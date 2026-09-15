// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

// Shared harness for the thread-explosion demo pair:
// flight_threads_sync_example.cc and flight_threads_async_example.cc.
// It holds the flags, the /proc thread-count monitor, the example dataset,
// the client workload, and the main() skeleton, so each example only shows
// its own server half. Not a general-purpose header.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/flight/api.h>
#include <gflags/gflags.h>

DEFINE_string(mode, "server", "server or clients");
DEFINE_int32(port, -1, "Server port to listen on");
DEFINE_int32(clients, 50, "Number of concurrent clients (clients mode)");
DEFINE_int32(batches, 20, "Record batches per stream");
DEFINE_int32(batch_delay_ms, 200, "Delay before each batch is produced, in ms");

namespace flight_threads_demo {

namespace flight = ::arrow::flight;

/// Count the OS threads of this process (Linux: /proc/self/status "Threads:").
inline std::optional<int> ProcessThreadCount() {
#ifdef __linux__
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.starts_with("Threads:")) {
      return std::atoi(line.c_str() + 8);
    }
  }
#endif
  return std::nullopt;
}

/// Periodically sample this process's thread count while the server runs and
/// report the peak at shutdown.
class ThreadMonitor {
 public:
  ThreadMonitor(const char* label, std::shared_ptr<std::atomic<int>> active)
      : label_(label), active_(std::move(active)), thread_([this] { Run(); }) {}

  void StopAndReport() {
    stop_.store(true);
    if (thread_.joinable()) {
      thread_.join();
    }
    std::cout << label_ << " peak: threads=" << peak_threads_ << " (at " << peak_time_
              << "s, " << peak_active_ << " RPCs active)" << std::endl;
  }

 private:
  void Run() {
    const auto start = std::chrono::steady_clock::now();
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      const int threads = ProcessThreadCount().value_or(-1);
      const int active = active_->load();
      if (threads > peak_threads_) {
        peak_threads_ = threads;
        peak_time_ = elapsed;
        peak_active_ = active;
      }
      std::cout << label_ << " t=" << elapsed << "s threads=" << threads
                << " active_rpcs=" << active << std::endl;
    }
  }

  const char* label_;
  std::shared_ptr<std::atomic<int>> active_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  int peak_threads_ = 0;
  double peak_time_ = 0;
  int peak_active_ = 0;
};

/// Build the example dataset: one immutable batch repeated FLAGS_batches times.
inline arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ExampleBatches() {
  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues({1, 2, 3, 4}));
  std::shared_ptr<arrow::Array> values;
  ARROW_RETURN_NOT_OK(builder.Finish(&values));
  auto schema = arrow::schema({arrow::field("value", arrow::int64())});
  auto batch = arrow::RecordBatch::Make(schema, values->length(), {values});
  return std::vector<std::shared_ptr<arrow::RecordBatch>>(
      static_cast<size_t>(FLAGS_batches), batch);
}

/// Open FLAGS_clients concurrent DoGet streams, drain them completely, and
/// verify each stream delivered exactly FLAGS_batches batches.
inline arrow::Status RunClients(const char* label) {
  flight::Location location;
  ARROW_RETURN_NOT_OK(flight::Location::ForGrpcTcp("127.0.0.1", FLAGS_port).Value(&location));

  std::atomic<int32_t> ready{0};
  std::atomic<int32_t> failures{0};
  std::atomic<int64_t> total_batches{0};
  std::vector<std::thread> clients;
  const auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < FLAGS_clients; ++i) {
    clients.emplace_back([&] {
      ++ready;
      while (ready.load() < FLAGS_clients) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      auto maybe_client = flight::FlightClient::Connect(location);
      if (!maybe_client.ok()) {
        ++failures;
        return;
      }
      auto client = std::move(*maybe_client);
      auto maybe_stream = client->DoGet(flight::Ticket{"slow"});
      if (!maybe_stream.ok()) {
        ++failures;
        return;
      }
      auto stream = std::move(*maybe_stream);
      int64_t batches = 0;
      while (true) {
        auto maybe_chunk = stream->Next();
        if (!maybe_chunk.ok()) {
          ++failures;
          return;
        }
        auto chunk = std::move(*maybe_chunk);
        if (!chunk.data && !chunk.app_metadata) {
          break;  // End of stream.
        }
        if (chunk.data) {
          ++batches;
        }
      }
      total_batches += batches;
      if (batches != FLAGS_batches) {
        ++failures;
      }
      if (!client->Close().ok()) {
        ++failures;
      }
    });
  }
  for (auto& client : clients) {
    client.join();
  }
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << label << ": " << FLAGS_clients << " clients x " << FLAGS_batches
            << " batches, " << failures.load() << " failures, " << total_batches.load()
            << " batches received in " << elapsed << "s" << std::endl;
  if (failures.load() != 0) {
    return arrow::Status::Invalid("Client failures: ", failures.load());
  }
  return arrow::Status::OK();
}

/// Construct, run, and measure one demo server, then report the thread-count
/// peak. `Server` must expose the shared server lifecycle: Init(), location(),
/// SetShutdownOnSignals(), Serve(), and active().
template <typename Server>
arrow::Status RunServer(const char* label) {
  auto server = std::make_unique<Server>();
  flight::Location location;
  ARROW_RETURN_NOT_OK(
      flight::Location::ForGrpcTcp("0.0.0.0", FLAGS_port).Value(&location));
  ARROW_RETURN_NOT_OK(server->Init(flight::FlightServerOptions(location)));
  std::cout << label << " server listening on " << server->location().ToString()
            << " (batches=" << FLAGS_batches << ", batch_delay_ms=" << FLAGS_batch_delay_ms
            << ")" << std::endl;
  ThreadMonitor monitor(label, server->active());
  ARROW_RETURN_NOT_OK(server->SetShutdownOnSignals({SIGINT, SIGTERM}));
  ARROW_RETURN_NOT_OK(server->Serve());
  monitor.StopAndReport();
  return arrow::Status::OK();
}

/// Shared main(): parse flags, keep ctest happy (a bare run exits cleanly),
/// then run the server or the client workload.
template <typename Server>
int RunExampleMain(int argc, char** argv, const char* label) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_port < 0) {
    std::cout << "Must specify a port with --port (see the header comment"
                 " for usage)"
              << std::endl;
    return EXIT_SUCCESS;
  }
  arrow::Status status;
  if (FLAGS_mode == "server") {
    status = RunServer<Server>(label);
  } else if (FLAGS_mode == "clients") {
    status = RunClients(label);
  } else {
    std::cerr << "Unknown --mode " << FLAGS_mode << " (expected server or clients)"
              << std::endl;
    return EXIT_FAILURE;
  }
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace flight_threads_demo
