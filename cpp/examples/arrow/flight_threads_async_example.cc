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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <arrow/api.h>
#include <arrow/flight/api.h>
#include <gflags/gflags.h>

// Thread-explosion demo, async half.
//
// The asynchronous gRPC Flight server (callback API) never blocks its
// transport threads: DoGet returns a Future-based stream, and this example
// paces every batch through ONE shared scheduler thread that resolves the
// batch futures. The server process thread count therefore stays flat no
// matter how many concurrent slow streams are in flight.
//
// This is the async half of a pair; see flight_threads_sync_example.cc for
// the sync counterpart, which needs ~1 thread per in-flight RPC for the same
// workload.
//
// Usage (server and clients are two processes of this binary):
//   ./flight-threads-async-example --mode=server  --port=31337 [--batches=20] [--batch_delay_ms=200]
//   ./flight-threads-async-example --mode=clients --port=31337 --clients=50
//
// The server prints the process thread count every 250 ms and the peak at
// shutdown. Expected (Linux): peak threads ~= small baseline (gRPC pollers +
// scheduler + monitor), independent of --clients. Thread counting uses
// /proc/self/status and is Linux-only; other platforms print "threads=-1".

DEFINE_string(mode, "server", "server or clients");
DEFINE_int32(port, -1, "Server port to listen on");
DEFINE_int32(clients, 50, "Number of concurrent clients (clients mode)");
DEFINE_int32(batches, 20, "Record batches per stream");
DEFINE_int32(batch_delay_ms, 200, "Delay before each batch is produced, in ms");

namespace flight = ::arrow::flight;

namespace {

/// Count the OS threads of this process (Linux: /proc/self/status "Threads:").
std::optional<int> ProcessThreadCount() {
#ifdef __linux__
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0) {
      return std::atoi(line.c_str() + 8);
    }
  }
#endif
  return std::nullopt;
}

/// Periodically sample the process thread count while the server runs and
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

/// One thread that resolves delayed callbacks for every stream. This is the
/// only extra thread the server needs to pace all batches: it stands in for
/// whatever async primitives (timers, I/O completions) a real application
/// would use.
// ponytail: O(n) scan per wakeup; fine for a demo, use a heap for many tasks.
class DelayScheduler {
 public:
  DelayScheduler() : thread_([this] { Run(); }) {}

  ~DelayScheduler() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }

  /// Run `fn` on the scheduler thread after `delay`.
  void After(std::chrono::milliseconds delay, std::function<void()> fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push_back({std::chrono::steady_clock::now() + delay, std::move(fn)});
    cv_.notify_all();
  }

 private:
  struct Task {
    std::chrono::steady_clock::time_point due;
    std::function<void()> fn;
  };

  void Run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
      if (tasks_.empty()) {
        cv_.wait(lock);
        continue;
      }
      auto earliest = std::min_element(
          tasks_.begin(), tasks_.end(),
          [](const Task& a, const Task& b) { return a.due < b.due; });
      if (earliest->due > std::chrono::steady_clock::now()) {
        cv_.wait_until(lock, earliest->due);  // May wake early; the loop re-checks.
        continue;
      }
      auto task = std::move(*earliest);
      tasks_.erase(earliest);
      lock.unlock();
      task.fn();
      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<Task> tasks_;
  std::thread thread_;
  bool stop_ = false;
};

/// Build the example dataset: one immutable batch repeated FLAGS_batches times.
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ExampleBatches() {
  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues({1, 2, 3, 4}));
  std::shared_ptr<arrow::Array> values;
  ARROW_RETURN_NOT_OK(builder.Finish(&values));
  auto schema = arrow::schema({arrow::field("value", arrow::int64())});
  auto batch = arrow::RecordBatch::Make(schema, values->length(), {values});
  return std::vector<std::shared_ptr<arrow::RecordBatch>>(
      static_cast<size_t>(FLAGS_batches), batch);
}

/// Serialize the batches once, then serve the payloads as an async stream.
arrow::Result<std::vector<flight::FlightPayload>> ExamplePayloads() {
  ARROW_ASSIGN_OR_RAISE(auto batches, ExampleBatches());
  ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make(batches));
  flight::RecordBatchStream stream(std::move(reader));
  std::vector<flight::FlightPayload> payloads;
  ARROW_ASSIGN_OR_RAISE(auto schema_payload, stream.GetSchemaPayload());
  payloads.push_back(std::move(schema_payload));
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto payload, stream.Next());
    if (payload.ipc_message.metadata == nullptr) {
      break;  // End of stream.
    }
    payloads.push_back(std::move(payload));
  }
  return payloads;
}

/// Async stream that paces each batch through the shared scheduler. No gRPC
/// thread is ever blocked; the batch futures resolve on the scheduler thread.
class SlowAsyncStream final : public flight::AsyncFlightDataStream {
 public:
  SlowAsyncStream(std::shared_ptr<DelayScheduler> scheduler,
                  std::vector<flight::FlightPayload> payloads, int64_t delay_ms,
                  std::shared_ptr<std::atomic<int>> active)
      : scheduler_(std::move(scheduler)),
        payloads_(std::move(payloads)),
        delay_ms_(delay_ms),
        active_(std::move(active)) {}

  arrow::Future<flight::FlightPayload> GetSchemaPayload() override {
    if (payloads_.empty()) {
      return arrow::Future<flight::FlightPayload>::MakeFinished(
          arrow::Status::Invalid("No payloads prepared"));
    }
    return arrow::Future<flight::FlightPayload>::MakeFinished(payloads_.front());
  }

  arrow::Future<flight::FlightPayload> Next() override {
    if (next_index_ >= payloads_.size()) {
      return arrow::Future<flight::FlightPayload>::MakeFinished(flight::FlightPayload{});
    }
    auto out = arrow::Future<flight::FlightPayload>::Make();
    auto payload = payloads_[next_index_++];
    scheduler_->After(std::chrono::milliseconds(delay_ms_),
                      [out, payload = std::move(payload)]() mutable {
                        out.MarkFinished(std::move(payload));
                      });
    return out;
  }

  arrow::Future<> Close() override {
    if (!closed_.exchange(true)) {
      active_->fetch_sub(1);
    }
    return arrow::Future<>::MakeFinished();
  }

 private:
  std::shared_ptr<DelayScheduler> scheduler_;
  std::vector<flight::FlightPayload> payloads_;  // [0] is the schema payload.
  size_t next_index_ = 1;
  int64_t delay_ms_;
  std::shared_ptr<std::atomic<int>> active_;
  std::atomic<bool> closed_{false};
};

class SlowAsyncFlightServer final : public flight::AsyncFlightServerBase {
 public:
  SlowAsyncFlightServer() : scheduler_(std::make_shared<DelayScheduler>()) {}

  std::shared_ptr<std::atomic<int>> active() const { return active_; }

  arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>> DoGet(
      const flight::ServerCallContext&, const flight::Ticket&) override {
    active_->fetch_add(1);
    auto payloads = ExamplePayloads();
    if (!payloads.ok()) {
      return arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>>::MakeFinished(
          payloads.status());
    }
    return arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>>::MakeFinished(
        std::make_unique<SlowAsyncStream>(scheduler_, std::move(*payloads),
                                          FLAGS_batch_delay_ms, active_));
  }

 private:
  std::shared_ptr<DelayScheduler> scheduler_;
  std::shared_ptr<std::atomic<int>> active_ = std::make_shared<std::atomic<int>>(0);
};

arrow::Status RunServer(const char* label) {
  auto server = std::make_unique<SlowAsyncFlightServer>();
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

arrow::Status RunClients(const char* label) {
  flight::Location location;
  ARROW_RETURN_NOT_OK(
      flight::Location::ForGrpcTcp("127.0.0.1", FLAGS_port).Value(&location));

  std::atomic<int> ready{0};
  std::atomic<int> failures{0};
  std::atomic<long> total_batches{0};
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
      long batches = 0;
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

}  // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_port < 0) {
    // For CI: ctest runs the example with no arguments; exit instead
    // of starting a server that would block forever.
    std::cout << "Must specify a port with --port (see the header comment"
                 " for usage)"
              << std::endl;
    return EXIT_SUCCESS;
  }

  arrow::Status status;
  if (FLAGS_mode == "server") {
    status = RunServer("[async]");
  } else if (FLAGS_mode == "clients") {
    status = RunClients("[async]");
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
