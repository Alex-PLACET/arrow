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

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <arrow/api.h>
#include <arrow/flight/api.h>
#include <gflags/gflags.h>

// Measure the thread count of flight-async-generic-server-example under many
// parallel DoGet streams.
//
// The async generic server pumps each DoGet's FlightDataStream on the gRPC
// callback thread serving the RPC, so a stream whose batches are slow holds
// that thread until the stream is done. This program points --streams clients
// at such a server, releases them all at once through a barrier, drains each
// DoGet and reports how far the server's thread count moved.
//
// The server's thread count is read from the kernel, not from the server: it
// is the "Threads:" field of the server process's /proc/<pid>/status, sampled
// every --sample_ms (Linux-only; without --server_pid, or if that file cannot
// be read, only this process's own thread count is sampled, for context).
//
// Usage:
//   flight-async-generic-threads-example [--streams=N] [--server_batches=N]
//                                        [--server_batch_delay_ms=MS]
//                                        [--timeout_s=N] [--sample_ms=N]
//                                        [--server_binary=PATH]
//
// With no --port this program starts the server itself (its sibling binary
// flight-async-generic-server-example, same directory, same build), reads the
// pid and port it prints, and stops it at the end: one command measures the
// async server.  Point --port (and --server_pid) at an already running server
// instead to measure that one:
//
//   flight-async-generic-server-example --port=0 --batches=10 --batch_delay_ms=100
//   flight-async-generic-threads-example --port=45823 --server_pid=995166 --streams=50
//
// Run with no arguments to print this message and exit.

DEFINE_int32(port, 0, "Port of an already running server; 0 starts one instead");
DEFINE_int32(server_pid, 0,
             "Pid of the server process to sample; 0 skips server sampling");
DEFINE_int32(streams, 50, "Number of parallel DoGet streams");
DEFINE_int32(expect_batches, -1,
             "Batches each stream is expected to deliver; negative means do not verify");
DEFINE_int32(timeout_s, 60, "Per-DoGet timeout in seconds");
DEFINE_int32(sample_ms, 250, "Milliseconds between thread count samples");
DEFINE_int32(server_batches, 10, "Batches the started server streams per DoGet");
DEFINE_int32(server_batch_delay_ms, 100,
             "Delay the started server puts between batches (makes streams slow)");
DEFINE_string(server_binary, "",
              "Server binary to start; empty means the sibling of this binary");

namespace flight = arrow::flight;

void PrintUsage(const char* program) {
  std::cout << "Usage: " << program
            << " [--streams=N] [--server_batches=N] [--server_batch_delay_ms=MS]"
            << " [--timeout_s=N] [--sample_ms=N] [--server_binary=PATH]" << std::endl
            << "   or: " << program
            << " --port=PORT [--server_pid=PID] [--streams=N] [--expect_batches=N]"
            << " [--timeout_s=N] [--sample_ms=N]" << std::endl;
}

/// \brief The "Threads:" field of /proc/<pid>/status, or -1 if it cannot be
/// read (wrong pid, non-Linux). The kernel's count for the process is what
/// this program measures; nothing else can report it.
int ReadThreads(int pid) {
  std::ifstream status("/proc/" + std::to_string(pid) + "/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.starts_with("Threads:")) {
      return std::atoi(line.c_str() + 8);
    }
  }
  return -1;
}

/// \brief One sampled series: the first sample is the baseline, the largest is
/// the peak, kept together with the second it was seen at.
struct Series {
  int baseline = -1;
  int peak = -1;
  double peak_at_s = 0;
  const char* label = "";
  int pid = 0;
  bool enabled = false;
  bool warned = false;
};

/// \brief Sample one series once, printing its field, and warn (a single time)
/// if it cannot be read.
void SampleSeries(Series* series, double elapsed_s, std::string* warning) {
  if (!series->enabled) {
    std::cout << " " << series->label << "_threads=n/a";
    return;
  }
  int threads = ReadThreads(series->pid);
  if (threads < 0) {
    if (!series->warned) {
      series->warned = true;
      *warning = std::string("cannot read /proc/") + std::to_string(series->pid) +
                 "/status; skipping the " + series->label + " thread count series";
    }
    std::cout << " " << series->label << "_threads=n/a";
    return;
  }
  if (series->baseline < 0) series->baseline = threads;
  if (threads > series->peak) {
    series->peak = threads;
    series->peak_at_s = elapsed_s;
  }
  std::cout << " " << series->label << "_threads=" << threads;
}

/// \brief Samples the thread counts of the server process and of this process
/// every --sample_ms, printing each sample.
class ThreadSampler {
 public:
  ThreadSampler(int server_pid, int sample_ms)
      : sample_ms_(std::max(1, sample_ms)), start_(std::chrono::steady_clock::now()) {
    server_.label = "server";
    server_.pid = server_pid;
    server_.enabled = server_pid > 0;
    self_.label = "self";
    self_.pid = getpid();
    self_.enabled = true;
  }

  /// Print and record one sample of both series.
  void SampleNow() {
    double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    std::string warning;
    std::cout << "[" << elapsed_s << "s]";
    SampleSeries(&server_, elapsed_s, &warning);
    SampleSeries(&self_, elapsed_s, &warning);
    std::cout << std::endl;
    if (!warning.empty()) std::cout << "Warning: " << warning << std::endl;
  }

  /// Keep sampling periodically until Stop().
  void Run() {
    while (!stop_.load()) {
      SampleNow();
      std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms_));
    }
  }

  void Stop() { stop_.store(true); }

  const Series& server() const { return server_; }
  const Series& self() const { return self_; }

 private:
  Series server_;
  Series self_;
  int sample_ms_;
  std::chrono::steady_clock::time_point start_;
  std::atomic<bool> stop_{false};
};

/// \brief What one client stream did.
struct StreamResult {
  int64_t batches = 0;
  std::string failure;  // empty when the stream finished without a problem
};

/// \brief One client: wait for the barrier, connect, DoGet the "slow" ticket,
/// drain the stream counting batches, then let the client go.
void RunStream(const flight::FlightCallOptions& options, std::barrier<>* sync,
               StreamResult* result) {
  // Every stream arrives, so the barrier always releases; the clients then
  // connect and hit the server at the same time.
  sync->arrive_and_wait();

  auto location = flight::Location::ForGrpcTcp("127.0.0.1", FLAGS_port);
  if (!location.ok()) {
    result->failure = location.status().ToString();
    return;
  }
  auto client = flight::FlightClient::Connect(*location);
  if (!client.ok()) {
    result->failure = "connect: " + client.status().ToString();
    return;
  }
  auto reader = (*client)->DoGet(options, flight::Ticket("slow"));
  if (!reader.ok()) {
    result->failure = "DoGet: " + reader.status().ToString();
    return;
  }
  while (true) {
    auto chunk = (*reader)->Next();
    if (!chunk.ok()) {
      result->failure = "drain: " + chunk.status().ToString();
      break;
    }
    if (chunk->data == nullptr) break;  // end of stream
    ++result->batches;
  }
  // The reader has no Close(): dropping it ends the RPC. The client is closed
  // explicitly so its channel goes away too.
  reader->reset();
  auto status = (*client)->Close();
  if (!status.ok() && result->failure.empty()) {
    result->failure = "close: " + status.ToString();
  }
}

/// \brief Print a series' baseline and peak, or why it is missing.
void PrintSeries(const char* name, const Series& series) {
  if (series.baseline < 0) {
    std::cout << name << ": not sampled" << std::endl;
    return;
  }
  std::cout << name << ": " << series.baseline << " baseline -> " << series.peak
            << " peak (+" << (series.peak - series.baseline) << ") at "
            << series.peak_at_s << "s" << std::endl;
}

/// \brief The server this program measures: started by this program unless
/// --port points at one that is already running.
struct ServerProcess {
  int pid = 0;
  int port = 0;
  std::string log_path;
  bool spawned = false;
};

/// \brief The server binary to start: the sibling of this binary (same build
/// directory), or --server_binary.
std::string ServerBinaryPath(const char* program) {
  if (!FLAGS_server_binary.empty()) return FLAGS_server_binary;
  std::filesystem::path self(program);
  std::filesystem::path dir = self.has_parent_path() ? self.parent_path() : ".";
  return (dir / "flight-async-generic-server-example").string();
}

/// \brief Start the server example and wait for the port it prints.  Its output
/// goes to its own log file rather than to this program's stdout, so the
/// samples stay readable.
arrow::Result<ServerProcess> StartServer(const std::string& binary, int batches,
                                         int batch_delay_ms) {
  ServerProcess server;
  server.spawned = true;
  std::error_code ec;
  server.log_path = (std::filesystem::temp_directory_path(ec) /
                     ("flight-async-generic-server-" + std::to_string(getpid()) + ".log"))
                        .string();
  int fd = ::open(server.log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    return arrow::Status::IOError("cannot open the server log ", server.log_path);
  }
  pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fd);
    return arrow::Status::IOError("fork failed");
  }
  if (pid == 0) {
    // The server: a process of its own, output in the log file.
    ::dup2(fd, STDOUT_FILENO);
    ::dup2(fd, STDERR_FILENO);
    if (fd > STDERR_FILENO) ::close(fd);
    const std::string port_arg = "--port=0";  // let the kernel pick a free one
    const std::string batches_arg = "--batches=" + std::to_string(batches);
    const std::string delay_arg = "--batch_delay_ms=" + std::to_string(batch_delay_ms);
    ::execl(binary.c_str(), binary.c_str(), port_arg.c_str(), batches_arg.c_str(),
            delay_arg.c_str(), static_cast<char*>(nullptr));
    ::_exit(127);
  }
  ::close(fd);
  server.pid = pid;

  // Wait for the "Server pid N, port P" line, or for the server to give up.
  for (int attempt = 0; attempt < 100; attempt++) {
    std::ifstream log(server.log_path);
    std::string line;
    while (std::getline(log, line)) {
      auto at = line.find(", port ");
      if (at != std::string::npos) server.port = std::atoi(line.c_str() + at + 7);
    }
    if (server.port > 0) break;
    int status = 0;
    if (::waitpid(pid, &status, WNOHANG) == pid) {
      return arrow::Status::IOError("the server exited during startup; see ",
                                    server.log_path);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (server.port <= 0) {
    ::kill(pid, SIGTERM);
    ::waitpid(pid, nullptr, 0);
    return arrow::Status::IOError("the server never reported a port; see ",
                                  server.log_path);
  }
  return server;
}

/// \brief Stop a server this program started, and report what it served.  The
/// count is evidence of which server answered (only the async server prints
/// "DoGet: ticket="); it is approximate, since concurrent DoGet handlers write
/// to the same stream and their lines can interleave.
void StopServer(const ServerProcess& server) {
  if (!server.spawned) return;
  ::kill(server.pid, SIGTERM);
  int status = 0;
  ::waitpid(server.pid, &status, 0);
  int served = 0;
  std::ifstream log(server.log_path);
  std::string line;
  while (std::getline(log, line)) {
    if (line.rfind("DoGet: ticket=", 0) == 0) served++;
  }
  std::cout << "Server log:   " << server.log_path << " (" << served
            << " DoGet calls seen; the tail may still be buffered)" << std::endl;
}

int main(int argc, char** argv) {
  if (argc == 1) {
    // As in the other examples: a bare run (e.g. from ctest) does not load
    // anything, it just prints the usage.
    PrintUsage(argv[0]);
    return EXIT_SUCCESS;
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_streams < 1) {
    std::cerr << "--streams must be at least 1" << std::endl;
    return EXIT_FAILURE;
  }
  // With no --port, measure a server this program starts (and stops) itself:
  // the async server example is then the only server it can be, and its pid is
  // known here rather than passed in.
  ServerProcess server;
  if (FLAGS_port <= 0) {
    const std::string binary = ServerBinaryPath(argv[0]);
    auto started = StartServer(binary, FLAGS_server_batches, FLAGS_server_batch_delay_ms);
    if (!started.ok()) {
      std::cerr << started.status() << std::endl;
      return EXIT_FAILURE;
    }
    server = *started;
    FLAGS_port = server.port;
    FLAGS_server_pid = server.pid;
    if (FLAGS_expect_batches < 0) FLAGS_expect_batches = FLAGS_server_batches;
    std::cout << "Started " << binary << " pid " << server.pid << " on port "
              << server.port << " (" << FLAGS_server_batches << " batches, "
              << FLAGS_server_batch_delay_ms << " ms apart; log " << server.log_path
              << ")" << std::endl;
  }
  if (FLAGS_server_pid <= 0) {
    std::cout << "Not sampling the server process: --server_pid is not set" << std::endl;
  }

  const int kStreams = FLAGS_streams;
  ThreadSampler sampler(FLAGS_server_pid, FLAGS_sample_ms);
  // The baseline is taken before any client starts, so the first sample is the
  // server's idle thread count.
  sampler.SampleNow();

  // FlightCallOptions::timeout is a deadline in (fractional) seconds.
  flight::FlightCallOptions call_options;
  call_options.timeout = flight::TimeoutDuration(FLAGS_timeout_s);

  std::vector<StreamResult> results(kStreams);
  std::barrier sync(kStreams + 1);
  std::vector<std::thread> clients;
  clients.reserve(kStreams);
  for (int i = 0; i < kStreams; i++) {
    clients.emplace_back([&, i]() { RunStream(call_options, &sync, &results[i]); });
  }

  std::thread sampler_thread([&]() { sampler.Run(); });

  std::cout << "Load: " << kStreams
            << " parallel DoGet streams against 127.0.0.1:" << FLAGS_port
            << " (ticket \"slow\", timeout " << FLAGS_timeout_s << "s)" << std::endl;
  auto start = std::chrono::steady_clock::now();
  sync.arrive_and_wait();  // release every client at once
  for (auto& client : clients) {
    client.join();
  }
  double wall_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  sampler.Stop();
  sampler_thread.join();
  sampler.SampleNow();  // one sample after the load is gone

  int64_t total_batches = 0;
  int failures = 0;
  std::map<std::string, int> failure_reasons;
  for (auto& result : results) {
    total_batches += result.batches;
    if (FLAGS_expect_batches >= 0 && result.failure.empty() &&
        result.batches != FLAGS_expect_batches) {
      result.failure = "streamed " + std::to_string(result.batches) +
                       " batches, expected " + std::to_string(FLAGS_expect_batches);
    }
    if (!result.failure.empty()) {
      failures++;
      failure_reasons[result.failure]++;
    }
  }

  std::cout << std::endl << "==== Summary ====" << std::endl;
  std::cout << "Streams:     " << kStreams << std::endl;
  std::cout << "Total batches: " << total_batches << std::endl
            << "Failures:      " << failures << std::endl;
  for (const auto& [reason, count] : failure_reasons) {
    std::cout << "  " << count << "x " << reason << std::endl;
  }
  std::cout << "Wall time:     " << wall_s << " s" << std::endl;
  std::cout << "Streams/s:     " << (kStreams / wall_s) << std::endl;
  PrintSeries("Server threads", sampler.server());
  PrintSeries("Self threads", sampler.self());
  StopServer(server);

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
