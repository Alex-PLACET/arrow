// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

// Monitor the number of threads of a process, sampled from the kernel.
//
// The sample is the "Threads:" field of /proc/<pid>/status (Linux only).  A
// thread count is what tells you whether a server grows a thread per stream or
// serves them all from a fixed pool; point this at a running Flight server
// while clients drive load at it:
//
//   flight-async-generic-doget-example --port=31337   # prints "Server pid N"
//   flight-async-generic-threads-example --pid=N --duration_s=30 --names
//
// With --names each sample also prints the per-thread-name breakdown, read from
// /proc/<pid>/task/*/comm - which is how you tell gRPC's own threads
// (event_engine, grpc_global_timer, lifeguard) from an example's per-stream
// workers (doget_worker): same process, different owners.  A thread that never
// set a name shows the process's own name (the binary, truncated to 15
// characters); Arrow's signal-handler thread is one.
//
// With --pid=0 (the default) this process itself is sampled.
//
// Usage:
//   flight-async-generic-threads-example [--pid=PID] [--interval_ms=MS]
//                                        [--duration_s=N] [--names]
//
// Run with no arguments to print this message and exit.

DEFINE_int32(pid, 0, "Process to sample; 0 samples this process itself");
DEFINE_int32(interval_ms, 100, "Milliseconds between samples");
DEFINE_int32(duration_s, 5, "How long to sample; 0 samples until interrupted");
DEFINE_bool(names, false,
            "Also print the per-thread-name breakdown (/proc/<pid>/task/*/comm)");

namespace {

/// Set from the SIGINT/SIGTERM handler to stop the sampling loop.
volatile std::sig_atomic_t stop_requested = 0;

void OnSignal(int) { stop_requested = 1; }

/// \brief The "Threads:" field of /proc/<pid>/status, or -1 if it cannot be
/// read (wrong pid, non-Linux).
int ReadThreads(int pid) {
  std::ifstream status("/proc/" + std::to_string(pid) + "/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0) {
      return std::atoi(line.c_str() + 8);
    }
  }
  return -1;
}

/// \brief The comm (name) of every thread of the process, most frequent first.
std::vector<std::pair<std::string, int>> ReadThreadNames(int pid) {
  std::map<std::string, int> counts;
  const std::string tasks = "/proc/" + std::to_string(pid) + "/task";
  DIR* dir = opendir(tasks.c_str());
  if (dir == nullptr) return {};
  while (dirent* entry = readdir(dir)) {
    const std::string tid = entry->d_name;
    if (tid == "." || tid == "..") continue;
    std::ifstream comm(tasks + "/" + tid + "/comm");
    std::string name;
    if (std::getline(comm, name)) counts[name]++;
  }
  closedir(dir);
  std::vector<std::pair<std::string, int>> sorted(counts.begin(), counts.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  return sorted;
}

/// \brief One line like `event_engine=56 doget_stream_worker=40 lifeguard=1`.
std::string FormatThreadNames(int pid) {
  std::string out;
  for (const auto& [name, count] : ReadThreadNames(pid)) {
    if (!out.empty()) out += " ";
    out += name + "=" + std::to_string(count);
  }
  return out.empty() ? "n/a" : out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    std::cout << "Usage: " << argv[0]
              << " [--pid=PID] [--interval_ms=MS] [--duration_s=N] [--names]"
              << std::endl;
    return EXIT_SUCCESS;
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const int pid = FLAGS_pid > 0 ? FLAGS_pid : static_cast<int>(getpid());
  if (ReadThreads(pid) < 0) {
    std::cerr << "Cannot read the thread count of pid " << pid
              << " (missing process, or not Linux)" << std::endl;
    return EXIT_FAILURE;
  }
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  const auto start = std::chrono::steady_clock::now();
  int baseline = -1;
  int peak = -1;
  int last = -1;
  bool lost = false;
  while (!stop_requested) {
    const int threads = ReadThreads(pid);
    if (threads < 0) {
      lost = true;  // the process exited while we were watching
      break;
    }
    if (baseline < 0) baseline = threads;
    peak = std::max(peak, threads);
    last = threads;
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "[" << elapsed_s << "s] pid=" << pid << " threads=" << threads;
    if (FLAGS_names) std::cout << " | " << FormatThreadNames(pid);
    std::cout << std::endl;
    if (FLAGS_duration_s > 0 && elapsed_s >= FLAGS_duration_s) break;
    std::this_thread::sleep_for(
        std::chrono::milliseconds(std::max(1, FLAGS_interval_ms)));
  }
  if (lost) std::cout << "pid " << pid << " exited" << std::endl;
  std::cout << "pid " << pid << ": " << baseline << " baseline -> " << peak
            << " peak (last " << last << ")" << std::endl;
  if (FLAGS_names) {
    std::cout << "pid " << pid << " threads by name: " << FormatThreadNames(pid)
              << std::endl;
  }
  return EXIT_SUCCESS;
}
