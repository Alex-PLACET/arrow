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

#include "flight_threads_common.h"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>

// Thread-explosion demo, async half.
//
// The asynchronous gRPC Flight server (callback API) never blocks its
// transport threads: DoGet returns a Future-based stream, and this example
// paces every batch through ONE shared scheduler thread that resolves the
// batch futures. The server process thread count therefore stays flat no
// matter how many concurrent slow streams are in flight.
//
// This is the async half of a pair; see flight_threads_sync_example.cc for
// the sync counterpart. The shared harness (flags, monitor, dataset, client
// workload, main) lives in flight_threads_common.h.
//
// Usage (server and clients are two processes of this binary):
//   ./flight-threads-async-example --mode=server --port=31337 [--batches=20] [--batch_delay_ms=200]
//   ./flight-threads-async-example --mode=clients --port=31337 --clients=50
//
// The server prints the process thread count every 250 ms and the peak at
// shutdown. Expected (Linux): peak threads ~= the startup baseline, independent
// of --clients. Thread counting uses /proc/self/status and is Linux-only.

namespace flight = ::arrow::flight;

namespace {

/// One thread that resolves delayed callbacks for every stream. This is the
/// only extra thread the server needs to pace all batches: it stands in for
/// whatever async primitives (timers, I/O completions) a real application
/// would use. Tasks are kept in a binary heap so the scheduler stays
/// O(log n) per task even with thousands of concurrent streams.
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
    tasks_.push_back({.due=std::chrono::steady_clock::now() + delay, .fn=std::move(fn)});
    std::push_heap(tasks_.begin(), tasks_.end(), Later);
    cv_.notify_all();
  }

 private:
  struct Task {
    std::chrono::steady_clock::time_point due;
    std::function<void()> fn;
  };

  /// Min-heap ordering: with std::push_heap/pop_heap, the heap's front is the
  /// EARLIEST due task when `Later(a, b)` means "a is due after b".
  static bool Later(const Task& a, const Task& b) { return a.due > b.due; }

  void Run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
      if (tasks_.empty()) {
        cv_.wait(lock);
        continue;
      }
      if (tasks_.front().due > std::chrono::steady_clock::now()) {
        cv_.wait_until(lock, tasks_.front().due);  // May wake early; the loop re-checks.
        continue;
      }
      std::pop_heap(tasks_.begin(), tasks_.end(), Later);
      Task task = std::move(tasks_.back());
      tasks_.pop_back();
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

/// Serialize the batches once, then serve the payloads as an async stream.
arrow::Result<std::vector<flight::FlightPayload>> ExamplePayloads() {
  ARROW_ASSIGN_OR_RAISE(auto batches, flight_threads_demo::ExampleBatches());
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
/// thread is ever blocked, the batch futures resolve on the scheduler thread.
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

}  // namespace

int main(int argc, char** argv) {
  return flight_threads_demo::RunExampleMain<SlowAsyncFlightServer>(argc, argv, "[async]");
}
