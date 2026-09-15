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

// Thread-explosion demo, sync half.
//
// The synchronous gRPC Flight server serves each RPC on a dedicated handler
// thread, and the whole DoGet exchange (writing every payload) runs on that
// thread. When the stream is slow, every concurrent in-flight RPC keeps one
// server thread busy: N concurrent slow streams -> ~N server threads.
//
// This is the sync half of a pair; see flight_threads_async_example.cc for
// the async counterpart, which serves the same workload with a flat thread
// count. The shared harness (flags, monitor, dataset, client workload, main)
// lives in flight_threads_common.h.
//
// Usage (server and clients are two processes of this binary):
//   ./flight-threads-sync-example --mode=server --port=31337 [--batches=20] [--batch_delay_ms=200]
//   ./flight-threads-sync-example --mode=clients --port=31337 --clients=50
//
// The server prints the process thread count every 250 ms and the peak at
// shutdown. Expected (Linux): peak threads ~= clients + small baseline.
// Thread counting uses /proc/self/status and is Linux-only; other platforms
// print "threads=-1" but the demo still runs.

namespace flight = ::arrow::flight;

namespace {

/// A RecordBatchStream that sleeps before each batch. The sleep runs on the
/// gRPC handler thread that is serving this RPC: with the sync server, every
/// concurrent slow stream keeps one server thread busy.
class SlowSyncStream final : public flight::RecordBatchStream {
 public:
  SlowSyncStream(std::shared_ptr<arrow::RecordBatchReader> reader, int64_t delay_ms,
                 std::shared_ptr<std::atomic<int>> active)
      : flight::RecordBatchStream(reader),
        delay_ms_(delay_ms),
        active_(std::move(active)) {}

  ~SlowSyncStream() override { Release(); }

  arrow::Result<flight::FlightPayload> Next() override {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
    ARROW_ASSIGN_OR_RAISE(auto payload, flight::RecordBatchStream::Next());
    if (payload.ipc_message.metadata == nullptr) {
      Release();  // End of stream.
    }
    return payload;
  }

 private:
  void Release() {
    if (!released_.exchange(true)) {
      active_->fetch_sub(1);
    }
  }

  int64_t delay_ms_;
  std::shared_ptr<std::atomic<int>> active_;
  std::atomic<bool> released_{false};
};

class SlowFlightServer final : public flight::FlightServerBase {
 public:
  std::shared_ptr<std::atomic<int>> active() const { return active_; }

  arrow::Status DoGet(const flight::ServerCallContext&, const flight::Ticket&,
                      std::unique_ptr<flight::FlightDataStream>* stream) override {
    active_->fetch_add(1);
    ARROW_ASSIGN_OR_RAISE(auto batches, flight_threads_demo::ExampleBatches());
    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make(batches));
    *stream =
        std::make_unique<SlowSyncStream>(std::move(reader), FLAGS_batch_delay_ms, active_);
    return arrow::Status::OK();
  }

 private:
  std::shared_ptr<std::atomic<int>> active_ = std::make_shared<std::atomic<int>>(0);
};

}  // namespace

int main(int argc, char** argv) {
  return flight_threads_demo::RunExampleMain<SlowFlightServer>(argc, argv, "[sync]");
}
