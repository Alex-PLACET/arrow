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

#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <arrow/api.h>
#include <arrow/flight/api.h>
#include <gflags/gflags.h>
#include <condition_variable>

// Async DoPut on the async generic Flight server.
//
// Deriving from AsyncGenericFlightServerBase selects the async generic
// transport.  On this path DoPut is not answered by FlightServerBase::DoPut:
// the transport hands each decoded upload to the FlightDataListener returned by
// FlightServerOptions::listener_factory, one listener per RPC.  The callbacks
// are:
//
// * OnDescriptor(const FlightDescriptor&) - the descriptor of the upload;
//   returning non-OK rejects the upload (the client sees that status when it
//   closes the writer).
// * OnSchemaDecoded(std::shared_ptr<Schema>) - from ipc::Listener, the decoded
//   schema.
// * OnNext(FlightStreamChunk) - once per decoded message; chunk.data is the
//   RecordBatch, or nullptr for metadata-only messages.
//
// All of them run on a gRPC callback thread, so none of them may block: a slow
// callback stalls the RPC and every other call that thread would serve.  The
// listener interface has no completion callback (no OnFinish/OnError) - the
// RPC's completion is the transport's business - so this example drains its own
// queue instead of waiting for a notification.
//
// AsyncUploadHandler::OnNext therefore only pushes the chunk onto an
// UploadQueue and returns OK at once.  A worker thread pops the chunks and
// processes them one at a time, sleeping --slow_processing_ms per batch to
// stand in for real work.
//
// --demo proves the callbacks never block: the client writes 3 batches and the
// whole DoPut (DoPut() through Close()) must complete in well under the
// worker's total processing time (3 x --slow_processing_ms).  If a callback
// thread were blocked by the processing, the client-visible DoPut would take at
// least that long.
//
// Usage:
//   flight-async-generic-doput-example --port=31337
//   flight-async-generic-doput-example --demo --slow_processing_ms=100
//
// Run with no arguments to print this message and exit.

DEFINE_int32(port, 0, "Port to listen on (0 picks a free port)");
DEFINE_int32(slow_processing_ms, 100,
             "Time the worker thread spends on each uploaded batch, standing in for "
             "real work");
DEFINE_bool(demo, false, "Run the async DoPut self-check against the server, then exit");

namespace flight = arrow::flight;

/// \brief Make a one-column batch of `num_rows` int64 values starting at `value`.
arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeInt64Batch(
    const std::shared_ptr<arrow::Schema>& schema, int64_t value, int64_t num_rows) {
  arrow::Int64Builder builder;
  for (int64_t i = 0; i < num_rows; i++) {
    ARROW_RETURN_NOT_OK(builder.Append(value + i));
  }
  ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
  return arrow::RecordBatch::Make(schema, array->length(), {array});
}

/// \brief The hand-off between the gRPC callback threads and one worker thread.
///
/// Push() is called from a callback thread and only appends to a deque under a
/// mutex: it never waits for processing.  The worker pops one chunk at a time
/// and sleeps --slow_processing_ms per batch to stand in for real work.
class UploadQueue {
 public:
  explicit UploadQueue(int32_t slow_processing_ms)
      : slow_processing_ms_(slow_processing_ms), worker_([this] { Run(); }) {}

  ~UploadQueue() { Stop(); }

  /// \brief Hand a decoded chunk to the worker; returns immediately.
  void Push(flight::FlightStreamChunk chunk) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      chunks_.push_back(std::move(chunk));
    }
    ready_.notify_one();
  }

  /// \brief Block until every queued chunk has been processed.
  void Drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this] { return chunks_.empty() && !processing_; });
  }

  /// \brief Stop and join the worker.  Idempotent; call after the last Push().
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    ready_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  /// Counters, readable once Drain() has returned.
  int64_t num_batches() const { return num_batches_.load(); }
  int64_t num_rows() const { return num_rows_.load(); }

 private:
  void Run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      if (chunks_.empty()) {
        if (stopped_) {
          return;
        }
        ready_.wait(lock);
        continue;
      }
      auto chunk = std::move(chunks_.front());
      chunks_.pop_front();
      processing_ = true;
      lock.unlock();

      // The deliberately slow part, off the callback thread.
      std::this_thread::sleep_for(std::chrono::milliseconds(slow_processing_ms_));
      if (chunk.data != nullptr) {  // null for metadata-only messages
        num_batches_.fetch_add(1);
        num_rows_.fetch_add(chunk.data->num_rows());
      }

      lock.lock();
      processing_ = false;
      if (chunks_.empty()) {
        idle_.notify_all();
      }
    }
  }

  const int32_t slow_processing_ms_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable idle_;
  std::deque<flight::FlightStreamChunk> chunks_;
  bool processing_ = false;
  bool stopped_ = false;
  std::atomic<int64_t> num_batches_{0};
  std::atomic<int64_t> num_rows_{0};
  std::thread worker_;
};

/// \brief The FlightDataListener serving one DoPut RPC.
///
/// Every method runs on a gRPC callback thread and returns promptly: the
/// descriptor and the schema are printed as they arrive, and each chunk is
/// handed to the queue.  The processing happens on the queue's worker thread.
class AsyncUploadHandler : public flight::FlightDataListener {
 public:
  explicit AsyncUploadHandler(std::shared_ptr<UploadQueue> queue)
      : queue_(std::move(queue)) {}

  arrow::Status OnDescriptor(const flight::FlightDescriptor& descriptor) override {
    std::cout << "DoPut descriptor: " << descriptor.ToString() << std::endl;
    // Returning non-OK here would reject the upload; the client sees that
    // status when it closes the writer.
    return arrow::Status::OK();
  }

  arrow::Status OnSchemaDecoded(std::shared_ptr<arrow::Schema> schema) override {
    std::cout << "DoPut schema:";
    for (const auto& field : schema->fields()) {
      std::cout << " " << field->name() << ":" << field->type()->ToString();
    }
    std::cout << std::endl;
    return arrow::Status::OK();
  }

  arrow::Status OnNext(flight::FlightStreamChunk chunk) override {
    // Hand the chunk over and return at once: the worker thread does the slow
    // processing, this callback thread stays free for the transport.
    queue_->Push(std::move(chunk));
    return arrow::Status::OK();
  }

 private:
  std::shared_ptr<UploadQueue> queue_;
};

/// \brief The server.  Deriving from AsyncGenericFlightServerBase selects the
/// async generic transport; uploads are served by the listener from
/// options.listener_factory set in main(), never by a DoPut handler method.
class UploadServer : public flight::AsyncGenericFlightServerBase {};

/// \brief The --demo self-check.  The assertion that matters: the client-visible
/// DoPut finishes long before the worker thread has processed the batches, so
/// no callback thread was blocked.
bool RunDemo(int port, const std::shared_ptr<UploadQueue>& queue) {
  bool ok = true;

  auto location = flight::Location::ForGrpcTcp("127.0.0.1", port);
  if (!location.ok()) {
    std::cerr << "Failed to build client location: " << location.status() << std::endl;
    return false;
  }
  auto client = flight::FlightClient::Connect(*location);
  if (!client.ok()) {
    std::cerr << "Failed to connect: " << client.status() << std::endl;
    return false;
  }

  constexpr int kNumBatches = 3;
  constexpr int64_t kRowsPerBatch = 4;
  auto schema = arrow::schema({arrow::field("value", arrow::int64())});

  // Time the whole client-visible DoPut: the initial call through the final
  // Close().  The worker's processing happens after this window.
  const auto start = std::chrono::steady_clock::now();
  auto put = (*client)->DoPut(flight::FlightDescriptor::Command("upload"), schema);
  if (!put.ok()) {
    std::cerr << "DoPut failed: " << put.status() << std::endl;
    return false;
  }
  arrow::Status status;
  for (int i = 0; i < kNumBatches && status.ok(); i++) {
    auto batch = MakeInt64Batch(schema, i * kRowsPerBatch, kRowsPerBatch);
    if (!batch.ok()) {
      std::cerr << "Failed to make a batch: " << batch.status() << std::endl;
      return false;
    }
    status = put->writer->WriteRecordBatch(**batch);
  }
  if (status.ok()) {
    status = put->writer->DoneWriting();
  }
  if (status.ok()) {
    status = put->writer->Close();
  }
  const double put_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
          .count();
  if (!status.ok()) {
    std::cerr << "DoPut upload failed: " << status << std::endl;
    return false;
  }

  const double worker_ms = kNumBatches * FLAGS_slow_processing_ms;
  std::cout << "Client-visible DoPut (DoPut() to Close()) took " << put_ms
            << " ms; the worker still has " << worker_ms << " ms of processing queued"
            << std::endl;
  if (put_ms < 150.0) {
    std::cout << "PASS: client-visible DoPut finished in " << put_ms
              << " ms, well under the 150 ms budget" << std::endl;
  } else {
    std::cout << "FAIL: client-visible DoPut took " << put_ms << " ms, expected < 150 ms"
              << std::endl;
    ok = false;
  }
  if (put_ms < worker_ms) {
    std::cout << "PASS: DoPut returned " << put_ms << " ms in, before the worker's "
              << worker_ms << " ms of processing could have finished" << std::endl;
  } else {
    std::cout << "FAIL: DoPut took " << put_ms << " ms, no faster than the worker's "
              << worker_ms << " ms of processing: a callback thread was blocked"
              << std::endl;
    ok = false;
  }

  // Now let the worker finish and check what it saw.
  queue->Drain();
  if (queue->num_batches() == kNumBatches &&
      queue->num_rows() == kNumBatches * kRowsPerBatch) {
    std::cout << "PASS: worker processed " << queue->num_batches() << " batches, "
              << queue->num_rows() << " rows" << std::endl;
  } else {
    std::cout << "FAIL: worker processed " << queue->num_batches() << " batches, "
              << queue->num_rows() << " rows, expected " << kNumBatches << " batches, "
              << kNumBatches * kRowsPerBatch << " rows" << std::endl;
    ok = false;
  }
  return ok;
}

int main(int argc, char** argv) {
  if (argc == 1) {
    // As in the other examples: a bare run (e.g. from ctest) does not start a
    // server, it just prints the usage.
    std::cout << "Usage: " << argv[0]
              << " [--port=PORT] [--slow_processing_ms=MS] [--demo]" << std::endl;
    return EXIT_SUCCESS;
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto queue = std::make_shared<UploadQueue>(FLAGS_slow_processing_ms);
  UploadServer server;

  auto location = flight::Location::ForGrpcTcp("0.0.0.0", FLAGS_port);
  if (!location.ok()) {
    std::cerr << location.status() << std::endl;
    return EXIT_FAILURE;
  }
  flight::FlightServerOptions options(*location);
  // UploadServer derives from AsyncGenericFlightServerBase: the server class
  // alone selects the async generic transport, so there is no option to set.
  // The factory is called once per DoPut RPC and returns a fresh listener,
  // since uploads may overlap; returning nullptr would refuse the upload.
  options.listener_factory = [queue]() -> std::shared_ptr<flight::FlightDataListener> {
    return std::make_shared<AsyncUploadHandler>(queue);
  };

  auto status = server.Init(options);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return EXIT_FAILURE;
  }
  int port = server.port();
  std::cout << "Serving Flight on " << server.location().ToString() << std::endl;
  std::cout << "Server pid " << getpid() << ", port " << port << std::endl;

  if (FLAGS_demo) {
    bool ok = RunDemo(port, queue);
    ARROW_WARN_NOT_OK(server.Shutdown(), "Error shutting down server");
    ARROW_WARN_NOT_OK(server.Wait(), "Error waiting for server shutdown");
    queue->Stop();  // join the worker now that no more uploads can arrive
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  status = server.SetShutdownOnSignals({SIGINT, SIGTERM});
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return EXIT_FAILURE;
  }
  status = server.Serve();
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
