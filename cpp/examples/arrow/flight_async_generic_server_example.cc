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
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <arrow/api.h>
#include <arrow/flight/api.h>
#include <arrow/util/future.h>
#include <gflags/gflags.h>
#include <condition_variable>
#include <functional>
#include <map>

// Serve Arrow Flight over gRPC's generic callback API.
//
// Deriving from AsyncGenericFlightServerBase serves the RPCs through a gRPC
// generic (callback) service instead of the typed sync service. On this path
// only two RPCs are implemented:
//
// * DoGet is answered by DoGetAsync: the stream it returns paces its payloads
//   on a background timer thread, so a slow next payload never blocks the gRPC
//   callback thread serving the RPC (see TimedAsyncStream).
// * DoPut is NOT answered by FlightServerBase::DoPut: uploads are handed to the
//   FlightDataListener returned by options.listener_factory, one listener per
//   RPC. The descriptor arrives through FlightDataListener::OnDescriptor, and
//   returning non-OK from there rejects the upload.
//
// Every other RPC - GetFlightInfo included - answers UNIMPLEMENTED, so a stock
// client that calls GetFlightInfo before DoGet will not work against this
// server.
//
// Middleware runs on this path: ServerMiddlewareFactory::StartCall, the
// SendingHeaders hook and CallCompleted are all invoked.  The blocking
// ServerAuthHandler is still refused by Init(); this server instead derives
// from flight::AsyncGenericFlightServerBase (which selects the async transport) and
// overrides Handshake / ValidateToken to serve the handshake and check the
// per-call token.
//
// With --require_token those hooks are active: a client must Authenticate()
// first, and every later RPC carries the token its
// ClientAuthHandler::GetToken() returns in the auth-token-bin header.  The
// demo then also checks that an unauthenticated RPC is rejected before it
// authenticates.  Without the flag the hooks delegate to the base class:
// Handshake answers UNIMPLEMENTED and no token is required.
//
// Usage:
//   flight-async-generic-server-example --port=31337
//   flight-async-generic-server-example --port=0 --demo --batches=5 --batch_delay_ms=10
//   flight-async-generic-server-example --port=0 --demo --require_token
//
// DoGet is served by DoGetAsync: the stream it returns (TimedAsyncStream) paces
// its payloads on a background timer thread, so the gRPC callback thread
// serving the RPC is never blocked.
//
// Run with no arguments to print this message and exit.

DEFINE_int32(port, 0, "Port to listen on (0 picks a free port)");
DEFINE_int32(batches, 20, "Number of batches in the DoGet stream");
DEFINE_int32(batch_delay_ms, 100,
             "Delay before each batch in the DoGet stream (0 = non-blocking: the "
             "callback thread is not held)");
DEFINE_bool(demo, false, "Run the DoGet/DoPut self-check against the server, then exit");
DEFINE_bool(require_token, false,
            "Require a client handshake and a per-call token (Handshake / "
            "ValidateToken below)");

/// The credentials the --require_token demo accepts.  The handshake reads the
/// password; the per-call token the client sends is whatever its
/// ClientAuthHandler::GetToken() returns (this demo's client returns the
/// password), and ValidateToken checks it on every other RPC.
constexpr char kDemoUser[] = "user";
constexpr char kDemoPassword[] = "p4ssw0rd";

namespace flight = arrow::flight;

/// \brief Make a one-column batch of three int64 values starting at `value`.
arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeInt64Batch(
    const std::shared_ptr<arrow::Schema>& schema, int64_t value) {
  arrow::Int64Builder builder;
  for (int64_t i = 0; i < 3; i++) {
    ARROW_RETURN_NOT_OK(builder.Append(value + i));
  }
  ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
  return arrow::RecordBatch::Make(schema, array->length(), {array});
}

/// \brief A RecordBatchReader whose ReadNext() can sleep before each batch.
///
/// DoGetAsync hands it to TimedAsyncStream with delay 0: the --batch_delay_ms
/// pacing happens on DeadlineQueue's thread, so ReadNext() never holds a gRPC
/// callback thread.
class DelayingReader : public arrow::RecordBatchReader {
 public:
  DelayingReader(std::shared_ptr<arrow::Schema> schema, int32_t num_batches,
                 int32_t delay_ms)
      : schema_(std::move(schema)), num_batches_(num_batches), delay_ms_(delay_ms) {}

  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
    if (next_batch_ >= num_batches_) {
      *batch = nullptr;  // end of stream
      return arrow::Status::OK();
    }
    if (delay_ms_ > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
    }
    ARROW_ASSIGN_OR_RAISE(auto next, MakeInt64Batch(schema_, next_batch_));
    *batch = std::move(next);
    ++next_batch_;
    return arrow::Status::OK();
  }

 private:
  std::shared_ptr<arrow::Schema> schema_;
  int32_t num_batches_;
  int32_t delay_ms_;
  int32_t next_batch_ = 0;
};

/// \brief A timer: one thread for the whole process completes tasks at their
/// deadline.
///
/// This is what keeps the pacing of the async stream below off the gRPC
/// callback threads: a callback thread only hands out a future; the waiting
/// (and the IPC serialization of the wrapped stream) happens here.  One thread
/// serves every stream, so N streams cost one thread, not N.
class DeadlineQueue {
 public:
  static DeadlineQueue& Instance() {
    static DeadlineQueue queue;
    return queue;
  }

  /// \brief Run `task` on the queue's thread after `delay`.
  void After(std::chrono::milliseconds delay, std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace(std::chrono::steady_clock::now() + delay, std::move(task));
    }
    ready_.notify_one();
  }

 private:
  DeadlineQueue() : worker_([this] { Run(); }) {}

  ~DeadlineQueue() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    ready_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  DeadlineQueue(const DeadlineQueue&) = delete;
  DeadlineQueue& operator=(const DeadlineQueue&) = delete;

  void Run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stopped_) {
      if (tasks_.empty()) {
        ready_.wait(lock);
        continue;
      }
      if (ready_.wait_until(lock, tasks_.begin()->first) != std::cv_status::timeout) {
        continue;  // woken with a new task: re-check the earliest deadline
      }
      auto task = std::move(tasks_.begin()->second);
      tasks_.erase(tasks_.begin());
      lock.unlock();
      task();
      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::multimap<std::chrono::steady_clock::time_point, std::function<void()>> tasks_;
  std::thread worker_;
  bool stopped_ = false;
};

/// \brief An AsyncFlightDataStream that paces its payloads with the timer.
///
/// This is what the async stream interface is for: NextAsync() returns
/// immediately, and the payload -- the wait, plus the serialization of the
/// wrapped synchronous stream -- is produced on the queue's thread.  The gRPC
/// callback thread serving the RPC is never blocked, so slow streams do not
/// starve other connections.
class TimedAsyncStream : public flight::AsyncFlightDataStream {
 public:
  TimedAsyncStream(std::unique_ptr<flight::FlightDataStream> stream, int32_t delay_ms)
      : stream_(std::move(stream)), delay_ms_(delay_ms) {}

  std::shared_ptr<arrow::Schema> schema() override { return stream_->schema(); }

  arrow::Future<flight::FlightPayload> GetSchemaPayloadAsync() override {
    return AfterDelay([this] { return stream_->GetSchemaPayload(); });
  }

  arrow::Future<flight::FlightPayload> NextAsync() override {
    return AfterDelay([this] { return stream_->Next(); });
  }

  arrow::Status Close() override {
    // A pending payload must be resolved here.  The transport is waiting on its
    // future, and a future that never completes would keep the RPC alive:
    // Close() is the cancellation hook of the interface.
    arrow::Future<flight::FlightPayload> pending;
    bool has_pending = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
      pending = std::move(pending_);
      has_pending = has_pending_;
      has_pending_ = false;
    }
    if (has_pending) {
      pending.MarkFinished(arrow::Status::Cancelled());
    }
    return stream_->Close();
  }

 private:
  /// \brief Produce the next payload after --batch_delay_ms, on the queue's
  /// thread, and return the future the transport waits on.
  arrow::Future<flight::FlightPayload> AfterDelay(
      std::function<arrow::Result<flight::FlightPayload>()> produce) {
    arrow::Future<flight::FlightPayload> future =
        arrow::Future<flight::FlightPayload>::Make();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        future.MarkFinished(arrow::Status::Cancelled());
        return future;
      }
      pending_ = future;
      has_pending_ = true;
    }
    DeadlineQueue::Instance().After(
        std::chrono::milliseconds(delay_ms_),
        [this, future, produce = std::move(produce)]() mutable {
          bool cancelled;
          {
            std::lock_guard<std::mutex> lock(mutex_);
            cancelled = closed_;
            has_pending_ = false;
          }
          if (cancelled) {
            return;  // Close() already resolved the future
          }
          future.MarkFinished(produce());
        });
    return future;
  }

  std::unique_ptr<flight::FlightDataStream> stream_;
  int32_t delay_ms_;
  std::mutex mutex_;
  bool closed_ = false;
  /// The payload the transport is waiting for, so Close() can resolve it.
  arrow::Future<flight::FlightPayload> pending_ =
      arrow::Future<flight::FlightPayload>::Make();
  bool has_pending_ = false;
};

/// \brief The FlightDataListener serving one DoPut RPC.
///
/// The callbacks run on gRPC threads, so the counters the demo reads back are
/// atomics; the descriptor is guarded because it is not a scalar.
class ExampleUploadHandler : public flight::FlightDataListener {
 public:
  arrow::Status OnDescriptor(const flight::FlightDescriptor& descriptor) override {
    std::cout << "DoPut descriptor: " << descriptor.ToString() << std::endl;
    // Returning non-OK here would reject the upload; the client sees that
    // status when it closes the writer.
    std::lock_guard<std::mutex> lock(mutex_);
    descriptor_ = descriptor;
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
    if (chunk.data != nullptr) {  // null for metadata-only messages
      num_batches_.fetch_add(1);
      num_rows_.fetch_add(chunk.data->num_rows());
    }
    return arrow::Status::OK();
  }

  int64_t num_batches() const { return num_batches_.load(); }
  int64_t num_rows() const { return num_rows_.load(); }

  flight::FlightDescriptor descriptor() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return descriptor_;
  }

 private:
  std::atomic<int64_t> num_batches_{0};
  std::atomic<int64_t> num_rows_{0};
  mutable std::mutex mutex_;
  flight::FlightDescriptor descriptor_;
};

/// \brief The client half of the --require_token handshake: send the password,
/// expect the username back, then hand the password to every later call as the
/// token (ClientAuthHandler::GetToken is what the transport puts in the
/// auth-token-bin header).
class DemoClientAuthHandler : public flight::ClientAuthHandler {
 public:
  DemoClientAuthHandler(std::string username, std::string password)
      : username_(std::move(username)), password_(std::move(password)) {}

  arrow::Status Authenticate(flight::ClientAuthSender* outgoing,
                             flight::ClientAuthReader* incoming) override {
    ARROW_RETURN_NOT_OK(outgoing->Write(password_));
    std::string username;
    ARROW_RETURN_NOT_OK(incoming->Read(&username));
    if (username != username_) {
      return flight::MakeFlightError(flight::FlightStatusCode::Unauthenticated,
                                     "Invalid token");
    }
    return arrow::Status::OK();
  }

  arrow::Status GetToken(std::string* token) override {
    *token = password_;
    return arrow::Status::OK();
  }

 private:
  std::string username_;
  std::string password_;
};

/// \brief The server. DoGetAsync is the only data handler overridden: uploads
/// are served by the listener factory set in main(), never by a handler method.
/// The Handshake / ValidateToken hooks below serve --require_token; with it off
/// they delegate to the base class, which answers UNIMPLEMENTED and requires no
/// token.
class ExampleServer : public flight::AsyncGenericFlightServerBase {
 public:
  /// \brief The async DoGet: a future that may complete later, and a stream
  /// that produces its payloads asynchronously, so the gRPC callback thread
  /// serving the RPC is never held.
  arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>> DoGetAsync(
      const flight::ServerCallContext& context, const flight::Ticket& request) override {
    std::cout << "DoGetAsync: ticket=" << request.ticket << std::endl;
    auto schema = arrow::schema({arrow::field("value", arrow::int64())});

    // The reader itself never waits: TimedAsyncStream owns the pacing, on the
    // transport's behalf but not on its threads.
    auto reader = std::make_shared<DelayingReader>(schema, FLAGS_batches, /*delay_ms=*/0);
    auto stream = std::make_unique<TimedAsyncStream>(
        std::make_unique<flight::RecordBatchStream>(reader), FLAGS_batch_delay_ms);

    // Completing the future later is the point of DoGetAsync: preparing the
    // stream (a query plan, a schema lookup, an I/O open) must not run on the
    // callback thread either.
    arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>> future =
        arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>>::Make();
    // The queue runs std::function<void()>, which must be copyable: hand the
    // move-only stream over in a shared_ptr.
    auto handed_over = std::make_shared<std::unique_ptr<flight::AsyncFlightDataStream>>(
        std::move(stream));
    DeadlineQueue::Instance().After(
        std::chrono::milliseconds(FLAGS_batch_delay_ms), [future, handed_over]() mutable {
          future.MarkFinished(
              arrow::Result<std::unique_ptr<flight::AsyncFlightDataStream>>(
                  std::move(*handed_over)));
        });
    return future;
  }

  arrow::Status Handshake(const flight::ServerCallContext& context,
                          const std::string& password, std::string* response) override {
    if (!FLAGS_require_token) {
      return flight::AsyncGenericFlightServerBase::Handshake(context, password, response);
    }
    if (password != kDemoPassword) {
      return flight::MakeFlightError(flight::FlightStatusCode::Unauthenticated,
                                     "Invalid token");
    }
    *response = kDemoUser;
    return arrow::Status::OK();
  }

  arrow::Status ValidateToken(const flight::ServerCallContext& context,
                              const std::string& token,
                              std::string* peer_identity) override {
    if (!FLAGS_require_token) {
      return flight::AsyncGenericFlightServerBase::ValidateToken(context, token,
                                                                 peer_identity);
    }
    if (token != kDemoPassword) {
      return flight::MakeFlightError(flight::FlightStatusCode::Unauthenticated,
                                     "Invalid token");
    }
    *peer_identity = kDemoUser;
    return arrow::Status::OK();
  }
};

/// \brief The --demo self-check: drain a DoGet stream, then upload one batch
/// and verify the upload listener saw the descriptor and the rows. The
/// handler is only read after the client's RPCs have returned.
bool RunDemo(int port, const std::shared_ptr<ExampleUploadHandler>& handler) {
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

  // With --require_token the handshake comes first.  GetFlightInfo is a unary
  // call, so its status is the call's own: before Authenticate() it must be
  // rejected, and that rejection is UNAUTHENTICATED (mapped to IOError on the
  // client) rather than the UNIMPLEMENTED an unauthenticated server answers
  // with - auth runs before the method is dispatched.
  if (FLAGS_require_token) {
    auto info =
        (*client)->GetFlightInfo(flight::FlightDescriptor::Path({"example", "probe"}));
    if (!info.ok() && info.status().code() == arrow::StatusCode::IOError) {
      std::cout << "PASS: unauthenticated call rejected (UNAUTHENTICATED)" << std::endl;
    } else {
      std::cout << "FAIL: unauthenticated GetFlightInfo was not rejected: "
                << info.status() << std::endl;
      ok = false;
    }
    auto auth = (*client)->Authenticate(
        {}, std::make_unique<DemoClientAuthHandler>(kDemoUser, kDemoPassword));
    if (!auth.ok()) {
      std::cout << "FAIL: Authenticate failed: " << auth << std::endl;
      return false;
    }
    std::cout << "PASS: handshake accepted, token sent on later calls" << std::endl;
  }

  // DoGet: drain the whole stream. The server delays each batch by
  // --batch_delay_ms.
  auto reader = (*client)->DoGet(flight::Ticket("demo"));
  if (!reader.ok()) {
    std::cerr << "DoGet failed: " << reader.status() << std::endl;
    return false;
  }
  int64_t num_batches = 0;
  while (true) {
    auto chunk = (*reader)->Next();
    if (!chunk.ok()) {
      std::cerr << "DoGet drain failed: " << chunk.status() << std::endl;
      ok = false;
      break;
    }
    if (chunk->data == nullptr) break;  // end of stream
    ++num_batches;
  }
  if (num_batches == FLAGS_batches) {
    std::cout << "PASS: DoGet streamed " << num_batches << " batches" << std::endl;
  } else {
    std::cout << "FAIL: DoGet streamed " << num_batches << " batches, expected "
              << FLAGS_batches << std::endl;
    ok = false;
  }

  // DoPut: served by the listener from listener_factory, not by the server's
  // DoPut.
  auto schema = arrow::schema({arrow::field("value", arrow::int64())});
  auto descriptor = flight::FlightDescriptor::Path({"example", "upload"});
  auto batch = MakeInt64Batch(schema, 100);
  if (!batch.ok()) {
    std::cerr << "Failed to make a batch: " << batch.status() << std::endl;
    return false;
  }
  auto put = (*client)->DoPut(descriptor, schema);
  if (!put.ok()) {
    std::cerr << "DoPut failed: " << put.status() << std::endl;
    return false;
  }
  auto status = put->writer->WriteRecordBatch(**batch);
  if (status.ok()) status = put->writer->DoneWriting();
  if (status.ok()) status = put->writer->Close();
  if (!status.ok()) {
    std::cerr << "DoPut upload failed: " << status << std::endl;
    return false;
  }

  if (handler->descriptor().path == descriptor.path) {
    std::cout << "PASS: upload handler recorded descriptor "
              << handler->descriptor().ToString() << std::endl;
  } else {
    std::cout << "FAIL: upload handler recorded descriptor "
              << handler->descriptor().ToString() << ", expected "
              << descriptor.ToString() << std::endl;
    ok = false;
  }
  if (handler->num_batches() == 1 && handler->num_rows() == (*batch)->num_rows()) {
    std::cout << "PASS: upload handler saw " << handler->num_batches() << " batch, "
              << handler->num_rows() << " rows" << std::endl;
  } else {
    std::cout << "FAIL: upload handler saw " << handler->num_batches() << " batches, "
              << handler->num_rows() << " rows, expected 1 batch, "
              << (*batch)->num_rows() << " rows" << std::endl;
    ok = false;
  }
  return ok;
}

int main(int argc, char** argv) {
  if (argc == 1) {
    // As in the other examples: a bare run (e.g. from ctest) does not start a
    // server, it just prints the usage.
    std::cout << "Usage: " << argv[0]
              << " [--port=PORT] [--batches=N] [--batch_delay_ms=MS] [--demo] "
                 "[--require_token]"
              << std::endl;
    return EXIT_SUCCESS;
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto handler = std::make_shared<ExampleUploadHandler>();
  ExampleServer server;

  auto location = flight::Location::ForGrpcTcp("0.0.0.0", FLAGS_port);
  if (!location.ok()) {
    std::cerr << location.status() << std::endl;
    return EXIT_FAILURE;
  }
  flight::FlightServerOptions options(*location);
  // ExampleServer derives from AsyncGenericFlightServerBase: the server class
  // alone selects the async generic transport, so there is no option to set.
  // Called once per DoPut RPC. This demo hands every upload the same handler
  // so main() can inspect the result; a real server would return a fresh
  // listener per RPC, since uploads may overlap.
  options.listener_factory = [&]() -> std::shared_ptr<flight::FlightDataListener> {
    return handler;
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
    bool ok = RunDemo(port, handler);
    ARROW_WARN_NOT_OK(server.Shutdown(), "Error shutting down server");
    ARROW_WARN_NOT_OK(server.Wait(), "Error waiting for server shutdown");
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
