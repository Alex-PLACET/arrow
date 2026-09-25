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

// Serve Arrow Flight over gRPC's generic callback API.
//
// Setting FlightServerOptions::use_async_grpc serves the RPCs through a gRPC
// generic (callback) service instead of the typed sync service. On this path
// only two RPCs are implemented:
//
// * DoGet is answered by FlightServerBase::DoGet. The FlightDataStream it
//   returns is pumped payload by payload on the gRPC callback thread that
//   serves the RPC, so a slow next payload blocks that thread for the whole
//   stream (see DelayingReader); flight-async-generic-threads-example measures
//   the cost.
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
// from flight::AsyncGenericFlightServerBase (which implies use_async_grpc) and
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

/// \brief A RecordBatchReader whose ReadNext() is slow (--batch_delay_ms).
///
/// The async generic DoGet path calls ReadNext() on the gRPC callback thread
/// serving the RPC, so this sleep blocks that thread for the whole stream:
/// nothing else is done on it until the stream is finished.
/// flight-async-generic-threads-example measures that.  With
/// --batch_delay_ms=0 ReadNext() returns immediately and the callback thread is
/// never held: the same streams from the same client burst are then served
/// cleanly, which is the control for that measurement.
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

/// \brief The server. Only DoGet is overridden: on the async generic path
/// uploads are served by the listener factory, never by
/// FlightServerBase::DoPut, so overriding DoPut here would have no effect.
/// Deriving from AsyncGenericFlightServerBase also brings in the Handshake /
/// ValidateToken hooks implemented below; with --require_token off they
/// delegate to the base class, which answers UNIMPLEMENTED and requires no
/// token.
class ExampleServer : public flight::AsyncGenericFlightServerBase {
 public:
  arrow::Status DoGet(const flight::ServerCallContext& context,
                      const flight::Ticket& request,
                      std::unique_ptr<flight::FlightDataStream>* stream) override {
    std::cout << "DoGet: ticket=" << request.ticket << std::endl;
    auto schema = arrow::schema({arrow::field("value", arrow::int64())});
    auto reader =
        std::make_shared<DelayingReader>(schema, FLAGS_batches, FLAGS_batch_delay_ms);
    *stream = std::make_unique<flight::RecordBatchStream>(reader);
    return arrow::Status::OK();
  }

  arrow::Future<> Handshake(
      const flight::ServerCallContext& context,
      std::unique_ptr<flight::AsyncServerAuthSender> outgoing,
      std::unique_ptr<flight::AsyncServerAuthReader> incoming) override {
    if (!FLAGS_require_token) {
      return flight::AsyncGenericFlightServerBase::Handshake(context, std::move(outgoing),
                                                             std::move(incoming));
    }
    // Nothing here blocks: each step resolves a Future and the continuation
    // runs on a gRPC callback thread without holding it.
    auto sender = std::make_shared<std::unique_ptr<flight::AsyncServerAuthSender>>(
        std::move(outgoing));
    auto reader = std::make_shared<std::unique_ptr<flight::AsyncServerAuthReader>>(
        std::move(incoming));
    auto done = arrow::Future<>::Make();
    (*reader)->Read().AddCallback(
        [sender, done](const arrow::Result<std::string>& password) mutable {
          if (!password.ok()) {
            done.MarkFinished(password.status());
            return;
          }
          if (*password != kDemoPassword) {
            done.MarkFinished(flight::MakeFlightError(
                flight::FlightStatusCode::Unauthenticated, "Invalid token"));
            return;
          }
          (*sender)->Write(kDemoUser).AddCallback(
              [done](const arrow::Status& status) mutable { done.MarkFinished(status); });
        });
    return done;
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
  // ExampleServer derives from AsyncGenericFlightServerBase, whose Init()
  // forces use_async_grpc, so the flag is not set here.
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
