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

// End-to-end tests of the async generic gRPC Flight service: a real
// FlightServerBase served through AsyncGenericFlightService, driven by the
// regular sync FlightClient.

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include "arrow/flight/client.h"
#include "arrow/flight/server.h"
#include "arrow/flight/server_middleware.h"
#include "arrow/flight/test_auth_handlers.h"
#include "arrow/flight/transport/grpc/async_grpc_service.h"
#include "arrow/scalar.h"
#include "arrow/testing/gtest_util.h"

namespace arrow::flight::transport::grpc {

namespace {

// The ticket the test server answers with a stream that has a schema and no
// batches at all; every other ticket gets the 3-batch stream.
constexpr const char* kSchemaOnlyTicket = "schema-only-do-get-ticket";

// Serves 3 batches of 5 rows for any ticket, and records the ticket it was
// asked for. DoGet runs on a gRPC thread; the test reads the ticket only
// after the server has shut down.
class TestFlightServer : public FlightServerBase {
 public:
  arrow::Status DoGet(const ServerCallContext& context, const Ticket& request,
                      std::unique_ptr<FlightDataStream>* stream) override {
    ticket_ = request.ticket;

    auto schema = arrow::schema(
        {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
    if (request.ticket == kSchemaOnlyTicket) {
      // An empty reader still carries the schema; Next() reports the end of
      // stream right away.
      ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make({}, schema));
      *stream = std::make_unique<arrow::flight::RecordBatchStream>(std::move(reader));
      return arrow::Status::OK();
    }

    auto batch = arrow::RecordBatch::Make(
        schema, 5,
        {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3, 4, 5]"),
         arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30, 40, 50]")});
    ARROW_ASSIGN_OR_RAISE(auto reader,
                          arrow::RecordBatchReader::Make({batch, batch, batch}));
    *stream = std::make_unique<arrow::flight::RecordBatchStream>(std::move(reader));
    return arrow::Status::OK();
  }

  const std::string& ticket() const { return ticket_; }

 private:
  std::string ticket_;
};

// Records what the decoder hands to the application: the schema, every chunk,
// and whether the consumer accepted it. OnNext returns whatever status the
// test configured, so the upload's error path is reachable from here.
class RecordingListener : public FlightDataListener {
 public:
  arrow::Status OnSchemaDecoded(std::shared_ptr<arrow::Schema> schema) override {
    ++schema_count_;
    schema_ = std::move(schema);
    return arrow::Status::OK();
  }

  arrow::Status OnNext(FlightStreamChunk chunk) override {
    if (chunk.data) {
      batches_.push_back(std::move(chunk.data));
    } else {
      ++metadata_only_count_;
      metadata_chunks_.push_back(std::move(chunk.app_metadata));
    }
    last_status_ = next_status_;
    return next_status_;
  }

  // Records the upload's descriptor and reports whatever status the test
  // configured, so the descriptor rejection path is reachable from here too.
  arrow::Status OnDescriptor(const FlightDescriptor& descriptor) override {
    descriptors_.push_back(descriptor);
    return descriptor_status_;
  }

  // Consumer-side rejection of the upload, as the decoder tests do.
  void set_next_status(arrow::Status status) { next_status_ = std::move(status); }
  // Rejection before any schema or data arrives.
  void set_descriptor_status(arrow::Status status) {
    descriptor_status_ = std::move(status);
  }

  int schema_count() const { return schema_count_; }
  int descriptor_count() const { return static_cast<int>(descriptors_.size()); }
  int metadata_only_count() const { return metadata_only_count_; }
  const std::shared_ptr<arrow::Schema>& schema() const { return schema_; }
  const std::vector<FlightDescriptor>& descriptors() const { return descriptors_; }
  const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches() const {
    return batches_;
  }
  const std::vector<std::shared_ptr<arrow::Buffer>>& metadata_chunks() const {
    return metadata_chunks_;
  }
  const arrow::Status& last_status() const { return last_status_; }

 private:
  int schema_count_ = 0;
  int metadata_only_count_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<FlightDescriptor> descriptors_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  std::vector<std::shared_ptr<arrow::Buffer>> metadata_chunks_;
  arrow::Status descriptor_status_;
  arrow::Status next_status_;
  arrow::Status last_status_;
};

// Middleware execution counts, shared between the factory, the middleware and
// the test thread (all three touch it).
struct MiddlewareTrace {
  std::mutex mutex;
  int start_call = 0;
  int sending_headers = 0;
  int call_completed = 0;
};

// Records middleware execution so a test can assert StartCall/CallCompleted
// ran on the async path (the sync path is covered by TestMiddleware).
class RecordingServerMiddleware : public ServerMiddleware {
 public:
  explicit RecordingServerMiddleware(std::shared_ptr<MiddlewareTrace> trace)
      : trace_(std::move(trace)) {}

  std::string name() const override { return "recording"; }
  void SendingHeaders(AddCallHeaders* /*outgoing_headers*/) override {
    std::lock_guard<std::mutex> guard(trace_->mutex);
    trace_->sending_headers++;
  }
  void CallCompleted(const Status& /*status*/) override {
    std::lock_guard<std::mutex> guard(trace_->mutex);
    trace_->call_completed++;
  }

 private:
  std::shared_ptr<MiddlewareTrace> trace_;
};

class RecordingServerMiddlewareFactory : public ServerMiddlewareFactory {
 public:
  explicit RecordingServerMiddlewareFactory(std::shared_ptr<MiddlewareTrace> trace)
      : trace_(std::move(trace)) {}

  Status StartCall(const CallInfo& /*info*/, const ServerCallContext& /*context*/,
                   std::shared_ptr<ServerMiddleware>* middleware) override {
    {
      std::lock_guard<std::mutex> guard(trace_->mutex);
      trace_->start_call++;
    }
    *middleware = std::make_shared<RecordingServerMiddleware>(trace_);
    return Status::OK();
  }

 private:
  std::shared_ptr<MiddlewareTrace> trace_;
};

class RejectingServerMiddlewareFactory : public ServerMiddlewareFactory {
 public:
  Status StartCall(const CallInfo&, const ServerCallContext&,
                   std::shared_ptr<ServerMiddleware>*) override {
    return Status::Invalid("rejected by middleware");
  }
};

// The gRPC transport builds the generic service with the shared context
// helper (which runs middleware and auth). Tests that register the service on a
// hand-built grpc::ServerBuilder must pass one explicitly.
std::shared_ptr<GrpcServerCallContextHelper<::grpc::CallbackServerContext>>
MakeAsyncHelper() {
  return std::make_shared<GrpcServerCallContextHelper<::grpc::CallbackServerContext>>(
      /*auth_handler=*/nullptr, MiddlewareFactoryList{});
}

// Upload one batch and report whichever surface first reports an error: the
// DoPut call, the write, or Close.
std::pair<std::string, arrow::Status> UploadOneBatch(
    FlightClient* client, const FlightDescriptor& descriptor,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<arrow::RecordBatch>& batch) {
  auto result = client->DoPut(descriptor, schema);
  if (!result.ok()) {
    return {"DoPut", result.status()};
  }
  auto write_status = result->writer->WriteRecordBatch(*batch);
  if (!write_status.ok()) {
    return {"WriteRecordBatch", write_status};
  }
  return {"Close", result->writer->Close()};
}

TEST(AsyncGrpcTest, BasicDoGet) {
  TestFlightServer flight_server;
  AsyncGenericFlightService service(&flight_server, {}, MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  // Connect the existing arrow::flight::FlightClient implementation.
  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  // Call DoGet: the ticket must reach FlightServerBase::DoGet.
  Ticket ticket{"basic-do-get-ticket"};
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(ticket));

  // Read schema.
  ASSERT_OK_AND_ASSIGN(auto schema, reader->GetSchema());
  ASSERT_EQ(schema->num_fields(), 2);
  ASSERT_EQ(schema->field(0)->name(), "a");
  ASSERT_EQ(schema->field(1)->name(), "b");

  // Read batches.
  int batch_count = 0;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
    if (!chunk.data) break;
    ASSERT_EQ(chunk.data->num_rows(), 5);
    ASSERT_OK_AND_ASSIGN(auto value, chunk.data->column(0)->GetScalar(0));
    ASSERT_TRUE(value->Equals(*arrow::MakeScalar(int64_t(1))));
    batch_count++;
  }
  ASSERT_EQ(batch_count, 3);

  // Cleanup. The ticket is asserted only after the server stopped, so the
  // DoGet callback cannot race with the read.
  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  ASSERT_EQ(flight_server.ticket(), "basic-do-get-ticket");
}

TEST(AsyncGrpcTest, BasicDoPut) {
  TestFlightServer flight_server;
  // One listener for the whole test: shared with the service's factory and
  // kept here so it can be asserted on after the RPC.
  auto listener = std::make_shared<RecordingListener>();
  AsyncGenericFlightService service(
      &flight_server, [listener]() { return listener; }, MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  // Connect the existing arrow::flight::FlightClient implementation.
  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  // Call DoPut and upload the same batch twice. Close() returns the server's
  // final status, i.e. it only succeeds once the acknowledgement written on
  // end of stream has been read.
  FlightDescriptor descriptor = FlightDescriptor::Path({"test"});
  ASSERT_OK_AND_ASSIGN(auto result, client->DoPut(descriptor, schema));
  ASSERT_OK(result.writer->WriteRecordBatch(*batch));
  ASSERT_OK(result.writer->WriteRecordBatch(*batch));
  ASSERT_OK(result.writer->Close());

  // Cleanup. The listener is asserted only after the server stopped, so the
  // DoPut callback cannot race with the reads.
  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  // The listener saw the schema once and both batches, in order.
  ASSERT_EQ(listener->schema_count(), 1);
  ASSERT_EQ(listener->schema()->num_fields(), 2);
  ASSERT_EQ(listener->schema()->field(0)->name(), "a");
  ASSERT_EQ(listener->schema()->field(1)->name(), "b");
  ASSERT_EQ(listener->batches().size(), 2);
  for (const auto& chunk : listener->batches()) {
    ASSERT_EQ(chunk->num_rows(), 3);
    ASSERT_EQ(chunk->num_columns(), 2);
    ASSERT_OK_AND_ASSIGN(auto value, chunk->column(0)->GetScalar(0));
    ASSERT_TRUE(value->Equals(*arrow::MakeScalar(int64_t(1))));
  }
  // No metadata-only (error/app-metadata) chunks, and nothing the consumer
  // rejected: a non-OK OnNext status would have failed DoPut with no ack.
  ASSERT_EQ(listener->metadata_only_count(), 0);
  ASSERT_OK(listener->last_status());
}

TEST(AsyncGrpcTest, SchemaOnlyDoGet) {
  TestFlightServer flight_server;
  AsyncGenericFlightService service(&flight_server, {}, MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  // This ticket is served as a stream with no batches at all.
  Ticket ticket{kSchemaOnlyTicket};
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(ticket));

  // The schema is served...
  ASSERT_OK_AND_ASSIGN(auto schema, reader->GetSchema());
  ASSERT_EQ(schema->num_fields(), 2);
  ASSERT_EQ(schema->field(0)->name(), "a");
  ASSERT_EQ(schema->field(1)->name(), "b");

  // ...and the very first read reports the end of stream instead of failing.
  ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
  ASSERT_EQ(chunk.data, nullptr);

  // Cleanup. The ticket is asserted only after the server stopped, so the
  // DoGet callback cannot race with the read.
  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  ASSERT_EQ(flight_server.ticket(), kSchemaOnlyTicket);
}

TEST(AsyncGrpcTest, EmptyDoPut) {
  TestFlightServer flight_server;
  auto listener = std::make_shared<RecordingListener>();
  AsyncGenericFlightService service(
      &flight_server, [listener]() { return listener; }, MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});

  // Upload nothing at all: the acknowledgement the server writes back on the
  // client's half-close is what makes Close() return.
  FlightDescriptor descriptor = FlightDescriptor::Path({"empty-put"});
  ASSERT_OK_AND_ASSIGN(auto result, client->DoPut(descriptor, schema));
  ASSERT_OK(result.writer->Close());

  // Cleanup before looking at the listener, so the DoPut callback cannot race
  // with the reads.
  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  // No data of any kind reached the listener.
  ASSERT_EQ(listener->batches().size(), 0);
  ASSERT_EQ(listener->metadata_only_count(), 0);
  ASSERT_OK(listener->last_status());
  // The sync client sends the schema payload eagerly, even when no batch
  // follows, so the listener does see the schema here. Deliberately not
  // asserted: that is the client's business, not the transport's.
}

TEST(AsyncGrpcTest, MetadataOnlyPutChunk) {
  TestFlightServer flight_server;
  auto listener = std::make_shared<RecordingListener>();
  AsyncGenericFlightService service(
      &flight_server, [listener]() { return listener; }, MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});

  // RFC 3.2: a message carrying application metadata and no data is a valid
  // chunk of the upload.
  FlightDescriptor descriptor = FlightDescriptor::Path({"metadata-only-put"});
  ASSERT_OK_AND_ASSIGN(auto result, client->DoPut(descriptor, schema));
  auto app_metadata = arrow::Buffer::FromString("metadata-only-upload");
  ASSERT_OK(result.writer->WriteMetadata(app_metadata));
  ASSERT_OK(result.writer->Close());

  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  // Exactly one chunk reached the listener: no data, that exact metadata.
  ASSERT_EQ(listener->batches().size(), 0);
  ASSERT_EQ(listener->metadata_only_count(), 1);
  ASSERT_EQ(listener->metadata_chunks().size(), 1);
  ASSERT_NE(listener->metadata_chunks()[0], nullptr);
  ASSERT_EQ(listener->metadata_chunks()[0]->ToString(), "metadata-only-upload");
}

TEST(AsyncGrpcTest, DoPutRejectedByListener) {
  TestFlightServer flight_server;
  auto listener = std::make_shared<RecordingListener>();
  // The consumer rejects the upload.
  listener->set_next_status(arrow::Status::Invalid("listener rejected this upload"));
  AsyncGenericFlightService service(
      &flight_server, [listener]() { return listener; }, MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  // The rejection must reach the client as the upload's status, whichever of
  // the three surfaces reports it first (observed: the writer's Close()).
  FlightDescriptor descriptor = FlightDescriptor::Path({"rejected-put"});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, schema, batch);
  ASSERT_FALSE(status.ok())
      << "the listener's rejection never reached the client, surfaced on " << surface;
  ASSERT_EQ(status.code(), arrow::StatusCode::Invalid);
  ASSERT_NE(status.message().find("listener rejected this upload"), std::string::npos)
      << "unexpected status: " << status;

  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();
}

TEST(AsyncGrpcTest, DoPutWithoutFactoryIsUnimplemented) {
  TestFlightServer flight_server;
  // No listener factory: there is nothing to hand an upload to, so DoPut is
  // answered like any other method the service does not serve.
  AsyncGenericFlightService service(&flight_server, {}, MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  FlightDescriptor descriptor = FlightDescriptor::Path({"factory-less-put"});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, schema, batch);
  ASSERT_FALSE(status.ok())
      << "DoPut without a listener factory must be refused, surfaced on " << surface;
  // The gRPC UNIMPLEMENTED fallback replies with an empty message, so only the
  // code is asserted here (observed: the writer's Close()).
  ASSERT_EQ(status.code(), arrow::StatusCode::NotImplemented);

  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();
}

TEST(AsyncGrpcTest, OtherMethodsAreUnimplemented) {
  TestFlightServer flight_server;
  auto listener = std::make_shared<RecordingListener>();
  AsyncGenericFlightService service(
      &flight_server, [listener]() { return listener; }, MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  // DoGet and DoPut are served; every other method falls through to the
  // UNIMPLEMENTED fallback reactor, which the client sees as NotImplemented.
  FlightDescriptor descriptor = FlightDescriptor::Path({"not-served"});
  auto info = client->GetFlightInfo(descriptor);
  ASSERT_FALSE(info.ok()) << "GetFlightInfo must not be served by the async service";
  ASSERT_EQ(info.status().code(), arrow::StatusCode::NotImplemented);

  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlag) {
  // The flag must default off: the sync typed service is the default path.
  ASSERT_OK_AND_ASSIGN(auto default_location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions default_options(default_location);
  ASSERT_FALSE(default_options.use_async_grpc);

  // Same server class as the tests above, but now started through the real
  // FlightServerBase::Init instead of a hand-built grpc::ServerBuilder.
  TestFlightServer flight_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  options.use_async_grpc = true;
  ASSERT_OK(flight_server.Init(options));
  ASSERT_GT(flight_server.port(), 0);

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  // (i) DoGet went through the generic reactor: the ticket reached
  // FlightServerBase::DoGet and the whole stream came back.
  Ticket ticket{"flag-path-do-get-ticket"};
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(ticket));
  ASSERT_OK_AND_ASSIGN(auto schema, reader->GetSchema());
  ASSERT_EQ(schema->num_fields(), 2);
  ASSERT_EQ(schema->field(0)->name(), "a");
  ASSERT_EQ(schema->field(1)->name(), "b");
  int batch_count = 0;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
    if (!chunk.data) break;
    ASSERT_EQ(chunk.data->num_rows(), 5);
    ASSERT_OK_AND_ASSIGN(auto value, chunk.data->column(0)->GetScalar(0));
    ASSERT_TRUE(value->Equals(*arrow::MakeScalar(int64_t(1))));
    batch_count++;
  }
  ASSERT_EQ(batch_count, 3);

  // (ii) Every other RPC falls through to the UNIMPLEMENTED fallback reactor:
  // the typed sync service is NOT registered alongside the generic one.
  FlightDescriptor descriptor = FlightDescriptor::Path({"not-served"});
  auto info = client->GetFlightInfo(descriptor);
  ASSERT_FALSE(info.ok()) << "the sync service must not be registered with the flag on";
  ASSERT_EQ(info.status().code(), arrow::StatusCode::NotImplemented);

  // (iii) The flag path takes no listener factory yet, so uploads are refused
  // the same way.
  auto put_schema = arrow::schema({arrow::field("a", arrow::int64())});
  auto batch = arrow::RecordBatch::Make(put_schema, 1,
                                        {arrow::ArrayFromJSON(arrow::int64(), "[1]")});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, put_schema, batch);
  ASSERT_FALSE(status.ok()) << "DoPut must be refused, surfaced on " << surface;
  ASSERT_EQ(status.code(), arrow::StatusCode::NotImplemented);

  // Cleanup. The ticket is asserted only after the server stopped, so the
  // DoGet callback cannot race with the read.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  ASSERT_EQ(flight_server.ticket(), "flag-path-do-get-ticket");
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagServesDoPut) {
  // The flag path serves uploads once it is configured with a listener
  // factory: FlightServerOptions::listener_factory hands out one listener per
  // DoPut RPC instead of the RPC falling through to UNIMPLEMENTED.
  TestFlightServer flight_server;
  auto listener = std::make_shared<RecordingListener>();
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  options.use_async_grpc = true;
  options.listener_factory = [listener]() { return listener; };
  ASSERT_OK(flight_server.Init(options));
  ASSERT_GT(flight_server.port(), 0);

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  const FlightDescriptor descriptor = FlightDescriptor::Path({"wave", "h2"});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, schema, batch);
  ASSERT_TRUE(status.ok()) << "upload failed on " << surface << ": " << status;

  // Cleanup before looking at the listener: its callbacks run on a gRPC
  // thread, and nothing may read the recordings while that thread writes.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  // The listener saw the descriptor the client sent, the schema, and the batch.
  ASSERT_EQ(listener->descriptor_count(), 1);
  ASSERT_EQ(listener->descriptors()[0], descriptor);
  ASSERT_EQ(listener->schema_count(), 1);
  ASSERT_EQ(listener->schema()->num_fields(), 2);
  ASSERT_EQ(listener->schema()->field(0)->name(), "a");
  ASSERT_EQ(listener->schema()->field(1)->name(), "b");
  ASSERT_EQ(listener->batches().size(), 1);
  const auto& chunk = listener->batches()[0];
  ASSERT_EQ(chunk->num_rows(), 3);
  ASSERT_EQ(chunk->num_columns(), 2);
  ASSERT_OK_AND_ASSIGN(auto value, chunk->column(0)->GetScalar(0));
  ASSERT_TRUE(value->Equals(*arrow::MakeScalar(int64_t(1))));
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagUploadRejectedOnDescriptor) {
  // A handler that rejects the descriptor rejects the whole upload through
  // the same surface as an OnNext rejection: the writer's status.
  TestFlightServer flight_server;
  auto listener = std::make_shared<RecordingListener>();
  listener->set_descriptor_status(
      arrow::Status::Invalid("listener rejected this descriptor"));
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  options.use_async_grpc = true;
  options.listener_factory = [listener]() { return listener; };
  ASSERT_OK(flight_server.Init(options));
  ASSERT_GT(flight_server.port(), 0);

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  const FlightDescriptor descriptor = FlightDescriptor::Path({"rejected-h2"});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, schema, batch);
  ASSERT_FALSE(status.ok())
      << "the descriptor rejection never reached the client, surfaced on " << surface;
  ASSERT_EQ(status.code(), arrow::StatusCode::Invalid);
  ASSERT_NE(status.message().find("listener rejected this descriptor"), std::string::npos)
      << "unexpected status: " << status;

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  // The upload was rejected on its descriptor: the handler saw it, and no
  // schema or data followed it.
  ASSERT_EQ(listener->descriptor_count(), 1);
  ASSERT_EQ(listener->descriptors()[0], descriptor);
  ASSERT_EQ(listener->schema_count(), 0);
  ASSERT_EQ(listener->batches().size(), 0);
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagRefusesBlockingAuthHandler) {
  // The callback path cannot run the blocking ServerAuthHandler (it would run
  // on callback threads), so a server configured with one must refuse to start
  // instead of silently serving unauthenticated requests.  Middleware *is*
  // supported now - see UseAsyncGrpcFlagAllowsMiddleware.
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));

  TestFlightServer auth_server;
  FlightServerOptions with_auth(location);
  with_auth.use_async_grpc = true;
  with_auth.auth_handler = std::make_shared<TestServerAuthHandler>("user", "pass");
  auto auth_status = auth_server.Init(with_auth);
  ASSERT_FALSE(auth_status.ok()) << "an auth handler must not be silently skipped";
  ASSERT_EQ(auth_status.code(), arrow::StatusCode::NotImplemented);
  ASSERT_NE(auth_status.message().find("AsyncGenericFlightServerBase"), std::string::npos)
      << auth_status;
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagAllowsMiddleware) {
  // Middleware configured through FlightServerOptions must run on the async
  // path: StartCall at call creation, SendingHeaders and CallCompleted at the
  // end - success or failure alike.
  auto trace = std::make_shared<MiddlewareTrace>();
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));

  TestFlightServer flight_server;
  FlightServerOptions options(location);
  options.use_async_grpc = true;
  options.middleware = {
      {"recording", std::make_shared<RecordingServerMiddlewareFactory>(trace)}};
  ASSERT_OK(flight_server.Init(options));

  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", flight_server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(Ticket{"middleware-ticket"}));
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
    if (!chunk.data) break;
  }

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  std::lock_guard<std::mutex> guard(trace->mutex);
  EXPECT_EQ(1, trace->start_call);
  EXPECT_EQ(1, trace->sending_headers);
  EXPECT_EQ(1, trace->call_completed);
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagMiddlewareCanRejectCall) {
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  TestFlightServer flight_server;
  FlightServerOptions options(location);
  options.use_async_grpc = true;
  options.middleware = {
      {"rejecting", std::make_shared<RejectingServerMiddlewareFactory>()}};
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", flight_server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto status = client->DoGet(Ticket{"rejected"}).status();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("rejected by middleware"));

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

// ---------------------------------------------------------------------------
// Handshake + per-RPC token auth on the async generic service.
// ---------------------------------------------------------------------------

// Counters shared between the server callbacks and the test thread.
struct AsyncAuthState {
  std::mutex mutex;
  int handshakes = 0;
  int validations = 0;
  std::string peer_identity;
};

// Async-transport twin of TestServerAuthHandler: the handshake answers the
// client's password with the username; later RPCs must carry that password as
// the token, which is what TestClientAuthHandler::GetToken() puts in the
// `auth-token-bin` header (the sync twin validates it the same way in
// TestServerAuthHandler::IsValid).  It is a server class: the two hooks below
// are the only configuration the transport needs.
class AsyncAuthTestServer : public AsyncGenericFlightServerBase {
 public:
  AsyncAuthTestServer(std::string username, std::string password, std::string ticket,
                      std::shared_ptr<AsyncAuthState> state = nullptr)
      : username_(std::move(username)),
        password_(std::move(password)),
        ticket_(std::move(ticket)),
        state_(state ? std::move(state) : std::make_shared<AsyncAuthState>()) {}

  const std::shared_ptr<AsyncAuthState>& state() const { return state_; }

  Status DoGet(const ServerCallContext& context, const Ticket& request,
               std::unique_ptr<FlightDataStream>* stream) override {
    {
      std::lock_guard<std::mutex> guard(state_->mutex);
      state_->peer_identity = context.peer_identity();
    }
    if (request.ticket != ticket_) {
      return Status::KeyError("No such ticket: ", request.ticket);
    }
    auto schema = arrow::schema({arrow::field("a", arrow::int64())});
    auto batch = arrow::RecordBatch::Make(
        schema, 2, {arrow::ArrayFromJSON(arrow::int64(), "[1, 2]")});
    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make({batch}));
    *stream = std::make_unique<arrow::flight::RecordBatchStream>(std::move(reader));
    return Status::OK();
  }

  // The hook: the transport read the client's password; answer with the
  // username, or reject the RPC the way the sync twin does.
  Status Handshake(const ServerCallContext& /*context*/, const std::string& password,
                   std::string* response) override {
    {
      std::lock_guard<std::mutex> guard(state_->mutex);
      state_->handshakes++;
    }
    if (password != password_) {
      return MakeFlightError(FlightStatusCode::Unauthenticated, "Invalid token");
    }
    *response = username_;
    return Status::OK();
  }

  Status ValidateToken(const ServerCallContext& /*context*/, const std::string& token,
                       std::string* peer_identity) override {
    std::lock_guard<std::mutex> guard(state_->mutex);
    state_->validations++;
    if (token != password_) {
      return MakeFlightError(FlightStatusCode::Unauthenticated, "Invalid token");
    }
    *peer_identity = username_;
    return Status::OK();
  }

 private:
  std::string username_;
  std::string password_;
  std::string ticket_;
  std::shared_ptr<AsyncAuthState> state_;
};

// A server with the async auth hooks installed, and a client connected to it.
// `status` carries the setup result; construction never fails a test.
struct AuthenticatedHarness {
  AsyncAuthTestServer server{"user", "p4ssw0rd", "auth-ticket"};
  std::unique_ptr<FlightClient> client;
  Status status;

  AuthenticatedHarness() : status(Start()) {}

  ~AuthenticatedHarness() {
    if (client) {
      ARROW_WARN_NOT_OK(client->Close(), "Close()");
    }
    ARROW_WARN_NOT_OK(server.Shutdown(), "Shutdown()");
  }

 private:
  Status Start() {
    ARROW_ASSIGN_OR_RAISE(auto location, Location::Parse("grpc://localhost:0"));
    // No use_async_grpc flag and no hook plumbing: the class and its virtuals
    // are the whole configuration.
    FlightServerOptions options(location);
    RETURN_NOT_OK(server.Init(options));
    ARROW_ASSIGN_OR_RAISE(auto client_location,
                          Location::ForScheme("grpc", "127.0.0.1", server.port()));
    ARROW_ASSIGN_OR_RAISE(client, FlightClient::Connect(client_location));
    return Status::OK();
  }
};

TEST(AsyncGrpcTest, HandshakeWithoutHookIsUnimplemented) {
  // A bare async server: Handshake is answered by the class's default virtual
  // with UNIMPLEMENTED, and the client's Authenticate() surfaces it.
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  AsyncGenericFlightServerBase flight_server;
  FlightServerOptions options(location);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", flight_server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto status = client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd"));
  ASSERT_FALSE(status.ok()) << "handshake must not succeed without a hook";
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("authentication mechanism"));

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, AsyncGenericFlightServerBaseImpliesAsyncTransport) {
  // Subclassing the async class is enough: no use_async_grpc on the options and
  // no hooks, yet the RPC is served by the generic callback service.  DoPut is
  // the discriminator: FlightServerBase's default (the sync service) answers
  // "NYI", the async service with no listener factory answers with its own
  // message.
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  AsyncGenericFlightServerBase flight_server;
  FlightServerOptions options(location);
  ASSERT_FALSE(options.use_async_grpc) << "this test must not opt in via the flag";
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", flight_server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  // DoPut is the discriminator: FlightServerBase's default (the sync service)
  // answers "NYI", the async service with no listener factory answers with its
  // own message.  The refusal can surface on either call: FlightClient::DoPut
  // writes the schema payload eagerly, so when the server's UNIMPLEMENTED
  // arrives first the DoPut call itself fails; otherwise the writer's Close()
  // reports it.
  auto put = client->DoPut(FlightDescriptor::Path({"x"}), arrow::schema({}));
  auto status = put.status();
  if (status.ok()) {
    status = put->writer->Close();
  }
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("no listener available"))
      << "the async generic service must have served this call, not the sync path";

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, HandshakeAuthenticatesAndTokenIsRequired) {
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);

  // 1. Without a handshake the data call is rejected before it is served.
  ASSERT_OK_AND_ASSIGN(auto unauthenticated_location,
                       Location::ForScheme("grpc", "127.0.0.1", harness.server.port()));
  ASSERT_OK_AND_ASSIGN(auto unauthenticated_client,
                       FlightClient::Connect(unauthenticated_location));
  auto status = unauthenticated_client->DoGet(Ticket{"auth-ticket"}).status();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Invalid token"));
  ASSERT_OK(unauthenticated_client->Close());

  // 2. After the handshake the same call succeeds and the server sees the
  //    validated identity.
  ASSERT_OK(harness.client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
  ASSERT_OK_AND_ASSIGN(auto reader, harness.client->DoGet(Ticket{"auth-ticket"}));
  int batches = 0;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
    if (!chunk.data) break;
    batches++;
  }
  EXPECT_EQ(1, batches);
  {
    std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
    EXPECT_EQ(1, harness.server.state()->handshakes);
    EXPECT_EQ("user", harness.server.state()->peer_identity);
    EXPECT_GE(harness.server.state()->validations, 1);
  }
}

TEST(AsyncGrpcTest, BadPasswordIsRejectedAtHandshake) {
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  auto status = harness.client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "wrong"));
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Invalid token"));
}

TEST(AsyncGrpcTest, UnauthenticatedUnknownMethodIsRejectedBeforeUnimplemented) {
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  // GetFlightInfo is not served by the async generic service, but auth runs
  // first: the client must see the auth failure, not UNIMPLEMENTED.
  auto status = harness.client->GetFlightInfo(FlightDescriptor::Path({"x"})).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Invalid token"));
  EXPECT_EQ(arrow::StatusCode::IOError, status.code());
}

// ---------------------------------------------------------------------------
// The synchronous request/response Handshake hook: payload fidelity, edge
// cases, and reactor cleanup (the reactor finishes on the strength of "one
// terminal path per direction", so the concurrency test is its real check).
// ---------------------------------------------------------------------------

// Send `password`, require the server to echo `expect` back.
class EchoClientAuthHandler : public ClientAuthHandler {
 public:
  EchoClientAuthHandler(std::string password, std::string expect)
      : password_(std::move(password)), expect_(std::move(expect)) {}

  Status Authenticate(ClientAuthSender* outgoing, ClientAuthReader* incoming) override {
    ARROW_RETURN_NOT_OK(outgoing->Write(password_));
    ARROW_RETURN_NOT_OK(incoming->Read(&response_));
    if (response_ != expect_) {
      return MakeFlightError(FlightStatusCode::Unauthenticated,
                             "unexpected handshake response");
    }
    return Status::OK();
  }

  Status GetToken(std::string* token) override {
    *token = password_;
    return Status::OK();
  }

  const std::string& response() const { return response_; }

 private:
  std::string password_;
  std::string expect_;
  std::string response_;
};

TEST(AsyncGrpcTest, HandshakePayloadIsBinarySafe) {
  // The hook's request is a std::string taken from the protobuf and its
  // response is echoed back through the client's reader: an embedded NUL and
  // non-ASCII bytes must survive both directions unchanged.
  const std::string binary("p4ss\0w0rd\xC3\xA9", 11);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  AsyncAuthTestServer server(binary, binary, "bin-ticket");
  FlightServerOptions options(location);
  ASSERT_OK(server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto handler = std::make_unique<EchoClientAuthHandler>(binary, binary);
  const auto* observed = handler.get();
  ASSERT_OK(client->Authenticate({}, std::move(handler)));
  EXPECT_EQ(binary, observed->response());

  // The token is the password, so an authenticated call must also work.
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(Ticket{"bin-ticket"}));
  ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
  EXPECT_NE(nullptr, chunk.data);

  ASSERT_OK(client->Close());
  ASSERT_OK(server.Shutdown());
  ASSERT_OK(server.Wait());
}

TEST(AsyncGrpcTest, EmptyHandshakeRequestIsRejectedNotHung) {
  // An empty HandshakeRequest is a valid protobuf message: the read succeeds,
  // the hook sees an empty string, and its rejection must reach the client.
  class SilentClientAuthHandler : public ClientAuthHandler {
   public:
    Status Authenticate(ClientAuthSender* outgoing, ClientAuthReader*) override {
      return outgoing->Write("");
    }
    Status GetToken(std::string*) override { return Status::OK(); }
  };

  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  auto status =
      harness.client->Authenticate({}, std::make_unique<SilentClientAuthHandler>());
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Invalid token"));
  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(1, harness.server.state()->handshakes)
      << "the hook must run on an empty request, not fail the read";
}

TEST(AsyncGrpcTest, AbandonedHandshakeLeavesServerUsable) {
  // A client that never sends a message: the server's read completes with
  // ok=false, the reactor must still finish, and the server must keep serving.
  class AbandoningClientAuthHandler : public ClientAuthHandler {
   public:
    Status Authenticate(ClientAuthSender*, ClientAuthReader*) override {
      return Status::Invalid("client never speaks");
    }
    Status GetToken(std::string*) override { return Status::OK(); }
  };

  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  EXPECT_FALSE(
      harness.client->Authenticate({}, std::make_unique<AbandoningClientAuthHandler>())
          .ok());

  ASSERT_OK_AND_ASSIGN(auto location,
                       Location::ForScheme("grpc", "127.0.0.1", harness.server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));
  ASSERT_OK(client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(Ticket{"auth-ticket"}));
  ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
  EXPECT_NE(nullptr, chunk.data);
  ASSERT_OK(client->Close());

  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(1, harness.server.state()->handshakes)
      << "the abandoned call must never have reached the hook";
}

TEST(AsyncGrpcTest, ConcurrentHandshakesAllComplete) {
  constexpr int kClients = 32;
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);

  std::vector<std::thread> threads;
  std::vector<Status> statuses(kClients);
  for (int i = 0; i < kClients; i++) {
    threads.emplace_back([&harness, &statuses, i] {
      auto location = Location::ForScheme("grpc", "127.0.0.1", harness.server.port());
      if (!location.ok()) {
        statuses[i] = location.status();
        return;
      }
      auto client = FlightClient::Connect(*location);
      if (!client.ok()) {
        statuses[i] = client.status();
        return;
      }
      auto authenticated = (*client)->Authenticate(
          {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd"));
      if (!authenticated.ok()) {
        statuses[i] = authenticated;
        return;
      }
      auto reader = (*client)->DoGet(Ticket{"auth-ticket"});
      if (!reader.ok()) {
        statuses[i] = reader.status();
        return;
      }
      auto chunk = (*reader)->Next();
      if (!chunk.ok()) {
        statuses[i] = chunk.status();
        return;
      }
      statuses[i] = (*client)->Close();
    });
  }
  for (auto& thread : threads) thread.join();
  for (int i = 0; i < kClients; i++) {
    EXPECT_TRUE(statuses[i].ok()) << "client " << i << ": " << statuses[i];
  }

  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(kClients, harness.server.state()->handshakes);
}

TEST(AsyncGrpcTest, RepeatedHandshakesOnFreshChannels) {
  constexpr int kRounds = 50;
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  for (int i = 0; i < kRounds; i++) {
    ASSERT_OK_AND_ASSIGN(auto location,
                         Location::ForScheme("grpc", "127.0.0.1", harness.server.port()));
    ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));
    ASSERT_OK(client->Authenticate(
        {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
    ASSERT_OK(client->Close());
  }
  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(kRounds, harness.server.state()->handshakes);
}

TEST(AsyncGrpcTest, LateHandshakeResponseLeavesServerUsable) {
  // The client gives up while the hook is still running, so the server's write
  // fails: the reactor's write-side terminal path must still Finish, and the
  // server must keep serving.
  class SlowAnsweringServer : public AsyncGenericFlightServerBase {
   public:
    Status Handshake(const ServerCallContext&, const std::string&,
                     std::string* response) override {
      handshakes.fetch_add(1);
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      *response = "user";
      return Status::OK();
    }
    std::atomic<int> handshakes{0};
  };

  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  SlowAnsweringServer server;
  FlightServerOptions options(location);
  ASSERT_OK(server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  FlightCallOptions impatient;
  impatient.timeout = std::chrono::milliseconds(50);
  auto status = client->Authenticate(
      impatient, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd"));
  EXPECT_FALSE(status.ok()) << "the client's deadline must expire during the hook";
  EXPECT_NE(std::string::npos, status.ToString().find("Deadline"))
      << "expected a deadline error, got: " << status.ToString();
  EXPECT_EQ(1, server.handshakes.load())
      << "the first call must have reached the hook, so its write had to finish";

  ASSERT_OK_AND_ASSIGN(auto second_location,
                       Location::ForScheme("grpc", "127.0.0.1", server.port()));
  ASSERT_OK_AND_ASSIGN(auto second_client, FlightClient::Connect(second_location));
  ASSERT_OK(second_client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));

  ASSERT_OK(second_client->Close());
  ASSERT_OK(client->Close());
  ASSERT_OK(server.Shutdown());
  ASSERT_OK(server.Wait());
}

TEST(AsyncGrpcTest, EmptyHandshakeResponseIsDelivered) {
  // A hook that answers nothing still produces exactly one (empty) response
  // message: the client's Read must return an empty string rather than hang.
  class NoResponseServer : public AsyncGenericFlightServerBase {
   public:
    Status Handshake(const ServerCallContext&, const std::string&,
                     std::string*) override {
      return Status::OK();
    }
  };

  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  NoResponseServer server;
  FlightServerOptions options(location);
  ASSERT_OK(server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto handler = std::make_unique<EchoClientAuthHandler>("anything", std::string());
  const auto* observed = handler.get();
  ASSERT_OK(client->Authenticate({}, std::move(handler)));
  EXPECT_EQ(std::string(), observed->response());

  ASSERT_OK(client->Close());
  ASSERT_OK(server.Shutdown());
  ASSERT_OK(server.Wait());
}

TEST(AsyncGrpcTest, RehandshakeOnTheSameChannelWorks) {
  // The Handshake RPC is per-call, so authenticating twice on one channel must
  // work: the second handshake reaches the hook and the channel keeps serving.
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  ASSERT_OK(harness.client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
  ASSERT_OK(harness.client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
  ASSERT_OK_AND_ASSIGN(auto reader, harness.client->DoGet(Ticket{"auth-ticket"}));
  ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
  EXPECT_NE(nullptr, chunk.data);

  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(2, harness.server.state()->handshakes);
}

}  // namespace

}  // namespace arrow::flight::transport::grpc
