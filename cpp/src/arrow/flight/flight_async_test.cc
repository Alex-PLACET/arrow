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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "arrow/array/array_dict.h"
#include "arrow/array/util.h"
#include "arrow/flight/api.h"
#include "arrow/flight/test_definitions.h"
#include "arrow/flight/test_flight_server.h"
#include "arrow/flight/test_util.h"
#include "arrow/flight/types_async.h"
#include "arrow/testing/future_util.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/gtest_util.h"

namespace arrow::flight {
namespace {

using ::testing::HasSubstr;

static constexpr char kExpectedMetadata[] = "foo bar";

template <typename T, typename Fn>
Future<T> WrapSyncOutcome(Fn&& fn) {
  T out{};
  auto st = fn(&out);
  if (!st.ok()) {
    return Future<T>::MakeFinished(std::move(st));
  }
  return Future<T>::MakeFinished(std::move(out));
}

template <typename Fn>
Future<> WrapSyncStatus(Fn&& fn) {
  return Future<>::MakeFinished(fn());
}

template <typename T, typename... Args>
Status MakeAsyncServer(
    const Location& location, std::unique_ptr<AsyncFlightServerBase>* server,
    std::unique_ptr<FlightClient>* client,
    std::function<Status(FlightServerOptions*)> make_server_options,
    std::function<Status(FlightClientOptions*)> make_client_options,
    Args&&... server_args) {
  *server = std::make_unique<T>(std::forward<Args>(server_args)...);
  FlightServerOptions server_options(location);
  RETURN_NOT_OK(make_server_options(&server_options));
  RETURN_NOT_OK((*server)->Init(server_options));
  ARROW_ASSIGN_OR_RAISE(auto real_location,
                        Location::ForGrpcTcp("127.0.0.1", (*server)->port()));
  FlightClientOptions client_options = FlightClientOptions::Defaults();
  RETURN_NOT_OK(make_client_options(&client_options));
  return FlightClient::Connect(real_location, client_options).Value(client);
}

template <typename T, typename... Args>
Status MakeAsyncServer(
    std::unique_ptr<AsyncFlightServerBase>* server, std::unique_ptr<FlightClient>* client,
    std::function<Status(FlightServerOptions*)> make_server_options,
    std::function<Status(FlightClientOptions*)> make_client_options,
    Args&&... server_args) {
  ARROW_ASSIGN_OR_RAISE(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
  return MakeAsyncServer<T>(location, server, client, std::move(make_server_options),
                            std::move(make_client_options),
                            std::forward<Args>(server_args)...);
}

class GetFlightInfoListener : public AsyncListener<FlightInfo> {
 public:
  void OnNext(FlightInfo message) override {
    info = std::move(message);
    counter++;
  }

  void OnFinish(Status status) override {
    ASSERT_FALSE(future.is_finished());
    if (status.ok()) {
      future.MarkFinished(std::move(info));
    } else {
      future.MarkFinished(std::move(status));
    }
  }

  FlightInfo info = FlightInfo(FlightInfo::Data{});
  int counter = 0;
  Future<FlightInfo> future = Future<FlightInfo>::Make();
};

class AsyncAdapterFlightServer : public AsyncFlightServerBase {
 public:
  Future<std::unique_ptr<FlightListing>> ListFlights(const ServerCallContext& context,
                                                     const Criteria* criteria) override {
    return WrapSyncOutcome<std::unique_ptr<FlightListing>>(
        [&](auto* out) { return impl_.ListFlights(context, criteria, out); });
  }

  Future<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const ServerCallContext& context, const FlightDescriptor& request) override {
    return WrapSyncOutcome<std::unique_ptr<FlightInfo>>(
        [&](auto* out) { return impl_.GetFlightInfo(context, request, out); });
  }

  Future<std::unique_ptr<SchemaResult>> GetSchema(
      const ServerCallContext& context, const FlightDescriptor& request) override {
    return WrapSyncOutcome<std::unique_ptr<SchemaResult>>(
        [&](auto* out) { return impl_.GetSchema(context, request, out); });
  }

  Future<std::unique_ptr<FlightDataStream>> DoGet(const ServerCallContext& context,
                                                  const Ticket& request) override {
    return WrapSyncOutcome<std::unique_ptr<FlightDataStream>>(
        [&](auto* out) { return impl_.DoGet(context, request, out); });
  }

  Future<> DoPut(const ServerCallContext& context,
                 std::unique_ptr<FlightMessageReader> reader,
                 std::unique_ptr<FlightMetadataWriter> writer) override {
    return WrapSyncStatus(
        [&] { return impl_.DoPut(context, std::move(reader), std::move(writer)); });
  }

  Future<> DoExchange(const ServerCallContext& context,
                      std::unique_ptr<FlightMessageReader> reader,
                      std::unique_ptr<FlightMessageWriter> writer) override {
    return WrapSyncStatus([&] {
      return impl_.DoExchange(context, std::move(reader), std::move(writer));
    });
  }

  Future<std::unique_ptr<ResultStream>> DoAction(const ServerCallContext& context,
                                                 const Action& action) override {
    return WrapSyncOutcome<std::unique_ptr<ResultStream>>(
        [&](auto* out) { return impl_.DoAction(context, action, out); });
  }

  Future<std::vector<ActionType>> ListActions(const ServerCallContext& context) override {
    return WrapSyncOutcome<std::vector<ActionType>>(
        [&](auto* out) { return impl_.ListActions(context, out); });
  }

 private:
  TestFlightServer impl_;
};

class AsyncDoPutTestServer : public AsyncFlightServerBase {
 public:
  Future<> DoPut(const ServerCallContext&,
                 std::unique_ptr<FlightMessageReader> reader,
                 std::unique_ptr<FlightMetadataWriter> writer) override {
    descriptor_ = reader->descriptor();

    if (descriptor_.type == FlightDescriptor::DescriptorType::CMD &&
        descriptor_.cmd == "TestUndrained") {
      return Future<>::MakeFinished();
    }

    int counter = 0;
    FlightStreamChunk chunk;
    while (true) {
      ARROW_ASSIGN_OR_RAISE(chunk, reader->Next());
      if (!chunk.data) {
        break;
      }
      if (counter % 2 == 1) {
        if (!chunk.app_metadata) {
          return Future<>::MakeFinished(Status::Invalid("Expected app_metadata"));
        }
        if (chunk.app_metadata->ToString() != std::to_string(counter)) {
          return Future<>::MakeFinished(
              Status::Invalid("Expected app_metadata to be ", counter, " but got ",
                              chunk.app_metadata->ToString()));
        }
      } else if (chunk.app_metadata) {
        return Future<>::MakeFinished(Status::Invalid("Expected no app_metadata"));
      }
      batches_.push_back(std::move(chunk.data));
      auto buffer = Buffer::FromString(std::to_string(counter));
      RETURN_NOT_OK(writer->WriteMetadata(*buffer));
      counter++;
    }

    if (!chunk.app_metadata) {
      return Future<>::MakeFinished(
          Status::Invalid("Expected app_metadata at end of stream (#1)"));
    }
    if (chunk.app_metadata->ToString() != kExpectedMetadata) {
      return Future<>::MakeFinished(Status::Invalid(
          "Expected app_metadata to be ", kExpectedMetadata, " but got ",
          chunk.app_metadata->ToString()));
    }
    return Future<>::MakeFinished();
  }

  FlightDescriptor descriptor_;
  RecordBatchVector batches_;
};

class AsyncAppMetadataTestServer : public AsyncFlightServerBase {
 public:
  Future<std::unique_ptr<FlightDataStream>> DoGet(const ServerCallContext& context,
                                                  const Ticket& request) override {
    return WrapSyncOutcome<std::unique_ptr<FlightDataStream>>(
        [&](auto* out) { return impl_.DoGet(context, request, out); });
  }

  Future<> DoPut(const ServerCallContext& context,
                 std::unique_ptr<FlightMessageReader> reader,
                 std::unique_ptr<FlightMetadataWriter> writer) override {
    return WrapSyncStatus(
        [&] { return impl_.DoPut(context, std::move(reader), std::move(writer)); });
  }

 private:
  AppMetadataTestServer impl_;
};

class AsyncIpcOptionsTestServer : public AsyncFlightServerBase {
 public:
  Future<std::unique_ptr<FlightDataStream>> DoGet(const ServerCallContext&,
                                                  const Ticket&) override {
    RecordBatchVector batches;
    RETURN_NOT_OK(ExampleNestedBatches(&batches));
    ARROW_ASSIGN_OR_RAISE(auto reader, RecordBatchReader::Make(batches));
    return Future<std::unique_ptr<FlightDataStream>>::MakeFinished(
        std::make_unique<RecordBatchStream>(reader));
  }

  Future<> DoPut(const ServerCallContext&, std::unique_ptr<FlightMessageReader> reader,
                 std::unique_ptr<FlightMetadataWriter> writer) override {
    int counter = 0;
    while (true) {
      ARROW_ASSIGN_OR_RAISE(FlightStreamChunk chunk, reader->Next());
      if (chunk.data == nullptr) {
        break;
      }
      counter++;
    }
    auto metadata = Buffer::FromString(std::to_string(counter));
    return Future<>::MakeFinished(writer->WriteMetadata(*metadata));
  }

  Future<> DoExchange(const ServerCallContext&,
                      std::unique_ptr<FlightMessageReader> reader,
                      std::unique_ptr<FlightMessageWriter> writer) override {
    auto options = ipc::IpcWriteOptions::Defaults();
    options.max_recursion_depth = 1;
    bool begun = false;
    while (true) {
      ARROW_ASSIGN_OR_RAISE(FlightStreamChunk chunk, reader->Next());
      if (!chunk.data && !chunk.app_metadata) {
        break;
      }
      if (!begun && chunk.data) {
        begun = true;
        RETURN_NOT_OK(writer->Begin(chunk.data->schema(), options));
      }
      if (chunk.data && chunk.app_metadata) {
        RETURN_NOT_OK(writer->WriteWithMetadata(*chunk.data, chunk.app_metadata));
      } else if (chunk.data) {
        RETURN_NOT_OK(writer->WriteRecordBatch(*chunk.data));
      } else if (chunk.app_metadata) {
        RETURN_NOT_OK(writer->WriteMetadata(chunk.app_metadata));
      }
    }
    return Future<>::MakeFinished();
  }
};

class SyncThreadScalingTestServer : public FlightServerBase {
 public:
  Status GetFlightInfo(const ServerCallContext&, const FlightDescriptor& request,
                       std::unique_ptr<FlightInfo>* info) override {
    ARROW_ASSIGN_OR_RAISE(auto flight_info,
                          FlightInfo::Make(ExampleStringSchema(), request, {}, -1, -1));
    *info = std::make_unique<FlightInfo>(std::move(flight_info));
    return Status::OK();
  }
};

class AsyncThreadScalingTestServer : public AsyncFlightServerBase {
 public:
  Future<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const ServerCallContext&, const FlightDescriptor& request) override {
    auto maybe_info = FlightInfo::Make(ExampleStringSchema(), request, {}, -1, -1);
    if (!maybe_info.ok()) {
      return Future<std::unique_ptr<FlightInfo>>::MakeFinished(maybe_info.status());
    }
    auto info = std::move(maybe_info).ValueUnsafe();
    return Future<std::unique_ptr<FlightInfo>>::MakeFinished(
        std::make_unique<FlightInfo>(std::move(info)));
  }
};

#ifdef __linux__
::arrow::Result<int> GetProcessThreadCount() {
  std::ifstream status("/proc/self/status");
  if (!status) {
    return Status::IOError("Could not open /proc/self/status");
  }
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0) {
      auto value = line.substr(std::strlen("Threads:"));
      return std::stoi(value);
    }
  }
  return Status::Invalid("Could not find thread count in /proc/self/status");
}

template <typename ServerType>
::arrow::Result<int> MeasureThreadDeltaForConnections(int num_clients) {
  ARROW_ASSIGN_OR_RAISE(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
  FlightServerOptions options(location);
  auto server = std::make_unique<ServerType>();
  RETURN_NOT_OK(server->Init(options));

  std::vector<std::unique_ptr<FlightClient>> clients;
  clients.reserve(num_clients);
  ARROW_ASSIGN_OR_RAISE(auto client_location,
                        Location::ForGrpcTcp("127.0.0.1", server->port()));
  ARROW_ASSIGN_OR_RAISE(int before_threads, GetProcessThreadCount());

  FlightDescriptor descriptor = FlightDescriptor::Path({"thread-scaling"});
  for (int i = 0; i < num_clients; ++i) {
    ARROW_ASSIGN_OR_RAISE(auto client, FlightClient::Connect(client_location));
    ARROW_ASSIGN_OR_RAISE(auto info, client->GetFlightInfo(descriptor));
    EXPECT_EQ(descriptor, info->descriptor());
    clients.push_back(std::move(client));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  ARROW_ASSIGN_OR_RAISE(int after_threads, GetProcessThreadCount());

  for (auto& client : clients) {
    RETURN_NOT_OK(client->Close());
  }
  RETURN_NOT_OK(server->Shutdown());
  RETURN_NOT_OK(server->Wait());
  return after_threads - before_threads;
}
#endif

class AsyncConnectivityTest : public ::testing::Test {
 protected:
  void TestGetPort() {
    auto server = std::make_unique<AsyncAdapterFlightServer>();
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    FlightServerOptions options(location);
    ASSERT_OK(server->Init(options));
    ASSERT_GT(server->port(), 0);
    ASSERT_OK(server->Shutdown());
    ASSERT_OK(server->Wait());
  }

  void TestBuilderHook() {
    auto server = std::make_unique<AsyncAdapterFlightServer>();
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    FlightServerOptions options(location);
    bool builder_hook_run = false;
    options.builder_hook = [&builder_hook_run](void* builder) {
      ASSERT_NE(nullptr, builder);
      builder_hook_run = true;
    };
    ASSERT_OK(server->Init(options));
    ASSERT_TRUE(builder_hook_run);
    ASSERT_GT(server->port(), 0);
    ASSERT_OK(server->Shutdown());
    ASSERT_OK(server->Wait());
  }

  void TestShutdown() {
    constexpr int kIterations = 10;
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    for (int i = 0; i < kIterations; ++i) {
      auto server = std::make_unique<AsyncAdapterFlightServer>();
      FlightServerOptions options(location);
      ASSERT_OK(server->Init(options));
      ASSERT_GT(server->port(), 0);
      std::thread t([&]() { ASSERT_OK(server->Serve()); });
      ASSERT_OK(server->Shutdown());
      ASSERT_OK(server->Wait());
      t.join();
    }
  }

  void TestShutdownWithDeadline() {
    auto server = std::make_unique<AsyncAdapterFlightServer>();
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    FlightServerOptions options(location);
    ASSERT_OK(server->Init(options));
    ASSERT_GT(server->port(), 0);
    auto deadline = std::chrono::system_clock::now() + std::chrono::microseconds(10);
    ASSERT_OK(server->Shutdown(&deadline));
    ASSERT_OK(server->Wait());
  }

  void TestBrokenConnection() {
    auto server = std::make_unique<AsyncAdapterFlightServer>();
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    FlightServerOptions options(location);
    ASSERT_OK(server->Init(options));

    std::unique_ptr<FlightClient> client;
    ASSERT_OK_AND_ASSIGN(location, Location::ForGrpcTcp("127.0.0.1", server->port()));
    ASSERT_OK_AND_ASSIGN(client, FlightClient::Connect(location));

    ASSERT_OK(server->Shutdown());
    ASSERT_OK(server->Wait());

    auto status = client->GetFlightInfo(FlightDescriptor::Command(""));
    ASSERT_NOT_OK(status);
    ASSERT_THAT(status.status().code(),
                ::testing::AnyOf(StatusCode::IOError, StatusCode::UnknownError));
  }
};

#define ARROW_FLIGHT_ASYNC_TEST_CONNECTIVITY(FIXTURE) \
  TEST_F(FIXTURE, GetPort) { TestGetPort(); }         \
  TEST_F(FIXTURE, BuilderHook) { TestBuilderHook(); } \
  TEST_F(FIXTURE, Shutdown) { TestShutdown(); }       \
  TEST_F(FIXTURE, ShutdownWithDeadline) {             \
    TestShutdownWithDeadline();                       \
  }                                                   \
  TEST_F(FIXTURE, BrokenConnection) { TestBrokenConnection(); }

class AsyncDataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    server_ = std::make_unique<AsyncAdapterFlightServer>();
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    FlightServerOptions options(location);
    ASSERT_OK(server_->Init(options));
    ASSERT_OK(ConnectClient());
  }

  void TearDown() override {
    ASSERT_OK(client_->Close());
    ASSERT_OK(server_->Shutdown());
    ASSERT_OK(server_->Wait());
  }

  Status ConnectClient() {
    ARROW_ASSIGN_OR_RAISE(auto location, Location::ForGrpcTcp("127.0.0.1", server_->port()));
    ARROW_ASSIGN_OR_RAISE(client_, FlightClient::Connect(location));
    return Status::OK();
  }

  void CheckDoGet(const FlightDescriptor& descr, const RecordBatchVector& expected_batches,
                  std::function<void(const std::vector<FlightEndpoint>&)> check_endpoints) {
    auto expected_schema = expected_batches[0]->schema();

    ASSERT_OK_AND_ASSIGN(auto info, client_->GetFlightInfo(descr));
    check_endpoints(info->endpoints());

    auto listener = std::make_shared<GetFlightInfoListener>();
    client_->GetFlightInfoAsync(descr, listener);
    ASSERT_FINISHES_OK(listener->future);
    ASSERT_EQ(1, listener->counter);
    check_endpoints(listener->future.MoveResult()->endpoints());

    ipc::DictionaryMemo dict_memo;
    ASSERT_OK_AND_ASSIGN(auto schema, info->GetSchema(&dict_memo));
    AssertSchemaEqual(*expected_schema, *schema);

    Ticket ticket = info->endpoints()[0].ticket;
    CheckDoGet(ticket, expected_batches);
  }

  void CheckDoGet(const Ticket& ticket, const RecordBatchVector& expected_batches) {
    auto num_batches = static_cast<int>(expected_batches.size());
    ASSERT_GE(num_batches, 2);

    ASSERT_OK_AND_ASSIGN(auto stream, client_->DoGet(ticket));
    ASSERT_OK_AND_ASSIGN(auto stream2, client_->DoGet(ticket));
    ASSERT_OK_AND_ASSIGN(auto reader, MakeRecordBatchReader(std::move(stream2)));

    std::shared_ptr<RecordBatch> batch;
    for (int i = 0; i < num_batches; ++i) {
      ASSERT_OK_AND_ASSIGN(auto chunk, stream->Next());
      ASSERT_OK(reader->ReadNext(&batch));
      ASSERT_NE(nullptr, chunk.data);
      ASSERT_NE(nullptr, batch);
#if !defined(__MINGW32__)
      ASSERT_BATCHES_EQUAL(*expected_batches[i], *chunk.data);
      ASSERT_BATCHES_EQUAL(*expected_batches[i], *batch);
#else
      ASSERT_BATCHES_APPROX_EQUAL(*expected_batches[i], *chunk.data);
      ASSERT_BATCHES_APPROX_EQUAL(*expected_batches[i], *batch);
#endif
    }

    ASSERT_OK_AND_ASSIGN(auto chunk, stream->Next());
    ASSERT_OK(reader->ReadNext(&batch));
    ASSERT_EQ(nullptr, chunk.data);
    ASSERT_EQ(nullptr, batch);
  }

  void TestDoGetInts();
  void TestDoGetFloats();
  void TestDoGetDicts();
  void TestDoGetLargeBatch();
  void TestFlightDataStreamError();
  void TestOverflowServerBatch();
  void TestOverflowClientBatch();
  void TestDoExchange();
  void TestDoExchangeNoData();
  void TestDoExchangeWriteOnlySchema();
  void TestDoExchangeGet();
  void TestDoExchangePut();
  void TestDoExchangeEcho();
  void TestDoExchangeTotal();
  void TestDoExchangeError();
  void TestDoExchangeConcurrency();
  void TestDoExchangeUndrained();
  void TestIssue5095();

  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<AsyncFlightServerBase> server_;
};

void AsyncDataTest::TestDoGetInts() {
  auto descr = FlightDescriptor::Path({"examples", "ints"});
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleIntBatches(&expected_batches));
  CheckDoGet(descr, expected_batches, [](const std::vector<FlightEndpoint>& endpoints) {
    ASSERT_EQ(2, endpoints.size());
    ASSERT_EQ(Ticket{"ticket-ints-1"}, endpoints[0].ticket);
  });
}

void AsyncDataTest::TestDoGetFloats() {
  auto descr = FlightDescriptor::Path({"examples", "floats"});
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleFloatBatches(&expected_batches));
  CheckDoGet(descr, expected_batches, [](const std::vector<FlightEndpoint>& endpoints) {
    ASSERT_EQ(1, endpoints.size());
    ASSERT_EQ(Ticket{"ticket-floats-1"}, endpoints[0].ticket);
  });
}

void AsyncDataTest::TestDoGetDicts() {
  auto descr = FlightDescriptor::Path({"examples", "dicts"});
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleDictBatches(&expected_batches));
  CheckDoGet(descr, expected_batches, [](const std::vector<FlightEndpoint>& endpoints) {
    ASSERT_EQ(1, endpoints.size());
    ASSERT_EQ(Ticket{"ticket-dicts-1"}, endpoints[0].ticket);
  });
}

void AsyncDataTest::TestDoGetLargeBatch() {
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleLargeBatches(&expected_batches));
  CheckDoGet(Ticket{"ticket-large-batch-1"}, expected_batches);
}

void AsyncDataTest::TestFlightDataStreamError() {
  ASSERT_OK_AND_ASSIGN(auto stream, client_->DoGet(Ticket{"ticket-stream-error"}));
  Status status;
  while (true) {
    FlightStreamChunk chunk;
    status = stream->Next().Value(&chunk);
    if (!chunk.data || !status.ok()) {
      break;
    }
  }
  EXPECT_RAISES_WITH_MESSAGE_THAT(IOError, HasSubstr("Expected error"), status);
}

void AsyncDataTest::TestOverflowServerBatch() {
  {
    ASSERT_OK_AND_ASSIGN(auto stream, client_->DoGet(Ticket{"ARROW-13253-DoGet-Batch"}));
    EXPECT_RAISES_WITH_MESSAGE_THAT(
        Invalid, HasSubstr("Cannot send record batches exceeding 2GiB yet"),
        stream->Next());
  }
  {
    ASSERT_OK_AND_ASSIGN(auto exchange,
                         client_->DoExchange(FlightDescriptor::Command("large_batch")));
    RecordBatchVector batches;
    EXPECT_RAISES_WITH_MESSAGE_THAT(
        Invalid, HasSubstr("Cannot send record batches exceeding 2GiB yet"),
        exchange.reader->ToRecordBatches().Value(&batches));
    ARROW_UNUSED(exchange.writer->Close());
  }
}

void AsyncDataTest::TestOverflowClientBatch() {
  ASSERT_OK_AND_ASSIGN(auto batch, VeryLargeBatch());
  {
    ASSERT_OK_AND_ASSIGN(auto do_put_result,
                         client_->DoPut(FlightDescriptor::Path({""}), batch->schema()));
    EXPECT_RAISES_WITH_MESSAGE_THAT(
        Invalid, HasSubstr("Cannot send record batches exceeding 2GiB yet"),
        do_put_result.writer->WriteRecordBatch(*batch));
    ASSERT_OK(do_put_result.writer->Close());
  }
  {
    ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("counter")));
    ASSERT_OK(exchange.writer->Begin(batch->schema()));
    EXPECT_RAISES_WITH_MESSAGE_THAT(
        Invalid, HasSubstr("Cannot send record batches exceeding 2GiB yet"),
        exchange.writer->WriteRecordBatch(*batch));
    ASSERT_OK(exchange.writer->Close());
  }
}

void AsyncDataTest::TestDoExchange() {
  auto a1 = ArrayFromJSON(int32(), "[4, 5, 6, null]");
  auto schema = arrow::schema({field("f1", a1->type())});
  RecordBatchVector batches = {RecordBatch::Make(schema, a1->length(), {a1})};
  ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("counter")));
  ASSERT_OK(exchange.writer->Begin(schema));
  for (const auto& batch : batches) {
    ASSERT_OK(exchange.writer->WriteRecordBatch(*batch));
  }
  ASSERT_OK(exchange.writer->DoneWriting());
  ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
  ASSERT_NE(nullptr, chunk.app_metadata);
  ASSERT_EQ(nullptr, chunk.data);
  ASSERT_EQ("1", chunk.app_metadata->ToString());
  ASSERT_OK_AND_ASSIGN(auto server_schema, exchange.reader->GetSchema());
  AssertSchemaEqual(schema, server_schema);
  for (const auto& batch : batches) {
    ASSERT_OK_AND_ASSIGN(chunk, exchange.reader->Next());
    ASSERT_BATCHES_EQUAL(*batch, *chunk.data);
  }
  ASSERT_OK(exchange.writer->Close());
}

void AsyncDataTest::TestDoExchangeNoData() {
  ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("counter")));
  ASSERT_OK(exchange.writer->DoneWriting());
  ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
  ASSERT_EQ(nullptr, chunk.data);
  ASSERT_NE(nullptr, chunk.app_metadata);
  ASSERT_EQ("0", chunk.app_metadata->ToString());
  ASSERT_OK(exchange.writer->Close());
}

void AsyncDataTest::TestDoExchangeWriteOnlySchema() {
  ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("counter")));
  auto schema = arrow::schema({field("f1", arrow::int32())});
  ASSERT_OK(exchange.writer->Begin(schema));
  ASSERT_OK(exchange.writer->WriteMetadata(Buffer::FromString("foo")));
  ASSERT_OK(exchange.writer->DoneWriting());
  ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
  ASSERT_EQ(nullptr, chunk.data);
  ASSERT_NE(nullptr, chunk.app_metadata);
  ASSERT_EQ("0", chunk.app_metadata->ToString());
  ASSERT_OK(exchange.writer->Close());
}

void AsyncDataTest::TestDoExchangeGet() {
  ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("get")));
  ASSERT_OK_AND_ASSIGN(auto server_schema, exchange.reader->GetSchema());
  AssertSchemaEqual(*ExampleIntSchema(), *server_schema);
  RecordBatchVector batches;
  ASSERT_OK(ExampleIntBatches(&batches));
  for (const auto& batch : batches) {
    ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
    ASSERT_NE(nullptr, chunk.data);
    AssertBatchesEqual(*batch, *chunk.data);
  }
  ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
  ASSERT_EQ(nullptr, chunk.data);
  ASSERT_EQ(nullptr, chunk.app_metadata);
  ASSERT_OK(exchange.writer->Close());
}

void AsyncDataTest::TestDoExchangePut() {
  ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("put")));
  ASSERT_OK(exchange.writer->Begin(ExampleIntSchema()));
  RecordBatchVector batches;
  ASSERT_OK(ExampleIntBatches(&batches));
  for (const auto& batch : batches) {
    ASSERT_OK(exchange.writer->WriteRecordBatch(*batch));
  }
  ASSERT_OK(exchange.writer->DoneWriting());
  ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
  ASSERT_NE(nullptr, chunk.app_metadata);
  AssertBufferEqual(*chunk.app_metadata, "done");
  ASSERT_OK_AND_ASSIGN(chunk, exchange.reader->Next());
  ASSERT_EQ(nullptr, chunk.data);
  ASSERT_EQ(nullptr, chunk.app_metadata);
  ASSERT_OK(exchange.writer->Close());
}

void AsyncDataTest::TestDoExchangeEcho() {
  ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("echo")));
  ASSERT_OK(exchange.writer->Begin(ExampleIntSchema()));
  RecordBatchVector batches;
  ASSERT_OK(ExampleIntBatches(&batches));
  for (const auto& batch : batches) {
    ASSERT_OK(exchange.writer->WriteRecordBatch(*batch));
    ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
    ASSERT_NE(nullptr, chunk.data);
    ASSERT_EQ(nullptr, chunk.app_metadata);
    AssertBatchesEqual(*batch, *chunk.data);
  }
  ASSERT_OK(exchange.writer->WriteMetadata(Buffer::FromString("meta")));
  ASSERT_OK_AND_ASSIGN(auto metadata_only, exchange.reader->Next());
  ASSERT_EQ(nullptr, metadata_only.data);
  ASSERT_EQ("meta", metadata_only.app_metadata->ToString());
  ASSERT_OK(exchange.writer->Close());
}

void AsyncDataTest::TestDoExchangeTotal() {
  ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("total")));
  auto schema = arrow::schema({field("a", int64()), field("b", int64())});
  auto batch1 = RecordBatchFromJSON(schema, "[[1, 2], [4, null]]");
  auto batch2 = RecordBatchFromJSON(schema, "[[3, 5]]");
  ASSERT_OK(exchange.writer->Begin(schema));
  ASSERT_OK(exchange.writer->WriteRecordBatch(*batch1));
  ASSERT_OK_AND_ASSIGN(auto chunk1, exchange.reader->Next());
  ASSERT_BATCHES_EQUAL(*RecordBatchFromJSON(schema, "[[5, 2]]"), *chunk1.data);
  ASSERT_OK(exchange.writer->WriteRecordBatch(*batch2));
  ASSERT_OK_AND_ASSIGN(auto chunk2, exchange.reader->Next());
  ASSERT_BATCHES_EQUAL(*RecordBatchFromJSON(schema, "[[8, 7]]"), *chunk2.data);
  ASSERT_OK(exchange.writer->Close());
}

void AsyncDataTest::TestDoExchangeError() {
  {
    ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("error")));
    EXPECT_RAISES_WITH_MESSAGE_THAT(NotImplemented, HasSubstr("Expected error"),
                                    exchange.writer->Close());
  }
  {
    ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("error")));
    EXPECT_RAISES_WITH_MESSAGE_THAT(NotImplemented, HasSubstr("Expected error"),
                                    exchange.reader->Next());
    ARROW_UNUSED(exchange.writer->Close());
  }
}

void AsyncDataTest::TestDoExchangeConcurrency() {
  ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("echo")));
  ASSERT_OK(exchange.writer->Begin(ExampleIntSchema()));
  RecordBatchVector batches;
  ASSERT_OK(ExampleIntBatches(&batches));
  std::thread writer_thread([&] {
    for (const auto& batch : batches) {
      ASSERT_OK(exchange.writer->WriteRecordBatch(*batch));
    }
    ASSERT_OK(exchange.writer->DoneWriting());
  });
  for (const auto& batch : batches) {
    ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
    ASSERT_BATCHES_EQUAL(*batch, *chunk.data);
  }
  writer_thread.join();
  ASSERT_OK(exchange.writer->Close());
}

void AsyncDataTest::TestDoExchangeUndrained() {
  ASSERT_OK_AND_ASSIGN(auto exchange,
                       client_->DoExchange(FlightDescriptor::Command("TestUndrained")));
  ASSERT_OK(exchange.writer->Begin(ExampleIntSchema()));
  RecordBatchVector batches;
  ASSERT_OK(ExampleIntBatches(&batches));
  for (const auto& batch : batches) {
    ARROW_UNUSED(exchange.writer->WriteRecordBatch(*batch));
  }
  ASSERT_OK(exchange.writer->Close());
  CheckDoGet(Ticket{"ticket-ints-1"}, batches);
}

void AsyncDataTest::TestIssue5095() {
  {
    EXPECT_RAISES_WITH_MESSAGE_THAT(UnknownError, HasSubstr("Server-side error"),
                                    client_->DoGet(Ticket{"ARROW-5095-fail"}));
  }
  {
    EXPECT_RAISES_WITH_MESSAGE_THAT(KeyError, HasSubstr("No data"),
                                    client_->DoGet(Ticket{"ARROW-5095-success"}));
  }
}

#define ARROW_FLIGHT_ASYNC_TEST_DATA(FIXTURE)                             \
  TEST_F(FIXTURE, TestDoGetInts) { TestDoGetInts(); }                     \
  TEST_F(FIXTURE, TestDoGetFloats) { TestDoGetFloats(); }                 \
  TEST_F(FIXTURE, TestDoGetDicts) { TestDoGetDicts(); }                   \
  TEST_F(FIXTURE, TestDoGetLargeBatch) { TestDoGetLargeBatch(); }         \
  TEST_F(FIXTURE, TestFlightDataStreamError) { TestFlightDataStreamError(); } \
  TEST_F(FIXTURE, TestOverflowServerBatch) { TestOverflowServerBatch(); } \
  TEST_F(FIXTURE, TestOverflowClientBatch) { TestOverflowClientBatch(); } \
  TEST_F(FIXTURE, TestDoExchange) { TestDoExchange(); }                   \
  TEST_F(FIXTURE, TestDoExchangeNoData) { TestDoExchangeNoData(); }       \
  TEST_F(FIXTURE, TestDoExchangeWriteOnlySchema) {                        \
    TestDoExchangeWriteOnlySchema();                                      \
  }                                                                       \
  TEST_F(FIXTURE, TestDoExchangeGet) { TestDoExchangeGet(); }             \
  TEST_F(FIXTURE, TestDoExchangePut) { TestDoExchangePut(); }             \
  TEST_F(FIXTURE, TestDoExchangeEcho) { TestDoExchangeEcho(); }           \
  TEST_F(FIXTURE, TestDoExchangeTotal) { TestDoExchangeTotal(); }         \
  TEST_F(FIXTURE, TestDoExchangeError) { TestDoExchangeError(); }         \
  TEST_F(FIXTURE, TestDoExchangeConcurrency) { TestDoExchangeConcurrency(); } \
  TEST_F(FIXTURE, TestDoExchangeUndrained) { TestDoExchangeUndrained(); } \
  TEST_F(FIXTURE, TestIssue5095) { TestIssue5095(); }

class AsyncDoPutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    ASSERT_OK(MakeAsyncServer<AsyncDoPutTestServer>(
        location, &server_, &client_,
        [](FlightServerOptions*) { return Status::OK(); },
        [](FlightClientOptions*) { return Status::OK(); }));
  }

  void TearDown() override {
    ASSERT_OK(client_->Close());
    ASSERT_OK(server_->Shutdown());
    ASSERT_OK(server_->Wait());
    static_cast<AsyncDoPutTestServer*>(server_.get())->batches_.clear();
  }

  void CheckBatches(const FlightDescriptor& expected_descriptor,
                    const RecordBatchVector& expected_batches) {
    auto* do_put_server = static_cast<AsyncDoPutTestServer*>(server_.get());
    ASSERT_EQ(do_put_server->descriptor_, expected_descriptor);
    ASSERT_EQ(do_put_server->batches_.size(), expected_batches.size());
    for (size_t i = 0; i < expected_batches.size(); ++i) {
      ASSERT_BATCHES_EQUAL(*do_put_server->batches_[i], *expected_batches[i]);
    }
  }

  void CheckDoPut(const FlightDescriptor& descr, const std::shared_ptr<Schema>& schema,
                  const RecordBatchVector& batches) {
    ASSERT_OK_AND_ASSIGN(auto do_put_result, client_->DoPut(descr, schema));
    auto writer = std::move(do_put_result.writer);
    auto reader = std::move(do_put_result.reader);

    std::thread reader_thread([&reader, &batches]() {
      for (size_t i = 0; i < batches.size(); ++i) {
        std::shared_ptr<Buffer> out;
        ASSERT_OK(reader->ReadMetadata(&out));
      }
    });

    int64_t counter = 0;
    for (const auto& batch : batches) {
      if (counter % 2 == 0) {
        ASSERT_OK(writer->WriteRecordBatch(*batch));
      } else {
        ASSERT_OK(writer->WriteWithMetadata(*batch,
                                            Buffer::FromString(std::to_string(counter))));
      }
      counter++;
    }
    ASSERT_OK(writer->WriteMetadata(Buffer::FromString(kExpectedMetadata)));
    ASSERT_OK(writer->DoneWriting());
    reader_thread.join();
    ASSERT_OK(writer->Close());
    CheckBatches(descr, batches);
  }

  void TestInts();
  void TestFloats();
  void TestEmptyBatch();
  void TestDicts();
  void TestLargeBatch();
  void TestSizeLimit();
  void TestUndrained();

  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<AsyncFlightServerBase> server_;
};

void AsyncDoPutTest::TestInts() {
  auto descr = FlightDescriptor::Path({"ints"});
  RecordBatchVector batches;
  auto a0 = ArrayFromJSON(int8(), "[0, 1, 127, -128, null]");
  auto a1 = ArrayFromJSON(uint8(), "[0, 1, 127, 255, null]");
  auto a2 = ArrayFromJSON(int16(), "[0, 258, 32767, -32768, null]");
  auto a3 = ArrayFromJSON(uint16(), "[0, 258, 32767, 65535, null]");
  auto a4 = ArrayFromJSON(int32(), "[0, 65538, 2147483647, -2147483648, null]");
  auto a5 = ArrayFromJSON(uint32(), "[0, 65538, 2147483647, 4294967295, null]");
  auto a6 = ArrayFromJSON(
      int64(), "[0, 4294967298, 9223372036854775807, -9223372036854775808, null]");
  auto a7 = ArrayFromJSON(
      uint64(), "[0, 4294967298, 9223372036854775807, 18446744073709551615, null]");
  auto schema = arrow::schema({field("f0", a0->type()), field("f1", a1->type()),
                               field("f2", a2->type()), field("f3", a3->type()),
                               field("f4", a4->type()), field("f5", a5->type()),
                               field("f6", a6->type()), field("f7", a7->type())});
  batches.push_back(
      RecordBatch::Make(schema, a0->length(), {a0, a1, a2, a3, a4, a5, a6, a7}));
  CheckDoPut(descr, schema, batches);
}

void AsyncDoPutTest::TestFloats() {
  auto descr = FlightDescriptor::Path({"floats"});
  RecordBatchVector batches;
  auto a0 = ArrayFromJSON(float32(), "[0, 1.2, -3.4, 5.6, null]");
  auto a1 = ArrayFromJSON(float64(), "[0, 1.2, -3.4, 5.6, null]");
  auto schema = arrow::schema({field("f0", a0->type()), field("f1", a1->type())});
  batches.push_back(RecordBatch::Make(schema, a0->length(), {a0, a1}));
  CheckDoPut(descr, schema, batches);
}

void AsyncDoPutTest::TestEmptyBatch() {
  auto descr = FlightDescriptor::Path({"ints"});
  RecordBatchVector batches;
  auto a1 = ArrayFromJSON(int32(), "[]");
  auto schema = arrow::schema({field("f1", a1->type())});
  batches.push_back(RecordBatch::Make(schema, a1->length(), {a1}));
  CheckDoPut(descr, schema, batches);
}

void AsyncDoPutTest::TestDicts() {
  auto descr = FlightDescriptor::Path({"dicts"});
  RecordBatchVector batches;
  auto dict_values = ArrayFromJSON(utf8(), "[\"foo\", \"bar\", \"quux\"]");
  auto ty = dictionary(int8(), dict_values->type());
  auto schema = arrow::schema({field("f1", ty)});
  for (const char* json : {"[1, 0, 1]", "[null]", "[null, 1]"}) {
    auto indices = ArrayFromJSON(int8(), json);
    auto dict_array = std::make_shared<DictionaryArray>(ty, indices, dict_values);
    batches.push_back(RecordBatch::Make(schema, dict_array->length(), {dict_array}));
  }
  CheckDoPut(descr, schema, batches);
}

void AsyncDoPutTest::TestLargeBatch() {
  auto descr = FlightDescriptor::Path({"large-batches"});
  auto schema = ExampleLargeSchema();
  RecordBatchVector batches;
  ASSERT_OK(ExampleLargeBatches(&batches));
  CheckDoPut(descr, schema, batches);
}

void AsyncDoPutTest::TestSizeLimit() {
  const int64_t size_limit = 4096;
  ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", server_->port()));
  auto client_options = FlightClientOptions::Defaults();
  client_options.write_size_limit_bytes = size_limit;
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location, client_options));

  auto descr = FlightDescriptor::Command("simple");
  auto schema = arrow::schema({field("f1", arrow::int64())});
  auto batch = arrow::ConstantArrayGenerator::Zeroes(768, schema);
  auto batch1 = batch->Slice(0, 384);
  auto batch2 = batch->Slice(384);

  ASSERT_OK_AND_ASSIGN(auto do_put_result, client->DoPut(descr, schema));
  const auto status = do_put_result.writer->WriteRecordBatch(*batch);
  EXPECT_RAISES_WITH_MESSAGE_THAT(Invalid, HasSubstr("exceeded soft limit"), status);
  auto detail = FlightWriteSizeStatusDetail::UnwrapStatus(status);
  ASSERT_NE(nullptr, detail);
  ASSERT_EQ(size_limit, detail->limit());
  ASSERT_GT(detail->actual(), size_limit);

  ASSERT_OK(do_put_result.writer->WriteRecordBatch(*batch1));
  ASSERT_OK(do_put_result.writer->WriteWithMetadata(*batch2, Buffer::FromString("1")));
  ASSERT_OK(do_put_result.writer->WriteMetadata(Buffer::FromString(kExpectedMetadata)));
  ASSERT_OK(do_put_result.writer->DoneWriting());
  ASSERT_OK(do_put_result.writer->Close());
  CheckBatches(descr, {batch1, batch2});
}

void AsyncDoPutTest::TestUndrained() {
  auto descr = FlightDescriptor::Command("TestUndrained");
  auto schema = arrow::schema({arrow::field("ints", int64())});
  ASSERT_OK_AND_ASSIGN(auto do_put_result, client_->DoPut(descr, schema));
  auto batch = RecordBatchFromJSON(schema, "[[1], [2], [3], [4]]");
  ARROW_UNUSED(do_put_result.writer->WriteRecordBatch(*batch));
  ARROW_UNUSED(do_put_result.writer->WriteRecordBatch(*batch));
  ARROW_UNUSED(do_put_result.writer->WriteRecordBatch(*batch));
  ARROW_UNUSED(do_put_result.writer->WriteRecordBatch(*batch));
  ASSERT_OK(do_put_result.writer->Close());
  CheckDoPut(FlightDescriptor::Command("foo"), schema, {batch, batch});
}

#define ARROW_FLIGHT_ASYNC_TEST_DO_PUT(FIXTURE)       \
  TEST_F(FIXTURE, TestInts) { TestInts(); }           \
  TEST_F(FIXTURE, TestFloats) { TestFloats(); }       \
  TEST_F(FIXTURE, TestEmptyBatch) { TestEmptyBatch(); } \
  TEST_F(FIXTURE, TestDicts) { TestDicts(); }         \
  TEST_F(FIXTURE, TestLargeBatch) { TestLargeBatch(); } \
  TEST_F(FIXTURE, TestSizeLimit) { TestSizeLimit(); } \
  TEST_F(FIXTURE, TestUndrained) { TestUndrained(); }

class AsyncAppMetadataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    ASSERT_OK(MakeAsyncServer<AsyncAppMetadataTestServer>(
        location, &server_, &client_,
        [](FlightServerOptions*) { return Status::OK(); },
        [](FlightClientOptions*) { return Status::OK(); }));
  }

  void TearDown() override {
    ASSERT_OK(client_->Close());
    ASSERT_OK(server_->Shutdown());
    ASSERT_OK(server_->Wait());
  }

  void TestDoGet();
  void TestDoGetDictionaries();
  void TestDoPut();
  void TestDoPutDictionaries();
  void TestDoPutReadMetadata();

  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<AsyncFlightServerBase> server_;
};

void AsyncAppMetadataTest::TestDoGet() {
  ASSERT_OK_AND_ASSIGN(auto stream, client_->DoGet(Ticket{""}));
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleIntBatches(&expected_batches));
  for (size_t i = 0; i < expected_batches.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(auto chunk, stream->Next());
    ASSERT_NE(nullptr, chunk.data);
    ASSERT_NE(nullptr, chunk.app_metadata);
    ASSERT_BATCHES_EQUAL(*expected_batches[i], *chunk.data);
    ASSERT_EQ(std::to_string(i), chunk.app_metadata->ToString());
  }
  ASSERT_OK_AND_ASSIGN(auto chunk, stream->Next());
  ASSERT_EQ(nullptr, chunk.data);
}

void AsyncAppMetadataTest::TestDoGetDictionaries() {
  ASSERT_OK_AND_ASSIGN(auto stream, client_->DoGet(Ticket{"dicts"}));
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleDictBatches(&expected_batches));
  for (size_t i = 0; i < expected_batches.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(auto chunk, stream->Next());
    ASSERT_NE(nullptr, chunk.data);
    ASSERT_NE(nullptr, chunk.app_metadata);
    ASSERT_BATCHES_EQUAL(*expected_batches[i], *chunk.data);
    ASSERT_EQ(std::to_string(i), chunk.app_metadata->ToString());
  }
  ASSERT_OK_AND_ASSIGN(auto chunk, stream->Next());
  ASSERT_EQ(nullptr, chunk.data);
}

void AsyncAppMetadataTest::TestDoPut() {
  auto schema = ExampleIntSchema();
  ASSERT_OK_AND_ASSIGN(auto do_put_result, client_->DoPut(FlightDescriptor{}, schema));
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleIntBatches(&expected_batches));
  for (size_t i = 0; i < expected_batches.size(); ++i) {
    ASSERT_OK(do_put_result.writer->WriteWithMetadata(
        *expected_batches[i], Buffer::FromString(std::to_string(i))));
  }
  ASSERT_OK(do_put_result.writer->Close());
}

void AsyncAppMetadataTest::TestDoPutDictionaries() {
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleDictBatches(&expected_batches));
  ASSERT_OK_AND_ASSIGN(auto do_put_result,
                       client_->DoPut(FlightDescriptor{}, expected_batches[0]->schema()));
  for (size_t i = 0; i < expected_batches.size(); ++i) {
    ASSERT_OK(do_put_result.writer->WriteWithMetadata(
        *expected_batches[i], Buffer::FromString(std::to_string(i))));
  }
  ASSERT_OK(do_put_result.writer->Close());
}

void AsyncAppMetadataTest::TestDoPutReadMetadata() {
  auto schema = ExampleIntSchema();
  ASSERT_OK_AND_ASSIGN(auto do_put_result, client_->DoPut(FlightDescriptor{}, schema));
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleIntBatches(&expected_batches));
  for (size_t i = 0; i < expected_batches.size(); ++i) {
    ASSERT_OK(do_put_result.writer->WriteWithMetadata(
        *expected_batches[i], Buffer::FromString(std::to_string(i))));
    std::shared_ptr<Buffer> metadata;
    ASSERT_OK(do_put_result.reader->ReadMetadata(&metadata));
    ASSERT_NE(nullptr, metadata);
    ASSERT_EQ(std::to_string(i), metadata->ToString());
  }
  ASSERT_OK(do_put_result.writer->Close());
}

#define ARROW_FLIGHT_ASYNC_TEST_APP_METADATA(FIXTURE)      \
  TEST_F(FIXTURE, TestDoGet) { TestDoGet(); }              \
  TEST_F(FIXTURE, TestDoGetDictionaries) {                 \
    TestDoGetDictionaries();                               \
  }                                                        \
  TEST_F(FIXTURE, TestDoPut) { TestDoPut(); }              \
  TEST_F(FIXTURE, TestDoPutDictionaries) {                 \
    TestDoPutDictionaries();                               \
  }                                                        \
  TEST_F(FIXTURE, TestDoPutReadMetadata) {                 \
    TestDoPutReadMetadata();                               \
  }

class AsyncIpcOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    ASSERT_OK(MakeAsyncServer<AsyncIpcOptionsTestServer>(
        location, &server_, &client_,
        [](FlightServerOptions*) { return Status::OK(); },
        [](FlightClientOptions*) { return Status::OK(); }));
  }

  void TearDown() override {
    ASSERT_OK(client_->Close());
    ASSERT_OK(server_->Shutdown());
    ASSERT_OK(server_->Wait());
  }

  void TestDoGetReadOptions() {
    auto options = FlightCallOptions();
    options.read_options.max_recursion_depth = 1;
    std::unique_ptr<FlightStreamReader> stream;
    ASSERT_OK_AND_ASSIGN(stream, client_->DoGet(options, Ticket{""}));
    ASSERT_RAISES(Invalid, stream->Next());
  }

  void TestDoPutWriteOptions() {
    RecordBatchVector expected_batches;
    ASSERT_OK(ExampleNestedBatches(&expected_batches));
    auto options = FlightCallOptions();
    options.write_options.max_recursion_depth = 1;
    ASSERT_OK_AND_ASSIGN(auto do_put_result,
                         client_->DoPut(options, FlightDescriptor{},
                                        expected_batches[0]->schema()));
    for (const auto& batch : expected_batches) {
      ASSERT_RAISES(Invalid, do_put_result.writer->WriteRecordBatch(*batch));
    }
  }

  void TestDoExchangeClientWriteOptions() {
    auto options = FlightCallOptions();
    options.write_options.max_recursion_depth = 1;
    ASSERT_OK_AND_ASSIGN(auto exchange,
                         client_->DoExchange(options, FlightDescriptor::Command("")));
    RecordBatchVector batches;
    ASSERT_OK(ExampleNestedBatches(&batches));
    ASSERT_OK(exchange.writer->Begin(batches[0]->schema()));
    for (const auto& batch : batches) {
      ASSERT_RAISES(Invalid, exchange.writer->WriteRecordBatch(*batch));
    }
    ASSERT_OK(exchange.writer->DoneWriting());
    ASSERT_OK(exchange.writer->Close());
  }

  void TestDoExchangeClientWriteOptionsBegin() {
    ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("")));
    RecordBatchVector batches;
    ASSERT_OK(ExampleNestedBatches(&batches));
    auto options = ipc::IpcWriteOptions::Defaults();
    options.max_recursion_depth = 1;
    ASSERT_OK(exchange.writer->Begin(batches[0]->schema(), options));
    for (const auto& batch : batches) {
      ASSERT_RAISES(Invalid, exchange.writer->WriteRecordBatch(*batch));
    }
    ASSERT_OK(exchange.writer->DoneWriting());
    ASSERT_OK(exchange.writer->Close());
  }

  void TestDoExchangeServerWriteOptions() {
    ASSERT_OK_AND_ASSIGN(auto exchange, client_->DoExchange(FlightDescriptor::Command("")));
    RecordBatchVector batches;
    ASSERT_OK(ExampleNestedBatches(&batches));
    ASSERT_OK(exchange.writer->Begin(batches[0]->schema()));
    ASSERT_OK(exchange.writer->WriteRecordBatch(*batches[0]));
    ASSERT_OK(exchange.writer->DoneWriting());
    ASSERT_RAISES(Invalid, exchange.writer->Close());
  }

  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<AsyncFlightServerBase> server_;
};

#define ARROW_FLIGHT_ASYNC_TEST_IPC_OPTIONS(FIXTURE)      \
  TEST_F(FIXTURE, TestDoGetReadOptions) {                 \
    TestDoGetReadOptions();                               \
  }                                                       \
  TEST_F(FIXTURE, TestDoPutWriteOptions) {                \
    TestDoPutWriteOptions();                              \
  }                                                       \
  TEST_F(FIXTURE, TestDoExchangeClientWriteOptions) {     \
    TestDoExchangeClientWriteOptions();                   \
  }                                                       \
  TEST_F(FIXTURE, TestDoExchangeClientWriteOptionsBegin) { \
    TestDoExchangeClientWriteOptionsBegin();              \
  }                                                       \
  TEST_F(FIXTURE, TestDoExchangeServerWriteOptions) {     \
    TestDoExchangeServerWriteOptions();                   \
  }

class AsyncRpcCoverageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    server_ = std::make_unique<AsyncAdapterFlightServer>();
    FlightServerOptions options(location);
    ASSERT_OK(server_->Init(options));
    ASSERT_OK_AND_ASSIGN(auto client_location,
                         Location::ForGrpcTcp("127.0.0.1", server_->port()));
    ASSERT_OK_AND_ASSIGN(client_, FlightClient::Connect(client_location));
  }

  void TearDown() override {
    ASSERT_OK(client_->Close());
    ASSERT_OK(server_->Shutdown());
    ASSERT_OK(server_->Wait());
  }

  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<AsyncFlightServerBase> server_;
};

TEST_F(AsyncRpcCoverageTest, ListFlights) {
  ASSERT_OK_AND_ASSIGN(auto listings, client_->ListFlights());
  ASSERT_OK_AND_ASSIGN(auto first, listings->Next());
  ASSERT_NE(nullptr, first);
}

TEST_F(AsyncRpcCoverageTest, GetSchema) {
  auto descr = FlightDescriptor::Path({"examples", "ints"});
  ASSERT_OK_AND_ASSIGN(auto schema_result, client_->GetSchema(descr));
  ipc::DictionaryMemo dict_memo;
  ASSERT_OK_AND_ASSIGN(auto schema, schema_result->GetSchema(&dict_memo));
  AssertSchemaEqual(*ExampleIntSchema(), *schema);
}

TEST_F(AsyncRpcCoverageTest, ListActions) {
  ASSERT_OK_AND_ASSIGN(auto actions, client_->ListActions());
  ASSERT_FALSE(actions.empty());
}

TEST_F(AsyncRpcCoverageTest, DoAction) {
  Action action{"action1", Buffer::FromString("hello")};
  ASSERT_OK_AND_ASSIGN(auto results, client_->DoAction(action));
  ASSERT_OK_AND_ASSIGN(auto first, results->Next());
  ASSERT_NE(nullptr, first);
  ASSERT_EQ("hello-part0", first->body->ToString());
}

class GrpcAsyncServerConnectivityTest : public AsyncConnectivityTest {};
ARROW_FLIGHT_ASYNC_TEST_CONNECTIVITY(GrpcAsyncServerConnectivityTest);

class GrpcAsyncServerDataTest : public AsyncDataTest {};
ARROW_FLIGHT_ASYNC_TEST_DATA(GrpcAsyncServerDataTest);

class GrpcAsyncServerDoPutTest : public AsyncDoPutTest {};
ARROW_FLIGHT_ASYNC_TEST_DO_PUT(GrpcAsyncServerDoPutTest);

class GrpcAsyncServerAppMetadataTest : public AsyncAppMetadataTest {};
ARROW_FLIGHT_ASYNC_TEST_APP_METADATA(GrpcAsyncServerAppMetadataTest);

class GrpcAsyncServerIpcOptionsTest : public AsyncIpcOptionsTest {};
ARROW_FLIGHT_ASYNC_TEST_IPC_OPTIONS(GrpcAsyncServerIpcOptionsTest);

#ifdef __linux__
TEST(AsyncFlightServerTest, UsesFewerThreadsThanSyncServerUnderManyConnections) {
  constexpr int kNumClients = 24;

  ASSERT_OK_AND_ASSIGN(int sync_thread_delta,
                       MeasureThreadDeltaForConnections<SyncThreadScalingTestServer>(
                           kNumClients));
  ASSERT_OK_AND_ASSIGN(
      int async_thread_delta,
      MeasureThreadDeltaForConnections<AsyncThreadScalingTestServer>(kNumClients));

  EXPECT_GT(sync_thread_delta, 0);
  EXPECT_LT(async_thread_delta, sync_thread_delta);
  EXPECT_LE(async_thread_delta, kNumClients / 3 + 4);
}
#endif

}  // namespace
}  // namespace arrow::flight
