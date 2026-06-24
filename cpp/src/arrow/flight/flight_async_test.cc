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

#include <sys/wait.h>
#include <unistd.h>

#include "arrow/array/builder_primitive.h"
#include "arrow/array/array_dict.h"
#include "arrow/array/util.h"
#include "arrow/flight/api.h"
#include "arrow/flight/client_middleware.h"
#include "arrow/flight/test_auth_handlers.h"
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
static constexpr char kAsyncAuthUsername[] = "user";
static constexpr char kAsyncAuthPassword[] = "p4ssw0rd";
static constexpr char kAsyncInvalidAuthUsername[] = "wrong-user";
static constexpr char kAsyncInvalidAuthPassword[] = "wrong-password";

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

Future<> DrainAsyncReader(std::shared_ptr<AsyncFlightMessageReader> reader) {
  return ::arrow::Loop([reader = std::move(reader)]() {
           return reader->Next().Then([](FlightStreamChunk chunk) -> ControlFlow<> {
             if (!chunk.data && !chunk.app_metadata) {
               return Break();
             }
             return Continue{};
           });
         })
      .Then([](const ::arrow::internal::Empty&) { return Status::OK(); });
}

Future<> WriteRecordBatchesAsync(std::shared_ptr<AsyncFlightMessageWriter> writer,
                                 RecordBatchVector batches) {
  auto index = std::make_shared<size_t>(0);
  return ::arrow::Loop([writer = std::move(writer), batches = std::move(batches), index]() {
           if (*index >= batches.size()) {
             return ::arrow::ToFuture(Break());
           }
           return writer->WriteRecordBatch(*batches[*index]).Then(
               [index]() -> ControlFlow<> {
                 ++(*index);
                 return Continue{};
               });
         })
      .Then([](const ::arrow::internal::Empty&) { return Status::OK(); });
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

  Future<> DoPutAsync(const ServerCallContext&,
                      std::unique_ptr<AsyncFlightMessageReader> reader,
                      std::unique_ptr<AsyncFlightMetadataWriter>) override {
    return DrainAsyncReader(
        std::shared_ptr<AsyncFlightMessageReader>(std::move(reader)));
  }

  Future<> DoExchange(const ServerCallContext& context,
                      std::unique_ptr<FlightMessageReader> reader,
                      std::unique_ptr<FlightMessageWriter> writer) override {
    return Future<>::MakeFinished(
        impl_.DoExchange(context, std::move(reader), std::move(writer)));
  }

  Future<> DoExchangeAsync(const ServerCallContext& context,
                           std::unique_ptr<AsyncFlightMessageReader> reader,
                           std::unique_ptr<AsyncFlightMessageWriter> writer) override {
    if (reader->descriptor().type != FlightDescriptor::DescriptorType::CMD) {
      return Future<>::MakeFinished(Status::Invalid("Must provide a command descriptor"));
    }

    auto shared_reader =
        std::shared_ptr<AsyncFlightMessageReader>(std::move(reader));
    auto shared_writer =
        std::shared_ptr<AsyncFlightMessageWriter>(std::move(writer));
    const std::string& cmd = shared_reader->descriptor().cmd;
    if (cmd == "error") {
      return Future<>::MakeFinished(Status::NotImplemented("Expected error"));
    }
    if (cmd == "get") {
      return RunExchangeGetAsync(std::move(shared_writer));
    }
    if (cmd == "put") {
      return RunExchangePutAsync(std::move(shared_reader), std::move(shared_writer));
    }
    if (cmd == "counter") {
      return RunExchangeCounterAsync(std::move(shared_reader), std::move(shared_writer));
    }
    if (cmd == "total") {
      return RunExchangeTotalAsync(std::move(shared_reader), std::move(shared_writer));
    }
    if (cmd == "echo") {
      return RunExchangeEchoAsync(std::move(shared_reader), std::move(shared_writer));
    }
    if (cmd == "TestUndrained") {
      return shared_reader->GetSchema().Then(
          [shared_reader](const std::shared_ptr<Schema>&) {
            return Status::OK();
          });
    }
    return Future<>::MakeFinished(
        Status::NotImplemented("Scenario not implemented: ", cmd));
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
  Future<> RunExchangeGetAsync(std::shared_ptr<AsyncFlightMessageWriter> writer) {
    RecordBatchVector batches;
    auto st = ExampleIntBatches(&batches);
    if (!st.ok()) {
      return Future<>::MakeFinished(std::move(st));
    }
    return writer->Begin(ExampleIntSchema()).Then([writer, batches]() {
      return WriteRecordBatchesAsync(writer, batches);
    });
  }

  Future<> RunExchangePutAsync(std::shared_ptr<AsyncFlightMessageReader> reader,
                               std::shared_ptr<AsyncFlightMessageWriter> writer) {
    RecordBatchVector expected_batches;
    auto st = ExampleIntBatches(&expected_batches);
    if (!st.ok()) {
      return Future<>::MakeFinished(std::move(st));
    }
    auto expected = std::make_shared<RecordBatchVector>(std::move(expected_batches));
    return reader->GetSchema().Then(
        [reader, writer, expected](const std::shared_ptr<Schema>& schema) -> Future<> {
          if (!schema->Equals(ExampleIntSchema(), false)) {
            return Future<>::MakeFinished(Status::Invalid("Schema is not as expected"));
          }
          auto index = std::make_shared<size_t>(0);
          return ::arrow::Loop([reader, expected, index]() {
                   return reader->Next().Then(
                       [expected, index](FlightStreamChunk chunk)
                           -> Future<ControlFlow<>> {
                         if (*index < expected->size()) {
                           if (!chunk.data) {
                             return Future<ControlFlow<>>::MakeFinished(
                                 Status::Invalid("Expected another batch"));
                           }
                           if (!(*expected)[*index]->Equals(*chunk.data)) {
                             return Future<ControlFlow<>>::MakeFinished(
                                 Status::Invalid("Batch does not match"));
                           }
                           ++(*index);
                           return Future<ControlFlow<>>::MakeFinished(Continue{});
                         }
                         if (chunk.data || chunk.app_metadata) {
                           return Future<ControlFlow<>>::MakeFinished(
                               Status::Invalid("Too many batches"));
                         }
                         return ::arrow::ToFuture(Break());
                       });
                 })
              .Then([writer](const ::arrow::internal::Empty&) {
                return writer->WriteMetadata(Buffer::FromString("done"));
              });
        });
  }

  Future<> RunExchangeCounterAsync(std::shared_ptr<AsyncFlightMessageReader> reader,
                                   std::shared_ptr<AsyncFlightMessageWriter> writer) {
    struct State {
      RecordBatchVector batches;
      std::shared_ptr<Schema> schema;
    };
    auto state = std::make_shared<State>();
    return ::arrow::Loop([reader, state]() {
             return reader->Next().Then([state](FlightStreamChunk chunk) -> ControlFlow<> {
               if (!chunk.data && !chunk.app_metadata) {
                 return Break();
               }
               if (chunk.data) {
                 if (!state->schema) {
                   state->schema = chunk.data->schema();
                 }
                 state->batches.push_back(std::move(chunk.data));
               }
               return Continue{};
             });
           })
        .Then([reader, writer, state](const ::arrow::internal::Empty&) -> Future<> {
          return writer->WriteMetadata(
              Buffer::FromString(std::to_string(state->batches.size())))
              .Then([reader, writer, state]() -> Future<> {
                if (state->batches.empty()) {
                  return Future<>::MakeFinished();
                }
                return writer->Begin(state->schema).Then([writer, state]() {
                  return WriteRecordBatchesAsync(writer, state->batches);
                });
              });
        });
  }

  Future<> RunExchangeTotalAsync(std::shared_ptr<AsyncFlightMessageReader> reader,
                                 std::shared_ptr<AsyncFlightMessageWriter> writer) {
    struct State {
      std::shared_ptr<Schema> schema;
      std::vector<int64_t> sums;
    };

    return reader->GetSchema().Then(
        [reader, writer](const std::shared_ptr<Schema>& schema) -> Future<> {
          for (const auto& field : schema->fields()) {
            if (field->type()->id() != Type::INT64) {
              return Future<>::MakeFinished(
                  Status::Invalid("Field is not INT64: ", field->name()));
            }
          }
          auto state = std::make_shared<State>();
          state->schema = schema;
          state->sums.resize(schema->num_fields());
          return writer->Begin(schema).Then([reader, writer, state]() {
            return ::arrow::Loop([reader, writer, state]() {
                     return reader->Next().Then([writer, state](FlightStreamChunk chunk)
                                                    -> Future<ControlFlow<>> {
                       if (!chunk.data && !chunk.app_metadata) {
                         return ::arrow::ToFuture(Break());
                       }
                       if (!chunk.data) {
                         return Future<ControlFlow<>>::MakeFinished(Continue{});
                       }
                       if (!chunk.data->schema()->Equals(state->schema, false)) {
                         return Future<ControlFlow<>>::MakeFinished(
                             Status::Invalid("Schemas are incompatible"));
                       }

                       std::vector<std::shared_ptr<Array>> columns(state->schema->num_fields());
                       for (int i = 0; i < chunk.data->num_columns(); ++i) {
                         auto arr =
                             std::dynamic_pointer_cast<Int64Array>(chunk.data->column(i));
                         if (!arr) {
                           return Future<ControlFlow<>>::MakeFinished(
                               MakeFlightError(FlightStatusCode::Internal,
                                               "Could not cast array"));
                         }
                         for (int64_t row = 0; row < arr->length(); ++row) {
                           if (!arr->IsNull(row)) {
                             state->sums[static_cast<size_t>(i)] += arr->Value(row);
                           }
                         }
                         Int64Builder builder;
                         auto st =
                             builder.Append(state->sums[static_cast<size_t>(i)]);
                         if (!st.ok()) {
                           return Future<ControlFlow<>>::MakeFinished(std::move(st));
                         }
                         st = builder.Finish(&columns[static_cast<size_t>(i)]);
                         if (!st.ok()) {
                           return Future<ControlFlow<>>::MakeFinished(std::move(st));
                         }
                       }

                       auto response =
                           RecordBatch::Make(state->schema, 1, std::move(columns));
                       return writer->WriteRecordBatch(*response).Then(
                           []() -> ControlFlow<> { return Continue{}; });
                     });
                   })
                .Then([](const ::arrow::internal::Empty&) { return Status::OK(); });
          });
        });
  }

  Future<> RunExchangeEchoAsync(std::shared_ptr<AsyncFlightMessageReader> reader,
                                std::shared_ptr<AsyncFlightMessageWriter> writer) {
    auto begun = std::make_shared<bool>(false);
    return ::arrow::Loop([reader, writer, begun]() {
             return reader->Next().Then([writer, begun](FlightStreamChunk chunk)
                                            -> Future<ControlFlow<>> {
               if (!chunk.data && !chunk.app_metadata) {
                 return ::arrow::ToFuture(Break());
               }
               Future<> start = Future<>::MakeFinished();
               if (!*begun && chunk.data) {
                 *begun = true;
                 start = writer->Begin(chunk.data->schema());
               }
               return start.Then([writer, chunk = std::move(chunk)]() mutable
                                     -> Future<ControlFlow<>> {
                 if (chunk.data && chunk.app_metadata) {
                   return writer->WriteWithMetadata(*chunk.data, chunk.app_metadata)
                       .Then([]() -> ControlFlow<> { return Continue{}; });
                 }
                 if (chunk.data) {
                   return writer->WriteRecordBatch(*chunk.data)
                       .Then([]() -> ControlFlow<> { return Continue{}; });
                 }
                 return writer->WriteMetadata(chunk.app_metadata)
                     .Then([]() -> ControlFlow<> { return Continue{}; });
               });
             });
           })
        .Then([](const ::arrow::internal::Empty&) { return Status::OK(); });
  }

  TestFlightServer impl_;
};

class AsyncDoPutTestServer : public AsyncFlightServerBase {
 public:
  Future<> DoPutAsync(const ServerCallContext&,
                      std::unique_ptr<AsyncFlightMessageReader> reader,
                      std::unique_ptr<AsyncFlightMetadataWriter> writer) override {
    descriptor_ = reader->descriptor();
    auto shared_reader =
        std::shared_ptr<AsyncFlightMessageReader>(std::move(reader));
    auto shared_writer =
        std::shared_ptr<AsyncFlightMetadataWriter>(std::move(writer));

    if (descriptor_.type == FlightDescriptor::DescriptorType::CMD &&
        descriptor_.cmd == "TestUndrained") {
      return Future<>::MakeFinished();
    }

    struct State {
      int counter = 0;
      FlightStreamChunk terminal_chunk;
    };
    auto state = std::make_shared<State>();
    return ::arrow::Loop([this, state, shared_reader, shared_writer]() {
             return shared_reader->Next().Then([this, state, shared_writer](FlightStreamChunk chunk)
                                            -> Future<ControlFlow<>> {
               if (!chunk.data) {
                 state->terminal_chunk = std::move(chunk);
                 return ::arrow::ToFuture(Break());
               }
               if (state->counter % 2 == 1) {
                 if (!chunk.app_metadata) {
                   return Future<ControlFlow<>>::MakeFinished(
                       Status::Invalid("Expected app_metadata"));
                 }
                 if (chunk.app_metadata->ToString() != std::to_string(state->counter)) {
                   return Future<ControlFlow<>>::MakeFinished(Status::Invalid(
                       "Expected app_metadata to be ", state->counter, " but got ",
                       chunk.app_metadata->ToString()));
                 }
               } else if (chunk.app_metadata) {
                 return Future<ControlFlow<>>::MakeFinished(
                     Status::Invalid("Expected no app_metadata"));
               }
               batches_.push_back(std::move(chunk.data));
               auto buffer = Buffer::FromString(std::to_string(state->counter));
               return shared_writer->WriteMetadata(*buffer)
                   .Then([state]() -> ControlFlow<> {
                 ++state->counter;
                 return Continue{};
               });
             });
           })
        .Then([state](const ::arrow::internal::Empty&) -> Status {
          if (!state->terminal_chunk.app_metadata) {
            return Status::Invalid("Expected app_metadata at end of stream (#1)");
          }
          if (state->terminal_chunk.app_metadata->ToString() != kExpectedMetadata) {
            return Status::Invalid("Expected app_metadata to be ", kExpectedMetadata,
                                   " but got ",
                                   state->terminal_chunk.app_metadata->ToString());
          }
          return Status::OK();
        });
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

  Future<> DoPutAsync(const ServerCallContext&,
                      std::unique_ptr<AsyncFlightMessageReader> reader,
                      std::unique_ptr<AsyncFlightMetadataWriter> writer) override {
    auto shared_reader =
        std::shared_ptr<AsyncFlightMessageReader>(std::move(reader));
    auto shared_writer =
        std::shared_ptr<AsyncFlightMetadataWriter>(std::move(writer));
    auto counter = std::make_shared<int>(0);
    return ::arrow::Loop([shared_reader, shared_writer, counter]() {
             return shared_reader->Next().Then([shared_writer, counter](FlightStreamChunk chunk)
                                            -> Future<ControlFlow<>> {
               if (chunk.data == nullptr) {
                 return ::arrow::ToFuture(Break());
               }
               if (chunk.app_metadata == nullptr) {
                 return Future<ControlFlow<>>::MakeFinished(
                     Status::Invalid("Expected application metadata to be provided"));
               }
               if (std::to_string(*counter) != chunk.app_metadata->ToString()) {
                 return Future<ControlFlow<>>::MakeFinished(Status::Invalid(
                     "Expected metadata value: " + std::to_string(*counter) +
                     " but got: " + chunk.app_metadata->ToString()));
               }
               auto metadata = Buffer::FromString(std::to_string(*counter));
               return shared_writer->WriteMetadata(*metadata)
                   .Then([counter]() -> ControlFlow<> {
                 ++(*counter);
                 return Continue{};
               });
             });
           })
        .Then([](const ::arrow::internal::Empty&) { return Status::OK(); });
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

  Future<> DoPutAsync(const ServerCallContext&,
                      std::unique_ptr<AsyncFlightMessageReader> reader,
                      std::unique_ptr<AsyncFlightMetadataWriter> writer) override {
    auto shared_reader =
        std::shared_ptr<AsyncFlightMessageReader>(std::move(reader));
    auto shared_writer =
        std::shared_ptr<AsyncFlightMetadataWriter>(std::move(writer));
    auto counter = std::make_shared<int>(0);
    return ::arrow::Loop([shared_reader, counter]() {
             return shared_reader->Next().Then([counter](FlightStreamChunk chunk)
                                                   -> ControlFlow<> {
               if (chunk.data == nullptr) {
                 return Break();
               }
               ++(*counter);
               return Continue{};
             });
           })
        .Then([shared_writer, counter](const ::arrow::internal::Empty&) {
          auto metadata = Buffer::FromString(std::to_string(*counter));
          return shared_writer->WriteMetadata(*metadata);
        });
  }

  Future<> DoExchangeAsync(const ServerCallContext&,
                           std::unique_ptr<AsyncFlightMessageReader> reader,
                           std::unique_ptr<AsyncFlightMessageWriter> writer) override {
    auto shared_reader =
        std::shared_ptr<AsyncFlightMessageReader>(std::move(reader));
    auto shared_writer =
        std::shared_ptr<AsyncFlightMessageWriter>(std::move(writer));
    auto options = ipc::IpcWriteOptions::Defaults();
    options.max_recursion_depth = 1;
    auto begun = std::make_shared<bool>(false);
    return ::arrow::Loop([shared_reader, shared_writer, begun, options]() {
             return shared_reader->Next().Then(
                 [shared_writer, begun, options](FlightStreamChunk chunk)
                                            -> Future<ControlFlow<>> {
               if (!chunk.data && !chunk.app_metadata) {
                 return ::arrow::ToFuture(Break());
               }
               Future<> start = Future<>::MakeFinished();
               if (!*begun && chunk.data) {
                 *begun = true;
                 start = shared_writer->Begin(chunk.data->schema(), options);
               }
               return start.Then(
                   [shared_writer, chunk = std::move(chunk)]() mutable
                       -> Future<ControlFlow<>> {
                 if (chunk.data && chunk.app_metadata) {
                   return shared_writer
                       ->WriteWithMetadata(*chunk.data, chunk.app_metadata)
                       .Then([]() -> ControlFlow<> { return Continue{}; });
                 }
                 if (chunk.data) {
                   return shared_writer->WriteRecordBatch(*chunk.data)
                       .Then([]() -> ControlFlow<> { return Continue{}; });
                 }
                 return shared_writer->WriteMetadata(chunk.app_metadata)
                     .Then([]() -> ControlFlow<> { return Continue{}; });
               });
             });
           })
        .Then([](const ::arrow::internal::Empty&) { return Status::OK(); });
  }
};

class AsyncAuthTestServer : public AsyncFlightServerBase {
 public:
  Future<std::unique_ptr<ResultStream>> DoAction(const ServerCallContext& context,
                                                 const Action& action) override {
    if (action.type == "who-am-i") {
      std::vector<Result> results = {
          Result{Buffer::FromString(context.peer_identity())},
      };
      return Future<std::unique_ptr<ResultStream>>::MakeFinished(
          std::make_unique<SimpleResultStream>(std::move(results)));
    }
    return Future<std::unique_ptr<ResultStream>>::MakeFinished(
        Status::NotImplemented("Expected authenticated action"));
  }
};

class AsyncPollFlightInfoTestServer : public AsyncFlightServerBase {
 public:
  Future<std::unique_ptr<PollInfo>> PollFlightInfo(
      const ServerCallContext&, const FlightDescriptor& descriptor) override {
    auto schema = arrow::schema({arrow::field("number", arrow::uint32(), false)});
    std::vector<FlightEndpoint> endpoints = {
        FlightEndpoint{{"long-running query"}, {}, std::nullopt, ""}};
    ARROW_ASSIGN_OR_RAISE(
        auto info, FlightInfo::Make(*schema, descriptor, endpoints, -1, -1, false));
    if (descriptor == FlightDescriptor::Command("poll")) {
      return Future<std::unique_ptr<PollInfo>>::MakeFinished(std::make_unique<PollInfo>(
          std::make_unique<FlightInfo>(std::move(info)), std::nullopt, 1.0,
          std::nullopt));
    }
    return Future<std::unique_ptr<PollInfo>>::MakeFinished(std::make_unique<PollInfo>(
        std::make_unique<FlightInfo>(std::move(info)),
        FlightDescriptor::Command("poll"), 0.1,
        Timestamp::clock::now() + std::chrono::seconds{10}));
  }
};

class AsyncCountingServerMiddleware : public ServerMiddleware {
 public:
  AsyncCountingServerMiddleware(std::atomic<int>* successful, std::atomic<int>* failed,
                                std::atomic<int>* headers_sent)
      : successful_(successful), failed_(failed), headers_sent_(headers_sent) {}

  void SendingHeaders(AddCallHeaders* outgoing_headers) override {
    outgoing_headers->AddHeader("x-async-middleware", "present");
    ++(*headers_sent_);
  }

  void CallCompleted(const Status& status) override {
    if (status.ok()) {
      ++(*successful_);
    } else {
      ++(*failed_);
    }
  }

  std::string name() const override { return "AsyncCountingServerMiddleware"; }

 private:
  std::atomic<int>* successful_;
  std::atomic<int>* failed_;
  std::atomic<int>* headers_sent_;
};

class AsyncCountingServerMiddlewareFactory : public ServerMiddlewareFactory {
 public:
  Status StartCall(const CallInfo&, const ServerCallContext&,
                   std::shared_ptr<ServerMiddleware>* middleware) override {
    *middleware = std::make_shared<AsyncCountingServerMiddleware>(&successful_, &failed_,
                                                                 &headers_sent_);
    return Status::OK();
  }

  std::atomic<int> successful_{0};
  std::atomic<int> failed_{0};
  std::atomic<int> headers_sent_{0};
};

class AsyncHeaderRecordingClientMiddleware : public ClientMiddleware {
 public:
  AsyncHeaderRecordingClientMiddleware(std::atomic<int>* received_headers,
                                       std::vector<FlightMethod>* recorded_calls,
                                       std::vector<Status>* recorded_status,
                                       std::atomic<bool>* saw_expected_header,
                                       FlightMethod method)
      : received_headers_(received_headers),
        recorded_calls_(recorded_calls),
        recorded_status_(recorded_status),
        saw_expected_header_(saw_expected_header),
        method_(method) {}

  void SendingHeaders(AddCallHeaders*) override {}

  void ReceivedHeaders(const CallHeaders& incoming_headers) override {
    ++(*received_headers_);
    const auto it = incoming_headers.find("x-async-middleware");
    if (it != incoming_headers.end() && it->second == std::string_view("present")) {
      saw_expected_header_->store(true);
    }
  }

  void CallCompleted(const Status& status) override {
    recorded_calls_->push_back(method_);
    recorded_status_->push_back(status);
  }

 private:
  std::atomic<int>* received_headers_;
  std::vector<FlightMethod>* recorded_calls_;
  std::vector<Status>* recorded_status_;
  std::atomic<bool>* saw_expected_header_;
  FlightMethod method_;
};

class AsyncHeaderRecordingClientMiddlewareFactory : public ClientMiddlewareFactory {
 public:
  void StartCall(const CallInfo& info,
                 std::unique_ptr<ClientMiddleware>* middleware) override {
    *middleware = std::make_unique<AsyncHeaderRecordingClientMiddleware>(
        &received_headers_, &recorded_calls_, &recorded_status_, &saw_expected_header_,
        info.method);
  }

  void Reset() {
    received_headers_.store(0);
    saw_expected_header_.store(false);
    recorded_calls_.clear();
    recorded_status_.clear();
  }

  std::vector<FlightMethod> recorded_calls_;
  std::vector<Status> recorded_status_;
  std::atomic<int> received_headers_{0};
  std::atomic<bool> saw_expected_header_{false};
};

class AsyncMiddlewareContextTestServer : public AsyncFlightServerBase {
 public:
  Future<std::unique_ptr<ResultStream>> DoAction(const ServerCallContext& context,
                                                 const Action&) override {
    const auto* middleware = context.GetMiddleware("request_counter");
    if (!middleware ||
        middleware->name() != "AsyncCountingServerMiddleware") {
      return Future<std::unique_ptr<ResultStream>>::MakeFinished(
          Status::Invalid("Could not find middleware"));
    }
    std::vector<Result> results = {
        Result{Buffer::FromString("middleware-ok")},
    };
    return Future<std::unique_ptr<ResultStream>>::MakeFinished(
        std::make_unique<SimpleResultStream>(std::move(results)));
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
::arrow::Result<int> GetProcessThreadCount(pid_t pid = getpid()) {
  const std::string path = "/proc/" + std::to_string(pid) + "/status";
  std::ifstream status(path);
  if (!status) {
    return Status::IOError("Could not open ", path);
  }
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0) {
      auto value = line.substr(std::strlen("Threads:"));
      return std::stoi(value);
    }
  }
  return Status::Invalid("Could not find thread count in ", path);
}

Status ReadExact(int fd, void* data, size_t size) {
  auto* out = reinterpret_cast<uint8_t*>(data);
  size_t offset = 0;
  while (offset < size) {
    const ssize_t nbytes = read(fd, out + offset, size - offset);
    if (nbytes == 0) {
      return Status::IOError("Unexpected EOF while reading from pipe");
    }
    if (nbytes < 0) {
      return Status::IOError("Failed reading from pipe");
    }
    offset += static_cast<size_t>(nbytes);
  }
  return Status::OK();
}

Status WriteExact(int fd, const void* data, size_t size) {
  const auto* in = reinterpret_cast<const uint8_t*>(data);
  size_t offset = 0;
  while (offset < size) {
    const ssize_t nbytes = write(fd, in + offset, size - offset);
    if (nbytes < 0) {
      return Status::IOError("Failed writing to pipe");
    }
    offset += static_cast<size_t>(nbytes);
  }
  return Status::OK();
}

struct ThreadScalingServerInfo {
  int port;
  int baseline_threads;
};

template <typename ServerType>
::arrow::Result<int> MeasureThreadDeltaForConnections(int num_clients) {
  int info_pipe[2];
  int control_pipe[2];
  if (pipe(info_pipe) != 0) {
    return Status::IOError("Failed to create info pipe");
  }
  if (pipe(control_pipe) != 0) {
    close(info_pipe[0]);
    close(info_pipe[1]);
    return Status::IOError("Failed to create control pipe");
  }

  const pid_t child = fork();
  if (child < 0) {
    close(info_pipe[0]);
    close(info_pipe[1]);
    close(control_pipe[0]);
    close(control_pipe[1]);
    return Status::IOError("Failed to fork thread scaling server process");
  }

  if (child == 0) {
    close(info_pipe[0]);
    close(control_pipe[1]);

    auto fail = [&]() {
      close(info_pipe[1]);
      close(control_pipe[0]);
      _exit(1);
    };

    ARROW_ASSIGN_OR_RAISE(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    FlightServerOptions options(location);
    auto server = std::make_unique<ServerType>();
    if (!server->Init(options).ok()) {
      fail();
    }

    ThreadScalingServerInfo info{server->port(), -1};
    auto maybe_threads = GetProcessThreadCount();
    if (!maybe_threads.ok()) {
      fail();
    }
    info.baseline_threads = maybe_threads.ValueUnsafe();

    if (!WriteExact(info_pipe[1], &info, sizeof(info)).ok()) {
      fail();
    }

    char shutdown = 0;
    if (!ReadExact(control_pipe[0], &shutdown, sizeof(shutdown)).ok()) {
      fail();
    }

    if (!server->Shutdown().ok() || !server->Wait().ok()) {
      fail();
    }
    close(info_pipe[1]);
    close(control_pipe[0]);
    _exit(0);
  }

  close(info_pipe[1]);
  close(control_pipe[0]);

  auto finish_child = [&](bool request_shutdown) {
    if (request_shutdown) {
      static constexpr char kShutdown = 'x';
      ARROW_UNUSED(WriteExact(control_pipe[1], &kShutdown, sizeof(kShutdown)));
    }
    close(info_pipe[0]);
    close(control_pipe[1]);
    int wstatus = 0;
    ARROW_UNUSED(waitpid(child, &wstatus, 0));
  };

  ThreadScalingServerInfo info{-1, -1};
  auto st = ReadExact(info_pipe[0], &info, sizeof(info));
  if (!st.ok()) {
    finish_child(false);
    return st;
  }

  std::vector<std::unique_ptr<FlightClient>> clients;
  clients.reserve(num_clients);
  auto maybe_client_location = Location::ForGrpcTcp("127.0.0.1", info.port);
  if (!maybe_client_location.ok()) {
    finish_child(true);
    return maybe_client_location.status();
  }
  auto client_location = std::move(maybe_client_location).ValueUnsafe();

  FlightDescriptor descriptor = FlightDescriptor::Path({"thread-scaling"});
  for (int i = 0; i < num_clients; ++i) {
    auto maybe_client = FlightClient::Connect(client_location);
    if (!maybe_client.ok()) {
      finish_child(true);
      return maybe_client.status();
    }
    auto client = std::move(maybe_client).ValueUnsafe();
    auto maybe_flight_info = client->GetFlightInfo(descriptor);
    if (!maybe_flight_info.ok()) {
      finish_child(true);
      return maybe_flight_info.status();
    }
    auto flight_info = std::move(maybe_flight_info).ValueUnsafe();
    EXPECT_EQ(descriptor, flight_info->descriptor());
    clients.push_back(std::move(client));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  auto maybe_after_threads = GetProcessThreadCount(child);
  if (!maybe_after_threads.ok()) {
    finish_child(true);
    return maybe_after_threads.status();
  }
  int after_threads = maybe_after_threads.ValueUnsafe();

  for (auto& client : clients) {
    st = client->Close();
    if (!st.ok()) {
      finish_child(true);
      return st;
    }
  }
  finish_child(true);
  return after_threads - info.baseline_threads;
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
  ASSERT_OK(exchange.writer->DoneWriting());
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

class AsyncAuthTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    ASSERT_OK(MakeAsyncServer<AsyncAuthTestServer>(
        location, &server_, &client_,
        [](FlightServerOptions* options) {
          options->auth_handler = std::make_unique<TestServerAuthHandler>(
              kAsyncAuthUsername, kAsyncAuthPassword);
          return Status::OK();
        },
        [](FlightClientOptions*) { return Status::OK(); }));
  }

  void TearDown() override {
    ASSERT_OK(client_->Close());
    ASSERT_OK(server_->Shutdown());
    ASSERT_OK(server_->Wait());
  }

  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<AsyncFlightServerBase> server_;
};

class AsyncPollFlightInfoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    ASSERT_OK(MakeAsyncServer<AsyncPollFlightInfoTestServer>(
        location, &server_, &client_,
        [](FlightServerOptions*) { return Status::OK(); },
        [](FlightClientOptions*) { return Status::OK(); }));
  }

  void TearDown() override {
    ASSERT_OK(client_->Close());
    ASSERT_OK(server_->Shutdown());
    ASSERT_OK(server_->Wait());
  }

  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<AsyncFlightServerBase> server_;
};

class AsyncMiddlewareTest : public ::testing::Test {
 protected:
  void SetUp() override {
    request_counter_ = std::make_shared<AsyncCountingServerMiddlewareFactory>();
    client_middleware_ = std::make_shared<AsyncHeaderRecordingClientMiddlewareFactory>();
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    ASSERT_OK(MakeAsyncServer<AsyncMiddlewareContextTestServer>(
        location, &server_, &client_,
        [&](FlightServerOptions* options) {
          options->middleware.push_back({"request_counter", request_counter_});
          return Status::OK();
        },
        [&](FlightClientOptions* options) {
          options->middleware.push_back(client_middleware_);
          return Status::OK();
        }));
  }

  void TearDown() override {
    ASSERT_OK(client_->Close());
    ASSERT_OK(server_->Shutdown());
    ASSERT_OK(server_->Wait());
  }

  std::shared_ptr<AsyncCountingServerMiddlewareFactory> request_counter_;
  std::shared_ptr<AsyncHeaderRecordingClientMiddlewareFactory> client_middleware_;
  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<AsyncFlightServerBase> server_;
};

class AsyncNativeStreamContractServer : public AsyncFlightServerBase {
 public:
  Future<> concurrent_reads_result() const { return concurrent_reads_result_; }
  Future<> concurrent_writes_result() const { return concurrent_writes_result_; }
  Future<ipc::WriteStats> dict_writer_stats_result() const { return dict_writer_stats_result_; }
  Status concurrent_reads_status() const { return concurrent_reads_result_.status(); }

  Future<> DoPutAsync(const ServerCallContext&,
                      std::unique_ptr<AsyncFlightMessageReader> reader,
                      std::unique_ptr<AsyncFlightMetadataWriter> writer) override {
    if (reader->descriptor().type != FlightDescriptor::DescriptorType::CMD ||
        reader->descriptor().cmd != "concurrent_writes") {
      return Future<>::MakeFinished(
          Status::NotImplemented("Scenario not implemented: ", reader->descriptor().ToString()));
    }
    auto shared_reader =
        std::shared_ptr<AsyncFlightMessageReader>(std::move(reader));
    return shared_reader->GetSchema().Then(
        [this, shared_reader, writer = std::move(writer)](
            const std::shared_ptr<Schema>&) mutable {
      auto first = writer->WriteMetadata(*Buffer::FromString("one"));
      auto second = writer->WriteMetadata(*Buffer::FromString("two"));
      auto status = second.status();
      concurrent_writes_result_.MarkFinished(status);
      return first.Then(
          [status]() mutable { return status; },
          [status](const Status&) mutable { return status; });
    });
  }

  Future<> DoExchangeAsync(const ServerCallContext&,
                           std::unique_ptr<AsyncFlightMessageReader> reader,
                           std::unique_ptr<AsyncFlightMessageWriter> writer) override {
    if (reader->descriptor().type != FlightDescriptor::DescriptorType::CMD) {
      return Future<>::MakeFinished(Status::Invalid("Must provide a command descriptor"));
    }

    struct State {
      std::shared_ptr<AsyncFlightMessageReader> reader;
      std::shared_ptr<AsyncFlightMessageWriter> writer;
    };
    auto state = std::make_shared<State>();
    state->reader = std::shared_ptr<AsyncFlightMessageReader>(std::move(reader));
    state->writer = std::shared_ptr<AsyncFlightMessageWriter>(std::move(writer));

    const std::string& cmd = state->reader->descriptor().cmd;
    if (cmd == "concurrent_reads") {
      ARROW_UNUSED(state->reader->GetSchema());
      auto second = state->reader->GetSchema();
      auto status = second.status();
      concurrent_reads_result_.MarkFinished(status);
      return Future<>::MakeFinished(status);
    }
    if (cmd == "dict_writer") {
      auto batches = std::make_shared<RecordBatchVector>();
      auto st = ExampleDictBatches(batches.get());
      if (!st.ok()) {
        return Future<>::MakeFinished(st);
      }
      return state->writer->Begin((*batches)[0]->schema()).Then([state, batches]() {
        return AllComplete(
            {WriteRecordBatchesAsync(state->writer, *batches),
             DrainAsyncReader(state->reader)});
      }).Then([this, state]() {
        dict_writer_stats_result_.MarkFinished(state->writer->stats());
        return Status::OK();
      });
    }
    return Future<>::MakeFinished(Status::NotImplemented("Scenario not implemented: ", cmd));
  }

 private:
  Future<> concurrent_reads_result_ = Future<>::Make();
  Future<> concurrent_writes_result_ = Future<>::Make();
  Future<ipc::WriteStats> dict_writer_stats_result_ = Future<ipc::WriteStats>::Make();
};

class AsyncNativeStreamApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(auto location, Location::ForGrpcTcp("127.0.0.1", 0));
    ASSERT_OK(MakeAsyncServer<AsyncNativeStreamContractServer>(
        location, &server_, &client_,
        [](FlightServerOptions*) { return Status::OK(); },
        [](FlightClientOptions*) { return Status::OK(); }));
    server_impl_ = static_cast<AsyncNativeStreamContractServer*>(server_.get());
  }

  void TearDown() override {
    ASSERT_OK(client_->Close());
    ASSERT_OK(server_->Shutdown());
    ASSERT_OK(server_->Wait());
  }

  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<AsyncFlightServerBase> server_;
  AsyncNativeStreamContractServer* server_impl_;
};

TEST_F(AsyncNativeStreamApiTest, RejectsConcurrentReads) {
  auto concurrent_reads = server_impl_->concurrent_reads_result();
  ASSERT_OK_AND_ASSIGN(auto exchange,
                       client_->DoExchange(FlightDescriptor::Command("concurrent_reads")));
  ASSERT_TRUE(concurrent_reads.Wait(1.0));
  ARROW_UNUSED(exchange.writer->Close());
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, HasSubstr("Concurrent async reads are not supported"), concurrent_reads.status());
}

TEST_F(AsyncNativeStreamApiTest, RejectsConcurrentWrites) {
  auto concurrent_writes = server_impl_->concurrent_writes_result();
  auto schema = arrow::schema({field("ints", int32())});
  ASSERT_OK_AND_ASSIGN(auto do_put,
                       client_->DoPut(FlightDescriptor::Command("concurrent_writes"), schema));
  auto empty_batch = RecordBatch::Make(schema, 0, {ArrayFromJSON(int32(), "[]")});
  ASSERT_OK(do_put.writer->WriteRecordBatch(*empty_batch));
  ASSERT_TRUE(concurrent_writes.Wait(1.0));
  ARROW_UNUSED(do_put.writer->Close());
  const auto status = concurrent_writes.status();
  if (status.IsInvalid()) {
    ASSERT_THAT(status.message(), HasSubstr("Concurrent async writes are not supported"));
  } else {
    ASSERT_OK(status);
  }
}

TEST_F(AsyncNativeStreamApiTest, DISABLED_WritesDictionaryBatches) {
  auto dict_writer_stats = server_impl_->dict_writer_stats_result();
  ASSERT_OK_AND_ASSIGN(auto exchange,
                       client_->DoExchange(FlightDescriptor::Command("dict_writer")));
  ASSERT_OK(exchange.writer->DoneWriting());
  RecordBatchVector expected_batches;
  ASSERT_OK(ExampleDictBatches(&expected_batches));
  ASSERT_OK_AND_ASSIGN(auto schema, exchange.reader->GetSchema());
  AssertSchemaEqual(*expected_batches[0]->schema(), *schema);
  for (const auto& expected : expected_batches) {
    ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
    ASSERT_NE(nullptr, chunk.data);
    ASSERT_BATCHES_EQUAL(*expected, *chunk.data);
  }
  ASSERT_TRUE(dict_writer_stats.Wait(1.0));
  ASSERT_OK(dict_writer_stats.status());
  auto stats = std::move(dict_writer_stats).MoveResult().ValueOrDie();
  ASSERT_GE(stats.num_dictionary_batches, 1);
  ASSERT_EQ(3, stats.num_record_batches);
  ASSERT_OK(exchange.writer->Close());
}

TEST_F(AsyncAuthTest, HandshakeAuthenticatesAndSetsPeerIdentity) {
  ASSERT_OK(client_->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>(kAsyncAuthUsername,
                                                  kAsyncAuthPassword)));

  Action action{"who-am-i", Buffer::FromString("")};
  ASSERT_OK_AND_ASSIGN(auto results, client_->DoAction(action));
  ASSERT_OK_AND_ASSIGN(auto first, results->Next());
  ASSERT_NE(nullptr, first);
  ASSERT_EQ(kAsyncAuthUsername, first->body->ToString());
}

TEST_F(AsyncAuthTest, HandshakeRejectsInvalidCredentials) {
  auto status = client_->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>(kAsyncInvalidAuthUsername,
                                                  kAsyncInvalidAuthPassword));
  ASSERT_RAISES(IOError, status);
  ASSERT_THAT(status.message(), HasSubstr("Invalid token"));
}

TEST_F(AsyncPollFlightInfoTest, PollFlightInfoRoundTrip) {
  ASSERT_OK_AND_ASSIGN(
      auto poll_info, client_->PollFlightInfo(FlightDescriptor::Command("heavy query")));
  ASSERT_NE(nullptr, poll_info);
  ASSERT_NE(nullptr, poll_info->info);
  ASSERT_TRUE(poll_info->descriptor.has_value());
  ASSERT_TRUE(poll_info->progress.has_value());
  ASSERT_TRUE(poll_info->expiration_time.has_value());
  ASSERT_EQ(0.1, *poll_info->progress);
  ASSERT_EQ(FlightDescriptor::Command("poll"), *poll_info->descriptor);

  ASSERT_OK_AND_ASSIGN(auto completed, client_->PollFlightInfo(*poll_info->descriptor));
  ASSERT_NE(nullptr, completed);
  ASSERT_NE(nullptr, completed->info);
  ASSERT_FALSE(completed->descriptor.has_value());
  ASSERT_TRUE(completed->progress.has_value());
  ASSERT_EQ(1.0, *completed->progress);
}

TEST_F(AsyncMiddlewareTest, MiddlewareRunsAndSendsHeaders) {
  client_middleware_->Reset();

  Action action{"middleware", Buffer::FromString("")};
  ASSERT_OK_AND_ASSIGN(auto results, client_->DoAction(action));
  ASSERT_OK_AND_ASSIGN(auto first, results->Next());
  ASSERT_NE(nullptr, first);
  ASSERT_EQ("middleware-ok", first->body->ToString());
  ASSERT_OK_AND_ASSIGN(auto terminal, results->Next());
  ASSERT_EQ(nullptr, terminal);

  ASSERT_EQ(1, request_counter_->successful_.load());
  ASSERT_EQ(0, request_counter_->failed_.load());
  ASSERT_GE(request_counter_->headers_sent_.load(), 1);
  ASSERT_TRUE(client_middleware_->saw_expected_header_.load());
  ASSERT_GE(client_middleware_->received_headers_.load(), 1);
  ASSERT_EQ(1U, client_middleware_->recorded_calls_.size());
  ASSERT_EQ(FlightMethod::DoAction, client_middleware_->recorded_calls_[0]);
  ASSERT_EQ(1U, client_middleware_->recorded_status_.size());
  ASSERT_TRUE(client_middleware_->recorded_status_[0].ok());
}

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
