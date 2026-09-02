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

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "arrow/flight/protocol_internal.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/server_async.h"
#include "arrow/flight/transport/grpc/grpc_server_internal.h"
#include "arrow/flight/transport_server_async.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/uri.h"

namespace arrow::flight::transport::grpc::async_internal {

namespace pb = arrow::flight::protocol;
using FlightService = pb::FlightService;
using GrpcServerCallContext =
    ::arrow::flight::transport::grpc::GrpcServerCallContext<
        ::grpc::CallbackServerContext>;
using CallbackServiceHelper =
    ::arrow::flight::transport::grpc::GrpcServerCallContextHelper<
        ::grpc::CallbackServerContext>;

arrow::Result<std::shared_ptr<arrow::internal::ThreadPool>> MakeAsyncGrpcExecutor();

::grpc::Status PrepareAuthenticatedCall(const CallbackServiceHelper& helper,
                                        FlightMethod method,
                                        ::grpc::CallbackServerContext* context,
                                        GrpcServerCallContext* flight_context);

template <typename ProtoRequest, typename ArrowRequest>
Status ParseRequiredRequest(const ProtoRequest* request, const char* request_name,
                            ArrowRequest* out) {
  if (request == nullptr) {
    return Status::Invalid(request_name, " cannot be null");
  }
  return ::arrow::flight::internal::FromProto(*request, out);
}

template <typename ArrowResponse, typename ProtoResponse, typename ToProtoFn>
void FinishUnaryResult(::grpc::ServerUnaryReactor* reactor, ProtoResponse* response,
                       GrpcServerCallContext flight_context,
                       const arrow::Result<std::unique_ptr<ArrowResponse>>& result, ToProtoFn&& to_proto) {
  if (result.ok() && result.ValueUnsafe()) {
    reactor->Finish(
        flight_context.FinishRequest(to_proto(*result.ValueUnsafe(), response)));
    return;
  }
  reactor->Finish(flight_context.FinishRequest(
      result.ok() ? Status::KeyError("Flight not found") : result.status()));
}

struct AsyncMessageReader {
  std::unique_ptr<AsyncFlightMessageReader> reader;
  std::function<Future<>()> when_idle;

  std::unique_ptr<AsyncFlightMessageReader> TakeReader() {
    return std::move(reader);
  }

  Future<> WhenIdle() const { return when_idle(); }
};

using AsyncReadFn =
    std::function<Future<std::shared_ptr<::arrow::flight::internal::FlightData>>()>;

Future<AsyncMessageReader> MakeAsyncMessageReader(
    AsyncReadFn read_fn, std::shared_ptr<MemoryManager> memory_manager);

std::unique_ptr<AsyncFlightMetadataWriter> MakeAsyncMetadataWriter(
    std::function<Future<bool>(pb::PutResult)> write_fn);

std::unique_ptr<AsyncFlightMessageWriter> MakeAsyncMessageWriter(
    std::function<Future<bool>(FlightPayload)> write_fn);

class CallbackFlightService;

class AsyncGrpcServerTransport : public arrow::flight::internal::AsyncServerTransport {
 public:
  AsyncGrpcServerTransport(AsyncFlightServerBase* base,
                           std::shared_ptr<MemoryManager> memory_manager);
  ~AsyncGrpcServerTransport() override;

  Status Init(const FlightServerOptions& options, const arrow::util::Uri& uri) override;
  Status Shutdown() override;
  Status Shutdown(const std::chrono::system_clock::time_point& deadline) override;
  Status Wait() override;
  Location location() const override;

  const CallbackServiceHelper& helper() const { return *helper_; }
  std::shared_ptr<arrow::internal::ThreadPool> executor() const { return executor_pool_; }
  std::shared_ptr<MemoryManager> memory_manager() const { return memory_manager_; }

 private:
  std::shared_ptr<arrow::internal::ThreadPool> executor_pool_;
  std::unique_ptr<CallbackServiceHelper> helper_;
  std::unique_ptr<CallbackFlightService> grpc_service_;
  std::unique_ptr<::grpc::Server> grpc_server_;
  Location location_;
};

class CallbackFlightService final : public FlightService::CallbackService {
 public:
  CallbackFlightService(AsyncGrpcServerTransport* impl, CallbackServiceHelper helper)
      : impl_(impl), helper_(std::move(helper)) {}

  ::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>* Handshake(
      ::grpc::CallbackServerContext* context) override;

  ::grpc::ServerWriteReactor<pb::FlightInfo>* ListFlights(
      ::grpc::CallbackServerContext* context, const pb::Criteria* request) override;

  ::grpc::ServerUnaryReactor* GetFlightInfo(::grpc::CallbackServerContext* context,
                                            const pb::FlightDescriptor* request,
                                            pb::FlightInfo* response) override;

  ::grpc::ServerUnaryReactor* PollFlightInfo(::grpc::CallbackServerContext* context,
                                             const pb::FlightDescriptor* request,
                                             pb::PollInfo* response) override;

  ::grpc::ServerUnaryReactor* GetSchema(::grpc::CallbackServerContext* context,
                                        const pb::FlightDescriptor* request,
                                        pb::SchemaResult* response) override;

  ::grpc::ServerWriteReactor<pb::FlightData>* DoGet(
      ::grpc::CallbackServerContext* context, const pb::Ticket* request) override;

  ::grpc::ServerBidiReactor<pb::FlightData, pb::PutResult>* DoPut(
      ::grpc::CallbackServerContext* context) override;

  ::grpc::ServerBidiReactor<pb::FlightData, pb::FlightData>* DoExchange(
      ::grpc::CallbackServerContext* context) override;

  ::grpc::ServerWriteReactor<pb::ActionType>* ListActions(
      ::grpc::CallbackServerContext* context, const pb::Empty* request) override;

  ::grpc::ServerWriteReactor<pb::Result>* DoAction(::grpc::CallbackServerContext* context,
                                                   const pb::Action* request) override;

 private:
  AsyncGrpcServerTransport* impl_;
  CallbackServiceHelper helper_;
};

::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>*
MakeHandshakeReactor(::grpc::CallbackServerContext* context,
                     AsyncGrpcServerTransport* impl,
                     const CallbackServiceHelper& helper);

::grpc::ServerBidiReactor<pb::FlightData, pb::PutResult>* MakeDoPutReactor(
    ::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
    const CallbackServiceHelper& helper);

::grpc::ServerBidiReactor<pb::FlightData, pb::FlightData>* MakeDoExchangeReactor(
    ::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
    const CallbackServiceHelper& helper);

::grpc::ServerWriteReactor<pb::FlightInfo>* MakeListFlightsReactor(
    ::grpc::CallbackServerContext* context,
    std::shared_ptr<arrow::internal::ThreadPool> executor,
    GrpcServerCallContext flight_context,
    Future<std::unique_ptr<FlightListing>> future);

::grpc::ServerWriteReactor<pb::ActionType>* MakeListActionsReactor(
    ::grpc::CallbackServerContext* context,
    std::shared_ptr<arrow::internal::ThreadPool> executor,
    GrpcServerCallContext flight_context, Future<std::vector<ActionType>> future);

::grpc::ServerWriteReactor<pb::Result>* MakeDoActionReactor(
    ::grpc::CallbackServerContext* context,
    std::shared_ptr<arrow::internal::ThreadPool> executor,
    GrpcServerCallContext flight_context,
    Future<std::unique_ptr<ResultStream>> future);

::grpc::ServerWriteReactor<pb::FlightData>* MakeDoGetReactor(
    ::grpc::CallbackServerContext* context,
    std::shared_ptr<arrow::internal::ThreadPool> executor,
    GrpcServerCallContext flight_context,
    Future<std::unique_ptr<AsyncFlightDataStream>> future);

}  // namespace arrow::flight::transport::grpc::async_internal