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

#include "arrow/flight/transport/grpc/grpc_server_async_internal.h"

namespace arrow::flight::transport::grpc::async_internal {

arrow::Result<std::shared_ptr<arrow::internal::ThreadPool>> MakeAsyncGrpcExecutor() {
  return arrow::internal::ThreadPool::MakeEternal(
      arrow::internal::ThreadPool::DefaultCapacity());
}

::grpc::Status PrepareAuthenticatedCall(const CallbackServiceHelper& helper,
                                        FlightMethod method,
                                        ::grpc::CallbackServerContext* context,
                                        GrpcServerCallContext* flight_context) {
  auto st = helper.CheckAuth(method, context, flight_context);
  if (!st.ok()) {
    return st;
  }
  helper.AddMiddlewareHeaders(context, flight_context);
  return ::grpc::Status::OK;
}

AsyncGrpcServerTransport::AsyncGrpcServerTransport(
    AsyncFlightServerBase* base, std::shared_ptr<MemoryManager> memory_manager)
  : arrow::flight::internal::AsyncServerTransport(base, std::move(memory_manager)) {}

AsyncGrpcServerTransport::~AsyncGrpcServerTransport() = default;

Status AsyncGrpcServerTransport::Init(const FlightServerOptions& options,
                                      const arrow::util::Uri& uri) {
  ARROW_ASSIGN_OR_RAISE(executor_pool_, MakeAsyncGrpcExecutor());
  helper_ =
      std::make_unique<CallbackServiceHelper>(options.auth_handler, options.middleware);
  grpc_service_ = std::make_unique<CallbackFlightService>(this, *helper_);

  return transport::grpc::StartFlightGrpcServer(options, uri, grpc_service_.get(),
                                                /*callback_api=*/true, &grpc_server_,
                                                &location_);
}

Status AsyncGrpcServerTransport::Shutdown() {
  grpc_server_->Shutdown();
  return Status::OK();
}

Status AsyncGrpcServerTransport::Shutdown(
    const std::chrono::system_clock::time_point& deadline) {
  grpc_server_->Shutdown(deadline);
  return Status::OK();
}

Status AsyncGrpcServerTransport::Wait() {
  grpc_server_->Wait();
  return Status::OK();
}

Location AsyncGrpcServerTransport::location() const { return location_; }

/// Bridge the generated callback service to the async handshake reactor.
::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>*
CallbackFlightService::Handshake(::grpc::CallbackServerContext* context) {
  return MakeHandshakeReactor(context, impl_, helper_);
}

/// Authenticate ListFlights and stream the returned FlightListing iterator.
::grpc::ServerWriteReactor<pb::FlightInfo>* CallbackFlightService::ListFlights(
    ::grpc::CallbackServerContext* context, const pb::Criteria* request) {
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::ListFlights, context,
                                     &flight_context);
  if (!st.ok()) {
    return FinishWriteNow<pb::FlightInfo>(st);
  }

  Criteria criteria;
  if (request) {
    auto conv = internal::FromProto(*request, &criteria);
    if (!conv.ok()) {
      return FinishWriteNow<pb::FlightInfo>(flight_context.FinishRequest(conv));
    }
  }
  auto future = impl_->base()->ListFlights(flight_context, &criteria);
  return MakeListFlightsReactor(context, impl_->executor(), std::move(flight_context),
                                std::move(future));
}

/// Authenticate, parse, and asynchronously resolve GetFlightInfo.
::grpc::ServerUnaryReactor* CallbackFlightService::GetFlightInfo(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::FlightInfo* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::GetFlightInfo, context,
                                     &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  FlightDescriptor descr;
  auto arrow_st = ParseRequiredRequest(request, "FlightDescriptor", &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  impl_->base()
      ->GetFlightInfo(flight_context, descr)
      .AddCallback([reactor, response, flight_context = std::move(flight_context)](
                       const arrow::Result<std::unique_ptr<FlightInfo>>& result) mutable {
        FinishUnaryResult(reactor, response, std::move(flight_context), result,
                          "Flight not found",
                          [](const FlightInfo& info, pb::FlightInfo* out) {
                            return internal::ToProto(info, out);
                          });
      });
  return reactor;
}

/// Authenticate, parse, and asynchronously resolve PollFlightInfo.
::grpc::ServerUnaryReactor* CallbackFlightService::PollFlightInfo(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::PollInfo* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::PollFlightInfo, context,
                                     &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  FlightDescriptor descr;
  auto arrow_st = ParseRequiredRequest(request, "FlightDescriptor", &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  impl_->base()
      ->PollFlightInfo(flight_context, descr)
      .AddCallback([reactor, response, flight_context = std::move(flight_context)](
                       const arrow::Result<std::unique_ptr<PollInfo>>& result) mutable {
        FinishUnaryResult(reactor, response, std::move(flight_context), result,
                          "Flight not found",
                          [](const PollInfo& info, pb::PollInfo* out) {
                            return internal::ToProto(info, out);
                          });
      });
  return reactor;
}

/// Authenticate, parse, and asynchronously resolve GetSchema.
::grpc::ServerUnaryReactor* CallbackFlightService::GetSchema(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::SchemaResult* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::GetSchema, context,
                                     &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  FlightDescriptor descr;
  auto arrow_st = ParseRequiredRequest(request, "FlightDescriptor", &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  impl_->base()
      ->GetSchema(flight_context, descr)
      .AddCallback(
          [reactor, response, flight_context = std::move(flight_context)](
              const arrow::Result<std::unique_ptr<SchemaResult>>& result) mutable {
            FinishUnaryResult(reactor, response, std::move(flight_context), result,
                              "Flight not found",
                              [](const SchemaResult& schema, pb::SchemaResult* out) {
                                return internal::ToProto(schema, out);
                              });
          });
  return reactor;
}

/// Authenticate DoGet, parse its ticket, and stream the async source.
::grpc::ServerWriteReactor<pb::FlightData>* CallbackFlightService::DoGet(
    ::grpc::CallbackServerContext* context, const pb::Ticket* request) {
  GrpcServerCallContext flight_context(context);
  auto st =
      PrepareAuthenticatedCall(helper_, FlightMethod::DoGet, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::FlightData>(st);
  Ticket ticket;
  auto arrow_st = ParseRequiredRequest(request, "ticket", &ticket);
  if (!arrow_st.ok()) {
    return FinishWriteNow<pb::FlightData>(flight_context.FinishRequest(arrow_st));
  }
  auto future = impl_->base()->DoGet(flight_context, ticket);
  return MakeDoGetReactor(context, impl_->executor(), std::move(flight_context),
                          std::move(future));
}

/// Create the shared async bidi reactor for DoPut.
::grpc::ServerBidiReactor<pb::FlightData, pb::PutResult>* CallbackFlightService::DoPut(
    ::grpc::CallbackServerContext* context) {
  return MakeDoPutReactor(context, impl_, helper_);
}

/// Create the shared async bidi reactor for DoExchange.
::grpc::ServerBidiReactor<pb::FlightData, pb::FlightData>*
CallbackFlightService::DoExchange(::grpc::CallbackServerContext* context) {
  return MakeDoExchangeReactor(context, impl_, helper_);
}

/// Authenticate ListActions and stream its returned action-type vector.
::grpc::ServerWriteReactor<pb::ActionType>* CallbackFlightService::ListActions(
    ::grpc::CallbackServerContext* context, const pb::Empty*) {
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::ListActions, context,
                                     &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::ActionType>(st);
  auto future = impl_->base()->ListActions(flight_context);
  return MakeListActionsReactor(context, impl_->executor(), std::move(flight_context),
                                std::move(future));
}

/// Authenticate DoAction, parse its request, and stream action results.
::grpc::ServerWriteReactor<pb::Result>* CallbackFlightService::DoAction(
    ::grpc::CallbackServerContext* context, const pb::Action* request) {
  GrpcServerCallContext flight_context(context);
  auto st =
      PrepareAuthenticatedCall(helper_, FlightMethod::DoAction, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::Result>(st);
  Action action;
  auto arrow_st = ParseRequiredRequest(request, "Action", &action);
  if (!arrow_st.ok()) {
    return FinishWriteNow<pb::Result>(flight_context.FinishRequest(arrow_st));
  }
  auto future = impl_->base()->DoAction(flight_context, action);
  return MakeDoActionReactor(context, impl_->executor(), std::move(flight_context),
                             std::move(future));
}

}  // namespace arrow::flight::transport::grpc::async_internal

namespace arrow::flight::internal {

/// Create the callback-API gRPC transport selected by the async transport factory.
arrow::Result<std::unique_ptr<AsyncServerTransport>> MakeGrpcCallbackServerTransport(
    AsyncFlightServerBase* base, std::shared_ptr<MemoryManager> memory_manager) {
  return std::unique_ptr<AsyncServerTransport>(
      new transport::grpc::async_internal::AsyncGrpcServerTransport(
          base, std::move(memory_manager)));
}

}  // namespace arrow::flight::internal