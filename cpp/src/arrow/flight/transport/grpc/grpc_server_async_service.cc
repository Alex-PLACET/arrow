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

namespace {

template <typename Proto>
class ImmediateWriteReactor final : public ::grpc::ServerWriteReactor<Proto> {
 public:
  explicit ImmediateWriteReactor(::grpc::Status status) {
    this->Finish(std::move(status));
  }

  void OnDone() override { delete this; }
};

template <typename Proto>
::grpc::ServerWriteReactor<Proto>* FinishWriteNow(const ::grpc::Status& status) {
  return new ImmediateWriteReactor<Proto>(status);
}

template <typename Proto, typename Handler>
::grpc::ServerWriteReactor<Proto>* HandleAuthenticatedStreamingCall(
    const CallbackServiceHelper& helper, FlightMethod method,
    ::grpc::CallbackServerContext* context, Handler&& handler) {
  GrpcServerCallContext flight_context(context);
  const auto status = PrepareAuthenticatedCall(helper, method, context, &flight_context);
  if (!status.ok()) {
    return FinishWriteNow<Proto>(status);
  }
  return std::forward<Handler>(handler)(std::move(flight_context));
}

/// Authenticate, parse, and resolve a descriptor-based unary Flight request.
template <typename ArrowResponse, typename ProtoResponse, FlightMethod method,
          Future<std::unique_ptr<ArrowResponse>> (AsyncFlightServerBase::*resolve)(
              const ServerCallContext&, const FlightDescriptor&)>
::grpc::ServerUnaryReactor* HandleDescriptorUnary(AsyncGrpcServerTransport* impl,
                                                  const CallbackServiceHelper& helper,
                                                  ::grpc::CallbackServerContext* context,
                                                  const pb::FlightDescriptor* request,
                                                  ProtoResponse* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  const auto status = PrepareAuthenticatedCall(helper, method, context, &flight_context);
  if (!status.ok()) {
    reactor->Finish(status);
    return reactor;
  }
  FlightDescriptor descriptor;
  auto arrow_status = ParseRequiredRequest(request, "FlightDescriptor", &descriptor);
  if (!arrow_status.ok()) {
    reactor->Finish(flight_context.FinishRequest(arrow_status));
    return reactor;
  }
  (impl->base()->*resolve)(flight_context, descriptor)
      .AddCallback(
          [reactor, response, flight_context = std::move(flight_context)](
              const arrow::Result<std::unique_ptr<ArrowResponse>>& result) mutable {
            FinishUnaryResult(reactor, response, std::move(flight_context), result,
                              [](const ArrowResponse& value, ProtoResponse* out) {
                                return internal::ToProto(value, out);
                              });
          });
  return reactor;
}

}  // namespace

/// Bridge the generated callback service to the async handshake reactor.
::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>*
CallbackFlightService::Handshake(::grpc::CallbackServerContext* context) {
  return MakeHandshakeReactor(context, impl_, helper_);
}

/// Authenticate ListFlights and stream the returned FlightListing iterator.
::grpc::ServerWriteReactor<pb::FlightInfo>* CallbackFlightService::ListFlights(
    ::grpc::CallbackServerContext* context, const pb::Criteria* request) {
  return HandleAuthenticatedStreamingCall<pb::FlightInfo>(
      helper_, FlightMethod::ListFlights, context,
      [this, context, request](GrpcServerCallContext flight_context) {
        Criteria criteria;
        if (request) {
          auto conv = internal::FromProto(*request, &criteria);
          if (!conv.ok()) {
            return FinishWriteNow<pb::FlightInfo>(flight_context.FinishRequest(conv));
          }
        }
        auto future = impl_->base()->ListFlights(flight_context, &criteria);
        return MakeListFlightsReactor(context, impl_->executor(),
                                      std::move(flight_context), std::move(future));
      });
}

/// Authenticate, parse, and asynchronously resolve GetFlightInfo.
::grpc::ServerUnaryReactor* CallbackFlightService::GetFlightInfo(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::FlightInfo* response) {
  return HandleDescriptorUnary<FlightInfo, pb::FlightInfo, FlightMethod::GetFlightInfo,
                               &AsyncFlightServerBase::GetFlightInfo>(
      impl_, helper_, context, request, response);
}

/// Authenticate, parse, and asynchronously resolve PollFlightInfo.
::grpc::ServerUnaryReactor* CallbackFlightService::PollFlightInfo(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::PollInfo* response) {
  return HandleDescriptorUnary<PollInfo, pb::PollInfo, FlightMethod::PollFlightInfo,
                               &AsyncFlightServerBase::PollFlightInfo>(
      impl_, helper_, context, request, response);
}

/// Authenticate, parse, and asynchronously resolve GetSchema.
::grpc::ServerUnaryReactor* CallbackFlightService::GetSchema(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::SchemaResult* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  const auto status = PrepareAuthenticatedCall(helper_, FlightMethod::GetSchema, context,
                                               &flight_context);
  if (!status.ok()) {
    reactor->Finish(status);
    return reactor;
  }

  FlightDescriptor descr;
  const auto arrow_status = ParseRequiredRequest(request, "FlightDescriptor", &descr);
  if (!arrow_status.ok()) {
    reactor->Finish(flight_context.FinishRequest(arrow_status));
    return reactor;
  }

  impl_->base()
      ->GetSchema(flight_context, descr)
      .AddCallback(
          [reactor, response, flight_context = std::move(flight_context)](
              const arrow::Result<std::unique_ptr<SchemaResult>>& result) mutable {
            FinishUnaryResult(reactor, response, std::move(flight_context), result,
                              [](const SchemaResult& schema, pb::SchemaResult* out) {
                                return internal::ToProto(schema, out);
                              });
          });
  return reactor;
}

/// Authenticate DoGet, parse its ticket, and stream the async source.
::grpc::ServerWriteReactor<pb::FlightData>* CallbackFlightService::DoGet(
    ::grpc::CallbackServerContext* context, const pb::Ticket* request) {
  return HandleAuthenticatedStreamingCall<pb::FlightData>(
      helper_, FlightMethod::DoGet, context,
      [this, context, request](GrpcServerCallContext flight_context) {
        Ticket ticket;
        const auto arrow_status = ParseRequiredRequest(request, "ticket", &ticket);
        if (!arrow_status.ok()) {
          return FinishWriteNow<pb::FlightData>(
              flight_context.FinishRequest(arrow_status));
        }
        auto future = impl_->base()->DoGet(flight_context, ticket);
        return MakeDoGetReactor(context, impl_->executor(), std::move(flight_context),
                                std::move(future));
      });
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
  return HandleAuthenticatedStreamingCall<pb::ActionType>(
      helper_, FlightMethod::ListActions, context,
      [this, context](GrpcServerCallContext flight_context) {
        auto future = impl_->base()->ListActions(flight_context);
        return MakeListActionsReactor(context, impl_->executor(),
                                      std::move(flight_context), std::move(future));
      });
}

/// Authenticate DoAction, parse its request, and stream action results.
::grpc::ServerWriteReactor<pb::Result>* CallbackFlightService::DoAction(
    ::grpc::CallbackServerContext* context, const pb::Action* request) {
  return HandleAuthenticatedStreamingCall<pb::Result>(
      helper_, FlightMethod::DoAction, context,
      [this, context, request](GrpcServerCallContext flight_context) {
        Action action;
        const auto arrow_status = ParseRequiredRequest(request, "Action", &action);
        if (!arrow_status.ok()) {
          return FinishWriteNow<pb::Result>(flight_context.FinishRequest(arrow_status));
        }
        auto future = impl_->base()->DoAction(flight_context, action);
        return MakeDoActionReactor(context, impl_->executor(), std::move(flight_context),
                                   std::move(future));
      });
}

}  // namespace arrow::flight::transport::grpc::async_internal