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

#include "arrow/flight/transport/grpc/async_grpc_service.h"

#include <algorithm>
#include <string_view>
#include <utility>

#include "arrow/flight/transport/grpc/async_grpc_service_internal.h"

namespace arrow::flight::transport::grpc {

namespace {

namespace detail = arrow::flight::transport::grpc::detail;

constexpr std::string_view kPrefix = "/arrow.flight.protocol.FlightService/";
constexpr std::string_view kHandshakeMethod = "Handshake";
constexpr std::string_view kDoGetMethod = "DoGet";
constexpr std::string_view kDoPutMethod = "DoPut";
constexpr std::string_view kDoActionMethod = "DoAction";
constexpr std::string_view kListActionsMethod = "ListActions";
constexpr std::string_view kDoExchangeMethod = "DoExchange";
constexpr std::string_view kPollFlightInfoMethod = "PollFlightInfo";
constexpr std::string_view kListFlightsMethod = "ListFlights";
constexpr std::string_view kGetFlightInfoMethod = "GetFlightInfo";
constexpr std::string_view kGetSchemaMethod = "GetSchema";

FlightMethod MethodFromName(std::string_view method) {
  if (!method.starts_with(kPrefix)) {
    return FlightMethod::Invalid;
  }
  method.remove_prefix(kPrefix.size());
  constexpr std::pair<std::string_view, FlightMethod> kMethods[] = {
      {kHandshakeMethod, FlightMethod::Handshake},
      {kListFlightsMethod, FlightMethod::ListFlights},
      {kGetFlightInfoMethod, FlightMethod::GetFlightInfo},
      {kGetSchemaMethod, FlightMethod::GetSchema},
      {kDoGetMethod, FlightMethod::DoGet},
      {kDoPutMethod, FlightMethod::DoPut},
      {kDoActionMethod, FlightMethod::DoAction},
      {kListActionsMethod, FlightMethod::ListActions},
      {kDoExchangeMethod, FlightMethod::DoExchange},
      {kPollFlightInfoMethod, FlightMethod::PollFlightInfo},
  };

  const auto it = std::ranges::find_if(
      kMethods, [method](const auto& pair) { return pair.first == method; });
  return it != std::end(kMethods) ? it->second : FlightMethod::Invalid;
}

class Unimplemented : public ::grpc::ServerGenericBidiReactor {
 public:
  explicit Unimplemented(::grpc::Status status) : status_(std::move(status)) {
    Finish(status_);
  }
  void OnDone() override { delete this; }

 private:
  ::grpc::Status status_;
};

}  // namespace

AsyncGenericFlightService::AsyncGenericFlightService(
    AsyncGenericFlightServerBase* async_base,
    std::shared_ptr<GrpcServerCallContextHelper<::grpc::CallbackServerContext>> helper,
    HandshakeFn handshake_handler)
    : base_(async_base),
      helper_(std::move(helper)),
      handshake_handler_(std::move(handshake_handler)) {}

::grpc::ServerGenericBidiReactor* AsyncGenericFlightService::CreateReactor(
    ::grpc::GenericCallbackServerContext* context) {
  const std::string_view full_method = context->method();
  const std::string_view method =
      full_method.starts_with(kPrefix) ? full_method.substr(kPrefix.size()) : full_method;
  const FlightMethod flight_method = MethodFromName(full_method);
  detail::AsyncCallContext flight_context(context);

  const auto prepare_status =
      method == kHandshakeMethod
          ? helper_->MakeCallContext(flight_method, context, &flight_context)
          : helper_->CheckAuth(flight_method, context, &flight_context);
  if (!prepare_status.ok()) {
    return new Unimplemented(prepare_status);
  }

  if (method == kHandshakeMethod) {
    if (!handshake_handler_) {
      return new Unimplemented(::grpc::Status(
          ::grpc::StatusCode::UNIMPLEMENTED,
          "This service does not have an authentication mechanism enabled."));
    }
    return detail::MakeHandshakeReactor(std::move(flight_context), handshake_handler_);
  }
  if (method == kGetFlightInfoMethod) {
    return detail::MakeGetFlightInfoReactor(std::move(flight_context), base_);
  }
  if (method == kGetSchemaMethod) {
    return detail::MakeGetSchemaReactor(std::move(flight_context), base_);
  }
  if (method == kPollFlightInfoMethod) {
    return detail::MakePollFlightInfoReactor(std::move(flight_context), base_);
  }
  if (method == kListActionsMethod) {
    return detail::MakeListActionsReactor(std::move(flight_context), base_);
  }
  if (method == kDoActionMethod) {
    return detail::MakeDoActionReactor(std::move(flight_context), base_);
  }
  if (method == kListFlightsMethod) {
    return detail::MakeListFlightsReactor(std::move(flight_context), base_);
  }
  if (method == kDoExchangeMethod) {
    return detail::MakeExchangeReactor(std::move(flight_context), base_);
  }
  if (method == kDoGetMethod) {
    return detail::MakeDoGetReactor(std::move(flight_context), base_);
  }
  if (method == kDoPutMethod) {
    std::shared_ptr<AsyncFlightDataListener> listener =
        base_->CreateDoPutListener(flight_context);
    if (!listener) {
      return new Unimplemented(
          ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED,
                         "DoPut is not implemented: no listener available"));
    }
    return detail::MakeDoPutReactor(std::move(flight_context), std::move(listener));
  }
  return new Unimplemented(
      ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Unknown method"));
}

}  // namespace arrow::flight::transport::grpc
