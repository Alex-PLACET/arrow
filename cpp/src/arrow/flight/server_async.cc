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

// Standalone async Flight server implementation on top of the transport
// interface.

#include "arrow/flight/server_async.h"

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/flight/transport.h"
#include "arrow/flight/transport/grpc/grpc_server.h"
#include "arrow/flight/transport_server.h"
#include "arrow/flight/transport_server_internal.h"
#include "arrow/flight/types.h"

namespace arrow::flight {

AsyncFlightDataStream::~AsyncFlightDataStream() = default;


struct AsyncGenericFlightServerBase::Impl {
  std::unique_ptr<internal::ServerTransport> transport;
  internal::ServerSignalState signal_state;
};

AsyncGenericFlightServerBase::AsyncGenericFlightServerBase() : impl_(new Impl) {}

AsyncGenericFlightServerBase::~AsyncGenericFlightServerBase() = default;

Status AsyncGenericFlightServerBase::Init(const FlightServerOptions& options) {
  flight::transport::grpc::InitializeFlightGrpcServer();

  const auto scheme = options.location.scheme();
  ARROW_ASSIGN_OR_RAISE(impl_->transport,
                        internal::GetDefaultTransportRegistry()->MakeServerAsync(
                            scheme, this, options.memory_manager));
  ARROW_ASSIGN_OR_RAISE(auto uri, internal::ParseLocationUri(options.location));
  return impl_->transport->Init(options, uri);
}

// The five lifecycle methods below mirror FlightServerBase's (server.cc:89-140).
int AsyncGenericFlightServerBase::port() const {
  return internal::PortFromLocation(location());
}

Location AsyncGenericFlightServerBase::location() const {
  return impl_->transport->location();
}

Status AsyncGenericFlightServerBase::SetShutdownOnSignals(
    const std::vector<int> signals) {
  return impl_->signal_state.SetShutdownOnSignals(signals);
}

Status AsyncGenericFlightServerBase::Serve() {
  return impl_->signal_state.Serve(
      [this]() -> Status {
        if (!impl_->transport) {
          return Status::UnknownError("Server did not start properly");
        }
        return impl_->transport->Wait();
      },
      [this](const std::chrono::system_clock::time_point* deadline) -> Status {
        if (!impl_->transport) {
          return Status::Invalid(
              "Shutdown() on uninitialized AsyncGenericFlightServerBase");
        }
        if (deadline) {
          return impl_->transport->Shutdown(*deadline);
        }
        return impl_->transport->Shutdown();
      },
      "Server did not start properly", "Error shutting down server");
}

int AsyncGenericFlightServerBase::GotSignal() const {
  return impl_->signal_state.GotSignal();
}

Status AsyncGenericFlightServerBase::Shutdown(
    const std::chrono::system_clock::time_point* deadline) {
  // Shutdown() takes the callback and the deadline only; the two message strings
  // belong to Serve() alone.
  return impl_->signal_state.Shutdown(
      [this](const std::chrono::system_clock::time_point* maybe_deadline) -> Status {
        if (!impl_->transport) {
          return Status::Invalid(
              "Shutdown() on uninitialized AsyncGenericFlightServerBase");
        }
        if (maybe_deadline) {
          return impl_->transport->Shutdown(*maybe_deadline);
        }
        return impl_->transport->Shutdown();
      },
      deadline);
}

Status AsyncGenericFlightServerBase::Wait() {
  return impl_->signal_state.Wait([this] {
    if (!impl_->transport) {
      return Status::Invalid("Wait() on uninitialized AsyncGenericFlightServerBase");
    }
    return impl_->transport->Wait();
  });
}

// --- handler defaults: the same answers FlightServerBase's defaults give ---

Status AsyncGenericFlightServerBase::ListFlights(const ServerCallContext&,
                                                 const Criteria*,
                                                 std::unique_ptr<FlightListing>*) {
  return Status::NotImplemented("NYI");
}

Status AsyncGenericFlightServerBase::GetFlightInfo(const ServerCallContext&,
                                                   const FlightDescriptor&,
                                                   std::unique_ptr<FlightInfo>*) {
  return Status::NotImplemented("NYI");
}

Status AsyncGenericFlightServerBase::PollFlightInfo(const ServerCallContext&,
                                                    const FlightDescriptor&,
                                                    std::unique_ptr<PollInfo>*) {
  return Status::NotImplemented("NYI");
}

Status AsyncGenericFlightServerBase::GetSchema(const ServerCallContext&,
                                               const FlightDescriptor&,
                                               std::unique_ptr<SchemaResult>*) {
  return Status::NotImplemented("NYI");
}

Status AsyncGenericFlightServerBase::DoAction(const ServerCallContext&, const Action&,
                                              std::unique_ptr<ResultStream>*) {
  return Status::NotImplemented("NYI");
}

Status AsyncGenericFlightServerBase::ListActions(const ServerCallContext&,
                                                 std::vector<ActionType>*) {
  return Status::NotImplemented("NYI");
}

Status AsyncGenericFlightServerBase::DoExchange(const ServerCallContext&,
                                                std::unique_ptr<FlightMessageReader>,
                                                std::unique_ptr<FlightMessageWriter>) {
  return Status::NotImplemented("NYI");
}

arrow::Future<std::unique_ptr<AsyncFlightDataStream>>
AsyncGenericFlightServerBase::DoGetAsync(const ServerCallContext&, const Ticket&) {
  return arrow::Future<std::unique_ptr<AsyncFlightDataStream>>::MakeFinished(
      Status::NotImplemented("DoGetAsync is not implemented"));
}

Status AsyncGenericFlightServerBase::Handshake(const ServerCallContext&,
                                               const std::string&, std::string*) {
  return Status::NotImplemented(
      "This service does not have an authentication mechanism enabled.");
}

Status AsyncGenericFlightServerBase::ValidateToken(const ServerCallContext&,
                                                   const std::string&, std::string*) {
  // Keep the transport-level (TLS) identity; ignore the token.
  return Status::OK();
}

}  // namespace arrow::flight
