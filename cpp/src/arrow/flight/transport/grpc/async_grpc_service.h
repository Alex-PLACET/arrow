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

#include <memory>

#include <grpcpp/generic/callback_generic_service.h>

#include "arrow/flight/flight_data_decoder.h"
#include "arrow/flight/server_async.h"
#include "arrow/flight/transport/grpc/grpc_server_internal.h"

namespace arrow::flight::transport::grpc {

// The shared helper's async handshake hook: one Handshake RPC adapted to the
// server class's Handshake virtual.
using HandshakeFn =
    GrpcServerCallContextHelper<::grpc::CallbackServerContext>::HandshakeFn;

/// \brief Generic callback service dispatching Flight methods to reactors.
///
/// Constructed with the AsyncGenericFlightServerBase whose handlers are served.
/// The server's middleware runs for every call, through the shared context
/// helper, which also authenticates every method except Handshake.  Handshake,
/// DoGet, DoPut and the synchronous Flight methods (GetFlightInfo, GetSchema,
/// PollFlightInfo, ListActions, DoAction, ListFlights, DoExchange) have a
/// reactor; every other method is answered UNIMPLEMENTED.
class AsyncGenericFlightService : public ::grpc::CallbackGenericService {
 public:
  /// \param async_base is the async server whose handlers are served; it must
  /// outlive the service.
  /// \param listener_factory creates the FlightDataListener serving one DoPut
  /// RPC; the service consults it per RPC and refuses uploads when it is empty.
  /// \param memory_manager is the server transport's memory manager; the
  /// DoExchange reader uses it to view the bodies it reads.
  /// \param helper runs middleware and auth and builds the call context.
  /// \param handshake_handler is the server's Handshake hook; an empty hook
  /// means the server has no authentication mechanism.
  AsyncGenericFlightService(
      AsyncGenericFlightServerBase* async_base,
      FlightDataListenerFactory listener_factory,
      std::shared_ptr<MemoryManager> memory_manager,
      std::shared_ptr<GrpcServerCallContextHelper<::grpc::CallbackServerContext>> helper,
      HandshakeFn handshake_handler = {});

  /// \brief Creates a reactor for the incoming RPC.
  /// \param context is the gRPC callback server context for the RPC.
  /// \return a new reactor handling the RPC, or an Unimplemented reactor if the method is
  /// unknown.
  ::grpc::ServerGenericBidiReactor* CreateReactor(
      ::grpc::GenericCallbackServerContext* context) override;

 private:
  AsyncGenericFlightServerBase* base_;
  FlightDataListenerFactory listener_factory_;
  /// The server transport's memory manager; only DoExchange's reader uses it.
  std::shared_ptr<MemoryManager> memory_manager_;
  std::shared_ptr<GrpcServerCallContextHelper<::grpc::CallbackServerContext>> helper_;
  /// The server class's Handshake hook; empty when the server has none.
  HandshakeFn handshake_handler_;
};

}  // namespace arrow::flight::transport::grpc
