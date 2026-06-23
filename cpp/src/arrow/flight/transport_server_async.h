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
#include <memory>

#include "arrow/flight/server_async.h"
#include "arrow/flight/transport_server.h"
#include "arrow/flight/transport_server_internal.h"
#include "arrow/flight/type_fwd.h"
#include "arrow/flight/visibility.h"

namespace arrow::flight {

namespace internal {
class AsyncServerTransport;
}

/// \brief Create an async server transport backed by gRPC callback API.
///
/// \param[in] base The async Flight server whose RPC methods will be called.
/// \param[in] memory_manager The memory manager for buffer allocations.
ARROW_FLIGHT_EXPORT
arrow::Result<std::unique_ptr<internal::AsyncServerTransport>>
MakeGrpcCallbackServerTransport(AsyncFlightServerBase* base,
                                std::shared_ptr<MemoryManager> memory_manager);

}  // namespace arrow::flight

namespace arrow::flight::internal {

/// \brief An implementation of an async Flight server for a particular transport.
///
/// This class (the transport implementation) implements the underlying
/// async server and handles connections/incoming RPC calls. It should forward
/// RPC calls to the RPC handlers defined on this class, which work in terms of
/// the generic ServerDataStream interfaces. The RPC handlers then forward calls
/// to the underlying AsyncFlightServerBase instance that contains the actual
/// application RPC method handlers.
///
/// Used by AsyncFlightServerBase to manage the server lifecycle.
class ARROW_FLIGHT_EXPORT AsyncServerTransport : public ServerTransportBase {
 public:
  AsyncServerTransport(AsyncFlightServerBase* base,
                       std::shared_ptr<MemoryManager> memory_manager)
      : ServerTransportBase(std::move(memory_manager)), base_(base) {}
  virtual ~AsyncServerTransport() = default;

  /// \name Server Lifecycle Methods
  /// Transports implement these methods to start/shutdown the underlying
  /// async server.
  /// @{
  /// \brief Initialize the server.
  ///
  /// This method should launch the server in a background thread, i.e. it
  /// should not block. Once this returns, the server should be active.
  virtual Status Init(const FlightServerOptions& options,
                      const arrow::util::Uri& uri) = 0;
  /// \brief Shutdown the server.
  ///
  /// This should wait for active RPCs to finish. Once this returns, the
  /// server is no longer listening.
  virtual Status Shutdown() = 0;
  /// \brief Shutdown the server with a deadline.
  ///
  /// This should wait for active RPCs to finish, or for the deadline to
  /// expire. Once this returns, the server is no longer listening.
  virtual Status Shutdown(const std::chrono::system_clock::time_point& deadline) = 0;
  /// \brief Wait for the server to shutdown (but do not shut down the server).
  ///
  /// Once this returns, the server is no longer listening.
  virtual Status Wait() = 0;
  /// \brief Get the address the server is listening on, else an empty Location.
  virtual Location location() const = 0;
  ///@}

  /// \brief Get the AsyncFlightServerBase.
  ///
  /// Intended as an escape hatch for now since not all methods have been
  /// factored into a transport-agnostic interface.
  AsyncFlightServerBase* base() const { return base_; }

  /// \brief Implement DoGet in terms of a transport-level stream.
  ///
  /// Adapts the Arrow-native async server surface to the transport-facing
  /// ServerDataStream helpers shared with the synchronous transport.
  ///
  /// \param[in] context The server context.
  /// \param[in] request The request payload.
  /// \param[in] stream The transport-specific data stream
  ///   implementation. Must implement WriteData(const FlightPayload&).
  Status DoGet(const ServerCallContext& context, const Ticket& request,
               ServerDataStream* stream);

 protected:
  AsyncFlightServerBase* base_;
};

}  // namespace arrow::flight::internal
