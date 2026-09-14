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

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>

#include "arrow/flight/server_async.h"
#include "arrow/flight/transport_server.h"
#include "arrow/flight/transport_server_internal.h"
#include "arrow/flight/type_fwd.h"
#include "arrow/flight/visibility.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/future.h"

namespace arrow::flight::internal {

class AsyncServerTransport;

arrow::Result<std::unique_ptr<AsyncServerTransport>> MakeAsyncServerTransport(
    const std::string& scheme, AsyncFlightServerBase* base,
    std::shared_ptr<MemoryManager> memory_manager);

/// \brief An implementation of an async Flight server for a particular transport.
///
/// Transports implement the server lifecycle (Init/Shutdown/Wait/location)
/// and convert transport-specific RPC events (e.g. gRPC callback reactors)
/// into calls on the underlying AsyncFlightServerBase, reusing the generic
/// orchestration provided here (AsyncStreamDriver, and the async reader and
/// writer factories of the grpc layer).
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

 protected:
  AsyncFlightServerBase* base_;
};

/// Drives an AsyncFlightDataStream to completion through a write callback.
///
/// This is the async version of ServerTransportBase::WriteDataStream. It sends
/// the schema first, then sends each payload in order. It waits for each write
/// to finish before pulling the next payload. It closes the stream at the end.
/// A null stream fails with "No data in this flight". Each payload is validated
/// before it is sent.
///
/// A payload or validation error closes the stream and is returned by Run().
/// A failed write also closes the stream. In that case, Run() returns the close
/// status. This treats a lost connection as a clean end, like the sync path.
/// A payload error takes precedence over the close status.
///
/// `is_cancelled` is checked between steps. RequestClose() stops the stream
/// immediately and is safe to call from any thread. Once closing starts, the
/// write callback is not called again.
///
/// Keep the returned shared_ptr while the stream is running and while
/// RequestClose() may be called. Call Run() exactly once. Its future is the
/// only completion signal. WriteFn returns when the transport finishes a
/// payload, so only one write is pending at a time.
class AsyncStreamDriver : public std::enable_shared_from_this<AsyncStreamDriver> {
 public:
  /// Sink for one serialized payload, the boolean reports write success.
  using WriteFn = std::function<Future<bool>(FlightPayload)>;
  /// Polled between steps, true stops the stream.
  using CancelFn = std::function<bool()>;

  AsyncStreamDriver(std::unique_ptr<AsyncFlightDataStream> stream, CancelFn is_cancelled,
                    WriteFn write_fn);

  /// Start emitting schema, payloads, and the end-of-stream marker.
  /// Completes once the stream is closed, with the close status or the
  /// terminal failure.
  Future<> Run();

  /// Close the stream (e.g. on RPC cancellation), completing Run() with
  /// `failure` once the close finishes
  //  OK reports the close status.
  void RequestClose(Status failure = Status::OK());

 private:
  void PullNext(bool first);
  void StartWrite(FlightPayload payload);
  void BeginClose(Status failure = Status::OK());
  void CloseStream();

  std::unique_ptr<AsyncFlightDataStream> stream_;
  CancelFn is_cancelled_;
  WriteFn write_fn_;
  Future<> out_;
  /// Protects the single terminal close path.
  std::atomic<bool> close_started_{false};
  /// Terminal failure reported once the stream is closed (OK = close status).
  Status close_failure_;
};

arrow::Result<std::unique_ptr<AsyncServerTransport>> MakeGrpcCallbackServerTransport(
    AsyncFlightServerBase* base, std::shared_ptr<MemoryManager> memory_manager);

}  // namespace arrow::flight::internal