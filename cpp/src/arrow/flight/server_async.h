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
#include <string>
#include <vector>

#include "arrow/flight/server.h"
#include "arrow/flight/types.h"
#include "arrow/util/future.h"
#include "arrow/util/macros.h"

namespace arrow::flight {

/// \brief Interface that produces a sequence of IPC payloads to be sent in
/// FlightData protobuf messages, where a payload may arrive later.
///
/// The async transport pumps this stream through
/// NextAsync()/GetSchemaPayloadAsync() and never blocks a gRPC callback thread.
class ARROW_FLIGHT_EXPORT AsyncFlightDataStream {
 public:
  virtual ~AsyncFlightDataStream();

  virtual std::shared_ptr<Schema> schema() = 0;

  /// \brief Like GetSchemaPayload(), but the payload arrives later.
  virtual arrow::Future<FlightPayload> GetSchemaPayloadAsync() = 0;

  /// \brief Like Next(), but the payload arrives later.
  /// The last payload has null metadata, as in the synchronous interface.
  virtual arrow::Future<FlightPayload> NextAsync() = 0;

  virtual Status Close() { return Status::OK(); }
};

/// \brief a Flight server served by the async generic gRPC
/// callback service.
///
/// Derive from this class alone and implement the handlers you serve:
///
/// \code
/// class MyServer : public arrow::flight::AsyncGenericFlightServerBase {
///   arrow::Status GetFlightInfo(const arrow::flight::ServerCallContext&,
///                               const arrow::flight::FlightDescriptor&,
///                               std::unique_ptr<arrow::flight::FlightInfo>*) override;
///   arrow::Future<std::shared_ptr<arrow::flight::AsyncFlightDataStream>> DoGetAsync(
///       const arrow::flight::ServerCallContext&,
///       const arrow::flight::Ticket&) override;
/// };
/// \endcode
///
/// Every handler runs on a gRPC callback thread: return promptly and do the
/// waiting elsewhere (see `AsyncFlightDataStream`). Blocking a callback thread
/// starves new connections, because gRPC's thread pool grows at most about one
/// thread per second.
class ARROW_FLIGHT_EXPORT AsyncGenericFlightServerBase {
 public:
  AsyncGenericFlightServerBase();
  virtual ~AsyncGenericFlightServerBase();

  Status Init(const FlightServerOptions& options);
  int port() const;
  Location location() const;
  Status SetShutdownOnSignals(const std::vector<int> signals);
  Status Serve();
  int GotSignal() const;
  Status Shutdown(const std::chrono::system_clock::time_point* deadline = NULLPTR);
  Status Wait();

  /// \brief Retrieve a list of available fields given an optional opaque
  /// criteria.
  virtual Status ListFlights(const ServerCallContext& context, const Criteria* criteria,
                             std::unique_ptr<FlightListing>* listings);

  /// \brief Retrieve the schema and an access plan for the indicated
  /// descriptor.  See FlightServerBase::GetFlightInfo.
  virtual Status GetFlightInfo(const ServerCallContext& context,
                               const FlightDescriptor& request,
                               std::unique_ptr<FlightInfo>* info);

  /// \brief Retrieve the current status of the target query.  See
  /// FlightServerBase::PollFlightInfo.
  virtual Status PollFlightInfo(const ServerCallContext& context,
                                const FlightDescriptor& request,
                                std::unique_ptr<PollInfo>* info);

  /// \brief Retrieve the schema for the indicated descriptor.  See
  /// FlightServerBase::GetSchema.
  virtual Status GetSchema(const ServerCallContext& context,
                           const FlightDescriptor& request,
                           std::unique_ptr<SchemaResult>* schema);

  /// \brief Execute an action, return a stream of zero or more results.  See
  /// FlightServerBase::DoAction.
  virtual Status DoAction(const ServerCallContext& context, const Action& action,
                          std::unique_ptr<ResultStream>* result);

  /// \brief Retrieve the list of available actions.  See
  /// FlightServerBase::ListActions.
  virtual Status ListActions(const ServerCallContext& context,
                             std::vector<ActionType>* actions);

  /// \brief Process a bidirectional stream of IPC payloads.  See
  /// FlightServerBase::DoExchange.
  virtual Status DoExchange(const ServerCallContext& context,
                            std::unique_ptr<FlightMessageReader> reader,
                            std::unique_ptr<FlightMessageWriter> writer);

  /// \brief The stream serving one DoGet RPC, possibly later.
  ///
  /// \return Complete the returned future from your own thread or pool, the transport
  /// writes whatever stream it resolves to.
  /// \param[in] `context` is the server call context.
  /// \param[in] `request` is an opaque ticket The default answers UNIMPLEMENTED.
  virtual arrow::Future<std::shared_ptr<AsyncFlightDataStream>> DoGetAsync(
      const ServerCallContext& context, const Ticket& request);

  /// \brief Create the FlightDataListener serving one DoPut RPC.
  ///
  /// \param[in] `context` is the server call context.
  /// \return A shared pointer to the FlightDataListener handling the upload, or
  /// nullptr to refuse the upload.
  
  virtual std::shared_ptr<FlightDataListener> CreateDoPutListener(
      const ServerCallContext& context);

  /// \brief Handle the handshake protocol with the client.
  ///
  /// `request` is the client's handshake message; write the response into
  /// `*response`.  Return a non-OK status to fail the RPC
  /// (FlightStatusCode::Unauthenticated rejects the client).  Runs inline on a
  /// gRPC callback thread: do not block.  The default answers UNIMPLEMENTED.
  virtual Status Handshake(const ServerCallContext& context, const std::string& request,
                           std::string* response);

  /// \brief Validate the token sent in the `auth-token-bin` header of RPCs
  /// issued after a successful Handshake().
  ///
  /// Runs inline on the RPC dispatch path for every method except Handshake.  On
  /// success, set `peer_identity` to the authenticated identity.  The default
  /// keeps the transport-level (TLS) identity and ignores the token.
  virtual Status ValidateToken(const ServerCallContext& context, const std::string& token,
                               std::string* peer_identity);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace arrow::flight
