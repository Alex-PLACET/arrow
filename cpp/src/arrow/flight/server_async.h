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
#include <vector>

#include "arrow/flight/server.h"
#include "arrow/flight/server_auth.h"
#include "arrow/flight/visibility.h"
#include "arrow/util/future.h"

namespace arrow::flight {

/// \brief Skeleton asynchronous RPC server implementation.
class ARROW_FLIGHT_EXPORT AsyncFlightServerBase {
 public:
  AsyncFlightServerBase();
  virtual ~AsyncFlightServerBase();

  Status Init(const FlightServerOptions& options);
  int port() const;
  Location location() const;
  Status SetShutdownOnSignals(const std::vector<int> sigs);
  Status Serve();
  int GotSignal() const;
  Status Shutdown(const std::chrono::system_clock::time_point* deadline = NULLPTR);
  Status Wait();

  virtual Future<> Handshake(const ServerCallContext& context,
                             std::unique_ptr<ServerAuthSender> outgoing,
                             std::unique_ptr<ServerAuthReader> incoming);

  virtual Future<std::unique_ptr<FlightListing>> ListFlights(
      const ServerCallContext& context, const Criteria* criteria);
  virtual Future<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const ServerCallContext& context, const FlightDescriptor& request);
  virtual Future<std::unique_ptr<PollInfo>> PollFlightInfo(
      const ServerCallContext& context, const FlightDescriptor& request);
  virtual Future<std::unique_ptr<SchemaResult>> GetSchema(
      const ServerCallContext& context, const FlightDescriptor& request);
  virtual Future<std::unique_ptr<FlightDataStream>> DoGet(
      const ServerCallContext& context, const Ticket& request);
  virtual Future<> DoPut(const ServerCallContext& context,
                         std::unique_ptr<FlightMessageReader> reader,
                         std::unique_ptr<FlightMetadataWriter> writer);
  virtual Future<> DoExchange(const ServerCallContext& context,
                              std::unique_ptr<FlightMessageReader> reader,
                              std::unique_ptr<FlightMessageWriter> writer);
  virtual Future<std::unique_ptr<ResultStream>> DoAction(
      const ServerCallContext& context, const Action& action);
  virtual Future<std::vector<ActionType>> ListActions(
      const ServerCallContext& context);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace arrow::flight
