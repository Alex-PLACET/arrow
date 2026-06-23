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

class ARROW_FLIGHT_EXPORT AsyncFlightMessageReader {
 public:
  virtual ~AsyncFlightMessageReader() = default;

  /// \brief Return the descriptor from the first FlightData message.
  ///
  /// This is available synchronously once the RPC hook is invoked.
  virtual const FlightDescriptor& descriptor() const = 0;

  /// \brief Resolve the schema for the inbound stream.
  ///
  /// This may require consuming inbound Flight data and dictionary messages.
  /// It is safe to call multiple times after the schema has been decoded.
  /// Concurrent calls to GetSchema() and/or Next() are not supported.
  virtual Future<std::shared_ptr<Schema>> GetSchema() = 0;

  /// \brief Read the next logical Flight stream chunk.
  ///
  /// A returned chunk may contain a record batch, application metadata only,
  /// or both fields set to null to indicate end-of-stream.
  /// Concurrent calls to Next() and/or GetSchema() are not supported.
  virtual Future<FlightStreamChunk> Next() = 0;

  virtual arrow::ipc::ReadStats stats() const = 0;
};

class ARROW_FLIGHT_EXPORT AsyncFlightMetadataWriter {
 public:
  virtual ~AsyncFlightMetadataWriter() = default;

  /// \brief Send a DoPut application metadata message to the client.
  ///
  /// Calls are ordered. Concurrent writes are not supported.
  virtual Future<> WriteMetadata(const Buffer& app_metadata) = 0;
};

class ARROW_FLIGHT_EXPORT AsyncFlightMessageWriter {
 public:
  virtual ~AsyncFlightMessageWriter() = default;

  /// \brief Start the outbound stream by sending the schema.
  ///
  /// Begin() may be called at most once before writing batches.
  /// Concurrent writes are not supported.
  virtual Future<> Begin(
      const std::shared_ptr<Schema>& schema,
      const ipc::IpcWriteOptions& options = ipc::IpcWriteOptions::Defaults()) = 0;

  /// \brief Send a record batch.
  ///
  /// Begin() must have completed successfully first.
  virtual Future<> WriteRecordBatch(const RecordBatch& batch) = 0;

  /// \brief Send application metadata without a record batch.
  virtual Future<> WriteMetadata(std::shared_ptr<Buffer> app_metadata) = 0;

  /// \brief Send a record batch and application metadata together.
  virtual Future<> WriteWithMetadata(const RecordBatch& batch,
                                     std::shared_ptr<Buffer> app_metadata) = 0;

  /// \brief Mark the writer closed.
  ///
  /// This is idempotent transport cleanup, not a replacement for the RPC
  /// completion Future returned by the server hook.
  virtual Future<> Close() = 0;
  virtual arrow::ipc::WriteStats stats() const = 0;
};

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
  /// \brief Handle DoPut using the async-native stream API.
  ///
  /// The default implementation does not bridge to the synchronous DoPut hook.
  /// Subclasses must override this explicitly if they want to support DoPut on
  /// AsyncFlightServerBase.
  virtual Future<> DoPutAsync(const ServerCallContext& context,
                              std::unique_ptr<AsyncFlightMessageReader> reader,
                              std::unique_ptr<AsyncFlightMetadataWriter> writer);
  virtual Future<> DoPut(const ServerCallContext& context,
                         std::unique_ptr<FlightMessageReader> reader,
                         std::unique_ptr<FlightMetadataWriter> writer);
  /// \brief Handle DoExchange using the async-native stream API.
  ///
  /// The default implementation does not bridge to the synchronous DoExchange
  /// hook. Subclasses must override this explicitly if they want to support
  /// DoExchange on AsyncFlightServerBase.
  virtual Future<> DoExchangeAsync(const ServerCallContext& context,
                                   std::unique_ptr<AsyncFlightMessageReader> reader,
                                   std::unique_ptr<AsyncFlightMessageWriter> writer);
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
