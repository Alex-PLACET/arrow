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
  /// Concurrent calls to GetSchema() and/or Next() are not supported. Callers
  /// should wait for the returned Future to finish before issuing another read.
  virtual Future<std::shared_ptr<Schema>> GetSchema() = 0;

  /// \brief Read the next logical Flight stream chunk.
  ///
  /// A returned chunk may contain a record batch, application metadata only,
  /// or both fields set to null to indicate end-of-stream.
  /// Concurrent calls to Next() and/or GetSchema() are not supported. Chunks
  /// are ordered exactly as they arrive on the Flight stream.
  virtual Future<FlightStreamChunk> Next() = 0;

  virtual arrow::ipc::ReadStats stats() const = 0;
};

class ARROW_FLIGHT_EXPORT AsyncFlightMetadataWriter {
 public:
  virtual ~AsyncFlightMetadataWriter() = default;

  /// \brief Send a DoPut application metadata message to the client.
  ///
  /// Calls are ordered. Concurrent writes are not supported, so each returned
  /// Future should finish before the next write is issued.
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
///
/// AsyncFlightServerBase is a parallel server model to FlightServerBase. It
/// preserves the same Flight RPC surface and lifecycle methods, but RPC hooks
/// return Futures and can use async-native stream reader/writer interfaces for
/// bidirectional data movement.
///
/// The public API is transport-agnostic: subclasses do not interact with gRPC
/// reactor types directly. Transport adapters are responsible for bridging
/// between the callback transport and these Arrow-native interfaces.
///
/// There is intentionally no automatic bridge between the synchronous and
/// asynchronous server hooks. In particular, subclasses that need async DoPut
/// or DoExchange support must override DoPutAsync() and DoExchangeAsync()
/// explicitly.
class ARROW_FLIGHT_EXPORT AsyncFlightServerBase {
 public:
  AsyncFlightServerBase();
  virtual ~AsyncFlightServerBase();

  /// \brief Initialize the underlying server transport.
  ///
  /// Init() must be called before Serve(), Shutdown(), or Wait().
  Status Init(const FlightServerOptions& options);
  int port() const;
  Location location() const;
  Status SetShutdownOnSignals(const std::vector<int> sigs);
  Status Serve();
  int GotSignal() const;
  Status Shutdown(const std::chrono::system_clock::time_point* deadline = NULLPTR);
  Status Wait();

  /// \brief Authenticate the client handshake stream.
  ///
  /// The default implementation returns NotImplemented.
  virtual Future<> Handshake(const ServerCallContext& context,
                             std::unique_ptr<ServerAuthSender> outgoing,
                             std::unique_ptr<ServerAuthReader> incoming);

  /// \brief Enumerate available flights.
  virtual Future<std::unique_ptr<FlightListing>> ListFlights(
      const ServerCallContext& context, const Criteria* criteria);
  /// \brief Resolve FlightInfo for a descriptor.
  virtual Future<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const ServerCallContext& context, const FlightDescriptor& request);
  /// \brief Resolve PollInfo for a descriptor.
  virtual Future<std::unique_ptr<PollInfo>> PollFlightInfo(
      const ServerCallContext& context, const FlightDescriptor& request);
  /// \brief Resolve schema metadata for a descriptor.
  virtual Future<std::unique_ptr<SchemaResult>> GetSchema(
      const ServerCallContext& context, const FlightDescriptor& request);
  /// \brief Create a server-to-client data stream.
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
