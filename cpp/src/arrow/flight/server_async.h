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
  /// A cancelled RPC completes pending reads with end-of-stream.
  virtual Future<std::shared_ptr<Schema>> GetSchema() = 0;

  /// \brief Read the next logical Flight stream chunk.
  ///
  /// A returned chunk may contain a record batch, application metadata only,
  /// or both fields set to null to indicate end-of-stream.
  /// Concurrent calls to Next() and/or GetSchema() are not supported. Chunks
  /// are ordered exactly as they arrive on the Flight stream.
  virtual Future<FlightStreamChunk> Next() = 0;

  /// \brief Return IPC read statistics accumulated by this stream.
  virtual arrow::ipc::ReadStats stats() const = 0;
};

/// \brief An asynchronous server-to-client Flight stream.
///
/// Each operation is ordered: callers must wait for its Future before starting
/// the next operation. Implementations may complete a Future inline.
class ARROW_FLIGHT_EXPORT AsyncFlightDataStream {
 public:
  virtual ~AsyncFlightDataStream() = default;

  /// \brief Asynchronously return the schema payload.
  ///
  /// The transport calls this exactly once before Next().
  virtual Future<FlightPayload> GetSchemaPayload() = 0;

  /// \brief Asynchronously return the next Flight payload.
  ///
  /// A payload with null IPC metadata marks end-of-stream. After that, the
  /// transport calls Close().
  virtual Future<FlightPayload> Next() = 0;

  /// \brief Release stream resources.
  ///
  /// The transport calls this once after normal completion, an error, or
  /// cancellation. Implementations should make it idempotent.
  virtual Future<> Close() = 0;
};

/// \brief Adapt a legacy synchronous FlightDataStream to AsyncFlightDataStream.
///
/// "Legacy" refers only to the synchronous FlightDataStream interface. It is
/// supported as a compatibility bridge, not deprecated. This adapter completes
/// each Future by calling the synchronous stream method, so it does not make a
/// blocking stream non-blocking. New async servers should implement
/// AsyncFlightDataStream directly.
ARROW_FLIGHT_EXPORT std::unique_ptr<AsyncFlightDataStream>
MakeAsyncFlightDataStreamFromSync(std::unique_ptr<FlightDataStream> stream);

class ARROW_FLIGHT_EXPORT AsyncFlightMetadataWriter {
 public:
  virtual ~AsyncFlightMetadataWriter() = default;

  /// \brief Send a DoPut application metadata message to the client.
  ///
  /// Calls are ordered. Concurrent writes are not supported, so each returned
  /// Future should finish before the next write is issued. A cancelled RPC
  /// completes a pending write with an error.
  virtual Future<> WriteMetadata(const Buffer& app_metadata) = 0;
};

class ARROW_FLIGHT_EXPORT AsyncFlightMessageWriter {
 public:
  virtual ~AsyncFlightMessageWriter() = default;

  /// \brief Start the outbound stream by sending the schema.
  ///
  /// Begin() may be called at most once before writing batches. A null schema,
  /// or a call after Close(), is invalid. Concurrent writes are not supported.
  virtual Future<> Begin(
      const std::shared_ptr<Schema>& schema,
      const ipc::IpcWriteOptions& options = ipc::IpcWriteOptions::Defaults()) = 0;

  /// \brief Send a record batch.
  ///
  /// Begin() must have completed successfully first.
  virtual Future<> WriteRecordBatch(const RecordBatch& batch) = 0;

  /// \brief Send application metadata without a record batch.
  ///
  /// Calls are ordered with Begin(), WriteRecordBatch(), and
  /// WriteWithMetadata(). Wait for each returned Future before the next write.
  virtual Future<> WriteMetadata(std::shared_ptr<Buffer> app_metadata) = 0;

  /// \brief Send a record batch and application metadata together.
  ///
  /// Begin() must have completed successfully first.
  virtual Future<> WriteWithMetadata(const RecordBatch& batch,
                                     std::shared_ptr<Buffer> app_metadata) = 0;

  /// \brief Mark the writer closed.
  ///
  /// This is idempotent transport cleanup, not a replacement for the RPC
  /// completion Future returned by the server hook. It does not cancel an
  /// already pending write; callers must await it before calling Close(). All
  /// subsequent writes are invalid.
  virtual Future<> Close() = 0;

  /// \brief Return IPC write statistics accumulated by this stream.
  virtual arrow::ipc::WriteStats stats() const = 0;
};

/// \brief Experimental skeleton asynchronous RPC server implementation.
///
/// AsyncFlightServerBase is a parallel server model to FlightServerBase. It
/// preserves the Flight RPC surface and lifecycle methods, but RPC hooks return
/// Futures and use async-native stream reader/writer interfaces for
/// bidirectional data movement. The synchronous FlightServerBase hooks are not
/// part of this interface.
///
/// The public API is transport-agnostic: subclasses do not interact with gRPC
/// reactor types directly. Transport adapters are responsible for bridging
/// between the callback transport and these Arrow-native interfaces.
///
/// Continuations may execute inline on a transport callback thread. Applications
/// must not block them and must await all reader and writer Futures before the
/// RPC hook Future completes.
class ARROW_FLIGHT_EXPORT AsyncFlightServerBase {
 public:
  /// \brief Construct an uninitialized async Flight server.
  AsyncFlightServerBase();
  /// \brief Destroy the server and its transport state.
  virtual ~AsyncFlightServerBase();

  /// \brief Initialize the underlying server transport.
  ///
  /// Init() must be called before Serve(), Shutdown(), or Wait().
  Status Init(const FlightServerOptions& options);

  /// \brief Return the listening port after Init().
  int port() const;

  /// \brief Return the bound listening location after Init().
  Location location() const;

  /// \brief Arrange for the server to shut down when one of sigs is received.
  Status SetShutdownOnSignals(const std::vector<int> sigs);

  /// \brief Block until the server is shut down.
  Status Serve();

  /// \brief Return the signal that triggered shutdown, or zero if none.
  int GotSignal() const;

  /// \brief Stop accepting RPCs and wait for active RPCs to finish.
  ///
  /// If a deadline is supplied, return when it expires even if RPCs remain.
  Status Shutdown(const std::chrono::system_clock::time_point* deadline = NULLPTR);

  /// \brief Wait until the server has shut down without initiating shutdown.
  Status Wait();

  /// \brief Authenticate the client handshake stream.
  ///
  /// The default implementation returns NotImplemented.
  virtual Future<> Handshake(const ServerCallContext& context,
                             std::unique_ptr<ServerAuthSender> outgoing,
                             std::unique_ptr<ServerAuthReader> incoming);

  /// \brief Enumerate flights matching criteria.
  ///
  /// Return a listing whose Next() is consumed by the transport until exhausted.
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

  /// \brief Create an asynchronous server-to-client data stream for a ticket.
  ///
  /// This is the asynchronous counterpart to FlightServerBase::DoGet(). Return
  /// a Future containing the stream once it is ready. Use
  /// MakeAsyncFlightDataStreamFromSync() to adapt an existing FlightDataStream
  /// during migration.
  virtual Future<std::unique_ptr<AsyncFlightDataStream>> DoGet(
      const ServerCallContext& context, const Ticket& request);

  /// \brief Handle DoPut using async-native inbound and metadata streams.
  ///
  /// Resolve the returned Future after all reader and writer operations started
  /// by this hook have completed.
  virtual Future<> DoPut(const ServerCallContext& context,
                         std::unique_ptr<AsyncFlightMessageReader> reader,
                         std::unique_ptr<AsyncFlightMetadataWriter> writer);

  /// \brief Handle DoExchange using async-native bidirectional streams.
  ///
  /// Resolve the returned Future after all reader and writer operations started
  /// by this hook have completed.
  virtual Future<> DoExchange(const ServerCallContext& context,
                              std::unique_ptr<AsyncFlightMessageReader> reader,
                              std::unique_ptr<AsyncFlightMessageWriter> writer);

  /// \brief Execute an action and return its result stream.
  virtual Future<std::unique_ptr<ResultStream>> DoAction(const ServerCallContext& context,
                                                         const Action& action);

  /// \brief Return the action types supported by this server.
  virtual Future<std::vector<ActionType>> ListActions(const ServerCallContext& context);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace arrow::flight
