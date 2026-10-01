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
#include <optional>
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
///
/// The transport calls Close() at most once, from whichever ending comes
/// first (end of stream, write failure, or the client cancelling the RPC): a
/// stream whose payload future is still pending when the RPC dies gets
/// Close() from a transport thread, and resolving that pending future there
/// is how the stream is expected to stop.  Close() must not block.  When the
/// RPC is cancelled while DoGetAsync() is still preparing the stream, there
/// is no stream yet to Close(): the transport finishes the RPC anyway, and
/// preparation should observe ServerCallContext::is_cancelled() and give up
/// on its own.
class ARROW_FLIGHT_EXPORT AsyncFlightDataStream {
 public:
  virtual ~AsyncFlightDataStream();

  virtual std::shared_ptr<Schema> schema() = 0;

  /// \brief Like GetSchemaPayload(), but the payload arrives later.
  /// \return A future completed with the schema payload. A non-OK status fails the RPC.
  virtual arrow::Future<FlightPayload> GetSchemaPayloadAsync() = 0;

  /// \brief Like Next(), but the payload arrives later.
  /// \return A future completed with the next payload, or with a nullopt once the stream
  /// is exhausted. A non-OK status fails the RPC.
  virtual arrow::Future<std::optional<FlightPayload>> NextAsync() = 0;

  /// \brief Stop the stream: the RPC it was producing for is over.
  ///
  /// Called at most once, possibly from any thread.  A pending
  /// NextAsync()/GetSchemaPayloadAsync() future must be resolved here (or
  /// abandoned knowingly): the transport does not wait for it.  On the
  /// end-of-stream path the status Close() returns is the RPC's final status.
  virtual Status Close() { return Status::OK(); }
};

/// \brief An asynchronous iterator of FlightInfo instances, returned by
/// ListFlightsAsync.
///
/// The transport pulls it one NextAsync() at a time and writes each
/// FlightInfo to the client as it becomes available.
class ARROW_FLIGHT_EXPORT AsyncFlightListing {
 public:
  virtual ~AsyncFlightListing();

  /// \brief The next FlightInfo in the listing.
  /// \return A future completed with the next FlightInfo, or with a nullptr
  /// once the listing is exhausted.  A non-OK status fails the RPC.
  virtual arrow::Future<std::shared_ptr<FlightInfo>> NextAsync() = 0;
};

/// \brief An asynchronous iterator of Result instances, returned by
/// DoActionAsync.
class ARROW_FLIGHT_EXPORT AsyncResultStream {
 public:
  virtual ~AsyncResultStream();

  /// \brief The next Result in the stream.
  /// \return A future completed with the next Result, or with a nullptr once
  /// the stream is exhausted.  A non-OK status fails the RPC.
  virtual arrow::Future<std::shared_ptr<Result>> NextAsync() = 0;
};

/// \brief The reader of one DoExchange: the messages the client sends, pulled
/// one NextAsync() at a time.
///
/// Runs on the thread that calls it; at most one NextAsync() may be
/// outstanding.  Schema and dictionary messages are read through (they are not
/// chunks), a metadata-only message is a chunk with a null data member, and
/// the chunk whose members are both null ends the exchange.  The reader must
/// not be used after the future returned by DoExchangeAsync completes.
class ARROW_FLIGHT_EXPORT AsyncFlightMessageReader {
 public:
  virtual ~AsyncFlightMessageReader();

  /// \brief The descriptor of this exchange.
  virtual const FlightDescriptor& descriptor() const = 0;

  /// \brief The next message of the exchange, decoded.
  virtual arrow::Future<FlightStreamChunk> NextAsync() = 0;
};

/// \brief The writer of one DoExchange: the messages the server sends back.
///
/// Runs on the thread that calls it; await each returned future before
/// writing again.  The writer must not be used after the future returned by
/// DoExchangeAsync completes.
class ARROW_FLIGHT_EXPORT AsyncFlightMessageWriter {
 public:
  virtual ~AsyncFlightMessageWriter();

  /// \brief Start the writer with the schema of the messages it will write.
  virtual arrow::Future<> BeginAsync(std::shared_ptr<Schema> schema) = 0;

  /// \brief Write one record batch.
  virtual arrow::Future<> WriteRecordBatchAsync(const RecordBatch& batch) = 0;

  /// \brief Write one application-metadata-only message.
  virtual arrow::Future<> WriteMetadataAsync(std::shared_ptr<Buffer> app_metadata) = 0;

  /// \brief Write one record batch with application metadata attached.
  virtual arrow::Future<> WriteWithMetadataAsync(
      const RecordBatch& batch, std::shared_ptr<Buffer> app_metadata) = 0;

  /// \brief Finish the write side.
  virtual arrow::Future<> CloseAsync() = 0;
};

/// \brief a Flight server served by the async generic gRPC
/// callback service.
///
/// Derive from this class alone and implement the handlers you serve:
///
/// \code
/// class MyServer : public arrow::flight::AsyncGenericFlightServerBase {
///   arrow::Future<std::shared_ptr<arrow::flight::FlightInfo>> GetFlightInfoAsync(
///       const arrow::flight::ServerCallContext&,
///       const arrow::flight::FlightDescriptor&) override;
///   arrow::Future<std::shared_ptr<arrow::flight::AsyncFlightDataStream>> DoGetAsync(
///       const arrow::flight::ServerCallContext&,
///       const arrow::flight::Ticket&) override;
/// };
/// \endcode
///
/// Every handler returns a future: complete it from your own thread or pool
/// when the answer is ready, and no gRPC callback thread is ever held waiting
/// for one.  Handlers that do run inline should still return promptly: blocking
/// a callback thread starves new connections, because gRPC's thread pool grows
/// at most about one thread per second.
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

  /// \brief Asynchronously retrieve a list of available fields given an
  /// optional opaque criteria.  See FlightServerBase::ListFlights.
  ///
  /// \return A future completed with the listing, or with a non-OK status to
  /// fail the RPC.  A future completed with a nullptr listing serves an empty
  /// listing.  The transport pulls the listing one NextAsync() at a time; the
  /// criteria pointer is only valid for the duration of the call.
  virtual arrow::Future<std::shared_ptr<AsyncFlightListing>> ListFlightsAsync(
      const ServerCallContext& context, const Criteria* criteria);

  /// \brief Asynchronously retrieve the schema and an access plan for the
  /// indicated descriptor.  See FlightServerBase::GetFlightInfo.
  ///
  /// \return A future completed with the info, or with a non-OK status to
  /// fail the RPC.  A nullptr info answers `KeyError("Flight not found")`, as
  /// a synchronous server that returns OK without setting the response does.
  virtual arrow::Future<std::shared_ptr<FlightInfo>> GetFlightInfoAsync(
      const ServerCallContext& context, const FlightDescriptor& request);

  /// \brief Asynchronously retrieve the current status of the target query.
  /// See FlightServerBase::PollFlightInfo.
  virtual arrow::Future<std::shared_ptr<PollInfo>> PollFlightInfoAsync(
      const ServerCallContext& context, const FlightDescriptor& request);

  /// \brief Asynchronously retrieve the schema for the indicated descriptor.
  /// See FlightServerBase::GetSchema.
  virtual arrow::Future<std::shared_ptr<SchemaResult>> GetSchemaAsync(
      const ServerCallContext& context, const FlightDescriptor& request);

  /// \brief Asynchronously execute an action, returning a stream of zero or
  /// more results.  See FlightServerBase::DoAction.
  ///
  /// \return A future completed with the result stream, or with a non-OK
  /// status to fail the RPC.  A future completed with a nullptr stream
  /// answers CANCELLED, as a synchronous server that returns OK without a
  /// stream does.  The transport pulls the stream one NextAsync() at a time.
  virtual arrow::Future<std::shared_ptr<AsyncResultStream>> DoActionAsync(
      const ServerCallContext& context, const Action& action);

  /// \brief Asynchronously retrieve the list of available actions.  See
  /// FlightServerBase::ListActions.
  virtual arrow::Future<std::vector<ActionType>> ListActionsAsync(
      const ServerCallContext& context);

  /// \brief Asynchronously process a bidirectional stream of IPC payloads.
  /// See FlightServerBase::DoExchange.
  ///
  /// \return A future completed when the exchange is over; its status is the
  /// RPC's status.  While it runs, consume the client's messages through the
  /// reader and send the server's through the writer.  Both must not be used
  /// after the future completes.
  virtual arrow::Future<> DoExchangeAsync(
      const ServerCallContext& context, std::shared_ptr<AsyncFlightMessageReader> reader,
      std::shared_ptr<AsyncFlightMessageWriter> writer);

  /// \brief The stream serving one DoGet RPC, possibly later.
  ///
  /// \return Complete the returned future from your own thread or pool, the transport
  /// writes whatever stream it resolves to.
  /// \param[in] `context` is the server call context.
  /// \param[in] `request` is an opaque ticket The default answers UNIMPLEMENTED.
  ///
  /// If the RPC is cancelled while the returned future is pending, the
  /// transport finishes the RPC without waiting for it, and the stream it
  /// eventually resolves to is dropped unconsumed (Close() was already
  /// delivered, before the stream existed).  A producer that must give up
  /// early watches ServerCallContext::is_cancelled().
  virtual arrow::Future<std::shared_ptr<AsyncFlightDataStream>> DoGetAsync(
      const ServerCallContext& context, const Ticket& request);

  /// \brief Create the AsyncFlightDataListener serving one DoPut RPC.
  ///
  /// \param[in] `context` is the server call context.
  /// \return A shared pointer to the AsyncFlightDataListener handling the upload, or
  /// nullptr to refuse the upload.

  virtual std::shared_ptr<AsyncFlightDataListener> CreateDoPutListener(
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
