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

#include <functional>
#include <memory>
#include <mutex>

#include "arrow/flight/types.h"
#include "arrow/flight/visibility.h"
#include "arrow/ipc/options.h"
#include "arrow/ipc/reader.h"
#include "arrow/status.h"

namespace arrow::flight {

namespace internal {
struct FlightData;
}  // namespace internal

class AsyncFlightDataListener;

namespace internal {

/// \brief The per-RPC handle a transport gives a FlightDataListener so the
/// application can act on the RPC while it is in flight.
///
/// The transport owns this; the listener only holds a reference for as long as
/// the RPC is live, and asks for it under a lock the transport takes to clear
/// it (see FlightDataListener::LockTransport).  That is the same lifetime rule
/// the client-side AsyncListenerBase uses for its AsyncRpc state.
class ARROW_FLIGHT_EXPORT FlightDataListenerTransport {
 public:
  virtual ~FlightDataListenerTransport();

  /// \brief Finish the upload's RPC with `status`, from any thread.
  ///
  /// The transport must make this idempotent and safe to call after the RPC
  /// finished (as a no-op), so a racing Cancel() cannot use-after-free.
  virtual void CancelUpload(Status status) = 0;

  /// \brief Install `transport` on `listener` for the life of the RPC.
  /// Transport-only.  `transport` is the transport's own RPC state (for gRPC's
  /// callback API, the reactor itself); it stays valid until Clear(), and the
  /// listener only reaches it under the lock Clear() takes.
  static void Install(const std::shared_ptr<AsyncFlightDataListener>& listener,
                      FlightDataListenerTransport* transport);

  /// \brief Clear the installed transport.  Transport-only; called as the RPC
  /// finishes, before the transport object dies, so a concurrent Cancel() can
  /// never reach a dead RPC.
  static void Clear(const std::shared_ptr<AsyncFlightDataListener>& listener);

  /// \brief Report the upload's terminal status to the listener, once.
  /// Transport-only.  Later calls are dropped, so every ending can report
  /// without coordinating with the others.
  static Status ReportFinish(const std::shared_ptr<AsyncFlightDataListener>& listener,
                             Status status);
};

}  // namespace internal

/// \brief A general listener class to receive events from FlightMessageDecoder
///
/// User must implement callback methods for interested events.
class ARROW_FLIGHT_EXPORT AsyncFlightDataListener : public ipc::Listener {
 public:
  AsyncFlightDataListener();
  ~AsyncFlightDataListener() override;

  /// \brief Called for each decoded FlightStreamChunk.
  ///
  /// chunk.data is the decoded RecordBatch, or nullptr for metadata-only
  /// messages.
  virtual Future<> OnNext(FlightStreamChunk chunk) = 0;

  /// \brief Called when the descriptor of an upload is decoded.
  ///
  /// Fired before any schema or data of that upload, so the listener knows
  /// which upload it is being handed. A non-OK status rejects the upload.
  virtual Future<> OnDescriptor(const FlightDescriptor& descriptor) { return Future<>::MakeFinished(); }

  /// \brief Called once, when the upload ends, whichever way it ends.
  ///
  /// The counterpart of OnNext: after this, no other callback arrives.  Runs
  /// on a transport thread, so it must not block.  `status` is OK for an upload
  /// the client ended normally, and the failure otherwise (the client went
  /// away, the transport failed, or the upload was rejected).  This is where a
  /// consumer commits or discards what it accumulated; the default does
  /// nothing, since a stream that only counts chunks needs no completion.
  virtual Future<> OnFinish(Status status) { return Future<>::MakeFinished(); }

  /// \brief Cancel the upload with `status`, from any thread.
  ///
  /// Finishes the RPC with that status without waiting for the client to end
  /// the upload: the client's pending write or Close() reports it.  Use for a
  /// server-side rejection discovered mid-upload (quota, bad batch, upstream
  /// error).  Safe to call from any thread at any time: once the upload is no
  /// longer in flight this reports Invalid instead of reaching a finished RPC.
  /// \return OK when the cancel was handed to the transport, or Invalid when
  /// there is no upload in flight.
  Future<> Cancel(Status status);

 private:
  /// The transport installs and clears the state through
  /// FlightDataListenerTransport::Install/Clear, both under this lock.
  friend class internal::FlightDataListenerTransport;

  /// Whether OnFinish was already reported.  Atomic (not guarded by
  /// transport_mutex_) on purpose: Cancel() holds that lock across its call
  /// into the transport, and the transport reports the ending from inside it.
  std::atomic<bool> finished_{false};

  mutable std::mutex transport_mutex_;
  /// Not owned: the transport's own RPC state, valid until Clear().
  internal::FlightDataListenerTransport* transport_ = nullptr;
};

/// \brief Push style stream decoder that turns raw arrow Buffers into
/// FlightStreamChunks.
///
/// This class decodes Apache Arrow Flight data format from arrow::Buffer
/// and fires events on the provided FlightDataListener.
class ARROW_FLIGHT_EXPORT AsyncFlightMessageDecoder {
 public:
  explicit AsyncFlightMessageDecoder(
      std::shared_ptr<AsyncFlightDataListener> listener,
      ipc::IpcReadOptions options = ipc::IpcReadOptions::Defaults());
  ~AsyncFlightMessageDecoder();

  /// \brief Decode one FlightData message directly from a buffer.
  ///
  /// Fires listener->OnSchemaDecoded() on the first message containing
  /// a schema, listener->OnNext() for each subsequent record batch,
  /// metadata-only message or dictionary batch.
  ///
  /// \param[in] buffer a raw buffer directly from the transport. Example
  /// the arrow::Buffer extracted from the grpc::ByteBuffer from the gRPC transport.
  /// \return Status
  Future<> Consume(std::shared_ptr<Buffer> buffer);

  /// \brief Decode one FlightData message already deserialized by the transport.
  ///
  /// Same as Consume(Buffer) but avoids a serialization round-trip for transports
  /// that hand out an internal::FlightData directly (the gRPC client reads FlightData
  /// straight into internal::FlightData).  This overload is internal to Arrow;
  /// applications should use Consume(Buffer).
  Future<> Consume(internal::FlightData data);

  /// \brief The decoded schema.
  ///
  /// Available after the first Consume() call that contains a schema message.
  /// Returns nullptr if no schema has been decoded yet.
  std::shared_ptr<Schema> schema() const;

 private:
  class AsyncFlightMessageDecoderImpl;
  std::unique_ptr<AsyncFlightMessageDecoderImpl> impl_;
};

}  // namespace arrow::flight
