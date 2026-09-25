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

#include "arrow/flight/types.h"
#include "arrow/flight/visibility.h"
#include "arrow/ipc/options.h"
#include "arrow/ipc/reader.h"
#include "arrow/status.h"

namespace arrow::flight {

namespace internal {
struct FlightData;
}  // namespace internal

/// \brief A general listener class to receive events from FlightMessageDecoder
///
/// User must implement callback methods for interested events.
class ARROW_FLIGHT_EXPORT FlightDataListener : public ipc::Listener {
 public:
  /// \brief Called for each decoded FlightStreamChunk.
  ///
  /// chunk.data is the decoded RecordBatch, or nullptr for metadata-only
  /// messages.
  virtual Status OnNext(FlightStreamChunk chunk) = 0;

  /// \brief Called when the descriptor of an upload is decoded.
  ///
  /// Fired before any schema or data of that upload, so the listener knows
  /// which upload it is being handed. A non-OK status rejects the upload.
  virtual Status OnDescriptor(const FlightDescriptor& descriptor) { return Status::OK(); }
};

/// \brief Creates the FlightDataListener that serves one upload.
///
/// The transport calls this once per DoPut RPC. Return nullptr to refuse the
/// upload.
using FlightDataListenerFactory = std::function<std::shared_ptr<FlightDataListener>()>;

/// \brief Push style stream decoder that turns raw arrow Buffers into
/// FlightStreamChunks.
///
/// This class decodes Apache Arrow Flight data format from arrow::Buffer
/// and fires events on the provided FlightDataListener.
class ARROW_FLIGHT_EXPORT FlightMessageDecoder {
 public:
  explicit FlightMessageDecoder(
      std::shared_ptr<FlightDataListener> listener,
      ipc::IpcReadOptions options = ipc::IpcReadOptions::Defaults());
  ~FlightMessageDecoder();

  /// \brief Decode one FlightData message directly from a buffer.
  ///
  /// Fires listener->OnSchemaDecoded() on the first message containing
  /// a schema, listener->OnNext() for each subsequent record batch,
  /// metadata-only message or dictionary batch.
  ///
  /// \param[in] buffer a raw buffer directly from the transport. Example
  /// the arrow::Buffer extracted from the grpc::ByteBuffer from the gRPC transport.
  /// \return Status
  Status Consume(std::shared_ptr<Buffer> buffer);

  /// \brief Decode one FlightData message already deserialized by the transport.
  ///
  /// Same as Consume(Buffer) but avoids a serialization round-trip for transports
  /// that hand out an internal::FlightData directly (the gRPC client reads FlightData
  /// straight into internal::FlightData).  This overload is internal to Arrow;
  /// applications should use Consume(Buffer).
  Status Consume(internal::FlightData data);

  /// \brief The decoded schema.
  ///
  /// Available after the first Consume() call that contains a schema message.
  /// Returns nullptr if no schema has been decoded yet.
  std::shared_ptr<Schema> schema() const;

 private:
  class FlightMessageDecoderImpl;
  std::unique_ptr<FlightMessageDecoderImpl> impl_;
};

}  // namespace arrow::flight
