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

#include "arrow/flight/transport/grpc/grpc_server_async_message_internal.h"

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "arrow/buffer.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"

namespace arrow::flight::transport::grpc::async_internal {
namespace {

/// Serialize FlightData metadata and body for the IPC stream decoder.
arrow::Result<std::shared_ptr<Buffer>> SerializeIpcMessage(
    internal::FlightData& data, const ipc::IpcWriteOptions& options) {
  ARROW_ASSIGN_OR_RAISE(auto message, data.OpenMessage());
  ARROW_ASSIGN_OR_RAISE(auto sink, io::BufferOutputStream::Create());
  int64_t unused_size = 0;
  RETURN_NOT_OK(message->SerializeTo(sink.get(), options, &unused_size));
  return sink->Finish();
}

/// Make an inbound FlightData buffer usable by the IPC decoder.
arrow::Result<std::shared_ptr<internal::FlightData>> PrepareFlightData(
    std::shared_ptr<internal::FlightData> message,
    const std::shared_ptr<MemoryManager>& memory_manager) {
  if (!message) {
    return std::shared_ptr<internal::FlightData>{};
  }
  auto data = std::move(*message);
  if (data.body) {
    ARROW_ASSIGN_OR_RAISE(data.body, Buffer::ViewOrCopy(data.body, memory_manager));
  }
  return std::make_shared<internal::FlightData>(std::move(data));
}

/// Decodes inbound FlightData frames into the public async reader interface.
class NativeAsyncFlightMessageReader final
    : public AsyncFlightMessageReader,
      public std::enable_shared_from_this<NativeAsyncFlightMessageReader> {
 public:
  using ReadFn = AsyncReadFn;
  enum class ActiveOperation { kNone, kSchema, kNext };

  NativeAsyncFlightMessageReader(FlightDescriptor descriptor, ReadFn read_fn)
      : descriptor_(std::move(descriptor)),
        read_fn_(std::move(read_fn)),
        listener_(std::make_shared<ipc::CollectListener>()),
        decoder_(listener_) {}

  /// Return the descriptor captured from the first FlightData frame.
  const FlightDescriptor& descriptor() const override { return descriptor_; }

  /// Decode and return the stream schema, serializing concurrent reads.
  Future<std::shared_ptr<Schema>> GetSchema() override {
    if (listener_->schema()) {
      return Future<std::shared_ptr<Schema>>::MakeFinished(listener_->schema());
    }
    int64_t token = 0;
    const auto status = StartOperation(ActiveOperation::kSchema, &token);
    if (!status.ok()) {
      return Future<std::shared_ptr<Schema>>::MakeFinished(std::move(status));
    }
    auto out = Future<std::shared_ptr<Schema>>::Make();
    shared_from_this()->PumpSchema(out, token);
    return out;
  }

  /// Decode and return the next logical record-batch or metadata chunk.
  Future<FlightStreamChunk> Next() override {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!decoded_chunks_.empty()) {
        auto chunk = std::move(decoded_chunks_.front());
        decoded_chunks_.pop_front();
        return Future<FlightStreamChunk>::MakeFinished(std::move(chunk));
      }
      if (finished_) {
        return Future<FlightStreamChunk>::MakeFinished(FlightStreamChunk{});
      }
    }
    int64_t token = 0;
    auto status = StartOperation(ActiveOperation::kNext, &token);
    if (!status.ok()) {
      return Future<FlightStreamChunk>::MakeFinished(std::move(status));
    }
    auto out = Future<FlightStreamChunk>::Make();
    shared_from_this()->PumpNext(out, token);
    return out;
  }

  /// Return decoder statistics accumulated so far.
  ipc::ReadStats stats() const override { return decoder_.stats(); }

  /// Return a future that completes after the active reader operation.
  Future<> WhenIdle() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return idle_;
  }

 private:
  /// Request the next prepared frame from the composed source.
  Future<std::shared_ptr<internal::FlightData>> ReadDataAsync() {
    return read_fn_();
  }

  /// Feed one IPC frame to the decoder and queue any resulting record batch.
  Status ConsumeDataMessage(internal::FlightData& data) {
    ARROW_ASSIGN_OR_RAISE(auto buffer,
                          SerializeIpcMessage(data, ipc::IpcWriteOptions::Defaults()));
    const int64_t previous_batches = listener_->num_record_batches();
    RETURN_NOT_OK(decoder_.Consume(std::move(buffer)));
    const int64_t new_batches = listener_->num_record_batches();
    if (new_batches > previous_batches) {
      decoded_chunks_.emplace_back(listener_->PopRecordBatch(), std::move(data.app_metadata));
    }
    return Status::OK();
  }

  /// Start one serialized reader operation and return its cancellation token.
  Status StartOperation(ActiveOperation operation, int64_t* token) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_operation_ != ActiveOperation::kNone) {
      return Status::Invalid("Concurrent async reads are not supported");
    }
    active_operation_ = operation;
    idle_ = Future<>::Make();
    *token = ++active_token_;
    return Status::OK();
  }

  /// Check whether a callback still belongs to the current read operation.
  bool IsCurrentOperation(int64_t token, ActiveOperation operation) {
    std::lock_guard<std::mutex> guard(mutex_);
    return active_token_ == token && active_operation_ == operation;
  }

  /// Complete a serialized reader operation and its idle notification.
  template <typename T>
  void FinishOperationResult(Future<T> out, ::arrow::Result<T> result, int64_t token,
                             ActiveOperation operation) {
    Future<> idle;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (active_token_ != token || active_operation_ != operation) {
        return;
      }
      active_operation_ = ActiveOperation::kNone;
      idle = idle_;
    }
    out.MarkFinished(std::move(result));
    idle.MarkFinished();
  }

  template <typename T>
  void FinishOperation(Future<T> out, T value, int64_t token, ActiveOperation operation) {
    FinishOperationResult(std::move(out), ::arrow::Result<T>(std::move(value)), token,
                          operation);
  }

  template <typename T>
  void FinishOperation(Future<T> out, Status status, int64_t token,
                       ActiveOperation operation) {
    FinishOperationResult(std::move(out), ::arrow::Result<T>(std::move(status)), token,
                          operation);
  }

  /// Complete the schema operation from an inbound FlightData frame.
  void HandleSchemaResult(
      Future<std::shared_ptr<Schema>> out, int64_t token,
      const ::arrow::Result<std::shared_ptr<internal::FlightData>>& result) {
    if (!IsCurrentOperation(token, ActiveOperation::kSchema)) {
      return;
    }
    if (!result.ok()) {
      FinishOperation(out, result.status(), token, ActiveOperation::kSchema);
      return;
    }
    auto maybe_data = result.ValueUnsafe();
    if (!maybe_data) {
      FinishOperation(out, Status::IOError("Client never sent a data message"), token,
                      ActiveOperation::kSchema);
      return;
    }
    if (!maybe_data->metadata) {
      PumpSchema(out, token);
      return;
    }
    const auto status = ConsumeDataMessage(*maybe_data);
    if (!status.ok()) {
      FinishOperation(out, status, token, ActiveOperation::kSchema);
      return;
    }
    PumpSchema(out, token);
  }

  /// Read frames until the decoder produces a schema or the stream fails.
  void PumpSchema(Future<std::shared_ptr<Schema>> out, int64_t token) {
    if (listener_->schema()) {
      FinishOperation(out, listener_->schema(), token, ActiveOperation::kSchema);
      return;
    }
    ReadDataAsync().AddCallback(
        [self = shared_from_this(), out,
         token](const ::arrow::Result<std::shared_ptr<internal::FlightData>>&
                    result) {
          self->HandleSchemaResult(out, token, result);
        });
  }

  /// Complete the next operation from an inbound FlightData frame.
  void HandleNextResult(
      Future<FlightStreamChunk> out, int64_t token,
      const ::arrow::Result<std::shared_ptr<internal::FlightData>>& result) {
    if (!IsCurrentOperation(token, ActiveOperation::kNext)) {
      return;
    }
    if (!result.ok()) {
      FinishOperation(out, result.status(), token, ActiveOperation::kNext);
      return;
    }
    auto maybe_data = result.ValueUnsafe();
    if (!maybe_data) {
      {
        std::lock_guard<std::mutex> guard(mutex_);
        finished_ = true;
      }
      FinishOperation(out, FlightStreamChunk{}, token, ActiveOperation::kNext);
      return;
    }
    if (!maybe_data->metadata) {
      if (!maybe_data->app_metadata) {
        PumpNext(out, token);
        return;
      }
      FlightStreamChunk chunk;
      chunk.app_metadata = std::move(maybe_data->app_metadata);
      FinishOperation(out, std::move(chunk), token, ActiveOperation::kNext);
      return;
    }
    const auto status = ConsumeDataMessage(*maybe_data);
    if (!status.ok()) {
      FinishOperation(out, status, token, ActiveOperation::kNext);
      return;
    }
    PumpNext(out, token);
  }

  /// Read frames until a user-visible chunk or end-of-stream is available.
  void PumpNext(Future<FlightStreamChunk> out, int64_t token) {
    std::optional<FlightStreamChunk> ready_chunk;
    bool stream_finished = false;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!decoded_chunks_.empty()) {
        ready_chunk = std::move(decoded_chunks_.front());
        decoded_chunks_.pop_front();
      } else if (finished_) {
        stream_finished = true;
      }
    }
    if (ready_chunk.has_value()) {
      FinishOperation(out, std::move(*ready_chunk), token, ActiveOperation::kNext);
      return;
    }
    if (stream_finished) {
      FinishOperation(out, FlightStreamChunk{}, token, ActiveOperation::kNext);
      return;
    }
    ReadDataAsync().AddCallback(
        [self = shared_from_this(), out,
         token](const ::arrow::Result<std::shared_ptr<internal::FlightData>>&
            result) {
          self->HandleNextResult(out, token, result);
        });
  }

  /// Descriptor carried by the first inbound FlightData frame.
  FlightDescriptor descriptor_;
  /// Composed source yielding prepared frames: the retained first frame,
  /// then prepared frames from the underlying gRPC reads.
  ReadFn read_fn_;
  /// IPC listener collecting decoded schemas and record batches.
  std::shared_ptr<ipc::CollectListener> listener_;
  /// IPC stream decoder consuming serialized FlightData messages.
  ipc::StreamDecoder decoder_;
  /// Protects reader operation state and decoded chunk storage.
  mutable std::mutex mutex_;
  /// Decoded chunks waiting to be returned by Next().
  std::deque<FlightStreamChunk> decoded_chunks_;
  /// Whether the inbound stream has reached its end marker.
  bool finished_ = false;
  /// Reader operation currently consuming inbound frames.
  ActiveOperation active_operation_ = ActiveOperation::kNone;
  /// Monotonic identity used to reject callbacks from older operations.
  int64_t active_token_ = 0;
  /// Completes when the current serialized reader operation becomes idle.
  Future<> idle_ = Future<>::MakeFinished();
};

}  // namespace

Future<AsyncMessageReaderParts> MakeNativeAsyncMessageReader(
    AsyncReadFn read_fn, std::shared_ptr<MemoryManager> memory_manager) {
  auto first_message = read_fn();
  return first_message.Then(
      [read_fn = std::move(read_fn), memory_manager = std::move(memory_manager)](
          std::shared_ptr<internal::FlightData> message) mutable
          -> ::arrow::Result<AsyncMessageReaderParts> {
        if (!message) {
          return Status::IOError("Stream finished before first message sent");
        }
        auto data = std::move(*message);
        if (data.body) {
          ARROW_ASSIGN_OR_RAISE(data.body, Buffer::ViewOrCopy(data.body, memory_manager));
        }
        if (!data.descriptor) {
          return Status::IOError("Descriptor missing on first message");
        }
        auto descriptor = *data.descriptor;
        // Compose the reader's frame source: yield the retained first frame,
        // then prepare subsequent frames with the memory manager on read.
        auto first_frame =
            std::make_shared<std::optional<internal::FlightData>>(std::move(data));
        AsyncReadFn composed_read_fn =
            [read_fn = std::move(read_fn), memory_manager,
             first_frame = std::move(first_frame)]() mutable {
              if (first_frame->has_value()) {
                auto frame = std::make_shared<internal::FlightData>(
                    std::move(**first_frame));
                first_frame->reset();
                return Future<std::shared_ptr<internal::FlightData>>::MakeFinished(
                    std::move(frame));
              }
              return read_fn().Then(
                  [memory_manager](
                      std::shared_ptr<internal::FlightData> message) {
                    return PrepareFlightData(std::move(message), memory_manager);
                  });
            };
        auto impl = std::make_shared<NativeAsyncFlightMessageReader>(
            std::move(descriptor), std::move(composed_read_fn));
        return AsyncMessageReaderParts{.reader = impl,
                                       .when_idle = [impl] { return impl->WhenIdle(); }};
      });
}

}  // namespace arrow::flight::transport::grpc::async_internal