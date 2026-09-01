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

#include "arrow/flight/transport/grpc/grpc_server_async_internal.h"

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

#include "arrow/array/array_base.h"
#include "arrow/array/data.h"
#include "arrow/buffer.h"
#include "arrow/compare.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/types.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/dictionary.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"

namespace arrow::flight::transport::grpc::async_internal {
namespace {

/// Return whether an array contains a nested dictionary, which cannot use deltas.
bool HasNestedDictionary(const ArrayData& data) {
  if (data.type->id() == Type::DICTIONARY) {
    return true;
  }
  for (const auto& child : data.child_data) {
    if (HasNestedDictionary(*child)) {
      return true;
    }
  }
  return false;
}

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

  NativeAsyncFlightMessageReader(FlightDescriptor descriptor,
                                 internal::FlightData first_message, ReadFn read_fn,
                                 std::shared_ptr<MemoryManager> memory_manager)
      : descriptor_(std::move(descriptor)),
        pending_message_(std::move(first_message)),
        read_fn_(std::move(read_fn)),
        memory_manager_(std::move(memory_manager)),
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
    auto status = StartOperation(ActiveOperation::kSchema, &token);
    if (!status.ok()) {
      return Future<std::shared_ptr<Schema>>::MakeFinished(std::move(status));
    }
    auto out = Future<std::shared_ptr<Schema>>::Make();
    shared_from_this()->PumpSchema(out, token);
    return out;
  }

  /// Decode and return the next logical record-batch or metadata chunk.
  Future<FlightStreamChunk> Next() override {
    int64_t token = 0;
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
  /// Consume the retained first frame or request the next frame from gRPC.
  Future<std::shared_ptr<internal::FlightData>> ReadDataAsync() {
    std::shared_ptr<internal::FlightData> pending;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (pending_message_) {
        pending = std::make_shared<internal::FlightData>(std::move(*pending_message_));
        pending_message_.reset();
      }
    }
    if (pending) {
      return Future<std::shared_ptr<internal::FlightData>>::MakeFinished(
          std::move(pending));
    }
    return read_fn_().Then([memory_manager = memory_manager_](
                               std::shared_ptr<internal::FlightData> message) {
      return PrepareFlightData(std::move(message), memory_manager);
    });
  }

  /// Feed one IPC frame to the decoder and queue any resulting record batch.
  Status ConsumeDataMessage(internal::FlightData& data) {
    ARROW_ASSIGN_OR_RAISE(auto buffer,
                          SerializeIpcMessage(data, ipc::IpcWriteOptions::Defaults()));
    const auto previous_batches = listener_->num_record_batches();
    RETURN_NOT_OK(decoder_.Consume(std::move(buffer)));
    const auto new_batches = listener_->num_record_batches();
    if (new_batches > previous_batches) {
      auto batch = listener_->PopRecordBatch();
      FlightStreamChunk chunk;
      chunk.data = std::move(batch);
      chunk.app_metadata = std::move(data.app_metadata);
      decoded_chunks_.push_back(std::move(chunk));
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

  /// Read frames until the decoder produces a schema or the stream fails.
  void PumpSchema(Future<std::shared_ptr<Schema>> out, int64_t token) {
    if (listener_->schema()) {
      FinishOperation(out, listener_->schema(), token, ActiveOperation::kSchema);
      return;
    }
    ReadDataAsync().AddCallback(
        [self = shared_from_this(), out,
         token](const ::arrow::Result<std::shared_ptr<internal::FlightData>>&
                    result) mutable {
          if (!self->IsCurrentOperation(token, ActiveOperation::kSchema)) {
            return;
          }
          if (!result.ok()) {
            self->FinishOperation(out, result.status(), token, ActiveOperation::kSchema);
            return;
          }
          auto maybe_data = result.ValueUnsafe();
          if (!maybe_data) {
            self->FinishOperation(out,
                                  Status::IOError("Client never sent a data message"),
                                  token, ActiveOperation::kSchema);
            return;
          }
          if (!maybe_data->metadata) {
            // Descriptor-only or metadata-only FlightData frames do not advance the
            // IPC decoder toward a schema, so continue reading until an IPC message
            // arrives or the stream ends.
            self->PumpSchema(out, token);
            return;
          }
          auto st = self->ConsumeDataMessage(*maybe_data);
          if (!st.ok()) {
            self->FinishOperation(out, st, token, ActiveOperation::kSchema);
            return;
          }
          self->PumpSchema(out, token);
        });
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
                    result) mutable {
          if (!self->IsCurrentOperation(token, ActiveOperation::kNext)) {
            return;
          }
          if (!result.ok()) {
            self->FinishOperation(out, result.status(), token, ActiveOperation::kNext);
            return;
          }
          auto maybe_data = result.ValueUnsafe();
          if (!maybe_data) {
            {
              std::lock_guard<std::mutex> guard(self->mutex_);
              self->finished_ = true;
            }
            self->FinishOperation(out, FlightStreamChunk{}, token,
                                  ActiveOperation::kNext);
            return;
          }
          if (!maybe_data->metadata) {
            if (!maybe_data->app_metadata) {
              self->PumpNext(out, token);
              return;
            }
            FlightStreamChunk chunk;
            chunk.app_metadata = std::move(maybe_data->app_metadata);
            self->FinishOperation(out, std::move(chunk), token, ActiveOperation::kNext);
            return;
          }
          auto st = self->ConsumeDataMessage(*maybe_data);
          if (!st.ok()) {
            self->FinishOperation(out, st, token, ActiveOperation::kNext);
            return;
          }
          self->PumpNext(out, token);
        });
  }

  FlightDescriptor descriptor_;
  std::optional<internal::FlightData> pending_message_;
  ReadFn read_fn_;
  std::shared_ptr<MemoryManager> memory_manager_;
  std::shared_ptr<ipc::CollectListener> listener_;
  ipc::StreamDecoder decoder_;
  mutable std::mutex mutex_;
  std::deque<FlightStreamChunk> decoded_chunks_;
  bool finished_ = false;
  ActiveOperation active_operation_ = ActiveOperation::kNone;
  int64_t active_token_ = 0;
  Future<> idle_ = Future<>::MakeFinished();
};

/// Owns the shared reader state while exposing the unique_ptr public interface.
class SharedAsyncFlightMessageReader final : public AsyncFlightMessageReader {
 public:
  explicit SharedAsyncFlightMessageReader(
      std::shared_ptr<NativeAsyncFlightMessageReader> impl)
      : impl_(std::move(impl)) {}

  const FlightDescriptor& descriptor() const override { return impl_->descriptor(); }
  Future<std::shared_ptr<Schema>> GetSchema() override { return impl_->GetSchema(); }
  Future<FlightStreamChunk> Next() override { return impl_->Next(); }
  ipc::ReadStats stats() const override { return impl_->stats(); }

 private:
  std::shared_ptr<NativeAsyncFlightMessageReader> impl_;
};

/// Converts metadata buffers into PutResult protobuf writes.
class NativeAsyncFlightMetadataWriter final : public AsyncFlightMetadataWriter {
 public:
  using WriteFn = std::function<Future<bool>(pb::PutResult)>;

  explicit NativeAsyncFlightMetadataWriter(WriteFn write_fn)
      : write_fn_(std::move(write_fn)) {}

  /// Send one metadata frame and translate a failed gRPC write to an Arrow error.
  Future<> WriteMetadata(const Buffer& app_metadata) override {
    pb::PutResult result;
    result.set_app_metadata(app_metadata.data(), app_metadata.size());
    return write_fn_(std::move(result)).Then([](bool ok) -> Status {
      if (!ok) {
        return Status::IOError("Unknown error writing metadata.");
      }
      return Status::OK();
    });
  }

 private:
  WriteFn write_fn_;
};

/// Serializes Arrow schemas, batches, dictionaries, and metadata to FlightData.
class NativeAsyncFlightMessageWriter final
    : public AsyncFlightMessageWriter,
      public std::enable_shared_from_this<NativeAsyncFlightMessageWriter> {
 public:
  using WriteFn = std::function<Future<bool>(FlightPayload)>;

  explicit NativeAsyncFlightMessageWriter(WriteFn write_fn)
      : write_fn_(std::move(write_fn)) {}

  /// Serialize and write the schema, initializing dictionary tracking.
  Future<> Begin(const std::shared_ptr<Schema>& schema,
                 const ipc::IpcWriteOptions& options) override {
    std::vector<FlightPayload> payloads;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (begun_) {
        return Future<>::MakeFinished(
            Status::Invalid("This writer has already been started."));
      }
      if (closed_) {
        return Future<>::MakeFinished(Status::Invalid("This writer is already closed"));
      }
      if (failed_) {
        return Future<>::MakeFinished(failure_status_);
      }
      if (!schema) {
        return Future<>::MakeFinished(Status::Invalid("Schema cannot be null"));
      }
      options_ = options;
      schema_ = schema;
      mapper_ = std::make_unique<ipc::DictionaryFieldMapper>(*schema);
      FlightPayload payload;
      RETURN_NOT_OK(
          ipc::GetSchemaPayload(*schema_, options_, *mapper_, &payload.ipc_message));
      begun_ = true;
      payloads.push_back(std::move(payload));
    }
    return WritePayloads(std::move(payloads));
  }

  /// Write a record batch without application metadata.
  Future<> WriteRecordBatch(const RecordBatch& batch) override {
    return WriteWithMetadata(batch, nullptr);
  }

  /// Write an application-metadata-only FlightData frame.
  Future<> WriteMetadata(std::shared_ptr<Buffer> app_metadata) override {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      RETURN_NOT_OK(CheckWritableLocked());
    }
    FlightPayload payload;
    payload.app_metadata = std::move(app_metadata);
    return WritePayloads({std::move(payload)});
  }

  /// Write dictionary updates, a record batch, and optional metadata in order.
  Future<> WriteWithMetadata(const RecordBatch& batch,
                             std::shared_ptr<Buffer> app_metadata) override {
    std::vector<FlightPayload> payloads;
    std::vector<DictionaryUpdate> dictionary_updates;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      RETURN_NOT_OK(CheckStartedLocked());
      RETURN_NOT_OK(ReserveWriteLocked());
      auto status = BuildDictionaryPayloadsLocked(batch, &payloads, &dictionary_updates);
      if (!status.ok()) {
        write_in_flight_ = false;
        return Future<>::MakeFinished(std::move(status));
      }
      FlightPayload batch_payload;
      status = ipc::GetRecordBatchPayload(batch, options_, &batch_payload.ipc_message);
      if (!status.ok()) {
        write_in_flight_ = false;
        return Future<>::MakeFinished(std::move(status));
      }
      batch_payload.app_metadata = std::move(app_metadata);
      payloads.push_back(std::move(batch_payload));
      ++stats_.num_record_batches;
      stats_.total_raw_body_size += payloads.back().ipc_message.raw_body_length;
      stats_.total_serialized_body_size += payloads.back().ipc_message.body_length;
      CommitDictionaryUpdatesLocked(std::move(dictionary_updates));
    }
    return WritePayloads(std::move(payloads), /*reserved=*/true);
  }

  /// Prevent future writes without cancelling an active write.
  Future<> Close() override {
    std::lock_guard<std::mutex> guard(mutex_);
    closed_ = true;
    return Future<>::MakeFinished();
  }

  /// Return writer statistics protected by the writer mutex.
  ipc::WriteStats stats() const override {
    std::lock_guard<std::mutex> guard(mutex_);
    return stats_;
  }

 private:
  /// Validate that the writer can accept another operation while mutex_ is held.
  Status CheckWritableLocked() const {
    if (failed_) {
      return failure_status_;
    }
    if (closed_) {
      return Status::Invalid("This writer is already closed");
    }
    return Status::OK();
  }

  /// Validate that batch writes are legal while mutex_ is held.
  Status CheckStartedLocked() const {
    RETURN_NOT_OK(CheckWritableLocked());
    if (!begun_) {
      return Status::Invalid("This writer is not started. Call Begin() with a schema");
    }
    return Status::OK();
  }

  /// Reserve the sole outbound write before mutating dictionary state.
  Status ReserveWriteLocked() {
    if (write_in_flight_) {
      return Status::Invalid("Concurrent async writes are not supported");
    }
    write_in_flight_ = true;
    return Status::OK();
  }

  struct DictionaryUpdate {
    int64_t id;
    std::shared_ptr<Array> dictionary;
    bool is_delta;
    bool had_previous;
  };

  /// Collect and encode required dictionary replacements or deltas under mutex_.
  Status BuildDictionaryPayloadsLocked(const RecordBatch& batch,
                                       std::vector<FlightPayload>* payloads,
                                       std::vector<DictionaryUpdate>* updates) {
    ARROW_ASSIGN_OR_RAISE(const auto dictionaries,
                          ipc::CollectDictionaries(batch, *mapper_));
    const auto equal_options = EqualOptions().nans_equal(true);

    for (const auto& pair : dictionaries) {
      const int64_t dictionary_id = pair.first;
      const auto& dictionary = pair.second;
      auto* last_dictionary = &last_dictionaries_[dictionary_id];
      const bool dictionary_exists = (*last_dictionary != nullptr);
      int64_t delta_start = 0;
      if (dictionary_exists) {
        if ((*last_dictionary)->data() == dictionary->data()) {
          continue;
        }
        const int64_t last_length = (*last_dictionary)->length();
        const int64_t new_length = dictionary->length();
        if (new_length == last_length &&
            ((*last_dictionary)->Equals(dictionary, equal_options))) {
          continue;
        }
        if (new_length > last_length && options_.emit_dictionary_deltas &&
            !HasNestedDictionary(*dictionary->data()) &&
            ((*last_dictionary)
                 ->RangeEquals(dictionary, 0, last_length, 0, equal_options))) {
          delta_start = last_length;
        }
      }

      FlightPayload payload;
      if (delta_start) {
        RETURN_NOT_OK(ipc::GetDictionaryPayload(dictionary_id, /*is_delta=*/true,
                                                dictionary->Slice(delta_start), options_,
                                                &payload.ipc_message));
      } else {
        RETURN_NOT_OK(ipc::GetDictionaryPayload(dictionary_id, dictionary, options_,
                                                &payload.ipc_message));
      }
      payloads->push_back(std::move(payload));
      updates->push_back(
          {dictionary_id, dictionary, delta_start != 0, dictionary_exists});
    }
    return Status::OK();
  }

  /// Commit dictionary state after the complete batch payload is serialized.
  void CommitDictionaryUpdatesLocked(std::vector<DictionaryUpdate> updates) {
    for (auto& update : updates) {
      last_dictionaries_[update.id] = std::move(update.dictionary);
      ++stats_.num_dictionary_batches;
      if (update.had_previous) {
        if (update.is_delta) {
          ++stats_.num_dictionary_deltas;
        } else {
          ++stats_.num_replaced_dictionaries;
        }
      }
    }
  }

  /// Serialize a payload sequence through one gRPC write at a time.
  Future<> WritePayloads(std::vector<FlightPayload> payloads, bool reserved = false) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!reserved) {
        RETURN_NOT_OK(ReserveWriteLocked());
      }
      stats_.num_messages += static_cast<int64_t>(payloads.size());
    }
    auto out = Future<>::Make();
    auto state = std::make_shared<WriteState>();
    state->payloads = std::move(payloads);
    WritePayloadAt(state, 0, out);
    return out;
  }

  struct WriteState {
    std::vector<FlightPayload> payloads;
  };

  /// Issue the next payload and retain writer state until its callback runs.
  void WritePayloadAt(const std::shared_ptr<WriteState>& state, size_t index,
                      Future<> out) {
    if (index >= state->payloads.size()) {
      FinishWrite();
      out.MarkFinished();
      return;
    }
    write_fn_(std::move(state->payloads[index]))
        .AddCallback([self = shared_from_this(), state, index,
                      out](const ::arrow::Result<bool>& result) mutable {
          if (!result.ok()) {
            self->FinishWrite(result.status());
            out.MarkFinished(result.status());
            return;
          }
          if (!*result) {
            auto status = MakeFlightError(
                FlightStatusCode::Internal,
                "Could not write record batch to stream (client disconnect?)");
            self->FinishWrite(status);
            out.MarkFinished(std::move(status));
            return;
          }
          self->WritePayloadAt(state, index + 1, out);
        });
  }

  /// Clear the write-in-flight flag and remember a terminal write error.
  void FinishWrite(const Status& status = Status::OK()) {
    std::lock_guard<std::mutex> guard(mutex_);
    write_in_flight_ = false;
    if (!status.ok()) {
      failed_ = true;
      failure_status_ = status;
    }
  }

  WriteFn write_fn_;
  mutable std::mutex mutex_;
  std::shared_ptr<Schema> schema_;
  std::unique_ptr<ipc::DictionaryFieldMapper> mapper_;
  std::unordered_map<int64_t, std::shared_ptr<Array>> last_dictionaries_;
  ipc::IpcWriteOptions options_ = ipc::IpcWriteOptions::Defaults();
  ipc::WriteStats stats_;
  bool begun_ = false;
  bool closed_ = false;
  bool write_in_flight_ = false;
  bool failed_ = false;
  Status failure_status_;
};

/// Retains shared writer state behind the unique_ptr public writer interface.
class SharedAsyncFlightMessageWriter final : public AsyncFlightMessageWriter {
 public:
  explicit SharedAsyncFlightMessageWriter(
      std::shared_ptr<NativeAsyncFlightMessageWriter> impl)
      : impl_(std::move(impl)) {}

  Future<> Begin(const std::shared_ptr<Schema>& schema,
                 const ipc::IpcWriteOptions& options) override {
    return impl_->Begin(schema, options);
  }
  Future<> WriteRecordBatch(const RecordBatch& batch) override {
    return impl_->WriteRecordBatch(batch);
  }
  Future<> WriteMetadata(std::shared_ptr<Buffer> app_metadata) override {
    return impl_->WriteMetadata(std::move(app_metadata));
  }
  Future<> WriteWithMetadata(const RecordBatch& batch,
                             std::shared_ptr<Buffer> app_metadata) override {
    return impl_->WriteWithMetadata(batch, std::move(app_metadata));
  }
  Future<> Close() override { return impl_->Close(); }
  ipc::WriteStats stats() const override { return impl_->stats(); }

 private:
  std::shared_ptr<NativeAsyncFlightMessageWriter> impl_;
};

}  // namespace

Future<AsyncMessageReader> MakeAsyncMessageReader(
    AsyncReadFn read_fn, std::shared_ptr<MemoryManager> memory_manager) {
  auto first_message = read_fn();
  return first_message.Then(
      [read_fn = std::move(read_fn), memory_manager = std::move(memory_manager)](
          std::shared_ptr<internal::FlightData> message) mutable
          -> ::arrow::Result<AsyncMessageReader> {
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
        auto impl = std::make_shared<NativeAsyncFlightMessageReader>(
            std::move(descriptor), std::move(data), std::move(read_fn), memory_manager);
        return AsyncMessageReader{
            .reader = std::make_unique<SharedAsyncFlightMessageReader>(impl),
            .when_idle = [impl] { return impl->WhenIdle(); }};
      });
}

std::unique_ptr<AsyncFlightMetadataWriter> MakeAsyncMetadataWriter(
    std::function<Future<bool>(pb::PutResult)> write_fn) {
  return std::make_unique<NativeAsyncFlightMetadataWriter>(std::move(write_fn));
}

std::unique_ptr<AsyncFlightMessageWriter> MakeAsyncMessageWriter(
    std::function<Future<bool>(FlightPayload)> write_fn) {
  auto impl = std::make_shared<NativeAsyncFlightMessageWriter>(std::move(write_fn));
  return std::make_unique<SharedAsyncFlightMessageWriter>(std::move(impl));
}

}  // namespace arrow::flight::transport::grpc::async_internal