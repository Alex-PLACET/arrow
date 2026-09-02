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

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "arrow/array/array_base.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/ipc/dictionary.h"
#include "arrow/ipc/writer.h"

namespace arrow::flight::transport::grpc::async_internal {
namespace {

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
      switch(state_) {
        case State::kNotStarted:
          break;
        case State::kOpen:
          return Future<>::MakeFinished(
              Status::Invalid("This writer has already been started."));
        case State::kClosed:
          return Future<>::MakeFinished(Status::Invalid("This writer is already closed"));
        case State::kFailed:
          return Future<>::MakeFinished(failure_status_);
      } 
      if (!schema) {
        return Future<>::MakeFinished(Status::Invalid("Schema cannot be null"));
      }
      options_ = options;
      mapper_ = std::make_unique<ipc::DictionaryFieldMapper>(*schema);
      FlightPayload payload;
      RETURN_NOT_OK(
          ipc::GetSchemaPayload(*schema, options, *mapper_, &payload.ipc_message));
      state_ = State::kOpen;
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
    std::vector<ipc::internal::DictionaryFrame> dictionary_updates;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      RETURN_NOT_OK(CheckStartedLocked());
      RETURN_NOT_OK(ReserveWriteLocked());
      auto status = BuildDictionaryPayloadsLocked(batch, payloads, dictionary_updates);
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
    return WriteReservedPayloads(std::move(payloads));
  }

  /// Prevent future writes without cancelling an active write.
  Future<> Close() override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (state_ != State::kFailed) {
      state_ = State::kClosed;
    }
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
    switch (state_) {
      case State::kNotStarted:
        return Status::OK();
      case State::kOpen:
        return Status::OK();
      case State::kClosed:
        return Status::Invalid("This writer is already closed");
      case State::kFailed:
        return failure_status_;
    }
    return Status::Invalid("Invalid writer state");
  }

  /// Validate that batch writes are legal while mutex_ is held.
  Status CheckStartedLocked() const {
    RETURN_NOT_OK(CheckWritableLocked());
    if (state_ == State::kNotStarted) {
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

  /// Collect and encode required dictionary replacements or deltas under mutex_.
  /// `updates` carries the ipc-side bookkeeping until the batch is serialized.
  Status BuildDictionaryPayloadsLocked(const RecordBatch& batch,
                                       std::vector<FlightPayload>& payloads,
                                       std::vector<ipc::internal::DictionaryFrame>& updates) {
    RETURN_NOT_OK(ipc::internal::ComputeDictionaryFrames(
        batch, /*is_file_format=*/false, options_, *mapper_, &last_dictionaries_,
        &updates));
    for (auto& frame : updates) {
      FlightPayload payload;
      payload.ipc_message = std::move(frame.payload);
      payloads.push_back(std::move(payload));
    }
    return Status::OK();
  }

  /// Commit dictionary state after the complete batch payload is serialized.
  void CommitDictionaryUpdatesLocked(std::vector<ipc::internal::DictionaryFrame> updates) {
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

  /// Reserve the writer and serialize a payload sequence through one gRPC write at a time.
  Future<> WritePayloads(std::vector<FlightPayload> payloads) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      RETURN_NOT_OK(ReserveWriteLocked());
    }
    return WriteReservedPayloads(std::move(payloads));
  }

  /// Serialize a payload sequence after the caller has reserved the writer.
  Future<> WriteReservedPayloads(std::vector<FlightPayload> payloads) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      stats_.num_messages += static_cast<int64_t>(payloads.size());
    }
    auto out = Future<>::Make();
    auto state = std::make_shared<WriteState>(std::move(payloads));
    WritePayloadAt(state, 0, out);
    return out;
  }

  using WriteState = std::vector<FlightPayload>;

  /// Issue the next payload and retain writer state until its callback runs.
  void WritePayloadAt(const std::shared_ptr<WriteState>& state, size_t index,
                      Future<> out) {
    if (index >= state->size()) {
      FinishWrite();
      out.MarkFinished();
      return;
    }
    write_fn_(std::move((*state)[index]))
        .AddCallback([self = shared_from_this(), state, index,
                      out](const ::arrow::Result<bool>& result) mutable {
          if (!result.ok()) {
            self->FinishWrite(result.status());
            out.MarkFinished(result.status());
            return;
          }
          if (!*result) {
            const auto status = MakeFlightError(
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
      state_ = State::kFailed;
      failure_status_ = status;
    }
  }

  /// Callback used to submit serialized FlightData payloads to gRPC.
  WriteFn write_fn_;
  /// Protects writer state and dictionary bookkeeping.
  mutable std::mutex mutex_;
  /// Maps schema dictionary fields to IPC dictionary identifiers.
  std::unique_ptr<ipc::DictionaryFieldMapper> mapper_;
  /// Most recently serialized dictionary for each IPC dictionary identifier.
  std::unordered_map<int64_t, std::shared_ptr<Array>> last_dictionaries_;
  /// IPC options used for all payloads produced by this writer.
  ipc::IpcWriteOptions options_ = ipc::IpcWriteOptions::Defaults();
  /// Serialization statistics accumulated by this writer.
  ipc::WriteStats stats_;
  /// Whether one or more payloads are currently being written.
  bool write_in_flight_ = false;
  enum class State { kNotStarted, kOpen, kClosed, kFailed };
  /// Lifecycle state of the writer, excluding the independent write operation state.
  State state_ = State::kNotStarted;
  /// Error returned by subsequent operations after a terminal write failure.
  Status failure_status_;
};

}  // namespace

std::shared_ptr<AsyncFlightMessageWriter> MakeNativeAsyncMessageWriter(
    std::function<Future<bool>(FlightPayload)> write_fn) {
  return std::make_shared<NativeAsyncFlightMessageWriter>(std::move(write_fn));
}

}  // namespace arrow::flight::transport::grpc::async_internal