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

#include "arrow/flight/transport_server_async.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "arrow/buffer.h"
#include "arrow/array.h"
#include "arrow/array/data.h"
#include "arrow/flight/protocol_internal.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/server_middleware.h"
#include "arrow/flight/transport/grpc/customize_grpc.h"
#include "arrow/flight/transport/grpc/grpc_server.h"
#include "arrow/flight/transport/grpc/grpc_server_internal.h"
#include "arrow/flight/transport/grpc/serialization_internal.h"
#include "arrow/flight/transport/grpc/util_internal.h"
#include "arrow/flight/types.h"
#include "arrow/io/memory.h"
#include "arrow/compare.h"
#include "arrow/ipc/dictionary.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/logging.h"
#include "arrow/util/uri.h"

namespace arrow::flight {
namespace {

namespace pb = arrow::flight::protocol;
using FlightService = pb::FlightService;

template <typename GrpcMessage>
arrow::Result<std::shared_ptr<Buffer>> SerializeGrpcMessage(const GrpcMessage& message) {
  std::string serialized;
  if (!message.SerializeToString(&serialized)) {
    return Status::Invalid("Could not serialize gRPC message");
  }
  return Buffer::FromString(std::move(serialized));
}

arrow::Result<internal::FlightData> DeserializeGrpcFlightData(
    const pb::FlightData& message) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, SerializeGrpcMessage(message));
  return internal::DeserializeFlightData(buffer);
}

arrow::Result<pb::FlightData> SerializeFlightPayload(const FlightPayload& payload) {
  RETURN_NOT_OK(payload.Validate());
  ARROW_ASSIGN_OR_RAISE(auto buffers, internal::SerializePayloadToBuffers(payload));
  int64_t size = 0;
  for (const auto& buffer : buffers) {
    size += buffer->size();
  }
  std::string serialized;
  serialized.reserve(static_cast<size_t>(size));
  for (const auto& buffer : buffers) {
    serialized.append(reinterpret_cast<const char*>(buffer->data()),
                      static_cast<size_t>(buffer->size()));
  }
  pb::FlightData message;
  if (!message.ParseFromArray(serialized.data(), static_cast<int>(serialized.size()))) {
    return Status::Invalid("Could not parse serialized FlightData");
  }
  return message;
}

using GrpcServerCallContext =
    transport::grpc::GrpcServerCallContext<::grpc::CallbackServerContext>;
using CallbackServiceHelper =
    transport::grpc::GrpcServerCallContextHelper<::grpc::CallbackServerContext>;

constexpr int kMaxAsyncGrpcWorkerThreads = 4;

arrow::Result<std::shared_ptr<arrow::internal::ThreadPool>> MakeAsyncGrpcExecutor() {
  return arrow::internal::ThreadPool::MakeEternal(
      std::min(kMaxAsyncGrpcWorkerThreads,
               arrow::internal::ThreadPool::DefaultCapacity()));
}

::grpc::Status PrepareAuthenticatedCall(const CallbackServiceHelper& helper,
                                        FlightMethod method,
                                        ::grpc::CallbackServerContext* context,
                                        GrpcServerCallContext* flight_context) {
  auto st = helper.CheckAuth(method, context, flight_context);
  if (!st.ok()) {
    return st;
  }
  helper.AddMiddlewareHeaders(context, flight_context);
  return ::grpc::Status::OK;
}

template <typename ProtoRequest, typename ArrowRequest>
Status ParseRequiredRequest(const ProtoRequest* request, const char* request_name,
                            ArrowRequest* out) {
  if (request == nullptr) {
    return Status::Invalid(request_name, " cannot be null");
  }
  return internal::FromProto(*request, out);
}

template <typename ArrowResponse, typename ProtoResponse, typename ToProtoFn>
void FinishUnaryResult(::grpc::ServerUnaryReactor* reactor, ProtoResponse* response,
                       GrpcServerCallContext flight_context,
                       const arrow::Result<std::unique_ptr<ArrowResponse>>& result,
                       const char* not_found_message, ToProtoFn&& to_proto) {
  if (result.ok() && result.ValueUnsafe()) {
    reactor->Finish(
        flight_context.FinishRequest(to_proto(*result.ValueUnsafe(), response)));
    return;
  }
  reactor->Finish(flight_context.FinishRequest(
      result.ok() ? Status::KeyError(not_found_message) : result.status()));
}

template <typename Proto>
class ImmediateWriteReactor final : public ::grpc::ServerWriteReactor<Proto> {
 public:
  explicit ImmediateWriteReactor(::grpc::Status status) { this->Finish(std::move(status)); }
  void OnDone() override { delete this; }
};

template <typename Req, typename Resp>
class BidiReactorBase : public ::grpc::ServerBidiReactor<Req, Resp> {
 public:
  BidiReactorBase(::grpc::CallbackServerContext* context,
                  std::shared_ptr<arrow::internal::ThreadPool> executor)
      : context_(context), executor_(std::move(executor)) {
    if constexpr (std::is_same_v<Req, pb::FlightData>) {
      transport::grpc::RegisterGrpcFlightDataMessage(&read_buffer_);
      read_buffer_registered_ = true;
    }
  }

  Status StartWorker(std::function<void()> fn) {
    // For bidi RPCs the first read must be armed explicitly. After that, reads
    // are re-armed only from OnReadDone() so the callback path owns the
    // receive-side state machine.
    {
      std::unique_lock<std::mutex> lock(mutex_);
      read_started_ = true;
      read_in_flight_ = true;
    }
    this->StartRead(&read_buffer_);
    refs_.fetch_add(1, std::memory_order_relaxed);
    auto maybe_future = executor_->Submit([this, fn = std::move(fn)]() mutable {
      fn();
      ReleaseRef();
    });
    if (!maybe_future.ok()) {
      ReleaseRef();
      return maybe_future.status();
    }
    return Status::OK();
  }

  void OnReadDone(bool ok) override {
    std::optional<Req> completed_read;
    Future<std::optional<Req>> pending_future;
    bool resolve_pending = false;
    bool start_next_read = false;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      read_in_flight_ = false;
      if (ok) {
        completed_read.emplace(read_buffer_);
        if (!cancelled_) {
          read_in_flight_ = true;
          start_next_read = true;
        }
        if (pending_read_active_) {
          pending_future = pending_read_;
          pending_read_active_ = false;
          pending_read_ = Future<std::optional<Req>>();
          resolve_pending = true;
        } else {
          reads_.push_back(std::move(*completed_read));
        }
      } else {
        reads_done_ = true;
        if (pending_read_active_) {
          pending_future = pending_read_;
          pending_read_active_ = false;
          pending_read_ = Future<std::optional<Req>>();
          resolve_pending = true;
        }
      }
    }

    if (start_next_read) {
      // Post the next read before resolving the waiting Future so EOF can race
      // with user callbacks without stalling the receive loop.
      this->StartRead(&read_buffer_);
    }
    if (resolve_pending) {
      if (ok) {
        pending_future.MarkFinished(std::move(completed_read));
      } else {
        pending_future.MarkFinished(std::optional<Req>{});
      }
    }
    cv_.notify_all();
  }

  void OnWriteDone(bool ok) override {
    Future<bool> pending_future;
    bool resolve_pending = false;
    bool finish_now = false;
    std::unique_lock<std::mutex> lock(mutex_);
    if constexpr (std::is_same_v<Resp, pb::FlightData>) {
      if (flight_data_registered_) {
        transport::grpc::UnregisterGrpcFlightDataMessage(&current_write_);
        flight_data_registered_ = false;
      }
    }
    write_in_flight_ = false;
    write_ok_ = ok;
    if (pending_write_active_) {
      pending_future = pending_write_;
      pending_write_active_ = false;
      pending_write_ = Future<bool>();
      resolve_pending = true;
    }
    if (finish_requested_) {
      finish_now = true;
    }
    lock.unlock();
    if (resolve_pending) {
      pending_future.MarkFinished(!cancelled_ && write_ok_);
    }
    if (finish_now) {
      this->Finish(finish_status_);
    }
    cv_.notify_all();
  }

  void OnCancel() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cancelled_ = true;
    if (pending_read_active_) {
      auto future = pending_read_;
      pending_read_active_ = false;
      pending_read_ = Future<std::optional<Req>>();
      lock.unlock();
      future.MarkFinished(std::optional<Req>{});
      cv_.notify_all();
      return;
    }
    cv_.notify_all();
  }

  void OnDone() override {
    if constexpr (std::is_same_v<Req, pb::FlightData>) {
      if (read_buffer_registered_) {
        transport::grpc::UnregisterGrpcFlightDataMessage(&read_buffer_);
        read_buffer_registered_ = false;
      }
    }
    if constexpr (std::is_same_v<Resp, pb::FlightData>) {
      if (flight_data_registered_) {
        transport::grpc::UnregisterGrpcFlightDataMessage(&current_write_);
        flight_data_registered_ = false;
      }
    }
    ReleaseRef();
  }

  bool ReadOne(Req* out) { return PopRead(out); }
  bool WriteOnePublic(Resp message) { return WriteOne(std::move(message)); }
  Future<std::optional<Req>> ReadOneAsync() {
    bool start_initial_read = false;
    std::unique_lock<std::mutex> lock(mutex_);
    if (!reads_.empty()) {
      Req out = std::move(reads_.front());
      reads_.pop_front();
      return Future<std::optional<Req>>::MakeFinished(std::optional<Req>(std::move(out)));
    }
    if (cancelled_ || reads_done_) {
      return Future<std::optional<Req>>::MakeFinished(std::optional<Req>{});
    }
    if (pending_read_active_) {
      return Future<std::optional<Req>>::MakeFinished(
          Status::Invalid("Concurrent async reads are not supported"));
    }
    pending_read_ = Future<std::optional<Req>>::Make();
    pending_read_active_ = true;
    if (!read_started_ && !read_in_flight_) {
      // DoPut/DoExchange do not call StartWorker(); their first consumer read
      // is what arms the callback read loop.
      read_started_ = true;
      read_in_flight_ = true;
      start_initial_read = true;
    }
    auto future = pending_read_;
    lock.unlock();
    if (start_initial_read) {
      this->StartRead(&read_buffer_);
    }
    return future;
  }
  Future<bool> WriteOneAsync(Resp message) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cancelled_) {
      return Future<bool>::MakeFinished(false);
    }
    if (pending_write_active_ || write_in_flight_) {
      return Future<bool>::MakeFinished(
          Status::Invalid("Concurrent async writes are not supported"));
    }
    pending_write_ = Future<bool>::Make();
    pending_write_active_ = true;
    current_write_ = std::move(message);
    write_in_flight_ = true;
    this->StartWrite(&current_write_);
    return pending_write_;
  }
  Future<bool> WritePayloadAsync(FlightPayload payload) {
    static_assert(std::is_same_v<Resp, pb::FlightData>);
    ARROW_ASSIGN_OR_RAISE(current_write_, SerializeFlightPayload(payload));
    std::unique_lock<std::mutex> lock(mutex_);
    if (cancelled_) {
      return Future<bool>::MakeFinished(false);
    }
    if (pending_write_active_ || write_in_flight_) {
      return Future<bool>::MakeFinished(
          Status::Invalid("Concurrent async writes are not supported"));
    }
    pending_write_ = Future<bool>::Make();
    pending_write_active_ = true;
    write_in_flight_ = true;
    transport::grpc::RegisterGrpcFlightDataMessage(&current_write_);
    flight_data_registered_ = true;
    this->StartWrite(&current_write_);
    return pending_write_;
  }
  arrow::Result<bool> WritePayloadPublic(FlightPayload payload) {
    static_assert(std::is_same_v<Resp, pb::FlightData>);
    ARROW_ASSIGN_OR_RAISE(current_write_, SerializeFlightPayload(payload));
    std::unique_lock<std::mutex> lock(mutex_);
    if (cancelled_) return false;
    write_in_flight_ = true;
    write_ok_ = true;
    transport::grpc::RegisterGrpcFlightDataMessage(&current_write_);
    flight_data_registered_ = true;
    this->StartWrite(&current_write_);
    cv_.wait(lock, [&] { return !write_in_flight_ || cancelled_; });
    return !cancelled_ && write_ok_;
  }

  void Hold() { refs_.fetch_add(1, std::memory_order_relaxed); }
  void ReleaseHold() { ReleaseRef(); }

 protected:
  bool PopRead(Req* out) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return cancelled_ || reads_done_ || !reads_.empty(); });
    if (!reads_.empty()) {
      *out = std::move(reads_.front());
      reads_.pop_front();
      return true;
    }
    return false;
  }

  bool WriteOne(Resp message) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cancelled_) return false;
    current_write_ = std::move(message);
    write_in_flight_ = true;
    write_ok_ = true;
    this->StartWrite(&current_write_);
    cv_.wait(lock, [&] { return !write_in_flight_ || cancelled_; });
    return !cancelled_ && write_ok_;
  }

  void FinishFromWorker(::grpc::Status status) {
    std::unique_lock<std::mutex> lock(mutex_);
    finish_requested_ = true;
    finish_status_ = std::move(status);
    if (!write_in_flight_) {
      this->Finish(finish_status_);
    }
  }

  void ReleaseRef() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  ::grpc::CallbackServerContext* context_;
  std::shared_ptr<arrow::internal::ThreadPool> executor_;
  Req read_buffer_;
  std::deque<Req> reads_;
  Resp current_write_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool reads_done_ = false;
  Future<std::optional<Req>> pending_read_;
  bool pending_read_active_ = false;
  bool read_buffer_registered_ = false;
  bool read_started_ = false;
  bool read_in_flight_ = false;
  bool cancelled_ = false;
  Future<bool> pending_write_;
  bool pending_write_active_ = false;
  bool write_in_flight_ = false;
  bool write_ok_ = true;
  bool flight_data_registered_ = false;
  bool finish_requested_ = false;
  ::grpc::Status finish_status_;
  std::atomic<int> refs_{1};
};

template <typename ReactorT>
ReactorT* FinishNow(ReactorT* reactor, const ::grpc::Status& status) {
  reactor->Finish(status);
  return reactor;
}

template <typename Proto>
::grpc::ServerWriteReactor<Proto>* FinishWriteNow(const ::grpc::Status& status) {
  return new ImmediateWriteReactor<Proto>(status);
}

class HandshakeAuthReader : public ServerAuthReader {
 public:
  explicit HandshakeAuthReader(BidiReactorBase<pb::HandshakeRequest, pb::HandshakeResponse>* r)
      : reactor_(r) {}
  Status Read(std::string* token) override {
    pb::HandshakeRequest request;
    if (!reactor_->ReadOne(&request)) {
      return Status::IOError("Stream is closed.");
    }
    *token = request.payload();
    return Status::OK();
  }

 private:
  BidiReactorBase<pb::HandshakeRequest, pb::HandshakeResponse>* reactor_;
};

class HandshakeAuthWriter : public ServerAuthSender {
 public:
  explicit HandshakeAuthWriter(
      BidiReactorBase<pb::HandshakeRequest, pb::HandshakeResponse>* reactor)
      : reactor_(reactor) {}
  Status Write(const std::string& token) override {
    pb::HandshakeResponse response;
    response.set_payload(token);
    if (!reactor_->WriteOnePublic(std::move(response))) {
      return Status::IOError("Stream was closed.");
    }
    return Status::OK();
  }

 private:
  BidiReactorBase<pb::HandshakeRequest, pb::HandshakeResponse>* reactor_;
};

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

arrow::Result<std::shared_ptr<Buffer>> SerializeIpcMessage(
    internal::FlightData& data, const ipc::IpcWriteOptions& options) {
  ARROW_ASSIGN_OR_RAISE(auto message, data.OpenMessage());
  ARROW_ASSIGN_OR_RAISE(auto sink, io::BufferOutputStream::Create());
  int64_t unused_size = 0;
  RETURN_NOT_OK(message->SerializeTo(sink.get(), options, &unused_size));
  return sink->Finish();
}

class NativeAsyncFlightMessageReader final : public AsyncFlightMessageReader {
 public:
  using ReadFn = std::function<Future<std::optional<pb::FlightData>>()>;
  enum class ActiveOperation { kNone, kSchema, kNext };

  NativeAsyncFlightMessageReader(FlightDescriptor descriptor, internal::FlightData first_message,
                                 ReadFn read_fn,
                                 std::shared_ptr<MemoryManager> memory_manager)
      : descriptor_(std::move(descriptor)),
        pending_message_(std::move(first_message)),
        read_fn_(std::move(read_fn)),
        memory_manager_(std::move(memory_manager)),
        listener_(std::make_shared<ipc::CollectListener>()),
        decoder_(listener_) {}

  const FlightDescriptor& descriptor() const override { return descriptor_; }

  Future<std::shared_ptr<Schema>> GetSchema() override {
    if (listener_->schema()) {
      return Future<std::shared_ptr<Schema>>::MakeFinished(listener_->schema());
    }
    int64_t token = 0;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (operation_in_flight_) {
        return Future<std::shared_ptr<Schema>>::MakeFinished(
            Status::Invalid("Concurrent async reads are not supported"));
      }
      operation_in_flight_ = true;
      active_operation_ = ActiveOperation::kSchema;
      token = ++active_token_;
    }
    auto out = Future<std::shared_ptr<Schema>>::Make();
    PumpSchema(out, token);
    return out;
  }

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
      if (operation_in_flight_) {
        return Future<FlightStreamChunk>::MakeFinished(
            Status::Invalid("Concurrent async reads are not supported"));
      }
      operation_in_flight_ = true;
      active_operation_ = ActiveOperation::kNext;
      token = ++active_token_;
    }
    auto out = Future<FlightStreamChunk>::Make();
    PumpNext(out, token);
    return out;
  }

  ipc::ReadStats stats() const override { return decoder_.stats(); }

 private:
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
      return Future<std::shared_ptr<internal::FlightData>>::MakeFinished(std::move(pending));
    }
    return read_fn_().Then(
        [memory_manager = memory_manager_](std::optional<pb::FlightData> message)
            -> ::arrow::Result<std::shared_ptr<internal::FlightData>> {
      if (!message) {
        return std::shared_ptr<internal::FlightData>{};
      }
      ARROW_ASSIGN_OR_RAISE(auto data, DeserializeGrpcFlightData(*message));
      if (data.body) {
        ARROW_ASSIGN_OR_RAISE(data.body, Buffer::ViewOrCopy(data.body, memory_manager));
      }
      return std::make_shared<internal::FlightData>(std::move(data));
    });
  }

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

  bool IsCurrentOperation(int64_t token, ActiveOperation operation) {
    std::lock_guard<std::mutex> guard(mutex_);
    return operation_in_flight_ && active_token_ == token && active_operation_ == operation;
  }

  void FinishSchema(Future<std::shared_ptr<Schema>> out,
                    ::arrow::Result<std::shared_ptr<Schema>> result, int64_t token) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (active_token_ != token || active_operation_ != ActiveOperation::kSchema) {
        return;
      }
      operation_in_flight_ = false;
      active_operation_ = ActiveOperation::kNone;
    }
    out.MarkFinished(std::move(result));
  }

  void PumpSchema(Future<std::shared_ptr<Schema>> out, int64_t token) {
    if (listener_->schema()) {
      FinishSchema(out, listener_->schema(), token);
      return;
    }
    ReadDataAsync().AddCallback(
        [this, out, token](
            const ::arrow::Result<std::shared_ptr<internal::FlightData>>& result) mutable {
      if (!IsCurrentOperation(token, ActiveOperation::kSchema)) {
        return;
      }
      if (!result.ok()) {
        FinishSchema(out, result.status(), token);
        return;
      }
      auto maybe_data = result.ValueUnsafe();
      if (!maybe_data) {
        FinishSchema(out, Status::IOError("Client never sent a data message"), token);
        return;
      }
      if (!maybe_data->metadata) {
        // Descriptor-only or metadata-only FlightData frames do not advance the
        // IPC decoder toward a schema, so continue reading until an IPC message
        // arrives or the stream ends.
        PumpSchema(out, token);
        return;
      }
      auto st = ConsumeDataMessage(*maybe_data);
      if (!st.ok()) {
        FinishSchema(out, st, token);
        return;
      }
      PumpSchema(out, token);
    });
  }

  void FinishNext(Future<FlightStreamChunk> out,
                  ::arrow::Result<FlightStreamChunk> result, int64_t token) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (active_token_ != token || active_operation_ != ActiveOperation::kNext) {
        return;
      }
      operation_in_flight_ = false;
      active_operation_ = ActiveOperation::kNone;
    }
    out.MarkFinished(std::move(result));
  }

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
      FinishNext(out, std::move(*ready_chunk), token);
      return;
    }
    if (stream_finished) {
      FinishNext(out, FlightStreamChunk{}, token);
      return;
    }
    ReadDataAsync().AddCallback(
        [this, out, token](
            const ::arrow::Result<std::shared_ptr<internal::FlightData>>& result) mutable {
      if (!IsCurrentOperation(token, ActiveOperation::kNext)) {
        return;
      }
      if (!result.ok()) {
        FinishNext(out, result.status(), token);
        return;
      }
      auto maybe_data = result.ValueUnsafe();
      if (!maybe_data) {
        {
          std::lock_guard<std::mutex> guard(mutex_);
          finished_ = true;
        }
        FinishNext(out, FlightStreamChunk{}, token);
        return;
      }
      if (!maybe_data->metadata) {
        if (!maybe_data->app_metadata) {
          // The initial descriptor frame is visible here after
          // MakeAsyncMessageReader() peels it off the stream. It is not a user
          // chunk, so continue to the next FlightData message.
          PumpNext(out, token);
          return;
        }
        FlightStreamChunk chunk;
        chunk.app_metadata = std::move(maybe_data->app_metadata);
        FinishNext(out, std::move(chunk), token);
        return;
      }
      auto st = ConsumeDataMessage(*maybe_data);
      if (!st.ok()) {
        FinishNext(out, st, token);
        return;
      }
      PumpNext(out, token);
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
  bool operation_in_flight_ = false;
  ActiveOperation active_operation_ = ActiveOperation::kNone;
  int64_t active_token_ = 0;
};

class NativeAsyncFlightMetadataWriter final : public AsyncFlightMetadataWriter {
 public:
  using WriteFn = std::function<Future<bool>(pb::PutResult)>;

  explicit NativeAsyncFlightMetadataWriter(WriteFn write_fn) : write_fn_(std::move(write_fn)) {}

  Future<> WriteMetadata(const Buffer& app_metadata) override {
    pb::PutResult result;
    result.set_app_metadata(app_metadata.data(), app_metadata.size());
    return write_fn_(std::move(result))
        .Then([](bool ok) -> Status {
          if (!ok) {
            return Status::IOError("Unknown error writing metadata.");
          }
          return Status::OK();
        });
  }

 private:
  WriteFn write_fn_;
};

class NativeAsyncFlightMessageWriter final : public AsyncFlightMessageWriter {
 public:
  using WriteFn = std::function<Future<bool>(FlightPayload)>;

  explicit NativeAsyncFlightMessageWriter(WriteFn write_fn) : write_fn_(std::move(write_fn)) {}

  Future<> Begin(const std::shared_ptr<Schema>& schema,
                 const ipc::IpcWriteOptions& options) override {
    std::vector<FlightPayload> payloads;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (begun_) {
        return Future<>::MakeFinished(Status::Invalid(
            "This writer has already been started."));
      }
      options_ = options;
      schema_ = schema;
      mapper_ = std::make_unique<ipc::DictionaryFieldMapper>(*schema);
      FlightPayload payload;
      RETURN_NOT_OK(ipc::GetSchemaPayload(*schema_, options_, *mapper_, &payload.ipc_message));
      begun_ = true;
      payloads.push_back(std::move(payload));
    }
    return WritePayloads(std::move(payloads));
  }

  Future<> WriteRecordBatch(const RecordBatch& batch) override {
    return WriteWithMetadata(batch, nullptr);
  }

  Future<> WriteMetadata(std::shared_ptr<Buffer> app_metadata) override {
    FlightPayload payload;
    payload.app_metadata = std::move(app_metadata);
    return WritePayloads({std::move(payload)});
  }

  Future<> WriteWithMetadata(const RecordBatch& batch,
                             std::shared_ptr<Buffer> app_metadata) override {
    std::vector<FlightPayload> payloads;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      RETURN_NOT_OK(CheckStartedLocked());
      RETURN_NOT_OK(BuildDictionaryPayloadsLocked(batch, &payloads));
      FlightPayload batch_payload;
      RETURN_NOT_OK(ipc::GetRecordBatchPayload(batch, options_, &batch_payload.ipc_message));
      batch_payload.app_metadata = std::move(app_metadata);
      payloads.push_back(std::move(batch_payload));
      ++stats_.num_record_batches;
      stats_.total_raw_body_size += payloads.back().ipc_message.raw_body_length;
      stats_.total_serialized_body_size += payloads.back().ipc_message.body_length;
    }
    return WritePayloads(std::move(payloads));
  }

  Future<> Close() override {
    std::lock_guard<std::mutex> guard(mutex_);
    closed_ = true;
    return Future<>::MakeFinished();
  }

  ipc::WriteStats stats() const override {
    std::lock_guard<std::mutex> guard(mutex_);
    return stats_;
  }

 private:
  Status CheckStartedLocked() const {
    if (!begun_) {
      return Status::Invalid("This writer is not started. Call Begin() with a schema");
    }
    if (closed_) {
      return Status::Invalid("This writer is already closed");
    }
    return Status::OK();
  }

  Status BuildDictionaryPayloadsLocked(const RecordBatch& batch,
                                       std::vector<FlightPayload>* payloads) {
    ARROW_ASSIGN_OR_RAISE(const auto dictionaries, ipc::CollectDictionaries(batch, *mapper_));
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
      ++stats_.num_dictionary_batches;
      if (dictionary_exists) {
        if (delta_start) {
          ++stats_.num_dictionary_deltas;
        } else {
          ++stats_.num_replaced_dictionaries;
        }
      }
      *last_dictionary = dictionary;
    }
    return Status::OK();
  }

  Future<> WritePayloads(std::vector<FlightPayload> payloads) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (write_in_flight_) {
        return Future<>::MakeFinished(
            Status::Invalid("Concurrent async writes are not supported"));
      }
      write_in_flight_ = true;
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

  void WritePayloadAt(const std::shared_ptr<WriteState>& state, size_t index, Future<> out) {
    if (index >= state->payloads.size()) {
      {
        std::lock_guard<std::mutex> guard(mutex_);
        write_in_flight_ = false;
      }
      out.MarkFinished();
      return;
    }
    write_fn_(std::move(state->payloads[index]))
        .AddCallback(
            [this, state, index, out](const ::arrow::Result<bool>& result) mutable {
          if (!result.ok()) {
            {
              std::lock_guard<std::mutex> guard(mutex_);
              write_in_flight_ = false;
            }
            out.MarkFinished(result.status());
            return;
          }
          if (!*result) {
            {
              std::lock_guard<std::mutex> guard(mutex_);
              write_in_flight_ = false;
            }
            out.MarkFinished(MakeFlightError(
                FlightStatusCode::Internal,
                "Could not write record batch to stream (client disconnect?)"));
            return;
          }
          WritePayloadAt(state, index + 1, out);
        });
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
};

class AsyncGrpcServerTransport;

class CallbackFlightService final : public FlightService::CallbackService {
 public:
  CallbackFlightService(AsyncGrpcServerTransport* impl, CallbackServiceHelper helper)
      : impl_(impl), helper_(std::move(helper)) {}

  ::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>* Handshake(
      ::grpc::CallbackServerContext* context) override;
  ::grpc::ServerWriteReactor<pb::FlightInfo>* ListFlights(
      ::grpc::CallbackServerContext* context, const pb::Criteria* request) override;
  ::grpc::ServerUnaryReactor* GetFlightInfo(::grpc::CallbackServerContext* context,
                                            const pb::FlightDescriptor* request,
                                            pb::FlightInfo* response) override;
  ::grpc::ServerUnaryReactor* PollFlightInfo(::grpc::CallbackServerContext* context,
                                             const pb::FlightDescriptor* request,
                                             pb::PollInfo* response) override;
  ::grpc::ServerUnaryReactor* GetSchema(::grpc::CallbackServerContext* context,
                                        const pb::FlightDescriptor* request,
                                        pb::SchemaResult* response) override;
  ::grpc::ServerWriteReactor<pb::FlightData>* DoGet(::grpc::CallbackServerContext* context,
                                                    const pb::Ticket* request) override;
  ::grpc::ServerBidiReactor<pb::FlightData, pb::PutResult>* DoPut(
      ::grpc::CallbackServerContext* context) override;
  ::grpc::ServerBidiReactor<pb::FlightData, pb::FlightData>* DoExchange(
      ::grpc::CallbackServerContext* context) override;
  ::grpc::ServerWriteReactor<pb::ActionType>* ListActions(
      ::grpc::CallbackServerContext* context, const pb::Empty* request) override;
  ::grpc::ServerWriteReactor<pb::Result>* DoAction(::grpc::CallbackServerContext* context,
                                                   const pb::Action* request) override;

 private:
  AsyncGrpcServerTransport* impl_;
  CallbackServiceHelper helper_;
};

class AsyncGrpcServerTransport : public internal::AsyncServerTransport {
 public:
  AsyncGrpcServerTransport(AsyncFlightServerBase* base,
                           std::shared_ptr<MemoryManager> memory_manager)
      : internal::AsyncServerTransport(base, std::move(memory_manager)) {}

  Status Init(const FlightServerOptions& options, const arrow::util::Uri& uri) override {
    // Callback handlers should stay non-blocking; Arrow work runs on a small
    // transport-owned pool so multiple async servers do not share hidden
    // process-global state or oversubscribe threads under load.
    ARROW_ASSIGN_OR_RAISE(executor_pool_, MakeAsyncGrpcExecutor());
    helper_ = std::make_unique<CallbackServiceHelper>(options.auth_handler, options.middleware);
    grpc_service_ = std::make_unique<CallbackFlightService>(this, *helper_);

    ::grpc::ServerBuilder builder;
    // The callback API does not use the synchronous server completion queues.
    // Leaving the sync server machinery at gRPC defaults adds idle polling
    // threads that dominate the thread-scaling tests.
    builder.SetSyncServerOption(::grpc::ServerBuilder::SyncServerOption::NUM_CQS, 0);
    builder.SetSyncServerOption(::grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, 1);
    builder.SetSyncServerOption(::grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, 1);
    int port = 0;
    RETURN_NOT_OK(
        transport::grpc::AddServerListeningPort(options, uri, &builder, &location_, &port));

    builder.RegisterService(grpc_service_.get());
    transport::grpc::ConfigureServerBuilderOptions(options, &builder);

    grpc_server_ = builder.BuildAndStart();
    if (!grpc_server_) {
      return Status::UnknownError("Server did not start properly");
    }
    return transport::grpc::SetServerLocationFromUri(uri, port, &location_);
  }

  Status Shutdown() override {
    grpc_server_->Shutdown();
    return Status::OK();
  }
  Status Shutdown(const std::chrono::system_clock::time_point& deadline) override {
    grpc_server_->Shutdown(deadline);
    return Status::OK();
  }
  Status Wait() override {
    grpc_server_->Wait();
    return Status::OK();
  }
  Location location() const override { return location_; }

  const CallbackServiceHelper& helper() const { return *helper_; }
  std::shared_ptr<arrow::internal::ThreadPool> executor() const { return executor_pool_; }

  template <typename Resp>
  Future<std::unique_ptr<AsyncFlightMessageReader>> MakeAsyncMessageReader(
      BidiReactorBase<pb::FlightData, Resp>* reactor) {
    return reactor->ReadOneAsync().Then(
        [this, reactor](std::optional<pb::FlightData> message)
            -> ::arrow::Result<std::unique_ptr<AsyncFlightMessageReader>> {
          if (!message) {
            return Status::IOError("Stream finished before first message sent");
          }
          ARROW_ASSIGN_OR_RAISE(auto data, DeserializeGrpcFlightData(*message));
          if (!data.descriptor) {
            return Status::IOError("Descriptor missing on first message");
          }
          auto descriptor = *data.descriptor;
          return std::unique_ptr<AsyncFlightMessageReader>(
              new NativeAsyncFlightMessageReader(
                  std::move(descriptor), std::move(data),
                  [reactor]() { return reactor->ReadOneAsync(); }, memory_manager_));
        });
  }

  std::unique_ptr<AsyncFlightMetadataWriter> MakeAsyncMetadataWriter(
      BidiReactorBase<pb::FlightData, pb::PutResult>* reactor) {
    return std::unique_ptr<AsyncFlightMetadataWriter>(new NativeAsyncFlightMetadataWriter(
        [reactor](pb::PutResult result) { return reactor->WriteOneAsync(std::move(result)); }));
  }

  std::unique_ptr<AsyncFlightMessageWriter> MakeAsyncMessageWriter(
      BidiReactorBase<pb::FlightData, pb::FlightData>* reactor) {
    return std::unique_ptr<AsyncFlightMessageWriter>(new NativeAsyncFlightMessageWriter(
        [reactor](FlightPayload payload) { return reactor->WritePayloadAsync(std::move(payload)); }));
  }

 private:
  std::shared_ptr<arrow::internal::ThreadPool> executor_pool_;
  std::unique_ptr<CallbackServiceHelper> helper_;
  std::unique_ptr<CallbackFlightService> grpc_service_;
  std::unique_ptr<::grpc::Server> grpc_server_;
  Location location_;
};

template <typename Proto>
class AsyncWriteReactorBase : public ::grpc::ServerWriteReactor<Proto> {
 public:
  AsyncWriteReactorBase(::grpc::CallbackServerContext* context,
                        std::shared_ptr<arrow::internal::ThreadPool> executor,
                        GrpcServerCallContext flight_context)
      : context_(context),
        executor_(std::move(executor)),
        flight_context_(std::move(flight_context)) {}

  void OnCancel() override { cancelled_.store(true, std::memory_order_relaxed); }

  void OnDone() override { ReleaseRef(); }

 protected:
  // Reactor instances are self-owned: the gRPC callback path holds one
  // reference and each in-flight background task adds one more until its
  // callback completes. This keeps callback-thread work minimal while making
  // teardown deterministic.
  template <typename Fn>
  Status StartBackgroundWork(Fn&& fn) {
    refs_.fetch_add(1, std::memory_order_relaxed);
    auto maybe_future = executor_->Submit(
        [this, fn = std::forward<Fn>(fn)]() mutable {
          fn();
          ReleaseRef();
        });
    if (!maybe_future.ok()) {
      ReleaseRef();
      return maybe_future.status();
    }
    return Status::OK();
  }

  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

  void FinishOnce(::grpc::Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      this->Finish(std::move(status));
    }
  }

  ::grpc::CallbackServerContext* context_;
  std::shared_ptr<arrow::internal::ThreadPool> executor_;
  GrpcServerCallContext flight_context_;
  Proto current_write_;
  std::mutex mutex_;

 private:
  void ReleaseRef() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  std::atomic<bool> cancelled_{false};
  std::atomic<bool> finished_{false};
  std::atomic<int> refs_{1};
};

template <typename Proto, typename UserType>
class IteratorReactor : public AsyncWriteReactorBase<Proto> {
 public:
  using NextFn = std::function<arrow::Result<std::unique_ptr<UserType>>()>;
  using ToProtoFn = std::function<Status(const UserType&, Proto*)>;

  IteratorReactor(::grpc::CallbackServerContext* context,
                  std::shared_ptr<arrow::internal::ThreadPool> executor,
                  GrpcServerCallContext flight_context, NextFn next_fn, ToProtoFn to_proto)
      : AsyncWriteReactorBase<Proto>(context, std::move(executor),
                                     std::move(flight_context)),
        next_fn_(std::move(next_fn)),
        to_proto_(std::move(to_proto)) {}

  void Start() {
    auto status = ScheduleAdvance();
    if (!status.ok()) {
      this->FinishOnce(this->flight_context_.FinishRequest(status));
    }
  }

  void OnWriteDone(bool ok) override {
    if (!ok) {
      this->FinishOnce(this->flight_context_.FinishRequest(Status::OK()));
      return;
    }
    auto status = ScheduleAdvance();
    if (!status.ok()) {
      this->FinishOnce(this->flight_context_.FinishRequest(status));
    }
  }

 private:
  Status ScheduleAdvance() {
    return this->StartBackgroundWork([this] {
      if (this->cancelled()) {
        return;
      }

      auto maybe_value = next_fn_();
      if (!maybe_value.ok()) {
        this->FinishOnce(this->flight_context_.FinishRequest(maybe_value.status()));
        return;
      }

      auto value = std::move(maybe_value).ValueUnsafe();
      if (!value) {
        this->FinishOnce(this->flight_context_.FinishRequest(Status::OK()));
        return;
      }

      Proto proto;
      auto st = to_proto_(*value, &proto);
      if (!st.ok()) {
        this->FinishOnce(this->flight_context_.FinishRequest(st));
        return;
      }

      {
        std::lock_guard<std::mutex> lock(this->mutex_);
        if (this->cancelled()) {
          return;
        }
        this->current_write_ = std::move(proto);
      }
      this->StartWrite(&this->current_write_);
    });
  }

  NextFn next_fn_;
  ToProtoFn to_proto_;
};

class DoGetReactor : public AsyncWriteReactorBase<pb::FlightData> {
 public:
  DoGetReactor(::grpc::CallbackServerContext* context,
               std::shared_ptr<arrow::internal::ThreadPool> executor,
               GrpcServerCallContext flight_context, std::unique_ptr<FlightDataStream> stream)
      : AsyncWriteReactorBase<pb::FlightData>(context, std::move(executor),
                                              std::move(flight_context)),
        stream_(std::move(stream)) {}

  void Start() {
    auto status = ScheduleAdvance();
    if (!status.ok()) {
      this->FinishOnce(this->flight_context_.FinishRequest(status));
    }
  }

  void OnWriteDone(bool ok) override {
    if (flight_data_registered_) {
      transport::grpc::UnregisterGrpcFlightDataMessage(&this->current_write_);
      flight_data_registered_ = false;
    }
    if (!ok) {
      this->FinishOnce(this->flight_context_.FinishRequest(Status::OK()));
      return;
    }
    auto status = ScheduleAdvance();
    if (!status.ok()) {
      this->FinishOnce(this->flight_context_.FinishRequest(status));
    }
  }

  void OnDone() override {
    if (flight_data_registered_) {
      transport::grpc::UnregisterGrpcFlightDataMessage(&this->current_write_);
      flight_data_registered_ = false;
    }
    AsyncWriteReactorBase<pb::FlightData>::OnDone();
  }

 private:
  enum class Stage { kSchema, kPayloads, kFinish };

  Status ScheduleAdvance() {
    return this->StartBackgroundWork([this] {
      if (this->cancelled()) {
        return;
      }

      if (!stream_) {
        this->FinishOnce(
            this->flight_context_.FinishRequest(Status::KeyError("No data in this flight")));
        return;
      }

      if (stage_ == Stage::kSchema) {
        auto maybe_schema = stream_->GetSchemaPayload();
        if (!maybe_schema.ok()) {
          this->FinishOnce(this->flight_context_.FinishRequest(maybe_schema.status()));
          return;
        }
        auto status = StartPayloadWrite(std::move(maybe_schema).MoveValueUnsafe());
        if (!status.ok()) {
          this->FinishOnce(this->flight_context_.FinishRequest(status));
          return;
        }
        stage_ = Stage::kPayloads;
        return;
      }

      if (stage_ == Stage::kPayloads) {
        auto maybe_payload = stream_->Next();
        if (!maybe_payload.ok()) {
          this->FinishOnce(this->flight_context_.FinishRequest(maybe_payload.status()));
          return;
        }
        auto payload = std::move(maybe_payload).MoveValueUnsafe();
        if (payload.ipc_message.metadata == nullptr) {
          stage_ = Stage::kFinish;
        } else {
          auto status = StartPayloadWrite(std::move(payload));
          if (!status.ok()) {
            this->FinishOnce(this->flight_context_.FinishRequest(status));
          }
          return;
        }
      }

      auto close_status = stream_->Close();
      this->FinishOnce(this->flight_context_.FinishRequest(close_status));
    });
  }

  Status StartPayloadWrite(FlightPayload payload) {
    ARROW_ASSIGN_OR_RAISE(this->current_write_, SerializeFlightPayload(payload));
    if (this->cancelled()) {
      return Status::OK();
    }
    transport::grpc::RegisterGrpcFlightDataMessage(&this->current_write_);
    flight_data_registered_ = true;
    this->StartWrite(&this->current_write_);
    return Status::OK();
  }
  std::unique_ptr<FlightDataStream> stream_;
  Stage stage_ = Stage::kSchema;
  bool flight_data_registered_ = false;
};

::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>*
CallbackFlightService::Handshake(::grpc::CallbackServerContext* context) {
  class Reactor final : public BidiReactorBase<pb::HandshakeRequest, pb::HandshakeResponse> {
   public:
    Reactor(::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
            const CallbackServiceHelper& helper)
        : BidiReactorBase(context, impl->executor()),
          impl_(impl),
          helper_(helper),
          flight_context_(context) {}

    void Start() {
      auto st =
          helper_.MakeCallContext(FlightMethod::Handshake, this->context_, &flight_context_);
      if (!st.ok()) {
        this->Finish(st);
        return;
      }
      helper_.AddMiddlewareHeaders(this->context_, &flight_context_);
      auto status = this->StartWorker([this] {
        Status status;
        if (helper_.auth_handler()) {
          auto outgoing = std::make_unique<HandshakeAuthWriter>(this);
          auto incoming = std::make_unique<HandshakeAuthReader>(this);
          status = helper_.auth_handler()->Authenticate(flight_context_, outgoing.get(),
                                                        incoming.get());
        } else {
          auto outgoing = std::make_unique<HandshakeAuthWriter>(this);
          auto incoming = std::make_unique<HandshakeAuthReader>(this);
          status =
              impl_->base()->Handshake(flight_context_, std::move(outgoing), std::move(incoming))
                  .status();
        }
        this->FinishFromWorker(flight_context_.FinishRequest(status));
      });
      if (!status.ok()) {
        this->Finish(flight_context_.FinishRequest(status));
      }
    }

   private:
    AsyncGrpcServerTransport* impl_;
    const CallbackServiceHelper& helper_;
    GrpcServerCallContext flight_context_;
  };

  auto* reactor = new Reactor(context, impl_, helper_);
  reactor->Start();
  return reactor;
}

::grpc::ServerWriteReactor<pb::FlightInfo>* CallbackFlightService::ListFlights(
    ::grpc::CallbackServerContext* context, const pb::Criteria* request) {
  GrpcServerCallContext flight_context(context);
  auto st =
      PrepareAuthenticatedCall(helper_, FlightMethod::ListFlights, context, &flight_context);
  if (!st.ok()) {
    return FinishWriteNow<pb::FlightInfo>(st);
  }

  Criteria criteria;
  if (request) {
    auto conv = internal::FromProto(*request, &criteria);
    if (!conv.ok()) return FinishWriteNow<pb::FlightInfo>(flight_context.FinishRequest(conv));
  }
  auto maybe_listing = impl_->base()->ListFlights(flight_context, &criteria);
  if (!maybe_listing.status().ok()) {
    return FinishWriteNow<pb::FlightInfo>(flight_context.FinishRequest(maybe_listing.status()));
  }
  auto listing =
      std::make_shared<std::unique_ptr<FlightListing>>(
          std::move(maybe_listing).MoveResult().ValueUnsafe());
  auto* reactor = new IteratorReactor<pb::FlightInfo, FlightInfo>(
      context, impl_->executor(), std::move(flight_context),
      [listing]() mutable {
        return *listing ? (*listing)->Next()
                       : arrow::Result<std::unique_ptr<FlightInfo>>(
                             std::unique_ptr<FlightInfo>{});
      },
      [](const FlightInfo& info, pb::FlightInfo* out) { return internal::ToProto(info, out); });
  reactor->Start();
  return reactor;
}

::grpc::ServerUnaryReactor* CallbackFlightService::GetFlightInfo(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::FlightInfo* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::GetFlightInfo, context,
                                     &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  FlightDescriptor descr;
  auto arrow_st = ParseRequiredRequest(request, "FlightDescriptor", &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  impl_->base()->GetFlightInfo(flight_context, descr).AddCallback(
      [reactor, response, flight_context = std::move(flight_context)](
          const arrow::Result<std::unique_ptr<FlightInfo>>& result) mutable {
        FinishUnaryResult(
            reactor, response, std::move(flight_context), result, "Flight not found",
            [](const FlightInfo& info, pb::FlightInfo* out) {
              return internal::ToProto(info, out);
            });
      });
  return reactor;
}

::grpc::ServerUnaryReactor* CallbackFlightService::PollFlightInfo(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::PollInfo* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::PollFlightInfo, context,
                                     &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  FlightDescriptor descr;
  auto arrow_st = ParseRequiredRequest(request, "FlightDescriptor", &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  impl_->base()->PollFlightInfo(flight_context, descr).AddCallback(
      [reactor, response, flight_context = std::move(flight_context)](
          const arrow::Result<std::unique_ptr<PollInfo>>& result) mutable {
        FinishUnaryResult(
            reactor, response, std::move(flight_context), result, "Flight not found",
            [](const PollInfo& info, pb::PollInfo* out) {
              return internal::ToProto(info, out);
            });
      });
  return reactor;
}

::grpc::ServerUnaryReactor* CallbackFlightService::GetSchema(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::SchemaResult* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  auto st =
      PrepareAuthenticatedCall(helper_, FlightMethod::GetSchema, context, &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  FlightDescriptor descr;
  auto arrow_st = ParseRequiredRequest(request, "FlightDescriptor", &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  impl_->base()->GetSchema(flight_context, descr).AddCallback(
      [reactor, response, flight_context = std::move(flight_context)](
          const arrow::Result<std::unique_ptr<SchemaResult>>& result) mutable {
        FinishUnaryResult(
            reactor, response, std::move(flight_context), result, "Flight not found",
            [](const SchemaResult& schema, pb::SchemaResult* out) {
              return internal::ToProto(schema, out);
            });
      });
  return reactor;
}

::grpc::ServerWriteReactor<pb::FlightData>* CallbackFlightService::DoGet(
    ::grpc::CallbackServerContext* context, const pb::Ticket* request) {
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::DoGet, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::FlightData>(st);
  Ticket ticket;
  auto arrow_st = ParseRequiredRequest(request, "ticket", &ticket);
  if (!arrow_st.ok()) {
    return FinishWriteNow<pb::FlightData>(flight_context.FinishRequest(arrow_st));
  }
  auto maybe_stream = impl_->base()->DoGet(flight_context, ticket);
  if (!maybe_stream.status().ok()) {
    return FinishWriteNow<pb::FlightData>(flight_context.FinishRequest(maybe_stream.status()));
  }
  auto* reactor = new DoGetReactor(context, impl_->executor(), std::move(flight_context),
                                   std::move(maybe_stream).MoveResult().ValueUnsafe());
  reactor->Start();
  return reactor;
}

::grpc::ServerBidiReactor<pb::FlightData, pb::PutResult>* CallbackFlightService::DoPut(
    ::grpc::CallbackServerContext* context) {
  class Reactor final : public BidiReactorBase<pb::FlightData, pb::PutResult> {
   public:
    Reactor(::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
            const CallbackServiceHelper& helper)
        : BidiReactorBase(context, impl->executor()),
          impl_(impl),
          helper_(helper),
          flight_context_(context) {}

    void Start() {
      auto st = PrepareAuthenticatedCall(helper_, FlightMethod::DoPut, this->context_,
                                         &flight_context_);
      if (!st.ok()) {
        this->Finish(st);
        return;
      }
      this->Hold();
      impl_->MakeAsyncMessageReader(this).AddCallback(
          [this](const ::arrow::Result<std::unique_ptr<AsyncFlightMessageReader>>& maybe_reader)
              mutable {
            if (!maybe_reader.ok()) {
              this->Finish(flight_context_.FinishRequest(maybe_reader.status()));
              this->ReleaseHold();
              return;
            }
            auto writer = impl_->MakeAsyncMetadataWriter(this);
            auto reader = std::move(
                const_cast<::arrow::Result<std::unique_ptr<AsyncFlightMessageReader>>&>(
                    maybe_reader))
                              .MoveValueUnsafe();
            impl_->base()
                ->DoPutAsync(flight_context_, std::move(reader), std::move(writer))
                .AddCallback(
                    [this](const ::arrow::Result<::arrow::internal::Empty>& result) mutable {
                      this->Finish(flight_context_.FinishRequest(result.status()));
                      this->ReleaseHold();
                    });
          });
    }

   private:
    AsyncGrpcServerTransport* impl_;
    const CallbackServiceHelper& helper_;
    GrpcServerCallContext flight_context_;
  };
  auto* reactor = new Reactor(context, impl_, helper_);
  reactor->Start();
  return reactor;
}

::grpc::ServerBidiReactor<pb::FlightData, pb::FlightData>*
CallbackFlightService::DoExchange(::grpc::CallbackServerContext* context) {
  class Reactor final : public BidiReactorBase<pb::FlightData, pb::FlightData> {
   public:
    Reactor(::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
            const CallbackServiceHelper& helper)
        : BidiReactorBase(context, impl->executor()),
          impl_(impl),
          helper_(helper),
          flight_context_(context) {}

    void Start() {
      auto st = PrepareAuthenticatedCall(helper_, FlightMethod::DoExchange,
                                         this->context_, &flight_context_);
      if (!st.ok()) {
        this->Finish(st);
        return;
      }
      this->Hold();
      impl_->MakeAsyncMessageReader(this).AddCallback(
          [this](const ::arrow::Result<std::unique_ptr<AsyncFlightMessageReader>>& maybe_reader)
              mutable {
            if (!maybe_reader.ok()) {
              this->Finish(flight_context_.FinishRequest(maybe_reader.status()));
              this->ReleaseHold();
              return;
            }
            auto writer = impl_->MakeAsyncMessageWriter(this);
            auto reader = std::move(
                const_cast<::arrow::Result<std::unique_ptr<AsyncFlightMessageReader>>&>(
                    maybe_reader))
                              .MoveValueUnsafe();
            impl_->base()
                ->DoExchangeAsync(flight_context_, std::move(reader), std::move(writer))
                .AddCallback(
                    [this](const ::arrow::Result<::arrow::internal::Empty>& result) mutable {
                      this->Finish(flight_context_.FinishRequest(result.status()));
                      this->ReleaseHold();
                    });
          });
    }

   private:
    AsyncGrpcServerTransport* impl_;
    const CallbackServiceHelper& helper_;
    GrpcServerCallContext flight_context_;
  };
  auto* reactor = new Reactor(context, impl_, helper_);
  reactor->Start();
  return reactor;
}

::grpc::ServerWriteReactor<pb::ActionType>* CallbackFlightService::ListActions(
    ::grpc::CallbackServerContext* context, const pb::Empty*) {
  GrpcServerCallContext flight_context(context);
  auto st =
      PrepareAuthenticatedCall(helper_, FlightMethod::ListActions, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::ActionType>(st);
  auto maybe_actions = impl_->base()->ListActions(flight_context);
  if (!maybe_actions.status().ok()) {
    return FinishWriteNow<pb::ActionType>(
        flight_context.FinishRequest(maybe_actions.status()));
  }
  auto actions = std::move(maybe_actions).MoveResult().ValueUnsafe();
  auto next = [actions = std::move(actions), index = size_t{0}]() mutable
      -> arrow::Result<std::unique_ptr<ActionType>> {
    if (index >= actions.size()) return nullptr;
    return std::make_unique<ActionType>(actions[index++]);
  };
  auto* reactor = new IteratorReactor<pb::ActionType, ActionType>(
      context, impl_->executor(), std::move(flight_context), std::move(next),
      [](const ActionType& action, pb::ActionType* out) {
        return internal::ToProto(action, out);
      });
  reactor->Start();
  return reactor;
}

::grpc::ServerWriteReactor<pb::Result>* CallbackFlightService::DoAction(
    ::grpc::CallbackServerContext* context, const pb::Action* request) {
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::DoAction, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::Result>(st);
  Action action;
  auto arrow_st = ParseRequiredRequest(request, "Action", &action);
  if (!arrow_st.ok()) {
    return FinishWriteNow<pb::Result>(flight_context.FinishRequest(arrow_st));
  }
  auto maybe_results = impl_->base()->DoAction(flight_context, action);
  if (!maybe_results.status().ok()) {
    return FinishWriteNow<pb::Result>(flight_context.FinishRequest(maybe_results.status()));
  }
  auto results =
      std::make_shared<std::unique_ptr<ResultStream>>(
          std::move(maybe_results).MoveResult().ValueUnsafe());
  if (!*results) {
    return FinishWriteNow<pb::Result>(
        flight_context.FinishRequest(::grpc::Status::CANCELLED));
  }
  auto* reactor = new IteratorReactor<pb::Result, Result>(
      context, impl_->executor(), std::move(flight_context),
      [results]() mutable {
        return *results ? (*results)->Next()
                       : arrow::Result<std::unique_ptr<arrow::flight::Result>>(
                             std::unique_ptr<arrow::flight::Result>{});
      },
      [](const Result& result, pb::Result* out) { return internal::ToProto(result, out); });
  reactor->Start();
  return reactor;
}

}  // namespace

arrow::Result<std::unique_ptr<internal::AsyncServerTransport>>
MakeGrpcCallbackServerTransport(AsyncFlightServerBase* base,
                                std::shared_ptr<MemoryManager> memory_manager) {
  return std::unique_ptr<internal::AsyncServerTransport>(
      new AsyncGrpcServerTransport(base, std::move(memory_manager)));
}

}  // namespace arrow::flight
