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

#include "arrow/array.h"
#include "arrow/array/data.h"
#include "arrow/buffer.h"
#include "arrow/compare.h"
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
#include "arrow/ipc/dictionary.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/uri.h"

namespace arrow::flight {
namespace {

namespace pb = arrow::flight::protocol;
using FlightService = pb::FlightService;

using GrpcServerCallContext =
    transport::grpc::GrpcServerCallContext<::grpc::CallbackServerContext>;
using CallbackServiceHelper =
    transport::grpc::GrpcServerCallContextHelper<::grpc::CallbackServerContext>;

constexpr int kMaxAsyncGrpcWorkerThreads = 4;

/// Create the bounded worker pool used only for blocking compatibility paths.
arrow::Result<std::shared_ptr<arrow::internal::ThreadPool>> MakeAsyncGrpcExecutor() {
  return arrow::internal::ThreadPool::MakeEternal(std::min(
      kMaxAsyncGrpcWorkerThreads, arrow::internal::ThreadPool::DefaultCapacity()));
}

/// Authenticate an RPC and install middleware response headers.
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
/// Validate and convert a required protobuf request message.
Status ParseRequiredRequest(const ProtoRequest* request, const char* request_name,
                            ArrowRequest* out) {
  if (request == nullptr) {
    return Status::Invalid(request_name, " cannot be null");
  }
  return internal::FromProto(*request, out);
}

template <typename ArrowResponse, typename ProtoResponse, typename ToProtoFn>
/// Serialize a unary Arrow result and finish the gRPC reactor.
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
  /// Finish a server-streaming RPC without producing a message.
  explicit ImmediateWriteReactor(::grpc::Status status) {
    this->Finish(std::move(status));
  }
  /// Delete this self-owned immediate reactor once gRPC releases it.
  void OnDone() override { delete this; }
};

/// Shared state machine for callback bidi RPCs, including synchronous handshake
/// compatibility and future-based Flight data reads/writes.
template <typename Req, typename Resp>
class BidiReactorBase : public ::grpc::ServerBidiReactor<Req, Resp> {
 public:
  using ReadValue =
      std::conditional_t<std::is_same_v<Req, pb::FlightData>, internal::FlightData, Req>;
  using WriteValue =
      std::conditional_t<std::is_same_v<Resp, pb::FlightData>, FlightPayload, Resp>;
  using AsyncReadValue =
      std::conditional_t<std::is_same_v<Req, pb::FlightData>,
                         std::shared_ptr<internal::FlightData>, std::optional<Req>>;

  BidiReactorBase(::grpc::CallbackServerContext* context,
                  std::shared_ptr<arrow::internal::ThreadPool> executor)
      : context_(context), executor_(std::move(executor)) {}

  /// Arm the first read and run a blocking compatibility handler on the pool.
  Status StartWorker(std::function<void()> fn) {
    // For bidi RPCs the first read must be armed explicitly. After that, reads
    // are re-armed only from OnReadDone() so the callback path owns the
    // receive-side state machine.
    {
      std::unique_lock<std::mutex> lock(mutex_);
      read_in_flight_ = true;
    }
    this->StartRead(GrpcReadBuffer());
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

  /// Deliver one completed read or stream end to the pending async read.
  void OnReadDone(bool ok) override {
    std::optional<ReadValue> completed_read;
    Future<AsyncReadValue> pending_future;
    bool resolve_pending = false;
    bool start_next_read = false;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      read_in_flight_ = false;
      if (ok) {
        completed_read.emplace(std::move(read_buffer_));
        if (pending_read_active_) {
          pending_future = pending_read_;
          pending_read_active_ = false;
          pending_read_ = Future<AsyncReadValue>();
          resolve_pending = true;
          // gRPC requires the next read to be started from the completion
          // callback. This preserves that requirement while bounding prefetch
          // to one message when the application stops reading.
          if (!cancelled_) {
            read_in_flight_ = true;
            start_next_read = true;
          }
        } else {
          reads_.push_back(std::move(*completed_read));
        }
      } else {
        reads_done_ = true;
        if (pending_read_active_) {
          pending_future = pending_read_;
          pending_read_active_ = false;
          pending_read_ = Future<AsyncReadValue>();
          resolve_pending = true;
        }
      }
    }

    if (start_next_read) {
      this->StartRead(GrpcReadBuffer());
    }
    if (resolve_pending) {
      if (ok) {
        pending_future.MarkFinished(MakeAsyncReadValue(std::move(*completed_read)));
      } else {
        pending_future.MarkFinished(EndAsyncReadValue());
      }
    }
    cv_.notify_all();
  }

  /// Complete the pending write and finish a deferred RPC if necessary.
  void OnWriteDone(bool ok) override {
    Future<bool> pending_future;
    bool resolve_pending = false;
    bool finish_now = false;
    bool cancelled = false;
    ::grpc::Status finish_status;
    std::unique_lock<std::mutex> lock(mutex_);
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
      finish_status = finish_status_;
    }
    cancelled = cancelled_;
    lock.unlock();
    if (resolve_pending) {
      pending_future.MarkFinished(!cancelled && ok);
    }
    if (finish_now) {
      this->Finish(finish_status);
    }
    cv_.notify_all();
  }

  /// Wake pending reads and writes when gRPC cancels the RPC.
  void OnCancel() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cancelled_ = true;
    Future<bool> write_future;
    bool resolve_write = false;
    if (pending_write_active_) {
      write_future = pending_write_;
      pending_write_active_ = false;
      pending_write_ = Future<bool>();
      resolve_write = true;
    }
    if (pending_read_active_) {
      auto future = pending_read_;
      pending_read_active_ = false;
      pending_read_ = Future<AsyncReadValue>();
      lock.unlock();
      future.MarkFinished(EndAsyncReadValue());
      if (resolve_write) {
        write_future.MarkFinished(false);
      }
      cv_.notify_all();
      return;
    }
    lock.unlock();
    if (resolve_write) {
      write_future.MarkFinished(false);
    }
    cv_.notify_all();
  }

  /// Release gRPC's ownership reference.
  void OnDone() override { ReleaseRef(); }

  /// Synchronously read one message for legacy handshake authentication.
  bool ReadOne(Req* out) { return PopRead(out); }

  /// Synchronously write one message for legacy handshake authentication.
  bool WriteOnePublic(Resp message) { return WriteOne(std::move(message)); }
  /// Return the next inbound message while enforcing one outstanding read.
  Future<AsyncReadValue> ReadOneAsync() {
    bool start_read = false;
    std::unique_lock<std::mutex> lock(mutex_);
    if (!reads_.empty()) {
      ReadValue out = std::move(reads_.front());
      reads_.pop_front();
      if (!read_in_flight_ && !reads_done_ && !cancelled_) {
        read_in_flight_ = true;
        start_read = true;
      }
      auto future =
          Future<AsyncReadValue>::MakeFinished(MakeAsyncReadValue(std::move(out)));
      lock.unlock();
      if (start_read) {
        this->StartRead(GrpcReadBuffer());
      }
      return future;
    }
    if (cancelled_ || reads_done_) {
      return Future<AsyncReadValue>::MakeFinished(EndAsyncReadValue());
    }
    if (pending_read_active_) {
      return Future<AsyncReadValue>::MakeFinished(
          Status::Invalid("Concurrent async reads are not supported"));
    }
    pending_read_ = Future<AsyncReadValue>::Make();
    pending_read_active_ = true;
    if (!read_in_flight_) {
      // The application controls receive-side backpressure. Keep at most one
      // completed message buffered when it is not currently awaiting a read.
      read_in_flight_ = true;
      start_read = true;
    }
    auto future = pending_read_;
    lock.unlock();
    if (start_read) {
      this->StartRead(GrpcReadBuffer());
    }
    return future;
  }

  /// Start an asynchronous protobuf response write.
  Future<bool> WriteOneAsync(Resp message) { return StartAsyncWrite(std::move(message)); }

  /// Validate and start an asynchronous FlightData response write.
  Future<bool> WritePayloadAsync(FlightPayload payload) {
    static_assert(std::is_same_v<Resp, pb::FlightData>);
    RETURN_NOT_OK(payload.Validate());
    return StartAsyncWrite(std::move(payload));
  }

  /// Synchronously write a Flight payload for legacy server streams.
  arrow::Result<bool> WritePayloadPublic(FlightPayload payload) {
    static_assert(std::is_same_v<Resp, pb::FlightData>);
    RETURN_NOT_OK(payload.Validate());
    std::unique_lock<std::mutex> lock(mutex_);
    if (cancelled_) return false;
    current_write_ = std::move(payload);
    write_in_flight_ = true;
    write_ok_ = true;
    this->StartWrite(GrpcWriteBuffer());
    cv_.wait(lock, [&] { return !write_in_flight_ || cancelled_; });
    return !cancelled_ && write_ok_;
  }

  /// Retain this self-owned reactor across an asynchronous callback.
  void Hold() { refs_.fetch_add(1, std::memory_order_relaxed); }
  /// Release a reference acquired with Hold().
  void ReleaseHold() { ReleaseRef(); }

 protected:
  /// Install the single outstanding async write and hand it to gRPC.
  Future<bool> StartAsyncWrite(WriteValue message) {
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
    this->StartWrite(GrpcWriteBuffer());
    return pending_write_;
  }
  /// Block a compatibility caller until a queued read or stream end is available.
  bool PopRead(Req* out) {
    bool start_read = false;
    std::unique_lock<std::mutex> lock(mutex_);
    if (reads_.empty() && !cancelled_ && !reads_done_ && !read_in_flight_) {
      read_in_flight_ = true;
      start_read = true;
    }
    if (start_read) {
      lock.unlock();
      this->StartRead(GrpcReadBuffer());
      lock.lock();
    }
    cv_.wait(lock, [&] { return cancelled_ || reads_done_ || !reads_.empty(); });
    if (!reads_.empty()) {
      *out = std::move(reads_.front());
      reads_.pop_front();
      return true;
    }
    return false;
  }

  /// Block a compatibility caller until a gRPC write completes.
  bool WriteOne(Resp message) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cancelled_) return false;
    current_write_ = std::move(message);
    write_in_flight_ = true;
    write_ok_ = true;
    this->StartWrite(GrpcWriteBuffer());
    cv_.wait(lock, [&] { return !write_in_flight_ || cancelled_; });
    return !cancelled_ && write_ok_;
  }

  /// Finish after an active response write, preserving gRPC write ordering.
  void FinishFromWorker(::grpc::Status status) {
    bool finish_now = false;
    ::grpc::Status finish_status;
    std::unique_lock<std::mutex> lock(mutex_);
    finish_requested_ = true;
    finish_status_ = std::move(status);
    if (!write_in_flight_) {
      finish_now = true;
      finish_status = finish_status_;
    }
    lock.unlock();
    if (finish_now) {
      this->Finish(finish_status);
    }
  }

  /// Delete this reactor when both gRPC and background work have released it.
  void ReleaseRef() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  /// Return the protobuf storage used by gRPC for the next inbound message.
  Req* GrpcReadBuffer() {
    if constexpr (std::is_same_v<Req, pb::FlightData>) {
      return reinterpret_cast<Req*>(&read_buffer_);
    } else {
      return &read_buffer_;
    }
  }

  /// Return the protobuf storage used by gRPC for the active outbound message.
  Resp* GrpcWriteBuffer() {
    if constexpr (std::is_same_v<Resp, pb::FlightData>) {
      return reinterpret_cast<Resp*>(&current_write_);
    } else {
      return &current_write_;
    }
  }

  /// Convert a transport read value to the public async representation.
  AsyncReadValue MakeAsyncReadValue(ReadValue value) {
    if constexpr (std::is_same_v<Req, pb::FlightData>) {
      return std::make_shared<internal::FlightData>(std::move(value));
    } else {
      return std::optional<Req>(std::move(value));
    }
  }

  /// Return the public end-of-stream representation for this request type.
  AsyncReadValue EndAsyncReadValue() {
    if constexpr (std::is_same_v<Req, pb::FlightData>) {
      return nullptr;
    } else {
      return std::optional<Req>{};
    }
  }

  ::grpc::CallbackServerContext* context_;
  std::shared_ptr<arrow::internal::ThreadPool> executor_;
  ReadValue read_buffer_;
  std::deque<ReadValue> reads_;
  WriteValue current_write_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool reads_done_ = false;
  Future<AsyncReadValue> pending_read_;
  bool pending_read_active_ = false;
  bool read_in_flight_ = false;
  bool cancelled_ = false;
  Future<bool> pending_write_;
  bool pending_write_active_ = false;
  bool write_in_flight_ = false;
  bool write_ok_ = true;
  bool finish_requested_ = false;
  ::grpc::Status finish_status_;
  std::atomic<int> refs_{1};
};

template <typename ReactorT>
/// Finish a unary reactor and return it to gRPC.
ReactorT* FinishNow(ReactorT* reactor, const ::grpc::Status& status) {
  reactor->Finish(status);
  return reactor;
}

template <typename Proto>
/// Allocate and finish a server-streaming reactor without output.
::grpc::ServerWriteReactor<Proto>* FinishWriteNow(const ::grpc::Status& status) {
  return new ImmediateWriteReactor<Proto>(status);
}

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

/// Decodes inbound FlightData frames into the public async reader interface.
class NativeAsyncFlightMessageReader final
    : public AsyncFlightMessageReader,
      public std::enable_shared_from_this<NativeAsyncFlightMessageReader> {
 public:
  using ReadFn = std::function<Future<std::shared_ptr<internal::FlightData>>()>;
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
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (operation_in_flight_) {
        return Future<std::shared_ptr<Schema>>::MakeFinished(
            Status::Invalid("Concurrent async reads are not supported"));
      }
      operation_in_flight_ = true;
      idle_ = Future<>::Make();
      active_operation_ = ActiveOperation::kSchema;
      token = ++active_token_;
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
      if (operation_in_flight_) {
        return Future<FlightStreamChunk>::MakeFinished(
            Status::Invalid("Concurrent async reads are not supported"));
      }
      operation_in_flight_ = true;
      idle_ = Future<>::Make();
      active_operation_ = ActiveOperation::kNext;
      token = ++active_token_;
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
                               std::shared_ptr<internal::FlightData> message)
                               -> ::arrow::Result<std::shared_ptr<internal::FlightData>> {
      if (!message) {
        return std::shared_ptr<internal::FlightData>{};
      }
      auto data = std::move(*message);
      if (data.body) {
        ARROW_ASSIGN_OR_RAISE(data.body, Buffer::ViewOrCopy(data.body, memory_manager));
      }
      return std::make_shared<internal::FlightData>(std::move(data));
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

  /// Check whether a callback still belongs to the current read operation.
  bool IsCurrentOperation(int64_t token, ActiveOperation operation) {
    std::lock_guard<std::mutex> guard(mutex_);
    return operation_in_flight_ && active_token_ == token &&
           active_operation_ == operation;
  }

  void FinishSchema(Future<std::shared_ptr<Schema>> out,
                    ::arrow::Result<std::shared_ptr<Schema>> result, int64_t token) {
    Future<> idle;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (active_token_ != token || active_operation_ != ActiveOperation::kSchema) {
        return;
      }
      operation_in_flight_ = false;
      active_operation_ = ActiveOperation::kNone;
      idle = idle_;
    }
    out.MarkFinished(std::move(result));
    idle.MarkFinished();
  }

  /// Read frames until the decoder produces a schema or the stream fails.
  void PumpSchema(Future<std::shared_ptr<Schema>> out, int64_t token) {
    if (listener_->schema()) {
      FinishSchema(out, listener_->schema(), token);
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
            self->FinishSchema(out, result.status(), token);
            return;
          }
          auto maybe_data = result.ValueUnsafe();
          if (!maybe_data) {
            self->FinishSchema(out, Status::IOError("Client never sent a data message"),
                               token);
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
            self->FinishSchema(out, st, token);
            return;
          }
          self->PumpSchema(out, token);
        });
  }

  void FinishNext(Future<FlightStreamChunk> out,
                  ::arrow::Result<FlightStreamChunk> result, int64_t token) {
    Future<> idle;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (active_token_ != token || active_operation_ != ActiveOperation::kNext) {
        return;
      }
      operation_in_flight_ = false;
      active_operation_ = ActiveOperation::kNone;
      idle = idle_;
    }
    out.MarkFinished(std::move(result));
    idle.MarkFinished();
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
      FinishNext(out, std::move(*ready_chunk), token);
      return;
    }
    if (stream_finished) {
      FinishNext(out, FlightStreamChunk{}, token);
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
            self->FinishNext(out, result.status(), token);
            return;
          }
          auto maybe_data = result.ValueUnsafe();
          if (!maybe_data) {
            {
              std::lock_guard<std::mutex> guard(self->mutex_);
              self->finished_ = true;
            }
            self->FinishNext(out, FlightStreamChunk{}, token);
            return;
          }
          if (!maybe_data->metadata) {
            if (!maybe_data->app_metadata) {
              self->PumpNext(out, token);
              return;
            }
            FlightStreamChunk chunk;
            chunk.app_metadata = std::move(maybe_data->app_metadata);
            self->FinishNext(out, std::move(chunk), token);
            return;
          }
          auto st = self->ConsumeDataMessage(*maybe_data);
          if (!st.ok()) {
            self->FinishNext(out, st, token);
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
  bool operation_in_flight_ = false;
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
      if (closed_) {
        return Future<>::MakeFinished(Status::Invalid("This writer is already closed"));
      }
      if (failed_) {
        return Future<>::MakeFinished(failure_status_);
      }
    }
    FlightPayload payload;
    payload.app_metadata = std::move(app_metadata);
    return WritePayloads({std::move(payload)});
  }

  /// Write dictionary updates, a record batch, and optional metadata in order.
  Future<> WriteWithMetadata(const RecordBatch& batch,
                             std::shared_ptr<Buffer> app_metadata) override {
    std::vector<FlightPayload> payloads;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      RETURN_NOT_OK(CheckStartedLocked());
      RETURN_NOT_OK(ReserveWriteLocked());
      auto status = BuildDictionaryPayloadsLocked(batch, &payloads);
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
  /// Validate that batch writes are legal while mutex_ is held.
  Status CheckStartedLocked() const {
    if (failed_) {
      return failure_status_;
    }
    if (!begun_) {
      return Status::Invalid("This writer is not started. Call Begin() with a schema");
    }
    if (closed_) {
      return Status::Invalid("This writer is already closed");
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
  Status BuildDictionaryPayloadsLocked(const RecordBatch& batch,
                                       std::vector<FlightPayload>* payloads) {
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
      {
        std::lock_guard<std::mutex> guard(mutex_);
        write_in_flight_ = false;
      }
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

class AsyncGrpcServerTransport;

/// Maps generated gRPC callback methods to AsyncFlightServerBase hooks.
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
  ::grpc::ServerWriteReactor<pb::FlightData>* DoGet(
      ::grpc::CallbackServerContext* context, const pb::Ticket* request) override;
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

/// Owns the callback gRPC server and constructs transport-specific async streams.
class AsyncGrpcServerTransport : public internal::AsyncServerTransport {
 public:
  struct AsyncMessageReader {
    std::unique_ptr<AsyncFlightMessageReader> reader;
    std::shared_ptr<NativeAsyncFlightMessageReader> impl;

    /// Transfer public reader ownership to the application hook.
    std::unique_ptr<AsyncFlightMessageReader> TakeReader() { return std::move(reader); }
    /// Wait for a reader operation retained by the transport to complete.
    Future<> WhenIdle() const { return impl->WhenIdle(); }
  };
  AsyncGrpcServerTransport(AsyncFlightServerBase* base,
                           std::shared_ptr<MemoryManager> memory_manager)
      : internal::AsyncServerTransport(base, std::move(memory_manager)) {}

  /// Configure and start the callback gRPC server for the requested URI.
  Status Init(const FlightServerOptions& options, const arrow::util::Uri& uri) override {
    ARROW_ASSIGN_OR_RAISE(executor_pool_, MakeAsyncGrpcExecutor());
    helper_ =
        std::make_unique<CallbackServiceHelper>(options.auth_handler, options.middleware);
    grpc_service_ = std::make_unique<CallbackFlightService>(this, *helper_);

    return transport::grpc::StartFlightGrpcServer(options, uri, grpc_service_.get(),
                                                  /*callback_api=*/true, &grpc_server_,
                                                  &location_);
  }

  /// Shut down the gRPC server without a deadline.
  Status Shutdown() override {
    grpc_server_->Shutdown();
    return Status::OK();
  }
  /// Shut down the gRPC server, allowing active work until deadline.
  Status Shutdown(const std::chrono::system_clock::time_point& deadline) override {
    grpc_server_->Shutdown(deadline);
    return Status::OK();
  }
  /// Wait for gRPC shutdown to complete.
  Status Wait() override {
    grpc_server_->Wait();
    return Status::OK();
  }
  Location location() const override { return location_; }

  const CallbackServiceHelper& helper() const { return *helper_; }
  std::shared_ptr<arrow::internal::ThreadPool> executor() const { return executor_pool_; }

  /// Read and validate the first FlightData frame before invoking a bidi hook.
  template <typename Resp>
  Future<AsyncMessageReader> MakeAsyncMessageReader(
      BidiReactorBase<pb::FlightData, Resp>* reactor) {
    return reactor->ReadOneAsync().Then([this, reactor](
                                            std::shared_ptr<internal::FlightData> message)
                                            -> ::arrow::Result<AsyncMessageReader> {
      if (!message) {
        return Status::IOError("Stream finished before first message sent");
      }
      auto data = std::move(*message);
      if (data.body) {
        ARROW_ASSIGN_OR_RAISE(data.body, Buffer::ViewOrCopy(data.body, memory_manager_));
      }
      if (!data.descriptor) {
        return Status::IOError("Descriptor missing on first message");
      }
      auto descriptor = *data.descriptor;
      auto impl = std::make_shared<NativeAsyncFlightMessageReader>(
          std::move(descriptor), std::move(data),
          [reactor]() { return reactor->ReadOneAsync(); }, memory_manager_);
      return AsyncMessageReader{std::make_unique<SharedAsyncFlightMessageReader>(impl),
                                std::move(impl)};
    });
  }

  /// Build a DoPut metadata writer backed by this bidi reactor.
  std::unique_ptr<AsyncFlightMetadataWriter> MakeAsyncMetadataWriter(
      BidiReactorBase<pb::FlightData, pb::PutResult>* reactor) {
    return std::unique_ptr<AsyncFlightMetadataWriter>(
        new NativeAsyncFlightMetadataWriter([reactor](pb::PutResult result) {
          return reactor->WriteOneAsync(std::move(result));
        }));
  }

  /// Build a DoExchange message writer backed by this bidi reactor.
  std::unique_ptr<AsyncFlightMessageWriter> MakeAsyncMessageWriter(
      BidiReactorBase<pb::FlightData, pb::FlightData>* reactor) {
    auto impl = std::make_shared<NativeAsyncFlightMessageWriter>(
        [reactor](FlightPayload payload) {
          return reactor->WritePayloadAsync(std::move(payload));
        });
    return std::make_unique<SharedAsyncFlightMessageWriter>(std::move(impl));
  }

 private:
  std::shared_ptr<arrow::internal::ThreadPool> executor_pool_;
  std::unique_ptr<CallbackServiceHelper> helper_;
  std::unique_ptr<CallbackFlightService> grpc_service_;
  std::unique_ptr<::grpc::Server> grpc_server_;
  Location location_;
};

/// Dispatches DoPut or DoExchange after authentication and the first input frame.
template <typename Resp>
class AsyncBidiFlightReactor final : public BidiReactorBase<pb::FlightData, Resp> {
 public:
  AsyncBidiFlightReactor(::grpc::CallbackServerContext* context,
                         AsyncGrpcServerTransport* impl,
                         const CallbackServiceHelper& helper)
      : BidiReactorBase<pb::FlightData, Resp>(context, impl->executor()),
        impl_(impl),
        helper_(helper),
        flight_context_(context) {}

  /// Authenticate the RPC and asynchronously acquire its public reader.
  void Start() {
    constexpr auto kMethod = std::is_same_v<Resp, pb::PutResult>
                                 ? FlightMethod::DoPut
                                 : FlightMethod::DoExchange;
    auto st =
        PrepareAuthenticatedCall(helper_, kMethod, this->context_, &flight_context_);
    if (!st.ok()) {
      this->Finish(st);
      return;
    }
    this->Hold();
    impl_->MakeAsyncMessageReader(this).AddCallback(
        [this](const ::arrow::Result<AsyncGrpcServerTransport::AsyncMessageReader>&
                   maybe_reader) { HandleReader(maybe_reader); });
  }

 private:
  /// Invoke the selected server hook or finish an invalid inbound stream.
  void HandleReader(
      const ::arrow::Result<AsyncGrpcServerTransport::AsyncMessageReader>& result) {
    if (!result.ok()) {
      this->Finish(flight_context_.FinishRequest(result.status()));
      this->ReleaseHold();
      return;
    }
    auto reader_and_state =
        std::move(
            const_cast<::arrow::Result<AsyncGrpcServerTransport::AsyncMessageReader>&>(
                result))
            .MoveValueUnsafe();
    Future<> completion;
    if constexpr (std::is_same_v<Resp, pb::PutResult>) {
      completion = impl_->base()->DoPut(flight_context_, reader_and_state.TakeReader(),
                                        impl_->MakeAsyncMetadataWriter(this));
    } else {
      completion =
          impl_->base()->DoExchange(flight_context_, reader_and_state.TakeReader(),
                                    impl_->MakeAsyncMessageWriter(this));
    }
    FinishAfterReader(std::move(reader_and_state), std::move(completion));
  }

  /// Defer final gRPC completion until the application reader is idle.
  void FinishAfterReader(AsyncGrpcServerTransport::AsyncMessageReader reader,
                         Future<> completion) {
    completion.AddCallback(
        [this, reader = std::move(reader)](
            const ::arrow::Result<::arrow::internal::Empty>& result) mutable {
          auto status = flight_context_.FinishRequest(result.status());
          auto idle = reader.WhenIdle();
          idle.AddCallback([this, reader = std::move(reader), status = std::move(status)](
                               const ::arrow::Result<::arrow::internal::Empty>&) mutable {
            ARROW_UNUSED(reader);
            this->FinishFromWorker(std::move(status));
            this->ReleaseHold();
          });
        });
  }

  AsyncGrpcServerTransport* impl_;
  const CallbackServiceHelper& helper_;
  GrpcServerCallContext flight_context_;
};

/// Common lifetime, cancellation, and output storage for server-streaming RPCs.
template <typename Proto>
class AsyncWriteReactorBase : public ::grpc::ServerWriteReactor<Proto> {
 public:
  using WriteValue =
      std::conditional_t<std::is_same_v<Proto, pb::FlightData>, FlightPayload, Proto>;

  AsyncWriteReactorBase(::grpc::CallbackServerContext* context,
                        std::shared_ptr<arrow::internal::ThreadPool> executor,
                        GrpcServerCallContext flight_context)
      : context_(context),
        executor_(std::move(executor)),
        flight_context_(std::move(flight_context)) {}

  /// Remember gRPC cancellation for background producers.
  void OnCancel() override { cancelled_.store(true, std::memory_order_relaxed); }

  /// Release gRPC's ownership reference.
  void OnDone() override { ReleaseRef(); }

  const GrpcServerCallContext& flight_context() const { return flight_context_; }

 protected:
  template <typename Fn>
  /// Schedule blocking compatibility work while retaining the reactor.
  Status StartBackgroundWork(Fn&& fn) {
    refs_.fetch_add(1, std::memory_order_relaxed);
    auto maybe_future = executor_->Submit([this, fn = std::forward<Fn>(fn)]() mutable {
      fn();
      ReleaseRef();
    });
    if (!maybe_future.ok()) {
      ReleaseRef();
      return maybe_future.status();
    }
    return Status::OK();
  }

  /// Return whether gRPC has cancelled the RPC.
  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

  /// Finish at most once, even when multiple async operations fail concurrently.
  void FinishOnce(::grpc::Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      this->Finish(std::move(status));
    }
  }

  /// Return the protobuf storage submitted by StartWrite().
  Proto* GrpcWriteBuffer() {
    if constexpr (std::is_same_v<Proto, pb::FlightData>) {
      return reinterpret_cast<Proto*>(&current_write_);
    } else {
      return &current_write_;
    }
  }

  ::grpc::CallbackServerContext* context_;
  std::shared_ptr<arrow::internal::ThreadPool> executor_;
  GrpcServerCallContext flight_context_;
  WriteValue current_write_;
  std::mutex mutex_;

  /// Retain this self-owned reactor across an asynchronous callback.
  void Hold() { refs_.fetch_add(1, std::memory_order_relaxed); }
  /// Release a reference acquired with Hold().
  void ReleaseHold() { ReleaseRef(); }

 private:
  /// Delete this reactor when gRPC and background callbacks have released it.
  void ReleaseRef() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  std::atomic<bool> cancelled_{false};
  std::atomic<bool> finished_{false};
  std::atomic<int> refs_{1};
};

/// Streams a legacy pull iterator to a server-streaming gRPC response.
template <typename Proto, typename UserType>
class IteratorReactor : public AsyncWriteReactorBase<Proto> {
 public:
  using NextFn = std::function<arrow::Result<std::unique_ptr<UserType>>()>;
  using ToProtoFn = std::function<Status(const UserType&, Proto*)>;

  IteratorReactor(::grpc::CallbackServerContext* context,
                  std::shared_ptr<arrow::internal::ThreadPool> executor,
                  GrpcServerCallContext flight_context, NextFn next_fn,
                  ToProtoFn to_proto)
      : AsyncWriteReactorBase<Proto>(context, std::move(executor),
                                     std::move(flight_context)),
        next_fn_(std::move(next_fn)),
        to_proto_(std::move(to_proto)) {}

  IteratorReactor(::grpc::CallbackServerContext* context,
                  std::shared_ptr<arrow::internal::ThreadPool> executor,
                  GrpcServerCallContext flight_context, ToProtoFn to_proto)
      : AsyncWriteReactorBase<Proto>(context, std::move(executor),
                                     std::move(flight_context)),
        to_proto_(std::move(to_proto)) {}

  /// Begin producing values from an already-installed iterator.
  void Start() {
    auto status = ScheduleAdvance();
    if (!status.ok()) {
      this->FinishOnce(this->flight_context_.FinishRequest(status));
    }
  }

  template <typename T, typename MakeNextFn>
  /// Install an iterator once an asynchronous server hook has completed.
  void StartAfter(Future<T> future, MakeNextFn make_next_fn) {
    this->Hold();
    future.AddCallback([this, make_next_fn = std::move(make_next_fn)](
                           const arrow::Result<T>& result) mutable {
      if (!result.ok()) {
        this->FinishOnce(this->flight_context_.FinishRequest(result.status()));
      } else {
        auto value = std::move(const_cast<arrow::Result<T>&>(result)).MoveValueUnsafe();
        next_fn_ = make_next_fn(std::move(value));
        Start();
      }
      this->ReleaseHold();
    });
  }

  /// Advance after each successful response write.
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
  /// Pull, serialize, and start one response write on a worker thread.
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
      this->StartWrite(this->GrpcWriteBuffer());
    });
  }

  NextFn next_fn_;
  ToProtoFn to_proto_;
};

/// Streams an AsyncFlightDataStream to the DoGet gRPC response.
class DoGetReactor : public AsyncWriteReactorBase<pb::FlightData> {
 public:
  DoGetReactor(::grpc::CallbackServerContext* context,
               std::shared_ptr<arrow::internal::ThreadPool> executor,
               GrpcServerCallContext flight_context,
               std::unique_ptr<AsyncFlightDataStream> stream)
      : AsyncWriteReactorBase<pb::FlightData>(context, std::move(executor),
                                              std::move(flight_context)),
        stream_(std::move(stream)) {}

  /// Start stream progression after a source is available.
  void Start() {
    auto status = Advance();
    if (!status.ok()) {
      this->FinishOnce(this->flight_context_.FinishRequest(status));
    }
  }

  /// Install a source returned by AsyncFlightServerBase::DoGet.
  void StartAfter(Future<std::unique_ptr<AsyncFlightDataStream>> future) {
    this->Hold();
    future.AddCallback(
        [this](
            const arrow::Result<std::unique_ptr<AsyncFlightDataStream>>& result) mutable {
          if (!result.ok()) {
            this->FinishOnce(this->flight_context_.FinishRequest(result.status()));
          } else {
            stream_ =
                std::move(
                    const_cast<arrow::Result<std::unique_ptr<AsyncFlightDataStream>>&>(
                        result))
                    .MoveValueUnsafe();
            Start();
          }
          this->ReleaseHold();
        });
  }

  /// Continue with the next source operation after a successful write.
  void OnWriteDone(bool ok) override {
    if (!ok) {
      this->FinishOnce(this->flight_context_.FinishRequest(Status::OK()));
      return;
    }
    auto status = Advance();
    if (!status.ok()) {
      this->FinishOnce(this->flight_context_.FinishRequest(status));
    }
  }

  /// Release the application stream when gRPC cancels the RPC.
  void OnCancel() override {
    AsyncWriteReactorBase<pb::FlightData>::OnCancel();
    ARROW_UNUSED(CloseStream());
  }

  void OnDone() override { AsyncWriteReactorBase<pb::FlightData>::OnDone(); }

 private:
  enum class Stage { kSchema, kPayloads, kFinish };

  /// Select the next schema, payload, or close operation from the source.
  Status Advance() {
    if (this->cancelled()) return Status::OK();
    if (!stream_) {
      this->FinishOnce(this->flight_context_.FinishRequest(
          Status::KeyError("No data in this flight")));
      return Status::OK();
    }
    if (stage_ == Stage::kSchema) {
      stage_ = Stage::kPayloads;
      return ReadSchema();
    }
    if (stage_ == Stage::kPayloads) {
      return ReadNext();
    }
    return CloseStream();
  }

  /// Request the mandatory schema payload.
  Status ReadSchema() { return ReadPayload(stream_->GetSchemaPayload()); }

  /// Request the next payload after the schema.
  Status ReadNext() { return ReadPayload(stream_->Next()); }

  /// Close the source and finish the RPC with its close status.
  Status CloseStream() {
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (close_started_) return Status::OK();
      close_started_ = true;
    }
    this->Hold();
    stream_->Close().AddCallback(
        [this](const ::arrow::Result<::arrow::internal::Empty>& result) {
          this->FinishOnce(this->flight_context_.FinishRequest(result.status()));
          this->ReleaseHold();
        });
    return Status::OK();
  }

  /// Process a future payload, an error, or the end marker.
  Status ReadPayload(Future<FlightPayload> future) {
    this->Hold();
    future.AddCallback([this](const ::arrow::Result<FlightPayload>& result) {
      if (!result.ok()) {
        this->FinishOnce(this->flight_context_.FinishRequest(result.status()));
      } else {
        auto payload = result.ValueUnsafe();
        if (payload.ipc_message.metadata == nullptr) {
          stage_ = Stage::kFinish;
          auto status = Advance();
          if (!status.ok()) this->FinishOnce(this->flight_context_.FinishRequest(status));
        } else {
          auto status = StartPayloadWrite(std::move(payload));
          if (!status.ok()) this->FinishOnce(this->flight_context_.FinishRequest(status));
        }
      }
      this->ReleaseHold();
    });
    return Status::OK();
  }

  /// Validate and submit one FlightData payload to gRPC.
  Status StartPayloadWrite(FlightPayload payload) {
    RETURN_NOT_OK(payload.Validate());
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (this->cancelled()) {
        return Status::OK();
      }
      this->current_write_ = std::move(payload);
    }
    this->StartWrite(this->GrpcWriteBuffer());
    return Status::OK();
  }

  std::unique_ptr<AsyncFlightDataStream> stream_;
  Stage stage_ = Stage::kSchema;
  bool close_started_ = false;
};

/// Bridge gRPC handshake authentication to the configured synchronous or async hook.
::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>*
CallbackFlightService::Handshake(::grpc::CallbackServerContext* context) {
  class Reactor final
      : public BidiReactorBase<pb::HandshakeRequest, pb::HandshakeResponse> {
   public:
    Reactor(::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
            const CallbackServiceHelper& helper)
        : BidiReactorBase(context, impl->executor()),
          impl_(impl),
          helper_(helper),
          flight_context_(context) {}

    void Start() {
      auto st = helper_.MakeCallContext(FlightMethod::Handshake, this->context_,
                                        &flight_context_);
      if (!st.ok()) {
        this->Finish(st);
        return;
      }
      helper_.AddMiddlewareHeaders(this->context_, &flight_context_);
      auto status = this->StartWorker([this] {
          auto outgoing = std::make_unique<
              transport::grpc::GrpcServerAuthSender<pb::HandshakeResponse>>(
              [this](pb::HandshakeResponse response) {
                return this->WriteOnePublic(std::move(response));
              });
          auto incoming = std::make_unique<
              transport::grpc::GrpcServerAuthReader<pb::HandshakeRequest>>(
              [this](pb::HandshakeRequest* request) { return this->ReadOne(request); });
        if (helper_.auth_handler()) {
          auto status = helper_.auth_handler()->Authenticate(
              flight_context_, outgoing.get(), incoming.get());
          this->FinishFromWorker(flight_context_.FinishRequest(status));
        } else {

          this->Hold();
          impl_->base()
              ->Handshake(flight_context_, std::move(outgoing), std::move(incoming))
              .AddCallback([this](
                               const ::arrow::Result<::arrow::internal::Empty>& result) {
                this->FinishFromWorker(flight_context_.FinishRequest(result.status()));
                this->ReleaseHold();
              });
        }
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

/// Authenticate ListFlights and stream the returned FlightListing iterator.
::grpc::ServerWriteReactor<pb::FlightInfo>* CallbackFlightService::ListFlights(
    ::grpc::CallbackServerContext* context, const pb::Criteria* request) {
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::ListFlights, context,
                                     &flight_context);
  if (!st.ok()) {
    return FinishWriteNow<pb::FlightInfo>(st);
  }

  Criteria criteria;
  if (request) {
    auto conv = internal::FromProto(*request, &criteria);
    if (!conv.ok())
      return FinishWriteNow<pb::FlightInfo>(flight_context.FinishRequest(conv));
  }
  auto* reactor = new IteratorReactor<pb::FlightInfo, FlightInfo>(
      context, impl_->executor(), std::move(flight_context),
      [](const FlightInfo& info, pb::FlightInfo* out) {
        return internal::ToProto(info, out);
      });
  reactor->StartAfter(
      impl_->base()->ListFlights(reactor->flight_context(), &criteria),
      [](std::unique_ptr<FlightListing> listing) {
        auto state = std::make_shared<std::unique_ptr<FlightListing>>(std::move(listing));
        return [state]() mutable {
          return *state ? (*state)->Next()
                        : arrow::Result<std::unique_ptr<FlightInfo>>(
                              std::unique_ptr<FlightInfo>{});
        };
      });
  return reactor;
}

/// Authenticate, parse, and asynchronously resolve GetFlightInfo.
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
  impl_->base()
      ->GetFlightInfo(flight_context, descr)
      .AddCallback([reactor, response, flight_context = std::move(flight_context)](
                       const arrow::Result<std::unique_ptr<FlightInfo>>& result) mutable {
        FinishUnaryResult(reactor, response, std::move(flight_context), result,
                          "Flight not found",
                          [](const FlightInfo& info, pb::FlightInfo* out) {
                            return internal::ToProto(info, out);
                          });
      });
  return reactor;
}

/// Authenticate, parse, and asynchronously resolve PollFlightInfo.
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
  impl_->base()
      ->PollFlightInfo(flight_context, descr)
      .AddCallback([reactor, response, flight_context = std::move(flight_context)](
                       const arrow::Result<std::unique_ptr<PollInfo>>& result) mutable {
        FinishUnaryResult(reactor, response, std::move(flight_context), result,
                          "Flight not found",
                          [](const PollInfo& info, pb::PollInfo* out) {
                            return internal::ToProto(info, out);
                          });
      });
  return reactor;
}

/// Authenticate, parse, and asynchronously resolve GetSchema.
::grpc::ServerUnaryReactor* CallbackFlightService::GetSchema(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::SchemaResult* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::GetSchema, context,
                                     &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  FlightDescriptor descr;
  auto arrow_st = ParseRequiredRequest(request, "FlightDescriptor", &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  impl_->base()
      ->GetSchema(flight_context, descr)
      .AddCallback(
          [reactor, response, flight_context = std::move(flight_context)](
              const arrow::Result<std::unique_ptr<SchemaResult>>& result) mutable {
            FinishUnaryResult(reactor, response, std::move(flight_context), result,
                              "Flight not found",
                              [](const SchemaResult& schema, pb::SchemaResult* out) {
                                return internal::ToProto(schema, out);
                              });
          });
  return reactor;
}

/// Authenticate DoGet, parse its ticket, and stream the async source.
::grpc::ServerWriteReactor<pb::FlightData>* CallbackFlightService::DoGet(
    ::grpc::CallbackServerContext* context, const pb::Ticket* request) {
  GrpcServerCallContext flight_context(context);
  auto st =
      PrepareAuthenticatedCall(helper_, FlightMethod::DoGet, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::FlightData>(st);
  Ticket ticket;
  auto arrow_st = ParseRequiredRequest(request, "ticket", &ticket);
  if (!arrow_st.ok()) {
    return FinishWriteNow<pb::FlightData>(flight_context.FinishRequest(arrow_st));
  }
  auto* reactor =
      new DoGetReactor(context, impl_->executor(), std::move(flight_context), nullptr);
  reactor->StartAfter(impl_->base()->DoGet(reactor->flight_context(), ticket));
  return reactor;
}

/// Create the shared async bidi reactor for DoPut.
::grpc::ServerBidiReactor<pb::FlightData, pb::PutResult>* CallbackFlightService::DoPut(
    ::grpc::CallbackServerContext* context) {
  auto* reactor = new AsyncBidiFlightReactor<pb::PutResult>(context, impl_, helper_);
  reactor->Start();
  return reactor;
}

/// Create the shared async bidi reactor for DoExchange.
::grpc::ServerBidiReactor<pb::FlightData, pb::FlightData>*
CallbackFlightService::DoExchange(::grpc::CallbackServerContext* context) {
  auto* reactor = new AsyncBidiFlightReactor<pb::FlightData>(context, impl_, helper_);
  reactor->Start();
  return reactor;
}

/// Authenticate ListActions and stream its returned action-type vector.
::grpc::ServerWriteReactor<pb::ActionType>* CallbackFlightService::ListActions(
    ::grpc::CallbackServerContext* context, const pb::Empty*) {
  GrpcServerCallContext flight_context(context);
  auto st = PrepareAuthenticatedCall(helper_, FlightMethod::ListActions, context,
                                     &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::ActionType>(st);
  auto* reactor = new IteratorReactor<pb::ActionType, ActionType>(
      context, impl_->executor(), std::move(flight_context),
      [](const ActionType& action, pb::ActionType* out) {
        return internal::ToProto(action, out);
      });
  reactor->StartAfter(impl_->base()->ListActions(reactor->flight_context()),
                      [](std::vector<ActionType> actions) {
                        return [actions = std::move(actions), index = size_t{0}]() mutable
                                   -> arrow::Result<std::unique_ptr<ActionType>> {
                          if (index >= actions.size()) return nullptr;
                          return std::make_unique<ActionType>(actions[index++]);
                        };
                      });
  return reactor;
}

/// Authenticate DoAction, parse its request, and stream action results.
::grpc::ServerWriteReactor<pb::Result>* CallbackFlightService::DoAction(
    ::grpc::CallbackServerContext* context, const pb::Action* request) {
  GrpcServerCallContext flight_context(context);
  auto st =
      PrepareAuthenticatedCall(helper_, FlightMethod::DoAction, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::Result>(st);
  Action action;
  auto arrow_st = ParseRequiredRequest(request, "Action", &action);
  if (!arrow_st.ok()) {
    return FinishWriteNow<pb::Result>(flight_context.FinishRequest(arrow_st));
  }
  auto* reactor = new IteratorReactor<pb::Result, Result>(
      context, impl_->executor(), std::move(flight_context),
      [](const Result& result, pb::Result* out) {
        return internal::ToProto(result, out);
      });
  reactor->StartAfter(
      impl_->base()->DoAction(reactor->flight_context(), action),
      [](std::unique_ptr<ResultStream> results) {
        auto state = std::make_shared<std::unique_ptr<ResultStream>>(std::move(results));
        return [state]() mutable {
          return *state ? (*state)->Next()
                        : arrow::Result<std::unique_ptr<arrow::flight::Result>>(
                              std::unique_ptr<arrow::flight::Result>{});
        };
      });
  return reactor;
}

}  // namespace

/// Create the callback-API gRPC transport selected by the async transport factory.
arrow::Result<std::unique_ptr<internal::AsyncServerTransport>>
internal::MakeGrpcCallbackServerTransport(AsyncFlightServerBase* base,
                                          std::shared_ptr<MemoryManager> memory_manager) {
  return std::unique_ptr<internal::AsyncServerTransport>(
      new AsyncGrpcServerTransport(base, std::move(memory_manager)));
}

}  // namespace arrow::flight
