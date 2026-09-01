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

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

#include "arrow/flight/transport/grpc/grpc_server_internal.h"

namespace arrow::flight::transport::grpc::async_internal {
namespace {

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
    if (cancelled_)
    {   
        return false;
    }
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
    MakeAsyncMessageReader([this] { return this->ReadOneAsync(); },
                           impl_->memory_manager())
        .AddCallback([this](const ::arrow::Result<AsyncMessageReader>& maybe_reader) {
          HandleReader(maybe_reader);
        });
  }

 private:
  /// Invoke the selected server hook or finish an invalid inbound stream.
  void HandleReader(const ::arrow::Result<AsyncMessageReader>& result) {
    if (!result.ok()) {
      this->Finish(flight_context_.FinishRequest(result.status()));
      this->ReleaseHold();
      return;
    }
    auto reader_and_state =
        std::move(const_cast<::arrow::Result<AsyncMessageReader>&>(result))
            .MoveValueUnsafe();
    Future<> completion;
    if constexpr (std::is_same_v<Resp, pb::PutResult>) {
      completion = impl_->base()->DoPut(
          flight_context_, reader_and_state.TakeReader(),
          MakeAsyncMetadataWriter([this](pb::PutResult result) {
            return this->WriteOneAsync(std::move(result));
          }));
    } else {
      completion = impl_->base()->DoExchange(
          flight_context_, reader_and_state.TakeReader(),
          MakeAsyncMessageWriter([this](FlightPayload payload) {
            return this->WritePayloadAsync(std::move(payload));
          }));
    }
    FinishAfterReader(std::move(reader_and_state), std::move(completion));
  }

  /// Defer final gRPC completion until the application reader is idle.
  void FinishAfterReader(AsyncMessageReader reader, Future<> completion) {
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

}  // namespace

::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>*
MakeHandshakeReactor(::grpc::CallbackServerContext* context,
                     AsyncGrpcServerTransport* impl,
                     const CallbackServiceHelper& helper) {
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
            ::arrow::flight::transport::grpc::GrpcServerAuthSender<
                pb::HandshakeResponse>>([this](pb::HandshakeResponse response) {
          return this->WriteOnePublic(std::move(response));
        });
        auto incoming = std::make_unique<
            ::arrow::flight::transport::grpc::GrpcServerAuthReader<
                pb::HandshakeRequest>>([this](pb::HandshakeRequest* request) {
          return this->ReadOne(request);
        });
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

  auto* reactor = new Reactor(context, impl, helper);
  reactor->Start();
  return reactor;
}

::grpc::ServerBidiReactor<pb::FlightData, pb::PutResult>* MakeDoPutReactor(
    ::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
    const CallbackServiceHelper& helper) {
  auto* reactor = new AsyncBidiFlightReactor<pb::PutResult>(context, impl, helper);
  reactor->Start();
  return reactor;
}

::grpc::ServerBidiReactor<pb::FlightData, pb::FlightData>* MakeDoExchangeReactor(
    ::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
    const CallbackServiceHelper& helper) {
  auto* reactor = new AsyncBidiFlightReactor<pb::FlightData>(context, impl, helper);
  reactor->Start();
  return reactor;
}

}  // namespace arrow::flight::transport::grpc::async_internal