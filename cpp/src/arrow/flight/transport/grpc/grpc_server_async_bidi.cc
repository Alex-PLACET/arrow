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
template <typename Request, typename Response>
class BidiReactorBase
    : public ::grpc::ServerBidiReactor<Request, Response>,
      public SelfOwnedReactor<BidiReactorBase<Request, Response>> {
 public:
  using ReadValue = std::conditional_t<std::is_same_v<Request, pb::FlightData>,
                                       internal::FlightData, Request>;
  using WriteValue = std::conditional_t<std::is_same_v<Response, pb::FlightData>,
                                        FlightPayload, Response>;
  using AsyncReadValue =
      std::conditional_t<std::is_same_v<Request, pb::FlightData>,
                         std::shared_ptr<internal::FlightData>, std::optional<Request>>;

 private:
  struct ReadState {
    ReadValue buffer;
    std::deque<ReadValue> messages;
    Future<AsyncReadValue> pending;
    bool pending_active = false;
    bool in_flight = false;
    bool done = false;
  };

 public:
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
      read_state_.in_flight = true;
    }
    this->StartRead(GrpcReadBuffer());
    this->Hold();
    auto maybe_future = executor_->Submit([this, fn = std::move(fn)]() mutable {
      fn();
      this->ReleaseHold();
    });
    if (!maybe_future.ok()) {
      this->ReleaseHold();
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
      std::lock_guard<std::mutex> lock(mutex_);
      read_state_.in_flight = false;
      if (ok) {
        completed_read.emplace(std::move(read_state_.buffer));
        if (!read_state_.pending_active) {
          read_state_.messages.push_back(std::move(*completed_read));
        }
      } else {
        read_state_.done = true;
      }
      if (read_state_.pending_active) {
        pending_future = read_state_.pending;
        read_state_.pending_active = false;
        read_state_.pending = Future<AsyncReadValue>();
        resolve_pending = true;
        if (ok && !this->cancelled()) {
          read_state_.in_flight = true;
          start_next_read = true;
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
    write_state_.in_flight = false;
    write_state_.ok = ok;
    if (write_state_.pending_active) {
      pending_future = write_state_.pending;
      write_state_.pending_active = false;
      write_state_.pending = Future<bool>();
      resolve_pending = true;
    }
    if (finish_state_.requested) {
      finish_now = true;
      finish_status = finish_state_.status;
    }
    cancelled = this->cancelled();
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
    this->SetCanceled();
    std::unique_lock<std::mutex> lock(mutex_);
    Future<bool> write_future;
    bool resolve_write = false;
    if (write_state_.pending_active) {
      write_future = write_state_.pending;
      write_state_.pending_active = false;
      write_state_.pending = Future<bool>();
      resolve_write = true;
    }
    if (read_state_.pending_active) {
      auto future = read_state_.pending;
      read_state_.pending_active = false;
      read_state_.pending = Future<AsyncReadValue>();
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
  void OnDone() override { this->ReleaseHold(); }

  /// Synchronously read one message for legacy handshake authentication.
  bool ReadOne(Request* out) { return PopRead(out); }

  /// Block a compatibility caller until a gRPC write completes.
  bool WriteOne(Response message) { return WriteOneImpl(std::move(message)); }

  /// Return the next inbound message while enforcing one outstanding read.
  Future<AsyncReadValue> ReadOneAsync() {
    bool start_read = false;
    std::unique_lock<std::mutex> lock(mutex_);
    if (!read_state_.messages.empty()) {
      ReadValue out = std::move(read_state_.messages.front());
      read_state_.messages.pop_front();
      if (!read_state_.in_flight && !read_state_.done && !this->cancelled()) {
        read_state_.in_flight = true;
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
    if (this->cancelled() || read_state_.done) {
      return Future<AsyncReadValue>::MakeFinished(EndAsyncReadValue());
    }
    if (read_state_.pending_active) {
      return Future<AsyncReadValue>::MakeFinished(
          Status::Invalid("Concurrent async reads are not supported"));
    }
    read_state_.pending = Future<AsyncReadValue>::Make();
    read_state_.pending_active = true;
    if (!read_state_.in_flight) {
      read_state_.in_flight = true;
      start_read = true;
    }
    auto future = read_state_.pending;
    lock.unlock();
    if (start_read) {
      this->StartRead(GrpcReadBuffer());
    }
    return future;
  }

  /// Start an asynchronous protobuf response write.
  Future<bool> WriteOneAsync(Response message) {
    return StartAsyncWrite(std::move(message));
  }

  /// Validate and start an asynchronous FlightData response write.
  Future<bool> WritePayloadAsync(FlightPayload payload) {
    static_assert(std::is_same_v<Response, pb::FlightData>);
    RETURN_NOT_OK(payload.Validate());
    return StartAsyncWrite(std::move(payload));
  }

  /// Synchronously write a Flight payload for legacy server streams.
  arrow::Result<bool> WritePayloadPublic(FlightPayload payload) {
    static_assert(std::is_same_v<Response, pb::FlightData>);
    RETURN_NOT_OK(payload.Validate());
    return WriteOneImpl(std::move(payload));
  }

  protected:
  /// Start one outbound write while enforcing the single-write contract.
  Future<bool> StartAsyncWrite(WriteValue message) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (this->cancelled()) {
      return Future<bool>::MakeFinished(false);
    }
    if (write_state_.pending_active || write_state_.in_flight) {
      return Future<bool>::MakeFinished(
          Status::Invalid("Concurrent async writes are not supported"));
    }
    write_state_.pending = Future<bool>::Make();
    write_state_.pending_active = true;
    current_write_ = std::move(message);
    write_state_.in_flight = true;
    this->StartWrite(GrpcWriteBuffer());
    return write_state_.pending;
  }

  /// Block a compatibility caller until an inbound message or end-of-stream.
  bool PopRead(Request* out) {
    bool start_read = false;
    std::unique_lock<std::mutex> lock(mutex_);
    if (read_state_.messages.empty() && !this->cancelled() && !read_state_.done &&
        !read_state_.in_flight) {
      read_state_.in_flight = true;
      start_read = true;
    }
    if (start_read) {
      lock.unlock();
      this->StartRead(GrpcReadBuffer());
      lock.lock();
    }
    cv_.wait(lock, [&] {
      return this->cancelled() || read_state_.done || !read_state_.messages.empty();
    });
    if (!read_state_.messages.empty()) {
      *out = std::move(read_state_.messages.front());
      read_state_.messages.pop_front();
      return true;
    }
    return false;
  }

  /// Start a synchronous write for either a protobuf or FlightData response.
  bool WriteOneImpl(WriteValue message) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (this->cancelled()) {
      return false;
    }
    current_write_ = std::move(message);
    write_state_.in_flight = true;
    write_state_.ok = true;
    this->StartWrite(GrpcWriteBuffer());
    cv_.wait(lock, [&] { return !write_state_.in_flight || this->cancelled(); });
    return !this->cancelled() && write_state_.ok;
  }

  /// Request RPC completion, deferring it until an active write finishes.
  void FinishFromWorker(::grpc::Status status) {
    bool finish_now = false;
    std::unique_lock<std::mutex> lock(mutex_);
    finish_state_.requested = true;
    finish_state_.status = std::move(status);
    if (!write_state_.in_flight) {
      finish_now = true;
    }
    lock.unlock();
    if (finish_now) {
      this->Finish(finish_state_.status);
    }
  }

  /// Return the protobuf storage used by gRPC for the next inbound message.
  Request* GrpcReadBuffer() {
    if constexpr (std::is_same_v<Request, pb::FlightData>) {
      return reinterpret_cast<Request*>(&read_state_.buffer);
    } else {
      return &read_state_.buffer;
    }
  }

  /// Return the protobuf storage used by gRPC for the active outbound message.
  Response* GrpcWriteBuffer() {
    if constexpr (std::is_same_v<Response, pb::FlightData>) {
      return reinterpret_cast<Response*>(&current_write_);
    } else {
      return &current_write_;
    }
  }

  /// Convert a transport read value to the public async representation.
  AsyncReadValue MakeAsyncReadValue(ReadValue value) {
    if constexpr (std::is_same_v<Request, pb::FlightData>) {
      return std::make_shared<internal::FlightData>(std::move(value));
    } else {
      return std::optional<Request>(std::move(value));
    }
  }

  /// Return the public end-of-stream representation for this request type.
  AsyncReadValue EndAsyncReadValue() {
    if constexpr (std::is_same_v<Request, pb::FlightData>) {
      return nullptr;
    } else {
      return std::optional<Request>{};
    }
  }

  /// gRPC context associated with this reactor.
  ::grpc::CallbackServerContext* context_;
  /// Executor used for blocking compatibility handlers.
  std::shared_ptr<arrow::internal::ThreadPool> executor_;
  /// Receive-side state, including the buffer owned by the active gRPC read.
  ReadState read_state_;
  /// Protobuf or FlightPayload storage supplied to the active gRPC write.
  WriteValue current_write_;
  /// Protects all read, write, cancellation, and completion state.
  std::mutex mutex_;
  /// Wakes compatibility callers after reads, writes, or cancellation.
  std::condition_variable cv_;
  /// Write-side state, including the pending application write.
  struct WriteState {
    /// Future completed when the one application write currently in progress ends.
    Future<bool> pending;
    /// Whether pending contains an unresolved application write.
    bool pending_active = false;
    /// Whether a gRPC write is currently active.
    bool in_flight = false;
    /// Result reported by the most recent completed gRPC write.
    bool ok = true;
  };
  WriteState write_state_;
  /// Deferred RPC completion requested by a background worker.
  struct FinishState {
    bool requested = false;
    ::grpc::Status status;
  };
  FinishState finish_state_;
};

/// Dispatches DoPut or DoExchange after authentication and the first input frame.
template <typename Response>
class AsyncBidiFlightReactor final : public BidiReactorBase<pb::FlightData, Response> {
 public:
  AsyncBidiFlightReactor(::grpc::CallbackServerContext* context,
                         AsyncGrpcServerTransport* impl,
                         const CallbackServiceHelper& helper)
      : BidiReactorBase<pb::FlightData, Response>(context, impl->executor()),
        impl_(impl),
        helper_(helper),
        flight_context_(context) {}

  /// Authenticate the RPC and asynchronously acquire its public reader.
  void Start() {
    constexpr auto kMethod = std::is_same_v<Response, pb::PutResult>
                                 ? FlightMethod::DoPut
                                 : FlightMethod::DoExchange;
    const auto status =
        PrepareAuthenticatedCall(helper_, kMethod, this->context_, &flight_context_);
    if (!status.ok()) {
      this->Finish(status);
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
    if constexpr (std::is_same_v<Response, pb::PutResult>) {
      completion =
          impl_->base()->DoPut(flight_context_, reader_and_state.TakeReader(),
                               MakeAsyncMetadataWriter([this](pb::PutResult result) {
                                 return this->WriteOneAsync(std::move(result));
                               }));
    } else {
      completion =
          impl_->base()->DoExchange(flight_context_, reader_and_state.TakeReader(),
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

  /// Async transport providing the server implementation and executor.
  AsyncGrpcServerTransport* impl_;
  /// Authentication and middleware helper for this RPC.
  const CallbackServiceHelper& helper_;
  /// Flight context used for authentication and status conversion.
  GrpcServerCallContext flight_context_;
};

}  // namespace

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
    auto grpc_status = helper_.MakeCallContext(FlightMethod::Handshake, this->context_,
                                               &flight_context_);
    if (!grpc_status.ok()) {
      this->Finish(grpc_status);
      return;
    }
    helper_.AddMiddlewareHeaders(this->context_, &flight_context_);
    const auto status = this->StartWorker([this] { RunHandshake(); });
    if (!status.ok()) {
      this->Finish(flight_context_.FinishRequest(status));
    }
  }

 private:
  void RunHandshake() {
    auto outgoing = std::make_unique<
        ::arrow::flight::transport::grpc::GrpcServerAuthSender<pb::HandshakeResponse>>(
        [this](pb::HandshakeResponse response) {
          return this->WriteOne(std::move(response));
        });

    auto incoming = std::make_unique<
        ::arrow::flight::transport::grpc::GrpcServerAuthReader<pb::HandshakeRequest>>(
        [this](pb::HandshakeRequest* request) { return this->ReadOne(request); });

    if (helper_.auth_handler()) {
      const auto status = helper_.auth_handler()->Authenticate(flight_context_, outgoing.get(),
                                                               incoming.get());
      this->FinishFromWorker(flight_context_.FinishRequest(status));
    } else {
      this->Hold();
      impl_->base()
          ->Handshake(flight_context_, std::move(outgoing), std::move(incoming))
          .AddCallback([this](const ::arrow::Result<::arrow::internal::Empty>& result) {
            this->FinishFromWorker(flight_context_.FinishRequest(result.status()));
            this->ReleaseHold();
          });
    }
  }

  /// Async transport providing the server implementation.
  AsyncGrpcServerTransport* impl_;
  /// Authentication and middleware helper for the handshake.
  const CallbackServiceHelper& helper_;
  /// Flight context used for handshake authentication and completion.
  GrpcServerCallContext flight_context_;
};

::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>*
MakeHandshakeReactor(::grpc::CallbackServerContext* context,
                     AsyncGrpcServerTransport* impl,
                     const CallbackServiceHelper& helper) {
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