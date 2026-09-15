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
#include <mutex>
#include <optional>
#include <utility>

#include "arrow/flight/transport/grpc/grpc_server_internal.h"

namespace arrow::flight::transport::grpc::async_internal {
namespace {

/// Shared state machine for callback bidi RPCs, exposing future-based reads
/// and writes to the async server hooks.
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
    /// At most one read completed before a consumer asked for it (the read
    /// loop stops issuing reads while this is set).
    std::optional<ReadValue> buffered;
    Future<AsyncReadValue> pending;
    bool pending_active = false;
    bool in_flight = false;
    bool done = false;
  };

 public:
  explicit BidiReactorBase(::grpc::CallbackServerContext* context)
      : context_(context) {}

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
      } else {
        read_state_.done = true;
      }

      if (read_state_.pending_active) {
        pending_future = read_state_.pending;
        read_state_.pending_active = false;
        read_state_.pending = Future<AsyncReadValue>();
        resolve_pending = true;
      } else if (ok) {
        // No consumer waiting: buffer the message (at most one is kept).
        read_state_.buffered.emplace(std::move(*completed_read));
      }
      
      if (ok) {
        start_next_read = MaybeStartNextReadLocked();
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
      return;
    }
    lock.unlock();
    if (resolve_write) {
      write_future.MarkFinished(false);
    }
  }

  /// Release gRPC's ownership reference.
  void OnDone() override { this->ReleaseHold(); }

  /// Return the next inbound message while enforcing one outstanding read.
  Future<AsyncReadValue> ReadOneAsync() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (read_state_.buffered.has_value()) {
      ReadValue out = std::move(*read_state_.buffered);
      read_state_.buffered.reset();
      auto future =
          Future<AsyncReadValue>::MakeFinished(MakeAsyncReadValue(std::move(out)));
      const bool start_read = MaybeStartNextReadLocked();
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
    auto future = read_state_.pending;
    const bool start_read = MaybeStartNextReadLocked();
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

  /// The one read-loop rule: keep exactly one read in flight unless a message
  /// is already buffered, the stream ended, or the RPC was cancelled. Call
  /// under mutex_; if this returns true, start the read after unlocking.
  bool MaybeStartNextReadLocked() {
    if (read_state_.in_flight || read_state_.done || this->cancelled() ||
        read_state_.buffered.has_value()) {
      return false;
    }
    read_state_.in_flight = true;
    return true;
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
  /// Receive-side state, including the buffer owned by the active gRPC read.
  ReadState read_state_;
  /// Protobuf or FlightPayload storage supplied to the active gRPC write.
  WriteValue current_write_;
  /// Protects all read, write, cancellation, and completion state.
  std::mutex mutex_;
  /// Write-side state, including the pending application write.
  struct WriteState {
    /// Future completed when the one application write currently in progress ends.
    Future<bool> pending;
    /// Whether pending contains an unresolved application write.
    bool pending_active = false;
    /// Whether a gRPC write is currently active.
    bool in_flight = false;
  };
  WriteState write_state_;
  /// Deferred RPC completion requested via FinishFromWorker().
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
      : BidiReactorBase<pb::FlightData, Response>(context),
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

  /// Async transport providing the server implementation.
  AsyncGrpcServerTransport* impl_;
  /// Authentication and middleware helper for this RPC.
  const CallbackServiceHelper& helper_;
  /// Flight context used for authentication and status conversion.
  GrpcServerCallContext flight_context_;
};

}  // namespace

template <typename Request>
class AsyncGrpcServerAuthReader final : public AsyncServerAuthReader {
 public:
  using ReadFn = std::function<Future<std::optional<Request>>()>;
  explicit AsyncGrpcServerAuthReader(ReadFn read_fn) : read_fn_(std::move(read_fn)) {}

  Future<std::string> Read() override {
    return read_fn_().Then([](const std::optional<Request>& request) {
      if (!request) {
        return Future<std::string>::MakeFinished(Status::IOError("Stream is closed."));
      }
      return Future<std::string>::MakeFinished(std::string(request->payload()));
    });
  }

 private:
  ReadFn read_fn_;
};

template <typename Response>
class AsyncGrpcServerAuthSender final : public AsyncServerAuthSender {
 public:
  using WriteFn = std::function<Future<bool>(Response)>;
  explicit AsyncGrpcServerAuthSender(WriteFn write_fn) : write_fn_(std::move(write_fn)) {}

  Future<> Write(const std::string& token) override {
    Response response;
    response.set_payload(token);
    return write_fn_(std::move(response)).Then([](bool ok) -> Status {
      return ok ? Status::OK() : Status::IOError("Stream was closed.");
    });
  }

 private:
  WriteFn write_fn_;
};

class Reactor final
    : public BidiReactorBase<pb::HandshakeRequest, pb::HandshakeResponse> {
 public:
  Reactor(::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
          const CallbackServiceHelper& helper)
      : BidiReactorBase(context), impl_(impl), helper_(helper), flight_context_(context) {}

  void Start() {
    auto grpc_status = helper_.MakeCallContext(FlightMethod::Handshake, this->context_,
                                               &flight_context_);
    if (!grpc_status.ok()) {
      this->Finish(grpc_status);
      return;
    }
    helper_.AddMiddlewareHeaders(this->context_, &flight_context_);
    RunHandshake();
  }

 private:
  /// Drive the handshake inline: the async sender/reader never block, so the
  /// callback thread stays free.
  void RunHandshake() {
    auto outgoing = std::make_unique<AsyncGrpcServerAuthSender<pb::HandshakeResponse>>(
        [this](pb::HandshakeResponse response) {
          return this->WriteOneAsync(std::move(response));
        });
    auto incoming = std::make_unique<AsyncGrpcServerAuthReader<pb::HandshakeRequest>>(
        [this] { return this->ReadOneAsync(); });
    this->Hold();
    impl_->base()
        ->Handshake(flight_context_, std::move(outgoing), std::move(incoming))
        .AddCallback([this](const ::arrow::Result<::arrow::internal::Empty>& result) {
          this->Finish(flight_context_.FinishRequest(result.status()));
          this->ReleaseHold();
        });
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