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

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "arrow/buffer.h"
#include "arrow/flight/protocol_internal.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/server_middleware.h"
#include "arrow/flight/transport/grpc/customize_grpc.h"
#include "arrow/flight/transport/grpc/grpc_server.h"
#include "arrow/flight/transport/grpc/grpc_server_internal.h"
#include "arrow/flight/transport/grpc/serialization_internal.h"
#include "arrow/flight/transport/grpc/util_internal.h"
#include "arrow/flight/types.h"
#include "arrow/result.h"
#include "arrow/status.h"
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

template <typename Proto>
class WriteReactorBase : public ::grpc::ServerWriteReactor<Proto> {
 public:
  explicit WriteReactorBase(::grpc::CallbackServerContext* context) : context_(context) {}

  void OnWriteDone(bool ok) override {
    std::unique_lock<std::mutex> lock(mutex_);
    write_in_flight_ = false;
    write_ok_ = ok;
    if (finish_requested_) {
      this->Finish(finish_status_);
    }
    cv_.notify_all();
  }

  void OnCancel() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cancelled_ = true;
    cv_.notify_all();
  }

  void OnDone() override {
    if (worker_.joinable()) {
      worker_.join();
    }
    delete this;
  }

 protected:
  void StartWorker(std::function<void()> fn) { worker_ = std::thread(std::move(fn)); }

  bool WriteOne(Proto message) {
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

  ::grpc::CallbackServerContext* context_;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  Proto current_write_;
  bool write_in_flight_ = false;
  bool write_ok_ = true;
  bool cancelled_ = false;
  bool finish_requested_ = false;
  ::grpc::Status finish_status_;
};

template <typename Proto>
class ImmediateWriteReactor final : public ::grpc::ServerWriteReactor<Proto> {
 public:
  explicit ImmediateWriteReactor(::grpc::Status status) { this->Finish(std::move(status)); }
  void OnDone() override { delete this; }
};

class FlightDataWriteReactorBase : public ::grpc::ServerWriteReactor<pb::FlightData> {
 public:
  explicit FlightDataWriteReactorBase(::grpc::CallbackServerContext* context)
      : context_(context) {}

  void OnWriteDone(bool ok) override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (flight_data_registered_) {
      transport::grpc::UnregisterGrpcFlightDataMessage(&current_write_);
      flight_data_registered_ = false;
    }
    write_in_flight_ = false;
    write_ok_ = ok;
    if (finish_requested_) {
      this->Finish(finish_status_);
    }
    cv_.notify_all();
  }

  void OnCancel() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cancelled_ = true;
    cv_.notify_all();
  }

  void OnDone() override {
    if (flight_data_registered_) {
      transport::grpc::UnregisterGrpcFlightDataMessage(&current_write_);
      flight_data_registered_ = false;
    }
    if (worker_.joinable()) {
      worker_.join();
    }
    delete this;
  }

 protected:
  void StartWorker(std::function<void()> fn) { worker_ = std::thread(std::move(fn)); }

  arrow::Result<bool> WritePayloadOne(FlightPayload payload) {
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

  void FinishFromWorker(::grpc::Status status) {
    std::unique_lock<std::mutex> lock(mutex_);
    finish_requested_ = true;
    finish_status_ = std::move(status);
    if (!write_in_flight_) {
      this->Finish(finish_status_);
    }
  }

  ::grpc::CallbackServerContext* context_;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  pb::FlightData current_write_;
  bool write_in_flight_ = false;
  bool write_ok_ = true;
  bool cancelled_ = false;
  bool finish_requested_ = false;
  bool flight_data_registered_ = false;
  ::grpc::Status finish_status_;
};

template <typename Req, typename Resp>
class BidiReactorBase : public ::grpc::ServerBidiReactor<Req, Resp> {
 public:
  explicit BidiReactorBase(::grpc::CallbackServerContext* context) : context_(context) {
    if constexpr (std::is_same_v<Req, pb::FlightData>) {
      transport::grpc::RegisterGrpcFlightDataMessage(&read_buffer_);
      read_buffer_registered_ = true;
    }
  }

  void StartWorker(std::function<void()> fn) {
    this->StartRead(&read_buffer_);
    worker_ = std::thread(std::move(fn));
  }

  void OnReadDone(bool ok) override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (ok) {
      reads_.push_back(read_buffer_);
      this->StartRead(&read_buffer_);
    } else {
      reads_done_ = true;
    }
    cv_.notify_all();
  }

  void OnWriteDone(bool ok) override {
    std::unique_lock<std::mutex> lock(mutex_);
    if constexpr (std::is_same_v<Resp, pb::FlightData>) {
      if (flight_data_registered_) {
        transport::grpc::UnregisterGrpcFlightDataMessage(&current_write_);
        flight_data_registered_ = false;
      }
    }
    write_in_flight_ = false;
    write_ok_ = ok;
    if (finish_requested_) {
      this->Finish(finish_status_);
    }
    cv_.notify_all();
  }

  void OnCancel() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cancelled_ = true;
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
    if (worker_.joinable()) {
      worker_.join();
    }
    delete this;
  }

  bool ReadOne(Req* out) { return PopRead(out); }
  bool WriteOnePublic(Resp message) { return WriteOne(std::move(message)); }
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

  ::grpc::CallbackServerContext* context_;
  std::thread worker_;
  Req read_buffer_;
  std::deque<Req> reads_;
  Resp current_write_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool reads_done_ = false;
  bool read_buffer_registered_ = false;
  bool cancelled_ = false;
  bool write_in_flight_ = false;
  bool write_ok_ = true;
  bool flight_data_registered_ = false;
  bool finish_requested_ = false;
  ::grpc::Status finish_status_;
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

template <typename Req, typename Resp>
class BlockingDataStream : public internal::ServerDataStream {
 public:
  explicit BlockingDataStream(BidiReactorBase<Req, Resp>* reactor) : reactor_(reactor) {}

 protected:
  BidiReactorBase<Req, Resp>* reactor_;
};

class PutDataStream : public BlockingDataStream<pb::FlightData, pb::PutResult> {
 public:
  using BlockingDataStream::BlockingDataStream;

  bool ReadData(internal::FlightData* data) override {
    pb::FlightData message;
    if (!reactor_->ReadOne(&message)) {
      return false;
    }
    auto maybe_data = DeserializeGrpcFlightData(message);
    if (!maybe_data.ok()) {
      return false;
    }
    *data = std::move(maybe_data).ValueUnsafe();
    return true;
  }

  Status WritePutMetadata(const Buffer& metadata) override {
    pb::PutResult result;
    result.set_app_metadata(metadata.data(), metadata.size());
    if (!reactor_->WriteOnePublic(std::move(result))) {
      return Status::IOError("Unknown error writing metadata.");
    }
    return Status::OK();
  }
};

class ExchangeDataStream : public BlockingDataStream<pb::FlightData, pb::FlightData> {
 public:
  using BlockingDataStream::BlockingDataStream;

  bool ReadData(internal::FlightData* data) override {
    pb::FlightData message;
    if (!reactor_->ReadOne(&message)) {
      return false;
    }
    auto maybe_data = DeserializeGrpcFlightData(message);
    if (!maybe_data.ok()) {
      return false;
    }
    *data = std::move(maybe_data).ValueUnsafe();
    return true;
  }

  arrow::Result<bool> WriteData(const FlightPayload& payload) override {
    return reactor_->WritePayloadPublic(payload);
  }
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
    helper_ = std::make_unique<CallbackServiceHelper>(options.auth_handler, options.middleware);
    grpc_service_ = std::make_unique<CallbackFlightService>(this, *helper_);

    ::grpc::ServerBuilder builder;
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

 private:
  std::unique_ptr<CallbackServiceHelper> helper_;
  std::unique_ptr<CallbackFlightService> grpc_service_;
  std::unique_ptr<::grpc::Server> grpc_server_;
  Location location_;
};

template <typename Proto, typename UserType>
class IteratorReactor : public WriteReactorBase<Proto> {
 public:
  using NextFn = std::function<arrow::Result<std::unique_ptr<UserType>>()>;
  using ToProtoFn = std::function<Status(const UserType&, Proto*)>;

  IteratorReactor(::grpc::CallbackServerContext* context, GrpcServerCallContext flight_context,
                  NextFn next_fn, ToProtoFn to_proto)
      : WriteReactorBase<Proto>(context),
        flight_context_(std::move(flight_context)),
        next_fn_(std::move(next_fn)),
        to_proto_(std::move(to_proto)) {}

  void Start() {
    this->StartWorker([this] {
      while (true) {
        auto maybe_value = next_fn_();
        if (!maybe_value.ok()) {
          this->FinishFromWorker(flight_context_.FinishRequest(maybe_value.status()));
          return;
        }
        auto value = std::move(maybe_value).ValueUnsafe();
        if (!value) break;
        Proto proto;
        auto st = to_proto_(*value, &proto);
        if (!st.ok()) {
          this->FinishFromWorker(flight_context_.FinishRequest(st));
          return;
        }
        if (!this->WriteOne(std::move(proto))) break;
      }
      this->FinishFromWorker(flight_context_.FinishRequest(Status::OK()));
    });
  }

 private:
  GrpcServerCallContext flight_context_;
  NextFn next_fn_;
  ToProtoFn to_proto_;
};

class DoGetReactor : public FlightDataWriteReactorBase {
 public:
  DoGetReactor(::grpc::CallbackServerContext* context, GrpcServerCallContext flight_context,
               std::unique_ptr<FlightDataStream> stream)
      : FlightDataWriteReactorBase(context),
        flight_context_(std::move(flight_context)),
        stream_(std::move(stream)) {}

  void Start() {
    this->StartWorker([this] {
      if (!stream_) {
        this->FinishFromWorker(
            flight_context_.FinishRequest(Status::KeyError("No data in this flight")));
        return;
      }
      auto maybe_schema = stream_->GetSchemaPayload();
      if (!maybe_schema.ok()) {
        this->FinishFromWorker(flight_context_.FinishRequest(maybe_schema.status()));
        return;
      }
      auto maybe_wrote_schema =
          this->WritePayloadOne(std::move(maybe_schema).MoveValueUnsafe());
      if (!maybe_wrote_schema.ok()) {
        this->FinishFromWorker(flight_context_.FinishRequest(maybe_wrote_schema.status()));
        return;
      }
      if (!maybe_wrote_schema.MoveValueUnsafe()) {
        this->FinishFromWorker(flight_context_.FinishRequest(Status::OK()));
        return;
      }
      while (true) {
        auto maybe_payload = stream_->Next();
        if (!maybe_payload.ok()) {
          this->FinishFromWorker(flight_context_.FinishRequest(maybe_payload.status()));
          return;
        }
        auto payload = std::move(maybe_payload).MoveValueUnsafe();
        if (payload.ipc_message.metadata == nullptr) break;
        auto maybe_wrote_payload = this->WritePayloadOne(std::move(payload));
        if (!maybe_wrote_payload.ok()) {
          this->FinishFromWorker(
              flight_context_.FinishRequest(maybe_wrote_payload.status()));
          return;
        }
        if (!maybe_wrote_payload.MoveValueUnsafe()) break;
      }
      auto close_status = stream_->Close();
      this->FinishFromWorker(flight_context_.FinishRequest(close_status));
    });
  }

 private:
  GrpcServerCallContext flight_context_;
  std::unique_ptr<FlightDataStream> stream_;
};

::grpc::ServerBidiReactor<pb::HandshakeRequest, pb::HandshakeResponse>*
CallbackFlightService::Handshake(::grpc::CallbackServerContext* context) {
  class Reactor final : public BidiReactorBase<pb::HandshakeRequest, pb::HandshakeResponse> {
   public:
    Reactor(::grpc::CallbackServerContext* context, AsyncGrpcServerTransport* impl,
            const CallbackServiceHelper& helper)
        : BidiReactorBase(context), impl_(impl), helper_(helper), flight_context_(context) {}

    void Start() {
      auto st = helper_.MakeCallContext(FlightMethod::Handshake, this->context_,
                                        &flight_context_);
      if (!st.ok()) {
        this->Finish(st);
        return;
      }
      helper_.AddMiddlewareHeaders(this->context_, &flight_context_);
      this->StartWorker([this] {
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
  auto st = helper_.CheckAuth(FlightMethod::ListFlights, context, &flight_context);
  if (!st.ok()) {
    return FinishWriteNow<pb::FlightInfo>(st);
  }
  helper_.AddMiddlewareHeaders(context, &flight_context);

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
      context, std::move(flight_context),
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
  auto st = helper_.CheckAuth(FlightMethod::GetFlightInfo, context, &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  if (request == nullptr) {
    return FinishNow(reactor, flight_context.FinishRequest(
                                  Status::Invalid("FlightDescriptor cannot be null")));
  }
  FlightDescriptor descr;
  auto arrow_st = internal::FromProto(*request, &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  helper_.AddMiddlewareHeaders(context, &flight_context);
  impl_->base()->GetFlightInfo(flight_context, descr).AddCallback(
      [reactor, response, flight_context = std::move(flight_context)](
          const arrow::Result<std::unique_ptr<FlightInfo>>& result) mutable {
        if (result.ok() && result.ValueUnsafe()) {
          auto st = internal::ToProto(*result.ValueUnsafe(), response);
          reactor->Finish(flight_context.FinishRequest(st));
          return;
        }
        reactor->Finish(flight_context.FinishRequest(result.ok()
                                                         ? Status::KeyError("Flight not found")
                                                         : result.status()));
      });
  return reactor;
}

::grpc::ServerUnaryReactor* CallbackFlightService::PollFlightInfo(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::PollInfo* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  auto st = helper_.CheckAuth(FlightMethod::PollFlightInfo, context, &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  if (request == nullptr) {
    return FinishNow(reactor, flight_context.FinishRequest(
                                  Status::Invalid("FlightDescriptor cannot be null")));
  }
  FlightDescriptor descr;
  auto arrow_st = internal::FromProto(*request, &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  helper_.AddMiddlewareHeaders(context, &flight_context);
  impl_->base()->PollFlightInfo(flight_context, descr).AddCallback(
      [reactor, response, flight_context = std::move(flight_context)](
          const arrow::Result<std::unique_ptr<PollInfo>>& result) mutable {
        if (result.ok() && result.ValueUnsafe()) {
          auto st = internal::ToProto(*result.ValueUnsafe(), response);
          reactor->Finish(flight_context.FinishRequest(st));
          return;
        }
        reactor->Finish(flight_context.FinishRequest(result.ok()
                                                         ? Status::KeyError("Flight not found")
                                                         : result.status()));
      });
  return reactor;
}

::grpc::ServerUnaryReactor* CallbackFlightService::GetSchema(
    ::grpc::CallbackServerContext* context, const pb::FlightDescriptor* request,
    pb::SchemaResult* response) {
  auto* reactor = context->DefaultReactor();
  GrpcServerCallContext flight_context(context);
  auto st = helper_.CheckAuth(FlightMethod::GetSchema, context, &flight_context);
  if (!st.ok()) return FinishNow(reactor, st);
  if (request == nullptr) {
    return FinishNow(reactor, flight_context.FinishRequest(
                                  Status::Invalid("FlightDescriptor cannot be null")));
  }
  FlightDescriptor descr;
  auto arrow_st = internal::FromProto(*request, &descr);
  if (!arrow_st.ok()) return FinishNow(reactor, flight_context.FinishRequest(arrow_st));
  helper_.AddMiddlewareHeaders(context, &flight_context);
  impl_->base()->GetSchema(flight_context, descr).AddCallback(
      [reactor, response, flight_context = std::move(flight_context)](
          const arrow::Result<std::unique_ptr<SchemaResult>>& result) mutable {
        if (result.ok() && result.ValueUnsafe()) {
          auto st = internal::ToProto(*result.ValueUnsafe(), response);
          reactor->Finish(flight_context.FinishRequest(st));
          return;
        }
        reactor->Finish(flight_context.FinishRequest(result.ok()
                                                         ? Status::KeyError("Flight not found")
                                                         : result.status()));
      });
  return reactor;
}

::grpc::ServerWriteReactor<pb::FlightData>* CallbackFlightService::DoGet(
    ::grpc::CallbackServerContext* context, const pb::Ticket* request) {
  GrpcServerCallContext flight_context(context);
  auto st = helper_.CheckAuth(FlightMethod::DoGet, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::FlightData>(st);
  if (request == nullptr) {
    return FinishWriteNow<pb::FlightData>(
        flight_context.FinishRequest(Status::Invalid("ticket cannot be null")));
  }
  Ticket ticket;
  auto arrow_st = internal::FromProto(*request, &ticket);
  if (!arrow_st.ok()) {
    return FinishWriteNow<pb::FlightData>(flight_context.FinishRequest(arrow_st));
  }
  helper_.AddMiddlewareHeaders(context, &flight_context);
  auto maybe_stream = impl_->base()->DoGet(flight_context, ticket);
  if (!maybe_stream.status().ok()) {
    return FinishWriteNow<pb::FlightData>(flight_context.FinishRequest(maybe_stream.status()));
  }
  auto* reactor = new DoGetReactor(context, std::move(flight_context),
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
        : BidiReactorBase(context), impl_(impl), helper_(helper), flight_context_(context) {}

    void Start() {
      auto st = helper_.CheckAuth(FlightMethod::DoPut, this->context_, &flight_context_);
      if (!st.ok()) {
        this->Finish(st);
        return;
      }
      helper_.AddMiddlewareHeaders(this->context_, &flight_context_);
      this->StartWorker([this] {
        PutDataStream stream(this);
        auto status = impl_->DoPut(flight_context_, &stream);
        this->FinishFromWorker(flight_context_.FinishRequest(status));
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
        : BidiReactorBase(context), impl_(impl), helper_(helper), flight_context_(context) {}

    void Start() {
      auto st = helper_.CheckAuth(FlightMethod::DoExchange, this->context_, &flight_context_);
      if (!st.ok()) {
        this->Finish(st);
        return;
      }
      helper_.AddMiddlewareHeaders(this->context_, &flight_context_);
      this->StartWorker([this] {
        ExchangeDataStream stream(this);
        auto status = impl_->DoExchange(flight_context_, &stream);
        this->FinishFromWorker(flight_context_.FinishRequest(status));
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
  auto st = helper_.CheckAuth(FlightMethod::ListActions, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::ActionType>(st);
  helper_.AddMiddlewareHeaders(context, &flight_context);
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
      context, std::move(flight_context), std::move(next),
      [](const ActionType& action, pb::ActionType* out) {
        return internal::ToProto(action, out);
      });
  reactor->Start();
  return reactor;
}

::grpc::ServerWriteReactor<pb::Result>* CallbackFlightService::DoAction(
    ::grpc::CallbackServerContext* context, const pb::Action* request) {
  GrpcServerCallContext flight_context(context);
  auto st = helper_.CheckAuth(FlightMethod::DoAction, context, &flight_context);
  if (!st.ok()) return FinishWriteNow<pb::Result>(st);
  if (request == nullptr) {
    return FinishWriteNow<pb::Result>(
        flight_context.FinishRequest(Status::Invalid("Action cannot be null")));
  }
  Action action;
  auto arrow_st = internal::FromProto(*request, &action);
  if (!arrow_st.ok()) {
    return FinishWriteNow<pb::Result>(flight_context.FinishRequest(arrow_st));
  }
  helper_.AddMiddlewareHeaders(context, &flight_context);
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
      context, std::move(flight_context),
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
