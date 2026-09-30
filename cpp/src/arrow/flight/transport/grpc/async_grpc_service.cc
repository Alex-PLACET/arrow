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

// Reactors and dispatch for the async Flight service implemented on top of
// gRPC's generic callback API (apache/arrow#49339).  Serves every
// FlightService method: Handshake, DoGet, DoPut, the unary
// GetFlightInfo/GetSchema/PollFlightInfo, the streaming
// ListActions/DoAction/ListFlights, and DoExchange.
//
// Everything here except AsyncGenericFlightService is file-local: the reactors
// are an implementation detail of the gRPC transport.

#include "arrow/flight/transport/grpc/async_grpc_service.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>

#include <grpcpp/support/byte_buffer.h>
#include <grpcpp/support/slice.h>

#include "arrow/flight/flight_data_decoder.h"
#include "arrow/flight/protocol_internal.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/transport/grpc/serialization_internal.h"
#include "arrow/flight/transport_server.h"
#include "arrow/flight/transport_server_internal.h"
#include "arrow/flight/types.h"
#include "arrow/util/logging.h"

namespace arrow::flight::transport::grpc {

namespace pb = arrow::flight::protocol;

namespace {

// The Arrow-side call context every reactor on this path carries; it is built
// by the shared helper (which runs middleware and auth) from the raw gRPC
// context, and every finish goes through its FinishRequest.
using AsyncCallContext = GrpcServerCallContext<::grpc::CallbackServerContext>;

constexpr std::string_view kPrefix = "/arrow.flight.protocol.FlightService/";
constexpr std::string_view kHandshakeMethod = "Handshake";
constexpr std::string_view kDoGetMethod = "DoGet";
constexpr std::string_view kDoPutMethod = "DoPut";
constexpr std::string_view kDoActionMethod = "DoAction";
constexpr std::string_view kListActionsMethod = "ListActions";
constexpr std::string_view kDoExchangeMethod = "DoExchange";
constexpr std::string_view kPollFlightInfoMethod = "PollFlightInfo";
constexpr std::string_view kListFlightsMethod = "ListFlights";
constexpr std::string_view kGetFlightInfoMethod = "GetFlightInfo";
constexpr std::string_view kGetSchemaMethod = "GetSchema";

/// \return The corresponding FlightMethod, or FlightMethod::Invalid if the
/// name is unknown.
FlightMethod MethodFromName(std::string_view method) {
  if (!method.starts_with(kPrefix)) {
    return FlightMethod::Invalid;
  }
  method.remove_prefix(kPrefix.size());
  constexpr std::pair<std::string_view, FlightMethod> kMethods[] = {
      {kHandshakeMethod, FlightMethod::Handshake},
      {kListFlightsMethod, FlightMethod::ListFlights},
      {kGetFlightInfoMethod, FlightMethod::GetFlightInfo},
      {kGetSchemaMethod, FlightMethod::GetSchema},
      {kDoGetMethod, FlightMethod::DoGet},
      {kDoPutMethod, FlightMethod::DoPut},
      {kDoActionMethod, FlightMethod::DoAction},
      {kListActionsMethod, FlightMethod::ListActions},
      {kDoExchangeMethod, FlightMethod::DoExchange},
      {kPollFlightInfoMethod, FlightMethod::PollFlightInfo},
  };

  const auto it = std::ranges::find_if(
      kMethods, [method](const auto& pair) { return pair.first == method; });
  if (it != std::end(kMethods)) {
    return it->second;
  }
  return FlightMethod::Invalid;
}

/// Serialize one proto message into a ByteBuffer the caller must keep alive
/// until the write completes (a reactor member, never a local).
template <typename ProtoT>
::grpc::ByteBuffer MakeWriteBuffer(const ProtoT& message) {
  const std::string bytes = message.SerializeAsString();
  ::grpc::Slice slice(bytes);
  return ::grpc::ByteBuffer(&slice, 1);
}

/// \brief Read `buf` as the proto message `PbT`.
/// \param[in] what names the message in the "Failed to read/parse <what>"
/// errors ("Ticket", "FlightDescriptor", …).
template <typename PbT>
arrow::Result<PbT> ParseProtoRequest(const ::grpc::ByteBuffer& buf,
                                     std::string_view what) {
  ARROW_ASSIGN_OR_RAISE(std::string bytes,
                        BytesFromBuffer(buf, "Failed to read request"));
  PbT pb;
  if (!pb.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
    return arrow::Status::Invalid("Failed to parse ", what);
  }
  return pb;
}

/// \brief Read `buf` as the proto message `PbT` and convert it to `T`.
template <typename PbT, typename T>
arrow::Result<T> ParseProtoRequest(const ::grpc::ByteBuffer& buf, std::string_view what) {
  ARROW_ASSIGN_OR_RAISE(auto pb, ParseProtoRequest<PbT>(buf, what));
  T out;
  ARROW_RETURN_NOT_OK(internal::FromProto(pb, &out));
  return out;
}

/// \brief Serialize `value` into `*out`, or answer what the sync transport
/// answers for a handler that returned OK without setting its result
/// (grpc_server.cc: "Flight not found").
template <typename T, typename PbT>
arrow::Status SerializeOrNotFound(const std::unique_ptr<T>& value, PbT* out) {
  if (value == nullptr) {
    return arrow::Status::KeyError("Flight not found");
  }
  return internal::ToProto(*value, out);
}

/// \brief Serve one DoGet RPC over the generic callback API.
///
/// The generic callback API has no server-streaming reactor, so DoGet is
/// served on a bidi reactor used write-only: the request is read once, then
/// one payload is written per OnWriteDone turn until the FlightDataStream
/// ends.
class DoGetReactor : public ::grpc::ServerGenericBidiReactor {
 public:
  /// `flight_context` is prepared by the service (middleware/auth already
  /// ran), the underlying gRPC context is owned by gRPC and outlives the
  /// reactor (the reactor is deleted in OnDone, before the context is
  /// destroyed).
  /// \param[in] flight_context The context for the Flight RPC.
  /// \param[in] base The base server handling the RPC.
  DoGetReactor(AsyncCallContext flight_context, AsyncGenericFlightServerBase* base)
      : flight_context_(std::move(flight_context)), base_(base) {
    StartRead(&request_buf_);
  }

  /// \brief Called when a read operation has completed.
  /// \param[in] ok Whether the read was successful.
  void OnReadDone(bool ok) override {
    // Request has been read.
    if (!ok) {
      FinishOnce(flight_context_.FinishRequest(
          MakeFlightError(FlightStatusCode::Internal, "Failed to read request")));
      return;
    }

    const auto ticket = ParseTicket();
    if (!ticket.ok()) {
      FinishOnce(flight_context_.FinishRequest(ticket.status()));
      return;
    }

    // The server class may prepare the stream asynchronously. Hold the
    // reactor across the callback: it runs on whichever thread completes the
    // future, which can be after OnDone (gRPC has no server-side holds, so
    // the refcount below is ours).
    Hold();
    arrow::Future<std::shared_ptr<AsyncFlightDataStream>> future =
        base_->DoGetAsync(flight_context_, *ticket);
    future.AddCallback(
        [this, future](
            const arrow::Result<std::shared_ptr<AsyncFlightDataStream>>& result) mutable {
          if (!result.ok()) {
            FinishOnce(flight_context_.FinishRequest(result.status()));
            ReleaseHold();
            return;
          }
          if (cancelled_) {
            // The RPC died while the future was pending.
            auto stream = *future.MoveResult();
            if (stream != nullptr) {
              ARROW_WARN_NOT_OK(stream->Close(),
                                "DoGet: closing the stream of a cancelled call failed");
            }
            FinishOnce(flight_context_.FinishRequest(arrow::Status::Cancelled()));
            ReleaseHold();
            return;
          }
          async_data_stream_ = std::move(*future.MoveResult());
          if (async_data_stream_ == nullptr) {
            FinishOnce(flight_context_.FinishRequest(
                arrow::Status::KeyError("No data in this flight")));
            ReleaseHold();
            return;
          }
          WriteNextPayload();
          ReleaseHold();
        });
  }

  /// \brief Called when a write operation has completed.
  /// \param[in] ok Whether the write was successful.
  void OnWriteDone(bool ok) override {
    // We have finished writing. We can write the next payload or finish the
    // stream.
    if (!ok) {
      // The write failed (usually because the client went away): let the
      // producer stop before finishing.
      if (async_data_stream_ != nullptr) {
        // This is the owning thread, so the close is safe here.
        ARROW_WARN_NOT_OK(async_data_stream_->Close(),
                          "DoGet: closing the data stream after a failed write failed");
      }
      FinishOnce(flight_context_.FinishRequest(
          MakeFlightError(FlightStatusCode::Internal, "Write failed")));
      return;
    }
    // Continue writing the next payload.
    WriteNextPayload();
  }

  /// \brief Called when the client cancels the request.
  /// This allows the server to stop producing data for a request that will no longer be
  /// consumed.
  void OnCancel() override { cancelled_ = true; }

  /// \brief Called when the RPC is fully done, regardless of success or cancellation.
  /// This is the last callback that will be invoked for this RPC.
  void OnDone() override { ReleaseHold(); }

 private:
  /// Parse the request ByteBuffer as the pb::Ticket of the DoGet request.
  /// \return A Result containing the parsed Ticket, or an error if parsing failed.
  arrow::Result<Ticket> ParseTicket() {
    return ParseProtoRequest<pb::Ticket, Ticket>(request_buf_, "Ticket");
  }

  /// \brief Ensures that gRPC's Finish is called at most once, and never after OnDone.
  /// \param[in] status The gRPC status to finish the call with.
  void FinishOnce(::grpc::Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true)) {
      Finish(std::move(status));
    }
  }

  // gRPC's server callback API has no holds, so the reactor refcounts itself:
  // one reference for the RPC (released in OnDone) plus one per pending
  // continuation.  Whichever thread releases the last reference deletes it.
  void Hold() { refs_.fetch_add(1, std::memory_order_relaxed); }
  void ReleaseHold() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  /// Serialize the next payload of the data stream and start writing it.
  /// Called once from OnReadDone (schema payload) and then once per OnWriteDone.
  /// NextAsync() is the server's code: the payload can complete on any thread,
  /// so the reactor holds itself across that callback and never touches the call
  /// context once the RPC is dead.
  ///
  /// The payload sequence mirrors the sync DoGet contract,
  /// ServerTransportBase::WriteDataStream (transport_server_internal.cc): schema
  /// payload first, the end of the stream is the payload whose
  /// ipc_message.metadata is null, Close() last.
  void WriteNextPayload() {
    if (cancelled_) {
      FinishOnce(flight_context_.FinishRequest(arrow::Status::Cancelled()));
      return;
    }
    Hold();  // the payload may complete on the server's own thread
    arrow::Future<FlightPayload> next = wrote_schema_
                                            ? async_data_stream_->NextAsync()
                                            : async_data_stream_->GetSchemaPayloadAsync();

    next.AddCallback([this](arrow::Result<FlightPayload> result) {
      if (cancelled_) {
        // The RPC died while the payload was pending.
        FinishOnce(flight_context_.FinishRequest(arrow::Status::Cancelled()));
        ReleaseHold();
        return;
      }

      if (!result.ok()) {
        FinishOnce(flight_context_.FinishRequest(result.status()));
        ReleaseHold();
        return;
      }

      wrote_schema_ = true;
      FlightPayload payload = std::move(*result);

      // End of stream: the last payload has no metadata.
      if (payload.ipc_message.metadata == nullptr) {
        FinishOnce(flight_context_.FinishRequest(async_data_stream_->Close()));
        ReleaseHold();
        return;
      }

      bool own_buffer = false;
      const ::grpc::Status grpc_status =
          FlightDataSerialize(payload, &write_buf_, &own_buffer);

      if (!grpc_status.ok()) {
        FinishOnce(flight_context_.FinishRequest(
            MakeFlightError(FlightStatusCode::Internal, grpc_status.error_message())));
        ReleaseHold();
        return;
      }

      // StartWrite requires the buffer to remain valid until OnWriteDone.
      StartWrite(&write_buf_);
      ReleaseHold();  // the next hop is a gRPC callback (OnWriteDone)
    });
  }

  GrpcServerCallContext<::grpc::CallbackServerContext> flight_context_;
  AsyncGenericFlightServerBase* base_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
  std::shared_ptr<AsyncFlightDataStream> async_data_stream_;
  bool wrote_schema_ = false;
  /// Set by OnCancel: a pending DoGetAsync continuation must not write or touch
  /// the call context afterwards.
  std::atomic<bool> cancelled_{false};
  std::atomic<bool> finished_{false};
  /// One reference for the RPC (released in OnDone) plus one per pending
  /// continuation.
  std::atomic<int> refs_{1};
};

/// \brief Serve one DoPut RPC over the generic callback API.
///
/// Reads one message per OnReadDone turn and pushes it into the per-RPC
/// FlightMessageDecoder, which fires the listener's callbacks
/// The listener's status is the upload's status: a non-OK one rejects the upload (the
/// acknowledgement is not written, so the client's DoPut fails with it). On
/// end of stream the acknowledgement is written back, that is what makes the
/// client's DoPut return.
class DoPutReactor : public ::grpc::ServerGenericBidiReactor,
                     public internal::FlightDataListenerTransport {
 public:
  /// \param `flight_context` is prepared by the service (middleware/auth ran); the raw
  /// context is owned by gRPC and outlives the reactor (the reactor is
  /// deleted in OnDone, before the context is destroyed).
  /// \param `listener` is the per-RPC listener that will receive the uploaded data.
  DoPutReactor(AsyncCallContext flight_context,
               std::shared_ptr<FlightDataListener> listener)
      : flight_context_(std::move(flight_context)),
        listener_(std::move(listener)),
        decoder_(listener_) {
    // The listener can now Cancel() this RPC; cleared in OnDone, before the
    // reactor dies.
    internal::FlightDataListenerTransport::Install(listener_, this);
    StartRead(&request_buf_);
  }

  /// \brief Cancel the upload from the application side: finish the RPC with
  /// `status` without waiting for the client to end it.
  ///
  /// Callable from any thread (that is the point).  Idempotent: `finished_`
  /// makes only the first ending take effect.  A Cancel() with an OK status is
  /// meaningless (there is nothing to cancel successfully), so it is refused.
  ///
  /// The reactor refcounts itself here as DoGet's does: gRPC's callback API has
  /// no holds, and OnDone delete this, so a cancel arriving from another
  /// thread must keep the reactor alive across the call.
  void CancelUpload(Status status) override {
    if (status.ok()) {
      status = Status::Invalid("Cancel() needs a non-OK status");
    }
    // FinishUpload reports the ending and claims it atomically, so the read
    // that gRPC completes during teardown cannot report a clean end on top.
    Hold();
    FinishUpload(std::move(status));
    ReleaseHold();
  }

  /// \brief Called when a new message is available from the client.
  /// \param `ok` indicates whether the read was successful. If false, the client has
  /// closed its half of the stream.
  void OnReadDone(bool ok) override {
    if (!ok) {
      // The read ended.  That is the client's clean half-close only when the
      // RPC is still running with no ending claimed yet: on a server-side
      // Cancel() (or a decode rejection) the teardown makes the pending read
      // complete too, and that ending must keep its own status.
      if (finished_.load()) {
        return;
      }
      // The client closed its half of the stream: the upload ended normally.
      // Report that to the listener, then acknowledge the upload, which is
      // what makes the client's DoPut return.
      ARROW_WARN_NOT_OK(
          internal::FlightDataListenerTransport::ReportFinish(listener_, Status::OK()),
          "Reporting the end of an upload to the listener failed");
      pb::PutResult pb_result;
      // Not a local variable: StartWrite requires the ByteBuffer to remain
      // valid until OnWriteDone.
      write_buf_ = MakeWriteBuffer(pb_result);
      StartWrite(&write_buf_);
      return;
    }
    // Extract an Arrow buffer from the gRPC ByteBuffer, then feed it to the
    // FlightMessageDecoder which fires the listener callbacks (OnSchemaDecoded / OnNext).
    std::shared_ptr<arrow::Buffer> arrow_buf;
    const Status wrap_status = WrapGrpcBuffer(&request_buf_, &arrow_buf);
    if (!wrap_status.ok()) {
      FinishUpload(
          MakeFlightError(FlightStatusCode::Internal,
                          "Failed to wrap gRPC buffer: " + wrap_status.message()));
      return;
    }

    // Feed the Arrow buffer to the FlightMessageDecoder. This will trigger the
    // appropriate callbacks on the listener (OnSchemaDecoded / OnNext).
    const Status decode_status = decoder_.Consume(std::move(arrow_buf));
    if (!decode_status.ok()) {
      // The listener's status is the transport error rejecting the upload.
      FinishUpload(decode_status);
      return;
    }

    // Read next FlightData.
    StartRead(&request_buf_);
  }

  /// \brief Called when a write to the client has completed.
  /// \param `ok` indicates whether the write was successful. If false, the client has
  /// closed its half of the stream.
  void OnWriteDone(bool ok) override {
    if (!ok) {
      // The client closed its half of the stream before the write could complete.
      FinishUpload(MakeFlightError(FlightStatusCode::Internal, "Write failed"));
      return;
    }
    // The acknowledgement of the upload reached the client: the upload is done.
    FinishUpload(arrow::Status::OK());
  }

  /// \brief Called when the client cancels the RPC.
  void OnCancel() override {
    // The client went away before ending the upload: the application must hear
    // that the upload will not complete.
    ARROW_WARN_NOT_OK(
        internal::FlightDataListenerTransport::ReportFinish(
            listener_, Status::Cancelled("the client cancelled the upload")),
        "Reporting the cancellation of an upload to the listener failed");
  }

  /// \brief Called when the RPC is done, regardless of success, failure, or cancellation.
  void OnDone() override {
    // The RPC is over: the listener must not reach this reactor any more.
    internal::FlightDataListenerTransport::Clear(listener_);
    ReleaseHold();
  }

 private:
  // gRPC's server callback API has no holds, so the reactor refcounts itself:
  // one reference for the RPC (released here) plus one per call in flight on
  // another thread (CancelUpload).  Whichever thread releases the last
  // reference deletes the reactor.  This is the DoGet reactor's idiom.
  void Hold() { refs_.fetch_add(1, std::memory_order_relaxed); }
  void ReleaseHold() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  /// \brief Report the upload's ending to the listener, then finish the RPC.
  ///
  /// The single place both happen, so no ending can finish the RPC without
  /// telling the listener (the listener's OnFinish fires at most once, so the
  /// endings that race each other still report only one).
  /// \param[in] status the upload's terminal status.
  void FinishUpload(Status status) {
    ARROW_WARN_NOT_OK(
        internal::FlightDataListenerTransport::ReportFinish(listener_, status),
        "Reporting the end of an upload to the listener failed");
    FinishOnce(flight_context_.FinishRequest(std::move(status)));
  }

  /// \brief Ensures that gRPC's Finish is called at most once, and never after OnDone.
  /// \param[in] status The gRPC status to finish the call with.
  void FinishOnce(::grpc::Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true)) {
      Finish(std::move(status));
    }
  }

  GrpcServerCallContext<::grpc::CallbackServerContext> flight_context_;
  std::shared_ptr<FlightDataListener> listener_;
  FlightMessageDecoder decoder_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
  /// Set by the first ending: gRPC's Finish runs once, and so does each
  /// terminal report to the listener.
  std::atomic<bool> finished_{false};
  /// One reference for the RPC (released in OnDone) plus one per call in
  /// flight on another thread (CancelUpload).
  std::atomic<int> refs_{1};
};

/// \brief Serve the Handshake RPC over the generic callback API.
///
/// One read (the client's request), one call into the server's Handshake hook,
/// one write (the response).  The hook runs inline on the callback thread, so it
/// must not block for long.
class HandshakeReactor : public ::grpc::ServerGenericBidiReactor {
 public:
  /// `flight_context` is prepared by the service (middleware ran; the token
  /// check does not apply to Handshake), and `handshake_handler` is the server
  /// class's Handshake hook.  The raw gRPC context is owned by gRPC and
  /// outlives the reactor (which is deleted in OnDone).
  HandshakeReactor(AsyncCallContext flight_context, HandshakeFn handshake_handler)
      : flight_context_(std::move(flight_context)),
        handshake_handler_(std::move(handshake_handler)) {
    StartRead(&request_buf_);
  }

  /// \brief Called when the read of the request has completed.
  /// \param[in] ok Whether the read was successful.
  void OnReadDone(bool ok) override {
    if (!ok) {
      Finish(flight_context_.FinishRequest(
          MakeFlightError(FlightStatusCode::Internal, "Failed to read request")));
      return;
    }
    auto request = ParseRequest();
    if (!request.ok()) {
      Finish(flight_context_.FinishRequest(std::move(request).status()));
      return;
    }
    std::string response;
    const auto status = handshake_handler_(flight_context_, *request, &response);
    if (!status.ok()) {
      Finish(flight_context_.FinishRequest(status));
      return;
    }
    pb::HandshakeResponse pb_response;
    pb_response.set_payload(std::move(response));
    // Not a local: StartWrite requires the buffer to remain valid until OnWriteDone.
    write_buf_ = MakeWriteBuffer(pb_response);
    StartWrite(&write_buf_);
  }

  /// \brief Called when the write of the response has completed.
  /// \param[in] ok Whether the write was successful.
  void OnWriteDone(bool ok) override {
    // A failed read is the only other terminal path, and it never reaches the
    // write, so exactly one of the two calls Finish().
    Finish(flight_context_.FinishRequest(
        ok ? arrow::Status::OK()
           : MakeFlightError(FlightStatusCode::Internal, "Failed to write response")));
  }

  /// \brief Called when the RPC is fully done, regardless of success or
  /// cancellation.
  void OnDone() override { delete this; }

 private:
  /// Parse the request ByteBuffer as the pb::HandshakeRequest of the handshake
  /// and return its payload (the client's password).
  arrow::Result<std::string> ParseRequest() {
    ARROW_ASSIGN_OR_RAISE(auto request, ParseProtoRequest<pb::HandshakeRequest>(
                                            request_buf_, "HandshakeRequest"));
    return request.payload();
  }

  AsyncCallContext flight_context_;
  HandshakeFn handshake_handler_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
};

/// One request message in, one response message out, then finish.
///
/// The RPCs served here (the three whose request is a FlightDescriptor:
/// GetFlightInfo, GetSchema, PollFlightInfo) differ only in the server class
/// method that runs and the response type, so the handler supplied by the
/// service does both while the reactor does the request parsing, the
/// "handler returned nothing" answer and the serialization.
template <typename PbResponseT>
class UnaryReactor final : public ::grpc::ServerGenericBidiReactor {
 public:
  /// Convert the request, call the server class, serialize the response.
  using HandlerFn = std::function<arrow::Status(const ServerCallContext&,
                                                const FlightDescriptor&, PbResponseT*)>;

  UnaryReactor(AsyncCallContext flight_context, HandlerFn handler)
      : flight_context_(std::move(flight_context)), handler_(std::move(handler)) {
    StartRead(&request_buf_);
  }

  void OnReadDone(bool ok) override {
    if (!ok) {
      Finish(flight_context_.FinishRequest(
          MakeFlightError(FlightStatusCode::Internal, "Failed to read request")));
      return;
    }
    const auto descriptor = ParseProtoRequest<pb::FlightDescriptor, FlightDescriptor>(
        request_buf_, "FlightDescriptor");
    if (!descriptor.ok()) {
      Finish(flight_context_.FinishRequest(descriptor.status()));
      return;
    }
    PbResponseT response;
    const auto status = handler_(flight_context_, *descriptor, &response);
    if (!status.ok()) {
      Finish(flight_context_.FinishRequest(status));
      return;
    }
    response_buf_ = MakeWriteBuffer(response);
    StartWrite(&response_buf_);
  }

  void OnWriteDone(bool ok) override {
    // The failed-read path above never reaches the write, so exactly one of
    // the two calls Finish().
    Finish(flight_context_.FinishRequest(
        ok ? arrow::Status::OK()
           : MakeFlightError(FlightStatusCode::Internal, "Failed to write response")));
  }

  void OnDone() override { delete this; }

 private:
  AsyncCallContext flight_context_;
  HandlerFn handler_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer response_buf_;
};

/// One request message in, N response messages out, then finish.
///
/// One message is in flight at a time: OnWriteDone asks for the next, so the
/// loop costs one callback per message and no extra state.
class StreamingReactor : public ::grpc::ServerGenericBidiReactor {
 public:
  StreamingReactor(AsyncCallContext flight_context, AsyncGenericFlightServerBase* base)
      : flight_context_(std::move(flight_context)), base_(base) {
    StartRead(&request_buf_);
  }

  void OnReadDone(bool ok) override {
    if (!ok) {
      Finish(flight_context_.FinishRequest(
          MakeFlightError(FlightStatusCode::Internal, "Failed to read request")));
      return;
    }
    const auto status = HandleRequest();
    if (!status.ok()) {
      Finish(flight_context_.FinishRequest(status));
      return;
    }
    WriteNext();
  }

  void OnWriteDone(bool ok) override {
    if (!ok) {
      Finish(flight_context_.FinishRequest(
          MakeFlightError(FlightStatusCode::Internal, "Failed to write response")));
      return;
    }
    WriteNext();
  }

  void OnDone() override { delete this; }

 protected:
  /// Consume the request. Called once, before the first WriteNext().
  virtual arrow::Status HandleRequest() = 0;
  /// Serialize the next message into `write_buf_` and return true, or return
  /// false when the response stream is exhausted.
  virtual arrow::Result<bool> NextMessage() = 0;

  void WriteNext() {
    auto has_next = NextMessage();
    if (!has_next.ok()) {
      Finish(flight_context_.FinishRequest(std::move(has_next).status()));
      return;
    }
    if (!*has_next) {
      // Nothing left to send: finish without ever arming a write, so the RPC
      // does not wait for an OnWriteDone that will not come.
      Finish(flight_context_.FinishRequest(arrow::Status::OK()));
      return;
    }
    StartWrite(&write_buf_);
  }

  AsyncCallContext flight_context_;
  AsyncGenericFlightServerBase* base_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
};

/// \brief Serve one ListActions RPC: the server's action types are collected
/// up front (a plain vector, no iterator) and written one per OnWriteDone.
class ListActionsReactor final : public StreamingReactor {
 public:
  using StreamingReactor::StreamingReactor;

 protected:
  arrow::Status HandleRequest() override {
    // The request is pb::Empty; nothing to parse.
    std::vector<ActionType> actions;
    ARROW_RETURN_NOT_OK(base_->ListActions(flight_context_, &actions));
    actions_ = std::move(actions);
    next_ = 0;
    return arrow::Status::OK();
  }

  arrow::Result<bool> NextMessage() override {
    if (next_ >= actions_.size()) {
      return false;
    }
    pb::ActionType pb_type;
    ARROW_RETURN_NOT_OK(internal::ToProto(actions_[next_++], &pb_type));
    write_buf_ = MakeWriteBuffer(pb_type);
    return true;
  }

 private:
  std::vector<ActionType> actions_;
  size_t next_ = 0;
};

/// \brief Serve one DoAction RPC: the server's ResultStream is pulled one
/// Result per OnWriteDone until it reports the end of stream.
class DoActionReactor final : public StreamingReactor {
 public:
  using StreamingReactor::StreamingReactor;

 protected:
  arrow::Status HandleRequest() override {
    ARROW_ASSIGN_OR_RAISE(
        auto action, (ParseProtoRequest<pb::Action, Action>(request_buf_, "Action")));
    return base_->DoAction(flight_context_, action, &results_);
  }

  arrow::Result<bool> NextMessage() override {
    if (results_ == nullptr) {
      // A handler that returned OK without a stream answers CANCELLED on the
      // sync transport (grpc_server.cc:404-406), before any message is
      // written: same answer, same place.
      return arrow::Status::Cancelled();
    }
    ARROW_ASSIGN_OR_RAISE(auto result, results_->Next());
    // A null Result is the end-of-stream sentinel, as in the sync
    // WriteStream/DoAction loops (grpc_server.cc:408-421).
    if (result == nullptr) {
      return false;
    }
    pb::Result pb_result;
    ARROW_RETURN_NOT_OK(internal::ToProto(*result, &pb_result));
    write_buf_ = MakeWriteBuffer(pb_result);
    return true;
  }

 private:
  std::unique_ptr<ResultStream> results_;
};

/// \brief Serve one ListFlights RPC: the server's FlightListing is pulled one
/// FlightInfo per OnWriteDone until it is exhausted.
class ListFlightsReactor final : public StreamingReactor {
 public:
  using StreamingReactor::StreamingReactor;

 protected:
  arrow::Status HandleRequest() override {
    ARROW_ASSIGN_OR_RAISE(auto criteria, (ParseProtoRequest<pb::Criteria, Criteria>(
                                             request_buf_, "Criteria")));
    return base_->ListFlights(flight_context_, &criteria, &listing_);
  }

  arrow::Result<bool> NextMessage() override {
    // A null listing is "no flights available", as in the sync transport
    // (grpc_server.cc:258-261) -- zero messages, then OK.
    if (listing_ == nullptr) {
      return false;
    }
    ARROW_ASSIGN_OR_RAISE(auto info, listing_->Next());
    // Exhaustion is the listing returning null (grpc_server.cc:186-206).
    if (info == nullptr) {
      return false;
    }
    pb::FlightInfo pb_info;
    ARROW_RETURN_NOT_OK(internal::ToProto(*info, &pb_info));
    write_buf_ = MakeWriteBuffer(pb_info);
    return true;
  }

 private:
  std::unique_ptr<FlightListing> listing_;
};

// ---------------------------------------------------------------------------
// DoExchange: the bidirectional RPC.
//
// Its handler is synchronous user code (a FlightMessageReader + a
// FlightMessageWriter), so unlike every other reactor here it cannot run on a
// callback thread: the handler blocks while the reactor's callbacks complete
// the operation on another gRPC thread.  `ExchangeDataStream` is the
// internal::ServerDataStream that bridges the two, and the reader/writer
// classes the Arrow sync path already has (TransportMessageReader /
// TransportMessageWriter) run on top of it untouched.
//
// Invariants (the class below is the only enforcement):
//   1. At most one read outstanding at a time: `ReadData()` is the only place
//      that calls StartRead, and it does not return until the read it posted
//      has completed (that is the condvar predicate it waits on) -- so a second
//      StartRead can never be posted while the first is in flight.
//   2. At most one write outstanding at a time: `WriteData()` waits out any
//      previous write before arming the next one, and does not return until the
//      write it armed has completed; only `OnWriteDone` clears the flag that
//      wait tests.
//   3. Finish() is called exactly once, from whichever terminal path fires
//      first (the handler returned, or the request could not be read at all);
//      the CAS in FinishOnce() is what makes that true.  On a cancelled call
//      the unblocked handler still returns and still reaches that one Finish.
//   4. OnDone() deletes the reactor, and the handler runs on the same thread
//      that entered OnReadDone (RunHandler() is called inline from it), so it
//      cannot outlive the reactor: gRPC delivers OnDone only after that
//      callback returns (grpcpp/impl/server_callback_handlers.h: it calls
//      reactor->OnReadDone(ok) and only then MaybeDone()).
//
// ponytail: holding the callback thread for the whole exchange is the accepted
// tradeoff (matches the sync transport's thread-per-exchange behaviour). The
// cost is real -- a long exchange occupies one thread from the pool, and the
// pool grows ~1 thread/s while it is held -- but it buys a synchronous handler
// with no extra thread per exchange and no new public API.  Upgrade path: run
// the handler on a Flight thread pool and keep a Hold() on the reactor, if
// DoExchange throughput ever matters.
class ExchangeReactor;

/// The internal::ServerDataStream the sync reader/writer classes see; it
/// forwards each call to the reactor, which blocks until the corresponding
/// gRPC callback completes.  (Defined out-of-line, after ExchangeReactor.)
class ExchangeDataStream final : public internal::ServerDataStream {
 public:
  explicit ExchangeDataStream(ExchangeReactor* reactor);

  bool ReadData(internal::FlightData* data) override;

  arrow::Result<bool> WriteData(const FlightPayload& payload) override;

  // Not offered by the exchange: DoExchange's writer has no PutResult channel,
  // so the default (NotImplemented) is the honest answer and WriteMetadata()
  // goes through the data path instead (payload.app_metadata).
  Status WritePutMetadata(const Buffer& payload) override;

  Status WritesDone() override;

 private:
  ExchangeReactor* reactor_;
};

/// Serve one DoExchange RPC.
///
/// The handler runs inline in `RunHandler()`, which this reactor calls from the
/// first OnReadDone (i.e. on a gRPC callback thread): the server class's
/// synchronous DoExchange gets the sync TransportMessageReader/Writer over the
/// ExchangeDataStream.  While the handler runs, its blocking
/// ReadData()/WriteData() wait on a mutex + condvar for the OnReadDone /
/// OnWriteDone that gRPC delivers on its own threads.
class ExchangeReactor final : public ::grpc::ServerGenericBidiReactor {
 public:
  ExchangeReactor(AsyncCallContext flight_context, AsyncGenericFlightServerBase* base,
                  std::shared_ptr<MemoryManager> memory_manager)
      : flight_context_(std::move(flight_context)),
        base_(base),
        memory_manager_(std::move(memory_manager)) {
    // The first message carries the request descriptor; the reader's Init()
    // consumes it (see ReadData below).
    StartRead(&read_buf_);
  }

  // --- gRPC callbacks; every one of these runs on a gRPC thread -----------

  /// One message arrived (or the read side ended).  The first message starts
  /// the handler; every later one is buffered for it.
  void OnReadDone(bool ok) override {
    std::unique_lock<std::mutex> lock(mutex_);
    read_pending_ = false;
    if (!ok) {
      // The client half-closed (or the stream broke): the read side is done,
      // but the exchange is not -- the handler may still be writing, which is
      // the read-all-then-write-all shape.
      read_finished_ = true;
      if (!handler_started_) {
        // The client never sent the descriptor: there is no exchange to run.
        lock.unlock();
        FinishOnce(flight_context_.FinishRequest(
            MakeFlightError(FlightStatusCode::Internal, "Failed to read request")));
        return;
      }
      condvar_.notify_all();
      return;
    }
    const ::grpc::Status deserialized = FlightDataDeserialize(&read_buf_, &read_data_);
    if (!deserialized.ok()) {
      // A message that cannot be deserialized breaks the whole stream: nothing
      // written afterwards would make sense, so this is terminal.
      deserialize_error_ =
          MakeFlightError(FlightStatusCode::Internal, deserialized.error_message());
      read_finished_ = true;
      cancelled_ = true;
      condvar_.notify_all();
      return;
    }
    pending_read_ = std::move(read_data_);
    has_pending_read_ = true;
    const bool is_first = !handler_started_;
    if (is_first) {
      handler_started_ = true;
    }
    condvar_.notify_all();
    if (is_first) {
      // Invariant 4: this is the thread the handler runs on, so the handler
      // cannot outlive the reactor (OnDone is delivered only after this
      // callback returns -- see the class comment).
      lock.unlock();
      RunHandler();
    }
  }

  void OnWriteDone(bool ok) override {
    std::unique_lock<std::mutex> lock(mutex_);
    // Invariant 2: the write that just completed is the only one outstanding.
    write_pending_ = false;
    write_ok_ = ok;
    if (!ok) {
      // No further write-side operation will succeed (grpcpp/support/
      // server_callback.h): stop the handler's writer from trying again.
      write_failed_ = true;
    }
    condvar_.notify_all();
  }

  void OnCancel() override {
    std::unique_lock<std::mutex> lock(mutex_);
    // A cancelled call's outstanding read/write never complete: unblock the
    // handler so it can unwind.  The call is already being torn down, so there
    // is no status left to report (gRPC drops it).
    cancelled_ = true;
    read_finished_ = true;
    write_failed_ = true;
    condvar_.notify_all();
  }

  /// Invariant 4: this is the last callback for the RPC, and the handler ran to
  /// completion inside OnReadDone before it, so nothing can touch the reactor
  /// after this delete.
  void OnDone() override { delete this; }

  // --- called by the handler (through ExchangeDataStream) -----------------

  /// Block until a message is available, or report the end of the read side.
  bool ReadData(internal::FlightData* data) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (read_finished_ || cancelled_) {
      return false;
    }
    if (has_pending_read_) {
      *data = std::move(pending_read_);
      has_pending_read_ = false;
      return true;
    }
    // Invariant 1: exactly one read outstanding -- the one whose OnReadDone
    // this call now waits for.  There can be no other, because this method
    // blocks until that read completes.
    read_pending_ = true;
    StartRead(&read_buf_);
    condvar_.wait(lock, [this] {
      return !read_pending_ || has_pending_read_ || read_finished_ || cancelled_;
    });
    read_pending_ = false;
    if (has_pending_read_) {
      *data = std::move(pending_read_);
      has_pending_read_ = false;
      return true;
    }
    return false;
  }

  /// Serialize and post one payload, then wait for its write to complete.
  arrow::Result<bool> WriteData(const FlightPayload& payload) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (write_failed_ || cancelled_) {
      return false;
    }
    // Invariant 2: wait out any previous write before arming this one, so at
    // most one write is ever outstanding.  (Only this thread writes, so the
    // wait is a no-op in practice -- it is here so the invariant holds even if
    // that ever changes.)
    condvar_.wait(lock, [this] { return !write_pending_ || cancelled_; });
    if (cancelled_) {
      return false;
    }
    bool own_buffer = false;
    // Mirror the sync path's WritePayload(), which validates before writing:
    // the DoGet reactor's inline serialize does not, but the exchange reuses
    // the sync reader/writer stack, and a payload that cannot be written must
    // be a status here and not gRPC's own assertion ("returning error here
    // causes gRPC to fail an assertion" -- serialization_internal.cc).
    ARROW_RETURN_NOT_OK(payload.Validate());
    const ::grpc::Status serialize =
        FlightDataSerialize(payload, &write_buf_, &own_buffer);
    // own_buffer is true whenever SerializeAsString-style buffers were built
    // (serialization_internal.cc:201); the reactor owns write_buf_ either way,
    // so there is nothing to free here -- kept to mirror the sync call shape.
    (void)own_buffer;
    if (!serialize.ok()) {
      return MakeFlightError(FlightStatusCode::Internal, serialize.error_message());
    }
    write_pending_ = true;
    write_ok_ = false;
    StartWrite(&write_buf_);
    condvar_.wait(lock, [this] { return !write_pending_ || cancelled_; });
    if (!write_ok_) {
      // The client went away (or the write failed): report "not accepted", the
      // transport-level disconnect signal the writer stack turns into an
      // error.
      return false;
    }
    return true;
  }

  /// The client's writes are done: nothing is outstanding, so there is nothing
  /// to flush.  gRPC's generic callback API has no server-side WritesDone
  /// (StartWritesDone exists only on the client, grpcpp/support/
  /// client_callback.h), and DoExchange does not need one: unlike DoPut, whose
  /// client waits for a PutResult, the exchange's client sees the RPC status
  /// from Finish().
  Status WritesDone() { return Status::OK(); }

 private:
  /// The handler body, mirroring internal::ServerTransport::DoExchange
  /// (transport_server.cc:49-58): the sync reader/writer stack over this
  /// reactor's stream.  `reader->Init()` reads the descriptor (the message the
  /// first OnReadDone buffered), so a client that never sends one makes Init()
  /// report it.
  void RunHandler() {
    auto stream = std::make_unique<ExchangeDataStream>(this);
    std::unique_ptr<internal::TransportMessageReader> reader(
        new internal::TransportMessageReader(stream.get(), memory_manager_));
    std::unique_ptr<FlightMessageWriter> writer(
        new internal::TransportMessageWriter(stream.get()));

    Status status = reader->Init();
    if (status.ok()) {
      status = base_->DoExchange(flight_context_, std::move(reader), std::move(writer));
    }
    if (status.ok()) {
      // The last line of the sync ServerTransport::DoExchange; WritesDone() is
      // OK by construction here (see its comment).
      status = stream->WritesDone();
    }
    // A broken stream must not look like a clean finish.  Taken under the lock:
    // the callback that recorded it ran on another thread.
    if (status.ok() || cancelled_) {
      status = TakeDeserializeError();
    }
    FinishOnce(flight_context_.FinishRequest(std::move(status)));
  }

  /// Move out the status of a message that could not be deserialized (empty
  /// when every message was fine).  Locked: OnReadDone writes it.
  Status TakeDeserializeError() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::move(deserialize_error_);
  }

  /// Invariant 3, the only Finish(): the CAS is the guard, and every
  /// background callback releases its reference only after it returns
  /// (grpcpp/support/server_callback.h), so OnDone follows the last of them.
  void FinishOnce(::grpc::Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      Finish(std::move(status));
    }
  }

  AsyncCallContext flight_context_;
  AsyncGenericFlightServerBase* base_;
  std::shared_ptr<MemoryManager> memory_manager_;

  std::mutex mutex_;
  std::condition_variable condvar_;
  // At most one buffered message, owned here while the handler consumes it.
  internal::FlightData pending_read_;
  bool has_pending_read_ = false;
  bool handler_started_ = false;
  // Invariant 1 / 2 flags: at most one outstanding operation per direction.
  bool read_pending_ = false;
  bool write_pending_ = false;
  bool write_ok_ = false;
  // The read side ended (the client half-closed or the stream broke); writes
  // may still be outstanding and are still allowed.
  bool read_finished_ = false;
  // No further write-side operation can succeed.
  bool write_failed_ = false;
  // The client went away: every wait returns immediately so the handler can
  // unwind, and the status it returns is dropped by the cancelled call.
  bool cancelled_ = false;
  // Set when a message could not be deserialized; reported in RunHandler.
  Status deserialize_error_;
  std::atomic<bool> finished_{false};
  // gRPC reads into `read_buf_` and serializes `write_buf_`; both must stay
  // valid (and unmodified) until their callback arrives.
  ::grpc::ByteBuffer read_buf_;
  ::grpc::ByteBuffer write_buf_;
  internal::FlightData read_data_;
};

// ExchangeDataStream, now that ExchangeReactor is complete.
ExchangeDataStream::ExchangeDataStream(ExchangeReactor* reactor) : reactor_(reactor) {}

bool ExchangeDataStream::ReadData(internal::FlightData* data) {
  return reactor_->ReadData(data);
}

arrow::Result<bool> ExchangeDataStream::WriteData(const FlightPayload& payload) {
  return reactor_->WriteData(payload);
}

Status ExchangeDataStream::WritePutMetadata(const Buffer& payload) {
  (void)payload;
  return Status::NotImplemented(
      "DoExchange writes application metadata through its data stream");
}

Status ExchangeDataStream::WritesDone() { return reactor_->WritesDone(); }

// Reject unknown methods.  Finish() in the constructor is fine: gRPC backlogs
// operations issued before the reactor is returned to it.
class Unimplemented : public ::grpc::ServerGenericBidiReactor {
 public:
  explicit Unimplemented(::grpc::Status status) : status_(std::move(status)) {
    Finish(status_);
  }
  void OnDone() override { delete this; }

 private:
  ::grpc::Status status_;
};

}  // namespace

AsyncGenericFlightService::AsyncGenericFlightService(
    AsyncGenericFlightServerBase* async_base,
    std::shared_ptr<MemoryManager> memory_manager,
    std::shared_ptr<GrpcServerCallContextHelper<::grpc::CallbackServerContext>> helper,
    HandshakeFn handshake_handler)
    : base_(async_base),
      memory_manager_(std::move(memory_manager)),
      helper_(std::move(helper)),
      handshake_handler_(std::move(handshake_handler)) {}

::grpc::ServerGenericBidiReactor* AsyncGenericFlightService::CreateReactor(
    ::grpc::GenericCallbackServerContext* context) {
  // gRPC hands the reactor the full path
  // ("/arrow.flight.protocol.FlightService/DoGet"), while the dispatch below
  // compares bare method names.
  const std::string_view full_method = context->method();
  const std::string_view method =
      full_method.starts_with(kPrefix) ? full_method.substr(kPrefix.size()) : full_method;
  // The enum keeps middleware able to switch on the method;
  // unknown names map to FlightMethod::Invalid.
  const FlightMethod flight_method = MethodFromName(full_method);
  AsyncCallContext flight_context(context);

  // Handshake is how a client obtains a token: middleware runs, the token
  // check does not.  Every other method - implemented here or not - is
  // authenticated first, so an unauthenticated caller never learns whether
  // a method exists.
  const auto prepare_status =
      method == kHandshakeMethod
          ? helper_->MakeCallContext(flight_method, context, &flight_context)
          : helper_->CheckAuth(flight_method, context, &flight_context);
  if (!prepare_status.ok()) {
    return new Unimplemented(prepare_status);
  }

  if (method == kHandshakeMethod) {
    if (!handshake_handler_) {
      return new Unimplemented(::grpc::Status(
          ::grpc::StatusCode::UNIMPLEMENTED,
          "This service does not have an authentication mechanism enabled."));
    }
    return new HandshakeReactor(std::move(flight_context), handshake_handler_);
  }

  if (method == kGetFlightInfoMethod) {
    return new UnaryReactor<pb::FlightInfo>(
        std::move(flight_context),
        [base = base_](const ServerCallContext& context,
                       const FlightDescriptor& descriptor,
                       pb::FlightInfo* out) -> arrow::Status {
          std::unique_ptr<FlightInfo> info;
          ARROW_RETURN_NOT_OK(base->GetFlightInfo(context, descriptor, &info));
          return SerializeOrNotFound(info, out);
        });
  }

  if (method == kGetSchemaMethod) {
    return new UnaryReactor<pb::SchemaResult>(
        std::move(flight_context),
        [base = base_](const ServerCallContext& context,
                       const FlightDescriptor& descriptor,
                       pb::SchemaResult* out) -> arrow::Status {
          std::unique_ptr<SchemaResult> schema;
          ARROW_RETURN_NOT_OK(base->GetSchema(context, descriptor, &schema));
          return SerializeOrNotFound(schema, out);
        });
  }

  if (method == kPollFlightInfoMethod) {
    return new UnaryReactor<pb::PollInfo>(
        std::move(flight_context),
        [base = base_](const ServerCallContext& context,
                       const FlightDescriptor& descriptor,
                       pb::PollInfo* out) -> arrow::Status {
          std::unique_ptr<PollInfo> info;
          ARROW_RETURN_NOT_OK(base->PollFlightInfo(context, descriptor, &info));
          return SerializeOrNotFound(info, out);
        });
  }

  if (method == kListActionsMethod) {
    return new ListActionsReactor(std::move(flight_context), base_);
  }

  if (method == kDoActionMethod) {
    return new DoActionReactor(std::move(flight_context), base_);
  }

  if (method == kListFlightsMethod) {
    return new ListFlightsReactor(std::move(flight_context), base_);
  }

  if (method == kDoExchangeMethod) {
    return new ExchangeReactor(std::move(flight_context), base_, memory_manager_);
  }

  if (method == kDoGetMethod) {
    return new DoGetReactor(std::move(flight_context), base_);
  }

  if (method == kDoPutMethod) {
    // DoPut needs a listener to hand the incoming batches to; a server class
    // that returns none (the default) refuses uploads.
    std::shared_ptr<FlightDataListener> listener =
        base_->CreateDoPutListener(flight_context);
    if (!listener) {
      return new Unimplemented(
          ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED,
                         "DoPut is not implemented: no listener available"));
    }
    return new DoPutReactor(std::move(flight_context), std::move(listener));
  }
  return new Unimplemented(
      ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Unknown method"));
}

}  // namespace arrow::flight::transport::grpc
