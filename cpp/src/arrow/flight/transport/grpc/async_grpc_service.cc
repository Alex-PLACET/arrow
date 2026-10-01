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

#include "arrow/flight/transport/grpc/async_grpc_service.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <grpcpp/support/byte_buffer.h>
#include <grpcpp/support/slice.h>

#include "arrow/flight/flight_data_decoder.h"
#include "arrow/flight/protocol_internal.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/transport/grpc/serialization_internal.h"
#include "arrow/flight/types.h"
#include "arrow/ipc/writer.h"

namespace arrow::flight::transport::grpc {

namespace pb = arrow::flight::protocol;

namespace {

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
/// \tparam ProtoT The type of the proto message to serialize.
/// \param[in] message The proto message to serialize.
/// \return A gRPC ByteBuffer containing the serialized message.
template <typename ProtoT>
::grpc::ByteBuffer MakeWriteBuffer(const ProtoT& message) {
  const std::string bytes = message.SerializeAsString();
  ::grpc::Slice slice(bytes);
  return {&slice, 1};
}

/// \brief Read `buf` as the proto message `PbT`.
/// \param[in] what names the message in the "Failed to read/parse <what>"
/// errors ("Ticket", "FlightDescriptor", …).
/// \tparam PbT The type of the proto message to parse.
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
/// \tparam PbT The type of the proto message to parse.
/// \tparam T The type to convert the proto message to.
/// \param[in] buf The gRPC ByteBuffer containing the serialized proto message.
/// \param[in] what Names the message in the "Failed to read/parse <what>"
/// errors ("Ticket", "FlightDescriptor", …).
/// \return The converted message of type `T`.
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
/// \tparam T The type of the value to serialize.
/// \tparam PbT The type of the proto message to serialize into.
/// \param[in] value The value to serialize.
/// \param[out] out The proto message to serialize into.
/// \return Status indicating success or failure.
template <typename T, typename PbT>
arrow::Status SerializeOrNotFound(const std::shared_ptr<T>& value, PbT* out) {
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
      FinishOnce(MakeFlightError(FlightStatusCode::Internal, "Failed to read request"));
      return;
    }

    const auto ticket = ParseTicket();
    if (!ticket.ok()) {
      FinishOnce(ticket.status());
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
          if (finished_) {
            // The RPC is over (cancelled while the future was pending): the
            // arrived stream is dropped unwritten.
            ReleaseHold();
            return;
          }
          if (!result.ok()) {
            FinishOnce(result.status());
            ReleaseHold();
            return;
          }
          async_data_stream_ = std::move(*future.MoveResult());
          if (async_data_stream_ == nullptr) {
            FinishOnce(arrow::Status::KeyError("No data in this flight"));
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
    if (finished_) {
      return;  // the RPC is over. gRPC may still deliver this.
    }
    if (!ok) {
      // The write failed (usually because the client went away): the
      // transport's diagnosis is the final status, and the producer is told
      // to stop.
      FinishOnce(MakeFlightError(FlightStatusCode::Internal, "Write failed"));
      ARROW_WARN_NOT_OK(CloseStreamOnce(),
                        "DoGet: closing the data stream after a failed write failed");
      return;
    }
    // Continue writing the next payload.
    WriteNextPayload();
  }

  /// \brief Called when the client cancels the request.
  void OnCancel() override {
    FinishOnce(arrow::Status::Cancelled());
    ARROW_WARN_NOT_OK(CloseStreamOnce(),
                      "DoGet: closing the stream of a cancelled call failed");
  }

  /// \brief Called when the RPC is fully done, regardless of success or cancellation.
  /// This is the last callback that will be invoked for this RPC.
  void OnDone() override { ReleaseHold(); }

 private:
  /// Parse the request ByteBuffer as the pb::Ticket of the DoGet request.
  /// \return A Result containing the parsed Ticket, or an error if parsing failed.
  arrow::Result<Ticket> ParseTicket() {
    return ParseProtoRequest<pb::Ticket, Ticket>(request_buf_, "Ticket");
  }

  /// \brief Ensures that gRPC's Finish is called at most once, and never
  /// after OnDone.
  void FinishOnce(Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true)) {
      Finish(flight_context_.FinishRequest(status));
    }
  }

  /// \brief Close the producer at most once, from whichever terminal path
  /// runs first.
  /// \return Status indicating success or failure of closing the stream.
  Status CloseStreamOnce() {
    bool expected = false;
    if (!closed_.compare_exchange_strong(expected, true)) {
      return Status::OK();
    }
    if (async_data_stream_ == nullptr) {
      return Status::OK();
    }
    return async_data_stream_->Close();
  }

  // gRPC's server callback API has no holds, so the reactor refcounts itself:
  // one reference for the RPC (released in OnDone) plus one per pending
  // continuation. Whichever thread releases the last reference deletes it.
  void Hold() { refs_.fetch_add(1, std::memory_order_relaxed); }
  void ReleaseHold() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  /// Serialize the next payload of the data stream and start writing it.
  /// Called once from OnReadDone (schema payload) and then once per OnWriteDone.
  void WriteNextPayload() {
    if (finished_) {
      return;  // the RPC is over; gRPC may still deliver this
    }
    Hold();  // the payload may complete on the server's own thread

    arrow::Future<std::optional<FlightPayload>> next;
    if (!wrote_schema_) {
      next = async_data_stream_->GetSchemaPayloadAsync().Then(
          [](FlightPayload payload) -> std::optional<FlightPayload> {
            return {std::move(payload)};
          });
    } else {
      next = async_data_stream_->NextAsync();
    }
    next.AddCallback([this](arrow::Result<std::optional<FlightPayload>> result) {
      if (finished_) {
        // The RPC is over (cancelled): the payload is abandoned.
        ReleaseHold();
        return;
      }
      if (!result.ok()) {
        FinishOnce(result.status());
        ReleaseHold();
        return;
      }

      wrote_schema_ = true;
      std::optional<FlightPayload> payload = std::move(*result);

      // End of stream: the last payload has no metadata.
      // The producer's Close() status is the final status.
      if (!payload.has_value()) {
        FinishOnce(CloseStreamOnce());
        ReleaseHold();
        return;
      }

      bool own_buffer = false;
      const ::grpc::Status grpc_status =
          FlightDataSerialize(*payload, &write_buf_, &own_buffer);

      if (!grpc_status.ok()) {
        FinishOnce(
            MakeFlightError(FlightStatusCode::Internal, grpc_status.error_message()));
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
  /// Set by the first FinishOnce: a pending continuation must not write or
  /// touch the call context afterwards.
  std::atomic<bool> finished_{false};
  /// Set by the first CloseStreamOnce.
  std::atomic<bool> closed_{false};
  /// One reference for the RPC (released in OnDone) plus one per pending
  /// continuation.
  std::atomic<int> refs_{1};
};

/// \brief Serve one DoPut RPC over the generic callback API.
///
/// Reads one message per OnReadDone turn and pushes it into the per-RPC
/// AsyncFlightMessageDecoder, which fires the listener's callbacks
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
               std::shared_ptr<AsyncFlightDataListener> listener)
      : flight_context_(std::move(flight_context)),
        listener_(std::move(listener)),
        decoder_(listener_) {
    internal::FlightDataListenerTransport::Install(listener_, this);
    StartRead(&request_buf_);
  }

  /// \brief Cancel the upload from the application side: finish the RPC with
  /// `status` without waiting for the client to end it.
  /// \param[in] `status` the status to finish the upload with. Must be non-OK.
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
      if (finished_.load()) {
        return;  // an ending already claimed the RPC
      }
      pb::PutResult pb_result;
      write_buf_ = MakeWriteBuffer(pb_result);
      StartWrite(&write_buf_);
      return;
    }
    // Extract an Arrow buffer from the gRPC ByteBuffer, then feed it to the
    // AsyncFlightMessageDecoder which fires the listener callbacks (OnSchemaDecoded /
    // OnNext).
    std::shared_ptr<arrow::Buffer> arrow_buf;
    const Status wrap_status = WrapGrpcBuffer(&request_buf_, &arrow_buf);
    if (!wrap_status.ok()) {
      FinishUpload(
          MakeFlightError(FlightStatusCode::Internal,
                          "Failed to wrap gRPC buffer: " + wrap_status.message()));
      return;
    }

    // Feed the Arrow buffer to the AsyncFlightMessageDecoder. This will trigger the
    // appropriate callbacks on the listener (OnSchemaDecoded / OnNext).
    Future<> decode_status = decoder_.Consume(std::move(arrow_buf));
    Hold();
    decode_status.AddCallback([this](arrow::Status status) {
      if (finished_) {
        ReleaseHold();
        return;
      }
      if (!status.ok()) {
        // The listener's status is the transport error rejecting the upload.
        FinishUpload(std::move(status));
        ReleaseHold();
        return;
      }
      // Read next FlightData.
      StartRead(&request_buf_);
      ReleaseHold();
    });
  }

  /// \brief Called when a write to the client has completed.
  /// \param `ok` indicates whether the write was successful. If false, the client has
  /// closed its half of the stream.
  void OnWriteDone(bool ok) override {
    if (finished_) {
      return;  // the RPC is over; gRPC may still deliver this
    }
    if (!ok) {
      // The client went away before the acknowledgement could be delivered:
      // the upload did not end cleanly.
      FinishUpload(MakeFlightError(FlightStatusCode::Internal, "Write failed"));
      return;
    }
    // The acknowledgement reached the client: this is the clean ending of the
    // upload, and the only path that may report OK to the listener.
    FinishUpload(arrow::Status::OK());
  }

  /// \brief Called when the client cancels the RPC.
  ///
  /// The RPC must not wait for the listener's pending future: finish here, so
  /// gRPC can deliver OnDone; a pending decode continuation stands down when
  /// (if) it runs.
  void OnCancel() override {
    Status cancel_status = Status::Cancelled("the client cancelled the upload");
    ARROW_WARN_NOT_OK(
        internal::FlightDataListenerTransport::ReportFinish(listener_, cancel_status),
        "Reporting the cancellation of an upload to the listener failed");
    FinishOnce(std::move(cancel_status));
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
  // another thread (CancelUpload). Whichever thread releases the last
  // reference deletes the reactor.
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
    FinishOnce(std::move(status));
  }

  /// \brief Ensures that gRPC's Finish is called at most once, and never
  /// after OnDone.
  void FinishOnce(Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true)) {
      Finish(flight_context_.FinishRequest(status));
    }
  }

  GrpcServerCallContext<::grpc::CallbackServerContext> flight_context_;
  std::shared_ptr<AsyncFlightDataListener> listener_;
  AsyncFlightMessageDecoder decoder_;
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
/// The handler returns a future; the response is serialized and written when
/// the future resolves, on whatever thread resolved it, so the gRPC callback
/// thread serving the request is never held by the handler.  The RPCs served
/// here (the three whose request is a FlightDescriptor: GetFlightInfo,
/// GetSchema, PollFlightInfo) differ only in the handler and the response
/// type, so `HandlerFn` does the handler while the reactor does the request
/// parsing, the "handler answered nothing" answer and the serialization.
template <typename T, typename PbT>
class UnaryReactor final : public ::grpc::ServerGenericBidiReactor {
 public:
  using HandlerFn = std::function<arrow::Future<std::shared_ptr<T>>(
      const ServerCallContext&, const FlightDescriptor&)>;

  UnaryReactor(AsyncCallContext flight_context, HandlerFn handler)
      : flight_context_(std::move(flight_context)), handler_(std::move(handler)) {
    StartRead(&request_buf_);
  }

  void OnReadDone(bool ok) override {
    if (!ok) {
      FinishOnce(MakeFlightError(FlightStatusCode::Internal, "Failed to read request"));
      return;
    }
    const auto descriptor = ParseProtoRequest<pb::FlightDescriptor, FlightDescriptor>(
        request_buf_, "FlightDescriptor");
    if (!descriptor.ok()) {
      FinishOnce(descriptor.status());
      return;
    }
    // The handler may complete its future on any thread: hold the reactor
    // across the continuation (gRPC has no server-side holds).
    Hold();
    arrow::Future<std::shared_ptr<T>> future = handler_(flight_context_, *descriptor);
    future.AddCallback(
        [this, future](const arrow::Result<std::shared_ptr<T>>& result) mutable {
          if (finished_) {
            // The RPC is over (cancelled while the future was pending).
            ReleaseHold();
            return;
          }
          if (!result.ok()) {
            FinishOnce(result.status());
            ReleaseHold();
            return;
          }
          PbT response;
          const auto status = SerializeOrNotFound(*future.MoveResult(), &response);
          if (!status.ok()) {
            FinishOnce(status);
            ReleaseHold();
            return;
          }
          response_buf_ = MakeWriteBuffer(response);
          StartWrite(&response_buf_);
          ReleaseHold();  // the next hop is a gRPC callback (OnWriteDone)
        });
  }

  void OnWriteDone(bool ok) override {
    if (finished_) {
      return;  // the RPC is over; gRPC may still deliver this
    }
    // The failed-read and cancel paths never reach the write, so exactly one
    // path calls FinishOnce.
    FinishOnce(
        ok ? arrow::Status::OK()
           : MakeFlightError(FlightStatusCode::Internal, "Failed to write response"));
  }

  /// \brief Called when the client cancels the request.
  ///
  /// The RPC must not wait for the handler's pending future: finish here, so
  /// gRPC can deliver OnDone; the continuation stands down when (if) it runs.
  void OnCancel() override { FinishOnce(arrow::Status::Cancelled()); }

  void OnDone() override { ReleaseHold(); }

 private:
  /// \brief Ensures that gRPC's Finish is called at most once, and never
  /// after OnDone.  The middleware's CallCompleted runs inside the CAS:
  /// racing finishers (a cancel against a completing continuation) must not
  /// report the call completed twice.
  void FinishOnce(Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true)) {
      Finish(flight_context_.FinishRequest(status));
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

  AsyncCallContext flight_context_;
  HandlerFn handler_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer response_buf_;
  std::atomic<bool> finished_{false};
  std::atomic<int> refs_{1};
};

/// One request message in, N response messages out, then finish.
///
/// The handler's answer arrives as a future (the listing, the result stream,
/// the action vector); the reactor then pulls one message at a time through
/// NextMessage(), each pull possibly completing later, and writes each
/// message on its own OnWriteDone turn.
class StreamingReactor : public ::grpc::ServerGenericBidiReactor {
 public:
  StreamingReactor(AsyncCallContext flight_context, AsyncGenericFlightServerBase* base)
      : flight_context_(std::move(flight_context)), base_(base) {
    StartRead(&request_buf_);
  }

  void OnReadDone(bool ok) override {
    if (!ok) {
      FinishOnce(MakeFlightError(FlightStatusCode::Internal, "Failed to read request"));
      return;
    }
    // The handler may complete its future on any thread: hold the reactor
    // across the continuation (gRPC has no server-side holds).
    Hold();
    Start().AddCallback([this](const arrow::Status& status) {
      if (finished_) {
        // The RPC is over (cancelled while the future was pending).
        ReleaseHold();
        return;
      }
      if (!status.ok()) {
        // The handler's status is the RPC's status (a handler that completed
        // OK with no stream arrives here too: the concrete reactor mapped
        // that to CANCELLED, as the sync transport does).
        FinishOnce(status);
        ReleaseHold();
        return;
      }
      WriteNextMessage();
      ReleaseHold();
    });
  }

  void OnWriteDone(bool ok) override {
    if (finished_) {
      return;  // the RPC is over; gRPC may still deliver this
    }
    if (!ok) {
      FinishOnce(MakeFlightError(FlightStatusCode::Internal, "Failed to write response"));
      return;
    }
    WriteNextMessage();
  }

  /// \brief Called when the client cancels the request.
  ///
  /// The RPC must not wait for the handler's pending future: finish here, so
  /// gRPC can deliver OnDone; the continuation stands down when (if) it runs.
  void OnCancel() override { FinishOnce(arrow::Status::Cancelled()); }

  void OnDone() override { ReleaseHold(); }

 protected:
  /// Consume the request; the returned future resolves when the handler's
  /// answer is available, with a non-OK status to fail the RPC.
  virtual arrow::Future<> Start() = 0;
  /// Serialize the next message of the response into `write_buf_`.  A future
  /// completed with true writes it; false ends the response with OK; a
  /// non-OK status fails the RPC.
  virtual arrow::Future<bool> NextMessage() = 0;

  void WriteNextMessage() {
    if (finished_) {
      return;  // the RPC is over; gRPC may still deliver this
    }
    Hold();  // the message may be produced on any thread
    NextMessage().AddCallback([this](const arrow::Result<bool>& has_next) {
      if (finished_) {
        // The RPC is over (cancelled): the message is abandoned.
        ReleaseHold();
        return;
      }
      if (!has_next.ok()) {
        FinishOnce(has_next.status());
        ReleaseHold();
        return;
      }
      if (!*has_next) {
        // Nothing left to send: finish without ever arming a write, so the
        // RPC does not wait for an OnWriteDone that will not come.
        FinishOnce(arrow::Status::OK());
        ReleaseHold();
        return;
      }
      StartWrite(&write_buf_);
      ReleaseHold();  // the next hop is a gRPC callback (OnWriteDone)
    });
  }

 protected:
  /// \brief Ensures that gRPC's Finish is called at most once, and never
  /// after OnDone.  The middleware's CallCompleted runs inside the CAS:
  /// racing finishers (a cancel against a completing continuation) must not
  /// report the call completed twice.
  void FinishOnce(Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true)) {
      Finish(flight_context_.FinishRequest(status));
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

  AsyncCallContext flight_context_;
  AsyncGenericFlightServerBase* base_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
  std::atomic<bool> finished_{false};
  std::atomic<int> refs_{1};
};

/// \brief Serve one ListActions RPC: the handler's action types are written
/// one per OnWriteDone.
class ListActionsReactor final : public StreamingReactor {
 public:
  using StreamingReactor::StreamingReactor;

 protected:
  arrow::Future<> Start() override {
    // The request is pb::Empty; nothing to parse.
    return base_->ListActionsAsync(flight_context_)
        .Then([this](const std::vector<ActionType>& actions) -> arrow::Status {
          actions_ = actions;
          next_ = 0;
          return arrow::Status::OK();
        });
  }

  arrow::Future<bool> NextMessage() override {
    if (next_ >= actions_.size()) {
      return arrow::Future<bool>::MakeFinished(false);
    }
    pb::ActionType pb_type;
    const auto status = internal::ToProto(actions_[next_++], &pb_type);
    if (!status.ok()) {
      return arrow::Future<bool>::MakeFinished(status);
    }
    write_buf_ = MakeWriteBuffer(pb_type);
    return arrow::Future<bool>::MakeFinished(true);
  }

 private:
  std::vector<ActionType> actions_;
  size_t next_ = 0;
};

/// \brief Serve one DoAction RPC: the handler's AsyncResultStream is pulled
/// one Result per OnWriteDone until it reports the end of stream.
class DoActionReactor final : public StreamingReactor {
 public:
  using StreamingReactor::StreamingReactor;

 protected:
  arrow::Future<> Start() override {
    auto action = ParseProtoRequest<pb::Action, Action>(request_buf_, "Action");
    if (!action.ok()) {
      return arrow::Future<>::MakeFinished(action.status());
    }
    return base_->DoActionAsync(flight_context_, *action)
        .Then([this](const std::shared_ptr<AsyncResultStream>& stream) -> arrow::Status {
          if (stream == nullptr) {
            // A handler that completed OK with no stream answers CANCELLED on
            // the sync transport (grpc_server.cc:404-406), before any message
            // is written: the same answer, in the same place.
            return arrow::Status::Cancelled();
          }
          stream_ = stream;
          return arrow::Status::OK();
        });
  }

  arrow::Future<bool> NextMessage() override {
    return stream_->NextAsync().Then(
        [this](const std::shared_ptr<Result>& result) -> arrow::Result<bool> {
          // A null Result is the end-of-stream sentinel, as in the sync
          // WriteStream/DoAction loops (grpc_server.cc:408-421).
          if (result == nullptr) {
            return false;
          }
          pb::Result pb_result;
          ARROW_RETURN_NOT_OK(internal::ToProto(*result, &pb_result));
          write_buf_ = MakeWriteBuffer(pb_result);
          return true;
        });
  }

 private:
  std::shared_ptr<AsyncResultStream> stream_;
};

/// \brief Serve one ListFlights RPC: the handler's AsyncFlightListing is
/// pulled one FlightInfo per OnWriteDone until it is exhausted.
class ListFlightsReactor final : public StreamingReactor {
 public:
  using StreamingReactor::StreamingReactor;

 protected:
  arrow::Future<> Start() override {
    auto criteria = ParseProtoRequest<pb::Criteria, Criteria>(request_buf_, "Criteria");
    if (!criteria.ok()) {
      return arrow::Future<>::MakeFinished(criteria.status());
    }
    criteria_ = *criteria;
    return base_->ListFlightsAsync(flight_context_, &criteria_)
        .Then(
            [this](const std::shared_ptr<AsyncFlightListing>& listing) -> arrow::Status {
              listing_ = listing;
              return arrow::Status::OK();
            });
  }

  arrow::Future<bool> NextMessage() override {
    // A null listing is "no flights available", as in the sync transport
    // (grpc_server.cc:258-261) -- zero messages, then OK.
    if (listing_ == nullptr) {
      return arrow::Future<bool>::MakeFinished(false);
    }
    return listing_->NextAsync().Then(
        [this](const std::shared_ptr<FlightInfo>& info) -> arrow::Result<bool> {
          // Exhaustion is the listing returning null (grpc_server.cc:186-206).
          if (info == nullptr) {
            return false;
          }
          pb::FlightInfo pb_info;
          ARROW_RETURN_NOT_OK(internal::ToProto(*info, &pb_info));
          write_buf_ = MakeWriteBuffer(pb_info);
          return true;
        });
  }

 private:
  Criteria criteria_;
  std::shared_ptr<AsyncFlightListing> listing_;
};

// ---------------------------------------------------------------------------
// DoExchange: the bidirectional RPC.
//
// Its handler is asynchronous user code: it receives an
// AsyncFlightMessageReader/AsyncFlightMessageWriter pair and returns a future
// the reactor finishes the RPC with.  The reader pulls one client message per
// NextAsync() demand (the reactor arms exactly one read at a time), and the
// writer serializes through the same IPC writer stack the synchronous
// transport uses, the reactor writing one message per StartWrite.  No gRPC
// callback thread is ever held by the handler.

class ExchangeReactor;

/// \brief The AsyncFlightMessageReader of one DoExchange RPC.
///
/// Decoding reuses the push decoder DoPut uses, so the wire shapes behave
/// identically: the schema and dictionary messages are read through (they are
/// not chunks), a metadata-only message is a chunk with a null data member,
/// and the end of the exchange is the chunk whose members are both null.
/// Messages are pulled one demand at a time: NextAsync() arms the next read,
/// and the chunk (or the end, or the failure) resolves the demand.
///
/// The reactor drives the I/O (it owns the read buffer and calls OnMessage /
/// OnReadClosed / OnReadFailed); this class owns the decode state machine.
class ExchangeReader final : public AsyncFlightMessageReader {
 public:
  ExchangeReader()
      : listener_(std::make_shared<ChunkListener>(this)), decoder_(listener_) {}

  const FlightDescriptor& descriptor() const override { return descriptor_; }

  arrow::Future<FlightStreamChunk> NextAsync() override {
    if (pending_.is_valid() && !pending_.is_finished()) {
      return arrow::Future<FlightStreamChunk>::MakeFinished(
          arrow::Status::Invalid("one NextAsync at a time"));
    }
    if (buffered_.has_value()) {
      arrow::Future<FlightStreamChunk> out =
          arrow::Future<FlightStreamChunk>::MakeFinished(std::move(*buffered_));
      buffered_.reset();
      return out;
    }
    if (!read_error_.ok()) {
      return arrow::Future<FlightStreamChunk>::MakeFinished(read_error_);
    }
    if (read_closed_) {
      // The read side is over and nothing is buffered: the end of the
      // exchange is the all-null chunk.
      return arrow::Future<FlightStreamChunk>::MakeFinished(FlightStreamChunk{});
    }
    pending_ = arrow::Future<FlightStreamChunk>::Make();
    if (first_read_.has_value()) {
      // The first message was read before the handler started; consume it now.
      internal::FlightData data = std::move(*first_read_);
      first_read_.reset();
      Consume(std::move(data));
    } else {
      RequestRead();
    }
    return pending_;
  }

  /// \brief The reactor wires itself and the descriptor before the handler
  /// starts, and hands over the already-read first message.
  void SetReactor(ExchangeReactor* reactor) { reactor_ = reactor; }
  void SetDescriptor(const FlightDescriptor& descriptor) { descriptor_ = descriptor; }
  void OfferFirstRead(internal::FlightData data) { first_read_ = std::move(data); }

  /// \brief The reactor's OnReadDone: a message was read.
  void OnMessage(internal::FlightData data) {
    read_in_flight_ = false;
    Consume(std::move(data));
  }

  /// \brief The reactor's OnReadDone with ok == false: the read side is over.
  void OnReadClosed() {
    read_in_flight_ = false;
    read_closed_ = true;
    ResolvePending(FlightStreamChunk{});
  }

  /// \brief The reactor read a message that could not be deserialized.
  void OnReadFailed(Status status) {
    read_in_flight_ = false;
    read_error_ = std::move(status);
    ResolvePending(read_error_);
  }

  /// \brief The RPC was cancelled: fail the demand so the handler unwinds.
  void OnCancelled() {
    read_closed_ = true;
    ResolvePending(arrow::Status::Cancelled("the client cancelled the exchange"));
  }

 private:
  void ResolvePending(FlightStreamChunk chunk) {
    if (pending_.is_valid() && !pending_.is_finished()) {
      pending_.MarkFinished(std::move(chunk));
    }
  }

  void ResolvePending(const Status& status) {
    if (pending_.is_valid() && !pending_.is_finished()) {
      pending_.MarkFinished(status);
    }
  }

  /// \brief Decode one message and resolve the demand it satisfies, or keep
  /// reading when the message was read through (a schema or dictionary
  /// message).
  void Consume(internal::FlightData data) {
    if (read_closed_) {
      return;  // the demand was already resolved
    }
    decoder_.Consume(std::move(data)).AddCallback([this](const arrow::Status& status) {
      if (!status.ok()) {
        // A message that cannot be decoded breaks the exchange.
        read_error_ = status;
        ResolvePending(status);
      } else if (pending_.is_valid() && !pending_.is_finished()) {
        // Read through: the demand is still open, keep reading until a chunk
        // or the end arrives.
        RequestRead();
      }
    });
  }

  /// \brief Arm the next read through the reactor (defined after
  /// ExchangeReactor, whose full definition it needs).
  void RequestRead();

  /// The decoder's listener: chunks are handed to the reader; the descriptor
  /// and the schema are consumed (the reactor sets the descriptor before the
  /// handler starts).
  class ChunkListener final : public AsyncFlightDataListener {
   public:
    explicit ChunkListener(ExchangeReader* reader) : reader_(reader) {}

    arrow::Future<> OnNext(FlightStreamChunk chunk) override {
      if (reader_->pending_.is_valid() && !reader_->pending_.is_finished()) {
        reader_->ResolvePending(std::move(chunk));
      } else {
        // A chunk that arrived with no demand outstanding: buffered (a
        // one-chunk window), returned by the next NextAsync().  In practice
        // a demand is always open when a read is armed.
        reader_->buffered_ = std::move(chunk);
      }
      return arrow::Future<>::MakeFinished();
    }

    arrow::Future<> OnDescriptor(const FlightDescriptor&) override {
      return arrow::Future<>::MakeFinished();
    }

    arrow::Status OnSchemaDecoded(std::shared_ptr<Schema>) override {
      return arrow::Status::OK();
    }

   private:
    ExchangeReader* reader_;
  };

  ExchangeReactor* reactor_ = nullptr;  // owned by gRPC; alive for the RPC
  std::shared_ptr<ChunkListener> listener_;
  AsyncFlightMessageDecoder decoder_;
  FlightDescriptor descriptor_;
  std::optional<internal::FlightData> first_read_;
  std::optional<FlightStreamChunk> buffered_;
  arrow::Future<FlightStreamChunk> pending_;
  Status read_error_;
  bool read_in_flight_ = false;
  bool read_closed_ = false;
};

/// \brief The AsyncFlightMessageWriter of one DoExchange RPC.
///
/// Record batches are serialized through the same IPC writer stack the
/// synchronous transport uses (dictionary messages included) into an
/// IpcPayloadWriter sink that queues FlightPayloads; the reactor then writes
/// them one at a time.  Each method's future resolves when its messages have
/// been handed to the reactor, or with the first failure.
class ExchangeWriter final : public AsyncFlightMessageWriter {
 public:
  explicit ExchangeWriter(ExchangeReactor* reactor) : reactor_(reactor) {}

  arrow::Future<> BeginAsync(std::shared_ptr<Schema> schema) override {
    if (batch_writer_ != nullptr) {
      return arrow::Future<>::MakeFinished(
          arrow::Status::Invalid("This writer has already been started."));
    }
    auto sink = std::make_unique<PayloadSink>(this);
    ARROW_ASSIGN_OR_RAISE(batch_writer_, ipc::internal::OpenRecordBatchWriter(
                                             std::move(sink), schema, options_));
    return Flush();
  }

  arrow::Future<> WriteRecordBatchAsync(const RecordBatch& batch) override {
    pending_app_metadata_ = nullptr;
    return WriteBatch(batch);
  }

  arrow::Future<> WriteWithMetadataAsync(const RecordBatch& batch,
                                         std::shared_ptr<Buffer> app_metadata) override {
    pending_app_metadata_ = std::move(app_metadata);
    return WriteBatch(batch);
  }

  arrow::Future<> WriteMetadataAsync(std::shared_ptr<Buffer> app_metadata) override {
    FlightPayload payload;
    payload.app_metadata = std::move(app_metadata);
    queue_.push_back(std::move(payload));
    return Flush();
  }

  arrow::Future<> CloseAsync() override {
    if (batch_writer_ != nullptr) {
      auto status = batch_writer_->Close();
      batch_writer_ = nullptr;
      if (!status.ok()) {
        return arrow::Future<>::MakeFinished(std::move(status));
      }
    }
    return arrow::Future<>::MakeFinished();
  }

  /// \brief The reactor's OnWriteDone: continue flushing the queue.
  void OnWriteDone(bool ok) {
    write_in_flight_ = false;
    if (!ok) {
      FailFlush(MakeFlightError(FlightStatusCode::Internal, "Failed to write response"));
      return;
    }
    Pump();
  }

  /// \brief The RPC was cancelled: fail the pending write so the handler
  /// unwinds.
  void OnCancelled() {
    if (write_in_flight_) {
      write_in_flight_ = false;
      FailFlush(arrow::Status::Cancelled("the client cancelled the exchange"));
    }
  }

 private:
  /// \brief The IpcPayloadWriter that queues the IPC payloads the writer
  /// stack produces; the schema, dictionary and record-batch messages all
  /// travel this way, as on the synchronous path.
  class PayloadSink final : public ipc::internal::IpcPayloadWriter {
   public:
    explicit PayloadSink(ExchangeWriter* writer) : writer_(writer) {}

    arrow::Status Start() override { return arrow::Status::OK(); }

    arrow::Status WritePayload(const ipc::IpcPayload& ipc_payload) override {
      FlightPayload payload;
      payload.ipc_message = ipc_payload;
      if (ipc_payload.type == ipc::MessageType::RECORD_BATCH &&
          writer_->pending_app_metadata_ != nullptr) {
        payload.app_metadata = std::move(writer_->pending_app_metadata_);
      }
      writer_->queue_.push_back(std::move(payload));
      return arrow::Status::OK();
    }

    arrow::Status Close() override { return arrow::Status::OK(); }

   private:
    ExchangeWriter* writer_;
  };

  arrow::Future<> WriteBatch(const RecordBatch& batch) {
    if (batch_writer_ == nullptr) {
      return arrow::Future<>::MakeFinished(arrow::Status::Invalid(
          "This writer is not started. Call BeginAsync() with a schema"));
    }
    auto status = batch_writer_->WriteRecordBatch(batch);
    if (!status.ok()) {
      return arrow::Future<>::MakeFinished(std::move(status));
    }
    return Flush();
  }

  /// \brief Write the queued payloads one at a time; the returned future
  /// resolves when the queue is empty, or with the first failure.
  arrow::Future<> Flush() {
    if (!flush_.is_valid() || flush_.is_finished()) {
      flush_ = arrow::Future<>::Make();
    }
    Pump();
    return flush_;
  }

  /// \brief Write queued payloads through the reactor (defined after
  /// ExchangeReactor, whose full definition it needs).
  void Pump();

  void FailFlush(Status status) {
    if (flush_.is_valid() && !flush_.is_finished()) {
      flush_.MarkFinished(std::move(status));
    }
  }

  ExchangeReactor* reactor_;
  std::unique_ptr<ipc::RecordBatchWriter> batch_writer_;
  ipc::IpcWriteOptions options_ = ipc::IpcWriteOptions::Defaults();
  std::shared_ptr<Buffer> pending_app_metadata_;
  std::deque<FlightPayload> queue_;
  arrow::Future<> flush_;
  bool write_in_flight_ = false;
};

/// Serve one DoExchange RPC.
///
/// The handler receives an AsyncFlightMessageReader/Writer pair and returns a
/// future; the RPC then runs entirely on continuations.  The reader pulls one
/// message per NextAsync() demand (the reactor arms the read), the writer
/// serializes through the IPC writer stack and the reactor writes one message
/// per StartWrite, and the handler's future completing finishes the RPC with
/// its status.  No callback thread is ever held by the handler.
class ExchangeReactor final : public ::grpc::ServerGenericBidiReactor {
 public:
  ExchangeReactor(AsyncCallContext flight_context, AsyncGenericFlightServerBase* base)
      : flight_context_(std::move(flight_context)),
        base_(base),
        reader_(std::make_shared<ExchangeReader>()),
        writer_(std::make_shared<ExchangeWriter>(this)) {
    reader_->SetReactor(this);
    // The first message carries the request descriptor.
    StartRead(&read_buf_);
  }

  void OnReadDone(bool ok) override {
    if (finished_) {
      return;  // the RPC is over; gRPC may still deliver this
    }
    if (!ok) {
      if (!started_) {
        // The client never sent the descriptor message.
        FinishOnce(MakeFlightError(FlightStatusCode::Internal, "Failed to read request"));
        return;
      }
      // The client half-closed (or the stream broke): the read side is over.
      reader_->OnReadClosed();
      return;
    }
    const ::grpc::Status deserialized = FlightDataDeserialize(&read_buf_, &read_data_);
    if (!deserialized.ok()) {
      if (!started_) {
        FinishOnce(
            MakeFlightError(FlightStatusCode::Internal, deserialized.error_message()));
        return;
      }
      // A message that cannot be deserialized breaks the whole exchange.
      reader_->OnReadFailed(
          MakeFlightError(FlightStatusCode::Internal, deserialized.error_message()));
      return;
    }
    if (!started_) {
      started_ = true;
      if (read_data_.descriptor == nullptr) {
        FinishOnce(MakeFlightError(FlightStatusCode::Internal,
                                   "Descriptor missing on first message"));
        return;
      }
      reader_->SetDescriptor(*read_data_.descriptor);
      // The first message may carry data; the reader consumes it on its first
      // NextAsync, so hand it over before the handler can ask.
      reader_->OfferFirstRead(std::move(read_data_));
      // The handler may complete its future on any thread: hold the reactor
      // across it (gRPC has no server-side holds).
      Hold();
      arrow::Future<> exchange =
          base_->DoExchangeAsync(flight_context_, reader_, writer_);
      exchange.AddCallback([this](const arrow::Status& status) {
        if (finished_) {
          // The RPC is over (cancelled while the handler was running).
          ReleaseHold();
          return;
        }
        FinishOnce(status);
        ReleaseHold();
      });
      return;
    }
    // A message read because the reader asked for one.
    reader_->OnMessage(std::move(read_data_));
  }

  void OnWriteDone(bool ok) override {
    if (finished_) {
      return;  // the RPC is over; gRPC may still deliver this
    }
    writer_->OnWriteDone(ok);
  }

  /// \brief Called when the client cancels the request.
  ///
  /// Finish first (the handler's future may never resolve), then unwind the
  /// reader and writer so a handler parked on their futures wakes up.
  void OnCancel() override {
    FinishOnce(arrow::Status::Cancelled());
    reader_->OnCancelled();
    writer_->OnCancelled();
  }

  void OnDone() override { ReleaseHold(); }

  /// \brief The reader's demand: arm the next read.  The demand guarantees at
  /// most one read is outstanding (ExchangeReader::RequestRead is the only
  /// caller).
  void StartReadNext() {
    if (finished_) {
      return;
    }
    StartRead(&read_buf_);
  }

  /// \brief Serialize one payload into the write buffer and start the write.
  /// Called by ExchangeWriter; OnWriteDone() resumes it.
  arrow::Status WritePayload(FlightPayload payload) {
    if (finished_) {
      return arrow::Status::Cancelled("the exchange is over");
    }
    bool own_buffer = false;
    const ::grpc::Status serialize =
        FlightDataSerialize(payload, &write_buf_, &own_buffer);
    (void)own_buffer;
    if (!serialize.ok()) {
      return MakeFlightError(FlightStatusCode::Internal, serialize.error_message());
    }
    StartWrite(&write_buf_);
    return arrow::Status::OK();
  }

 private:
  /// \brief Ensures that gRPC's Finish is called at most once, and never
  /// after OnDone.  The middleware's CallCompleted runs inside the CAS:
  /// racing finishers (a cancel against a completing continuation) must not
  /// report the call completed twice.
  void FinishOnce(Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true)) {
      Finish(flight_context_.FinishRequest(status));
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

  AsyncCallContext flight_context_;
  AsyncGenericFlightServerBase* base_;
  std::shared_ptr<ExchangeReader> reader_;
  std::shared_ptr<ExchangeWriter> writer_;
  ::grpc::ByteBuffer read_buf_;
  ::grpc::ByteBuffer write_buf_;
  internal::FlightData read_data_;
  bool started_ = false;
  std::atomic<bool> finished_{false};
  std::atomic<int> refs_{1};
};

// ExchangeReader and ExchangeWriter members that call into ExchangeReactor,
// now that its definition above is complete.
void ExchangeReader::RequestRead() {
  if (read_in_flight_ || read_closed_ || reactor_ == nullptr) {
    return;
  }
  read_in_flight_ = true;
  reactor_->StartReadNext();
}

void ExchangeWriter::Pump() {
  if (write_in_flight_) {
    return;  // OnWriteDone resumes it
  }
  if (queue_.empty()) {
    if (flush_.is_valid() && !flush_.is_finished()) {
      flush_.MarkFinished();
    }
    return;
  }
  FlightPayload payload = std::move(queue_.front());
  queue_.pop_front();
  write_in_flight_ = true;
  const auto status = reactor_->WritePayload(std::move(payload));
  if (!status.ok()) {
    write_in_flight_ = false;
    FailFlush(std::move(status));
  }
}

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
    std::shared_ptr<GrpcServerCallContextHelper<::grpc::CallbackServerContext>> helper,
    HandshakeFn handshake_handler)
    : base_(async_base),
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
    return new UnaryReactor<FlightInfo, pb::FlightInfo>(
        std::move(flight_context), [base = base_](const ServerCallContext& context,
                                                  const FlightDescriptor& descriptor) {
          return base->GetFlightInfoAsync(context, descriptor);
        });
  }

  if (method == kGetSchemaMethod) {
    return new UnaryReactor<SchemaResult, pb::SchemaResult>(
        std::move(flight_context), [base = base_](const ServerCallContext& context,
                                                  const FlightDescriptor& descriptor) {
          return base->GetSchemaAsync(context, descriptor);
        });
  }

  if (method == kPollFlightInfoMethod) {
    return new UnaryReactor<PollInfo, pb::PollInfo>(
        std::move(flight_context), [base = base_](const ServerCallContext& context,
                                                  const FlightDescriptor& descriptor) {
          return base->PollFlightInfoAsync(context, descriptor);
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
    return new ExchangeReactor(std::move(flight_context), base_);
  }

  if (method == kDoGetMethod) {
    return new DoGetReactor(std::move(flight_context), base_);
  }

  if (method == kDoPutMethod) {
    // DoPut needs a listener to hand the incoming batches to; a server class
    // that returns none (the default) refuses uploads.
    std::shared_ptr<AsyncFlightDataListener> listener =
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
