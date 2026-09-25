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

// Async Flight service implemented on top of gRPC's generic callback API
// (apache/arrow#49339). Experimental: DoGet and DoPut for now, every other
// method is answered UNIMPLEMENTED.
//
// This is an internal header (no ARROW_FLIGHT_EXPORT): the reactors and the
// service are implementation details of the gRPC transport.

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <grpcpp/generic/callback_generic_service.h>

#include "arrow/flight/flight_data_decoder.h"
#include "arrow/flight/middleware.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/server.h"
#include "arrow/flight/server_auth.h"
#include "arrow/flight/server_middleware.h"
#include "arrow/flight/transport/grpc/grpc_server_internal.h"
#include "arrow/flight/transport/grpc/serialization_internal.h"
#include "arrow/util/logging.h"

namespace arrow::flight::transport::grpc {

namespace pb = arrow::flight::protocol;

// The shared helper's async handshake hook: one Handshake RPC adapted to the
// server class's Handshake virtual.
using HandshakeFn =
    GrpcServerCallContextHelper<::grpc::CallbackServerContext>::HandshakeFn;

namespace {

// Arrow has no "internal" status code: the transport's own convention is an
// IOError carrying a FlightStatusDetail(Internal), which ToGrpcStatus turns
// into ::grpc::StatusCode::INTERNAL.
arrow::Status Internal(std::string message) {
  return arrow::Status::IOError(std::move(message))
      .WithDetail(std::make_shared<FlightStatusDetail>(FlightStatusCode::Internal));
}

// Copy a gRPC ByteBuffer's slices into one contiguous string. 
// Dump() copies the bytes out, so the result does not alias the request buffer.
// \param[in] buffer The gRPC ByteBuffer to copy from.
// \param[in] error_message The error message to use if the buffer cannot be dumped.
// \return A contiguous string containing the buffer's bytes, or an Internal status on failure.
arrow::Result<std::string> BytesFromBuffer(const ::grpc::ByteBuffer& buffer,
                                           std::string_view error_message) {
  std::vector<::grpc::Slice> slices;
  if (!buffer.Dump(&slices).ok()) {
    return Internal(std::string(error_message));
  }
  std::string bytes;
  bytes.reserve(buffer.Length());
  for (const auto& slice : slices) {
    bytes.append(reinterpret_cast<const char*>(slice.begin()), slice.size());
  }
  return bytes;
}

// The Arrow-side call context every reactor on this path carries; it is built
// by the shared helper (which runs middleware and auth) from the raw gRPC
// context, and every finish goes through its FinishRequest.
using AsyncCallContext = GrpcServerCallContext<::grpc::CallbackServerContext>;

constexpr std::string_view kPrefix = "/arrow.flight.protocol.FlightService/";
constexpr std::string_view kHandshakeMethod = "Handshake";
constexpr std::string_view kDoGetMethod = "DoGet";
constexpr std::string_view kDoPutMethod = "DoPut";

// Map a gRPC method name to the middleware-visible Flight method; unknown
// names become FlightMethod::Invalid.
FlightMethod MethodFromName(std::string_view method) {
  if (!method.starts_with(kPrefix)) {
    return FlightMethod::Invalid;
  }
  method.remove_prefix(kPrefix.size());
  constexpr std::pair<std::string_view, FlightMethod> kMethods[] = {
      {"Handshake", FlightMethod::Handshake},
      {"ListFlights", FlightMethod::ListFlights},
      {"GetFlightInfo", FlightMethod::GetFlightInfo},
      {"GetSchema", FlightMethod::GetSchema},
      {"DoGet", FlightMethod::DoGet},
      {"DoPut", FlightMethod::DoPut},
      {"DoAction", FlightMethod::DoAction},
      {"ListActions", FlightMethod::ListActions},
      {"DoExchange", FlightMethod::DoExchange},
      {"PollFlightInfo", FlightMethod::PollFlightInfo},
  };

  const auto it = std::ranges::find_if(
      kMethods, [method](const auto& pair) { return pair.first == method; });
  if (it != std::end(kMethods)) {
    return it->second;
  }
  return FlightMethod::Invalid;
}

}  // namespace

/// \brief Serve one DoGet RPC over the generic callback API.
///
/// The generic callback API has no server-streaming reactor, so DoGet is
/// served on a bidi reactor used write-only: the request is read once, then
/// one payload is written per OnWriteDone turn until the FlightDataStream
/// ends. The payload sequence mirrors the sync transport
/// (ServerTransportBase::WriteDataStream): schema payload first, then Next()
/// until the last payload, which has no metadata.
class DoGetReactor : public ::grpc::ServerGenericBidiReactor {
 public:
  /// `flight_context` is prepared by the service (middleware/auth already
  /// ran), the underlying gRPC context is owned by gRPC and outlives the
  /// reactor (the reactor is deleted in OnDone, before the context is
  /// destroyed).
  DoGetReactor(AsyncCallContext flight_context, FlightServerBase* base)
      : flight_context_(std::move(flight_context)), base_(base) {
    StartRead(&request_buf_);
  }

  /// \brief Called when a read operation has completed.
  /// \param[in] ok Whether the read was successful.
  void OnReadDone(bool ok) override {
    // Request has been read.
    if (!ok) {
      Finish(flight_context_.FinishRequest(Internal("Failed to read request")));
      return;
    }

    const auto ticket = ParseTicket();
    if (!ticket.ok()) {
      Finish(flight_context_.FinishRequest(ticket.status()));
      return;
    }

    const auto status = base_->DoGet(flight_context_, *ticket, &data_stream_);
    if (!status.ok()) {
      Finish(flight_context_.FinishRequest(status));
      return;
    }
    if (data_stream_ == nullptr) {
      Finish(flight_context_.FinishRequest(
          arrow::Status::KeyError("No data in this flight")));
      return;
    }

    // Start writing the first payload.
    WriteNextPayload();
  }

  /// \brief Called when a write operation has completed.
  /// \param[in] ok Whether the write was successful.
  void OnWriteDone(bool ok) override {
    // We have finished writing. We can write the next payload or finish the
    // stream.
    if (!ok) {
      Finish(flight_context_.FinishRequest(Internal("Write failed")));
      return;
    }
    // Continue writing the next payload.
    WriteNextPayload();
  }

  /// \brief Called when the client cancels the request.
  /// This allows the server to stop producing data for a request that will no longer be
  /// consumed.
  void OnCancel() override {
    // The client went away: let the producer stop. The call is already being
    // torn down, so there is no status to report.
    if (data_stream_ != nullptr) {
      const auto status = data_stream_->Close();
      if (!status.ok()) {
        ARROW_LOG(WARNING) << "DoGet: closing the data stream after a client "
                              "cancel failed: "
                           << status;
      }
    }
  }

  /// \brief Called when the RPC is fully done, regardless of success or cancellation.
  /// This is the last callback that will be invoked for this RPC.
  void OnDone() override { delete this; }

 private:
  /// Parse the request ByteBuffer as the pb::Ticket of the DoGet request.
  /// \return A Result containing the parsed Ticket, or an error if parsing failed.
  arrow::Result<Ticket> ParseTicket() {
    ARROW_ASSIGN_OR_RAISE(std::string bytes,
                          BytesFromBuffer(request_buf_, "Failed to read request"));
    pb::Ticket pb_ticket;
    if (!pb_ticket.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
      return arrow::Status::Invalid("Failed to parse Ticket");
    }
    Ticket ticket;
    const auto status = internal::FromProto(pb_ticket, &ticket);
    if (!status.ok()) {
      return status;
    }
    return ticket;
  }

  // Serialize the next payload of the data stream and start writing it. Called
  // once from OnReadDone (schema payload) and then once per OnWriteDone.
  void WriteNextPayload() {
    arrow::Result<FlightPayload> next =
        wrote_schema_ ? data_stream_->Next() : data_stream_->GetSchemaPayload();
    if (!next.ok()) {
      Finish(flight_context_.FinishRequest(next.status()));
      return;
    }
    wrote_schema_ = true;
    FlightPayload payload = std::move(*next);

    // End of stream: the last payload has no metadata.
    if (payload.ipc_message.metadata == nullptr) {
      Finish(flight_context_.FinishRequest(data_stream_->Close()));
      return;
    }

    // Serialize with the shared zero-copy helper, the same one the sync
    // transport's typed path uses.
    bool own_buffer = false;
    const ::grpc::Status grpc_status =
        FlightDataSerialize(payload, &write_buf_, &own_buffer);
    if (!grpc_status.ok()) {
      Finish(flight_context_.FinishRequest(Internal(grpc_status.error_message())));
      return;
    }

    // StartWrite requires the buffer to remain valid until OnWriteDone.
    StartWrite(&write_buf_);
  }

  GrpcServerCallContext<::grpc::CallbackServerContext> flight_context_;
  FlightServerBase* base_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
  std::unique_ptr<FlightDataStream> data_stream_;
  bool wrote_schema_ = false;
};

/// \brief Serve one DoPut RPC over the generic callback API.
///
/// Reads one message per OnReadDone turn and pushes it into the per-RPC
/// FlightMessageDecoder, which fires the listener's callbacks. The listener's
/// status is the upload's status: a non-OK one rejects the upload (the
/// acknowledgement is not written, so the client's DoPut fails with it). On
/// end of stream the acknowledgement is written back, that is what makes the
/// client's DoPut return.
class DoPutReactor : public ::grpc::ServerGenericBidiReactor {
 public:
  /// \param `flight_context` is prepared by the service (middleware/auth ran); the raw
  /// context is owned by gRPC and outlives the reactor (the reactor is
  /// deleted in OnDone, before the context is destroyed).
  /// \param `listener` is the per-RPC listener that will receive the uploaded data.
  DoPutReactor(AsyncCallContext flight_context,
               std::shared_ptr<FlightDataListener> listener)
      : flight_context_(std::move(flight_context)), decoder_(std::move(listener)) {
    StartRead(&request_buf_);
  }

  /// \brief Called when a new message is available from the client.
  /// \param `ok` indicates whether the read was successful. If false, the client has
  /// closed its half of the stream.
  void OnReadDone(bool ok) override {
    if (!ok) {
      // The client closed its half of the stream: acknowledge the upload.
      // The client's DoPut does not return until it has read this message.
      pb::PutResult pb_result;
      ::grpc::Slice slice(pb_result.SerializeAsString());
      // Not a local variable: StartWrite requires the ByteBuffer to remain
      // valid until OnWriteDone.
      write_buf_ = ::grpc::ByteBuffer(&slice, 1);
      StartWrite(&write_buf_);
      return;
    }

    // Extract an Arrow buffer from the gRPC ByteBuffer, then feed it to the
    // FlightMessageDecoder which fires the listener callbacks
    // (OnSchemaDecoded / OnNext).
    std::shared_ptr<arrow::Buffer> arrow_buf;
    const Status wrap_status = WrapGrpcBuffer(&request_buf_, &arrow_buf);
    if (!wrap_status.ok()) {
      Finish(flight_context_.FinishRequest(
          Internal("Failed to wrap gRPC buffer: " + wrap_status.message())));
      return;
    }

    // Feed the Arrow buffer to the FlightMessageDecoder. This will trigger the
    // appropriate callbacks on the listener (OnSchemaDecoded / OnNext).
    const Status decode_status = decoder_.Consume(std::move(arrow_buf));
    if (!decode_status.ok()) {
      // The listener's status is the transport error rejecting the upload.
      Finish(flight_context_.FinishRequest(decode_status));
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
      Finish(flight_context_.FinishRequest(Internal("Write failed")));
      return;
    }
    Finish(flight_context_.FinishRequest(arrow::Status::OK()));
  }

  /// \brief Called when the client cancels the RPC.
  void OnCancel() override {
    // The client went away: there is no data stream to close here (the
    // listener is not owned by the call), and the call is already being torn
    // down, so there is no status to report.
  }

  /// \brief Called when the RPC is done, regardless of success, failure, or cancellation.
  void OnDone() override { delete this; }

 private:
  GrpcServerCallContext<::grpc::CallbackServerContext> flight_context_;
  FlightMessageDecoder decoder_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
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
      Finish(flight_context_.FinishRequest(Internal("Failed to read request")));
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
    const std::string bytes = pb_response.SerializeAsString();
    ::grpc::Slice slice(bytes);
    // Not a local: StartWrite requires the buffer to remain valid until OnWriteDone.
    write_buf_ = ::grpc::ByteBuffer(&slice, 1);
    StartWrite(&write_buf_);
  }

  /// \brief Called when the write of the response has completed.
  /// \param[in] ok Whether the write was successful.
  void OnWriteDone(bool ok) override {
    // A failed read is the only other terminal path, and it never reaches the
    // write, so exactly one of the two calls Finish().
    Finish(flight_context_.FinishRequest(ok ? arrow::Status::OK()
                                            : Internal("Failed to write response")));
  }

  /// \brief Called when the RPC is fully done, regardless of success or
  /// cancellation.
  void OnDone() override { delete this; }

 private:
  /// Parse the request ByteBuffer as the pb::HandshakeRequest of the handshake
  /// and return its payload (the client's password).
  arrow::Result<std::string> ParseRequest() {
    ARROW_ASSIGN_OR_RAISE(
        std::string bytes,
        BytesFromBuffer(request_buf_, "Failed to read HandshakeRequest"));
    pb::HandshakeRequest request;
    if (!request.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
      return arrow::Status::Invalid("Failed to parse HandshakeRequest");
    }
    return request.payload();
  }

  AsyncCallContext flight_context_;
  HandshakeFn handshake_handler_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
};

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

/// \brief Generic callback service dispatching Flight methods to reactors.
///
/// Constructed with the FlightServerBase whose methods are served.  The
/// server's middleware runs for every call, through the shared context
/// ⟪HERMES-CONTEXT-COMPRESSION: 370 of 570 chars omitted here by Hermes's context
/// compressor. This is NOT part of the original tool call and must never be reproduced in
/// new output — always write full, untruncated content.⟫
class AsyncGenericFlightService : public ::grpc::CallbackGenericService {
 public:
  /// `listener_factory` creates the FlightDataListener serving one DoPut RPC;
  /// the service consults it per RPC and refuses uploads when it is empty.
  /// \param base is the FlightServerBase whose methods are served.
  /// \param listener_factory is used to create FlightDataListener instances for DoPut
  /// RPCs. \param handshake_handler is the server's Handshake hook; an empty hook means
  /// the server has no authentication mechanism.
  AsyncGenericFlightService(
      FlightServerBase* base, FlightDataListenerFactory listener_factory,
      std::shared_ptr<GrpcServerCallContextHelper<::grpc::CallbackServerContext>> helper,
      HandshakeFn handshake_handler = {})
      : base_(base),
        listener_factory_(std::move(listener_factory)),
        helper_(std::move(helper)),
        handshake_handler_(std::move(handshake_handler)) {}

  /// \brief Creates a reactor for the incoming RPC.
  /// \param context is the gRPC callback server context for the RPC.
  /// \return a new reactor handling the RPC, or an Unimplemented reactor if the method is
  /// unknown.
  ::grpc::ServerGenericBidiReactor* CreateReactor(
      ::grpc::GenericCallbackServerContext* context) override {
    // gRPC hands the reactor the full path
    // ("/arrow.flight.protocol.FlightService/DoGet"), while the dispatch below
    // compares bare method names.
    const std::string_view full_method = context->method();
    const std::string_view method = full_method.starts_with(kPrefix)
                                        ? full_method.substr(kPrefix.size())
                                        : full_method;
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

    if (method == kDoGetMethod) {
      return new DoGetReactor(std::move(flight_context), base_);
    }
    // DoPut needs a listener to hand the incoming batches to, so a server
    // built without a factory does not accept uploads.
    if (method == kDoPutMethod) {
      // A server built without a factory, or whose factory declined this RPC,
      // does not accept uploads.
      if (auto listener = listener_factory_ ? listener_factory_() : nullptr) {
        return new DoPutReactor(std::move(flight_context), std::move(listener));
      }
      return new Unimplemented(
          ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED,
                         "DoPut is not implemented: no listener available"));
    }
    return new Unimplemented(
        ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Unknown method"));
  }

 private:
  FlightServerBase* base_;
  FlightDataListenerFactory listener_factory_;
  std::shared_ptr<GrpcServerCallContextHelper<::grpc::CallbackServerContext>> helper_;
  /// The server class's Handshake hook; empty when the server has none.
  HandshakeFn handshake_handler_;
};

}  // namespace arrow::flight::transport::grpc
