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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/generic/callback_generic_service.h>

#include "arrow/flight/flight_data_decoder.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/server.h"
#include "arrow/flight/transport/grpc/grpc_server_internal.h"
#include "arrow/flight/transport/grpc/serialization_internal.h"

namespace arrow::flight::transport::grpc {

namespace pb = arrow::flight::protocol;

namespace {

// Arrow has no "internal" status code: the transport's own convention is an
// IOError carrying a FlightStatusDetail(Internal), which ToGrpcStatus turns
// into ::grpc::StatusCode::INTERNAL.
arrow::Status Internal(std::string message) {
  return arrow::Status::IOError(std::move(message))
      .WithDetail(std::make_shared<FlightStatusDetail>(FlightStatusCode::Internal));
}

// Every finish, success or error, goes through the transport's own Arrow ->
// gRPC status conversion.
//
// GrpcServerCallContext<...>::FinishRequest(const Status&) is the natural
// route (it also runs the middleware CallCompleted hook), but this branch
// declares ToGrpcStatus(const Status&, ::grpc::CallbackServerContext*)
// without defining it - only the ::grpc::ServerContext* overload exists - so
// instantiating it fails to link. The middleware hook is inert on this path
// anyway: the context is constructed directly instead of through
// GrpcServerCallContextHelper::MakeCallContext, so its middleware list is
// empty. Switch to FinishRequest when middleware/auth are wired up.
::grpc::Status FinishStatus(const arrow::Status& status) { return ToGrpcStatus(status); }

}  // namespace

/// \brief Serve one DoGet RPC over the generic callback API.
///
/// The generic callback API has no server-streaming reactor, so DoGet is
/// served on a bidi reactor used write-only
//  The request is read once, then one payload is written per OnWriteDone turn until the
//  FlightDataStream
/// ends. The payload sequence mirrors the sync transport
/// (ServerTransportBase::WriteDataStream): schema payload first, then Next()
/// until the last payload, which has no metadata.
class DoGetReactor : public ::grpc::ServerGenericBidiReactor {
 public:
  /// `context` is owned by gRPC and outlives the reactor (the reactor is
  /// deleted in OnDone, before the context is destroyed).
  DoGetReactor(::grpc::CallbackServerContext* context, FlightServerBase* base)
      : flight_context_(context), base_(base) {
    StartRead(&request_buf_);
  }

  /// \brief Called when a read operation has completed.
  /// \param[in] ok Whether the read was successful.
  void OnReadDone(bool ok) override {
    // Request has been read.
    if (!ok) {
      Finish(FinishStatus(Internal("Failed to read request")));
      return;
    }

    const auto ticket = ParseTicket();
    if (!ticket.ok()) {
      Finish(FinishStatus(ticket.status()));
      return;
    }

    const auto status = base_->DoGet(flight_context_, *ticket, &data_stream_);
    if (!status.ok()) {
      Finish(FinishStatus(status));
      return;
    }
    if (data_stream_ == nullptr) {
      // Same as the sync transport, ServerTransportBase::WriteDataStream.
      Finish(FinishStatus(arrow::Status::KeyError("No data in this flight")));
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
      Finish(FinishStatus(Internal("Write failed")));
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
        // TODO:Log the error or handle it as needed.
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
    std::vector<::grpc::Slice> slices;
    if (!request_buf_.Dump(&slices).ok()) {
      return Internal("Failed to read request");
    }
    std::string bytes;
    bytes.reserve(request_buf_.Length());
    for (const auto& slice : slices) {
      bytes.append(reinterpret_cast<const char*>(slice.begin()), slice.size());
    }
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
      Finish(FinishStatus(next.status()));
      return;
    }
    wrote_schema_ = true;
    FlightPayload payload = std::move(*next);

    // End of stream: the last payload has no metadata.
    if (payload.ipc_message.metadata == nullptr) {
      Finish(FinishStatus(data_stream_->Close()));
      return;
    }

    const auto buffers = payload.SerializeToBuffers();
    if (!buffers.ok()) {
      Finish(FinishStatus(buffers.status()));
      return;
    }

    std::vector<::grpc::Slice> slices;
    slices.reserve(buffers->size());
    for (const auto& buffer : *buffers) {
      auto slice = SliceFromBuffer(buffer);
      if (!slice.ok()) {
        Finish(FinishStatus(slice.status()));
        return;
      }
      slices.push_back(std::move(*slice));
    }

    // StartWrite requires the buffer to remain valid until OnWriteDone.
    write_buf_ = ::grpc::ByteBuffer(slices.data(), slices.size());
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
  /// \param `context` is owned by gRPC and outlives the reactor (the reactor is
  /// deleted in OnDone, before the context is destroyed).
  /// \param `listener` is the per-RPC listener that will receive the uploaded data.
  DoPutReactor(::grpc::CallbackServerContext* context,
               std::shared_ptr<FlightDataListener> listener)
      : flight_context_(context), decoder_(std::move(listener)) {
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
      Finish(
          FinishStatus(Internal("Failed to wrap gRPC buffer: " + wrap_status.message())));
      return;
    }

    // Feed the Arrow buffer to the FlightMessageDecoder. This will trigger the
    // appropriate callbacks on the listener (OnSchemaDecoded / OnNext).
    const Status decode_status = decoder_.Consume(std::move(arrow_buf));
    if (!decode_status.ok()) {
      // The listener's status is the transport error rejecting the upload.
      Finish(FinishStatus(decode_status));
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
      Finish(FinishStatus(Internal("Write failed")));
      return;
    }
    Finish(FinishStatus(arrow::Status::OK()));
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

// Reject unknown methods.
  class Unimplemented : public ::grpc::ServerGenericBidiReactor {
    public:
    explicit Unimplemented(std::string_view error_message) { Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, error_message)); }
    void OnDone() override { delete this; }
  };


/// \brief Generic callback service dispatching Flight methods to reactors.
///
/// Constructed with the FlightServerBase whose methods are served; the
/// server's auth handler and middleware are NOT run on this path (see the
/// FinishStatus note). DoPut additionally needs a listener factory (one
/// listener per RPC); without it, DoPut is answered UNIMPLEMENTED.
class AsyncGenericFlightService : public ::grpc::CallbackGenericService {
 public:
  /// `listener_factory` creates the FlightDataListener serving one DoPut RPC;
  /// the service consults it per RPC and refuses uploads when it is empty.
  /// \param base is the FlightServerBase whose methods are served.
  /// \param listener_factory is used to create FlightDataListener instances for DoPut RPCs.
  explicit AsyncGenericFlightService(FlightServerBase* base,
                                     FlightDataListenerFactory listener_factory = {})
      : base_(base), listener_factory_(std::move(listener_factory)) {}

  /// \brief Creates a reactor for the incoming RPC.
  /// \param context is the gRPC callback server context for the RPC.
  /// \return a new reactor handling the RPC, or an Unimplemented reactor if the method is unknown.
  ::grpc::ServerGenericBidiReactor* CreateReactor(
      ::grpc::GenericCallbackServerContext* context) override {
    if (context->method() == "/arrow.flight.protocol.FlightService/DoGet") {
      return new DoGetReactor(context, base_);
    }
    // DoPut needs a listener to hand the incoming batches to, so a server
    // built without a factory does not accept uploads.
    if (context->method() == "/arrow.flight.protocol.FlightService/DoPut") {
      if (!listener_factory_) {
        return new Unimplemented("DoPut is not implemented: no listener available");
      }
      auto listener = listener_factory_();
      if (listener) {
        return new DoPutReactor(context, std::move(listener));
      }
    }
    return new Unimplemented("Unknown method");
  }

 private:
  FlightServerBase* base_;
  FlightDataListenerFactory listener_factory_;
};

}  // namespace arrow::flight::transport::grpc
