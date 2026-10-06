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

#include "arrow/flight/transport/grpc/async_grpc_service_internal.h"

#include "arrow/flight/transport/grpc/customize_grpc.h"

#include <memory>
#include <utility>

namespace arrow::flight::transport::grpc::detail {

namespace {

/// Serve one DoPut RPC over the generic callback API.
class DoPutReactor : public AsyncReactorBase,
                     public arrow::flight::internal::FlightDataListenerTransport {
 public:
  DoPutReactor(AsyncCallContext flight_context,
               std::shared_ptr<AsyncFlightDataListener> listener)
      : AsyncReactorBase(std::move(flight_context)),
        listener_(std::move(listener)),
        decoder_(listener_) {
    arrow::flight::internal::FlightDataListenerTransport::Install(listener_, this);
    StartRead(&request_buf_);
  }

  void CancelUpload(Status status) override {
    if (status.ok()) {
      status = Status::Invalid("Cancel() needs a non-OK status");
    }
    Hold();
    FinishUpload(std::move(status));
    ReleaseHold();
  }

  Status WriteMetadata(const Buffer& app_metadata) override {
    if (finished()) {
      return Status::Invalid(
          "the upload is over: WriteMetadata() is only usable while the "
          "upload's RPC is running");
    }
    if (!ack_in_flight_) {
      // One message in flight at a time: OnWriteDone serializes the PutResults.
      return Status::Invalid(
          "a PutResult message is already in flight: await it before writing "
          "another metadata message");
    }
    ack_in_flight_ = false;
    pb::PutResult pb_result;
    if (app_metadata.size() > 0) {
      pb_result.set_app_metadata(app_metadata.data(), app_metadata.size());
    }
    write_buf_ = MakeWriteBuffer(pb_result);
    StartWrite(&write_buf_);
    return Status::OK();
  }

  void OnReadDone(bool ok) override {
    if (!ok) {
      if (finished()) {
        return;
      }
      // The client half-closed: this ack ends the upload.
      ending_ = true;
      pb::PutResult pb_result;
      write_buf_ = MakeWriteBuffer(pb_result);
      StartWrite(&write_buf_);
      return;
    }
    std::shared_ptr<arrow::Buffer> arrow_buf;
    const Status wrap_status = WrapGrpcBuffer(&request_buf_, &arrow_buf);
    if (!wrap_status.ok()) {
      FinishUpload(
          MakeFlightError(FlightStatusCode::Internal,
                          "Failed to wrap gRPC buffer: " + wrap_status.message()));
      return;
    }

    if (!read_descriptor_) {
      // The first message must carry the descriptor of the upload, exactly as
      // the sync transport requires ("Descriptor missing on first message").
      // Without this the listener never hears OnDescriptor and the upload is
      // acknowledged as if it were well formed.
      read_descriptor_ = true;
      internal::FlightData data;
      const ::grpc::Status deserialized = FlightDataDeserialize(&request_buf_, &data);
      if (!deserialized.ok()) {
        FinishUpload(
            MakeFlightError(FlightStatusCode::Internal, deserialized.error_message()));
        return;
      }
      if (!data.descriptor) {
        FinishUpload(Status::IOError("Descriptor missing on first message"));
        return;
      }
    }

    Future<> decode_status = decoder_.Consume(std::move(arrow_buf));
    decode_status.AddCallback([this, token = hold()](arrow::Status status) {
      if (finished()) {
        // The RPC is over (cancelled while the listener was working): the
        // message is abandoned.
      } else {
        if (!status.ok()) {
          FinishUpload(std::move(status));
        } else {
          StartRead(&request_buf_);
        }
      }
    });
  }

  void OnWriteDone(bool ok) override {
    if (finished()) {
      return;
    }
    if (!ok) {
      FinishUpload(MakeFlightError(FlightStatusCode::Internal, "Write failed"));
      return;
    }
    if (ending_) {
      FinishUpload(arrow::Status::OK());
      return;
    }
    ack_in_flight_ = true;
  }

  void OnCancel() override {
    Status cancel_status = Status::Cancelled("the client cancelled the upload");
    ARROW_WARN_NOT_OK(arrow::flight::internal::FlightDataListenerTransport::ReportFinish(
                          listener_, cancel_status),
                      "Reporting the cancellation of an upload to the listener failed");
    FinishOnce(std::move(cancel_status));
  }

  void OnDone() override {
    arrow::flight::internal::FlightDataListenerTransport::Clear(listener_);
    ReleaseHold();
  }

 private:
  void FinishUpload(Status status) {
    ARROW_WARN_NOT_OK(arrow::flight::internal::FlightDataListenerTransport::ReportFinish(
                          listener_, status),
                      "Reporting the end of an upload to the listener failed");
    FinishOnce(std::move(status));
  }

  std::shared_ptr<AsyncFlightDataListener> listener_;
  AsyncFlightMessageDecoder decoder_;
  /// Whether the descriptor of the first message has been checked.
  bool read_descriptor_ = false;
  /// Whether a PutResult is on the wire.  At the start it is the empty ack the
  /// transport owes the client; a mid-upload WriteMetadata claims it and
  /// OnWriteDone releases it.
  bool ack_in_flight_ = true;
  /// Whether the write in flight is the ending ack (sent after the client
  /// half-closed) rather than a mid-upload metadata message.
  bool ending_ = false;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
};

}  // namespace

::grpc::ServerGenericBidiReactor* MakeDoPutReactor(
    AsyncCallContext flight_context, std::shared_ptr<AsyncFlightDataListener> listener) {
  return new DoPutReactor(std::move(flight_context), std::move(listener));
}

}  // namespace arrow::flight::transport::grpc::detail