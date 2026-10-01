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

#include <atomic>
#include <memory>
#include <utility>

namespace arrow::flight::transport::grpc::detail {

namespace {

/// Serve one DoPut RPC over the generic callback API.
class DoPutReactor : public ::grpc::ServerGenericBidiReactor,
                     public arrow::flight::internal::FlightDataListenerTransport {
 public:
  DoPutReactor(AsyncCallContext flight_context,
               std::shared_ptr<AsyncFlightDataListener> listener)
      : flight_context_(std::move(flight_context)),
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

  void OnReadDone(bool ok) override {
    if (!ok) {
      if (finished_.load()) {
        return;
      }
      // A failed read means the client half-closed -- or that the RPC is being
      // torn down (the client cancelled, or its deadline expired; gRPC
      // completes a pending read with ok=false either way).  Only the
      // acknowledgement write tells the two apart (it completes ok for a
      // half-close the client accepted, and !ok once the call is dead), so the
      // ending is reported from OnWriteDone, never here: reporting OK here
      // raced OnCancel's report and could call a cancelled upload a clean
      // finish.
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

    Future<> decode_status = decoder_.Consume(std::move(arrow_buf));
    // The listener's future may resolve on any thread, possibly after the RPC
    // is over: hold the reactor across the continuation (gRPC has no
    // server-side holds), and stand down when an ending has already run.
    Hold();
    decode_status.AddCallback([this](arrow::Status status) {
      if (finished_) {
        // The RPC is over (cancelled while the listener was working): the
        // message is abandoned.
        ReleaseHold();
        return;
      }
      if (!status.ok()) {
        FinishUpload(std::move(status));
        ReleaseHold();
        return;
      }
      StartRead(&request_buf_);
      ReleaseHold();
    });
  }

  void OnWriteDone(bool ok) override {
    if (finished_) {
      return;
    }
    if (!ok) {
      FinishUpload(MakeFlightError(FlightStatusCode::Internal, "Write failed"));
      return;
    }
    FinishUpload(arrow::Status::OK());
  }

  void OnCancel() override {
    Status cancel_status = Status::Cancelled("the client cancelled the upload");
    ARROW_WARN_NOT_OK(
      arrow::flight::internal::FlightDataListenerTransport::ReportFinish(listener_,
                                         cancel_status),
                      "Reporting the cancellation of an upload to the listener failed");
    FinishOnce(std::move(cancel_status));
  }

  void OnDone() override {
    arrow::flight::internal::FlightDataListenerTransport::Clear(listener_);
    ReleaseHold();
  }

 private:
  void Hold() { refs_.fetch_add(1, std::memory_order_relaxed); }
  void ReleaseHold() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  void FinishUpload(Status status) {
    ARROW_WARN_NOT_OK(
      arrow::flight::internal::FlightDataListenerTransport::ReportFinish(listener_, status),
                      "Reporting the end of an upload to the listener failed");
    FinishOnce(std::move(status));
  }

  void FinishOnce(Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true)) {
      Finish(flight_context_.FinishRequest(status));
    }
  }

  AsyncCallContext flight_context_;
  std::shared_ptr<AsyncFlightDataListener> listener_;
  AsyncFlightMessageDecoder decoder_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
  std::atomic<bool> finished_{false};
  std::atomic<int> refs_{1};
};

}  // namespace

::grpc::ServerGenericBidiReactor* MakeDoPutReactor(
    AsyncCallContext flight_context, std::shared_ptr<AsyncFlightDataListener> listener) {
  return new DoPutReactor(std::move(flight_context), std::move(listener));
}

}  // namespace arrow::flight::transport::grpc::detail