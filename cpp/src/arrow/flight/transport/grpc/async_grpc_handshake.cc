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

#include <string>
#include <utility>

namespace arrow::flight::transport::grpc::detail {

namespace {

/// Serve the Handshake RPC over the generic callback API.
class HandshakeReactor : public ::grpc::ServerGenericBidiReactor {
 public:
  HandshakeReactor(AsyncCallContext flight_context, HandshakeFn handshake_handler)
      : flight_context_(std::move(flight_context)),
        handshake_handler_(std::move(handshake_handler)) {
    StartRead(&request_buf_);
  }

  void OnReadDone(bool ok) override {
    if (!ok) {
      Finish(flight_context_.FinishRequest(
          MakeFlightError(FlightStatusCode::Internal, "Failed to read request")));
      return;
    }
    auto request = ParseProtoRequest<pb::HandshakeRequest>(request_buf_, "HandshakeRequest");
    if (!request.ok()) {
      Finish(flight_context_.FinishRequest(std::move(request).status()));
      return;
    }
    std::string response;
    const auto status = handshake_handler_(flight_context_, request->payload(), &response);
    if (!status.ok()) {
      Finish(flight_context_.FinishRequest(status));
      return;
    }
    pb::HandshakeResponse pb_response;
    pb_response.set_payload(std::move(response));
    write_buf_ = MakeWriteBuffer(pb_response);
    StartWrite(&write_buf_);
  }

  void OnWriteDone(bool ok) override {
    Finish(flight_context_.FinishRequest(
        ok ? arrow::Status::OK()
           : MakeFlightError(FlightStatusCode::Internal, "Failed to write response")));
  }

  void OnDone() override { delete this; }

 private:
  AsyncCallContext flight_context_;
  HandshakeFn handshake_handler_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer write_buf_;
};

}  // namespace

::grpc::ServerGenericBidiReactor* MakeHandshakeReactor(
    AsyncCallContext flight_context, HandshakeFn handshake_handler) {
  return new HandshakeReactor(std::move(flight_context), std::move(handshake_handler));
}

}  // namespace arrow::flight::transport::grpc::detail