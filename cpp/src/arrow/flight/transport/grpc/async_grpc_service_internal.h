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

#pragma once

#include <string>
#include <string_view>

#include <grpcpp/generic/callback_generic_service.h>
#include <grpcpp/support/byte_buffer.h>
#include <grpcpp/support/slice.h>

#include "arrow/flight/flight_data_decoder.h"
#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/server_async.h"
#include "arrow/flight/transport/grpc/async_grpc_service.h"
#include "arrow/flight/transport/grpc/serialization_internal.h"

namespace arrow::flight::transport::grpc::detail {

namespace pb = arrow::flight::protocol;

using AsyncCallContext = GrpcServerCallContext<::grpc::CallbackServerContext>;

template <typename ProtoT>
::grpc::ByteBuffer MakeWriteBuffer(const ProtoT& message) {
  const std::string bytes = message.SerializeAsString();
  ::grpc::Slice slice(bytes);
  return {&slice, 1};
}

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

template <typename PbT, typename T>
arrow::Result<T> ParseProtoRequest(const ::grpc::ByteBuffer& buf, std::string_view what) {
  ARROW_ASSIGN_OR_RAISE(auto pb, ParseProtoRequest<PbT>(buf, what));
  T out;
  ARROW_RETURN_NOT_OK(arrow::flight::internal::FromProto(pb, &out));
  return out;
}

template <typename T, typename PbT>
arrow::Status SerializeOrNotFound(const std::shared_ptr<T>& value, PbT* out) {
  if (value == nullptr) {
    return arrow::Status::KeyError("Flight not found");
  }
  return arrow::flight::internal::ToProto(*value, out);
}

::grpc::ServerGenericBidiReactor* MakeHandshakeReactor(AsyncCallContext flight_context,
                                                       HandshakeFn handshake_handler);
::grpc::ServerGenericBidiReactor* MakeGetFlightInfoReactor(AsyncCallContext flight_context,
                                                           AsyncGenericFlightServerBase* base);
::grpc::ServerGenericBidiReactor* MakeGetSchemaReactor(AsyncCallContext flight_context,
                                                       AsyncGenericFlightServerBase* base);
::grpc::ServerGenericBidiReactor* MakePollFlightInfoReactor(AsyncCallContext flight_context,
                                                            AsyncGenericFlightServerBase* base);
::grpc::ServerGenericBidiReactor* MakeDoGetReactor(AsyncCallContext flight_context,
                                                   AsyncGenericFlightServerBase* base);
::grpc::ServerGenericBidiReactor* MakeDoPutReactor(
    AsyncCallContext flight_context, std::shared_ptr<AsyncFlightDataListener> listener);
::grpc::ServerGenericBidiReactor* MakeListActionsReactor(
    AsyncCallContext flight_context, AsyncGenericFlightServerBase* base);
::grpc::ServerGenericBidiReactor* MakeDoActionReactor(AsyncCallContext flight_context,
                                                      AsyncGenericFlightServerBase* base);
::grpc::ServerGenericBidiReactor* MakeListFlightsReactor(
    AsyncCallContext flight_context, AsyncGenericFlightServerBase* base);
::grpc::ServerGenericBidiReactor* MakeExchangeReactor(AsyncCallContext flight_context,
                                                      AsyncGenericFlightServerBase* base);

}  // namespace arrow::flight::transport::grpc::detail