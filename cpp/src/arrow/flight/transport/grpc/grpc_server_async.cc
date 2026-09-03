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

namespace arrow::flight::transport::grpc::async_internal {

arrow::Result<std::shared_ptr<arrow::internal::ThreadPool>> MakeAsyncGrpcExecutor() {
  return arrow::internal::ThreadPool::MakeEternal(
      arrow::internal::ThreadPool::DefaultCapacity());
}

::grpc::Status PrepareAuthenticatedCall(const CallbackServiceHelper& helper,
                                        FlightMethod method,
                                        ::grpc::CallbackServerContext* context,
                                        GrpcServerCallContext* flight_context) {
  const auto status = helper.CheckAuth(method, context, flight_context);
  if (!status.ok()) {
    return status;
  }
  helper.AddMiddlewareHeaders(context, flight_context);
  return ::grpc::Status::OK;
}

AsyncGrpcServerTransport::AsyncGrpcServerTransport(
    AsyncFlightServerBase* base, std::shared_ptr<MemoryManager> memory_manager)
    : arrow::flight::internal::AsyncServerTransport(base, std::move(memory_manager)) {}

AsyncGrpcServerTransport::~AsyncGrpcServerTransport() = default;

Status AsyncGrpcServerTransport::Init(const FlightServerOptions& options,
                                      const arrow::util::Uri& uri) {
  ARROW_ASSIGN_OR_RAISE(executor_pool_, MakeAsyncGrpcExecutor());
  helper_ =
      std::make_unique<CallbackServiceHelper>(options.auth_handler, options.middleware);
  grpc_service_ = std::make_unique<CallbackFlightService>(this, *helper_);

  return transport::grpc::StartFlightGrpcServer(options, uri, grpc_service_.get(),
                                                /*callback_api=*/true, &grpc_server_,
                                                &location_);
}

Status AsyncGrpcServerTransport::Shutdown() {
  grpc_server_->Shutdown();
  return Status::OK();
}

Status AsyncGrpcServerTransport::Shutdown(
    const std::chrono::system_clock::time_point& deadline) {
  grpc_server_->Shutdown(deadline);
  return Status::OK();
}

Status AsyncGrpcServerTransport::Wait() {
  grpc_server_->Wait();
  return Status::OK();
}

Location AsyncGrpcServerTransport::location() const { return location_; }

}  // namespace arrow::flight::transport::grpc::async_internal

namespace arrow::flight::internal {

/// Create the callback-API gRPC transport selected by the async transport factory.
arrow::Result<std::unique_ptr<AsyncServerTransport>> MakeGrpcCallbackServerTransport(
    AsyncFlightServerBase* base, std::shared_ptr<MemoryManager> memory_manager) {
  return std::unique_ptr<AsyncServerTransport>(
      new transport::grpc::async_internal::AsyncGrpcServerTransport(
          base, std::move(memory_manager)));
}

}  // namespace arrow::flight::internal
