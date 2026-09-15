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

#include "arrow/flight/transport/grpc/grpc_server_async_message_internal.h"

#include <memory>
#include <utility>

namespace arrow::flight::transport::grpc::async_internal {
namespace {

/// Retains shared writer state behind the unique_ptr public writer interface.
class SharedAsyncFlightMessageWriter final : public AsyncFlightMessageWriter {
 public:
  explicit SharedAsyncFlightMessageWriter(std::shared_ptr<AsyncFlightMessageWriter> impl)
      : impl_(std::move(impl)) {}

  Future<> Begin(const std::shared_ptr<Schema>& schema,
                 const ipc::IpcWriteOptions& options) override {
    return impl_->Begin(schema, options);
  }

  Future<> WriteRecordBatch(const RecordBatch& batch) override {
    return impl_->WriteRecordBatch(batch);
  }

  Future<> WriteMetadata(std::shared_ptr<Buffer> app_metadata) override {
    return impl_->WriteMetadata(std::move(app_metadata));
  }

  Future<> WriteWithMetadata(const RecordBatch& batch,
                             std::shared_ptr<Buffer> app_metadata) override {
    return impl_->WriteWithMetadata(batch, std::move(app_metadata));
  }

  Future<> Close() override { return impl_->Close(); }

  ipc::WriteStats stats() const override { return impl_->stats(); }

 private:
  /// Shared implementation retaining writer state across the public wrapper.
  std::shared_ptr<AsyncFlightMessageWriter> impl_;
};

}  // namespace

std::unique_ptr<AsyncFlightMessageWriter> MakeAsyncMessageWriter(
    std::function<Future<bool>(FlightPayload)> write_fn) {
  auto impl = MakeNativeAsyncMessageWriter(std::move(write_fn));
  return std::make_unique<SharedAsyncFlightMessageWriter>(std::move(impl));
}

}  // namespace arrow::flight::transport::grpc::async_internal