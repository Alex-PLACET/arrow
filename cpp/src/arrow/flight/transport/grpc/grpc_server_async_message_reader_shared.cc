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

#include "arrow/ipc/reader.h"

namespace arrow::flight::transport::grpc::async_internal {
namespace {

/// Owns the shared reader state while exposing the unique_ptr public interface.
class SharedAsyncFlightMessageReader final : public AsyncFlightMessageReader {
 public:
  explicit SharedAsyncFlightMessageReader(std::shared_ptr<AsyncFlightMessageReader> impl)
      : impl_(std::move(impl)) {}

  const FlightDescriptor& descriptor() const override { return impl_->descriptor(); }

  Future<std::shared_ptr<Schema>> GetSchema() override { return impl_->GetSchema(); }

  Future<FlightStreamChunk> Next() override { return impl_->Next(); }

  ipc::ReadStats stats() const override { return impl_->stats(); }

 private:
  /// Shared reader state retained behind the public unique_ptr interface.
  std::shared_ptr<AsyncFlightMessageReader> impl_;
};

}  // namespace

Future<AsyncMessageReader> MakeAsyncMessageReader(
    AsyncReadFn read_fn, std::shared_ptr<MemoryManager> memory_manager) {
  return MakeNativeAsyncMessageReader(std::move(read_fn), std::move(memory_manager))
      .Then([](AsyncMessageReaderParts parts) -> ::arrow::Result<AsyncMessageReader> {
        return AsyncMessageReader{
            .reader =
                std::make_unique<SharedAsyncFlightMessageReader>(std::move(parts.reader)),
            .when_idle = std::move(parts.when_idle)};
      });
}

}  // namespace arrow::flight::transport::grpc::async_internal