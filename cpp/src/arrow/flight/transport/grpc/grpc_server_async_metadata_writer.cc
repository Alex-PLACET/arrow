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

#include <functional>
#include <memory>
#include <utility>

#include "arrow/buffer.h"

namespace arrow::flight::transport::grpc::async_internal {
namespace {

/// Converts metadata buffers into PutResult protobuf writes.
class NativeAsyncFlightMetadataWriter final : public AsyncFlightMetadataWriter {
 public:
  using WriteFn = std::function<Future<bool>(pb::PutResult)>;

  explicit NativeAsyncFlightMetadataWriter(WriteFn write_fn)
      : write_fn_(std::move(write_fn)) {}

  /// Send one metadata frame and translate a failed gRPC write to an Arrow error.
  Future<> WriteMetadata(const Buffer& app_metadata) override {
    pb::PutResult result;
    result.set_app_metadata(app_metadata.data(), app_metadata.size());
    return write_fn_(std::move(result)).Then([](bool ok) -> Status {
      if (!ok) {
        return Status::IOError("Unknown error writing metadata.");
      }
      return Status::OK();
    });
  }

 private:
  /// Callback used to submit metadata protobufs to gRPC.
  WriteFn write_fn_;
};

}  // namespace

std::unique_ptr<AsyncFlightMetadataWriter> MakeAsyncMetadataWriter(
    std::function<Future<bool>(pb::PutResult)> write_fn) {
  return std::make_unique<NativeAsyncFlightMetadataWriter>(std::move(write_fn));
}

}  // namespace arrow::flight::transport::grpc::async_internal