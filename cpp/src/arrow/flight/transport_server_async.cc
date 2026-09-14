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

#include "arrow/flight/transport_server_async.h"

#include <memory>
#include <utility>

namespace arrow::flight::internal {

arrow::Result<std::unique_ptr<AsyncServerTransport>> MakeAsyncServerTransport(
    const std::string& scheme, AsyncFlightServerBase* base,
    std::shared_ptr<MemoryManager> memory_manager) {
  if (scheme == kSchemeGrpc || scheme == kSchemeGrpcTcp || scheme == kSchemeGrpcTls ||
      scheme == kSchemeGrpcUnix) {
    return MakeGrpcCallbackServerTransport(base, std::move(memory_manager));
  }
  return Status::KeyError("No async server transport implementation for ", scheme);
}

AsyncStreamDriver::AsyncStreamDriver(std::unique_ptr<AsyncFlightDataStream> stream,
                                     CancelFn is_cancelled, WriteFn write_fn)
    : stream_(std::move(stream)),
      is_cancelled_(std::move(is_cancelled)),
      write_fn_(std::move(write_fn)) {}

Future<> AsyncStreamDriver::Run() {
  out_ = Future<>::Make();
  if (!stream_) {
    out_.MarkFinished(Status::KeyError("No data in this flight"));
    return out_;
  }
  if (is_cancelled_()) {
    BeginClose();
    return out_;
  }
  PullNext(true);
  return out_;
}

void AsyncStreamDriver::RequestClose(Status failure) { BeginClose(std::move(failure)); }

void AsyncStreamDriver::PullNext(bool first) {
  auto self = shared_from_this();
  auto future = first ? stream_->GetSchemaPayload() : stream_->Next();
  future.AddCallback([self](const ::arrow::Result<FlightPayload>& result) {
    if (!result.ok()) {
      self->BeginClose(result.status());
      return;
    }
    auto payload = result.ValueUnsafe();
    if (payload.ipc_message.metadata == nullptr) {
      // End of stream
      self->BeginClose();
      return;
    }
    const auto status = payload.Validate();
    if (!status.ok()) {
      self->BeginClose(status);
      return;
    }
    self->StartWrite(std::move(payload));
  });
}

void AsyncStreamDriver::StartWrite(FlightPayload payload) {
  if (close_started_.load(std::memory_order_relaxed)) {
    return;
  }
  auto self = shared_from_this();
  write_fn_(std::move(payload))
      .AddCallback([self](const ::arrow::Result<bool>& result) {
        if (self->close_started_.load(std::memory_order_relaxed)) {
          return;
        }
        if (!result.ok() || !*result) {
          self->BeginClose();
          return;
        }
        self->PullNext(/*first=*/false);
      });
}

void AsyncStreamDriver::BeginClose(Status failure) {
  if (close_started_.exchange(true, std::memory_order_relaxed)) {
    return;
  }
  close_failure_ = std::move(failure);
  CloseStream();
}

void AsyncStreamDriver::CloseStream() {
  auto self = shared_from_this();
  stream_->Close().AddCallback(
      [self](const ::arrow::Result<::arrow::internal::Empty>& result) {
        self->out_.MarkFinished(
            self->close_failure_.ok() ? result.status() : self->close_failure_);
      });
}

}  // namespace arrow::flight::internal
