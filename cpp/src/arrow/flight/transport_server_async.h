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

#include <chrono>
#include <memory>

#include "arrow/flight/server_async.h"
#include "arrow/flight/transport_server.h"
#include "arrow/flight/transport_server_internal.h"
#include "arrow/flight/type_fwd.h"
#include "arrow/flight/visibility.h"

namespace arrow::flight {

namespace internal {
class AsyncServerTransport;
}

ARROW_FLIGHT_EXPORT
arrow::Result<std::unique_ptr<internal::AsyncServerTransport>>
MakeGrpcCallbackServerTransport(AsyncFlightServerBase* base,
                                std::shared_ptr<MemoryManager> memory_manager);

}  // namespace arrow::flight

namespace arrow::flight::internal {

class ARROW_FLIGHT_EXPORT AsyncServerTransport : public ServerTransportBase {
 public:
  AsyncServerTransport(AsyncFlightServerBase* base,
                       std::shared_ptr<MemoryManager> memory_manager)
      : ServerTransportBase(std::move(memory_manager)), base_(base) {}
  virtual ~AsyncServerTransport() = default;

  virtual Status Init(const FlightServerOptions& options,
                      const arrow::util::Uri& uri) = 0;
  virtual Status Shutdown() = 0;
  virtual Status Shutdown(const std::chrono::system_clock::time_point& deadline) = 0;
  virtual Status Wait() = 0;
  virtual Location location() const = 0;

  AsyncFlightServerBase* base() const { return base_; }

  Status DoGet(const ServerCallContext& context, const Ticket& request,
               ServerDataStream* stream);
  Status DoPut(const ServerCallContext& context, ServerDataStream* stream);
  Status DoExchange(const ServerCallContext& context, ServerDataStream* stream);

 protected:
  AsyncFlightServerBase* base_;
};

}  // namespace arrow::flight::internal
