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

#include "arrow/flight/transport_server_internal.h"

namespace arrow::flight::internal {

Status AsyncServerTransport::DoGet(const ServerCallContext& context, const Ticket& ticket,
                                   ServerDataStream* stream) {
  ARROW_ASSIGN_OR_RAISE(auto data_stream, base_->DoGet(context, ticket).MoveResult());
  return WriteDataStream(std::move(data_stream), stream);
}

Status AsyncServerTransport::DoPut(const ServerCallContext& context,
                                   ServerDataStream* stream) {
  ARROW_ASSIGN_OR_RAISE(auto reader, MakeMessageReader(stream));
  auto writer = MakeMetadataWriter(stream);
  RETURN_NOT_OK(base_->DoPut(context, std::move(reader), std::move(writer)).status());
  RETURN_NOT_OK(stream->WritesDone());
  return Status::OK();
}

Status AsyncServerTransport::DoExchange(const ServerCallContext& context,
                                        ServerDataStream* stream) {
  ARROW_ASSIGN_OR_RAISE(auto reader, MakeMessageReader(stream));
  auto writer = MakeMessageWriter(stream);
  RETURN_NOT_OK(base_->DoExchange(context, std::move(reader), std::move(writer)).status());
  RETURN_NOT_OK(stream->WritesDone());
  return Status::OK();
}

}  // namespace arrow::flight::internal
