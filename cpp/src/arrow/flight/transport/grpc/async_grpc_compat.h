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

// Version gating for the async Flight server, which is built on gRPC's generic
// callback API (grpc::CallbackGenericService and its grpc::ServerGenericBidiReactor).
//
// That API appeared in gRPC 1.65, and 1.66 moved its declarations from
// grpcpp/generic/async_generic_service.h to
// grpcpp/generic/callback_generic_service.h.
// Both details are hidden behind ARROW_FLIGHT_HAS_ASYNC_SERVER so the rest of the transport does not have to
// care: when it is 0 the async server is not compiled and the gRPC transport
// answers async server requests with NotImplemented.

#include <grpcpp/version_info.h>

#define ARROW_FLIGHT_GRPC_AT_LEAST(major, minor) \
  ((GRPC_CPP_VERSION_MAJOR > (major)) ||         \
   ((GRPC_CPP_VERSION_MAJOR) == (major) && (GRPC_CPP_VERSION_MINOR) >= (minor)))

#if ARROW_FLIGHT_GRPC_AT_LEAST(1, 65)
#  define ARROW_FLIGHT_HAS_ASYNC_SERVER 1
#  if ARROW_FLIGHT_GRPC_AT_LEAST(1, 66)
#    include <grpcpp/generic/callback_generic_service.h>
#  else
#    include <grpcpp/generic/async_generic_service.h>
#  endif
#else
#  define ARROW_FLIGHT_HAS_ASYNC_SERVER 0
#endif
