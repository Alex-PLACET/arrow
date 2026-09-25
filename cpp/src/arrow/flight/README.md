<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Arrow Flight RPC System for C++

## Development notes

The gRPC protobuf plugin requires that libprotoc is in your
`LD_LIBRARY_PATH`. Until we figure out a general solution, you may need to do:

```
export LD_LIBRARY_PATH=$PROTOBUF_HOME/lib:$LD_LIBRARY_PATH
```

Currently, to run the unit tests, the directory of executables must either be
your current working directory or you need to add it to your path, e.g.

```
PATH=debug:$PATH debug/flight-test
```

## Experimental: async server over gRPC's generic callback API

`FlightServerOptions::use_async_grpc` (default off) serves Flight over gRPC's
generic callback API instead of the default synchronous typed service.

- `DoGet` is served through the server's `FlightServerBase::DoGet`: the
  request is parsed as a `Ticket` and the `FlightDataStream` it returns is
  pumped one payload at a time. It runs on a bidi-shaped reactor used
  write-only, because the generic callback API has no server-streaming
  reactor (see apache/arrow#49339 and GH-37937 for the background).
- `DoPut` is served through a per-RPC `FlightDataListener` obtained from
  `FlightServerOptions::listener_factory`, which also hands the upload's
  descriptor to the listener through `FlightDataListener::OnDescriptor`.
  Without a factory, uploads answer `UNIMPLEMENTED`.
- Every other RPC answers `UNIMPLEMENTED`. The flag replaces the synchronous
  typed service rather than adding to it: a method claimed by a registered
  typed service never reaches the generic handler.
- Middleware runs on this path: `ServerMiddlewareFactory::StartCall`, the
  SendingHeaders hook and CallCompleted are all invoked. The blocking
  `ServerAuthHandler` is still refused by `Init()`: to serve the handshake
  and check a per-call token, derive from `AsyncGenericFlightServerBase`
  (`server.h`), which forces this flag and carries the `Handshake` /
  `ValidateToken` virtuals.