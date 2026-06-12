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

#include "arrow/flight/server_async.h"

#include "arrow/flight/transport/grpc/grpc_server.h"
#include "arrow/flight/transport_server_async.h"
#include "arrow/flight/transport_server_internal.h"
#include "arrow/status.h"
#include "arrow/util/thread_pool.h"

#include <memory>
#include <mutex>

namespace arrow::flight {

namespace {

class LegacyAsyncFlightDataStream final : public AsyncFlightDataStream {
 public:
  explicit LegacyAsyncFlightDataStream(std::unique_ptr<FlightDataStream> stream)
      : state_(std::make_shared<State>(std::move(stream))) {}

  Future<FlightPayload> GetSchemaPayload() override {
    return Run<FlightPayload>([](State& state) { return state.stream->GetSchemaPayload(); });
  }
  Future<FlightPayload> Next() override {
    return Run<FlightPayload>([](State& state) { return state.stream->Next(); });
  }
  Future<> Close() override {
    return Run<::arrow::internal::Empty>(
        [](State& state) -> ::arrow::Result<::arrow::internal::Empty> {
          RETURN_NOT_OK(state.stream->Close());
          return ::arrow::internal::Empty{};
        });
  }

 private:
  struct State {
    explicit State(std::unique_ptr<FlightDataStream> stream) : stream(std::move(stream)) {}
    std::unique_ptr<FlightDataStream> stream;
    std::mutex mutex;
  };

  template <typename T, typename Fn>
  Future<T> Run(Fn&& fn) {
    auto out = Future<T>::Make();
    auto state = state_;
    auto submitted = ::arrow::internal::GetCpuThreadPool()->Submit(
        [out, state, fn = std::forward<Fn>(fn)]() mutable {
          std::lock_guard<std::mutex> guard(state->mutex);
          out.MarkFinished(fn(*state));
        });
    if (!submitted.ok()) out.MarkFinished(submitted.status());
    return out;
  }

  std::shared_ptr<State> state_;
};

}  // namespace

std::unique_ptr<AsyncFlightDataStream> MakeAsyncFlightDataStreamFromSync(
    std::unique_ptr<FlightDataStream> stream) {
  if (!stream) return nullptr;
  return std::make_unique<LegacyAsyncFlightDataStream>(std::move(stream));
}

struct AsyncFlightServerBase::Impl {
  std::unique_ptr<internal::AsyncServerTransport> transport_;
  internal::ServerSignalState signal_state_;
};

AsyncFlightServerBase::AsyncFlightServerBase() : impl_(new Impl) {}

AsyncFlightServerBase::~AsyncFlightServerBase() = default;

Status AsyncFlightServerBase::Init(const FlightServerOptions& options) {
  flight::transport::grpc::InitializeFlightGrpcServer();
  const auto scheme = options.location.scheme();
  ARROW_ASSIGN_OR_RAISE(impl_->transport_, internal::MakeAsyncServerTransport(
                                               scheme, this, options.memory_manager));
  ARROW_ASSIGN_OR_RAISE(auto uri, internal::ParseLocationUri(options.location));
  return impl_->transport_->Init(options, uri);
}

int AsyncFlightServerBase::port() const { return internal::PortFromLocation(location()); }

Location AsyncFlightServerBase::location() const {
  if (!impl_->transport_) {
    return Location{};
  }
  return impl_->transport_->location();
}

Status AsyncFlightServerBase::SetShutdownOnSignals(const std::vector<int> sigs) {
  return impl_->signal_state_.SetShutdownOnSignals(sigs);
}

Status AsyncFlightServerBase::Serve() {
  return impl_->signal_state_.Serve(
      [this]() -> Status {
        if (!impl_->transport_) {
          return Status::UnknownError("Async Flight server did not start properly");
        }
        return impl_->transport_->Wait();
      },
      [this](const std::chrono::system_clock::time_point* deadline) -> Status {
        if (!impl_->transport_) {
          return Status::Invalid("Shutdown() on uninitialized AsyncFlightServerBase");
        }
        if (deadline) {
          return impl_->transport_->Shutdown(*deadline);
        }
        return impl_->transport_->Shutdown();
      },
      "Async Flight server did not start properly",
      "Error shutting down async Flight server");
}

int AsyncFlightServerBase::GotSignal() const { return impl_->signal_state_.GotSignal(); }

Status AsyncFlightServerBase::Shutdown(
    const std::chrono::system_clock::time_point* deadline) {
  return impl_->signal_state_.Shutdown(
      [this](const std::chrono::system_clock::time_point* maybe_deadline) -> Status {
        if (!impl_->transport_) {
          return Status::Invalid("Shutdown() on uninitialized AsyncFlightServerBase");
        }
        if (maybe_deadline) {
          return impl_->transport_->Shutdown(*maybe_deadline);
        }
        return impl_->transport_->Shutdown();
      },
      deadline);
}

Status AsyncFlightServerBase::Wait() {
  return impl_->signal_state_.Wait([this] {
    if (!impl_->transport_) {
      return Status::Invalid("Wait() on uninitialized AsyncFlightServerBase");
    }
    return impl_->transport_->Wait();
  });
}

Future<> AsyncFlightServerBase::Handshake(const ServerCallContext&,
                                          std::unique_ptr<ServerAuthSender>,
                                          std::unique_ptr<ServerAuthReader>) {
  return Future<>::MakeFinished(Status::NotImplemented("NYI"));
}

Future<std::unique_ptr<FlightListing>> AsyncFlightServerBase::ListFlights(
    const ServerCallContext&, const Criteria*) {
  return Future<std::unique_ptr<FlightListing>>::MakeFinished(
      Status::NotImplemented("NYI"));
}

Future<std::unique_ptr<FlightInfo>> AsyncFlightServerBase::GetFlightInfo(
    const ServerCallContext&, const FlightDescriptor&) {
  return Future<std::unique_ptr<FlightInfo>>::MakeFinished(Status::NotImplemented("NYI"));
}

Future<std::unique_ptr<PollInfo>> AsyncFlightServerBase::PollFlightInfo(
    const ServerCallContext&, const FlightDescriptor&) {
  return Future<std::unique_ptr<PollInfo>>::MakeFinished(Status::NotImplemented("NYI"));
}

Future<std::unique_ptr<SchemaResult>> AsyncFlightServerBase::GetSchema(
    const ServerCallContext&, const FlightDescriptor&) {
  return Future<std::unique_ptr<SchemaResult>>::MakeFinished(
      Status::NotImplemented("NYI"));
}

Future<std::unique_ptr<AsyncFlightDataStream>> AsyncFlightServerBase::DoGet(
    const ServerCallContext&, const Ticket&) {
  return Future<std::unique_ptr<AsyncFlightDataStream>>::MakeFinished(
      Status::NotImplemented("NYI"));
}

Future<> AsyncFlightServerBase::DoPut(const ServerCallContext&,
                                      std::unique_ptr<AsyncFlightMessageReader>,
                                      std::unique_ptr<AsyncFlightMetadataWriter>) {
  return Future<>::MakeFinished(Status::NotImplemented("DoPut must be overridden"));
}

Future<> AsyncFlightServerBase::DoExchange(const ServerCallContext&,
                                           std::unique_ptr<AsyncFlightMessageReader>,
                                           std::unique_ptr<AsyncFlightMessageWriter>) {
  return Future<>::MakeFinished(Status::NotImplemented("DoExchange must be overridden"));
}

Future<std::unique_ptr<ResultStream>> AsyncFlightServerBase::DoAction(
    const ServerCallContext&, const Action&) {
  return Future<std::unique_ptr<ResultStream>>::MakeFinished(
      Status::NotImplemented("NYI"));
}

Future<std::vector<ActionType>> AsyncFlightServerBase::ListActions(
    const ServerCallContext&) {
  return Future<std::vector<ActionType>>::MakeFinished(Status::NotImplemented("NYI"));
}

}  // namespace arrow::flight
