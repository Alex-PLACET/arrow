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

#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace arrow::flight::transport::grpc::async_internal {
namespace {

/// Write-buffer, context, and completion plumbing shared by server-streaming
/// reactors (DoGet and the source-based server streams).
template <typename Proto>
class AsyncWriteReactorBase
    : public ::grpc::ServerWriteReactor<Proto>,
      public SelfOwnedReactor<AsyncWriteReactorBase<Proto>> {
 public:
  using WriteValue =
      std::conditional_t<std::is_same_v<Proto, pb::FlightData>, FlightPayload, Proto>;

  explicit AsyncWriteReactorBase(GrpcServerCallContext flight_context)
      : flight_context_(std::move(flight_context)) {}

  /// Remember gRPC cancellation for background producers.
  void OnCancel() override { this->SetCanceled(); }

  /// Release gRPC's ownership reference.
  void OnDone() override { this->ReleaseHold(); }

  const GrpcServerCallContext& flight_context() const { return flight_context_; }

  /// Install an asynchronous source, `init_fn` configures it, then
  /// OnSourceReady() starts producing (or OnSourceCancelled() if cancelled).
  template <typename T, typename InitFn>
  void StartAfter(Future<T> future, InitFn init_fn) {
    this->Hold();
    future.AddCallback([this, init_fn = std::move(init_fn)](
                           const arrow::Result<T>& result) mutable {
      if (!result.ok()) {
        FinishWithError(result.status());
      } else {
        auto value = std::move(const_cast<arrow::Result<T>&>(result)).MoveValueUnsafe();
        init_fn(std::move(value));
        if (this->cancelled()) {
          OnSourceCancelled();
        } else {
          OnSourceReady();
        }
      }
      this->ReleaseHold();
    });
  }

 protected:
  /// Finish the RPC with an Arrow status.
  void FinishWithError(const Status& status) {
    this->FinishOnce(this->flight_context_.FinishRequest(status));
  }

  /// Called once an asynchronous source is installed and not cancelled.
  virtual void OnSourceReady() {}

  /// Called when the RPC was cancelled before the source could start.
  virtual void OnSourceCancelled() {}

  /// Return the protobuf storage submitted by StartWrite().
  Proto* GrpcWriteBuffer() {
    if constexpr (std::is_same_v<Proto, pb::FlightData>) {
      return reinterpret_cast<Proto*>(&current_write_);
    } else {
      return &current_write_;
    }
  }

  GrpcServerCallContext flight_context_;
  WriteValue current_write_;
  std::mutex mutex_;
};

/// The Next/Close pair produced by MakeAsyncSource.
template <typename T>
struct AsyncSourceFns {
  std::function<Future<std::unique_ptr<T>>()> next;
  std::function<Future<>()> close;
};

/// Own an async source and delegate Next()/Close() to it; `on_null` supplies
/// the terminal result when the source was never provided.
template <typename T, typename Source>
AsyncSourceFns<T> MakeAsyncSource(
    std::unique_ptr<Source> source,
    std::function<Future<std::unique_ptr<T>>()> on_null) {
  auto state = std::make_shared<std::unique_ptr<Source>>(std::move(source));
  AsyncSourceFns<T> fns;
  fns.next = [state, on_null = std::move(on_null)]() -> Future<std::unique_ptr<T>> {
    if (!*state) {
      return on_null();
    }
    return (*state)->Next();
  };
  fns.close = [state]() -> Future<> {
    if (!*state) {
      return Future<>::MakeFinished();
    }
    return (*state)->Close();
  };
  return fns;
}

/// Streams items pulled from an async source to a server-streaming gRPC
/// response, chaining each Next() future on the thread that completes it.
template <typename Proto, typename UserType>
class SourceReactor : public AsyncWriteReactorBase<Proto> {
 public:
  using NextFn = std::function<Future<std::unique_ptr<UserType>>()>;
  using CloseFn = std::function<Future<>()>;
  using ToProtoFn = std::function<Status(const UserType&, Proto*)>;

  SourceReactor(GrpcServerCallContext flight_context, ToProtoFn to_proto)
      : AsyncWriteReactorBase<Proto>(std::move(flight_context)),
        to_proto_(std::move(to_proto)) {}

  /// Install an async source once an asynchronous server hook has completed.
  template <typename T, typename MakeFnsFn>
  void StartAfter(Future<T> future, MakeFnsFn make_fns_fn) {
    AsyncWriteReactorBase<Proto>::StartAfter(
        std::move(future), [this, make_fns_fn = std::move(make_fns_fn)](T value) mutable {
          auto fns = make_fns_fn(std::move(value));
          next_fn_ = std::move(fns.next);
          close_fn_ = std::move(fns.close);
        });
  }

 private:
  /// A failed write finishes cleanly, as the sync server does.
  void OnWriteDone(bool ok) override {
    if (!ok) {
      this->FinishWithError(Status::OK());
      return;
    }
    Advance();
  }

  /// Start the pull loop once the source is installed.
  void OnSourceReady() override { Advance(); }

  /// Signal cancellation to the source so a pending Next() can complete.
  void OnCancel() override {
    AsyncWriteReactorBase<Proto>::OnCancel();
    if (close_fn_) {
      ARROW_UNUSED(close_fn_());
    }
  }

  /// Pull one item, serialize it, and start its write. The Next() future may
  /// complete on any thread; reactor refs keep the reactor alive until then.
  void Advance() {
    if (this->cancelled()) {
      this->FinishWithError(Status::OK());
      return;
    }
    this->Hold();
    next_fn_().AddCallback(
        [this](const arrow::Result<std::unique_ptr<UserType>>& maybe_value) {
          if (!maybe_value.ok()) {
            this->FinishWithError(maybe_value.status());
          } else if (!maybe_value.ValueUnsafe()) {
            // Source exhausted.
            this->FinishWithError(Status::OK());
          } else if (this->cancelled()) {
            this->FinishWithError(Status::OK());
          } else {
            Proto proto;
            auto st = to_proto_(*maybe_value.ValueUnsafe(), &proto);
            if (!st.ok()) {
              this->FinishWithError(st);
            } else {
              {
                std::lock_guard<std::mutex> lock(this->mutex_);
                this->current_write_ = std::move(proto);
              }
              if (this->cancelled()) {
                this->FinishWithError(Status::OK());
              } else {
                this->StartWrite(this->GrpcWriteBuffer());
              }
            }
          }
          this->ReleaseHold();
        });
  }

  NextFn next_fn_;
  CloseFn close_fn_;
  ToProtoFn to_proto_;
};

/// Streams an AsyncFlightDataStream to the DoGet gRPC response.
class DoGetReactor : public AsyncWriteReactorBase<pb::FlightData> {
 public:
  explicit DoGetReactor(GrpcServerCallContext flight_context)
      : AsyncWriteReactorBase<pb::FlightData>(std::move(flight_context)) {}

  /// Install a source returned by AsyncFlightServerBase::DoGet and drive it
  /// through the generic stream driver.
  void StartAfter(Future<std::unique_ptr<AsyncFlightDataStream>> future) {
    AsyncWriteReactorBase<pb::FlightData>::StartAfter(
        std::move(future), [this](std::unique_ptr<AsyncFlightDataStream> stream) {
          // Hold the reactor for the whole drive: the driver's sink and
          // completion callbacks may run on other threads after gRPC events.
          this->Hold();
          driver_ = std::make_shared<internal::AsyncStreamDriver>(
              std::move(stream), [this] { return this->cancelled(); },
              [this](FlightPayload payload) {
                return StartPayloadWrite(std::move(payload));
              });
          driver_->Run().AddCallback(
              [this](const ::arrow::Result<::arrow::internal::Empty>& result) {
                this->FinishWithError(result.status());
                this->ReleaseHold();
              });
        });
  }

  /// Interrupt the stream when gRPC cancels the RPC.
  void OnCancel() override {
    AsyncWriteReactorBase<pb::FlightData>::OnCancel();
    if (driver_) {
      driver_->RequestClose();
    }
  }

 private:
  /// Validate and submit one FlightData payload to gRPC.
  Future<bool> StartPayloadWrite(FlightPayload payload) {
    // Hold for the write lifetime: OnWriteDone may run inline during
    // StartWrite, and gRPC may complete the RPC while a sink call is in
    // flight after cancellation.
    this->Hold();
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (this->cancelled()) {
        this->ReleaseHold();
        return Future<bool>::MakeFinished(false);
      }
      this->current_write_ = std::move(payload);
    }
    pending_write_ = Future<bool>::Make();
    // Copy before StartWrite: an inline OnWriteDone resets the member.
    auto out = pending_write_;
    this->StartWrite(this->GrpcWriteBuffer());
    return out;
  }

  /// Resolve the pending write future; the driver chains from it.
  void OnWriteDone(bool ok) override {
    auto future = std::move(pending_write_);
    pending_write_ = Future<bool>();
    future.MarkFinished(!this->cancelled() && ok);
    this->ReleaseHold();
  }

  /// Generic driver owning the schema/payload/close orchestration.
  std::shared_ptr<internal::AsyncStreamDriver> driver_;
  /// Completes when the single active gRPC write finishes.
  Future<bool> pending_write_;
};

}  // namespace

::grpc::ServerWriteReactor<pb::FlightInfo>* MakeListFlightsReactor(
    GrpcServerCallContext flight_context,
    Future<std::unique_ptr<AsyncFlightListing>> future) {
  auto* reactor = new SourceReactor<pb::FlightInfo, FlightInfo>(
      std::move(flight_context),
      [](const FlightInfo& info, pb::FlightInfo* out) { return internal::ToProto(info, out); });
  reactor->StartAfter(std::move(future), [](std::unique_ptr<AsyncFlightListing> listing) {
    return MakeAsyncSource<FlightInfo>(std::move(listing), [] {
      // A null listing means no flights are available.
      return Future<std::unique_ptr<FlightInfo>>::MakeFinished(
          std::unique_ptr<FlightInfo>{});
    });
  });
  return reactor;
}

::grpc::ServerWriteReactor<pb::ActionType>* MakeListActionsReactor(
    GrpcServerCallContext flight_context, Future<std::vector<ActionType>> future) {
  auto* reactor = new SourceReactor<pb::ActionType, ActionType>(
      std::move(flight_context), [](const ActionType& action, pb::ActionType* out) {
        return internal::ToProto(action, out);
      });
  reactor->StartAfter(std::move(future), [](std::vector<ActionType> actions) {
    AsyncSourceFns<ActionType> fns;
    auto state = std::make_shared<std::vector<ActionType>>(std::move(actions));
    auto index = std::make_shared<size_t>(0);
    fns.next = [state, index]() -> Future<std::unique_ptr<ActionType>> {
      if (*index >= state->size()) {
        return Future<std::unique_ptr<ActionType>>::MakeFinished(
            std::unique_ptr<ActionType>{});
      }
      return Future<std::unique_ptr<ActionType>>::MakeFinished(
          std::make_unique<ActionType>((*state)[(*index)++]));
    };
    fns.close = []() { return Future<>::MakeFinished(); };
    return fns;
  });
  return reactor;
}

::grpc::ServerWriteReactor<pb::Result>* MakeDoActionReactor(
    GrpcServerCallContext flight_context,
    Future<std::unique_ptr<AsyncResultStream>> future) {
  auto* reactor = new SourceReactor<pb::Result, Result>(
      std::move(flight_context),
      [](const Result& result, pb::Result* out) { return internal::ToProto(result, out); });
  reactor->StartAfter(std::move(future), [](std::unique_ptr<AsyncResultStream> results) {
    return MakeAsyncSource<Result>(std::move(results), [] {
      // A null result stream surfaces as cancellation, matching the sync server.
      return Future<std::unique_ptr<Result>>::MakeFinished(Status::Cancelled());
    });
  });
  return reactor;
}

::grpc::ServerWriteReactor<pb::FlightData>* MakeDoGetReactor(
    GrpcServerCallContext flight_context,
    Future<std::unique_ptr<AsyncFlightDataStream>> future) {
  auto* reactor = new DoGetReactor(std::move(flight_context));
  reactor->StartAfter(std::move(future));
  return reactor;
}

}  // namespace arrow::flight::transport::grpc::async_internal