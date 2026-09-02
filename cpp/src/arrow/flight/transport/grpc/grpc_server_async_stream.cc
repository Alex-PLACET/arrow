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

/// Common lifetime, cancellation, and output storage for server-streaming RPCs.
template <typename Proto>
class AsyncWriteReactorBase : public ::grpc::ServerWriteReactor<Proto> {
 public:
  using WriteValue =
      std::conditional_t<std::is_same_v<Proto, pb::FlightData>, FlightPayload, Proto>;

  AsyncWriteReactorBase(::grpc::CallbackServerContext* context,
                        std::shared_ptr<arrow::internal::ThreadPool> executor,
                        GrpcServerCallContext flight_context)
      : context_(context),
        executor_(std::move(executor)),
        flight_context_(std::move(flight_context)) {}

  /// Remember gRPC cancellation for background producers.
  void OnCancel() override { cancelled_.store(true, std::memory_order_relaxed); }

  /// Release gRPC's ownership reference.
  void OnDone() override { ReleaseRef(); }

  const GrpcServerCallContext& flight_context() const { return flight_context_; }

  /// Start producing values from the configured source.
  void Start() { AdvanceAndFinish(); }

  /// Continue producing values after an asynchronous source is available.
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
        if (cancelled()) {
          OnSourceCancelled();
        } else {
          Start();
        }
      }
      this->ReleaseHold();
    });
  }

  /// Continue producing values after a successful response write.
  void OnWriteDone(bool ok) override {
    if (!ok) {
      OnWriteFailure();
      return;
    }
    AdvanceAndFinish();
  }

 protected:
  template <typename Fn>
  /// Schedule blocking compatibility work while retaining the reactor.
  Status StartBackgroundWork(Fn&& fn) {
    refs_.fetch_add(1, std::memory_order_relaxed);
    auto maybe_future = executor_->Submit([this, fn = std::forward<Fn>(fn)]() mutable {
      fn();
      ReleaseRef();
    });
    if (!maybe_future.ok()) {
      ReleaseRef();
      return maybe_future.status();
    }
    return Status::OK();
  }

  /// Return whether gRPC has cancelled the RPC.
  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

  /// Advance the source and translate immediate failures to gRPC completion.
  void AdvanceAndFinish() {
    auto status = Advance();
    if (!status.ok()) {
      FinishWithError(status);
    }
  }

  /// Finish the RPC with an Arrow status.
  void FinishWithError(const Status& status) {
    FinishOnce(this->flight_context_.FinishRequest(status));
  }

  virtual void OnSourceCancelled() {}

  virtual void OnWriteFailure() { FinishWithError(Status::OK()); }

  virtual Status Advance() = 0;

  /// Finish at most once, even when multiple async operations fail concurrently.
  void FinishOnce(::grpc::Status status) {
    bool expected = false;
    if (finished_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      this->Finish(std::move(status));
    }
  }

  /// Return the protobuf storage submitted by StartWrite().
  Proto* GrpcWriteBuffer() {
    if constexpr (std::is_same_v<Proto, pb::FlightData>) {
      return reinterpret_cast<Proto*>(&current_write_);
    } else {
      return &current_write_;
    }
  }

  ::grpc::CallbackServerContext* context_;
  std::shared_ptr<arrow::internal::ThreadPool> executor_;
  GrpcServerCallContext flight_context_;
  WriteValue current_write_;
  std::mutex mutex_;

  /// Retain this self-owned reactor across an asynchronous callback.
  void Hold() { refs_.fetch_add(1, std::memory_order_relaxed); }
  /// Release a reference acquired with Hold().
  void ReleaseHold() { ReleaseRef(); }

 private:
  /// Delete this reactor when gRPC and background callbacks have released it.
  void ReleaseRef() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  std::atomic<bool> cancelled_{false};
  std::atomic<bool> finished_{false};
  std::atomic<int> refs_{1};
};

/// Streams a legacy pull iterator to a server-streaming gRPC response.
template <typename Proto, typename UserType>
class IteratorReactor : public AsyncWriteReactorBase<Proto> {
 public:
  using NextFn = std::function<arrow::Result<std::unique_ptr<UserType>>()>;
  using ToProtoFn = std::function<Status(const UserType&, Proto*)>;

  IteratorReactor(::grpc::CallbackServerContext* context,
                  std::shared_ptr<arrow::internal::ThreadPool> executor,
                  GrpcServerCallContext flight_context, ToProtoFn to_proto)
      : AsyncWriteReactorBase<Proto>(context, std::move(executor),
                                     std::move(flight_context)),
        to_proto_(std::move(to_proto)) {}

  /// Install an iterator once an asynchronous server hook has completed.
  template <typename T, typename MakeNextFn>
  void StartAfter(Future<T> future, MakeNextFn make_next_fn) {
    AsyncWriteReactorBase<Proto>::StartAfter(
        std::move(future),
        [this, make_next_fn = std::move(make_next_fn)](T value) mutable {
          next_fn_ = make_next_fn(std::move(value));
        });
  }

 private:
  /// Pull, serialize, and start one response write on a worker thread.
  Status Advance() override {
    return this->StartBackgroundWork([this] {
      if (this->cancelled()) {
        return;
      }

      auto maybe_value = next_fn_();
      if (!maybe_value.ok()) {
        this->FinishWithError(maybe_value.status());
        return;
      }

      auto value = std::move(maybe_value).ValueUnsafe();
      if (!value) {
        this->FinishWithError(Status::OK());
        return;
      }

      Proto proto;
      auto st = to_proto_(*value, &proto);
      if (!st.ok()) {
        this->FinishWithError(st);
        return;
      }

      {
        std::lock_guard<std::mutex> lock(this->mutex_);
        if (this->cancelled()) {
          return;
        }
        this->current_write_ = std::move(proto);
      }
      this->StartWrite(this->GrpcWriteBuffer());
    });
  }

  NextFn next_fn_;
  ToProtoFn to_proto_;
};

/// Streams an AsyncFlightDataStream to the DoGet gRPC response.
class DoGetReactor : public AsyncWriteReactorBase<pb::FlightData> {
 public:
  DoGetReactor(::grpc::CallbackServerContext* context,
               std::shared_ptr<arrow::internal::ThreadPool> executor,
               GrpcServerCallContext flight_context,
               std::unique_ptr<AsyncFlightDataStream> stream)
      : AsyncWriteReactorBase<pb::FlightData>(context, std::move(executor),
                                              std::move(flight_context)),
        stream_(std::move(stream)) {}

  /// Install a source returned by AsyncFlightServerBase::DoGet.
  void StartAfter(Future<std::unique_ptr<AsyncFlightDataStream>> future) {
    AsyncWriteReactorBase<pb::FlightData>::StartAfter(
        std::move(future), [this](std::unique_ptr<AsyncFlightDataStream> stream) {
          std::lock_guard<std::mutex> lock(this->mutex_);
          stream_ = std::move(stream);
        });
  }

  /// Release the application stream when gRPC cancels the RPC.
  void OnCancel() override {
    AsyncWriteReactorBase<pb::FlightData>::OnCancel();
    ARROW_UNUSED(CloseStream());
  }

 private:
  enum class Stage { kSchema, kPayloads, kFinish };

  /// Select the next schema, payload, or close operation from the source.
  Status Advance() override {
    if (this->cancelled())
    {
        return Status::OK();
    }
    if (!stream_) {
      this->FinishOnce(this->flight_context_.FinishRequest(
          Status::KeyError("No data in this flight")));
      return Status::OK();
    }
    switch (stage_) {
      case Stage::kSchema:
        stage_ = Stage::kPayloads;
        return ReadPayload(stream_->GetSchemaPayload());
      case Stage::kPayloads:
        return ReadPayload(stream_->Next());
      case Stage::kFinish:
        return CloseStream();
    }
    return CloseStream(Status::Invalid("Invalid stage"));
  }

  /// Close the source and finish the RPC with its close status.
  Status CloseStream(Status failure = Status::OK()) {
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (close_started_) {
        return Status::OK();
      }
      if (!stream_) {
        return Status::OK();
      }
      close_started_ = true;
    }
    this->Hold();
    stream_->Close().AddCallback(
        [this, failure = std::move(failure)](
            const ::arrow::Result<::arrow::internal::Empty>& result) mutable {
          this->FinishWithError(failure.ok() ? result.status() : failure);
          this->ReleaseHold();
        });
    return Status::OK();
  }

  /// Process a future payload, an error, or the end marker.
  Status ReadPayload(Future<FlightPayload> future) {
    this->Hold();
    future.AddCallback([this](const ::arrow::Result<FlightPayload>& result) {
      if (!result.ok()) {
        ARROW_UNUSED(CloseStream(result.status()));
      } else {
        auto payload = result.ValueUnsafe();
        if (payload.ipc_message.metadata == nullptr) {
          stage_ = Stage::kFinish;
          const auto status = Advance();
          if (!status.ok()) {
            ARROW_UNUSED(CloseStream(status));
          }
        } else {
          const auto status = StartPayloadWrite(std::move(payload));
          if (!status.ok()) {
            ARROW_UNUSED(CloseStream(status));
          }
        }
      }
      this->ReleaseHold();
    });
    return Status::OK();
  }

  /// Validate and submit one FlightData payload to gRPC.
  Status StartPayloadWrite(FlightPayload payload) {
    RETURN_NOT_OK(payload.Validate());
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (this->cancelled()) {
        return Status::OK();
      }
      this->current_write_ = std::move(payload);
    }
    this->StartWrite(this->GrpcWriteBuffer());
    return Status::OK();
  }

  /// Close the source when a response write fails.
  void OnWriteFailure() override { ARROW_UNUSED(CloseStream()); }

  /// Close a source that became available after cancellation.
  void OnSourceCancelled() override { ARROW_UNUSED(CloseStream()); }

  std::unique_ptr<AsyncFlightDataStream> stream_;
  Stage stage_ = Stage::kSchema;
  bool close_started_ = false;
};

}  // namespace

::grpc::ServerWriteReactor<pb::FlightInfo>* MakeListFlightsReactor(
    ::grpc::CallbackServerContext* context,
    std::shared_ptr<arrow::internal::ThreadPool> executor,
    GrpcServerCallContext flight_context, Future<std::unique_ptr<FlightListing>> future) {
  auto* reactor = new IteratorReactor<pb::FlightInfo, FlightInfo>(
      context, std::move(executor), std::move(flight_context),
      [](const FlightInfo& info, pb::FlightInfo* out) {
        return internal::ToProto(info, out);
      });
  reactor->StartAfter(std::move(future), [](std::unique_ptr<FlightListing> listing) {
    auto state = std::make_shared<std::unique_ptr<FlightListing>>(std::move(listing));
    return [state]() mutable {
      return *state ? (*state)->Next()
                    : arrow::Result<std::unique_ptr<FlightInfo>>(
                          std::unique_ptr<FlightInfo>{});
    };
  });
  return reactor;
}

::grpc::ServerWriteReactor<pb::ActionType>* MakeListActionsReactor(
    ::grpc::CallbackServerContext* context,
    std::shared_ptr<arrow::internal::ThreadPool> executor,
    GrpcServerCallContext flight_context, Future<std::vector<ActionType>> future) {
  auto* reactor = new IteratorReactor<pb::ActionType, ActionType>(
      context, std::move(executor), std::move(flight_context),
      [](const ActionType& action, pb::ActionType* out) {
        return internal::ToProto(action, out);
      });
  reactor->StartAfter(std::move(future), [](std::vector<ActionType> actions) {
    return [actions = std::move(actions),
            index = size_t{0}]() mutable -> arrow::Result<std::unique_ptr<ActionType>> {
      if (index >= actions.size()) {
        return nullptr;
      }
      return std::make_unique<ActionType>(actions[index++]);
    };
  });
  return reactor;
}

::grpc::ServerWriteReactor<pb::Result>* MakeDoActionReactor(
    ::grpc::CallbackServerContext* context,
    std::shared_ptr<arrow::internal::ThreadPool> executor,
    GrpcServerCallContext flight_context, Future<std::unique_ptr<ResultStream>> future) {
  auto* reactor = new IteratorReactor<pb::Result, Result>(
      context, std::move(executor), std::move(flight_context),
      [](const Result& result, pb::Result* out) {
        return internal::ToProto(result, out);
      });
  reactor->StartAfter(std::move(future), [](std::unique_ptr<ResultStream> results) {
    auto state = std::make_shared<std::unique_ptr<ResultStream>>(std::move(results));
    return [state]() mutable -> arrow::Result<std::unique_ptr<arrow::flight::Result>> {
      if (!*state) {
        // Mirror the sync server: a null result stream is cancelled.
        return Status::Cancelled();
      }
      return (*state)->Next();
    };
  });
  return reactor;
}

::grpc::ServerWriteReactor<pb::FlightData>* MakeDoGetReactor(
    ::grpc::CallbackServerContext* context,
    std::shared_ptr<arrow::internal::ThreadPool> executor,
    GrpcServerCallContext flight_context,
    Future<std::unique_ptr<AsyncFlightDataStream>> future) {
  auto* reactor =
      new DoGetReactor(context, std::move(executor), std::move(flight_context), nullptr);
  reactor->StartAfter(std::move(future));
  return reactor;
}

}  // namespace arrow::flight::transport::grpc::async_internal