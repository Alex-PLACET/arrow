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

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "arrow/flight/platform.h"
#include "arrow/flight/server.h"
#include "arrow/flight/visibility.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"
#include "arrow/util/uri.h"

namespace arrow::flight::internal {

class ServerDataStream;

class ARROW_FLIGHT_EXPORT ServerTransportBase {
 public:
  explicit ServerTransportBase(std::shared_ptr<MemoryManager> memory_manager)
      : memory_manager_(std::move(memory_manager)) {}
  virtual ~ServerTransportBase() = default;

 protected:
  arrow::Result<std::unique_ptr<FlightMessageReader>> MakeMessageReader(
      ServerDataStream* stream) const;
  std::unique_ptr<FlightMetadataWriter> MakeMetadataWriter(ServerDataStream* stream) const;
  std::unique_ptr<FlightMessageWriter> MakeMessageWriter(ServerDataStream* stream) const;
  Status WriteDataStream(std::unique_ptr<FlightDataStream> data_stream,
                         ServerDataStream* stream) const;

  std::shared_ptr<MemoryManager> memory_manager_;
};

class ServerSignalHandler {
 public:
  ARROW_DISALLOW_COPY_AND_ASSIGN(ServerSignalHandler);
  ServerSignalHandler() = default;

  template <typename Fn>
  arrow::Result<std::shared_ptr<::arrow::internal::SelfPipe>> Init(Fn handler) {
    ARROW_ASSIGN_OR_RAISE(self_pipe_, ::arrow::internal::SelfPipe::Make(/*signal_safe=*/true));
    handle_signals_ = std::thread(handler, self_pipe_);
    return self_pipe_;
  }

  Status Shutdown() {
    RETURN_NOT_OK(self_pipe_->Shutdown());
    handle_signals_.join();
    return Status::OK();
  }

  ~ServerSignalHandler() { ARROW_CHECK_OK(Shutdown()); }

 private:
  std::shared_ptr<::arrow::internal::SelfPipe> self_pipe_;
  std::thread handle_signals_;
};

class ARROW_FLIGHT_EXPORT ServerSignalState {
 public:
  ServerSignalState() = default;

  Status SetShutdownOnSignals(const std::vector<int>& sigs);
  int GotSignal() const;

  template <typename WaitFn, typename ShutdownFn>
  Status Serve(WaitFn&& wait, ShutdownFn&& shutdown, const char* not_started_message,
               const char* shutdown_warning) {
    got_signal_ = 0;
    old_signal_handlers_.clear();
    running_instance_ = this;

    ServerSignalHandler signal_handler;
    ARROW_ASSIGN_OR_RAISE(self_pipe_, signal_handler.Init(&ServerSignalState::WaitForSignals));
    for (size_t i = 0; i < signals_.size(); ++i) {
      int signum = signals_[i];
      ::arrow::internal::SignalHandler new_handler(&ServerSignalState::HandleSignal),
          old_handler;
      ARROW_ASSIGN_OR_RAISE(
          old_handler, ::arrow::internal::SetSignalHandler(signum, new_handler));
      old_signal_handlers_.push_back(std::move(old_handler));
    }

    shutdown_ = [shutdown = std::forward<ShutdownFn>(shutdown)]() mutable {
      return shutdown(nullptr);
    };
    shutdown_warning_ = shutdown_warning;
    auto status = wait();
    if (!status.ok()) {
      running_instance_ = nullptr;
      shutdown_ = nullptr;
      shutdown_warning_ = nullptr;
      return status;
    }
    running_instance_ = nullptr;
    shutdown_ = nullptr;
    shutdown_warning_ = nullptr;

    for (size_t i = 0; i < signals_.size(); ++i) {
      RETURN_NOT_OK(::arrow::internal::SetSignalHandler(signals_[i], old_signal_handlers_[i])
                        .status());
    }
    return Status::OK();
  }

  template <typename ShutdownFn>
  Status Shutdown(ShutdownFn&& shutdown,
                  const std::chrono::system_clock::time_point* deadline) {
    running_instance_ = nullptr;
    return shutdown(deadline);
  }

  template <typename WaitFn>
  Status Wait(WaitFn&& wait) {
    RETURN_NOT_OK(wait());
    running_instance_ = nullptr;
    return Status::OK();
  }

 private:
  static void HandleSignal(int signum);
  void DoHandleSignal(int signum);
  static void WaitForSignals(std::shared_ptr<::arrow::internal::SelfPipe> self_pipe);

  static std::atomic<ServerSignalState*> running_instance_;

  std::shared_ptr<::arrow::internal::SelfPipe> self_pipe_;
  std::vector<int> signals_;
  std::vector<::arrow::internal::SignalHandler> old_signal_handlers_;
  std::atomic<int> got_signal_{0};
  std::function<Status()> shutdown_;
  const char* shutdown_warning_ = nullptr;
};

ARROW_FLIGHT_EXPORT
arrow::Result<arrow::util::Uri> ParseLocationUri(const Location& location);

ARROW_FLIGHT_EXPORT
int PortFromLocation(const Location& location);

}  // namespace arrow::flight::internal
