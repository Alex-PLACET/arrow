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

#include "arrow/flight/transport/grpc/async_grpc_service_internal.h"

#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <utility>

#include "arrow/ipc/writer.h"

namespace arrow::flight::transport::grpc::detail {

namespace {

class ExchangeReactor;

class ExchangeReader final : public AsyncFlightMessageReader {
 public:
  ExchangeReader()
      : listener_(std::make_shared<ChunkListener>(this)), decoder_(listener_) {}

  const FlightDescriptor& descriptor() const override { return descriptor_; }

  /// Asynchronously retrieves the next chunk from the exchange. Returns a future that
  /// will be completed when the next chunk is available, or with an error if the read
  /// fails.
  /// \returns A future that will be completed with the next chunk or an error.
  arrow::Future<FlightStreamChunk> NextAsync() override {
    using FSCFuture = arrow::Future<FlightStreamChunk>;

    if (pending_.is_valid() && !pending_.is_finished()) {
      return FSCFuture::MakeFinished(arrow::Status::Invalid("one NextAsync at a time"));
    }
    if (buffered_.has_value()) {
      auto out = FSCFuture::MakeFinished(std::move(*buffered_));
      buffered_.reset();
      return out;
    }

    if (!read_error_.ok()) {
      return FSCFuture::MakeFinished(read_error_);
    }

    if (read_closed_ || reactor_.load() == nullptr) {
      // End of exchange: the reactor either saw the read side close or the RPC
      // is over (and the reactor is gone), so no further read can be started.
      return FSCFuture::MakeFinished(FlightStreamChunk{});
    }

    pending_ = FSCFuture::Make();
    if (first_read_.has_value()) {
      internal::FlightData data = std::move(*first_read_);
      first_read_.reset();
      Consume(std::move(data));
    } else {
      RequestRead();
    }
    return pending_;
  }

  void SetReactor(ExchangeReactor* reactor) { reactor_.store(reactor); }
  /// Called by the reactor once the exchange is over: from then on the handler's
  /// reader must refuse to start reads instead of calling into a freed reactor.
  void DetachReactor() { reactor_.store(nullptr); }
  void SetDescriptor(const FlightDescriptor& descriptor) { descriptor_ = descriptor; }
  void OfferFirstRead(internal::FlightData data) { first_read_ = std::move(data); }

  void OnMessage(internal::FlightData data) {
    read_in_flight_ = false;
    Consume(std::move(data));
  }

  void OnReadClosed() {
    read_in_flight_ = false;
    read_closed_ = true;
    ResolvePending(FlightStreamChunk{});
  }

  void OnReadFailed(Status status) {
    read_in_flight_ = false;
    read_error_ = std::move(status);
    ResolvePending(read_error_);
  }

  void OnCancelled() {
    read_closed_ = true;
    ResolvePending(arrow::Status::Cancelled("the client cancelled the exchange"));
  }

 private:
  void ResolvePending(FlightStreamChunk chunk) {
    if (pending_.is_valid() && !pending_.is_finished()) {
      pending_.MarkFinished(std::move(chunk));
    }
  }

  void ResolvePending(const Status& status) {
    if (pending_.is_valid() && !pending_.is_finished()) {
      pending_.MarkFinished(status);
    }
  }

  void Consume(internal::FlightData data) {
    if (read_closed_) {
      return;
    }
    decoder_.Consume(std::move(data)).AddCallback([this](const arrow::Status& status) {
      if (!status.ok()) {
        read_error_ = status;
        ResolvePending(status);
      } else if (pending_.is_valid() && !pending_.is_finished()) {
        RequestRead();
      }
    });
  }

  void RequestRead();

  class ChunkListener final : public AsyncFlightDataListener {
   public:
    explicit ChunkListener(ExchangeReader* reader) : reader_(reader) {}

    arrow::Future<> OnNext(FlightStreamChunk chunk) override {
      if (reader_->pending_.is_valid() && !reader_->pending_.is_finished()) {
        reader_->ResolvePending(std::move(chunk));
      } else {
        reader_->buffered_ = std::move(chunk);
      }
      return arrow::Future<>::MakeFinished();
    }

    arrow::Future<> OnDescriptor(const FlightDescriptor&) override {
      return arrow::Future<>::MakeFinished();
    }

    arrow::Status OnSchemaDecoded(std::shared_ptr<Schema>) override {
      return arrow::Status::OK();
    }

   private:
    ExchangeReader* reader_;
  };

  std::atomic<ExchangeReactor*> reactor_{nullptr};
  std::shared_ptr<ChunkListener> listener_;
  AsyncFlightMessageDecoder decoder_;
  FlightDescriptor descriptor_;
  std::optional<internal::FlightData> first_read_;
  std::optional<FlightStreamChunk> buffered_;
  arrow::Future<FlightStreamChunk> pending_;
  Status read_error_;
  bool read_in_flight_ = false;
  bool read_closed_ = false;
};

class ExchangeWriter final : public AsyncFlightMessageWriter {
 public:
  explicit ExchangeWriter(ExchangeReactor* reactor) : reactor_(reactor) {}

  /// Called by the reactor once the exchange is over: from then on the handler's
  /// writer must not call into a freed reactor.
  void DetachReactor() { reactor_.store(nullptr); }

  arrow::Future<> BeginAsync(std::shared_ptr<Schema> schema) override {
    if (reactor_.load() == nullptr) {
      return arrow::Future<>::MakeFinished(arrow::Status::Cancelled(
          "the exchange is over: the client disconnected or cancelled"));
    }
    if (batch_writer_ != nullptr) {
      return arrow::Future<>::MakeFinished(
          arrow::Status::Invalid("This writer has already been started."));
    }
    auto sink = std::make_unique<PayloadSink>(this);
    ARROW_ASSIGN_OR_RAISE(batch_writer_, ipc::internal::OpenRecordBatchWriter(
                                             std::move(sink), schema, options_));
    return Flush();
  }

  arrow::Future<> WriteRecordBatchAsync(const RecordBatch& batch) override {
    pending_app_metadata_ = nullptr;
    return WriteBatch(batch);
  }

  arrow::Future<> WriteWithMetadataAsync(const RecordBatch& batch,
                                         std::shared_ptr<Buffer> app_metadata) override {
    pending_app_metadata_ = std::move(app_metadata);
    return WriteBatch(batch);
  }

  arrow::Future<> WriteMetadataAsync(std::shared_ptr<Buffer> app_metadata) override {
    FlightPayload payload;
    payload.app_metadata = std::move(app_metadata);
    queue_.push_back(std::move(payload));
    return Flush();
  }

  arrow::Future<> CloseAsync() override {
    if (batch_writer_ != nullptr) {
      auto status = batch_writer_->Close();
      batch_writer_ = nullptr;
      if (!status.ok()) {
        return arrow::Future<>::MakeFinished(std::move(status));
      }
    }
    return arrow::Future<>::MakeFinished();
  }

  void OnWriteDone(bool ok) {
    write_in_flight_ = false;
    if (!ok) {
      FailFlush(MakeFlightError(FlightStatusCode::Internal, "Failed to write response"));
      return;
    }
    Pump();
  }

  void OnCancelled() {
    if (write_in_flight_) {
      write_in_flight_ = false;
      FailFlush(arrow::Status::Cancelled("the client cancelled the exchange"));
    }
  }

 private:
  class PayloadSink final : public ipc::internal::IpcPayloadWriter {
   public:
    explicit PayloadSink(ExchangeWriter* writer) : writer_(writer) {}

    arrow::Status Start() override { return arrow::Status::OK(); }

    arrow::Status WritePayload(const ipc::IpcPayload& ipc_payload) override {
      FlightPayload payload;
      payload.ipc_message = ipc_payload;
      if (ipc_payload.type == ipc::MessageType::RECORD_BATCH &&
          writer_->pending_app_metadata_ != nullptr) {
        payload.app_metadata = std::move(writer_->pending_app_metadata_);
      }
      writer_->queue_.push_back(std::move(payload));
      return arrow::Status::OK();
    }

    arrow::Status Close() override { return arrow::Status::OK(); }

   private:
    ExchangeWriter* writer_;
  };

  arrow::Future<> WriteBatch(const RecordBatch& batch) {
    if (batch_writer_ == nullptr) {
      return arrow::Future<>::MakeFinished(arrow::Status::Invalid(
          "This writer is not started. Call BeginAsync() with a schema"));
    }
    auto status = batch_writer_->WriteRecordBatch(batch);
    if (!status.ok()) {
      return arrow::Future<>::MakeFinished(std::move(status));
    }
    return Flush();
  }

  arrow::Future<> Flush() {
    if (reactor_.load() == nullptr) {
      // The exchange is over: report it rather than pumping into a dead reactor.
      return arrow::Future<>::MakeFinished(arrow::Status::Cancelled(
          "the exchange is over: the client disconnected or cancelled"));
    }
    if (!flush_.is_valid() || flush_.is_finished()) {
      flush_ = arrow::Future<>::Make();
    }
    Pump();
    return flush_;
  }

  void Pump();

  void FailFlush(Status status) {
    if (flush_.is_valid() && !flush_.is_finished()) {
      flush_.MarkFinished(std::move(status));
    }
  }

  std::atomic<ExchangeReactor*> reactor_;
  std::unique_ptr<ipc::RecordBatchWriter> batch_writer_;
  ipc::IpcWriteOptions options_ = ipc::IpcWriteOptions::Defaults();
  std::shared_ptr<Buffer> pending_app_metadata_;
  std::deque<FlightPayload> queue_;
  arrow::Future<> flush_;
  bool write_in_flight_ = false;
};

class ExchangeReactor final : public AsyncReactorBase {
 public:
  ExchangeReactor(AsyncCallContext flight_context, AsyncGenericFlightServerBase* base)
      : AsyncReactorBase(std::move(flight_context)),
        base_(base),
        reader_(std::make_shared<ExchangeReader>()),
        writer_(std::make_shared<ExchangeWriter>(this)) {
    reader_->SetReactor(this);
    StartRead(&read_buf_);
  }

  void OnReadDone(bool ok) override {
    if (finished()) {
      return;
    }
    if (!ok) {
      if (!started_) {
        FinishOnce(MakeFlightError(FlightStatusCode::Internal, "Failed to read request"));
        return;
      }
      reader_->OnReadClosed();
      return;
    }
    const ::grpc::Status deserialized = FlightDataDeserialize(&read_buf_, &read_data_);
    if (!deserialized.ok()) {
      if (!started_) {
        FinishOnce(
            MakeFlightError(FlightStatusCode::Internal, deserialized.error_message()));
        return;
      }
      reader_->OnReadFailed(
          MakeFlightError(FlightStatusCode::Internal, deserialized.error_message()));
      return;
    }
    if (!started_) {
      started_ = true;
      if (read_data_.descriptor == nullptr) {
        FinishOnce(MakeFlightError(FlightStatusCode::Internal,
                                   "Descriptor missing on first message"));
        return;
      }
      reader_->SetDescriptor(*read_data_.descriptor);
      reader_->OfferFirstRead(std::move(read_data_));
      Hold();
      arrow::Future<> exchange =
          base_->DoExchangeAsync(flight_context(), reader_, writer_);
      exchange.AddCallback([this](const arrow::Status& status) {
        if (!finished()) {
          FinishOnce(status);
        }
        ReleaseHold();
      });
      return;
    }
    reader_->OnMessage(std::move(read_data_));
  }

  void OnWriteDone(bool ok) override {
    if (finished()) {
      return;
    }
    writer_->OnWriteDone(ok);
  }

  void OnCancel() override {
    FinishOnce(arrow::Status::Cancelled());
    reader_->OnCancelled();
    writer_->OnCancelled();
  }

  void OnDone() override {
    // The handler may still hold the reader/writer: cut their back-pointers
    // before this reactor is deleted so a late call fails cleanly instead of
    // dereferencing freed memory.
    reader_->DetachReactor();
    writer_->DetachReactor();
    AsyncReactorBase::OnDone();
  }

  void StartReadNext() {
    if (!finished()) {
      StartRead(&read_buf_);
    }
  }

  arrow::Status WritePayload(FlightPayload payload) {
    if (finished()) {
      return arrow::Status::Cancelled("the exchange is over");
    }
    bool own_buffer = false;
    const ::grpc::Status serialize =
        FlightDataSerialize(payload, &write_buf_, &own_buffer);
    (void)own_buffer;
    if (!serialize.ok()) {
      return MakeFlightError(FlightStatusCode::Internal, serialize.error_message());
    }
    StartWrite(&write_buf_);
    return arrow::Status::OK();
  }

 private:
  AsyncGenericFlightServerBase* base_;
  std::shared_ptr<ExchangeReader> reader_;
  std::shared_ptr<ExchangeWriter> writer_;
  ::grpc::ByteBuffer read_buf_;
  ::grpc::ByteBuffer write_buf_;
  internal::FlightData read_data_;
  bool started_ = false;
};

void ExchangeReader::RequestRead() {
  ExchangeReactor* reactor = reactor_.load();
  if (read_in_flight_ || read_closed_ || reactor == nullptr) {
    return;
  }
  read_in_flight_ = true;
  reactor->StartReadNext();
}

void ExchangeWriter::Pump() {
  if (write_in_flight_) {
    return;
  }
  if (queue_.empty()) {
    if (flush_.is_valid() && !flush_.is_finished()) {
      flush_.MarkFinished();
    }
    return;
  }
  ExchangeReactor* reactor = reactor_.load();
  if (reactor == nullptr) {
    write_in_flight_ = false;
    FailFlush(arrow::Status::Cancelled(
        "the exchange is over: the client disconnected or cancelled"));
    return;
  }
  FlightPayload payload = std::move(queue_.front());
  queue_.pop_front();
  write_in_flight_ = true;
  const auto status = reactor->WritePayload(std::move(payload));
  if (!status.ok()) {
    write_in_flight_ = false;
    FailFlush(std::move(status));
  }
}

}  // namespace

::grpc::ServerGenericBidiReactor* MakeExchangeReactor(
    AsyncCallContext flight_context, AsyncGenericFlightServerBase* base) {
  return new ExchangeReactor(std::move(flight_context), base);
}

}  // namespace arrow::flight::transport::grpc::detail