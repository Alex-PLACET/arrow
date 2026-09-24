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

#include "arrow/flight/flight_data_decoder.h"

#include <algorithm>
#include <deque>

#include "arrow/flight/serialization_internal.h"
#include "arrow/flight/transport.h"
#include "arrow/ipc/message.h"
#include "arrow/ipc/reader.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow::flight {

namespace {

// FlightDataMessageReader is an ipc::MessageReader that accepts one at a time
// FlightData messages. Analogous to MessageReader::Open(InputStream*) but for
// individual FlightData messages directly read from the received buffers.
//
// Messages are queued rather than handed over one by one: the IPC reader reads
// messages until it finds a record batch, walking through the dictionary
// messages that precede it, and to it an empty queue means end-of-stream.
// A dictionary message on its own would therefore end the stream early, so the
// decoder only hands a queued run over once the record batch that ends it has
// arrived (see HasRecordBatch).
class FlightDataMessageReader : public ipc::MessageReader {
 public:
  /// Queue one message, with the app metadata it arrived with. app_metadata is
  /// kept per message: the decoder reads the app metadata of the message the
  /// reader took last via ReadAppMetadata() after ReadNext() returns.
  void Push(std::unique_ptr<ipc::Message> message, std::shared_ptr<Buffer> app_metadata) {
    queue_.emplace_back(std::move(message), std::move(app_metadata));
  }

  /// Whether a record batch message is queued.
  //  The IPC reader reads through everything up to a record batch, so a queued run ending
  //  in one can be
  /// handed over without it seeing a premature end-of-stream.
  /// \return True if a record batch message is queued, false otherwise.
  bool HasRecordBatch() const {
    return std::ranges::any_of(queue_, [](const QueuedMessage& queued) {
      return queued.message->type() == ipc::MessageType::RECORD_BATCH;
    });
  }

  /// Take the app metadata of the last queued message, leaving that message
  /// with none: its app metadata has been delivered already.
  /// \return The app metadata of the last queued message, or nullptr if no messages are
  /// queued.
  std::shared_ptr<Buffer> TakeLastAppMetadata() {
    if (queue_.empty()) {
      return nullptr;
    }
    return std::move(queue_.back().app_metadata);
  }

  /// Take the next queued message, if any, and return it. The app metadata of
  /// the message taken is stored internally and can be retrieved with
  /// ReadAppMetadata(). Returns nullptr if no messages are queued.
  /// \return The next queued message, or nullptr if no messages are queued.
  ::arrow::Result<std::unique_ptr<ipc::Message>> ReadNextMessage() override {
    if (queue_.empty()) {
      return nullptr;
    }
    QueuedMessage queued = std::move(queue_.front());
    queue_.pop_front();
    app_metadata_ = std::move(queued.app_metadata);
    return std::move(queued.message);
  }

  std::shared_ptr<Buffer> ReadAppMetadata() { return app_metadata_; }

 private:
  /// Queue of messages that have been pushed but not yet read.
  struct QueuedMessage {
    QueuedMessage() = default;
    QueuedMessage(std::unique_ptr<ipc::Message> message,
                  std::shared_ptr<Buffer> app_metadata)
        : message(std::move(message)), app_metadata(std::move(app_metadata)) {}
    std::unique_ptr<ipc::Message> message;
    std::shared_ptr<Buffer> app_metadata;
  };

  std::deque<QueuedMessage> queue_;
  /// App metadata of the message the IPC reader took last.
  std::shared_ptr<Buffer> app_metadata_;
};

}  // namespace

class FlightMessageDecoder::FlightMessageDecoderImpl {
 public:
  FlightMessageDecoderImpl(std::shared_ptr<FlightDataListener> listener,
                           ipc::IpcReadOptions options)
      : listener_(std::move(listener)),
        options_(std::move(options)),
        message_reader_(new FlightDataMessageReader()) {}

  /// \brief Consume a chunk of Flight data.
  /// \param data The Flight data to consume.
  /// \return Status indicating success or failure.
  Status ConsumeData(internal::FlightData data) {
    if (data.descriptor) {
      RETURN_NOT_OK(listener_->OnDescriptor(*data.descriptor));
    }

    if (!data.metadata) {
      // Metadata-only message: no IPC content, just Flight app_metadata.
      if (data.app_metadata && data.app_metadata->size() > 0) {
        FlightStreamChunk chunk;
        chunk.app_metadata = std::move(data.app_metadata);
        RETURN_NOT_OK(listener_->OnNext(std::move(chunk)));
      }
      return Status::OK();
    }

    ARROW_ASSIGN_OR_RAISE(auto message, data.OpenMessage());

    if (!batch_reader_) {
      // Initialize RecordBatchStreamReader and read the first IPC message.
      // It must be a schema.
      // RecordBatchStreamReader requiring unique_ptr is slightly awkward
      // since we want to keep a reference to the message reader.
      message_reader_->Push(std::move(message), std::move(data.app_metadata));
      ARROW_ASSIGN_OR_RAISE(
          batch_reader_,
          ipc::RecordBatchStreamReader::Open(
              std::unique_ptr<ipc::MessageReader>(message_reader_), options_));
      return listener_->OnSchemaDecoded(batch_reader_->schema());
    }

    message_reader_->Push(std::move(message), std::move(data.app_metadata));

    if (!message_reader_->HasRecordBatch()) {
      // Hold the queued run until the record batch that ends it has arrived:
      // the transport hands the decoder one message at a time, and the IPC
      // reader taking a dictionary message on its own would ask for the next
      // message, find the queue empty, and declare the stream ended (it reads
      // through a run of dictionary messages before the record batch).
      // A held-back message carrying app_metadata is still a value: deliver it
      // now, in wire order, without waiting for the record batch.
      auto app_metadata = message_reader_->TakeLastAppMetadata();
      if (app_metadata && app_metadata->size() > 0) {
        FlightStreamChunk chunk;
        chunk.app_metadata = std::move(app_metadata);
        RETURN_NOT_OK(listener_->OnNext(std::move(chunk)));
      }
      return Status::OK();
    }

    std::shared_ptr<RecordBatch> batch;
    RETURN_NOT_OK(batch_reader_->ReadNext(&batch));
    auto app_metadata = message_reader_->ReadAppMetadata();

    if (batch) {
      FlightStreamChunk chunk{std::move(batch), std::move(app_metadata)};
      return listener_->OnNext(std::move(chunk));
    }

    // This has to be a Dictionary batch.
    // TODO: Add unit test validating assumption.
    if (app_metadata && app_metadata->size() > 0) {
      FlightStreamChunk chunk{nullptr, std::move(app_metadata)};
      return listener_->OnNext(std::move(chunk));
    }
    return Status::OK();
  }

  std::shared_ptr<Schema> schema() const {
    return batch_reader_ ? batch_reader_->schema() : nullptr;
  }

 private:
  std::shared_ptr<FlightDataListener> listener_;
  ipc::IpcReadOptions options_;
  // This is owned by the RecordBatchStreamReader once it's passed to it.
  // We want to keep a reference to it so we can extract the app_metadata.
  FlightDataMessageReader* message_reader_;
  std::shared_ptr<ipc::RecordBatchStreamReader> batch_reader_;
};

FlightMessageDecoder::FlightMessageDecoder(std::shared_ptr<FlightDataListener> listener,
                                           ipc::IpcReadOptions options)
    : impl_(std::make_unique<FlightMessageDecoderImpl>(std::move(listener),
                                                       std::move(options))) {}

FlightMessageDecoder::~FlightMessageDecoder() = default;

Status FlightMessageDecoder::Consume(std::shared_ptr<Buffer> buffer) {
  ARROW_ASSIGN_OR_RAISE(auto data, internal::DeserializeFlightData(buffer));
  return impl_->ConsumeData(std::move(data));
}

Status FlightMessageDecoder::Consume(internal::FlightData data) {
  return impl_->ConsumeData(std::move(data));
}

std::shared_ptr<Schema> FlightMessageDecoder::schema() const { return impl_->schema(); }

}  // namespace arrow::flight
