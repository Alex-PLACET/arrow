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

// End-to-end tests of the async generic gRPC Flight service: a real
// FlightServerBase served through AsyncGenericFlightService, driven by the
// regular sync FlightClient.

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <grpcpp/generic/generic_stub.h>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include "arrow/flight/client.h"
#include "arrow/flight/server.h"
#include "arrow/flight/server_async.h"
#include "arrow/flight/server_middleware.h"
#include "arrow/flight/test_auth_handlers.h"
#include "arrow/flight/transport/grpc/async_grpc_service.h"
#include "arrow/scalar.h"
#include "arrow/testing/gtest_util.h"

namespace arrow::flight::transport::grpc {

namespace {

// The ticket the test server answers with a stream that has a schema and no
// batches at all; every other ticket gets the 3-batch stream.
constexpr const char* kSchemaOnlyTicket = "schema-only-do-get-ticket";

// One canned FlightInfo; `descriptor` names it for the assertions.
arrow::Result<std::unique_ptr<FlightInfo>> MakeTestFlightInfo(
    FlightDescriptor descriptor) {
  ARROW_ASSIGN_OR_RAISE(
      auto made, FlightInfo::Make(*arrow::schema({arrow::field("value", arrow::int64())}),
                                  std::move(descriptor), std::vector<FlightEndpoint>{},
                                  /*total_records=*/-1, /*total_bytes=*/-1));
  return std::make_unique<FlightInfo>(std::move(made));
}

// A ResultStream that yields one Result and then fails: the reactor must
// surface the error as the RPC status, not truncate the stream into a success.
class MidStreamErrorResultStream : public ResultStream {
 public:
  arrow::Result<std::unique_ptr<Result>> Next() override {
    if (served_) {
      return arrow::Status::IOError("mid-stream action failure");
    }
    served_ = true;
    return std::make_unique<Result>(arrow::Buffer::FromString("first"));
  }

 private:
  bool served_ = false;
};

// The same shape for ListFlights: one FlightInfo, then an error.
class MidStreamErrorFlightListing : public FlightListing {
 public:
  arrow::Result<std::unique_ptr<FlightInfo>> Next() override {
    if (served_) {
      return arrow::Status::IOError("mid-stream listing failure");
    }
    served_ = true;
    return MakeTestFlightInfo(FlightDescriptor::Path({"first-listing"}));
  }

 private:
  bool served_ = false;
};

// Wraps a synchronous stream for the async interface: the payloads are already
// available, so each future is finished when it is created.  Test-local stand-in
// for the deleted library bridge, for tests that want an already-resolved stream.
class ResolvedStream : public AsyncFlightDataStream {
 public:
  explicit ResolvedStream(std::unique_ptr<FlightDataStream> stream)
      : stream_(std::move(stream)) {}
  std::shared_ptr<Schema> schema() override { return stream_->schema(); }
  arrow::Future<FlightPayload> GetSchemaPayloadAsync() override {
    return arrow::Future<FlightPayload>::MakeFinished(stream_->GetSchemaPayload());
  }
  arrow::Future<FlightPayload> NextAsync() override {
    return arrow::Future<FlightPayload>::MakeFinished(stream_->Next());
  }
  Status Close() override { return stream_->Close(); }

 private:
  std::unique_ptr<FlightDataStream> stream_;
};

// A vector-backed AsyncFlightListing: every NextAsync() resolves immediately.
class SimpleListingAdapter final : public AsyncFlightListing {
 public:
  explicit SimpleListingAdapter(std::vector<FlightInfo> flights)
      : flights_(std::move(flights)) {}

  arrow::Future<std::shared_ptr<FlightInfo>> NextAsync() override {
    if (next_ >= flights_.size()) {
      return arrow::Future<std::shared_ptr<FlightInfo>>::MakeFinished(
          std::shared_ptr<FlightInfo>{});
    }
    return arrow::Future<std::shared_ptr<FlightInfo>>::MakeFinished(
        std::make_shared<FlightInfo>(std::move(flights_[next_++])));
  }

 private:
  std::vector<FlightInfo> flights_;
  size_t next_ = 0;
};

// Bridges the sync inner server's FlightListing to the async interface: each
// NextAsync() pulls one FlightInfo on the calling thread.
class SyncListingAdapter final : public AsyncFlightListing {
 public:
  explicit SyncListingAdapter(std::unique_ptr<FlightListing> listing)
      : listing_(std::move(listing)) {}

  arrow::Future<std::shared_ptr<FlightInfo>> NextAsync() override {
    auto next = listing_->Next();
    if (!next.ok()) {
      return arrow::Future<std::shared_ptr<FlightInfo>>::MakeFinished(next.status());
    }
    if (*next == nullptr) {
      return arrow::Future<std::shared_ptr<FlightInfo>>::MakeFinished(
          std::shared_ptr<FlightInfo>{});
    }
    return arrow::Future<std::shared_ptr<FlightInfo>>::MakeFinished(
        std::shared_ptr<FlightInfo>(std::move(*next)));
  }

 private:
  std::unique_ptr<FlightListing> listing_;
};

// The same bridge for DoAction's ResultStream.
class SyncResultStreamAdapter final : public AsyncResultStream {
 public:
  explicit SyncResultStreamAdapter(std::unique_ptr<ResultStream> results)
      : results_(std::move(results)) {}

  arrow::Future<std::shared_ptr<Result>> NextAsync() override {
    auto next = results_->Next();
    if (!next.ok()) {
      return arrow::Future<std::shared_ptr<Result>>::MakeFinished(next.status());
    }
    if (*next == nullptr) {
      return arrow::Future<std::shared_ptr<Result>>::MakeFinished(
          std::shared_ptr<Result>{});
    }
    return arrow::Future<std::shared_ptr<Result>>::MakeFinished(
        std::shared_ptr<Result>(std::move(*next)));
  }

 private:
  std::unique_ptr<ResultStream> results_;
};

// Serves an existing FlightServerBase-derived test server on the async
// transport.  Deriving from the async class is what selects the async generic
// transport now, so wrapping is the test-side equivalent of the flag that used
// to do it.  The inner server must outlive the adapter (it holds a raw pointer).
class TestServerAsyncAdapter : public AsyncGenericFlightServerBase {
 public:
  explicit TestServerAsyncAdapter(
      FlightServerBase* inner,
      std::shared_ptr<AsyncFlightDataListener> listener = nullptr)
      : inner_(inner), listener_(std::move(listener)) {}

  /// \brief Hands out the listener the test installed, if any; nullptr (the
  /// default) refuses uploads, which is the base class behavior.
  std::shared_ptr<AsyncFlightDataListener> CreateDoPutListener(
      const ServerCallContext& context) override {
    return listener_;
  }

  arrow::Future<std::shared_ptr<FlightInfo>> GetFlightInfoAsync(
      const ServerCallContext& context, const FlightDescriptor& request) override {
    std::unique_ptr<FlightInfo> info;
    const auto status = inner_->GetFlightInfo(context, request, &info);
    if (!status.ok()) {
      return arrow::Future<std::shared_ptr<FlightInfo>>::MakeFinished(status);
    }
    return arrow::Future<std::shared_ptr<FlightInfo>>::MakeFinished(
        std::shared_ptr<FlightInfo>(std::move(info)));
  }

  arrow::Future<std::shared_ptr<SchemaResult>> GetSchemaAsync(
      const ServerCallContext& context, const FlightDescriptor& request) override {
    std::unique_ptr<SchemaResult> schema;
    const auto status = inner_->GetSchema(context, request, &schema);
    if (!status.ok()) {
      return arrow::Future<std::shared_ptr<SchemaResult>>::MakeFinished(status);
    }
    return arrow::Future<std::shared_ptr<SchemaResult>>::MakeFinished(
        std::shared_ptr<SchemaResult>(std::move(schema)));
  }

  arrow::Future<std::shared_ptr<PollInfo>> PollFlightInfoAsync(
      const ServerCallContext& context, const FlightDescriptor& request) override {
    std::unique_ptr<PollInfo> info;
    const auto status = inner_->PollFlightInfo(context, request, &info);
    if (!status.ok()) {
      return arrow::Future<std::shared_ptr<PollInfo>>::MakeFinished(status);
    }
    return arrow::Future<std::shared_ptr<PollInfo>>::MakeFinished(
        std::shared_ptr<PollInfo>(std::move(info)));
  }

  arrow::Future<std::shared_ptr<AsyncFlightListing>> ListFlightsAsync(
      const ServerCallContext& context, const Criteria* criteria) override {
    std::unique_ptr<FlightListing> listing;
    const auto status = inner_->ListFlights(context, criteria, &listing);
    if (!status.ok()) {
      return arrow::Future<std::shared_ptr<AsyncFlightListing>>::MakeFinished(status);
    }
    if (listing == nullptr) {
      return arrow::Future<std::shared_ptr<AsyncFlightListing>>::MakeFinished(
          std::shared_ptr<AsyncFlightListing>{});
    }
    return arrow::Future<std::shared_ptr<AsyncFlightListing>>::MakeFinished(
        std::make_shared<SyncListingAdapter>(std::move(listing)));
  }

  arrow::Future<std::vector<ActionType>> ListActionsAsync(
      const ServerCallContext& context) override {
    std::vector<ActionType> actions;
    const auto status = inner_->ListActions(context, &actions);
    if (!status.ok()) {
      return arrow::Future<std::vector<ActionType>>::MakeFinished(status);
    }
    return arrow::Future<std::vector<ActionType>>::MakeFinished(std::move(actions));
  }

  arrow::Future<std::shared_ptr<AsyncResultStream>> DoActionAsync(
      const ServerCallContext& context, const Action& action) override {
    std::unique_ptr<ResultStream> results;
    const auto status = inner_->DoAction(context, action, &results);
    if (!status.ok()) {
      return arrow::Future<std::shared_ptr<AsyncResultStream>>::MakeFinished(status);
    }
    if (results == nullptr) {
      // OK with no stream: the transport answers CANCELLED for this, so the
      // null must travel.
      return arrow::Future<std::shared_ptr<AsyncResultStream>>::MakeFinished(
          std::shared_ptr<AsyncResultStream>{});
    }
    return arrow::Future<std::shared_ptr<AsyncResultStream>>::MakeFinished(
        std::make_shared<SyncResultStreamAdapter>(std::move(results)));
  }

  arrow::Future<std::shared_ptr<AsyncFlightDataStream>> DoGetAsync(
      const ServerCallContext& context, const Ticket& request) override {
    std::unique_ptr<FlightDataStream> stream;
    const auto status = inner_->DoGet(context, request, &stream);
    if (!status.ok()) {
      return arrow::Future<std::shared_ptr<AsyncFlightDataStream>>::MakeFinished(status);
    }
    return arrow::Future<std::shared_ptr<AsyncFlightDataStream>>::MakeFinished(
        std::make_shared<ResolvedStream>(std::move(stream)));
  }

 private:
  FlightServerBase* inner_;
  std::shared_ptr<AsyncFlightDataListener> listener_;
};

// Serves 3 batches of 5 rows for any ticket, and records the ticket it was
// asked for. DoGet runs on a gRPC thread; the test reads the ticket only
// after the server has shut down.  DoExchange echoes every chunk back, in
// order, which requires the exchange's reader and writer to be live at the
// same time.
class TestFlightServer : public FlightServerBase {
 public:
  arrow::Status DoGet(const ServerCallContext& context, const Ticket& request,
                      std::unique_ptr<FlightDataStream>* stream) override {
    ticket_ = request.ticket;

    auto schema = arrow::schema(
        {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
    if (request.ticket == kSchemaOnlyTicket) {
      // An empty reader still carries the schema; Next() reports the end of
      // stream right away.
      ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make({}, schema));
      *stream = std::make_unique<arrow::flight::RecordBatchStream>(std::move(reader));
      return arrow::Status::OK();
    }

    auto batch = arrow::RecordBatch::Make(
        schema, 5,
        {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3, 4, 5]"),
         arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30, 40, 50]")});
    ARROW_ASSIGN_OR_RAISE(auto reader,
                          arrow::RecordBatchReader::Make({batch, batch, batch}));
    *stream = std::make_unique<arrow::flight::RecordBatchStream>(std::move(reader));
    return arrow::Status::OK();
  }

  // The unary trio: canned responses.  `last_descriptor_` records the request
  // of the last one served (it is written on a gRPC thread, so tests read it
  // only after the server has shut down); Command("fail") is the sentinel that
  // drives the error-path test.
  arrow::Status GetFlightInfo(const ServerCallContext&, const FlightDescriptor& request,
                              std::unique_ptr<FlightInfo>* info) override {
    last_descriptor_ = request;
    if (request == FlightDescriptor::Command("fail")) {
      return arrow::Status::KeyError("no such flight");
    }
    if (request == FlightDescriptor::Command("null-info")) {
      // Answered OK without a response: the transport must answer what the
      // sync transport does for that case (grpc_server.cc: "Flight not found").
      return arrow::Status::OK();
    }
    ARROW_ASSIGN_OR_RAISE(
        auto made,
        FlightInfo::Make(*arrow::schema({arrow::field("value", arrow::int64())}),
                         FlightDescriptor{}, std::vector<FlightEndpoint>{}, -1, -1));
    *info = std::make_unique<FlightInfo>(std::move(made));
    return arrow::Status::OK();
  }

  arrow::Status GetSchema(const ServerCallContext&, const FlightDescriptor& request,
                          std::unique_ptr<SchemaResult>* schema) override {
    last_descriptor_ = request;
    ARROW_ASSIGN_OR_RAISE(
        *schema,
        SchemaResult::Make(*arrow::schema({arrow::field("value", arrow::int64())})));
    return arrow::Status::OK();
  }

  arrow::Status PollFlightInfo(const ServerCallContext&, const FlightDescriptor& request,
                               std::unique_ptr<PollInfo>* info) override {
    last_descriptor_ = request;
    auto poll_info = std::make_unique<PollInfo>();
    poll_info->descriptor = FlightDescriptor::Command("poll-next");
    poll_info->progress = 0.5;
    *info = std::move(poll_info);
    return arrow::Status::OK();
  }

  // The streaming trio: ListActions (a vector), DoAction (a ResultStream) and
  // ListFlights (a FlightListing).  `last_criteria_` records the filter the
  // client sent (read only after the server stopped); the "fail-mid-stream"
  // sentinels drive the error-path tests.
  arrow::Status ListActions(const ServerCallContext&,
                            std::vector<ActionType>* actions) override {
    *actions = {ActionType("echo", "echo back"), ActionType("fail", "always fails")};
    if (empty_actions_) {
      actions->clear();
    }
    return arrow::Status::OK();
  }

  arrow::Status DoAction(const ServerCallContext&, const Action& action,
                         std::unique_ptr<ResultStream>* results) override {
    last_action_type_ = action.type;
    if (action.type == "fail-mid-stream") {
      *results = std::make_unique<MidStreamErrorResultStream>();
      return arrow::Status::OK();
    }
    if (action.type == "null-result-stream") {
      // Answered OK without a stream: the sync transport answers CANCELLED for
      // that case (grpc_server.cc:404-406), before writing anything.
      return arrow::Status::OK();
    }
    std::vector<Result> made;
    made.emplace_back(arrow::Buffer::FromString("one"));
    if (action.type == "multi") {
      made.emplace_back(arrow::Buffer::FromString("two"));
      made.emplace_back(arrow::Buffer::FromString("three"));
    }
    *results = std::make_unique<SimpleResultStream>(std::move(made));
    return arrow::Status::OK();
  }

  arrow::Status ListFlights(const ServerCallContext&, const Criteria* criteria,
                            std::unique_ptr<FlightListing>* listings) override {
    if (criteria != nullptr) {
      last_criteria_ = *criteria;
    }
    const std::string expression = criteria == nullptr ? "" : criteria->expression;
    if (expression == "fail-mid-stream") {
      *listings = std::make_unique<MidStreamErrorFlightListing>();
      return arrow::Status::OK();
    }
    if (expression.empty()) {
      // No criteria: the sync handler passes an empty (never null) Criteria
      // to FlightServerBase::ListFlights (grpc_server.cc:252-257); the async
      // reactor must do the same, so an empty expression is reachable here.
      *listings = std::make_unique<SimpleFlightListing>(std::vector<FlightInfo>{});
      return arrow::Status::OK();
    }
    std::vector<FlightInfo> flights;
    for (const auto& name : {"first-listing", "second-listing"}) {
      ARROW_ASSIGN_OR_RAISE(auto info,
                            MakeTestFlightInfo(FlightDescriptor::Path({name})));
      flights.push_back(std::move(*info));
    }
    *listings = std::make_unique<SimpleFlightListing>(std::move(flights));
    return arrow::Status::OK();
  }

  void set_empty_actions() { empty_actions_ = true; }

  const std::string& ticket() const { return ticket_; }
  const FlightDescriptor& last_descriptor() const { return last_descriptor_; }
  const std::string& last_action_type() const { return last_action_type_; }
  const Criteria& last_criteria() const { return last_criteria_; }

 private:
  std::string ticket_;
  FlightDescriptor last_descriptor_;
  std::string last_action_type_;
  Criteria last_criteria_;
  bool empty_actions_ = false;
};

// Echoes every chunk of a DoExchange back, in order, on the async API: the
// reader's demand drives the loop, each write is awaited, and the all-null
// chunk ends the exchange.  `exchanges_in_flight` proves an abandoned
// exchange still leaves the handler: the state object is freed when the last
// continuation dies and decrements the counter on destruction.
class EchoExchangeTestServer : public AsyncGenericFlightServerBase {
 public:
  arrow::Future<> DoExchangeAsync(
      const ServerCallContext&, std::shared_ptr<AsyncFlightMessageReader> reader,
      std::shared_ptr<AsyncFlightMessageWriter> writer) override {
    exchange_count_.fetch_add(1);
    exchanges_in_flight_.fetch_add(1);
    auto state = std::make_shared<EchoState>();
    state->reader = std::move(reader);
    state->writer = std::move(writer);
    state->in_flight = &exchanges_in_flight_;
    return EchoNext(state);
  }

  int exchanges_in_flight() const { return exchanges_in_flight_.load(); }
  int exchange_count() const { return exchange_count_.load(); }

 private:
  struct EchoState {
    std::shared_ptr<AsyncFlightMessageReader> reader;
    std::shared_ptr<AsyncFlightMessageWriter> writer;
    std::atomic<int>* in_flight = nullptr;
    bool begun = false;
    ~EchoState() { in_flight->fetch_sub(1); }
  };

  arrow::Future<> EchoNext(const std::shared_ptr<EchoState>& state) {
    // ponytail: the recursion depth follows the batch count of one exchange.
    // Flatten into a loop if a single exchange ever carries thousands.
    return state->reader->NextAsync().Then(
        [this, state](const FlightStreamChunk& chunk) -> arrow::Future<> {
          if (chunk.data == nullptr && chunk.app_metadata == nullptr) {
            // End of the exchange.
            return arrow::Future<>::MakeFinished();
          }
          arrow::Future<> written = arrow::Future<>::MakeFinished();
          if (chunk.data != nullptr && !state->begun) {
            state->begun = true;
            written = state->writer->BeginAsync(chunk.data->schema());
          }
          if (chunk.data != nullptr && chunk.app_metadata != nullptr) {
            written = written.Then([state, chunk]() {
              return state->writer->WriteWithMetadataAsync(*chunk.data,
                                                           chunk.app_metadata);
            });
          } else if (chunk.data != nullptr) {
            written = written.Then([state, chunk]() {
              return state->writer->WriteRecordBatchAsync(*chunk.data);
            });
          } else {
            written = written.Then([state, chunk]() {
              return state->writer->WriteMetadataAsync(chunk.app_metadata);
            });
          }
          return written.Then([this, state]() { return EchoNext(state); });
        });
  }

  std::atomic<int> exchanges_in_flight_{0};
  std::atomic<int> exchange_count_{0};
};

// Records what the decoder hands to the application: the schema, every chunk,
// and whether the consumer accepted it. OnNext returns whatever status the
// test configured, so the upload's error path is reachable from here.
class RecordingListener : public AsyncFlightDataListener {
 public:
  arrow::Status OnSchemaDecoded(std::shared_ptr<arrow::Schema> schema) override {
    ++schema_count_;
    schema_ = std::move(schema);
    return arrow::Status::OK();
  }

  arrow::Future<> OnNext(FlightStreamChunk chunk) override {
    if (chunk.data) {
      batches_.push_back(std::move(chunk.data));
    } else {
      ++metadata_only_count_;
      metadata_chunks_.push_back(std::move(chunk.app_metadata));
    }
    last_status_ = next_status_;
    return arrow::Future<>::MakeFinished(next_status_);
  }

  // Records the upload's descriptor and reports whatever status the test
  // configured, so the descriptor rejection path is reachable from here too.
  arrow::Future<> OnDescriptor(const FlightDescriptor& descriptor) override {
    descriptors_.push_back(descriptor);
    return arrow::Future<>::MakeFinished(descriptor_status_);
  }

  // Consumer-side rejection of the upload, as the decoder tests do.
  void set_next_status(arrow::Status status) { next_status_ = std::move(status); }
  // Rejection before any schema or data arrives.
  void set_descriptor_status(arrow::Status status) {
    descriptor_status_ = std::move(status);
  }

  int schema_count() const { return schema_count_; }
  int descriptor_count() const { return static_cast<int>(descriptors_.size()); }
  int metadata_only_count() const { return metadata_only_count_; }
  const std::shared_ptr<arrow::Schema>& schema() const { return schema_; }
  const std::vector<FlightDescriptor>& descriptors() const { return descriptors_; }
  const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches() const {
    return batches_;
  }
  const std::vector<std::shared_ptr<arrow::Buffer>>& metadata_chunks() const {
    return metadata_chunks_;
  }
  const arrow::Status& last_status() const { return last_status_; }

 private:
  int schema_count_ = 0;
  int metadata_only_count_ = 0;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<FlightDescriptor> descriptors_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  std::vector<std::shared_ptr<arrow::Buffer>> metadata_chunks_;
  arrow::Status descriptor_status_;
  arrow::Status next_status_;
  arrow::Status last_status_;
};

// A listener that records the terminal events on top of RecordingListener:
// OnFinish (the upload ended, whichever way) and what Cancel() reported, so
// the tests can assert on an upload cancelled from the server side.
class FinishRecordingListener : public RecordingListener {
 public:
  arrow::Future<> OnFinish(Status status) override {
    std::lock_guard<std::mutex> lock(mutex_);
    finish_count_++;
    finish_status_ = std::move(status);
    return arrow::Future<>::MakeFinished(arrow::Status::OK());
  }

  /// Cancel from whatever thread the test drives; records what Cancel() said.
  void CancelWith(Status status) {
    const arrow::Status cancel_status = Cancel(std::move(status)).status();
    std::lock_guard<std::mutex> lock(mutex_);
    cancel_status_ = cancel_status;
    ++cancel_count_;
  }

  int finish_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return finish_count_;
  }
  int cancel_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancel_count_;
  }
  Status finish_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return finish_status_;
  }
  Status cancel_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancel_status_;
  }

 private:
  mutable std::mutex mutex_;
  int finish_count_ = 0;
  int cancel_count_ = 0;
  Status finish_status_;
  Status cancel_status_;
};

// The upload half of TestServerAsyncAdapter as a server class method: the
// transport asks the server for the per-RPC listener through the
// CreateDoPutListener() virtual.  One fresh listener per RPC, as overlapping uploads
// require; the test reads them only after the server has stopped, or through
// WaitForListener (which is what a cancel test needs: it acts while the upload is in
// flight).
class TestUploadServerAsyncAdapter : public AsyncGenericFlightServerBase {
 public:
  std::shared_ptr<AsyncFlightDataListener> CreateDoPutListener(
      const ServerCallContext& context) override {
    auto listener = std::make_shared<FinishRecordingListener>();
    std::lock_guard<std::mutex> lock(mutex_);
    listeners_.push_back(listener);
    return listener;
  }

  int calls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(listeners_.size());
  }

  std::shared_ptr<FinishRecordingListener> listener(int i) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return listeners_[i];
  }

  /// \brief Wait (bounded) for the listener of the i-th upload to exist.
  /// The factory runs on a gRPC thread; a test that must act mid-upload waits
  /// for it here instead of racing the read.
  std::shared_ptr<FinishRecordingListener> WaitForListener(int i) const {
    for (int attempt = 0; attempt < 500; attempt++) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (static_cast<int>(listeners_.size()) > i) return listeners_[i];
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return nullptr;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<FinishRecordingListener>> listeners_;
};

// Middleware execution counts, shared between the factory, the middleware and
// the test thread (all three touch it).
struct MiddlewareTrace {
  std::mutex mutex;
  int start_call = 0;
  int sending_headers = 0;
  int call_completed = 0;
};

// Records middleware execution so a test can assert StartCall/CallCompleted
// ran on the async path (the sync path is covered by TestMiddleware).
class RecordingServerMiddleware : public ServerMiddleware {
 public:
  explicit RecordingServerMiddleware(std::shared_ptr<MiddlewareTrace> trace)
      : trace_(std::move(trace)) {}

  std::string name() const override { return "recording"; }
  void SendingHeaders(AddCallHeaders* /*outgoing_headers*/) override {
    std::lock_guard<std::mutex> guard(trace_->mutex);
    trace_->sending_headers++;
  }
  void CallCompleted(const Status& /*status*/) override {
    std::lock_guard<std::mutex> guard(trace_->mutex);
    trace_->call_completed++;
  }

 private:
  std::shared_ptr<MiddlewareTrace> trace_;
};

class RecordingServerMiddlewareFactory : public ServerMiddlewareFactory {
 public:
  explicit RecordingServerMiddlewareFactory(std::shared_ptr<MiddlewareTrace> trace)
      : trace_(std::move(trace)) {}

  Status StartCall(const CallInfo& /*info*/, const ServerCallContext& /*context*/,
                   std::shared_ptr<ServerMiddleware>* middleware) override {
    {
      std::lock_guard<std::mutex> guard(trace_->mutex);
      trace_->start_call++;
    }
    *middleware = std::make_shared<RecordingServerMiddleware>(trace_);
    return Status::OK();
  }

 private:
  std::shared_ptr<MiddlewareTrace> trace_;
};

class RejectingServerMiddlewareFactory : public ServerMiddlewareFactory {
 public:
  Status StartCall(const CallInfo&, const ServerCallContext&,
                   std::shared_ptr<ServerMiddleware>*) override {
    return Status::Invalid("rejected by middleware");
  }
};

// The gRPC transport builds the generic service with the shared context
// helper (which runs middleware and auth) and the server's memory manager
// (DoExchange's reader views bodies with it). Tests that register the service
// on a hand-built grpc::ServerBuilder must pass both explicitly.
std::shared_ptr<GrpcServerCallContextHelper<::grpc::CallbackServerContext>>
MakeAsyncHelper() {
  return std::make_shared<GrpcServerCallContextHelper<::grpc::CallbackServerContext>>(
      /*auth_handler=*/nullptr, MiddlewareFactoryList{});
}

/// The memory manager the hand-built service in each test uses; the default
/// CPU one, exactly what the real transport passes.
std::shared_ptr<MemoryManager> MakeAsyncMemoryManager() {
  return default_cpu_memory_manager();
}

// The method name no FlightService stub knows: the transport must answer it
// with the UNIMPLEMENTED fallback reactor rather than any typed path.  It is
// called over a raw generic stub, because only a generic client can name a
// method the service does not have.
constexpr const char* kUnknownMethod =
    "/arrow.flight.protocol.FlightService/NoSuchMethod";

// Call `kUnknownMethod` on the server listening on `port` with the sync-style
// generic stub API, and return the server's status.
::grpc::Status CallUnknownMethod(int port) {
  std::shared_ptr<::grpc::Channel> channel = ::grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(port), ::grpc::InsecureChannelCredentials());
  ::grpc::GenericStub stub(channel);
  ::grpc::ClientContext context;
  ::grpc::ByteBuffer request;
  ::grpc::ByteBuffer response;
  ::grpc::CompletionQueue cq;
  std::unique_ptr<::grpc::ClientAsyncResponseReader<::grpc::ByteBuffer>> call =
      stub.PrepareUnaryCall(&context, kUnknownMethod, request, &cq);
  call->StartCall();
  ::grpc::Status status;
  call->Finish(&response, &status, reinterpret_cast<void*>(1));
  void* tag = nullptr;
  bool ok = false;
  cq.Next(&tag, &ok);
  return status;
}

// Poll `predicate` until it holds or the timeout expires; the DoExchange tests
// use it to bound server-side cleanup instead of sleeping a fixed time.
template <typename Predicate>
bool WaitFor(Predicate predicate,
             std::chrono::milliseconds timeout = std::chrono::seconds(10)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return predicate();
}

// Upload one batch and report whichever surface first reports an error: the
// DoPut call, the write, or Close.
std::pair<std::string, arrow::Status> UploadOneBatch(
    FlightClient* client, const FlightDescriptor& descriptor,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<arrow::RecordBatch>& batch) {
  auto result = client->DoPut(descriptor, schema);
  if (!result.ok()) {
    return {"DoPut", result.status()};
  }
  auto write_status = result->writer->WriteRecordBatch(*batch);
  if (!write_status.ok()) {
    return {"WriteRecordBatch", write_status};
  }
  return {"Close", result->writer->Close()};
}

TEST(AsyncGrpcTest, BasicDoGet) {
  TestFlightServer inner_server;
  TestServerAsyncAdapter flight_server(&inner_server);
  AsyncGenericFlightService service(&flight_server, MakeAsyncMemoryManager(),
                                    MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  // Connect the existing arrow::flight::FlightClient implementation.
  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  // Call DoGet: the ticket must reach FlightServerBase::DoGet.
  Ticket ticket{"basic-do-get-ticket"};
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(ticket));

  // Read schema.
  ASSERT_OK_AND_ASSIGN(auto schema, reader->GetSchema());
  ASSERT_EQ(schema->num_fields(), 2);
  ASSERT_EQ(schema->field(0)->name(), "a");
  ASSERT_EQ(schema->field(1)->name(), "b");

  // Read batches.
  int batch_count = 0;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
    if (!chunk.data) break;
    ASSERT_EQ(chunk.data->num_rows(), 5);
    ASSERT_OK_AND_ASSIGN(auto value, chunk.data->column(0)->GetScalar(0));
    ASSERT_TRUE(value->Equals(*arrow::MakeScalar(int64_t(1))));
    batch_count++;
  }
  ASSERT_EQ(batch_count, 3);

  // Cleanup. The ticket is asserted only after the server stopped, so the
  // DoGet callback cannot race with the read.
  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  ASSERT_EQ(inner_server.ticket(), "basic-do-get-ticket");
}

TEST(AsyncGrpcTest, BasicDoPut) {
  TestFlightServer inner_server;
  // One listener for the whole test: shared with the service's factory and
  // kept here so it can be asserted on after the RPC.
  auto listener = std::make_shared<RecordingListener>();
  TestServerAsyncAdapter flight_server(&inner_server, listener);
  AsyncGenericFlightService service(&flight_server, MakeAsyncMemoryManager(),
                                    MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  // Connect the existing arrow::flight::FlightClient implementation.
  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  // Call DoPut and upload the same batch twice. Close() returns the server's
  // final status, i.e. it only succeeds once the acknowledgement written on
  // end of stream has been read.
  FlightDescriptor descriptor = FlightDescriptor::Path({"test"});
  ASSERT_OK_AND_ASSIGN(auto result, client->DoPut(descriptor, schema));
  ASSERT_OK(result.writer->WriteRecordBatch(*batch));
  ASSERT_OK(result.writer->WriteRecordBatch(*batch));
  ASSERT_OK(result.writer->Close());

  // Cleanup. The listener is asserted only after the server stopped, so the
  // DoPut callback cannot race with the reads.
  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  // The listener saw the schema once and both batches, in order.
  ASSERT_EQ(listener->schema_count(), 1);
  ASSERT_EQ(listener->schema()->num_fields(), 2);
  ASSERT_EQ(listener->schema()->field(0)->name(), "a");
  ASSERT_EQ(listener->schema()->field(1)->name(), "b");
  ASSERT_EQ(listener->batches().size(), 2);
  for (const auto& chunk : listener->batches()) {
    ASSERT_EQ(chunk->num_rows(), 3);
    ASSERT_EQ(chunk->num_columns(), 2);
    ASSERT_OK_AND_ASSIGN(auto value, chunk->column(0)->GetScalar(0));
    ASSERT_TRUE(value->Equals(*arrow::MakeScalar(int64_t(1))));
  }
  // No metadata-only (error/app-metadata) chunks, and nothing the consumer
  // rejected: a non-OK OnNext status would have failed DoPut with no ack.
  ASSERT_EQ(listener->metadata_only_count(), 0);
  ASSERT_OK(listener->last_status());
}

TEST(AsyncGrpcTest, SchemaOnlyDoGet) {
  TestFlightServer inner_server;
  TestServerAsyncAdapter flight_server(&inner_server);
  AsyncGenericFlightService service(&flight_server, MakeAsyncMemoryManager(),
                                    MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  // This ticket is served as a stream with no batches at all.
  Ticket ticket{kSchemaOnlyTicket};
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(ticket));

  // The schema is served...
  ASSERT_OK_AND_ASSIGN(auto schema, reader->GetSchema());
  ASSERT_EQ(schema->num_fields(), 2);
  ASSERT_EQ(schema->field(0)->name(), "a");
  ASSERT_EQ(schema->field(1)->name(), "b");

  // ...and the very first read reports the end of stream instead of failing.
  ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
  ASSERT_EQ(chunk.data, nullptr);

  // Cleanup. The ticket is asserted only after the server stopped, so the
  // DoGet callback cannot race with the read.
  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  ASSERT_EQ(inner_server.ticket(), kSchemaOnlyTicket);
}

TEST(AsyncGrpcTest, EmptyDoPut) {
  TestFlightServer inner_server;
  auto listener = std::make_shared<RecordingListener>();
  TestServerAsyncAdapter flight_server(&inner_server, listener);
  AsyncGenericFlightService service(&flight_server, MakeAsyncMemoryManager(),
                                    MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});

  // Upload nothing at all: the acknowledgement the server writes back on the
  // client's half-close is what makes Close() return.
  FlightDescriptor descriptor = FlightDescriptor::Path({"empty-put"});
  ASSERT_OK_AND_ASSIGN(auto result, client->DoPut(descriptor, schema));
  ASSERT_OK(result.writer->Close());

  // Cleanup before looking at the listener, so the DoPut callback cannot race
  // with the reads.
  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  // No data of any kind reached the listener.
  ASSERT_EQ(listener->batches().size(), 0);
  ASSERT_EQ(listener->metadata_only_count(), 0);
  ASSERT_OK(listener->last_status());
  // The sync client sends the schema payload eagerly, even when no batch
  // follows, so the listener does see the schema here. Deliberately not
  // asserted: that is the client's business, not the transport's.
}

TEST(AsyncGrpcTest, MetadataOnlyPutChunk) {
  TestFlightServer inner_server;
  auto listener = std::make_shared<RecordingListener>();
  TestServerAsyncAdapter flight_server(&inner_server, listener);
  AsyncGenericFlightService service(&flight_server, MakeAsyncMemoryManager(),
                                    MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});

  // RFC 3.2: a message carrying application metadata and no data is a valid
  // chunk of the upload.
  FlightDescriptor descriptor = FlightDescriptor::Path({"metadata-only-put"});
  ASSERT_OK_AND_ASSIGN(auto result, client->DoPut(descriptor, schema));
  auto app_metadata = arrow::Buffer::FromString("metadata-only-upload");
  ASSERT_OK(result.writer->WriteMetadata(app_metadata));
  ASSERT_OK(result.writer->Close());

  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();

  // Exactly one chunk reached the listener: no data, that exact metadata.
  ASSERT_EQ(listener->batches().size(), 0);
  ASSERT_EQ(listener->metadata_only_count(), 1);
  ASSERT_EQ(listener->metadata_chunks().size(), 1);
  ASSERT_NE(listener->metadata_chunks()[0], nullptr);
  ASSERT_EQ(listener->metadata_chunks()[0]->ToString(), "metadata-only-upload");
}

TEST(AsyncGrpcTest, DoPutRejectedByListener) {
  TestFlightServer inner_server;
  auto listener = std::make_shared<FinishRecordingListener>();
  // The consumer rejects the upload.
  listener->set_next_status(arrow::Status::Invalid("listener rejected this upload"));
  TestServerAsyncAdapter flight_server(&inner_server, listener);
  AsyncGenericFlightService service(&flight_server, MakeAsyncMemoryManager(),
                                    MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  // The rejection must reach the client as the upload's status, whichever of
  // the three surfaces reports it first (observed: the writer's Close()).
  FlightDescriptor descriptor = FlightDescriptor::Path({"rejected-put"});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, schema, batch);
  ASSERT_FALSE(status.ok())
      << "the listener's rejection never reached the client, surfaced on " << surface;
  ASSERT_EQ(status.code(), arrow::StatusCode::Invalid);
  ASSERT_NE(status.message().find("listener rejected this upload"), std::string::npos)
      << "unexpected status: " << status;
  // The rejection must also be what OnFinish reports.
  ASSERT_EQ(listener->finish_status().code(), arrow::StatusCode::Invalid)
      << "OnFinish got: " << listener->finish_status();

  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();
}

TEST(AsyncGrpcTest, DoPutWithoutFactoryIsUnimplemented) {
  TestFlightServer inner_server;
  TestServerAsyncAdapter flight_server(&inner_server);
  // No listener factory: there is nothing to hand an upload to, so DoPut is
  // answered like any other method the service does not serve.
  AsyncGenericFlightService service(&flight_server, MakeAsyncMemoryManager(),
                                    MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  FlightDescriptor descriptor = FlightDescriptor::Path({"factory-less-put"});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, schema, batch);
  ASSERT_FALSE(status.ok())
      << "DoPut without a listener factory must be refused, surfaced on " << surface;
  // The gRPC UNIMPLEMENTED fallback replies with an empty message, so only the
  // code is asserted here (observed: the writer's Close()).
  ASSERT_EQ(status.code(), arrow::StatusCode::NotImplemented);

  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();
}

TEST(AsyncGrpcTest, OtherMethodsAreUnimplemented) {
  TestFlightServer inner_server;
  auto listener = std::make_shared<RecordingListener>();
  TestServerAsyncAdapter flight_server(&inner_server, listener);
  AsyncGenericFlightService service(&flight_server, MakeAsyncMemoryManager(),
                                    MakeAsyncHelper());

  int port = 0;
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(), &port);
  builder.RegisterCallbackGenericService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  ASSERT_NE(port, 0);

  std::string uri = "grpc://localhost:" + std::to_string(port);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));

  // Every typed FlightService method is served now; the fallback reactor is
  // still reachable through a method name no stub knows (which maps to
  // FlightMethod::Invalid in MethodFromName), and that is the only pin left
  // that an unknown method cannot be answered by a typed path.
  const ::grpc::Status status = CallUnknownMethod(port);
  ASSERT_FALSE(status.ok()) << "an unknown method must not be served";
  ASSERT_EQ(::grpc::StatusCode::UNIMPLEMENTED, status.error_code())
      << status.error_message();

  ASSERT_OK(client->Close());
  server->Shutdown();
  server->Wait();
}

// ---------------------------------------------------------------------------
// The unary trio: GetFlightInfo, GetSchema and PollFlightInfo share the
// UnaryReactor (one request in, one response out), so the trio has one
// happy-path test each plus one error-path test that pins the base class.
// ---------------------------------------------------------------------------

TEST(AsyncGrpcTest, GetFlightInfoIsServedAsync) {
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto info,
                       client->GetFlightInfo(FlightDescriptor::Command("ping")));
  ipc::DictionaryMemo dict_memo;
  ASSERT_OK_AND_ASSIGN(auto schema, info->GetSchema(&dict_memo));
  ASSERT_EQ(1, schema->num_fields());
  ASSERT_EQ("value", schema->field(0)->name());

  // Cleanup before looking at the recording: it was written on a gRPC thread.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  ASSERT_EQ("ping", inner_server.last_descriptor().cmd);
}

TEST(AsyncGrpcTest, GetFlightInfoWithNullResponseIsNotFound) {
  // A handler that answers OK without setting the response: the sync transport
  // answers NOT_FOUND with "Flight not found" (grpc_server.cc:282-286), and the
  // async reactor must answer the same, not INTERNAL.
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  const auto status =
      client->GetFlightInfo(FlightDescriptor::Command("null-info")).status();
  ASSERT_RAISES(KeyError, status);
  ASSERT_THAT(status.message(), ::testing::HasSubstr("Flight not found"));

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, GetSchemaIsServedAsync) {
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto schema_result,
                       client->GetSchema(FlightDescriptor::Command("ping")));
  ipc::DictionaryMemo dict_memo;
  ASSERT_OK_AND_ASSIGN(auto schema, schema_result->GetSchema(&dict_memo));
  ASSERT_EQ(1, schema->num_fields());
  ASSERT_EQ("value", schema->field(0)->name());

  // Cleanup before looking at the recording: it was written on a gRPC thread.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  ASSERT_EQ("ping", inner_server.last_descriptor().cmd);
}

TEST(AsyncGrpcTest, PollFlightInfoIsServedAsync) {
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto poll_info,
                       client->PollFlightInfo(FlightDescriptor::Command("ping")));
  ASSERT_TRUE(poll_info->progress.has_value());
  ASSERT_DOUBLE_EQ(0.5, *poll_info->progress);
  ASSERT_TRUE(poll_info->descriptor.has_value());
  ASSERT_EQ("poll-next", poll_info->descriptor->cmd);

  // Cleanup before looking at the recording: it was written on a gRPC thread.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  ASSERT_EQ("ping", inner_server.last_descriptor().cmd);
}

TEST(AsyncGrpcTest, UnaryReactorPropagatesServerError) {
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  // The server's KeyError must reach the client as a status failure carrying
  // the server's message, not as a successful (empty) response. The message is
  // matched by substring because this build defines ARROW_EXTRA_ERROR_CONTEXT,
  // which appends the call-site trail to the server's message.
  auto status = client->GetFlightInfo(FlightDescriptor::Command("fail")).status();
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(arrow::StatusCode::KeyError, status.code());
  ASSERT_THAT(status.message(), ::testing::HasSubstr("no such flight"));

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, GetFlightInfoAsyncCompletesOnAnotherThread) {
  // The handler completes its future from a helper thread after the dispatch
  // has returned: the answer must arrive on the client anyway, which is the
  // whole point of the async handler.
  class DeferredInfoServer : public AsyncGenericFlightServerBase {
   public:
    arrow::Future<std::shared_ptr<FlightInfo>> GetFlightInfoAsync(
        const ServerCallContext&, const FlightDescriptor& request) override {
      auto out = arrow::Future<std::shared_ptr<FlightInfo>>::Make();
      std::thread([out, request]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto info =
            FlightInfo::Make(*arrow::schema({arrow::field("value", arrow::int64())}),
                             request, std::vector<FlightEndpoint>{}, -1, -1);
        if (!info.ok()) {
          out.MarkFinished(info.status());
          return;
        }
        out.MarkFinished(std::make_shared<FlightInfo>(std::move(*info)));
      }).detach();
      return out;
    }
  };

  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  DeferredInfoServer flight_server;
  FlightServerOptions options(location);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto info,
                       client->GetFlightInfo(FlightDescriptor::Command("deferred")));
  ipc::DictionaryMemo dict_memo;
  ASSERT_OK_AND_ASSIGN(auto schema, info->GetSchema(&dict_memo));
  ASSERT_EQ(1, schema->num_fields());
  ASSERT_EQ("value", schema->field(0)->name());

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

// ---------------------------------------------------------------------------
// The streaming trio: ListActions (a vector), DoAction (a ResultStream) and
// ListFlights (a FlightListing) share StreamingReactor (one request in, N
// responses out, driven by OnWriteDone -> WriteNext).  Each RPC has a
// happy-path test; the reactor never writing a message must still finish the
// RPC (the empty tests, run under a shell timeout); the two iterator-backed
// ones pin the mid-stream error rule.
// ---------------------------------------------------------------------------

TEST(AsyncGrpcTest, ListActionsIsServedAsync) {
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto actions, client->ListActions());
  ASSERT_EQ(actions.size(), 2);
  ASSERT_EQ(actions[0].type, "echo");
  ASSERT_EQ(actions[0].description, "echo back");
  ASSERT_EQ(actions[1].type, "fail");
  ASSERT_EQ(actions[1].description, "always fails");

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, ListActionsEmptyFinishesCleanly) {
  // Zero response messages: WriteNext() finds the vector exhausted before it
  // ever calls StartWrite(), so the RPC must be finished with OK rather than
  // left waiting for a write completion that will never arrive.
  TestFlightServer inner_server;
  inner_server.set_empty_actions();
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto actions, client->ListActions());
  ASSERT_TRUE(actions.empty());

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, DoActionIsServedAsync) {
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  const Action action{"ping", Buffer::FromString("request-body")};
  ASSERT_OK_AND_ASSIGN(auto results, client->DoAction(action));
  ASSERT_OK_AND_ASSIGN(auto result, results->Next());
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->body->ToString(), "one");
  // The stream ends with a null Result, not with an error.
  ASSERT_OK_AND_ASSIGN(auto end, results->Next());
  ASSERT_EQ(end, nullptr);

  // Cleanup before looking at the recording: it was written on a gRPC thread.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  ASSERT_EQ("ping", inner_server.last_action_type());
}

TEST(AsyncGrpcTest, DoActionStreamsMultipleResults) {
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  const Action action{"multi", Buffer::FromString("")};
  ASSERT_OK_AND_ASSIGN(auto results, client->DoAction(action));

  std::vector<std::string> bodies;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto result, results->Next());
    if (result == nullptr) break;
    bodies.push_back(result->body->ToString());
  }
  ASSERT_EQ(bodies, (std::vector<std::string>{"one", "two", "three"}));

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, DoActionMidStreamErrorFinishesTheRpc) {
  // The stream yields one Result and then fails.  The error must become the
  // RPC status -- the client must not see a truncated success.
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  const Action action{"fail-mid-stream", Buffer::FromString("")};
  ASSERT_OK_AND_ASSIGN(auto results, client->DoAction(action));
  ASSERT_OK_AND_ASSIGN(auto first, results->Next());
  ASSERT_NE(first, nullptr);
  ASSERT_EQ(first->body->ToString(), "first");

  auto second = results->Next();
  ASSERT_FALSE(second.ok()) << "the mid-stream error must reach the client";
  ASSERT_EQ(second.status().code(), arrow::StatusCode::IOError);
  ASSERT_THAT(second.status().message(),
              ::testing::HasSubstr("mid-stream action failure"));

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, DoActionWithNullResultStreamIsCancelled) {
  // A handler that answers OK without a ResultStream answers CANCELLED on the
  // sync transport (grpc_server.cc:404-406), before any message is written; the
  // async reactor must answer the same.  The client may see it on DoAction() or
  // on the first Next(), so both surfaces are accepted.
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  const Action action{"null-result-stream", Buffer::FromString("")};
  auto results = client->DoAction(action);
  if (results.ok()) {
    // The stream was handed out before the status arrived: CANCELLED is then
    // reported by the first read.
    ASSERT_RAISES(Cancelled, (*results)->Next());
  } else {
    ASSERT_RAISES(Cancelled, results.status());
  }

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, ListFlightsIsServedAsync) {
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  const Criteria criteria("filter-me");
  ASSERT_OK_AND_ASSIGN(auto listing, client->ListFlights({}, criteria));

  std::vector<std::vector<std::string>> paths;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto info, listing->Next());
    if (info == nullptr) break;
    paths.push_back(info->descriptor().path);
  }
  ASSERT_EQ(paths, (std::vector<std::vector<std::string>>{{"first-listing"},
                                                          {"second-listing"}}));

  // Cleanup before looking at the recording: it was written on a gRPC thread.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  ASSERT_EQ("filter-me", inner_server.last_criteria().expression);
}

TEST(AsyncGrpcTest, ListFlightsEmptyFinishesCleanly) {
  // A listing with no flights: the server answers OK with zero messages, and
  // the client sees an exhausted listing rather than a hang.
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto listing, client->ListFlights({}, Criteria("")));
  ASSERT_OK_AND_ASSIGN(auto info, listing->Next());
  ASSERT_EQ(info, nullptr);
  // The client sent no criteria; the server must still have received the
  // (empty) Criteria, exactly as the sync handler does.
  ASSERT_EQ("", inner_server.last_criteria().expression);

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, ListFlightsMidStreamErrorFinishesTheRpc) {
  // A listing that fails on its second Next(): the error must become the RPC
  // status, not a silently truncated success.
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto listing = client->ListFlights({}, Criteria("fail-mid-stream"));
  ASSERT_FALSE(listing.ok()) << "the mid-stream error must reach the client";
  ASSERT_EQ(listing.status().code(), arrow::StatusCode::IOError);
  ASSERT_THAT(listing.status().message(),
              ::testing::HasSubstr("mid-stream listing failure"));

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, ListFlightsAsyncCompletesOnAnotherThread) {
  // The listing object itself arrives on a helper thread: the transport must
  // serve the response from wherever the future resolves.
  class DeferredListingServer : public AsyncGenericFlightServerBase {
   public:
    arrow::Future<std::shared_ptr<AsyncFlightListing>> ListFlightsAsync(
        const ServerCallContext&, const Criteria*) override {
      auto out = arrow::Future<std::shared_ptr<AsyncFlightListing>>::Make();
      std::thread([out]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::vector<FlightInfo> flights;
        auto info = MakeTestFlightInfo(FlightDescriptor::Path({"deferred"}));
        if (!info.ok()) {
          out.MarkFinished(info.status());
          return;
        }
        flights.push_back(std::move(**info));
        out.MarkFinished(std::make_shared<SimpleListingAdapter>(std::move(flights)));
      }).detach();
      return out;
    }
  };

  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  DeferredListingServer flight_server;
  FlightServerOptions options(location);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto listing, client->ListFlights());
  int seen = 0;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto info, listing->Next());
    if (info == nullptr) break;
    ++seen;
  }
  ASSERT_EQ(1, seen);

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

// ---------------------------------------------------------------------------
// DoExchange: the bidirectional RPC.  Its handler is synchronous user code
// running on the callback thread, with the sync reader/writer stack over the
// reactor's blocking ServerDataStream (see ExchangeReactor).
// ---------------------------------------------------------------------------

TEST(AsyncGrpcTest, DoExchangeEchoesBatches) {
  // Two batches up, two batches back, in order: this is the case that needs a
  // read and a write outstanding at the same time, which is what the bridge
  // exists for.
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  EchoExchangeTestServer flight_server;
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto exchange,
                       client->DoExchange(FlightDescriptor::Command("echo")));

  auto schema = arrow::schema({arrow::field("a", arrow::int64())});
  std::vector<std::shared_ptr<RecordBatch>> sent = {
      arrow::RecordBatch::Make(schema, 2,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2]")}),
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[3, 4, 5]")})};
  ASSERT_OK(exchange.writer->Begin(schema));
  for (const auto& batch : sent) {
    ASSERT_OK(exchange.writer->WriteRecordBatch(*batch));
  }
  ASSERT_OK(exchange.writer->DoneWriting());

  ASSERT_OK_AND_ASSIGN(auto server_schema, exchange.reader->GetSchema());
  AssertSchemaEqual(*schema, *server_schema);
  for (const auto& batch : sent) {
    ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
    ASSERT_NE(nullptr, chunk.data);
    ASSERT_BATCHES_EQUAL(*batch, *chunk.data);
  }
  ASSERT_OK_AND_ASSIGN(auto end, exchange.reader->Next());
  ASSERT_EQ(nullptr, end.data);
  ASSERT_OK(exchange.writer->Close());

  // Cleanup before looking at the recording: it was written on a gRPC thread.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  ASSERT_EQ(1, flight_server.exchange_count());
  ASSERT_EQ(0, flight_server.exchanges_in_flight());
}

TEST(AsyncGrpcTest, DoExchangeMetadataOnlyDoesNotHang) {
  // A client that writes metadata and nothing else, then reads: the server's
  // handler must see the end of the exchange and reply, not wait for a schema
  // message that will never come (the case DataTest::TestDoExchangeNoData
  // exists for).  A hang here is the failure mode this test exists for, which
  // is why it runs under a shell timeout.
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  EchoExchangeTestServer flight_server;
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto exchange,
                       client->DoExchange(FlightDescriptor::Command("echo")));

  ASSERT_OK(exchange.writer->WriteMetadata(arrow::Buffer::FromString("only-metadata")));
  ASSERT_OK(exchange.writer->DoneWriting());

  // The echo server reflects the metadata chunk back with no data.
  ASSERT_OK_AND_ASSIGN(auto chunk, exchange.reader->Next());
  ASSERT_EQ(nullptr, chunk.data);
  ASSERT_NE(nullptr, chunk.app_metadata);
  ASSERT_EQ("only-metadata", chunk.app_metadata->ToString());
  ASSERT_OK_AND_ASSIGN(auto end, exchange.reader->Next());
  ASSERT_EQ(nullptr, end.data);
  ASSERT_OK(exchange.writer->Close());

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  ASSERT_EQ(1, flight_server.exchange_count());
  ASSERT_EQ(0, flight_server.exchanges_in_flight());
}

TEST(AsyncGrpcTest, DoExchangeClientDisconnectFinishes) {
  // A client that vanishes mid-exchange: the server's read fails, the reactor
  // must let the handler unwind and finish the RPC, and the exchange counter
  // added to the test server must come back to zero -- a hang or a leaked
  // handler leaves it above zero.
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  EchoExchangeTestServer flight_server;
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(
      auto client_location,
      Location::Parse("grpc://localhost:" + std::to_string(flight_server.port())));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  ASSERT_OK_AND_ASSIGN(auto exchange,
                       client->DoExchange(FlightDescriptor::Command("echo")));

  auto schema = arrow::schema({arrow::field("a", arrow::int64())});
  ASSERT_OK(exchange.writer->Begin(schema));
  ASSERT_OK(exchange.writer->WriteRecordBatch(*arrow::RecordBatch::Make(
      schema, 2, {arrow::ArrayFromJSON(arrow::int64(), "[1, 2]")})));
  // The server has entered its handler by now (it answered the schema); the
  // exchange is live.
  ASSERT_TRUE(WaitFor([&flight_server] { return flight_server.exchange_count() == 1; }));

  // Walk away without finishing: no DoneWriting, no Close, no reader drain.
  // The client's destructor closes the channel, which cancels the RPC.
  exchange.reader->Cancel();
  exchange.writer.reset();
  exchange.reader.reset();
  ASSERT_OK(client->Close());

  // The handler must leave, whichever status path it takes.
  const bool drained =
      WaitFor([&flight_server] { return flight_server.exchanges_in_flight() == 0; });
  ASSERT_TRUE(drained) << "the exchange handler never left (in flight: "
                       << flight_server.exchanges_in_flight() << ")";

  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlag) {
  ASSERT_OK_AND_ASSIGN(auto default_location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions default_options(default_location);

  // Same adapter as the tests above, but now started through the async
  // server's own Init instead of a hand-built grpc::ServerBuilder.
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_GT(flight_server.port(), 0);

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  // (i) DoGet went through the generic reactor: the ticket reached
  // FlightServerBase::DoGet and the whole stream came back.
  Ticket ticket{"flag-path-do-get-ticket"};
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(ticket));
  ASSERT_OK_AND_ASSIGN(auto schema, reader->GetSchema());
  ASSERT_EQ(schema->num_fields(), 2);
  ASSERT_EQ(schema->field(0)->name(), "a");
  ASSERT_EQ(schema->field(1)->name(), "b");
  int batch_count = 0;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
    if (!chunk.data) break;
    ASSERT_EQ(chunk.data->num_rows(), 5);
    ASSERT_OK_AND_ASSIGN(auto value, chunk.data->column(0)->GetScalar(0));
    ASSERT_TRUE(value->Equals(*arrow::MakeScalar(int64_t(1))));
    batch_count++;
  }
  ASSERT_EQ(batch_count, 3);

  // (ii) Every typed method is served on this path too; the fallback reactor
  // still answers a method name no stub knows, which is what pins "the typed
  // sync service is NOT registered alongside the generic one".
  const ::grpc::Status unknown_status = CallUnknownMethod(flight_server.port());
  ASSERT_FALSE(unknown_status.ok()) << "an unknown method must not be served";
  ASSERT_EQ(::grpc::StatusCode::UNIMPLEMENTED, unknown_status.error_code())
      << unknown_status.error_message();

  // (iii) The async server takes no listener factory here, so uploads are
  // refused the same way.
  FlightDescriptor descriptor = FlightDescriptor::Path({"not-served"});
  auto put_schema = arrow::schema({arrow::field("a", arrow::int64())});
  auto batch = arrow::RecordBatch::Make(put_schema, 1,
                                        {arrow::ArrayFromJSON(arrow::int64(), "[1]")});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, put_schema, batch);
  ASSERT_FALSE(status.ok()) << "DoPut must be refused, surfaced on " << surface;
  ASSERT_EQ(status.code(), arrow::StatusCode::NotImplemented);

  // Cleanup. The ticket is asserted only after the server stopped, so the
  // DoGet callback cannot race with the read.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  ASSERT_EQ(inner_server.ticket(), "flag-path-do-get-ticket");
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagServesDoPut) {
  // The async server serves uploads because the server class hands out the
  // per-RPC listener through CreateDoPutListener(), instead of the RPC falling
  // through to UNIMPLEMENTED.
  TestFlightServer inner_server;
  auto listener = std::make_shared<RecordingListener>();
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server, listener);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_GT(flight_server.port(), 0);

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  const FlightDescriptor descriptor = FlightDescriptor::Path({"wave", "h2"});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, schema, batch);
  ASSERT_TRUE(status.ok()) << "upload failed on " << surface << ": " << status;

  // Cleanup before looking at the listener: its callbacks run on a gRPC
  // thread, and nothing may read the recordings while that thread writes.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  // The listener saw the descriptor the client sent, the schema, and the batch.
  ASSERT_EQ(listener->descriptor_count(), 1);
  ASSERT_EQ(listener->descriptors()[0], descriptor);
  ASSERT_EQ(listener->schema_count(), 1);
  ASSERT_EQ(listener->schema()->num_fields(), 2);
  ASSERT_EQ(listener->schema()->field(0)->name(), "a");
  ASSERT_EQ(listener->schema()->field(1)->name(), "b");
  ASSERT_EQ(listener->batches().size(), 1);
  const auto& chunk = listener->batches()[0];
  ASSERT_EQ(chunk->num_rows(), 3);
  ASSERT_EQ(chunk->num_columns(), 2);
  ASSERT_OK_AND_ASSIGN(auto value, chunk->column(0)->GetScalar(0));
  ASSERT_TRUE(value->Equals(*arrow::MakeScalar(int64_t(1))));
}

TEST(AsyncGrpcTest, CreateDoPutListenerServesUpload) {
  // The upload half of the server class API: CreateDoPutListener() hands out
  // the per-RPC listener, with no options-level factory.
  TestUploadServerAsyncAdapter flight_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_GT(flight_server.port(), 0);

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  const FlightDescriptor descriptor = FlightDescriptor::Path({"virtual", "h2"});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, schema, batch);
  ASSERT_TRUE(status.ok()) << "upload failed on " << surface << ": " << status;

  // Cleanup before reading the recordings: the listener's callbacks run on a
  // gRPC thread.
  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  // The virtual was consulted exactly once, and its listener saw the upload.
  ASSERT_EQ(flight_server.calls(), 1);
  const auto& listener = flight_server.listener(0);
  ASSERT_EQ(listener->descriptor_count(), 1);
  ASSERT_EQ(listener->descriptors()[0], descriptor);
  ASSERT_EQ(listener->schema_count(), 1);
  ASSERT_EQ(listener->batches().size(), 1);
  ASSERT_EQ(listener->batches()[0]->num_rows(), 3);
}

TEST(AsyncGrpcTest, DoPutWithoutListenerIsUnimplemented) {
  // A server class that does not hand out a listener (CreateDoPutListener()
  // returns nullptr, the base default) does not accept uploads: the RPC is
  // answered UNIMPLEMENTED.  There is no options-level fallback.
  TestFlightServer inner_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  ASSERT_OK(flight_server.Init(options));

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto schema = arrow::schema({arrow::field("a", arrow::int64())});
  auto batch = arrow::RecordBatch::Make(schema, 2,
                                        {arrow::ArrayFromJSON(arrow::int64(), "[7, 8]")});
  auto [surface, status] = UploadOneBatch(
      client.get(), FlightDescriptor::Path({"no-listener"}), schema, batch);
  ASSERT_FALSE(status.ok()) << "the upload succeeded on " << surface
                            << " with no listener to serve it";
  ASSERT_EQ(status.code(), arrow::StatusCode::NotImplemented) << status;

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, DoPutListenerSeesCleanFinish) {
  // A client that ends the upload normally (DoneWriting + Close) makes the
  // listener's OnFinish fire once, with OK: that is how a consumer knows the
  // upload ended and it can commit what it accumulated.
  TestUploadServerAsyncAdapter flight_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  ASSERT_OK(flight_server.Init(options));

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto schema = arrow::schema({arrow::field("a", arrow::int64())});
  auto batch = arrow::RecordBatch::Make(schema, 2,
                                        {arrow::ArrayFromJSON(arrow::int64(), "[1, 2]")});
  auto [surface, status] =
      UploadOneBatch(client.get(), FlightDescriptor::Path({"finish"}), schema, batch);
  ASSERT_TRUE(status.ok()) << "upload failed on " << surface << ": " << status;

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  // OnFinish ran exactly once, and said the upload ended cleanly.
  ASSERT_EQ(flight_server.listener(0)->finish_count(), 1);
  ASSERT_TRUE(flight_server.listener(0)->finish_status().ok())
      << "unexpected finish status: " << flight_server.listener(0)->finish_status();
}

TEST(AsyncGrpcTest, DoPutCancelFromServerSideFinishesTheUpload) {
  // The server aborts the upload without waiting for the client: Cancel()
  // finishes the RPC with the given status, the client's write or Close
  // reports it, and the listener hears it as its OnFinish status.
  TestUploadServerAsyncAdapter flight_server;
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  ASSERT_OK(flight_server.Init(options));

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto schema = arrow::schema({arrow::field("a", arrow::int64())});
  auto batch = arrow::RecordBatch::Make(schema, 2,
                                        {arrow::ArrayFromJSON(arrow::int64(), "[3, 4]")});

  // Open the upload and write one batch, but do not end it.
  auto put = client->DoPut(FlightDescriptor::Path({"cancel"}), schema);
  ASSERT_TRUE(put.ok()) << put.status();
  auto& writer = *put->writer;
  ASSERT_OK(writer.WriteRecordBatch(*batch));

  // Wait for the listener the virtual handed out, then cancel it from this
  // (non-gRPC) thread: that is the mid-upload server-side abort.
  auto listener = flight_server.WaitForListener(0);
  ASSERT_NE(listener, nullptr) << "the server never handed out a DoPut listener";
  listener->CancelWith(arrow::Status::Cancelled("server said stop"));

  // The client sees the cancellation on the surface that observes the RPC.
  const Status close_status = writer.Close();
  ASSERT_FALSE(close_status.ok()) << "the server-side cancel never reached the client";
  ASSERT_EQ(close_status.code(), arrow::StatusCode::Cancelled);
  ASSERT_NE(close_status.message().find("server said stop"), std::string::npos)
      << "unexpected status: " << close_status;

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  // Cancel() reached the transport, and the listener was told the upload ended
  // with that status exactly once.
  ASSERT_EQ(listener->cancel_count(), 1);
  ASSERT_OK(listener->cancel_status()) << listener->cancel_status();
  ASSERT_EQ(listener->finish_count(), 1);
  // OnFinish must carry the ending that caused it, not a generic status.
  ASSERT_EQ(listener->finish_status().code(), arrow::StatusCode::Cancelled)
      << "OnFinish got: " << listener->finish_status();

  // After the upload, Cancel() has nothing to cancel.
  listener->CancelWith(arrow::Status::Cancelled("too late"));
  ASSERT_EQ(listener->cancel_status().code(), arrow::StatusCode::Invalid);
}

TEST(AsyncGrpcTest, DoPutCancelAfterUploadIsInvalid) {
  // Cancel() is only meaningful while the upload's RPC is running: once it
  // finished, the listener says so instead of reaching a dead RPC.
  auto listener = std::make_shared<FinishRecordingListener>();
  listener->CancelWith(arrow::Status::Cancelled("nothing in flight"));
  ASSERT_EQ(listener->cancel_status().code(), arrow::StatusCode::Invalid);
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagUploadRejectedOnDescriptor) {
  // A handler that rejects the descriptor rejects the whole upload through
  // the same surface as an OnNext rejection: the writer's status.
  TestFlightServer inner_server;
  auto listener = std::make_shared<RecordingListener>();
  listener->set_descriptor_status(
      arrow::Status::Invalid("listener rejected this descriptor"));
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server, listener);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_GT(flight_server.port(), 0);

  std::string uri = "grpc://localhost:" + std::to_string(flight_server.port());
  ASSERT_OK_AND_ASSIGN(auto client_location, Location::Parse(uri));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto schema = arrow::schema(
      {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
  auto batch =
      arrow::RecordBatch::Make(schema, 3,
                               {arrow::ArrayFromJSON(arrow::int64(), "[1, 2, 3]"),
                                arrow::ArrayFromJSON(arrow::int64(), "[10, 20, 30]")});

  const FlightDescriptor descriptor = FlightDescriptor::Path({"rejected-h2"});
  auto [surface, status] = UploadOneBatch(client.get(), descriptor, schema, batch);
  ASSERT_FALSE(status.ok())
      << "the descriptor rejection never reached the client, surfaced on " << surface;
  ASSERT_EQ(status.code(), arrow::StatusCode::Invalid);
  ASSERT_NE(status.message().find("listener rejected this descriptor"), std::string::npos)
      << "unexpected status: " << status;

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  // The upload was rejected on its descriptor: the handler saw it, and no
  // schema or data followed it.
  ASSERT_EQ(listener->descriptor_count(), 1);
  ASSERT_EQ(listener->descriptors()[0], descriptor);
  ASSERT_EQ(listener->schema_count(), 0);
  ASSERT_EQ(listener->batches().size(), 0);
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagRefusesBlockingAuthHandler) {
  // The callback path cannot run the blocking ServerAuthHandler (it would run
  // on callback threads), so a server configured with one must refuse to start
  // instead of silently serving unauthenticated requests.  Middleware *is*
  // supported now - see UseAsyncGrpcFlagAllowsMiddleware.
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));

  TestFlightServer inner_server;
  FlightServerOptions with_auth(location);
  TestServerAsyncAdapter auth_server(&inner_server);
  with_auth.auth_handler = std::make_shared<TestServerAuthHandler>("user", "pass");
  auto auth_status = auth_server.Init(with_auth);
  ASSERT_FALSE(auth_status.ok()) << "an auth handler must not be silently skipped";
  ASSERT_EQ(auth_status.code(), arrow::StatusCode::NotImplemented);
  ASSERT_NE(auth_status.message().find("AsyncGenericFlightServerBase"), std::string::npos)
      << auth_status;
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagAllowsMiddleware) {
  // Middleware configured through FlightServerOptions must run on the async
  // path: StartCall at call creation, SendingHeaders and CallCompleted at the
  // end - success or failure alike.
  auto trace = std::make_shared<MiddlewareTrace>();
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));

  TestFlightServer inner_server;
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  options.middleware = {
      {"recording", std::make_shared<RecordingServerMiddlewareFactory>(trace)}};
  ASSERT_OK(flight_server.Init(options));

  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", flight_server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(Ticket{"middleware-ticket"}));
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
    if (!chunk.data) break;
  }

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());

  std::lock_guard<std::mutex> guard(trace->mutex);
  EXPECT_EQ(1, trace->start_call);
  EXPECT_EQ(1, trace->sending_headers);
  EXPECT_EQ(1, trace->call_completed);
}

TEST(AsyncGrpcTest, UseAsyncGrpcFlagMiddlewareCanRejectCall) {
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  TestFlightServer inner_server;
  FlightServerOptions options(location);
  TestServerAsyncAdapter flight_server(&inner_server);
  options.middleware = {
      {"rejecting", std::make_shared<RejectingServerMiddlewareFactory>()}};
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", flight_server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto status = client->DoGet(Ticket{"rejected"}).status();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("rejected by middleware"));

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

// ---------------------------------------------------------------------------
// Handshake + per-RPC token auth on the async generic service.
// ---------------------------------------------------------------------------

// Counters shared between the server callbacks and the test thread.
struct AsyncAuthState {
  std::mutex mutex;
  int handshakes = 0;
  int validations = 0;
  std::string peer_identity;
};

// Async-transport twin of TestServerAuthHandler: the handshake answers the
// client's password with the username; later RPCs must carry that password as
// the token, which is what TestClientAuthHandler::GetToken() puts in the
// `auth-token-bin` header (the sync twin validates it the same way in
// TestServerAuthHandler::IsValid).  It is a server class: the two hooks below
// are the only configuration the transport needs.
class AsyncAuthTestServer : public AsyncGenericFlightServerBase {
 public:
  AsyncAuthTestServer(std::string username, std::string password, std::string ticket,
                      std::shared_ptr<AsyncAuthState> state = nullptr)
      : username_(std::move(username)),
        password_(std::move(password)),
        ticket_(std::move(ticket)),
        state_(state ? std::move(state) : std::make_shared<AsyncAuthState>()) {}

  const std::shared_ptr<AsyncAuthState>& state() const { return state_; }

  arrow::Future<std::shared_ptr<AsyncFlightDataStream>> DoGetAsync(
      const ServerCallContext& context, const Ticket& request) override {
    {
      std::lock_guard<std::mutex> guard(state_->mutex);
      state_->peer_identity = context.peer_identity();
    }
    if (request.ticket != ticket_) {
      return arrow::Future<std::shared_ptr<AsyncFlightDataStream>>::MakeFinished(
          Status::KeyError("No such ticket: ", request.ticket));
    }
    auto schema = arrow::schema({arrow::field("a", arrow::int64())});
    auto batch = arrow::RecordBatch::Make(
        schema, 2, {arrow::ArrayFromJSON(arrow::int64(), "[1, 2]")});
    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make({batch}));
    return arrow::Future<std::shared_ptr<AsyncFlightDataStream>>::MakeFinished(
        std::make_unique<ResolvedStream>(
            std::make_unique<arrow::flight::RecordBatchStream>(std::move(reader))));
  }

  // The hook: the transport read the client's password; answer with the
  // username, or reject the RPC the way the sync twin does.
  Status Handshake(const ServerCallContext& /*context*/, const std::string& password,
                   std::string* response) override {
    {
      std::lock_guard<std::mutex> guard(state_->mutex);
      state_->handshakes++;
    }
    if (password != password_) {
      return MakeFlightError(FlightStatusCode::Unauthenticated, "Invalid token");
    }
    *response = username_;
    return Status::OK();
  }

  Status ValidateToken(const ServerCallContext& /*context*/, const std::string& token,
                       std::string* peer_identity) override {
    std::lock_guard<std::mutex> guard(state_->mutex);
    state_->validations++;
    if (token != password_) {
      return MakeFlightError(FlightStatusCode::Unauthenticated, "Invalid token");
    }
    *peer_identity = username_;
    return Status::OK();
  }

 private:
  std::string username_;
  std::string password_;
  std::string ticket_;
  std::shared_ptr<AsyncAuthState> state_;
};

// A server with the async auth hooks installed, and a client connected to it.
// `status` carries the setup result; construction never fails a test.
struct AuthenticatedHarness {
  AsyncAuthTestServer server{"user", "p4ssw0rd", "auth-ticket"};
  std::unique_ptr<FlightClient> client;
  Status status;

  AuthenticatedHarness() : status(Start()) {}

  ~AuthenticatedHarness() {
    if (client) {
      ARROW_WARN_NOT_OK(client->Close(), "Close()");
    }
    ARROW_WARN_NOT_OK(server.Shutdown(), "Shutdown()");
  }

 private:
  Status Start() {
    ARROW_ASSIGN_OR_RAISE(auto location, Location::Parse("grpc://localhost:0"));
    // The server class alone selects the async transport: deriving from it and
    // overriding its virtuals are the whole configuration.
    FlightServerOptions options(location);
    RETURN_NOT_OK(server.Init(options));
    ARROW_ASSIGN_OR_RAISE(auto client_location,
                          Location::ForScheme("grpc", "127.0.0.1", server.port()));
    ARROW_ASSIGN_OR_RAISE(client, FlightClient::Connect(client_location));
    return Status::OK();
  }
};

// A minimal server that mixes in the async hooks: the server class alone
// selects the async generic transport.
class AsyncGenericTestServer : public AsyncGenericFlightServerBase {};

TEST(AsyncGrpcTest, HandshakeWithoutHookIsUnimplemented) {
  // A bare async server: Handshake is answered by the class's default virtual
  // with UNIMPLEMENTED, and the client's Authenticate() surfaces it.
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  AsyncGenericTestServer flight_server;
  FlightServerOptions options(location);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", flight_server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto status = client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd"));
  ASSERT_FALSE(status.ok()) << "handshake must not succeed without a hook";
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("authentication mechanism"));

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, AsyncGenericFlightServerBaseImpliesAsyncTransport) {
  // Deriving from the async class is enough: no hooks and no option to set,
  // yet the RPC is served by the generic callback service.  DoPut is
  // the discriminator: FlightServerBase's default (the sync service) answers
  // "NYI", the async service with no listener factory answers with its own
  // message.
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  AsyncGenericTestServer flight_server;
  FlightServerOptions options(location);
  ASSERT_OK(flight_server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", flight_server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  // DoPut is the discriminator: FlightServerBase's default (the sync service)
  // answers "NYI", the async service with no listener factory answers with its
  // own message.  The refusal can surface on either call: FlightClient::DoPut
  // writes the schema payload eagerly, so when the server's UNIMPLEMENTED
  // arrives first the DoPut call itself fails; otherwise the writer's Close()
  // reports it.
  auto put = client->DoPut(FlightDescriptor::Path({"x"}), arrow::schema({}));
  auto status = put.status();
  if (status.ok()) {
    status = put->writer->Close();
  }
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("no listener available"))
      << "the async generic service must have served this call, not the sync path";

  ASSERT_OK(client->Close());
  ASSERT_OK(flight_server.Shutdown());
  ASSERT_OK(flight_server.Wait());
}

TEST(AsyncGrpcTest, HandshakeAuthenticatesAndTokenIsRequired) {
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);

  // 1. Without a handshake the data call is rejected before it is served.
  ASSERT_OK_AND_ASSIGN(auto unauthenticated_location,
                       Location::ForScheme("grpc", "127.0.0.1", harness.server.port()));
  ASSERT_OK_AND_ASSIGN(auto unauthenticated_client,
                       FlightClient::Connect(unauthenticated_location));
  auto status = unauthenticated_client->DoGet(Ticket{"auth-ticket"}).status();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Invalid token"));
  ASSERT_OK(unauthenticated_client->Close());

  // 2. After the handshake the same call succeeds and the server sees the
  //    validated identity.
  ASSERT_OK(harness.client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
  ASSERT_OK_AND_ASSIGN(auto reader, harness.client->DoGet(Ticket{"auth-ticket"}));
  int batches = 0;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
    if (!chunk.data) break;
    batches++;
  }
  EXPECT_EQ(1, batches);
  {
    std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
    EXPECT_EQ(1, harness.server.state()->handshakes);
    EXPECT_EQ("user", harness.server.state()->peer_identity);
    EXPECT_GE(harness.server.state()->validations, 1);
  }
}

TEST(AsyncGrpcTest, BadPasswordIsRejectedAtHandshake) {
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  auto status = harness.client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "wrong"));
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Invalid token"));
}

TEST(AsyncGrpcTest, UnauthenticatedUnknownMethodIsRejectedBeforeUnimplemented) {
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  // GetFlightInfo is not served by the async generic service, but auth runs
  // first: the client must see the auth failure, not UNIMPLEMENTED.
  auto status = harness.client->GetFlightInfo(FlightDescriptor::Path({"x"})).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Invalid token"));
  EXPECT_EQ(arrow::StatusCode::IOError, status.code());
}

// ---------------------------------------------------------------------------
// The synchronous request/response Handshake hook: payload fidelity, edge
// cases, and reactor cleanup (the reactor finishes on the strength of "one
// terminal path per direction", so the concurrency test is its real check).
// ---------------------------------------------------------------------------

// Send `password`, require the server to echo `expect` back.
class EchoClientAuthHandler : public ClientAuthHandler {
 public:
  EchoClientAuthHandler(std::string password, std::string expect)
      : password_(std::move(password)), expect_(std::move(expect)) {}

  Status Authenticate(ClientAuthSender* outgoing, ClientAuthReader* incoming) override {
    ARROW_RETURN_NOT_OK(outgoing->Write(password_));
    ARROW_RETURN_NOT_OK(incoming->Read(&response_));
    if (response_ != expect_) {
      return MakeFlightError(FlightStatusCode::Unauthenticated,
                             "unexpected handshake response");
    }
    return Status::OK();
  }

  Status GetToken(std::string* token) override {
    *token = password_;
    return Status::OK();
  }

  const std::string& response() const { return response_; }

 private:
  std::string password_;
  std::string expect_;
  std::string response_;
};

TEST(AsyncGrpcTest, HandshakePayloadIsBinarySafe) {
  // The hook's request is a std::string taken from the protobuf and its
  // response is echoed back through the client's reader: an embedded NUL and
  // non-ASCII bytes must survive both directions unchanged.
  const std::string binary("p4ss\0w0rd\xC3\xA9", 11);
  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  AsyncAuthTestServer server(binary, binary, "bin-ticket");
  FlightServerOptions options(location);
  ASSERT_OK(server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto handler = std::make_unique<EchoClientAuthHandler>(binary, binary);
  const auto* observed = handler.get();
  ASSERT_OK(client->Authenticate({}, std::move(handler)));
  EXPECT_EQ(binary, observed->response());

  // The token is the password, so an authenticated call must also work.
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(Ticket{"bin-ticket"}));
  ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
  EXPECT_NE(nullptr, chunk.data);

  ASSERT_OK(client->Close());
  ASSERT_OK(server.Shutdown());
  ASSERT_OK(server.Wait());
}

TEST(AsyncGrpcTest, EmptyHandshakeRequestIsRejectedNotHung) {
  // An empty HandshakeRequest is a valid protobuf message: the read succeeds,
  // the hook sees an empty string, and its rejection must reach the client.
  class SilentClientAuthHandler : public ClientAuthHandler {
   public:
    Status Authenticate(ClientAuthSender* outgoing, ClientAuthReader*) override {
      return outgoing->Write("");
    }
    Status GetToken(std::string*) override { return Status::OK(); }
  };

  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  auto status =
      harness.client->Authenticate({}, std::make_unique<SilentClientAuthHandler>());
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Invalid token"));
  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(1, harness.server.state()->handshakes)
      << "the hook must run on an empty request, not fail the read";
}

TEST(AsyncGrpcTest, AbandonedHandshakeLeavesServerUsable) {
  // A client that never sends a message: the server's read completes with
  // ok=false, the reactor must still finish, and the server must keep serving.
  class AbandoningClientAuthHandler : public ClientAuthHandler {
   public:
    Status Authenticate(ClientAuthSender*, ClientAuthReader*) override {
      return Status::Invalid("client never speaks");
    }
    Status GetToken(std::string*) override { return Status::OK(); }
  };

  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  EXPECT_FALSE(
      harness.client->Authenticate({}, std::make_unique<AbandoningClientAuthHandler>())
          .ok());

  ASSERT_OK_AND_ASSIGN(auto location,
                       Location::ForScheme("grpc", "127.0.0.1", harness.server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));
  ASSERT_OK(client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
  ASSERT_OK_AND_ASSIGN(auto reader, client->DoGet(Ticket{"auth-ticket"}));
  ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
  EXPECT_NE(nullptr, chunk.data);
  ASSERT_OK(client->Close());

  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(1, harness.server.state()->handshakes)
      << "the abandoned call must never have reached the hook";
}

TEST(AsyncGrpcTest, ConcurrentHandshakesAllComplete) {
  constexpr int kClients = 32;
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);

  std::vector<std::thread> threads;
  std::vector<Status> statuses(kClients);
  for (int i = 0; i < kClients; i++) {
    threads.emplace_back([&harness, &statuses, i] {
      auto location = Location::ForScheme("grpc", "127.0.0.1", harness.server.port());
      if (!location.ok()) {
        statuses[i] = location.status();
        return;
      }
      auto client = FlightClient::Connect(*location);
      if (!client.ok()) {
        statuses[i] = client.status();
        return;
      }
      auto authenticated = (*client)->Authenticate(
          {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd"));
      if (!authenticated.ok()) {
        statuses[i] = authenticated;
        return;
      }
      auto reader = (*client)->DoGet(Ticket{"auth-ticket"});
      if (!reader.ok()) {
        statuses[i] = reader.status();
        return;
      }
      auto chunk = (*reader)->Next();
      if (!chunk.ok()) {
        statuses[i] = chunk.status();
        return;
      }
      statuses[i] = (*client)->Close();
    });
  }
  for (auto& thread : threads) thread.join();
  for (int i = 0; i < kClients; i++) {
    EXPECT_TRUE(statuses[i].ok()) << "client " << i << ": " << statuses[i];
  }

  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(kClients, harness.server.state()->handshakes);
}

TEST(AsyncGrpcTest, RepeatedHandshakesOnFreshChannels) {
  constexpr int kRounds = 50;
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  for (int i = 0; i < kRounds; i++) {
    ASSERT_OK_AND_ASSIGN(auto location,
                         Location::ForScheme("grpc", "127.0.0.1", harness.server.port()));
    ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(location));
    ASSERT_OK(client->Authenticate(
        {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
    ASSERT_OK(client->Close());
  }
  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(kRounds, harness.server.state()->handshakes);
}

TEST(AsyncGrpcTest, LateHandshakeResponseLeavesServerUsable) {
  // The client gives up while the hook is still running, so the server's write
  // fails: the reactor's write-side terminal path must still Finish, and the
  // server must keep serving.
  class SlowAnsweringServer : public AsyncGenericFlightServerBase {
   public:
    Status Handshake(const ServerCallContext&, const std::string&,
                     std::string* response) override {
      handshakes.fetch_add(1);
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      *response = "user";
      return Status::OK();
    }
    std::atomic<int> handshakes{0};
  };

  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  SlowAnsweringServer server;
  FlightServerOptions options(location);
  ASSERT_OK(server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  FlightCallOptions impatient;
  impatient.timeout = std::chrono::milliseconds(50);
  auto status = client->Authenticate(
      impatient, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd"));
  EXPECT_FALSE(status.ok()) << "the client's deadline must expire during the hook";
  EXPECT_NE(std::string::npos, status.ToString().find("Deadline"))
      << "expected a deadline error, got: " << status.ToString();
  EXPECT_EQ(1, server.handshakes.load())
      << "the first call must have reached the hook, so its write had to finish";

  ASSERT_OK_AND_ASSIGN(auto second_location,
                       Location::ForScheme("grpc", "127.0.0.1", server.port()));
  ASSERT_OK_AND_ASSIGN(auto second_client, FlightClient::Connect(second_location));
  ASSERT_OK(second_client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));

  ASSERT_OK(second_client->Close());
  ASSERT_OK(client->Close());
  ASSERT_OK(server.Shutdown());
  ASSERT_OK(server.Wait());
}

TEST(AsyncGrpcTest, EmptyHandshakeResponseIsDelivered) {
  // A hook that answers nothing still produces exactly one (empty) response
  // message: the client's Read must return an empty string rather than hang.
  class NoResponseServer : public AsyncGenericFlightServerBase {
   public:
    Status Handshake(const ServerCallContext&, const std::string&,
                     std::string*) override {
      return Status::OK();
    }
  };

  ASSERT_OK_AND_ASSIGN(auto location, Location::Parse("grpc://localhost:0"));
  NoResponseServer server;
  FlightServerOptions options(location);
  ASSERT_OK(server.Init(options));
  ASSERT_OK_AND_ASSIGN(auto client_location,
                       Location::ForScheme("grpc", "127.0.0.1", server.port()));
  ASSERT_OK_AND_ASSIGN(auto client, FlightClient::Connect(client_location));

  auto handler = std::make_unique<EchoClientAuthHandler>("anything", std::string());
  const auto* observed = handler.get();
  ASSERT_OK(client->Authenticate({}, std::move(handler)));
  EXPECT_EQ(std::string(), observed->response());

  ASSERT_OK(client->Close());
  ASSERT_OK(server.Shutdown());
  ASSERT_OK(server.Wait());
}

TEST(AsyncGrpcTest, RehandshakeOnTheSameChannelWorks) {
  // The Handshake RPC is per-call, so authenticating twice on one channel must
  // work: the second handshake reaches the hook and the channel keeps serving.
  AuthenticatedHarness harness;
  ASSERT_OK(harness.status);
  ASSERT_OK(harness.client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
  ASSERT_OK(harness.client->Authenticate(
      {}, std::make_unique<TestClientAuthHandler>("user", "p4ssw0rd")));
  ASSERT_OK_AND_ASSIGN(auto reader, harness.client->DoGet(Ticket{"auth-ticket"}));
  ASSERT_OK_AND_ASSIGN(auto chunk, reader->Next());
  EXPECT_NE(nullptr, chunk.data);

  std::lock_guard<std::mutex> guard(harness.server.state()->mutex);
  EXPECT_EQ(2, harness.server.state()->handshakes);
}

}  // namespace

}  // namespace arrow::flight::transport::grpc
