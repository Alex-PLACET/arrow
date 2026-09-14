// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/flight/api.h>
#include <arrow/ipc/dictionary.h>
#include <gflags/gflags.h>

// A small end-to-end example for AsyncFlightServerBase.
//
// Default mode:
//   flight_async_server_example --port=31337
//
// This starts the async Flight server, connects a Flight client to it, then
// exercises every server hook at least once:
//    1. Handshake          (a rejected attempt, then a successful one)
//    2. ListFlights
//    3. GetFlightInfo
//    4. PollFlightInfo     (an in-progress poll, then a completed poll)
//    5. GetSchema
//    6. DoGet
//    7. DoPut
//    8. ListActions
//    9. DoAction           ("greet" and "who-am-i")
//   10. DoExchange         ("echo")
//
// After the handshake, every RPC carries the token issued to the client and
// goes through ValidateToken(), which sets the peer identity returned by the
// "who-am-i" action. 
// 
// Demo credentials: demo-user / demo-password.
//
// Server-only mode:
//   flight_async_server_example --client_demo=false --port=31337

DEFINE_int32(port, 31337, "Server port to listen on");
DEFINE_bool(client_demo, true,
            "Start the async Flight server and run a small client workflow");
DEFINE_bool(
    interactive, false,
    "Pause between client demo steps to make the terminal output easier to follow");

namespace flight = ::arrow::flight;

namespace {

constexpr const char* kDivider =
    "========================================================================";
constexpr const char* kExampleTicket = "ints";
constexpr const char* kUploadCommand = "upload";
constexpr const char* kEchoCommand = "echo";
constexpr const char* kGreetAction = "greet";
constexpr const char* kWhoAmIAction = "who-am-i";
constexpr const char* kPollQueryCommand = "heavy-query";
constexpr const char* kPollFollowupCommand = "heavy-query-poll";
// Demo-only credentials for the Handshake/ValidateToken flow.
constexpr const char* kExampleUser = "demo-user";
constexpr const char* kExamplePassword = "demo-password";
constexpr const char* kExampleBadPassword = "wrong-password";


// Logging ------------------------------------

// Print a full-width divider between major demo sections.
void PrintDivider() { std::cout << kDivider << std::endl; }

// Print a titled section header for the terminal transcript.
void PrintSection(std::string_view title) {
  std::cout << std::endl;
  PrintDivider();
  std::cout << title << std::endl;
  PrintDivider();
  std::cout << std::flush;
}

// Print a one-line event attributed to either the client or server.
void PrintEvent(const char* actor, std::string_view message) {
  std::cout << "[" << actor << "] " << message << std::endl;
}

// Print multiline values in an indented block so descriptors, batches, and
// schemas remain readable in a terminal.
void PrintIndentedBlock(std::string_view label, std::string_view text,
                        const char* actor = nullptr) {
  if (actor) {
    std::cout << "[" << actor << "] ";
  }
  std::cout << label << ":" << std::endl;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t end = text.find('\n', start);
    if (end == std::string::npos) {
      if (start < text.size()) {
        std::cout << "    " << text.substr(start) << std::endl;
      }
      break;
    }
    std::cout << "    " << text.substr(start, end - start) << std::endl;
    start = end + 1;
  }
}

// Print the standard heading for a client RPC step and its primary request
// value.
void PrintClientStep(std::string_view title, std::string_view request_label,
                     std::string_view request_value) {
  PrintSection(std::string(title));
  PrintEvent("client", request_label);
  PrintIndentedBlock("value", request_value, "client");
}

// ------------------------------------

// Optionally pause the guided demo so a reader can step through one RPC at a
// time.
void MaybePause(std::string_view next_step) {
  if (!FLAGS_interactive) {
    return;
  }
  std::cout << std::endl;
  std::cout << "Press Enter to continue to: " << next_step << std::endl;
  std::string line;
  std::getline(std::cin, line);
}

// Return the schema shared by every example RPC in this file.
std::shared_ptr<arrow::Schema> ExampleSchema() {
  return arrow::schema({arrow::field("value", arrow::int64())});
}

// Return the single path descriptor served by the example.
flight::FlightDescriptor ExamplePathDescriptor() {
  return flight::FlightDescriptor::Path({"examples", "ints"});
}

// Return the command descriptor used by the DoPut upload example.
flight::FlightDescriptor ExampleUploadDescriptor() {
  return flight::FlightDescriptor::Command(kUploadCommand);
}

// Return the command descriptor used by the DoExchange echo example.
flight::FlightDescriptor ExampleEchoDescriptor() {
  return flight::FlightDescriptor::Command(kEchoCommand);
}

// Check whether a descriptor matches the example path-based dataset.
bool IsExamplePathDescriptor(const flight::FlightDescriptor& descriptor) {
  return descriptor.type == flight::FlightDescriptor::DescriptorType::PATH &&
         descriptor.path == ExamplePathDescriptor().path;
}

// Check whether a descriptor matches the given example command.
bool IsExampleCommandDescriptor(const flight::FlightDescriptor& descriptor,
                                std::string_view command) {
  return descriptor.type == flight::FlightDescriptor::DescriptorType::CMD &&
         descriptor.cmd == command;
}



template <typename T>
// Convert a synchronous Result<T> into an already-completed Future<T> so the
// example server can return async results without introducing extra threading.
arrow::Future<T> ToFinishedFuture(arrow::Result<T> result) {
  if (result.ok()) {
    return arrow::Future<T>::MakeFinished(std::move(result).MoveValueUnsafe());
  }
  return arrow::Future<T>::MakeFinished(result.status());
}

// Build a single record batch that is reused by DoGet, DoPut, and DoExchange.
arrow::Result<arrow::RecordBatchVector> ExampleBatches() {
  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues({1, 2, 3, 4}));
  std::shared_ptr<arrow::Array> values;
  ARROW_RETURN_NOT_OK(builder.Finish(&values));
  return arrow::RecordBatchVector{
      arrow::RecordBatch::Make(ExampleSchema(), values->length(), {values})};
}

// Build the FlightInfo that advertises the example DoGet stream.
arrow::Result<std::unique_ptr<flight::FlightInfo>> MakeExampleFlightInfo(
    const flight::FlightDescriptor& descriptor, const flight::Location& location) {
  const flight::FlightEndpoint endpoint{
      flight::Ticket{kExampleTicket}, {location}, std::nullopt, ""};
  ARROW_ASSIGN_OR_RAISE(auto info, flight::FlightInfo::Make(*ExampleSchema(), descriptor,
                                                            {endpoint}, 4, -1));
  return std::make_unique<flight::FlightInfo>(std::move(info));
}

// Build the SchemaResult returned by the GetSchema RPC for the same dataset.
arrow::Result<std::unique_ptr<flight::SchemaResult>> MakeExampleSchemaResult(
    const flight::FlightDescriptor& descriptor, const flight::Location& location) {
  ARROW_ASSIGN_OR_RAISE(auto info, MakeExampleFlightInfo(descriptor, location));
  return std::make_unique<flight::SchemaResult>(info->serialized_schema());
}

// An in-memory native async DoGet source. Operations complete immediately in
// this example, but a production implementation can complete their futures
// from storage, compute, or another asynchronous source.
class ExampleAsyncDataStream final : public flight::AsyncFlightDataStream {
 public:
  explicit ExampleAsyncDataStream(std::vector<flight::FlightPayload> payloads)
      : payloads_(std::move(payloads)) {}

  arrow::Future<flight::FlightPayload> GetSchemaPayload() override {
    if (payloads_.empty()) {
      return arrow::Future<flight::FlightPayload>::MakeFinished(
          arrow::Status::Invalid("Example stream has no schema payload"));
    }
    return arrow::Future<flight::FlightPayload>::MakeFinished(payloads_.front());
  }

  arrow::Future<flight::FlightPayload> Next() override {
    if (next_index_ >= payloads_.size()) {
      return arrow::Future<flight::FlightPayload>::MakeFinished(flight::FlightPayload{});
    }
    return arrow::Future<flight::FlightPayload>::MakeFinished(payloads_[next_index_++]);
  }

  arrow::Future<> Close() override {
    closed_ = true;
    return arrow::Future<>::MakeFinished();
  }

 private:
  std::vector<flight::FlightPayload> payloads_;
  size_t next_index_ = 1;
  bool closed_ = false;
};

// Serialize the example batches once, then expose them through the native async
// source above. This keeps the example focused on the AsyncFlightDataStream API.
arrow::Result<std::unique_ptr<flight::AsyncFlightDataStream>> MakeExampleAsyncStream() {
  ARROW_ASSIGN_OR_RAISE(auto batches, ExampleBatches());
  ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make(batches));
  flight::RecordBatchStream stream(std::move(reader));
  std::vector<flight::FlightPayload> payloads;
  ARROW_ASSIGN_OR_RAISE(auto schema, stream.GetSchemaPayload());
  payloads.push_back(std::move(schema));
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto payload, stream.Next());
    if (!payload.ipc_message.metadata) break;
    payloads.push_back(std::move(payload));
  }
  return std::make_unique<ExampleAsyncDataStream>(std::move(payloads));
}

// An in-memory native async source for ListFlights. Futures resolve inline in
// this example, a production source can resolve them from storage, a queue,
// or another asynchronous system.
class ExampleAsyncListing final : public flight::AsyncFlightListing {
 public:
  explicit ExampleAsyncListing(std::vector<flight::FlightInfo> infos)
      : infos_(std::move(infos)) {}

  arrow::Future<std::unique_ptr<flight::FlightInfo>> Next() override {
    if (index_ >= infos_.size()) {
      return arrow::Future<std::unique_ptr<flight::FlightInfo>>::MakeFinished(
          std::unique_ptr<flight::FlightInfo>{});
    }
    return arrow::Future<std::unique_ptr<flight::FlightInfo>>::MakeFinished(
        std::make_unique<flight::FlightInfo>(infos_[index_++]));
  }

  arrow::Future<> Close() override {
    infos_.clear();
    return arrow::Future<>::MakeFinished();
  }

 private:
  std::vector<flight::FlightInfo> infos_;
  size_t index_ = 0;
};

// An in-memory native async source for DoAction results. Same contract as
// ExampleAsyncListing: a null Result marks the end of the stream.
class ExampleAsyncResultStream final : public flight::AsyncResultStream {
 public:
  explicit ExampleAsyncResultStream(std::vector<flight::Result> results)
      : results_(std::move(results)) {}

  arrow::Future<std::unique_ptr<flight::Result>> Next() override {
    if (index_ >= results_.size()) {
      return arrow::Future<std::unique_ptr<flight::Result>>::MakeFinished(
          std::unique_ptr<flight::Result>{});
    }
    return arrow::Future<std::unique_ptr<flight::Result>>::MakeFinished(
        std::make_unique<flight::Result>(results_[index_++]));
  }

  arrow::Future<> Close() override {
    results_.clear();
    return arrow::Future<>::MakeFinished();
  }

 private:
  std::vector<flight::Result> results_;
  size_t index_ = 0;
};

// Client-side half of the handshake: send the password, then check the
// identity the server sends back. GetToken() supplies the token attached to
// every RPC issued after a successful authentication.
class ExampleClientAuthHandler final : public flight::ClientAuthHandler {
 public:
  ExampleClientAuthHandler(std::string username, std::string password)
      : username_(std::move(username)), password_(std::move(password)) {}

  arrow::Status Authenticate(flight::ClientAuthSender* outgoing,
                             flight::ClientAuthReader* incoming) override {
    ARROW_RETURN_NOT_OK(outgoing->Write(password_));
    std::string identity;
    ARROW_RETURN_NOT_OK(incoming->Read(&identity));
    if (identity != username_) {
      return flight::MakeFlightError(flight::FlightStatusCode::Unauthenticated,
                                     "Invalid identity: " + identity);
    }
    return arrow::Status::OK();
  }

  arrow::Status GetToken(std::string* token) override {
    *token = password_;
    return arrow::Status::OK();
  }

 private:
  std::string username_;
  std::string password_;
};

// Drain a Flight stream on the client side and print every chunk as part of the
// demo transcript.
arrow::Status PrintStreamBatches(flight::FlightStreamReader* reader) {
  int batch_index = 0;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto chunk, reader->Next());
    if (!chunk.data && !chunk.app_metadata) {
      PrintEvent("client", "stream completed");
      return arrow::Status::OK();
    }
    if (chunk.app_metadata) {
      PrintIndentedBlock("metadata", chunk.app_metadata->ToString(), "client");
    }
    if (chunk.data) {
      ++batch_index;
      PrintIndentedBlock("batch #" + std::to_string(batch_index), chunk.data->ToString(),
                         "client");
    }
  }
}

// Read the server's DoPut metadata response and print it.
arrow::Status PrintDoPutMetadata(flight::FlightMetadataReader* reader) {
  std::shared_ptr<arrow::Buffer> metadata;
  ARROW_RETURN_NOT_OK(reader->ReadMetadata(&metadata));
  if (metadata) {
    PrintIndentedBlock("server metadata", metadata->ToString(), "client");
  }
  return arrow::Status::OK();
}

// Implement the async DoExchange "echo" behavior by mirroring each inbound
// chunk back to the client in the same order.
arrow::Future<> EchoExchange(
    const std::shared_ptr<flight::AsyncFlightMessageReader>& reader,
    const std::shared_ptr<flight::AsyncFlightMessageWriter>& writer) {
  auto begun = std::make_shared<bool>(false);
  return ::arrow::Loop([reader, writer, begun]() {
           return reader->Next().Then([reader, writer,
                                       begun](flight::FlightStreamChunk chunk)
                                          -> arrow::Future<arrow::ControlFlow<>> {
             if (!chunk.data && !chunk.app_metadata) {
               return ::arrow::ToFuture(arrow::Break());
             }

             arrow::Future<> start = arrow::Future<>::MakeFinished();
             if (!*begun && chunk.data) {
               *begun = true;
               start = writer->Begin(chunk.data->schema());
             }

             return start.Then([writer, chunk = std::move(chunk)]() mutable
                               -> arrow::Future<arrow::ControlFlow<>> {
               if (chunk.data && chunk.app_metadata) {
                 return writer->WriteWithMetadata(*chunk.data, chunk.app_metadata)
                     .Then([]() -> arrow::ControlFlow<> { return arrow::Continue{}; });
               }
               if (chunk.data) {
                 return writer->WriteRecordBatch(*chunk.data)
                     .Then([]() -> arrow::ControlFlow<> { return arrow::Continue{}; });
               }
               return writer->WriteMetadata(chunk.app_metadata)
                   .Then([]() -> arrow::ControlFlow<> { return arrow::Continue{}; });
             });
           });
         })
      .Then([](const arrow::internal::Empty&) { return arrow::Status::OK(); });
}

// Consume a DoPut upload, log what arrived, and send one metadata summary back
// to the client when the upload finishes.
arrow::Future<> DrainPutReader(
    const std::shared_ptr<flight::AsyncFlightMessageReader>& reader,
    const std::shared_ptr<flight::AsyncFlightMetadataWriter>& writer) {
  auto num_batches = std::make_shared<int64_t>(0);
  return reader->GetSchema().Then(
      [reader, writer, num_batches](const std::shared_ptr<arrow::Schema>& schema) {
        PrintIndentedBlock("received schema", schema->ToString(), "server");
        return ::arrow::Loop([reader, num_batches]() {
                 return reader->Next().Then([num_batches](flight::FlightStreamChunk chunk)
                                                -> arrow::Result<arrow::ControlFlow<>> {
                   if (!chunk.data && !chunk.app_metadata) {
                     PrintEvent("server", "DoPut client stream completed");
                     return arrow::Break();
                   }
                   if (chunk.data) {
                     ++(*num_batches);
                     PrintIndentedBlock("received batch #" + std::to_string(*num_batches),
                                        chunk.data->ToString(), "server");
                   }
                   if (chunk.app_metadata) {
                    PrintIndentedBlock("received metadata", chunk.app_metadata->ToString(),
                                       "server");
                   }
                   return arrow::Continue();
                 });
               })
            .Then([writer, num_batches](const arrow::internal::Empty&) {
              auto summary = arrow::Buffer::FromString(
                  "received " + std::to_string(*num_batches) + " batches");
              return writer->WriteMetadata(*summary);
            });
      });
}

// Minimal async Flight server used by the example. It implements one dataset,
// one upload command, one bidirectional echo command, a long-running query,
// two actions, and a handshake/ValidateToken pair.
class ExampleAsyncFlightServer : public flight::AsyncFlightServerBase {
 public:
  // Authenticate the client: read the password, check it, then send the
  // username back as the handshake acknowledgment.
  arrow::Future<> Handshake(
      const flight::ServerCallContext&,
      std::unique_ptr<flight::AsyncServerAuthSender> outgoing,
      std::unique_ptr<flight::AsyncServerAuthReader> incoming) override {
    PrintEvent("server", "Handshake");
    auto sender =
        std::shared_ptr<flight::AsyncServerAuthSender>(std::move(outgoing));
    auto reader =
        std::shared_ptr<flight::AsyncServerAuthReader>(std::move(incoming));
    return reader->Read().Then([sender](const std::string& password) -> arrow::Future<> {
      if (password != kExamplePassword) {
        PrintEvent("server", "handshake rejected: invalid credentials");
        return arrow::Future<>::MakeFinished(flight::MakeFlightError(
            flight::FlightStatusCode::Unauthenticated, "Invalid token"));
      }
      return sender->Write(kExampleUser);
    });
  }

  // Validate the per-RPC token carried in the authorization header after a
  // successful Handshake(). Runs inline on the RPC dispatch path, so keep it
  // cheap and non-blocking.
  arrow::Status ValidateToken(const flight::ServerCallContext&,
                              const std::string& token,
                              std::string* peer_identity) override {
    if (token != kExamplePassword) {
      return flight::MakeFlightError(flight::FlightStatusCode::Unauthenticated,
                                     "Invalid token");
    }
    *peer_identity = kExampleUser;
    PrintEvent("server",
               std::string("ValidateToken ok (identity: ") + *peer_identity + ")");
    return arrow::Status::OK();
  }

  // Advertise the example dataset through the async pull-source API.
  arrow::Future<std::unique_ptr<flight::AsyncFlightListing>> ListFlights(
      const flight::ServerCallContext&, const flight::Criteria*) override {
    PrintEvent("server", "ListFlights");
    auto info = MakeExampleFlightInfo(ExamplePathDescriptor(), location());
    if (!info.ok()) {
      return arrow::Future<std::unique_ptr<flight::AsyncFlightListing>>::MakeFinished(
          info.status());
    }
    std::vector<flight::FlightInfo> infos;
    infos.push_back(std::move(**info));
    return arrow::Future<std::unique_ptr<flight::AsyncFlightListing>>::MakeFinished(
        std::make_unique<ExampleAsyncListing>(std::move(infos)));
  }

  // Resolve metadata for the example path-based dataset.
  arrow::Future<std::unique_ptr<flight::FlightInfo>> GetFlightInfo(
      const flight::ServerCallContext&,
      const flight::FlightDescriptor& descriptor) override {
    PrintEvent("server", "GetFlightInfo");
    PrintIndentedBlock("descriptor", descriptor.ToString(), "server");
    if (!IsExamplePathDescriptor(descriptor)) {
      return arrow::Future<std::unique_ptr<flight::FlightInfo>>::MakeFinished(
          arrow::Status::KeyError("Unknown descriptor: ", descriptor.ToString()));
    }
    return ToFinishedFuture(MakeExampleFlightInfo(descriptor, location()));
  }

  // Serve the long-running query poll lifecycle: the first poll reports
  // partial progress and a follow-up descriptorpolling that descriptor
  // returns the completed result.
  arrow::Future<std::unique_ptr<flight::PollInfo>> PollFlightInfo(
      const flight::ServerCallContext&,
      const flight::FlightDescriptor& descriptor) override {
    PrintEvent("server", "PollFlightInfo");
    PrintIndentedBlock("descriptor", descriptor.ToString(), "server");
    if (!IsExamplePathDescriptor(descriptor) &&
        descriptor != flight::FlightDescriptor::Command(kPollQueryCommand) &&
        descriptor != flight::FlightDescriptor::Command(kPollFollowupCommand)) {
      return arrow::Future<std::unique_ptr<flight::PollInfo>>::MakeFinished(
          arrow::Status::KeyError("Unknown descriptor: ", descriptor.ToString()));
    }
    auto info = MakeExampleFlightInfo(descriptor, location());
    if (!info.ok()) {
      return arrow::Future<std::unique_ptr<flight::PollInfo>>::MakeFinished(
          info.status());
    }
    if (descriptor == flight::FlightDescriptor::Command(kPollFollowupCommand)) {
      return arrow::Future<std::unique_ptr<flight::PollInfo>>::MakeFinished(
          std::make_unique<flight::PollInfo>(
              std::make_unique<flight::FlightInfo>(std::move(**info)), std::nullopt, 1.0,
              std::nullopt));
    }
    return arrow::Future<std::unique_ptr<flight::PollInfo>>::MakeFinished(
        std::make_unique<flight::PollInfo>(
            std::make_unique<flight::FlightInfo>(std::move(**info)),
            flight::FlightDescriptor::Command(kPollFollowupCommand), 0.25,
            flight::Timestamp::clock::now() + std::chrono::seconds{10}));
  }

  // Return the serialized schema for the example dataset.
  arrow::Future<std::unique_ptr<flight::SchemaResult>> GetSchema(
      const flight::ServerCallContext&,
      const flight::FlightDescriptor& descriptor) override {
    PrintEvent("server", "GetSchema");
    PrintIndentedBlock("descriptor", descriptor.ToString(), "server");
    if (!IsExamplePathDescriptor(descriptor)) {
      return arrow::Future<std::unique_ptr<flight::SchemaResult>>::MakeFinished(
          arrow::Status::KeyError("Unknown descriptor: ", descriptor.ToString()));
    }
    return ToFinishedFuture(MakeExampleSchemaResult(descriptor, location()));
  }

  // Serve the example dataset when the advertised ticket is redeemed.
  arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>> DoGet(
      const flight::ServerCallContext&, const flight::Ticket& ticket) override {
    PrintEvent("server", "DoGet");
    PrintIndentedBlock("ticket", ticket.ticket, "server");
    if (ticket.ticket != kExampleTicket) {
      return arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>>::MakeFinished(
          arrow::Status::KeyError("Unknown ticket: ", ticket.ticket));
    }
    PrintEvent("server", "DoGet returning example record batches");
    auto stream = MakeExampleAsyncStream();
    if (!stream.ok()) {
      return arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>>::MakeFinished(
          stream.status());
    }
    return arrow::Future<std::unique_ptr<flight::AsyncFlightDataStream>>::MakeFinished(
        std::move(stream).MoveValueUnsafe());
  }

  // Accept one example upload command and summarize what the client sent.
  arrow::Future<> DoPut(
      const flight::ServerCallContext&,
      std::unique_ptr<flight::AsyncFlightMessageReader> reader,
      std::unique_ptr<flight::AsyncFlightMetadataWriter> writer) override {
    PrintEvent("server", "DoPut");
    PrintIndentedBlock("descriptor", reader->descriptor().ToString(), "server");
    if (!IsExampleCommandDescriptor(reader->descriptor(), kUploadCommand)) {
      return arrow::Future<>::MakeFinished(arrow::Status::NotImplemented(
          "Only the '", kUploadCommand, "' command is implemented"));
    }
    return DrainPutReader(
        std::shared_ptr<flight::AsyncFlightMessageReader>(std::move(reader)),
        std::shared_ptr<flight::AsyncFlightMetadataWriter>(std::move(writer)));
  }

  // Accept one example exchange command and echo the client's stream back.
  arrow::Future<> DoExchange(
      const flight::ServerCallContext&,
      std::unique_ptr<flight::AsyncFlightMessageReader> reader,
      std::unique_ptr<flight::AsyncFlightMessageWriter> writer) override {
    PrintEvent("server", "DoExchange");
    PrintIndentedBlock("descriptor", reader->descriptor().ToString(), "server");
    if (!IsExampleCommandDescriptor(reader->descriptor(), kEchoCommand)) {
      return arrow::Future<>::MakeFinished(arrow::Status::NotImplemented(
          "Only the '", kEchoCommand, "' command is implemented"));
    }
    PrintEvent("server", "DoExchange entering echo loop");
    return EchoExchange(
        std::shared_ptr<flight::AsyncFlightMessageReader>(std::move(reader)),
        std::shared_ptr<flight::AsyncFlightMessageWriter>(std::move(writer)));
  }

  // Serve the example actions through the async result-stream API.
  arrow::Future<std::unique_ptr<flight::AsyncResultStream>> DoAction(
      const flight::ServerCallContext& context, const flight::Action& action) override {
    PrintEvent("server", "DoAction");
    PrintIndentedBlock("action", action.type, "server");
    if (action.type == kGreetAction) {
      const std::string name = action.body ? action.body->ToString() : "world";
      std::vector<flight::Result> results = {flight::Result{
          arrow::Buffer::FromString("Hello, " + name + "!")}};
      return arrow::Future<std::unique_ptr<flight::AsyncResultStream>>::MakeFinished(
          std::make_unique<ExampleAsyncResultStream>(std::move(results)));
    }
    if (action.type == kWhoAmIAction) {
      std::vector<flight::Result> results = {
          flight::Result{arrow::Buffer::FromString(context.peer_identity())}};
      return arrow::Future<std::unique_ptr<flight::AsyncResultStream>>::MakeFinished(
          std::make_unique<ExampleAsyncResultStream>(std::move(results)));
    }
    return arrow::Future<std::unique_ptr<flight::AsyncResultStream>>::MakeFinished(
        arrow::Status::KeyError("Unknown action: ", action.type));
  }

  // Advertise the actions supported by this example.
  arrow::Future<std::vector<flight::ActionType>> ListActions(
      const flight::ServerCallContext&) override {
    PrintEvent("server", "ListActions");
    std::vector<flight::ActionType> actions = {
        {kGreetAction, "Greet the name passed in the action body"},
        {kWhoAmIAction, "Return the authenticated peer identity"},
    };
    return arrow::Future<std::vector<flight::ActionType>>::MakeFinished(
        std::move(actions));
  }
};

// Construct and initialize the example async server using the configured port.
arrow::Result<std::unique_ptr<ExampleAsyncFlightServer>> MakeExampleServer() {
  auto server = std::make_unique<ExampleAsyncFlightServer>();
  flight::Location bind_location;
  ARROW_RETURN_NOT_OK(
      flight::Location::ForGrpcTcp("0.0.0.0", FLAGS_port).Value(&bind_location));
  flight::FlightServerOptions options(bind_location);
  ARROW_RETURN_NOT_OK(server->Init(options));
  return server;
}

// Run the server in standalone mode and wait until it is shut down by a
// signal or by another thread.
arrow::Status RunServerOnly(ExampleAsyncFlightServer* server) {
  const auto path_descriptor = ExamplePathDescriptor();
  PrintSection("Async Flight Server Example");
  PrintEvent("server", "listening on " + server->location().ToString());
  PrintIndentedBlock("DoGet descriptor", path_descriptor.ToString(), "server");
  PrintIndentedBlock("DoPut command", ExampleUploadDescriptor().ToString(), "server");
  PrintIndentedBlock("DoExchange command", ExampleEchoDescriptor().ToString(), "server");

  ARROW_RETURN_NOT_OK(server->SetShutdownOnSignals({SIGTERM, SIGINT}));
  return server->Serve();
}

// Run the full in-process demo: start the server, connect a client, exercise
// the supported RPCs, then shut everything down cleanly.
arrow::Status RunClientDemo() {
  ARROW_ASSIGN_OR_RAISE(auto server, MakeExampleServer());
  std::thread server_thread([&server] {
    auto st = RunServerOnly(server.get());
    if (!st.ok()) {
      std::cerr << "Server thread failed: " << st.ToString() << std::endl;
    }
  });

  auto stop_server = [&](arrow::Status status) -> arrow::Status {
    auto shutdown_status = server->Shutdown();
    if (server_thread.joinable()) {
      server_thread.join();
    }
    if (!status.ok()) {
      return status;
    }
    if (!shutdown_status.ok()) {
      return shutdown_status;
    }
    return arrow::Status::OK();
  };

  auto run_demo = [&]() -> arrow::Status {
    auto path_descriptor = ExamplePathDescriptor();
    auto upload_descriptor = ExampleUploadDescriptor();
    auto echo_descriptor = ExampleEchoDescriptor();
    ARROW_ASSIGN_OR_RAISE(auto client, flight::FlightClient::Connect(server->location()));

    MaybePause("Handshake");
    PrintSection("Client Step 1: Handshake");
    // First show the rejection path: bad credentials must be refused.
    PrintEvent("client", "sending handshake with invalid credentials");
    auto rejected = client->Authenticate(
        {}, std::make_unique<ExampleClientAuthHandler>(kExampleUser, kExampleBadPassword));
    if (rejected.ok()) {
      return arrow::Status::Invalid("Expected the handshake to be rejected");
    }
    // Keep the transcript to one line: the full status appends a context trace.
    std::string reason = rejected.ToString();
    if (auto newline = reason.find('\n'); newline != std::string::npos) {
      reason.resize(newline);
    }
    PrintEvent("client", "handshake rejected as expected: " + reason);
    // Then authenticate for real; every following RPC carries the token and
    // goes through ValidateToken().
    PrintEvent("client", "sending handshake with valid credentials");
    ARROW_RETURN_NOT_OK(client->Authenticate(
        {}, std::make_unique<ExampleClientAuthHandler>(kExampleUser, kExamplePassword)));
    PrintEvent("client", std::string("handshake ok, authenticated as ") + kExampleUser);

    MaybePause("ListFlights");
    PrintSection("Client Step 2: ListFlights");
  PrintEvent("client", "requesting available flights");
    ARROW_ASSIGN_OR_RAISE(auto listings, client->ListFlights());
  PrintEvent("client", "received ListFlights response");
    int listed = 0;
    while (true) {
      ARROW_ASSIGN_OR_RAISE(auto listed_info, listings->Next());
      if (!listed_info) {
        break;
      }
      ++listed;
      PrintEvent("client", "flight #" + std::to_string(listed) + ": " +
                             listed_info->descriptor().ToString() + " (" +
                             std::to_string(listed_info->total_records()) + " records)");
    }
    if (listed == 0) {
      return arrow::Status::Invalid("Expected at least one listed flight");
    }

    MaybePause("GetFlightInfo");
    PrintClientStep("Client Step 3: GetFlightInfo", "sending descriptor",
                    path_descriptor.ToString());
    ARROW_ASSIGN_OR_RAISE(auto info, client->GetFlightInfo(path_descriptor));
    PrintEvent("client", "GetFlightInfo response received");
    PrintIndentedBlock("endpoints", std::to_string(info->endpoints().size()), "client");
    PrintIndentedBlock("total records", std::to_string(info->total_records()), "client");
    PrintIndentedBlock("total bytes", std::to_string(info->total_bytes()), "client");

    MaybePause("PollFlightInfo");
    PrintClientStep("Client Step 4: PollFlightInfo", "sending descriptor",
                    flight::FlightDescriptor::Command(kPollQueryCommand).ToString());
    ARROW_ASSIGN_OR_RAISE(
        auto poll,
        client->PollFlightInfo(flight::FlightDescriptor::Command(kPollQueryCommand)));
    PrintEvent("client", "received poll response");
    PrintIndentedBlock("progress",
                       poll->progress.has_value() ? std::to_string(*poll->progress) : "unknown",
                       "client");
    if (!poll->descriptor.has_value()) {
      return arrow::Status::Invalid("Expected the query to still be running");
    }
    PrintEvent("client", "requesting poll again");
    PrintIndentedBlock("descriptor", poll->descriptor->ToString(), "client");
    ARROW_ASSIGN_OR_RAISE(auto completed, client->PollFlightInfo(*poll->descriptor));
    if (completed->descriptor.has_value()) {
      return arrow::Status::Invalid("Expected the completed query to be final");
    }
    PrintEvent("client", "received completed poll response");
    PrintIndentedBlock("progress",
               completed->progress.has_value()
                 ? std::to_string(*completed->progress)
                 : "unknown",
               "client");
    PrintIndentedBlock("endpoints", std::to_string(completed->info->endpoints().size()),
               "client");
    PrintEvent("client", "poll completed");

    MaybePause("GetSchema");
    PrintClientStep("Client Step 5: GetSchema", "sending descriptor",
                    path_descriptor.ToString());
    ARROW_ASSIGN_OR_RAISE(auto schema_result, client->GetSchema(path_descriptor));
    arrow::ipc::DictionaryMemo dict_memo;
    ARROW_ASSIGN_OR_RAISE(auto schema, schema_result->GetSchema(&dict_memo));
    PrintEvent("client", "received schema");
    PrintIndentedBlock("schema", schema->ToString(), "client");

    MaybePause("DoGet");
    PrintClientStep("Client Step 6: DoGet", "sending ticket",
                    info->endpoints().front().ticket.ticket);
    ARROW_ASSIGN_OR_RAISE(auto stream, client->DoGet(info->endpoints().front().ticket));
    ARROW_RETURN_NOT_OK(PrintStreamBatches(stream.get()));

    MaybePause("DoPut");
    PrintClientStep("Client Step 7: DoPut", "sending descriptor",
                    upload_descriptor.ToString());
    ARROW_ASSIGN_OR_RAISE(auto do_put, client->DoPut(upload_descriptor, ExampleSchema()));
    ARROW_ASSIGN_OR_RAISE(auto batches, ExampleBatches());
    PrintIndentedBlock("batch", batches.front()->ToString(), "client");
    ARROW_RETURN_NOT_OK(do_put.writer->WriteRecordBatch(*batches.front()));
    ARROW_RETURN_NOT_OK(do_put.writer->DoneWriting());
    ARROW_RETURN_NOT_OK(PrintDoPutMetadata(do_put.reader.get()));
    ARROW_RETURN_NOT_OK(do_put.writer->Close());
    PrintEvent("client", "DoPut completed");

    MaybePause("ListActions");
    PrintSection("Client Step 8: ListActions");
    PrintEvent("client", "requesting supported actions");
    ARROW_ASSIGN_OR_RAISE(auto actions, client->ListActions());
    PrintEvent("client", "received supported actions");
    for (const auto& action : actions) {
      PrintEvent("client", "action: " + action.type + " - " + action.description);
    }

    MaybePause("DoAction");
    PrintSection("Client Step 9: DoAction");
    {
      flight::Action greet{kGreetAction, arrow::Buffer::FromString("async world")};
      PrintEvent("client", "sending greet action (body: async world)");
      ARROW_ASSIGN_OR_RAISE(auto results, client->DoAction(greet));
      ARROW_ASSIGN_OR_RAISE(auto result, results->Next());
      if (!result) {
        return arrow::Status::Invalid("Expected a result from the greet action");
      }
      PrintEvent("client", "received greet result");
      PrintIndentedBlock("result", result->body->ToString(), "client");
    }
    {
      flight::Action who{kWhoAmIAction, arrow::Buffer::FromString("")};
      PrintEvent("client", "sending who-am-i action");
      ARROW_ASSIGN_OR_RAISE(auto results, client->DoAction(who));
      ARROW_ASSIGN_OR_RAISE(auto result, results->Next());
      if (!result) {
        return arrow::Status::Invalid("Expected a result from the who-am-i action");
      }
      PrintEvent("client", "received who-am-i result");
      PrintIndentedBlock("result (peer identity)", result->body->ToString(), "client");
    }

    MaybePause("DoExchange");
    PrintClientStep("Client Step 10: DoExchange echo", "sending descriptor",
                    echo_descriptor.ToString());
    ARROW_ASSIGN_OR_RAISE(auto exchange, client->DoExchange(echo_descriptor));
    auto metadata = arrow::Buffer::FromString("client-metadata");
    PrintIndentedBlock("batch", batches.front()->ToString(), "client");
    PrintIndentedBlock("metadata", metadata->ToString(), "client");
    ARROW_RETURN_NOT_OK(exchange.writer->Begin(ExampleSchema()));
    ARROW_RETURN_NOT_OK(exchange.writer->WriteWithMetadata(*batches.front(), metadata));
    ARROW_RETURN_NOT_OK(exchange.writer->DoneWriting());
    ARROW_RETURN_NOT_OK(PrintStreamBatches(exchange.reader.get()));
    ARROW_RETURN_NOT_OK(exchange.writer->Close());
    PrintSection("Demo Completed");
    PrintEvent("client", "all example RPCs completed successfully");
    return arrow::Status::OK();
  };

  return stop_server(run_demo());
}

}  // namespace

// Parse flags and choose between standalone server mode and the guided
// client/server demo mode.
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  arrow::Status status;
  if (FLAGS_client_demo) {
    status = RunClientDemo();
  } else {
    auto maybe_server = MakeExampleServer();
    if (!maybe_server.ok()) {
      status = maybe_server.status();
    } else {
      auto server = std::move(maybe_server).MoveValueUnsafe();
      status = RunServerOnly(server.get());
    }
  }
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
