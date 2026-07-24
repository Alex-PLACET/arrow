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
#include <gflags/gflags.h>

// A small end-to-end example for AsyncFlightServerBase.
//
// Default mode:
//   flight_async_server_example --port=31337
//
// This starts the async Flight server, connects a Flight client to it, then
// runs four operations:
//   1. GetFlightInfo on a path descriptor
//   2. DoGet on the returned ticket
//   3. DoPut with one uploaded record batch
//   4. DoExchange with an "echo" command
//
// Example output:
//   Async Flight server listening on grpc+tcp://0.0.0.0:31337
//   DoGet descriptor: FlightDescriptor<path = 'examples/ints'>
//   DoExchange command: echo
//
//   Client: GetFlightInfo
//     endpoints: 1
//     total_records: 4
//
//   Client: DoGet
//     batch: record_batch:
//     ...
//
//   Client: DoPut
//     server metadata: received 1 batches
//
//   Client: DoExchange echo
//     metadata: client-metadata
//     batch: record_batch:
//     ...
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
void PrintIndentedBlock(std::string_view label, std::string_view text) {
  std::cout << "  " << label << ":" << std::endl;
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

// Print the standard heading for a client RPC step and its primary request
// value.
void PrintClientStep(std::string_view title, std::string_view request_label,
                     std::string_view request_value) {
  PrintSection(std::string(title));
  PrintIndentedBlock(request_label, request_value);
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
      PrintIndentedBlock("received metadata", chunk.app_metadata->ToString());
    }
    if (chunk.data) {
      ++batch_index;
      PrintIndentedBlock("received batch #" + std::to_string(batch_index),
                         chunk.data->ToString());
    }
  }
}

// Read the server's DoPut metadata response and print it.
arrow::Status PrintDoPutMetadata(flight::FlightMetadataReader* reader) {
  std::shared_ptr<arrow::Buffer> metadata;
  ARROW_RETURN_NOT_OK(reader->ReadMetadata(&metadata));
  if (metadata) {
    PrintIndentedBlock("server metadata", metadata->ToString());
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
        PrintIndentedBlock("server received schema", schema->ToString());
        return ::arrow::Loop([reader, num_batches]() {
                 return reader->Next().Then([num_batches](flight::FlightStreamChunk chunk)
                                                -> arrow::Result<arrow::ControlFlow<>> {
                   if (!chunk.data && !chunk.app_metadata) {
                     PrintEvent("server", "DoPut client stream completed");
                     return arrow::Break();
                   }
                   if (chunk.data) {
                     ++(*num_batches);
                     PrintIndentedBlock(
                         "server received batch #" + std::to_string(*num_batches),
                         chunk.data->ToString());
                   }
                   if (chunk.app_metadata) {
                     PrintIndentedBlock("server received metadata",
                                        chunk.app_metadata->ToString());
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
// one upload command, and one bidirectional echo command.
class ExampleAsyncFlightServer : public flight::AsyncFlightServerBase {
 public:
  // Resolve metadata for the example path-based dataset.
  arrow::Future<std::unique_ptr<flight::FlightInfo>> GetFlightInfo(
      const flight::ServerCallContext&,
      const flight::FlightDescriptor& descriptor) override {
    PrintEvent("server", "GetFlightInfo");
    PrintIndentedBlock("descriptor", descriptor.ToString());
    if (!IsExamplePathDescriptor(descriptor)) {
      return arrow::Future<std::unique_ptr<flight::FlightInfo>>::MakeFinished(
          arrow::Status::KeyError("Unknown descriptor: ", descriptor.ToString()));
    }
    return ToFinishedFuture(MakeExampleFlightInfo(descriptor, location()));
  }

  // Return the serialized schema for the example dataset.
  arrow::Future<std::unique_ptr<flight::SchemaResult>> GetSchema(
      const flight::ServerCallContext&,
      const flight::FlightDescriptor& descriptor) override {
    PrintEvent("server", "GetSchema");
    PrintIndentedBlock("descriptor", descriptor.ToString());
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
    PrintIndentedBlock("ticket", ticket.ticket);
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
    PrintIndentedBlock("descriptor", reader->descriptor().ToString());
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
    PrintIndentedBlock("descriptor", reader->descriptor().ToString());
    if (!IsExampleCommandDescriptor(reader->descriptor(), kEchoCommand)) {
      return arrow::Future<>::MakeFinished(arrow::Status::NotImplemented(
          "Only the '", kEchoCommand, "' command is implemented"));
    }
    PrintEvent("server", "DoExchange entering echo loop");
    return EchoExchange(
        std::shared_ptr<flight::AsyncFlightMessageReader>(std::move(reader)),
        std::shared_ptr<flight::AsyncFlightMessageWriter>(std::move(writer)));
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
  PrintIndentedBlock("DoGet descriptor", path_descriptor.ToString());
  PrintIndentedBlock("DoPut command", ExampleUploadDescriptor().ToString());
  PrintIndentedBlock("DoExchange command", ExampleEchoDescriptor().ToString());

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

    MaybePause("GetFlightInfo");
    PrintClientStep("Client Step 1: GetFlightInfo", "sending descriptor",
                    path_descriptor.ToString());
    ARROW_ASSIGN_OR_RAISE(auto info, client->GetFlightInfo(path_descriptor));
    PrintEvent("client", "GetFlightInfo response received");
    std::cout << "  endpoints: " << info->endpoints().size() << std::endl;
    std::cout << "  total_records: " << info->total_records() << std::endl;
    std::cout << "  total_bytes: " << info->total_bytes() << std::endl;

    MaybePause("DoGet");
    PrintClientStep("Client Step 2: DoGet", "sending ticket",
                    info->endpoints().front().ticket.ticket);
    ARROW_ASSIGN_OR_RAISE(auto stream, client->DoGet(info->endpoints().front().ticket));
    ARROW_RETURN_NOT_OK(PrintStreamBatches(stream.get()));

    MaybePause("DoPut");
    PrintClientStep("Client Step 3: DoPut", "sending descriptor",
                    upload_descriptor.ToString());
    ARROW_ASSIGN_OR_RAISE(auto do_put, client->DoPut(upload_descriptor, ExampleSchema()));
    ARROW_ASSIGN_OR_RAISE(auto batches, ExampleBatches());
    PrintIndentedBlock("sending batch", batches.front()->ToString());
    ARROW_RETURN_NOT_OK(do_put.writer->WriteRecordBatch(*batches.front()));
    ARROW_RETURN_NOT_OK(do_put.writer->DoneWriting());
    ARROW_RETURN_NOT_OK(PrintDoPutMetadata(do_put.reader.get()));
    ARROW_RETURN_NOT_OK(do_put.writer->Close());

    MaybePause("DoExchange");
    PrintClientStep("Client Step 4: DoExchange echo", "sending descriptor",
                    echo_descriptor.ToString());
    ARROW_ASSIGN_OR_RAISE(auto exchange, client->DoExchange(echo_descriptor));
    auto metadata = arrow::Buffer::FromString("client-metadata");
    PrintIndentedBlock("sending batch", batches.front()->ToString());
    PrintIndentedBlock("sending metadata", metadata->ToString());
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
