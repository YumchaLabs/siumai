//! HTTP MCP Client Example with OpenAI Responses API and GPT-5
//!
//! This example demonstrates how to create an HTTP MCP client that connects to
//! an MCP server, retrieves available tools, and integrates them with OpenAI's
//! Responses API using GPT-5 for advanced tool calling functionality.
//!
//! Features:
//! - HTTP MCP client implementation
//! - Tool discovery and conversion
//! - OpenAI Responses API integration with GPT-5
//! - Complete request-response flow with tool calling
//! - Real HTTP communication with MCP server
//! - Streaming and non-streaming tool call support
//!
//! To run this example:
//! 1. First start the MCP server: `cargo run --example http_mcp_server`
//! 2. Set your OpenAI API key: export OPENAI_API_KEY=your_key
//! 3. Then run this client: `cargo run --example openai_response_api_http_mcp_client`

use serde::{Deserialize, Serialize};
use serde_json::json;
use siumai::prelude::*;
use siumai::providers::openai::OpenAiClient;
use siumai::providers::openai::config::OpenAiConfig;

/// JSON-RPC request structure for MCP communication
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: String,
    params: Option<serde_json::Value>,
    id: u64,
}

/// JSON-RPC response structure from MCP server
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
    #[allow(dead_code)]
    id: Option<serde_json::Value>,
}

/// JSON-RPC error structure
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[allow(dead_code)]
    data: Option<serde_json::Value>,
}

/// HTTP MCP Client that communicates with MCP server via HTTP
#[derive(Clone)]
pub struct HttpMcpClient {
    http_client: reqwest::Client,
    server_url: String,
    request_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl HttpMcpClient {
    /// Create a new HTTP MCP client
    pub fn new(server_url: String) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            server_url,
            request_id: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// Generate next request ID
    fn next_id(&self) -> u64 {
        self.request_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Send JSON-RPC request to MCP server
    async fn send_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, LlmError> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: self.next_id(),
        };

        println!("📤 Sending MCP request: {method}");

        let response = self
            .http_client
            .post(&self.server_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::provider_error("MCP", format!("HTTP request failed: {e}")))?;

        let json_response: JsonRpcResponse = response.json().await.map_err(|e| {
            LlmError::provider_error("MCP", format!("Failed to parse JSON response: {e}"))
        })?;

        if let Some(error) = json_response.error {
            return Err(LlmError::provider_error(
                "MCP",
                format!("MCP server error: {} (code: {})", error.message, error.code),
            ));
        }

        json_response
            .result
            .ok_or_else(|| LlmError::provider_error("MCP", "No result in MCP response"))
    }

    /// Initialize connection with MCP server
    pub async fn initialize(&self) -> Result<(), LlmError> {
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "siumai-http-mcp-client",
                "version": "1.0.0"
            }
        });

        let _result = self.send_request("initialize", Some(params)).await?;
        println!("✅ MCP client initialized successfully");
        Ok(())
    }

    /// Get available tools from MCP server
    pub async fn get_tools(&self) -> Result<Vec<Tool>, LlmError> {
        let result = self.send_request("tools/list", None).await?;

        let tools_array = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| LlmError::provider_error("MCP", "Invalid tools response format"))?;

        let mut llm_tools = Vec::new();

        for tool_value in tools_array {
            let name = tool_value
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| LlmError::provider_error("MCP", "Tool missing name"))?;

            let description = tool_value
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("No description");

            let input_schema = tool_value
                .get("inputSchema")
                .ok_or_else(|| LlmError::provider_error("MCP", "Tool missing inputSchema"))?;

            // Convert MCP tool schema to siumai Tool
            let llm_tool = Tool::function(
                name.to_string(),
                description.to_string(),
                input_schema.clone(),
            );

            llm_tools.push(llm_tool);
        }

        println!("📋 Retrieved {} tools from MCP server", llm_tools.len());
        Ok(llm_tools)
    }

    /// Execute a tool call via MCP server
    pub async fn call_tool(&self, tool_call: &ToolCall) -> Result<String, LlmError> {
        let function = tool_call
            .function
            .as_ref()
            .ok_or_else(|| LlmError::provider_error("MCP", "Tool call missing function"))?;

        let arguments: serde_json::Value = serde_json::from_str(&function.arguments)
            .map_err(|e| LlmError::provider_error("MCP", format!("Invalid tool arguments: {e}")))?;

        let params = json!({
            "name": function.name,
            "arguments": arguments
        });

        println!(
            "🔧 Calling tool: {} with args: {}",
            function.name, arguments
        );

        let result = self.send_request("tools/call", Some(params)).await?;

        // Extract text content from MCP response
        let content = result
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("text"))
            .and_then(|text| text.as_str())
            .ok_or_else(|| LlmError::provider_error("MCP", "Invalid tool call response format"))?;

        println!("✅ Tool call result: {content}");
        Ok(content.to_string())
    }
}

/// Demonstration of HTTP MCP + OpenAI Responses API integration
pub struct HttpMcpLlmDemo {
    mcp_client: HttpMcpClient,
    openai_client: Option<OpenAiClient>,
}

impl HttpMcpLlmDemo {
    pub async fn new(server_url: String) -> Result<Self, LlmError> {
        // Create HTTP MCP client
        let mcp_client = HttpMcpClient::new(server_url);

        // Initialize MCP connection
        mcp_client.initialize().await?;

        // Try to create OpenAI Responses API client with GPT-5
        let openai_client = if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if api_key != "demo-key" && !api_key.is_empty() {
                println!("🤖 Initializing OpenAI Responses API with GPT-5...");
                let config = OpenAiConfig::new(&api_key)
                    .with_model("gpt-5")
                    .with_responses_api(true);
                let client = OpenAiClient::new(config, reqwest::Client::new());
                Some(client)
            } else {
                None
            }
        } else {
            None
        };

        if openai_client.is_some() {
            println!("✅ OpenAI Responses API client initialized with GPT-5");
        } else {
            println!("⚠️  No OpenAI API key found - set OPENAI_API_KEY for full integration");
        }

        Ok(Self {
            mcp_client,
            openai_client,
        })
    }

    /// Run a complete demo showing HTTP MCP tool integration with OpenAI Responses API
    pub async fn run_demo(&self) -> Result<(), LlmError> {
        println!("🚀 HTTP MCP + OpenAI Responses API (GPT-5) Integration Demo");
        println!("============================================================");
        println!();

        // Step 1: Get available tools from HTTP MCP server
        println!("📋 Step 1: Discovering tools from HTTP MCP server...");
        let tools = self.mcp_client.get_tools().await?;
        println!("✅ Found {} tools:", tools.len());
        for tool in &tools {
            println!(
                "   • {} - {}",
                tool.function.name, tool.function.description
            );
        }
        println!();

        // Step 2: Test tool calls directly (simulating LLM behavior)
        println!("🔧 Step 2: Testing tool calls via HTTP MCP...");
        let test_tool_calls = vec![
            ToolCall {
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: Some(FunctionCall {
                    name: "add".to_string(),
                    arguments: json!({"a": 15, "b": 27}).to_string(),
                }),
            },
            ToolCall {
                id: "call_2".to_string(),
                r#type: "function".to_string(),
                function: Some(FunctionCall {
                    name: "get_time".to_string(),
                    arguments: json!({"timezone": "UTC"}).to_string(),
                }),
            },
        ];

        let mut tool_results = Vec::new();
        for tool_call in &test_tool_calls {
            let result = self.mcp_client.call_tool(tool_call).await?;
            tool_results.push(result);
        }
        println!();

        // Step 3: Show integration with OpenAI Responses API (if available)
        if let Some(ref openai_client) = self.openai_client {
            println!("🤖 Step 3: OpenAI Responses API (GPT-5) Integration Examples...");

            // Run non-streaming example
            self.run_non_streaming_example(openai_client, &tools)
                .await?;

            // Run streaming example
            self.run_streaming_example(openai_client, &tools).await?;
        } else {
            println!(
                "💡 Step 3: No OpenAI API client available (set OPENAI_API_KEY for full integration)"
            );
        }
        println!();

        // Step 4: Summary
        println!("📊 Demo Summary:");
        println!("   ✅ HTTP MCP server communication: Working");
        println!("   ✅ Tool discovery: {} tools found", tools.len());
        println!(
            "   ✅ Tool execution: {} calls successful",
            tool_results.len()
        );
        if self.openai_client.is_some() {
            println!(
                "   ✅ Non-streaming OpenAI Responses API (GPT-5) integration: Complete workflow"
            );
            println!("   ✅ Streaming OpenAI Responses API (GPT-5) integration: Complete workflow");
        } else {
            println!("   ⚠️ OpenAI Responses API integration: Not configured");
        }
        println!();

        println!("✨ HTTP MCP integration demo completed!");
        println!("🔄 This demonstrates:");
        println!("   1. HTTP communication with MCP server");
        println!("   2. Real-time tool discovery via JSON-RPC");
        println!("   3. Tool execution with parameter passing");
        println!(
            "   4. Non-streaming: GPT-5 (Responses API) → Tool Calls → MCP Execution → Final Response"
        );
        println!(
            "   5. Streaming: GPT-5 (Responses API) Stream → Tool Calls → MCP Execution → Final Response"
        );
        println!(
            "   6. Complete integration of OpenAI Responses API streaming with MCP tool execution"
        );

        Ok(())
    }

    /// Demonstrate non-streaming OpenAI Responses API integration with MCP tools
    async fn run_non_streaming_example(
        &self,
        openai_client: &OpenAiClient,
        tools: &[Tool],
    ) -> Result<(), LlmError> {
        println!("📋 Non-Streaming OpenAI Responses API (GPT-5) + MCP Integration");
        println!("================================================================");

        // Create messages for the LLM
        let user_message =
            "Please add the numbers 15 and 27, then tell me the current time in UTC.";
        println!("👤 User: {user_message}");

        let messages = vec![ChatMessage::user(user_message).build()];

        // Test complete chat_with_tools flow with MCP execution using OpenAI Responses API
        println!("\n🔧 Testing OpenAI Responses API chat_with_tools with MCP execution:");
        match openai_client
            .chat_with_tools(messages.clone(), Some(tools.to_vec()))
            .await
        {
            Ok(response) => {
                if let Some(tool_calls) = response.get_tool_calls() {
                    println!("📞 LLM requested {} tool calls:", tool_calls.len());

                    // Execute each tool call via MCP
                    let mut tool_results = Vec::new();
                    for tool_call in tool_calls {
                        if let Some(function) = &tool_call.function {
                            println!(
                                "   🔧 Calling: {} with args: {}",
                                function.name, function.arguments
                            );

                            // Execute the tool via MCP
                            match self.mcp_client.call_tool(tool_call).await {
                                Ok(result) => {
                                    println!("   ✅ Result: {result}");
                                    tool_results.push((tool_call.clone(), result));
                                }
                                Err(e) => {
                                    println!("   ❌ Error: {e}");
                                    tool_results.push((tool_call.clone(), format!("Error: {e}")));
                                }
                            }
                        }
                    }

                    // Create follow-up messages with tool results
                    let mut follow_up_messages = messages.clone();

                    // Add assistant message with tool calls
                    let tool_calls = response.get_tool_calls().unwrap_or(&vec![]).clone();
                    follow_up_messages.push(
                        ChatMessage::assistant(
                            "I'll help you with those calculations and get the current time.",
                        )
                        .with_tool_calls(tool_calls)
                        .build(),
                    );

                    // Add tool result messages
                    for (tool_call, result) in tool_results {
                        follow_up_messages.push(ChatMessage::tool(&result, &tool_call.id).build());
                    }

                    // Get final response from OpenAI Responses API
                    println!("\n🤖 Getting final response from OpenAI Responses API (GPT-5)...");
                    match openai_client.chat(follow_up_messages).await {
                        Ok(final_response) => {
                            if let MessageContent::Text(content) = &final_response.content {
                                println!(
                                    "🎯 Final OpenAI Responses API (GPT-5) response: {content}"
                                );
                            }
                        }
                        Err(e) => {
                            println!("⚠️ Error getting final response: {e}");
                        }
                    }
                } else {
                    println!(
                        "📝 OpenAI Responses API response (no tool calls): {:?}",
                        response.content
                    );
                }
            }
            Err(e) => {
                println!("⚠️ Error with chat_with_tools: {e}");
            }
        }

        println!("✅ Non-streaming example completed!\n");
        Ok(())
    }

    /// Demonstrate streaming OpenAI Responses API integration with MCP tools
    async fn run_streaming_example(
        &self,
        openai_client: &OpenAiClient,
        tools: &[Tool],
    ) -> Result<(), LlmError> {
        use futures::StreamExt;
        use siumai::stream::{ChatStreamEvent, StreamProcessor};

        println!("🌊 Streaming OpenAI Responses API (GPT-5) + MCP Integration");
        println!("============================================================");

        // Create messages for the LLM
        let user_message =
            "Please add the numbers 15 and 27, then tell me the current time in UTC.";
        println!("👤 User: {user_message}");

        let messages = vec![ChatMessage::user(user_message).build()];

        // Test streaming version with complete MCP execution using OpenAI Responses API
        println!("\n🌊 Testing OpenAI Responses API chat_stream with MCP execution:");
        match openai_client
            .chat_stream(messages.clone(), Some(tools.to_vec()))
            .await
        {
            Ok(mut stream) => {
                let mut processor = StreamProcessor::new();
                let mut content_buffer = String::new();

                while let Some(event) = stream.next().await {
                    match event {
                        Ok(ChatStreamEvent::ToolCallDelta {
                            id,
                            function_name,
                            arguments_delta,
                            index,
                        }) => {
                            println!(
                                "🔧 Tool call delta - ID: {id}, Name: {function_name:?}, Args: {arguments_delta:?}, Index: {index:?}"
                            );
                            processor.process_event(ChatStreamEvent::ToolCallDelta {
                                id,
                                function_name,
                                arguments_delta,
                                index,
                            });
                        }
                        Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                            print!("{delta}");
                            content_buffer.push_str(&delta);
                            processor.process_event(ChatStreamEvent::ContentDelta {
                                delta,
                                index: None,
                            });
                        }
                        Ok(ChatStreamEvent::StreamEnd { .. }) => {
                            println!("\n🏁 Stream ended");
                            let final_response = processor.build_final_response();

                            // Execute tool calls if any
                            if let Some(tool_calls) = final_response.get_tool_calls() {
                                println!(
                                    "📋 Executing {} tool calls from stream:",
                                    tool_calls.len()
                                );

                                let mut stream_tool_results = Vec::new();
                                for tool_call in tool_calls {
                                    if let Some(function) = &tool_call.function {
                                        println!(
                                            "   📞 Executing: {} with args: {}",
                                            function.name, function.arguments
                                        );

                                        // Execute the tool via MCP
                                        match self.mcp_client.call_tool(tool_call).await {
                                            Ok(result) => {
                                                println!("   ✅ Result: {result}");
                                                stream_tool_results
                                                    .push((tool_call.clone(), result));
                                            }
                                            Err(e) => {
                                                println!("   ❌ Error: {e}");
                                                stream_tool_results.push((
                                                    tool_call.clone(),
                                                    format!("Error: {e}"),
                                                ));
                                            }
                                        }
                                    }
                                }

                                // Create follow-up messages with tool results for streaming
                                let mut stream_follow_up_messages = messages.clone();

                                // Add assistant message with tool calls
                                let stream_tool_calls =
                                    final_response.get_tool_calls().unwrap_or(&vec![]).clone();
                                stream_follow_up_messages.push(
                                    ChatMessage::assistant("I'll help you with those calculations and get the current time.")
                                        .with_tool_calls(stream_tool_calls)
                                        .build(),
                                );

                                // Add tool result messages
                                for (tool_call, result) in stream_tool_results {
                                    stream_follow_up_messages
                                        .push(ChatMessage::tool(&result, &tool_call.id).build());
                                }

                                // Get final response from OpenAI Responses API for streaming
                                println!(
                                    "\n🤖 Getting final response from OpenAI Responses API (GPT-5) (streaming)..."
                                );
                                match openai_client.chat(stream_follow_up_messages).await {
                                    Ok(final_response) => {
                                        if let MessageContent::Text(content) =
                                            &final_response.content
                                        {
                                            println!(
                                                "🎯 Final OpenAI Responses API (GPT-5) response (streaming): {content}"
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        println!(
                                            "⚠️ Error getting final response (streaming): {e}"
                                        );
                                    }
                                }
                            } else {
                                println!("📝 No tool calls in streaming response");
                                if !content_buffer.is_empty() {
                                    println!("💬 Direct response: {content_buffer}");
                                }
                            }
                            break;
                        }
                        Ok(other) => {
                            println!("📡 Other event: {other:?}");
                        }
                        Err(e) => {
                            println!("❌ Stream error: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                println!("⚠️ Error with chat_stream: {e}");
            }
        }

        println!("✅ Streaming example completed!\n");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    siumai::tracing::init_default_tracing().ok();

    println!("🌐 HTTP MCP Client - OpenAI Responses API (GPT-5) Integration Demo");
    println!("===================================================================");
    println!();

    // Check if MCP server is running
    println!("💡 Make sure the HTTP MCP server is running:");
    println!("   cargo run --example http_mcp_server");
    println!("   Server should be available at: http://127.0.0.1:3000/mcp");
    println!();

    let server_url = "http://127.0.0.1:3000/mcp".to_string();

    // Create and run the demo
    match HttpMcpLlmDemo::new(server_url).await {
        Ok(demo) => {
            if let Err(e) = demo.run_demo().await {
                eprintln!("❌ Demo failed: {e}");
                eprintln!("💡 Make sure the HTTP MCP server is running on port 3000");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize HTTP MCP client: {e}");
            eprintln!("💡 Make sure the HTTP MCP server is running:");
            eprintln!("   cargo run --example http_mcp_server");
            std::process::exit(1);
        }
    }

    Ok(())
}
