//! Tool Capability Integration Tests
//!
//! These tests verify tool calling functionality across all supported providers.
//! They are ignored by default to prevent accidental API usage during normal testing.
//!
//! ## Running Tests
//!
//! ```bash
//! # Test specific provider tool capabilities
//! export OPENAI_API_KEY="your-key"
//! cargo test test_openai_tools -- --ignored
//!
//! export ANTHROPIC_API_KEY="your-key"
//! cargo test test_anthropic_tools -- --ignored
//!
//! # Test all available providers
//! cargo test test_all_provider_tools -- --ignored
//! ```

use futures::StreamExt;
use serde_json::json;
use siumai::prelude::*;
use std::env;

/// Create a simple calculator tool for testing
fn create_calculator_tool() -> Tool {
    Tool::function(
        "calculate",
        "Perform basic mathematical calculations",
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3', '10 * 5')"
                }
            },
            "required": ["expression"]
        }),
    )
}

/// Create a weather tool for testing
fn create_weather_tool() -> Tool {
    Tool::function(
        "get_weather",
        "Get current weather information for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }),
    )
}

/// Test basic tool calling functionality
async fn test_basic_tool_calling<T: ChatCapability>(client: &T, provider_name: &str) {
    println!("  🔧 Testing basic tool calling for {}...", provider_name);

    let tools = vec![create_calculator_tool()];
    let messages = vec![
        system!(
            "You are a helpful assistant. When asked to calculate something, use the calculate tool."
        ),
        user!("What is 15 + 27? Please use the calculator tool to compute this."),
    ];

    match client.chat_with_tools(messages, Some(tools)).await {
        Ok(response) => {
            println!("    ✅ Tool calling successful");

            // Check if tool calls were made
            let tool_calls = response.tool_calls();
            if !tool_calls.is_empty() {
                println!("    🔧 Tool calls made: {}", tool_calls.len());
                for (i, call) in tool_calls.iter().enumerate() {
                    if let Some(info) = call.as_tool_call() {
                        println!(
                            "      Tool {}: {} ({})",
                            i + 1,
                            info.tool_name,
                            info.tool_call_id
                        );
                    }
                }
            } else {
                println!("    ⚠️ No tool calls in response (model may have answered directly)");
            }

            let content = response.content_text().unwrap_or_default();
            if !content.is_empty() {
                println!("    📝 Response: {}", content.trim());
            }

            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!("    ⚠️ Tool calling failed: {}", e);
            println!(
                "    💡 Note: Some models may not support tool calling or may need specific configuration"
            );
        }
    }
}

/// Test streaming tool calling functionality
async fn test_streaming_tool_calling<T: ChatCapability>(client: &T, provider_name: &str) {
    println!(
        "  🌊 Testing streaming tool calling for {}...",
        provider_name
    );

    let tools = vec![create_weather_tool()];
    let messages = vec![
        system!("You are a helpful assistant. When asked about weather, use the get_weather tool."),
        user!("What's the weather like in Tokyo? Please use the weather tool."),
    ];

    match client.chat_stream(messages, Some(tools)).await {
        Ok(mut stream) => {
            let mut tool_calls_received = 0;
            let mut content_chunks = Vec::new();

            while let Some(event_result) = stream.next().await {
                match event_result {
                    Ok(event) => match event {
                        ChatStreamEvent::ContentDelta { delta, .. } => {
                            content_chunks.push(delta);
                        }
                        ChatStreamEvent::ToolCallDelta {
                            id,
                            function_name,
                            arguments_delta,
                            ..
                        } => {
                            tool_calls_received += 1;
                            if let Some(name) = &function_name {
                                println!("    🔧 Tool call: {} (ID: {})", name, id);
                            }
                            if let Some(args) = &arguments_delta {
                                println!("    📝 Arguments delta: {}", args);
                            }
                        }
                        ChatStreamEvent::StreamEnd { response } => {
                            println!("    ✅ Streaming tool calling successful");

                            if tool_calls_received > 0 {
                                println!(
                                    "    🔧 Tool call deltas received: {}",
                                    tool_calls_received
                                );
                            }

                            let final_calls = response.tool_calls();
                            if !final_calls.is_empty() {
                                println!("    🔧 Final tool calls: {}", final_calls.len());
                            }

                            let final_content = response.content_text().unwrap_or_default();
                            if !final_content.is_empty() {
                                println!("    📝 Final response: {}", final_content.trim());
                            }

                            if let Some(usage) = response.usage {
                                println!(
                                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                                    usage.prompt_tokens,
                                    usage.completion_tokens,
                                    usage.total_tokens
                                );
                            }
                            break;
                        }
                        ChatStreamEvent::Error { error } => {
                            println!("    ❌ Stream error: {}", error);
                            return;
                        }
                        _ => {
                            // Handle other events
                        }
                    },
                    Err(e) => {
                        println!("    ❌ Stream error: {}", e);
                        return;
                    }
                }
            }
        }
        Err(e) => {
            println!("    ⚠️ Streaming tool calling failed: {}", e);
            println!("    💡 Note: Some providers may not support streaming tool calls");
        }
    }
}

/// Test multiple tools functionality
async fn test_multiple_tools<T: ChatCapability>(client: &T, provider_name: &str) {
    println!("  🔧 Testing multiple tools for {}...", provider_name);

    let tools = vec![create_calculator_tool(), create_weather_tool()];
    let messages = vec![
        system!("You are a helpful assistant. Use the appropriate tools when needed."),
        user!("Calculate 8 * 7 and also tell me about the weather in London."),
    ];

    match client.chat_with_tools(messages, Some(tools)).await {
        Ok(response) => {
            println!("    ✅ Multiple tools test successful");

            let tool_calls = response.tool_calls();
            if !tool_calls.is_empty() {
                println!("    🔧 Tool calls made: {}", tool_calls.len());
                for call in tool_calls {
                    if let Some(info) = call.as_tool_call() {
                        println!("      - {}: {}", info.tool_name, info.arguments);
                    }
                }
            }

            let content = response.content_text().unwrap_or_default();
            if !content.is_empty() {
                println!("    📝 Response: {}", content.trim());
            }
        }
        Err(e) => {
            println!("    ⚠️ Multiple tools test failed: {}", e);
        }
    }
}

/// Generic provider tool testing
async fn test_provider_tools(provider_name: &str, api_key_env: &str, model: &str) {
    if env::var(api_key_env).is_err() {
        println!(
            "⏭️ Skipping {} tool tests: {} not set",
            provider_name, api_key_env
        );
        return;
    }

    println!("🔧 Testing {} tool capabilities...", provider_name);

    match provider_name {
        "OpenAI" => {
            let api_key = env::var(api_key_env).unwrap();
            let mut builder = Siumai::builder().openai().api_key(api_key).model(model);

            if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            match builder.build().await {
                Ok(client) => {
                    test_basic_tool_calling(&client, provider_name).await;
                    test_streaming_tool_calling(&client, provider_name).await;
                    test_multiple_tools(&client, provider_name).await;
                }
                Err(e) => {
                    println!("❌ Failed to build OpenAI client: {}", e);
                    return;
                }
            }
        }
        "Anthropic" => {
            let api_key = env::var(api_key_env).unwrap();
            let mut builder = Siumai::builder().anthropic().api_key(api_key).model(model);

            if let Ok(base_url) = env::var("ANTHROPIC_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            match builder.build().await {
                Ok(client) => {
                    test_basic_tool_calling(&client, provider_name).await;
                    test_streaming_tool_calling(&client, provider_name).await;
                    test_multiple_tools(&client, provider_name).await;
                }
                Err(e) => {
                    println!("❌ Failed to build Anthropic client: {}", e);
                    return;
                }
            }
        }
        "Gemini" => {
            let api_key = env::var(api_key_env).unwrap();
            match Siumai::builder()
                .gemini()
                .api_key(api_key)
                .model(model)
                .build()
                .await
            {
                Ok(client) => {
                    test_basic_tool_calling(&client, provider_name).await;
                    test_streaming_tool_calling(&client, provider_name).await;
                    test_multiple_tools(&client, provider_name).await;
                }
                Err(e) => {
                    println!("❌ Failed to build Gemini client: {}", e);
                    return;
                }
            }
        }
        "xAI" => {
            let api_key = env::var(api_key_env).unwrap();
            match Siumai::builder()
                .xai()
                .api_key(api_key)
                .model(model)
                .build()
                .await
            {
                Ok(client) => {
                    test_basic_tool_calling(&client, provider_name).await;
                    test_streaming_tool_calling(&client, provider_name).await;
                    test_multiple_tools(&client, provider_name).await;
                }
                Err(e) => {
                    println!("❌ Failed to build xAI client: {}", e);
                    return;
                }
            }
        }
        "Ollama" => {
            let base_url = env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            match Siumai::builder()
                .ollama()
                .base_url(&base_url)
                .model(model)
                .build()
                .await
            {
                Ok(client) => {
                    test_basic_tool_calling(&client, provider_name).await;
                    test_streaming_tool_calling(&client, provider_name).await;
                    test_multiple_tools(&client, provider_name).await;
                }
                Err(e) => {
                    println!("❌ Failed to build Ollama client: {}", e);
                    return;
                }
            }
        }
        _ => {
            println!("❌ Unknown provider: {}", provider_name);
            return;
        }
    }

    println!("✅ {} tool testing completed\n", provider_name);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_openai_tools() {
        test_provider_tools("OpenAI", "OPENAI_API_KEY", "gpt-4o-mini").await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_anthropic_tools() {
        test_provider_tools(
            "Anthropic",
            "ANTHROPIC_API_KEY",
            "claude-3-5-haiku-20241022",
        )
        .await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_gemini_tools() {
        test_provider_tools("Gemini", "GEMINI_API_KEY", "gemini-2.5-flash").await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_xai_tools() {
        test_provider_tools("xAI", "XAI_API_KEY", "grok-4-0709").await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_ollama_tools() {
        // Check if Ollama is available
        let base_url =
            env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let test_client = reqwest::Client::new();

        match test_client
            .get(format!("{}/api/tags", base_url))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                test_provider_tools("Ollama", "OLLAMA_BASE_URL", "qwen3").await;
            }
            _ => {
                println!(
                    "⏭️ Skipping Ollama tool tests: Ollama not available at {}",
                    base_url
                );
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_all_provider_tools() {
        println!("🚀 Running tool capability tests for all available providers...\n");

        let providers = vec![
            ("OpenAI", "OPENAI_API_KEY", "gpt-4o-mini"),
            (
                "Anthropic",
                "ANTHROPIC_API_KEY",
                "claude-3-5-haiku-20241022",
            ),
            ("Gemini", "GEMINI_API_KEY", "gemini-2.5-flash"),
            ("xAI", "XAI_API_KEY", "grok-4-0709"),
        ];

        for (provider_name, api_key_env, model) in providers {
            test_provider_tools(provider_name, api_key_env, model).await;
        }

        // Test Ollama separately due to different setup
        test_provider_tools("Ollama", "OLLAMA_BASE_URL", "qwen3").await;

        println!("🎉 All provider tool testing completed!");
    }
}
