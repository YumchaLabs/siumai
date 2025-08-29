//! Tool Call Streaming Integration Test
//!
//! This integration test verifies that tool call streaming works correctly
//! and produces the same results as non-streaming tool calls.

use futures::StreamExt;
use serde_json::json;
use siumai::prelude::*;
use siumai::stream::ChatStreamEvent;

#[tokio::test]
#[ignore] // Requires API key
async fn test_tool_call_streaming_vs_non_streaming() {
    // Get API key
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() && key != "demo-key" => key,
        _ => {
            println!("⚠️  OPENAI_API_KEY not set, skipping integration test");
            return;
        }
    };

    // Create client
    let client = LlmBuilder::new()
        .openai()
        .api_key(&api_key)
        .model("gpt-3.5-turbo")
        .temperature(0.1)
        .build()
        .await
        .expect("Failed to create client");

    // Define a simple tool
    let tools = vec![Tool::function(
        "query_search".to_string(),
        "Search for information with a specific count parameter".to_string(),
        json!({
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of results to return"
                },
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["count", "query"]
        }),
    )];

    let messages = vec![
        ChatMessage::user("Please search for 'rust programming' and return exactly 50 results.")
            .build(),
    ];

    // Test 1: Non-streaming tool calls
    println!("📋 Test 1: Using chat_with_tools");
    let non_streaming_result = client
        .chat_with_tools(messages.clone(), Some(tools.clone()))
        .await
        .expect("Non-streaming tool call failed");

    let non_streaming_tool_calls = non_streaming_result.tool_calls.unwrap_or_default();
    assert!(
        !non_streaming_tool_calls.is_empty(),
        "Should have tool calls"
    );

    let first_tool_call = &non_streaming_tool_calls[0];
    let non_streaming_args: serde_json::Value =
        serde_json::from_str(&first_tool_call.function.as_ref().unwrap().arguments)
            .expect("Failed to parse non-streaming arguments");

    // Test 2: Streaming tool calls
    println!("🌊 Test 2: Using chat_stream");
    let mut stream = client
        .chat_stream(messages, Some(tools))
        .await
        .expect("Failed to create stream");

    let mut tool_call_deltas = 0;
    let mut accumulated_args = String::new();
    let mut stream_ended = false;

    while let Some(event) = stream.next().await {
        match event {
            Ok(ChatStreamEvent::ToolCallDelta {
                arguments_delta, ..
            }) => {
                tool_call_deltas += 1;
                if let Some(delta) = arguments_delta {
                    accumulated_args.push_str(&delta);
                }
            }
            Ok(ChatStreamEvent::StreamEnd { .. }) => {
                stream_ended = true;
                break;
            }
            Ok(_) => {} // Ignore other events
            Err(e) => panic!("Stream error: {e}"),
        }
    }

    // Verify streaming results
    assert!(stream_ended, "Stream should have ended");
    assert!(
        tool_call_deltas > 0,
        "Should have received tool call deltas"
    );
    assert!(
        !accumulated_args.is_empty(),
        "Should have accumulated arguments"
    );

    // Parse streaming arguments
    let streaming_args: serde_json::Value =
        serde_json::from_str(&accumulated_args).expect("Failed to parse streaming arguments");

    // Compare results
    assert_eq!(
        non_streaming_args, streaming_args,
        "Streaming and non-streaming results should be identical"
    );

    // Verify specific parameters
    assert_eq!(streaming_args["count"], 50);
    assert_eq!(streaming_args["query"], "rust programming");

    println!("✅ Tool call streaming test passed!");
    println!("   Non-streaming args: {non_streaming_args}");
    println!("   Streaming args: {streaming_args}");
    println!("   Tool call deltas received: {tool_call_deltas}");
}

#[tokio::test]
#[ignore] // Requires API key
async fn test_deepseek_tool_call_streaming() {
    // Get API key
    let api_key = match std::env::var("DEEPSEEK_API_KEY") {
        Ok(key) if !key.is_empty() && key != "demo-key" => key,
        _ => {
            println!("⚠️  DEEPSEEK_API_KEY not set, skipping DeepSeek test");
            return;
        }
    };

    // Create client
    let client = LlmBuilder::new()
        .deepseek()
        .api_key(&api_key)
        .model("deepseek-chat")
        .temperature(0.1)
        .build()
        .await
        .expect("Failed to create DeepSeek client");

    // Define a simple tool
    let tools = vec![Tool::function(
        "query_search".to_string(),
        "Search for information".to_string(),
        json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "query": {"type": "string"}
            },
            "required": ["count", "query"]
        }),
    )];

    let messages = vec![
        ChatMessage::user("Please search for 'rust programming' and return exactly 50 results.")
            .build(),
    ];

    // Test streaming with DeepSeek
    let mut stream = client
        .chat_stream(messages, Some(tools))
        .await
        .expect("Failed to create DeepSeek stream");

    let mut tool_call_deltas = 0;
    let mut accumulated_args = String::new();
    let mut stream_ended = false;

    while let Some(event) = stream.next().await {
        match event {
            Ok(ChatStreamEvent::ToolCallDelta {
                arguments_delta, ..
            }) => {
                tool_call_deltas += 1;
                if let Some(delta) = arguments_delta {
                    accumulated_args.push_str(&delta);
                }
            }
            Ok(ChatStreamEvent::StreamEnd { .. }) => {
                stream_ended = true;
                break;
            }
            Ok(_) => {} // Ignore other events
            Err(e) => panic!("DeepSeek stream error: {e}"),
        }
    }

    // Verify DeepSeek streaming results
    assert!(stream_ended, "DeepSeek stream should have ended");
    assert!(
        tool_call_deltas > 0,
        "Should have received tool call deltas from DeepSeek"
    );
    assert!(
        !accumulated_args.is_empty(),
        "Should have accumulated arguments from DeepSeek"
    );

    // Parse arguments
    let streaming_args: serde_json::Value = serde_json::from_str(&accumulated_args)
        .expect("Failed to parse DeepSeek streaming arguments");

    // Verify specific parameters
    assert_eq!(streaming_args["count"], 50);
    assert_eq!(streaming_args["query"], "rust programming");

    println!("✅ DeepSeek tool call streaming test passed!");
    println!("   Streaming args: {streaming_args}");
    println!("   Tool call deltas received: {tool_call_deltas}");
}
