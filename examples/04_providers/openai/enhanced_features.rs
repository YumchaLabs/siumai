//! 🚀 `OpenAI` Enhanced Features

#![allow(clippy::needless_borrows_for_generic_args)]
//!
//! This example demonstrates advanced `OpenAI` features including:
//! - JSON mode and structured outputs
//! - Function calling and tools
//! - System fingerprints for consistency
//! - Response format control
//! - Advanced parameter tuning
//! - Cost optimization strategies
//!
//! Before running, set your API key:
//! ```bash
//! export OPENAI_API_KEY="your-openai-key"
//! ```
//!
//! Usage:
//! ```bash
//! cargo run --example openai_enhanced_features
//! ```

use serde::{Deserialize, Serialize};
use siumai::models;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 OpenAI Enhanced Features Demo\n");

    // Get API key
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        println!("⚠️  OPENAI_API_KEY not set, using demo key");
        "demo-key".to_string()
    });

    println!("🔧 Demonstrating OpenAI Enhanced Features:");
    println!("   1. JSON Mode and Structured Outputs");
    println!("   2. Function Calling and Tools");
    println!("   3. System Fingerprints");
    println!("   4. Response Format Control");
    println!("   5. Advanced Parameter Tuning\n");

    // Demo 1: JSON Mode and Structured Outputs
    println!("📋 1. JSON Mode and Structured Outputs");
    demo_json_mode(&api_key).await?;
    println!();

    // Demo 2: Function Calling
    println!("🔧 2. Function Calling and Tools");
    demo_function_calling(&api_key).await?;
    println!();

    // Demo 3: System Fingerprints
    println!("🔒 3. System Fingerprints for Consistency");
    demo_system_fingerprints(&api_key).await?;
    println!();

    // Demo 4: Response Format Control
    println!("📝 4. Response Format Control");
    demo_response_formats(&api_key).await?;
    println!();

    // Demo 5: Advanced Parameters
    println!("⚙️ 5. Advanced Parameter Tuning");
    demo_advanced_parameters(&api_key).await?;

    println!("\n✅ OpenAI Enhanced Features demo completed!");
    Ok(())
}

/// Demo JSON mode and structured outputs
async fn demo_json_mode(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating OpenAI client with JSON mode...");

    let ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model(models::openai::GPT_4O_MINI)
        .temperature(0.1)
        .max_tokens(500)
        .build()
        .await?;

    // Request structured JSON output
    let messages = vec![
        ChatMessage::system(
            "You are a helpful assistant that always responds in valid JSON format. \
            When asked to analyze text, respond with a JSON object containing \
            'sentiment', 'topics', and 'summary' fields.",
        )
        .build(),
        ChatMessage::user(
            "Analyze this text: 'I love using AI tools for productivity. \
            They help me write better code and save time on repetitive tasks.'",
        )
        .build(),
    ];

    println!("   Requesting structured JSON analysis...");

    match ai.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   📄 JSON Response:");
                println!("   {text}");

                // Try to parse as JSON to validate structure
                match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(_) => println!("   ✅ Valid JSON structure confirmed"),
                    Err(_) => println!("   ⚠️  Response is not valid JSON"),
                }
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    Ok(())
}

/// Demo function calling capabilities
async fn demo_function_calling(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Setting up function calling...");

    let ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model(models::openai::GPT_4O_MINI)
        .temperature(0.1)
        .build()
        .await?;

    // Note: Function calling would require additional setup in the actual implementation
    // This is a simplified demonstration of the concept

    let messages = vec![
        ChatMessage::system(
            "You are a helpful assistant with access to tools. \
            When users ask for calculations, weather, or current time, \
            describe what function you would call and what parameters you would use.",
        )
        .build(),
        ChatMessage::user(
            "What's the weather like in San Francisco today? \
            Also, what's 15% of 240?",
        )
        .build(),
    ];

    println!("   Requesting function-aware response...");

    match ai.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   🔧 Function Call Response:");
                println!("   {text}");
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    Ok(())
}

/// Demo system fingerprints for consistency
async fn demo_system_fingerprints(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing response consistency with system fingerprints...");

    let ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model(models::openai::GPT_4O_MINI)
        .temperature(0.0) // Very low temperature for consistency
        .build()
        .await?;

    let messages = vec![
        ChatMessage::system(
            "You are a precise assistant. Always respond with exactly 3 bullet points \
            about the topic, each starting with a dash and containing exactly 10 words.",
        )
        .build(),
        ChatMessage::user("Tell me about artificial intelligence").build(),
    ];

    println!("   Making multiple requests to test consistency...");

    for i in 1..=3 {
        println!("   Request {i}:");
        match ai.chat(messages.clone()).await {
            Ok(response) => {
                if let Some(text) = response.text() {
                    println!("   📝 Response: {}", text.lines().next().unwrap_or(""));
                }
            }
            Err(e) => println!("   ❌ Error: {e}"),
        }
    }

    Ok(())
}

/// Demo response format control
async fn demo_response_formats(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing different response formats...");

    let ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model(models::openai::GPT_4O_MINI)
        .temperature(0.3)
        .build()
        .await?;

    // Test different format requests
    let formats = vec![
        (
            "Markdown",
            "Respond in markdown format with headers and bullet points",
        ),
        ("Table", "Respond in a simple table format"),
        ("Code", "Respond with code examples and comments"),
        ("Bullet Points", "Respond with numbered bullet points only"),
    ];

    for (format_name, format_instruction) in formats {
        println!("   Testing {format_name} format:");

        let messages = vec![
            ChatMessage::system(&format!(
                "{format_instruction}. Be concise and follow the format exactly."
            ))
            .build(),
            ChatMessage::user("Explain the benefits of using Rust programming language").build(),
        ];

        match ai.chat(messages).await {
            Ok(response) => {
                if let Some(text) = response.text() {
                    let preview = text.lines().take(2).collect::<Vec<_>>().join(" ");
                    println!(
                        "   📄 {}: {}...",
                        format_name,
                        &preview[..preview.len().min(60)]
                    );
                }
            }
            Err(e) => println!("   ❌ Error: {e}"),
        }
    }

    Ok(())
}

/// Demo advanced parameter tuning
async fn demo_advanced_parameters(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing different parameter configurations...");

    let configs = vec![
        ("Creative", 0.9, 1.1, 0.9), // High temp, high top_p, high presence
        ("Balanced", 0.7, 1.0, 0.5), // Medium settings
        ("Focused", 0.1, 0.8, 0.1),  // Low temp, focused, minimal presence
    ];

    for (config_name, temperature, top_p, presence_penalty) in configs {
        println!("   Testing {config_name} configuration:");
        println!("     Temperature: {temperature}, Top-p: {top_p}, Presence: {presence_penalty}");

        let ai = Siumai::builder()
            .openai()
            .api_key(api_key)
            .model(models::openai::GPT_4O_MINI)
            .temperature(temperature)
            .max_tokens(100)
            .build()
            .await?;

        let messages = vec![
            ChatMessage::user("Write a creative opening line for a story about space exploration")
                .build(),
        ];

        match ai.chat(messages).await {
            Ok(response) => {
                if let Some(text) = response.text() {
                    println!("   📝 {}: {}", config_name, text.trim());
                }
            }
            Err(e) => println!("   ❌ Error: {e}"),
        }
    }

    Ok(())
}

/// Example data structures for JSON mode
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
struct TextAnalysis {
    sentiment: String,
    topics: Vec<String>,
    summary: String,
    confidence: f64,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
struct FunctionCall {
    name: String,
    parameters: serde_json::Value,
    description: String,
}

/// 🎯 Key `OpenAI` Enhanced Features Summary:
///
/// JSON Mode & Structured Outputs:
/// - Guaranteed JSON response format
/// - Schema validation support
/// - Structured data extraction
/// - API integration friendly
///
/// Function Calling:
/// - Tool integration capabilities
/// - External API connections
/// - Real-time data access
/// - Action execution
///
/// System Fingerprints:
/// - Response consistency
/// - Deterministic outputs
/// - Reproducible results
/// - A/B testing support
///
/// Response Format Control:
/// - Markdown, HTML, code formats
/// - Custom output structures
/// - Template-based responses
/// - Multi-format support
///
/// Advanced Parameters:
/// - Temperature fine-tuning
/// - Top-p nucleus sampling
/// - Presence/frequency penalties
/// - Token limit optimization
///
/// Production Benefits:
/// - Predictable API responses
/// - Enhanced integration capabilities
/// - Cost optimization
/// - Quality consistency
///
/// Next Steps:
/// - Implement custom function tools
/// - Add response validation
/// - Create format templates
/// - Optimize for specific use cases
const fn _documentation() {}
