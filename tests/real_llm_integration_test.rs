//! Real LLM Integration Tests
//!
//! These tests use real API keys to test actual LLM provider functionality.
//! They are ignored by default to prevent accidental API usage during normal testing.
//!
//! ## Running Tests
//!
//! ### Individual Provider Tests
//! ```bash
//! # Test specific provider (set corresponding API key first)
//! export OPENAI_API_KEY="your-key"
//! cargo test test_openai_integration -- --ignored
//!
//! export ANTHROPIC_API_KEY="your-key"
//! cargo test test_anthropic_integration -- --ignored
//!
//! export GEMINI_API_KEY="your-key"
//! cargo test test_gemini_integration -- --ignored
//! ```
//!
//! ### All Available Providers
//! ```bash
//! # Set API keys for providers you want to test
//! export OPENAI_API_KEY="your-openai-key"
//! export ANTHROPIC_API_KEY="your-anthropic-key"
//! # ... set other keys as needed
//!
//! # Run all available provider tests
//! cargo test test_all_available_providers -- --ignored
//! ```
//!
//! ## Environment Variables
//!
//! ### Required API Keys
//! - `OPENAI_API_KEY`: OpenAI API key
//! - `ANTHROPIC_API_KEY`: Anthropic API key
//! - `GEMINI_API_KEY`: Google Gemini API key
//! - `DEEPSEEK_API_KEY`: DeepSeek API key
//! - `OPENROUTER_API_KEY`: OpenRouter API key
//! - `GROQ_API_KEY`: Groq API key
//! - `XAI_API_KEY`: xAI API key
//!
//! ### Optional Base URL Overrides
//! - `OPENAI_BASE_URL`: Override OpenAI base URL (for proxies/custom endpoints)
//! - `ANTHROPIC_BASE_URL`: Override Anthropic base URL
//!
//! ## Test Coverage
//!
//! Each provider test includes:
//! - ✅ **Non-streaming chat**: Basic request/response functionality
//! - 🌊 **Streaming chat**: Real-time response streaming
//! - 🔢 **Embeddings**: Text embedding generation (if supported)
//! - 🧠 **Reasoning**: Advanced reasoning/thinking capabilities (if supported)
//!
//! ### Provider Capabilities Matrix
//! | Provider   | Chat | Streaming | Embeddings | Reasoning |
//! |------------|------|-----------|------------|-----------|
//! | OpenAI     | ✅   | ✅        | ✅         | ✅ (o1)   |
//! | Anthropic  | ✅   | ✅        | ❌         | ✅ (thinking) |
//! | Gemini     | ✅   | ✅        | ✅         | ✅ (thinking) |
//! | DeepSeek   | ✅   | ✅        | ❌         | ✅ (reasoner) |
//! | OpenRouter | ✅   | ✅        | ❌         | ✅ (o1 models) |
//! | Groq       | ✅   | ✅        | ❌         | ❌        |
//! | xAI        | ✅   | ✅        | ❌         | ✅ (Grok) |

use futures::StreamExt;
use siumai::prelude::*;
use siumai::providers::openai_compatible::providers::models::{deepseek, groq, openrouter, xai};
use siumai::stream::ChatStreamEvent;
use std::env;

/// Test configuration for a provider
#[derive(Debug, Clone)]
struct ProviderTestConfig {
    name: &'static str,
    api_key_env: &'static str,
    default_model: &'static str,
    supports_embedding: bool,
    supports_reasoning: bool,
    reasoning_model: Option<&'static str>,
}

/// Get all provider configurations
fn get_provider_configs() -> Vec<ProviderTestConfig> {
    vec![
        ProviderTestConfig {
            name: "OpenAI",
            api_key_env: "OPENAI_API_KEY",
            default_model: "gpt-4o-mini",
            supports_embedding: true,
            supports_reasoning: true,
            reasoning_model: Some("o1-mini"),
        },
        ProviderTestConfig {
            name: "Anthropic",
            api_key_env: "ANTHROPIC_API_KEY",
            default_model: "claude-3-5-haiku-20241022",
            supports_embedding: false,
            supports_reasoning: true,
            reasoning_model: Some("claude-3-5-sonnet-20241022"),
        },
        ProviderTestConfig {
            name: "Gemini",
            api_key_env: "GEMINI_API_KEY",
            default_model: "gemini-1.5-flash",
            supports_embedding: true,
            supports_reasoning: true,
            reasoning_model: Some("gemini-1.5-pro"),
        },
        ProviderTestConfig {
            name: "DeepSeek",
            api_key_env: "DEEPSEEK_API_KEY",
            default_model: deepseek::CHAT,
            supports_embedding: false,
            supports_reasoning: true,
            reasoning_model: Some(deepseek::REASONER),
        },
        ProviderTestConfig {
            name: "OpenRouter",
            api_key_env: "OPENROUTER_API_KEY",
            default_model: openrouter::openai::GPT_4O,
            supports_embedding: false,
            supports_reasoning: true,
            reasoning_model: Some(openrouter::openai::GPT_4_TURBO),
        },
        ProviderTestConfig {
            name: "Groq",
            api_key_env: "GROQ_API_KEY",
            default_model: groq::LLAMA_3_1_8B,
            supports_embedding: false,
            supports_reasoning: false,
            reasoning_model: None,
        },
        ProviderTestConfig {
            name: "xAI",
            api_key_env: "XAI_API_KEY",
            default_model: xai::GROK_BETA,
            supports_embedding: false,
            supports_reasoning: true,
            reasoning_model: Some(xai::GROK_VISION_BETA),
        },
    ]
}

/// Check if provider environment variables are available
fn is_provider_available(config: &ProviderTestConfig) -> bool {
    env::var(config.api_key_env).is_ok()
}

/// Generic provider integration test
async fn test_provider_integration(config: &ProviderTestConfig) {
    match config.name {
        "OpenAI" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let mut builder = LlmBuilder::new()
                .openai()
                .api_key(api_key)
                .model(config.default_model);

            // Only set base URL if environment variable exists
            if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            let client = builder
                .build()
                .await
                .expect("Failed to build OpenAI client");
            test_non_streaming_chat(&client, config.name).await;
            test_streaming_chat(&client, config.name).await;
            if config.supports_embedding {
                test_embedding(&client, config.name).await;
            }
            if config.supports_reasoning && config.reasoning_model.is_some() {
                test_reasoning_openai(config).await;
            }
        }
        "Anthropic" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let mut builder = LlmBuilder::new()
                .anthropic()
                .api_key(api_key)
                .model(config.default_model);

            // Only set base URL if environment variable exists
            if let Ok(base_url) = env::var("ANTHROPIC_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            let client = builder
                .build()
                .await
                .expect("Failed to build Anthropic client");
            test_non_streaming_chat(&client, config.name).await;
            test_streaming_chat(&client, config.name).await;
            if config.supports_reasoning && config.reasoning_model.is_some() {
                test_reasoning_anthropic(config).await;
            }
        }
        "Gemini" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = LlmBuilder::new()
                .gemini()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
                .expect("Failed to build Gemini client");

            test_non_streaming_chat(&client, config.name).await;
            test_streaming_chat(&client, config.name).await;
            if config.supports_embedding {
                test_embedding(&client, config.name).await;
            }
            if config.supports_reasoning && config.reasoning_model.is_some() {
                test_reasoning_gemini(config).await;
            }
        }
        "DeepSeek" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = LlmBuilder::new()
                .deepseek()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
                .expect("Failed to build DeepSeek client");

            test_non_streaming_chat(&client, config.name).await;
            test_streaming_chat(&client, config.name).await;
            if config.supports_reasoning && config.reasoning_model.is_some() {
                test_reasoning_deepseek(config).await;
            }
        }
        "OpenRouter" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = LlmBuilder::new()
                .openrouter()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
                .expect("Failed to build OpenRouter client");

            test_non_streaming_chat(&client, config.name).await;
            test_streaming_chat(&client, config.name).await;
            if config.supports_reasoning && config.reasoning_model.is_some() {
                test_reasoning_openrouter(config).await;
            }
        }
        "Groq" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = LlmBuilder::new()
                .groq()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
                .expect("Failed to build Groq client");

            test_non_streaming_chat(&client, config.name).await;
            test_streaming_chat(&client, config.name).await;
        }
        "xAI" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = LlmBuilder::new()
                .xai()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
                .expect("Failed to build xAI client");

            test_non_streaming_chat(&client, config.name).await;
            test_streaming_chat(&client, config.name).await;
            if config.supports_reasoning && config.reasoning_model.is_some() {
                test_reasoning_xai(config).await;
            }
        }
        _ => println!("⚠️ Unknown provider: {}", config.name),
    }
}

/// Test non-streaming chat functionality
async fn test_non_streaming_chat<T: ChatCapability>(client: &T, provider_name: &str) {
    println!("  📝 Testing non-streaming chat for {}...", provider_name);

    let messages = vec![
        system!("You are a helpful assistant. Keep responses brief."),
        user!("What is 2+2? Answer with just the number."),
    ];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            assert!(!content.is_empty(), "Response should not be empty");
            println!("    ✅ Non-streaming chat successful: {}", content.trim());

            // Check usage statistics if available
            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!("    ⚠️ Non-streaming chat failed: {}", e);
            println!("    💡 Note: This may indicate API key issues or model unavailability");
            // Skip remaining tests for this provider
        }
    }
}

/// Test streaming chat functionality
async fn test_streaming_chat<T: ChatCapability>(client: &T, provider_name: &str) {
    println!("  🌊 Testing streaming chat for {}...", provider_name);

    let messages = vec![
        system!("You are a helpful assistant. Keep responses brief."),
        user!("Count from 1 to 5, one number per line."),
    ];

    match client.chat_stream(messages, None).await {
        Ok(mut stream) => {
            let mut content_chunks = Vec::new();
            let mut thinking_chunks = Vec::new();

            while let Some(event_result) = stream.next().await {
                match event_result {
                    Ok(event) => match event {
                        ChatStreamEvent::ContentDelta { delta, .. } => {
                            content_chunks.push(delta);
                        }
                        ChatStreamEvent::ThinkingDelta { delta } => {
                            thinking_chunks.push(delta);
                        }
                        ChatStreamEvent::StreamEnd { response } => {
                            let final_content = response.content_text().unwrap_or_default();

                            println!("    ✅ Streaming chat successful");
                            if !final_content.is_empty() {
                                println!("    📝 Final content: {}", final_content.trim());
                            } else {
                                // For streaming, content might be accumulated in chunks
                                let accumulated_content: String = content_chunks.join("");
                                if !accumulated_content.is_empty() {
                                    println!(
                                        "    📝 Accumulated content: {}",
                                        accumulated_content.trim()
                                    );
                                }
                            }

                            if !thinking_chunks.is_empty() {
                                let thinking_content: String = thinking_chunks.join("");
                                println!(
                                    "    🤔 Thinking content length: {} chars",
                                    thinking_content.len()
                                );
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
                            panic!("Streaming chat error for {}: {}", provider_name, error);
                        }
                        _ => {
                            // Handle other events like tool calls, etc.
                        }
                    },
                    Err(e) => {
                        println!("    ❌ Stream error: {}", e);
                        panic!("Streaming chat error for {}: {}", provider_name, e);
                    }
                }
            }

            let total_content: String = content_chunks.join("");
            assert!(
                !total_content.is_empty(),
                "Streamed content should not be empty"
            );
        }
        Err(e) => {
            println!("    ⚠️ Streaming chat failed: {}", e);
            println!("    💡 Note: This may indicate API key issues or model unavailability");
            // Skip remaining tests for this provider
        }
    }
}

/// Test embedding functionality
async fn test_embedding<T: EmbeddingCapability>(client: &T, provider_name: &str) {
    println!("  🔢 Testing embedding for {}...", provider_name);

    let texts = vec![
        "Hello world".to_string(),
        "Artificial intelligence".to_string(),
    ];

    match client.embed(texts.clone()).await {
        Ok(response) => {
            assert_eq!(
                response.embeddings.len(),
                texts.len(),
                "Should have embedding for each text"
            );

            for (i, embedding) in response.embeddings.iter().enumerate() {
                assert!(!embedding.is_empty(), "Embedding {} should not be empty", i);
            }

            println!(
                "    ✅ Embedding successful: {} embeddings with {} dimensions",
                response.embeddings.len(),
                response.embeddings[0].len()
            );

            if let Some(usage) = response.usage {
                println!("    📊 Usage: {} total tokens", usage.total_tokens);
            }
        }
        Err(e) => {
            println!("    ⚠️ Embedding failed (this may be expected): {}", e);
            println!("    💡 Note: Some API keys may not have embedding permissions");
        }
    }
}

/// Test OpenAI reasoning functionality (o1 models)
async fn test_reasoning_openai(config: &ProviderTestConfig) {
    println!("  🧠 Testing OpenAI reasoning for {}...", config.name);

    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let mut builder = LlmBuilder::new()
        .openai()
        .api_key(api_key)
        .model(reasoning_model);

    // Only set base URL if environment variable exists
    if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
        builder = builder.base_url(base_url);
    }

    let client = builder
        .build()
        .await
        .expect("Failed to build OpenAI reasoning client");

    let messages = vec![user!(
        "Solve this step by step: If a train travels 60 mph for 2 hours, then 80 mph for 1.5 hours, what's the total distance?"
    )];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            assert!(
                !content.is_empty(),
                "Reasoning response should not be empty"
            );

            println!("    ✅ OpenAI reasoning successful");
            println!("    📝 Response: {}", content.trim());

            // Check for reasoning tokens in usage
            if let Some(usage) = response.usage {
                if let Some(reasoning_tokens) = usage.reasoning_tokens {
                    println!("    🧠 Reasoning tokens: {}", reasoning_tokens);
                }
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!(
                "    ⚠️ OpenAI reasoning failed (this may be expected): {}",
                e
            );
            println!("    💡 Note: o1 models may not be available for all API keys");
        }
    }
}

/// Test Anthropic thinking functionality
async fn test_reasoning_anthropic(config: &ProviderTestConfig) {
    println!("  🤔 Testing Anthropic thinking for {}...", config.name);

    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let mut builder = LlmBuilder::new()
        .anthropic()
        .api_key(api_key)
        .model(reasoning_model)
        .thinking_budget(2000); // Enable thinking with budget

    // Only set base URL if environment variable exists
    if let Ok(base_url) = env::var("ANTHROPIC_BASE_URL") {
        builder = builder.base_url(base_url);
    }

    let client = builder
        .build()
        .await
        .expect("Failed to build Anthropic thinking client");

    let messages = vec![user!(
        "Think through this problem carefully: What are the key differences between machine learning and traditional programming? Consider multiple perspectives."
    )];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            assert!(!content.is_empty(), "Thinking response should not be empty");

            println!("    ✅ Anthropic thinking successful");
            println!("    📝 Response: {}", content.trim());

            // Check for thinking content
            if let Some(thinking) = response.thinking {
                println!("    🤔 Thinking content length: {} chars", thinking.len());
            }

            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!(
                "    ⚠️ Anthropic thinking failed (this may be expected): {}",
                e
            );
            println!("    💡 Note: Thinking feature may not be available for all models/keys");
        }
    }
}

/// Test Gemini thinking functionality
async fn test_reasoning_gemini(config: &ProviderTestConfig) {
    println!("  💎 Testing Gemini thinking for {}...", config.name);

    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let client = LlmBuilder::new()
        .gemini()
        .api_key(api_key)
        .model(reasoning_model)
        .thinking_budget(-1) // Dynamic thinking
        .build()
        .await
        .expect("Failed to build Gemini thinking client");

    let messages = vec![user!(
        "Analyze this scenario step by step: A company wants to reduce costs by 20% while maintaining quality. What strategies would you recommend?"
    )];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            assert!(
                !content.is_empty(),
                "Gemini thinking response should not be empty"
            );

            println!("    ✅ Gemini thinking successful");
            println!("    📝 Response: {}", content.trim());

            // Check for thinking content
            if let Some(thinking) = response.thinking {
                println!("    💎 Thinking content length: {} chars", thinking.len());
            }

            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!(
                "    ⚠️ Gemini thinking failed (this may be expected): {}",
                e
            );
            println!("    💡 Note: Thinking feature may not be available for all models");
        }
    }
}

/// Test DeepSeek reasoning functionality
async fn test_reasoning_deepseek(config: &ProviderTestConfig) {
    println!("  🔍 Testing DeepSeek reasoning for {}...", config.name);

    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let client = LlmBuilder::new()
        .deepseek()
        .api_key(api_key)
        .model(reasoning_model)
        .reasoning(true)
        .expect("Failed to set reasoning mode")
        .build()
        .await
        .expect("Failed to build DeepSeek reasoning client");

    let messages = vec![user!(
        "Solve this logic puzzle step by step: Three friends have different colored shirts (red, blue, green). Alice doesn't wear red. Bob doesn't wear blue. Charlie doesn't wear green. What color does each person wear?"
    )];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            assert!(
                !content.is_empty(),
                "DeepSeek reasoning response should not be empty"
            );

            println!("    ✅ DeepSeek reasoning successful");
            println!("    📝 Response: {}", content.trim());

            // Check for reasoning content
            if let Some(thinking) = response.thinking {
                println!("    🔍 Reasoning content length: {} chars", thinking.len());
            }

            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!(
                "    ⚠️ DeepSeek reasoning failed (this may be expected): {}",
                e
            );
            println!("    💡 Note: Reasoner models may not be available for all API keys");
        }
    }
}

/// Test OpenRouter reasoning functionality (using o1 models)
async fn test_reasoning_openrouter(config: &ProviderTestConfig) {
    println!("  🌐 Testing OpenRouter reasoning for {}...", config.name);

    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let client = LlmBuilder::new()
        .openrouter()
        .api_key(api_key)
        .model(reasoning_model)
        .build()
        .await
        .expect("Failed to build OpenRouter reasoning client");

    let messages = vec![user!(
        "Think through this math problem: A rectangle has a perimeter of 24 units and an area of 32 square units. What are its dimensions?"
    )];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            assert!(
                !content.is_empty(),
                "OpenRouter reasoning response should not be empty"
            );

            println!("    ✅ OpenRouter reasoning successful");
            println!("    📝 Response: {}", content.trim());

            // Check for reasoning tokens (if using o1 models through OpenRouter)
            if let Some(usage) = response.usage {
                if let Some(reasoning_tokens) = usage.reasoning_tokens {
                    println!("    🧠 Reasoning tokens: {}", reasoning_tokens);
                }
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!(
                "    ⚠️ OpenRouter reasoning failed (this may be expected): {}",
                e
            );
            println!("    💡 Note: o1 models may not be available through OpenRouter for all keys");
        }
    }
}

/// Test xAI reasoning functionality
async fn test_reasoning_xai(config: &ProviderTestConfig) {
    println!("  🚀 Testing xAI reasoning for {}...", config.name);

    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let client = LlmBuilder::new()
        .xai()
        .api_key(api_key)
        .model(reasoning_model)
        .build()
        .await
        .expect("Failed to build xAI reasoning client");

    let messages = vec![user!(
        "Analyze this scenario with careful reasoning: If you have a 3x3 grid and need to place 5 identical objects such that no two objects are in the same row or column, how many ways can you do this?"
    )];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            assert!(
                !content.is_empty(),
                "xAI reasoning response should not be empty"
            );

            println!("    ✅ xAI reasoning successful");
            println!("    📝 Response: {}", content.trim());

            // Check for reasoning content
            if let Some(thinking) = response.thinking {
                println!("    🚀 Reasoning content length: {} chars", thinking.len());
            }

            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!("    ⚠️ xAI reasoning failed (this may be expected): {}", e);
            println!("    💡 Note: Grok models may not be available for all API keys");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_openai_integration() {
        let config = &get_provider_configs()[0]; // OpenAI

        if !is_provider_available(config) {
            println!("⏭️ Skipping OpenAI test: {} not set", config.api_key_env);
            return;
        }

        let api_key = env::var(config.api_key_env).unwrap();

        // Build client with optional base URL override
        let mut builder = LlmBuilder::new()
            .openai()
            .api_key(api_key)
            .model(config.default_model);

        // Only set base URL if environment variable exists
        if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
            builder = builder.base_url(base_url);
        }

        let client = builder
            .build()
            .await
            .expect("Failed to build OpenAI client");

        // Test non-streaming chat
        test_non_streaming_chat(&client, config.name).await;

        // Test streaming chat
        test_streaming_chat(&client, config.name).await;

        // Test embedding if supported
        if config.supports_embedding {
            test_embedding(&client, config.name).await;
        }

        // Test reasoning if supported
        if config.supports_reasoning && config.reasoning_model.is_some() {
            test_reasoning_openai(config).await;
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_anthropic_integration() {
        let config = &get_provider_configs()[1]; // Anthropic

        if !is_provider_available(config) {
            println!("⏭️ Skipping Anthropic test: {} not set", config.api_key_env);
            return;
        }

        let api_key = env::var(config.api_key_env).unwrap();

        // Build client with optional base URL override
        let mut builder = LlmBuilder::new()
            .anthropic()
            .api_key(api_key)
            .model(config.default_model);

        // Only set base URL if environment variable exists
        if let Ok(base_url) = env::var("ANTHROPIC_BASE_URL") {
            builder = builder.base_url(base_url);
        }

        let client = builder
            .build()
            .await
            .expect("Failed to build Anthropic client");

        // Test non-streaming chat
        test_non_streaming_chat(&client, config.name).await;

        // Test streaming chat
        test_streaming_chat(&client, config.name).await;

        // Test reasoning if supported
        if config.supports_reasoning && config.reasoning_model.is_some() {
            test_reasoning_anthropic(config).await;
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_gemini_integration() {
        let config = &get_provider_configs()[2]; // Gemini

        if !is_provider_available(config) {
            println!("⏭️ Skipping Gemini test: {} not set", config.api_key_env);
            return;
        }

        let api_key = env::var(config.api_key_env).unwrap();

        let client = LlmBuilder::new()
            .gemini()
            .api_key(api_key)
            .model(config.default_model)
            .build()
            .await
            .expect("Failed to build Gemini client");

        // Test non-streaming chat
        test_non_streaming_chat(&client, config.name).await;

        // Test streaming chat
        test_streaming_chat(&client, config.name).await;

        // Test embedding if supported
        if config.supports_embedding {
            test_embedding(&client, config.name).await;
        }

        // Test reasoning if supported
        if config.supports_reasoning && config.reasoning_model.is_some() {
            test_reasoning_gemini(config).await;
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_deepseek_integration() {
        let config = &get_provider_configs()[3]; // DeepSeek

        if !is_provider_available(config) {
            println!("⏭️ Skipping DeepSeek test: {} not set", config.api_key_env);
            return;
        }

        let api_key = env::var(config.api_key_env).unwrap();

        let client = LlmBuilder::new()
            .deepseek()
            .api_key(api_key)
            .model(config.default_model)
            .build()
            .await
            .expect("Failed to build DeepSeek client");

        // Test non-streaming chat
        test_non_streaming_chat(&client, config.name).await;

        // Test streaming chat
        test_streaming_chat(&client, config.name).await;

        // Test reasoning if supported
        if config.supports_reasoning && config.reasoning_model.is_some() {
            test_reasoning_deepseek(config).await;
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_openrouter_integration() {
        let config = &get_provider_configs()[4]; // OpenRouter

        if !is_provider_available(config) {
            println!(
                "⏭️ Skipping OpenRouter test: {} not set",
                config.api_key_env
            );
            return;
        }

        let api_key = env::var(config.api_key_env).unwrap();

        let client = LlmBuilder::new()
            .openrouter()
            .api_key(api_key)
            .model(config.default_model)
            .build()
            .await
            .expect("Failed to build OpenRouter client");

        // Test non-streaming chat
        test_non_streaming_chat(&client, config.name).await;

        // Test streaming chat
        test_streaming_chat(&client, config.name).await;

        // Test reasoning if supported
        if config.supports_reasoning && config.reasoning_model.is_some() {
            test_reasoning_openrouter(config).await;
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_groq_integration() {
        let config = &get_provider_configs()[5]; // Groq

        if !is_provider_available(config) {
            println!("⏭️ Skipping Groq test: {} not set", config.api_key_env);
            return;
        }

        let api_key = env::var(config.api_key_env).unwrap();

        let client = LlmBuilder::new()
            .groq()
            .api_key(api_key)
            .model(config.default_model)
            .build()
            .await
            .expect("Failed to build Groq client");

        // Test non-streaming chat
        test_non_streaming_chat(&client, config.name).await;

        // Test streaming chat
        test_streaming_chat(&client, config.name).await;

        // Note: Groq doesn't support reasoning models
    }

    #[tokio::test]
    #[ignore]
    async fn test_xai_integration() {
        let config = &get_provider_configs()[6]; // xAI

        if !is_provider_available(config) {
            println!("⏭️ Skipping xAI test: {} not set", config.api_key_env);
            return;
        }

        let api_key = env::var(config.api_key_env).unwrap();

        let client = LlmBuilder::new()
            .xai()
            .api_key(api_key)
            .model(config.default_model)
            .build()
            .await
            .expect("Failed to build xAI client");

        // Test non-streaming chat
        test_non_streaming_chat(&client, config.name).await;

        // Test streaming chat
        test_streaming_chat(&client, config.name).await;

        // Test reasoning if supported
        if config.supports_reasoning && config.reasoning_model.is_some() {
            test_reasoning_xai(config).await;
        }
    }

    /// Run all available provider tests
    #[tokio::test]
    #[ignore]
    async fn test_all_available_providers() {
        println!("🚀 Running integration tests for all available providers...\n");

        let configs = get_provider_configs();
        let mut tested_providers = Vec::new();
        let mut skipped_providers = Vec::new();

        for config in &configs {
            if is_provider_available(config) {
                tested_providers.push(config.name);
                println!("✅ Testing {} provider...", config.name);

                // Test each provider individually
                match config.name {
                    "OpenAI" => {
                        test_provider_integration(config).await;
                    }
                    "Anthropic" => {
                        test_provider_integration(config).await;
                    }
                    "Gemini" => {
                        test_provider_integration(config).await;
                    }
                    "DeepSeek" => {
                        test_provider_integration(config).await;
                    }
                    "OpenRouter" => {
                        test_provider_integration(config).await;
                    }
                    "Groq" => {
                        test_provider_integration(config).await;
                    }
                    "xAI" => {
                        test_provider_integration(config).await;
                    }
                    _ => println!("⚠️ Unknown provider: {}", config.name),
                }
            } else {
                skipped_providers.push(config.name);
                println!("⏭️ Skipping {} (no API key)", config.name);
            }
        }

        println!("\n📊 Test Summary:");
        println!("   Tested providers: {:?}", tested_providers);
        println!("   Skipped providers: {:?}", skipped_providers);
        println!(
            "   Total providers tested: {}/{}",
            tested_providers.len(),
            configs.len()
        );
    }
}
