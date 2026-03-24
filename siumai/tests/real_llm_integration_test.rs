#![cfg(all(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "xai",
    feature = "ollama",
    feature = "groq",
    feature = "deepseek"
))]
#![allow(deprecated)]
//! Real LLM Integration Tests
//!
//! This file is the broad, manual live-provider suite.
//! It intentionally goes beyond the focused env smoke by probing optional capabilities such as:
//! - model listing
//! - embeddings
//! - reasoning / thinking
//! - wider provider coverage such as OpenRouter, xAI, and Ollama
//!
//! Prefer `provider_env_smoke_test.rs` plus `scripts/test-env-smoke.{sh,bat}` for refactor
//! regression checks around:
//! - environment-driven builder / registry wiring
//! - default model and provider-option merge behavior
//! - basic non-streaming / streaming reachability
//!
//! Keep using this file for slower, broader, more exploratory manual checks when you want to see
//! how real providers behave across extended capabilities. Some sub-checks intentionally log
//! warnings instead of failing hard because provider entitlements, quotas, and model availability
//! vary significantly across accounts.
//! The runtime output now prints per-provider `ok / warn / fail / skip` summaries so optional
//! capability drift is visible without being confused with core regressions.
//!
//! These tests are ignored by default to prevent accidental API usage during normal testing.
//!
//! ## Running Tests
//!
//! ### Recommended First Step For Refactors
//! ```bash
//! ./scripts/test-env-smoke.sh
//! ```
//!
//! ### Individual Provider Tests
//! ```bash
//! # Test specific provider (set corresponding API key first)
//! export OPENAI_API_KEY="your-key"
//! cargo test --test real_llm_integration_test test_openai_integration -- --ignored --nocapture
//!
//! export ANTHROPIC_API_KEY="your-key"
//! cargo test --test real_llm_integration_test test_anthropic_integration -- --ignored --nocapture
//!
//! export GEMINI_API_KEY="your-key"
//! cargo test --test real_llm_integration_test test_gemini_integration -- --ignored --nocapture
//!
//! # For Ollama (make sure Ollama is running locally)
//! export OLLAMA_BASE_URL="http://localhost:11434"
//! cargo test --test real_llm_integration_test test_ollama_integration -- --ignored --nocapture
//! ```
//!
//! ### All Available Providers
//! ```bash
//! # Set API keys for providers you want to test
//! export OPENAI_API_KEY="your-openai-key"
//! export ANTHROPIC_API_KEY="your-anthropic-key"
//! # ... set other keys as needed
//!
//! # Run the broad manual sweep
//! cargo test --test real_llm_integration_test test_all_available_providers -- --ignored --nocapture
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
//! - `OLLAMA_BASE_URL`: Ollama base URL (default: http://localhost:11434)
//!
//! ### Optional Base URL Overrides
//! - `OPENAI_BASE_URL`: Override OpenAI base URL (for proxies/custom endpoints)
//! - `ANTHROPIC_BASE_URL`: Override Anthropic base URL
//!
//! ## Test Coverage
//!
//! Typical coverage in this extended suite:
//! - **Non-streaming chat**: Basic request/response functionality
//! - **Streaming chat**: Real-time response streaming
//! - **Embeddings**: Text embedding generation (if supported)
//! - **Reasoning**: Advanced reasoning/thinking capabilities (if supported)
//! - **Model listing**: Provider metadata surface checks
//!
//! Important:
//! - This suite is not the primary env-wiring regression gate.
//! - Capability sub-checks may downgrade provider/account limitations into warnings.
//! - Focused `provider_env_smoke_test.rs` remains the preferred first-line live regression check.
//!
//! ### Provider Capabilities Matrix
//! | Provider   | Chat | Streaming | Embeddings | Reasoning |
//! |------------|------|-----------|------------|-----------|
//! | OpenAI     | yes  | yes       | yes        | yes (o1)         |
//! | Anthropic  | yes  | yes       | no         | yes (thinking)   |
//! | Gemini     | yes  | yes       | yes        | yes (thinking)   |
//! | DeepSeek   | yes  | yes       | no         | yes (reasoner)   |
//! | OpenRouter | yes  | yes       | no         | yes (o1 models)  |
//! | Groq       | yes  | yes       | no         | no               |
//! | xAI        | yes  | yes       | no         | yes (Grok)       |

use futures::StreamExt;
use siumai::extensions::ModelListingCapability;
use siumai::models::groq;
use siumai::models::openai_compatible::deepseek;
use siumai::prelude::*;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveCheckStatus {
    Pass,
    Warn,
    Fail,
    Skip,
}

impl LiveCheckStatus {
    const fn label(self) -> &'static str {
        match self {
            Self::Pass => "ok",
            Self::Warn => "warn",
            Self::Fail => "error",
            Self::Skip => "skip",
        }
    }
}

#[derive(Debug, Clone)]
struct LiveCheckOutcome {
    check: &'static str,
    status: LiveCheckStatus,
    detail: String,
}

impl LiveCheckOutcome {
    fn pass(check: &'static str, detail: impl Into<String>) -> Self {
        Self {
            check,
            status: LiveCheckStatus::Pass,
            detail: detail.into(),
        }
    }

    fn warn(check: &'static str, detail: impl Into<String>) -> Self {
        Self {
            check,
            status: LiveCheckStatus::Warn,
            detail: detail.into(),
        }
    }

    fn fail(check: &'static str, detail: impl Into<String>) -> Self {
        Self {
            check,
            status: LiveCheckStatus::Fail,
            detail: detail.into(),
        }
    }

    fn skip(check: &'static str, detail: impl Into<String>) -> Self {
        Self {
            check,
            status: LiveCheckStatus::Skip,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone)]
struct ProviderRunSummary {
    provider: &'static str,
    outcomes: Vec<LiveCheckOutcome>,
}

impl ProviderRunSummary {
    fn new(provider: &'static str) -> Self {
        Self {
            provider,
            outcomes: Vec::new(),
        }
    }

    fn push(&mut self, outcome: LiveCheckOutcome) {
        println!(
            "    [{}] {}: {}",
            outcome.status.label(),
            outcome.check,
            outcome.detail
        );
        self.outcomes.push(outcome);
    }

    fn counts(&self) -> (usize, usize, usize, usize) {
        let mut passed = 0;
        let mut warned = 0;
        let mut failed = 0;
        let mut skipped = 0;

        for outcome in &self.outcomes {
            match outcome.status {
                LiveCheckStatus::Pass => passed += 1,
                LiveCheckStatus::Warn => warned += 1,
                LiveCheckStatus::Fail => failed += 1,
                LiveCheckStatus::Skip => skipped += 1,
            }
        }

        (passed, warned, failed, skipped)
    }

    fn has_failures(&self) -> bool {
        self.outcomes
            .iter()
            .any(|outcome| outcome.status == LiveCheckStatus::Fail)
    }

    fn failure_summary(&self) -> String {
        self.outcomes
            .iter()
            .filter(|outcome| outcome.status == LiveCheckStatus::Fail)
            .map(|outcome| format!("{}: {}", outcome.check, outcome.detail))
            .collect::<Vec<_>>()
            .join(" | ")
    }

    fn print_summary(&self) {
        let (passed, warned, failed, skipped) = self.counts();
        println!(
            "  [summary] {} => ok: {}, warn: {}, fail: {}, skip: {}",
            self.provider, passed, warned, failed, skipped
        );
    }
}

fn preview_text(text: &str) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let preview: String = chars.by_ref().take(96).collect();

    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

fn response_reasoning_chars(response: &ChatResponse) -> usize {
    response.reasoning().iter().map(|chunk| chunk.len()).sum()
}

fn assert_provider_summary_clean(summary: &ProviderRunSummary) {
    summary.print_summary();
    assert!(
        !summary.has_failures(),
        "{} live suite failed: {}",
        summary.provider,
        summary.failure_summary()
    );
}

fn print_manual_suite_summary(
    title: &str,
    summaries: &[ProviderRunSummary],
    skipped_providers: &[&'static str],
) {
    let mut passed = 0;
    let mut warned = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for summary in summaries {
        let (provider_passed, provider_warned, provider_failed, provider_skipped) =
            summary.counts();
        passed += provider_passed;
        warned += provider_warned;
        failed += provider_failed;
        skipped += provider_skipped;
    }

    println!("\n[info] {} summary:", title);
    println!("   Providers checked: {}", summaries.len());
    println!("   Provider env-skips: {:?}", skipped_providers);
    println!("   Check totals => ok: {passed}, warn: {warned}, fail: {failed}, skip: {skipped}");
}

async fn probe_ollama(base_url: &str) -> Result<(), String> {
    match reqwest::Client::new()
        .get(format!("{}/api/tags", base_url))
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => Ok(()),
        Ok(response) => Err(format!("HTTP {}", response.status())),
        Err(err) => Err(err.to_string()),
    }
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
            reasoning_model: Some("gpt-5"),
        },
        ProviderTestConfig {
            name: "Anthropic",
            api_key_env: "ANTHROPIC_API_KEY",
            default_model: "claude-3-5-haiku-20241022",
            supports_embedding: false,
            supports_reasoning: true,
            reasoning_model: Some("claude-sonnet-4-20250514"),
        },
        ProviderTestConfig {
            name: "Gemini",
            api_key_env: "GEMINI_API_KEY",
            default_model: "gemini-2.5-flash",
            supports_embedding: true,
            supports_reasoning: true,
            reasoning_model: Some("gemini-2.5-pro"),
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
            default_model: "qwen/qwen3-4b:free",
            supports_embedding: false,
            supports_reasoning: true,
            reasoning_model: Some("qwen/qwen3-4b:free"),
        },
        ProviderTestConfig {
            name: "Groq",
            api_key_env: "GROQ_API_KEY",
            default_model: groq::LLAMA_3_1_8B_INSTANT,
            supports_embedding: false,
            supports_reasoning: false,
            reasoning_model: None,
        },
        ProviderTestConfig {
            name: "xAI",
            api_key_env: "XAI_API_KEY",
            default_model: "grok-4-0709",
            supports_embedding: false,
            supports_reasoning: true,
            reasoning_model: Some("grok-4-0709"),
        },
        ProviderTestConfig {
            name: "Ollama",
            api_key_env: "OLLAMA_BASE_URL", // Use base URL as "key" for Ollama
            default_model: "llama3.2:latest",
            supports_embedding: true,
            supports_reasoning: true,
            reasoning_model: Some("deepseek-r1:8b"),
        },
    ]
}

/// Check if provider environment variables are available
fn is_provider_available(config: &ProviderTestConfig) -> bool {
    if config.name == "Ollama" {
        // For Ollama, we just check if the base URL is set or use default
        true
    } else {
        env::var(config.api_key_env).is_ok()
    }
}

/// Generic provider integration test
async fn test_provider_integration(config: &ProviderTestConfig) -> ProviderRunSummary {
    let mut summary = ProviderRunSummary::new(config.name);

    match config.name {
        "OpenAI" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let mut builder = Siumai::builder()
                .openai()
                .api_key(api_key)
                .model(config.default_model);

            if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            let client = match builder.build().await {
                Ok(client) => client,
                Err(err) => {
                    summary.push(LiveCheckOutcome::fail(
                        "build",
                        format!("OpenAI client build failed: {err}"),
                    ));
                    return summary;
                }
            };

            summary.push(LiveCheckOutcome::pass(
                "build",
                format!("model={}", config.default_model),
            ));
            summary.push(test_non_streaming_chat(&client, config.name).await);
            summary.push(test_streaming_chat(&client, config.name).await);
            summary.push(test_model_listing(&client, config.name).await);

            if config.supports_embedding {
                let mut embedding_builder = Siumai::builder()
                    .openai()
                    .api_key(env::var(config.api_key_env).unwrap())
                    .model("text-embedding-3-small");
                if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
                    embedding_builder = embedding_builder.base_url(base_url);
                }
                match embedding_builder.build().await {
                    Ok(embedding_client) => {
                        summary.push(test_embedding(&embedding_client, config.name).await)
                    }
                    Err(err) => summary.push(LiveCheckOutcome::warn(
                        "embedding",
                        format!("OpenAI embedding client build failed: {err}"),
                    )),
                }
            }

            if config.supports_reasoning && config.reasoning_model.is_some() {
                summary.push(test_reasoning_openai(config).await);
            }
        }
        "Anthropic" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let mut builder = Siumai::builder()
                .anthropic()
                .api_key(api_key)
                .model(config.default_model);

            if let Ok(base_url) = env::var("ANTHROPIC_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            let client = match builder.build().await {
                Ok(client) => client,
                Err(err) => {
                    summary.push(LiveCheckOutcome::fail(
                        "build",
                        format!("Anthropic client build failed: {err}"),
                    ));
                    return summary;
                }
            };

            summary.push(LiveCheckOutcome::pass(
                "build",
                format!("model={}", config.default_model),
            ));
            summary.push(test_non_streaming_chat(&client, config.name).await);
            summary.push(test_streaming_chat(&client, config.name).await);
            summary.push(test_model_listing(&client, config.name).await);

            if config.supports_reasoning && config.reasoning_model.is_some() {
                summary.push(test_reasoning_anthropic(config).await);
            }
        }
        "Gemini" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = match Siumai::builder()
                .gemini()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => client,
                Err(err) => {
                    summary.push(LiveCheckOutcome::fail(
                        "build",
                        format!("Gemini client build failed: {err}"),
                    ));
                    return summary;
                }
            };

            summary.push(LiveCheckOutcome::pass(
                "build",
                format!("model={}", config.default_model),
            ));
            summary.push(test_non_streaming_chat(&client, config.name).await);
            summary.push(test_streaming_chat(&client, config.name).await);
            summary.push(test_model_listing(&client, config.name).await);

            if config.supports_embedding {
                match Siumai::builder()
                    .gemini()
                    .api_key(env::var(config.api_key_env).unwrap())
                    .model("text-embedding-004")
                    .build()
                    .await
                {
                    Ok(embedding_client) => {
                        summary.push(test_embedding(&embedding_client, config.name).await)
                    }
                    Err(err) => summary.push(LiveCheckOutcome::warn(
                        "embedding",
                        format!("Gemini embedding client build failed: {err}"),
                    )),
                }
            }

            if config.supports_reasoning && config.reasoning_model.is_some() {
                summary.push(test_reasoning_gemini(config).await);
            }
        }
        "DeepSeek" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = match Siumai::builder()
                .openai()
                .deepseek()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => client,
                Err(err) => {
                    summary.push(LiveCheckOutcome::fail(
                        "build",
                        format!("DeepSeek client build failed: {err}"),
                    ));
                    return summary;
                }
            };

            summary.push(LiveCheckOutcome::pass(
                "build",
                format!("model={}", config.default_model),
            ));
            summary.push(test_non_streaming_chat(&client, config.name).await);
            summary.push(test_streaming_chat(&client, config.name).await);
            summary.push(test_model_listing(&client, config.name).await);

            if config.supports_reasoning && config.reasoning_model.is_some() {
                summary.push(test_reasoning_deepseek(config).await);
            }
        }
        "OpenRouter" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = match Siumai::builder()
                .openai()
                .openrouter()
                .api_key(api_key)
                .model(config.default_model)
                .max_tokens(1024)
                .build()
                .await
            {
                Ok(client) => client,
                Err(err) => {
                    summary.push(LiveCheckOutcome::fail(
                        "build",
                        format!("OpenRouter client build failed: {err}"),
                    ));
                    return summary;
                }
            };

            summary.push(LiveCheckOutcome::pass(
                "build",
                format!("model={}", config.default_model),
            ));
            summary.push(test_non_streaming_chat(&client, config.name).await);
            summary.push(test_streaming_chat(&client, config.name).await);
            summary.push(test_model_listing(&client, config.name).await);

            if config.supports_reasoning && config.reasoning_model.is_some() {
                summary.push(test_reasoning_openrouter(config).await);
            }
        }
        "Groq" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = match Siumai::builder()
                .groq()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => client,
                Err(err) => {
                    summary.push(LiveCheckOutcome::fail(
                        "build",
                        format!("Groq client build failed: {err}"),
                    ));
                    return summary;
                }
            };

            summary.push(LiveCheckOutcome::pass(
                "build",
                format!("model={}", config.default_model),
            ));
            summary.push(test_non_streaming_chat(&client, config.name).await);
            summary.push(test_streaming_chat(&client, config.name).await);
            summary.push(test_model_listing(&client, config.name).await);
        }
        "xAI" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let client = match Siumai::builder()
                .xai()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => client,
                Err(err) => {
                    summary.push(LiveCheckOutcome::fail(
                        "build",
                        format!("xAI client build failed: {err}"),
                    ));
                    return summary;
                }
            };

            summary.push(LiveCheckOutcome::pass(
                "build",
                format!("model={}", config.default_model),
            ));
            summary.push(test_non_streaming_chat(&client, config.name).await);
            summary.push(test_streaming_chat(&client, config.name).await);
            summary.push(test_model_listing(&client, config.name).await);

            if config.supports_reasoning && config.reasoning_model.is_some() {
                summary.push(test_reasoning_xai(config).await);
            }
        }
        "Ollama" => {
            let base_url = env::var(config.api_key_env)
                .unwrap_or_else(|_| "http://localhost:11434".to_string());

            if let Err(err) = probe_ollama(&base_url).await {
                summary.push(LiveCheckOutcome::skip(
                    "availability",
                    format!("Ollama not reachable at {base_url}: {err}"),
                ));
                return summary;
            }

            let client = match Siumai::builder()
                .ollama()
                .base_url(&base_url)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => client,
                Err(err) => {
                    summary.push(LiveCheckOutcome::fail(
                        "build",
                        format!("Ollama client build failed: {err}"),
                    ));
                    return summary;
                }
            };

            summary.push(LiveCheckOutcome::pass(
                "build",
                format!("model={} base_url={base_url}", config.default_model),
            ));
            summary.push(test_non_streaming_chat(&client, config.name).await);
            summary.push(test_streaming_chat(&client, config.name).await);
            summary.push(test_model_listing(&client, config.name).await);

            if config.supports_embedding {
                match Siumai::builder()
                    .ollama()
                    .base_url(&base_url)
                    .model("nomic-embed-text")
                    .build()
                    .await
                {
                    Ok(embedding_client) => {
                        summary.push(test_embedding(&embedding_client, config.name).await)
                    }
                    Err(err) => summary.push(LiveCheckOutcome::warn(
                        "embedding",
                        format!("Ollama embedding client build failed: {err}"),
                    )),
                }
            }

            if config.supports_reasoning && config.reasoning_model.is_some() {
                summary.push(test_reasoning_ollama(config).await);
            }
        }
        _ => {
            summary.push(LiveCheckOutcome::fail(
                "provider",
                format!("Unknown provider: {}", config.name),
            ));
        }
    }

    summary
}

/// Test non-streaming chat functionality
async fn test_non_streaming_chat<T: ChatCapability>(
    client: &T,
    provider_name: &str,
) -> LiveCheckOutcome {
    let messages = vec![
        system!("You are a helpful assistant. Keep responses brief."),
        user!("What is 2+2? Answer with just the number."),
    ];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default().to_string();
            if content.trim().is_empty() {
                return LiveCheckOutcome::fail(
                    "chat.non_stream",
                    format!("{provider_name} returned empty response content"),
                );
            }

            let detail = if let Some(usage) = response.usage {
                format!(
                    "content='{}' total_tokens={}",
                    preview_text(content.trim()),
                    usage.total_tokens
                )
            } else {
                format!("content='{}'", preview_text(content.trim()))
            };

            LiveCheckOutcome::pass("chat.non_stream", detail)
        }
        Err(e) => LiveCheckOutcome::warn(
            "chat.non_stream",
            format!("{provider_name} request failed: {e}"),
        ),
    }
}

/// Test streaming chat functionality
async fn test_streaming_chat<T: ChatCapability>(
    client: &T,
    provider_name: &str,
) -> LiveCheckOutcome {
    let messages = vec![
        system!("You are a helpful assistant. Keep responses brief."),
        user!("Count from 1 to 5, one number per line."),
    ];

    match client.chat_stream(messages, None).await {
        Ok(mut stream) => {
            let mut content_chunks = Vec::new();
            let mut thinking_chunks = Vec::new();
            let mut final_content = String::new();
            let mut saw_stream_end = false;
            let mut total_tokens = None;

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
                            final_content = response.content_text().unwrap_or_default().to_string();
                            total_tokens = response.usage.map(|usage| usage.total_tokens);
                            saw_stream_end = true;
                            break;
                        }
                        ChatStreamEvent::Error { error } => {
                            return LiveCheckOutcome::fail(
                                "chat.stream",
                                format!("{provider_name} stream error: {error}"),
                            );
                        }
                        _ => {
                            // Handle other events like tool calls, etc.
                        }
                    },
                    Err(e) => {
                        return LiveCheckOutcome::fail(
                            "chat.stream",
                            format!("{provider_name} stream transport error: {e}"),
                        );
                    }
                }
            }

            if !saw_stream_end {
                return LiveCheckOutcome::fail(
                    "chat.stream",
                    format!("{provider_name} stream ended without StreamEnd"),
                );
            }

            let streamed_content = if final_content.trim().is_empty() {
                content_chunks.join("")
            } else {
                final_content
            };

            if streamed_content.trim().is_empty() {
                return LiveCheckOutcome::fail(
                    "chat.stream",
                    format!("{provider_name} stream produced empty content"),
                );
            }

            let thinking_chars: usize = thinking_chunks.iter().map(|chunk| chunk.len()).sum();
            let mut detail = format!("content='{}'", preview_text(streamed_content.trim()));
            if thinking_chars > 0 {
                detail.push_str(&format!(" thinking_chars={thinking_chars}"));
            }
            if let Some(total_tokens) = total_tokens {
                detail.push_str(&format!(" total_tokens={total_tokens}"));
            }

            LiveCheckOutcome::pass("chat.stream", detail)
        }
        Err(e) => LiveCheckOutcome::warn(
            "chat.stream",
            format!("{provider_name} stream setup failed: {e}"),
        ),
    }
}

/// Test embedding functionality
async fn test_embedding<T: EmbeddingCapability>(
    client: &T,
    provider_name: &str,
) -> LiveCheckOutcome {
    let texts = vec![
        "Hello world".to_string(),
        "Artificial intelligence".to_string(),
    ];

    match client.embed(texts.clone()).await {
        Ok(response) => {
            if response.embeddings.len() != texts.len() {
                return LiveCheckOutcome::fail(
                    "embedding",
                    format!(
                        "{provider_name} returned {} embeddings for {} texts",
                        response.embeddings.len(),
                        texts.len()
                    ),
                );
            }

            let dimension = response
                .embeddings
                .first()
                .map(|embedding| embedding.len())
                .unwrap_or(0);
            if dimension == 0
                || response
                    .embeddings
                    .iter()
                    .any(|embedding| embedding.is_empty())
            {
                return LiveCheckOutcome::fail(
                    "embedding",
                    format!("{provider_name} returned empty embedding vectors"),
                );
            }

            let detail = if let Some(usage) = response.usage {
                format!(
                    "vectors={} dimensions={} total_tokens={}",
                    response.embeddings.len(),
                    dimension,
                    usage.total_tokens
                )
            } else {
                format!(
                    "vectors={} dimensions={}",
                    response.embeddings.len(),
                    dimension
                )
            };

            LiveCheckOutcome::pass("embedding", detail)
        }
        Err(e) => LiveCheckOutcome::warn(
            "embedding",
            format!("{provider_name} embedding unavailable: {e}"),
        ),
    }
}

fn reasoning_pass_outcome(model: &str, response: ChatResponse) -> LiveCheckOutcome {
    let content = response.content_text().unwrap_or_default().to_string();
    if content.trim().is_empty() {
        return LiveCheckOutcome::fail(
            "reasoning",
            format!("model={model} returned empty reasoning content"),
        );
    }

    let reasoning_chars = response_reasoning_chars(&response);
    let mut detail = format!("model={model} content='{}'", preview_text(content.trim()));
    if reasoning_chars > 0 {
        detail.push_str(&format!(" reasoning_chars={reasoning_chars}"));
    }
    if let Some(usage) = response.usage {
        detail.push_str(&format!(" total_tokens={}", usage.total_tokens));
    }

    LiveCheckOutcome::pass("reasoning", detail)
}

fn reasoning_warn_outcome(
    provider_name: &str,
    model: &str,
    err: impl std::fmt::Display,
) -> LiveCheckOutcome {
    LiveCheckOutcome::warn(
        "reasoning",
        format!("{provider_name} reasoning unavailable for model {model}: {err}"),
    )
}

/// Test OpenAI reasoning functionality (o1 models)
async fn test_reasoning_openai(config: &ProviderTestConfig) -> LiveCheckOutcome {
    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let mut builder = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model(reasoning_model)
        // Explicitly route to Responses API to exercise that code path
        // alongside the default Chat Completions path tested earlier.
        .openai_responses();

    // Only set base URL if environment variable exists
    if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
        builder = builder.base_url(base_url);
    }

    let client = match builder.build().await {
        Ok(client) => client,
        Err(err) => return reasoning_warn_outcome(config.name, reasoning_model, err),
    };

    let messages = vec![user!("What is 3 + 5? Show your work.")];

    match client.chat(messages).await {
        Ok(response) => reasoning_pass_outcome(reasoning_model, response),
        Err(err) => reasoning_warn_outcome(config.name, reasoning_model, err),
    }
}

/// Test Anthropic thinking functionality
async fn test_reasoning_anthropic(config: &ProviderTestConfig) -> LiveCheckOutcome {
    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let mut builder = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model(reasoning_model)
        .reasoning_budget(2000); // Enable thinking with budget

    // Only set base URL if environment variable exists
    if let Ok(base_url) = env::var("ANTHROPIC_BASE_URL") {
        builder = builder.base_url(base_url);
    }

    let client = match builder.build().await {
        Ok(client) => client,
        Err(err) => return reasoning_warn_outcome(config.name, reasoning_model, err),
    };

    let messages = vec![user!("What is 4  3? Think step by step.")];

    match client.chat(messages).await {
        Ok(response) => reasoning_pass_outcome(reasoning_model, response),
        Err(err) => reasoning_warn_outcome(config.name, reasoning_model, err),
    }
}

/// Test Gemini thinking functionality
async fn test_reasoning_gemini(config: &ProviderTestConfig) -> LiveCheckOutcome {
    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let client = match Siumai::builder()
        .gemini()
        .api_key(api_key)
        .model(reasoning_model)
        .reasoning_budget(-1) // Dynamic thinking (automatically enables thought summaries)
        .build()
        .await
    {
        Ok(client) => client,
        Err(err) => return reasoning_warn_outcome(config.name, reasoning_model, err),
    };

    let messages = vec![user!("What is 10  2? Show your reasoning.")];

    match client.chat(messages).await {
        Ok(response) => reasoning_pass_outcome(reasoning_model, response),
        Err(err) => reasoning_warn_outcome(config.name, reasoning_model, err),
    }
}

/// Test DeepSeek reasoning functionality
async fn test_reasoning_deepseek(config: &ProviderTestConfig) -> LiveCheckOutcome {
    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let client = match Siumai::builder()
        .openai()
        .deepseek()
        .api_key(api_key)
        .model(reasoning_model)
        .build()
        .await
    {
        Ok(client) => client,
        Err(err) => return reasoning_warn_outcome(config.name, reasoning_model, err),
    };

    let messages = vec![user!("What is 7 - 3? Explain briefly.")];

    match client.chat(messages).await {
        Ok(response) => reasoning_pass_outcome(reasoning_model, response),
        Err(err) => reasoning_warn_outcome(config.name, reasoning_model, err),
    }
}

/// Test OpenRouter reasoning functionality (using o1 models)
async fn test_reasoning_openrouter(config: &ProviderTestConfig) -> LiveCheckOutcome {
    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let client = match Siumai::builder()
        .openai()
        .openrouter()
        .api_key(api_key)
        .model(reasoning_model)
        // Keep a conservative cap for compatibility
        .max_tokens(1024)
        .build()
        .await
    {
        Ok(client) => client,
        Err(err) => return reasoning_warn_outcome(config.name, reasoning_model, err),
    };

    let messages = vec![user!("What is 6 + 4? Explain your answer.")];

    match client.chat(messages).await {
        Ok(response) => reasoning_pass_outcome(reasoning_model, response),
        Err(err) => reasoning_warn_outcome(config.name, reasoning_model, err),
    }
}

/// Test xAI reasoning functionality
async fn test_reasoning_xai(config: &ProviderTestConfig) -> LiveCheckOutcome {
    let api_key = env::var(config.api_key_env).unwrap();
    let reasoning_model = config.reasoning_model.unwrap();

    let client = match Siumai::builder()
        .xai()
        .api_key(api_key)
        .model(reasoning_model)
        .build()
        .await
    {
        Ok(client) => client,
        Err(err) => return reasoning_warn_outcome(config.name, reasoning_model, err),
    };

    let messages = vec![user!("What is 8 - 5? Think about it step by step.")];

    match client.chat(messages).await {
        Ok(response) => reasoning_pass_outcome(reasoning_model, response),
        Err(err) => reasoning_warn_outcome(config.name, reasoning_model, err),
    }
}

/// Test Ollama reasoning functionality
async fn test_reasoning_ollama(config: &ProviderTestConfig) -> LiveCheckOutcome {
    let base_url =
        env::var(config.api_key_env).unwrap_or_else(|_| "http://localhost:11434".to_string());
    let reasoning_model = config.reasoning_model.unwrap();

    let client = match Siumai::builder()
        .ollama()
        .base_url(&base_url)
        .model(reasoning_model)
        .reasoning(true) // Enable reasoning mode
        .build()
        .await
    {
        Ok(client) => client,
        Err(err) => return reasoning_warn_outcome(config.name, reasoning_model, err),
    };

    let messages = vec![user!("What is 2+2? Think step by step.")];

    match client.chat(messages).await {
        Ok(response) => reasoning_pass_outcome(reasoning_model, response),
        Err(err) => reasoning_warn_outcome(config.name, reasoning_model, err),
    }
}

/// Test model listing capability
async fn test_model_listing<T>(client: &T, provider_name: &str) -> LiveCheckOutcome
where
    T: ModelListingCapability + Send + Sync,
{
    // Test list_models
    match client.list_models().await {
        Ok(models) => {
            if models.is_empty() {
                return LiveCheckOutcome::warn(
                    "models",
                    format!("{provider_name} returned zero models"),
                );
            }

            let first_model_id = models[0].id.clone();
            match client.get_model(first_model_id.clone()).await {
                Ok(model_info) => {
                    let sample_name = model_info.name.unwrap_or_else(|| model_info.id.clone());
                    let detail = format!(
                        "listed={} sample='{}' capabilities={}",
                        models.len(),
                        sample_name,
                        if model_info.capabilities.is_empty() {
                            "none".to_string()
                        } else {
                            model_info.capabilities.join(", ")
                        }
                    );
                    LiveCheckOutcome::pass("models", detail)
                }
                Err(e) => LiveCheckOutcome::warn(
                    "models",
                    format!(
                        "{provider_name} listed {} models but get_model('{}') failed: {e}",
                        models.len(),
                        first_model_id
                    ),
                ),
            }
        }
        Err(e) => LiveCheckOutcome::warn(
            "models",
            format!("{provider_name} model listing unavailable: {e}"),
        ),
    }
}

/// Test model listing for a specific provider
async fn test_provider_model_listing(config: &ProviderTestConfig) -> ProviderRunSummary {
    let mut summary = ProviderRunSummary::new(config.name);

    match config.name {
        "OpenAI" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let mut builder = Siumai::builder()
                .openai()
                .api_key(api_key)
                .model(config.default_model);

            if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            match builder.build().await {
                Ok(client) => {
                    summary.push(LiveCheckOutcome::pass(
                        "build",
                        format!("model={}", config.default_model),
                    ));
                    summary.push(test_model_listing(&client, config.name).await);
                }
                Err(err) => summary.push(LiveCheckOutcome::fail(
                    "build",
                    format!("OpenAI client build failed: {err}"),
                )),
            }
        }
        "Anthropic" => {
            let api_key = env::var(config.api_key_env).unwrap();
            let mut builder = Siumai::builder()
                .anthropic()
                .api_key(api_key)
                .model(config.default_model);

            if let Ok(base_url) = env::var("ANTHROPIC_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            match builder.build().await {
                Ok(client) => {
                    summary.push(LiveCheckOutcome::pass(
                        "build",
                        format!("model={}", config.default_model),
                    ));
                    summary.push(test_model_listing(&client, config.name).await);
                }
                Err(err) => summary.push(LiveCheckOutcome::fail(
                    "build",
                    format!("Anthropic client build failed: {err}"),
                )),
            }
        }
        "Gemini" => {
            let api_key = env::var(config.api_key_env).unwrap();
            match Siumai::builder()
                .gemini()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => {
                    summary.push(LiveCheckOutcome::pass(
                        "build",
                        format!("model={}", config.default_model),
                    ));
                    summary.push(test_model_listing(&client, config.name).await);
                }
                Err(err) => summary.push(LiveCheckOutcome::fail(
                    "build",
                    format!("Gemini client build failed: {err}"),
                )),
            }
        }
        "DeepSeek" => {
            let api_key = env::var(config.api_key_env).unwrap();
            match Siumai::builder()
                .openai()
                .deepseek()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => {
                    summary.push(LiveCheckOutcome::pass(
                        "build",
                        format!("model={}", config.default_model),
                    ));
                    summary.push(test_model_listing(&client, config.name).await);
                }
                Err(err) => summary.push(LiveCheckOutcome::fail(
                    "build",
                    format!("DeepSeek client build failed: {err}"),
                )),
            }
        }
        "OpenRouter" => {
            let api_key = env::var(config.api_key_env).unwrap();
            match Siumai::builder()
                .openai()
                .openrouter()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => {
                    summary.push(LiveCheckOutcome::pass(
                        "build",
                        format!("model={}", config.default_model),
                    ));
                    summary.push(test_model_listing(&client, config.name).await);
                }
                Err(err) => summary.push(LiveCheckOutcome::fail(
                    "build",
                    format!("OpenRouter client build failed: {err}"),
                )),
            }
        }
        "Groq" => {
            let api_key = env::var(config.api_key_env).unwrap();
            match Siumai::builder()
                .groq()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => {
                    summary.push(LiveCheckOutcome::pass(
                        "build",
                        format!("model={}", config.default_model),
                    ));
                    summary.push(test_model_listing(&client, config.name).await);
                }
                Err(err) => summary.push(LiveCheckOutcome::fail(
                    "build",
                    format!("Groq client build failed: {err}"),
                )),
            }
        }
        "xAI" => {
            let api_key = env::var(config.api_key_env).unwrap();
            match Siumai::builder()
                .xai()
                .api_key(api_key)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => {
                    summary.push(LiveCheckOutcome::pass(
                        "build",
                        format!("model={}", config.default_model),
                    ));
                    summary.push(test_model_listing(&client, config.name).await);
                }
                Err(err) => summary.push(LiveCheckOutcome::fail(
                    "build",
                    format!("xAI client build failed: {err}"),
                )),
            }
        }
        "Ollama" => {
            let base_url = env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            if let Err(err) = probe_ollama(&base_url).await {
                summary.push(LiveCheckOutcome::skip(
                    "availability",
                    format!("Ollama not reachable at {base_url}: {err}"),
                ));
                return summary;
            }
            match Siumai::builder()
                .ollama()
                .base_url(&base_url)
                .model(config.default_model)
                .build()
                .await
            {
                Ok(client) => {
                    summary.push(LiveCheckOutcome::pass(
                        "build",
                        format!("model={} base_url={base_url}", config.default_model),
                    ));
                    summary.push(test_model_listing(&client, config.name).await);
                }
                Err(err) => summary.push(LiveCheckOutcome::fail(
                    "build",
                    format!("Ollama client build failed: {err}"),
                )),
            }
        }
        _ => {
            summary.push(LiveCheckOutcome::fail(
                "provider",
                format!("Unknown provider: {}", config.name),
            ));
        }
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run_single_provider_integration(index: usize) {
        let configs = get_provider_configs();
        let config = &configs[index];

        if !is_provider_available(config) {
            println!(
                "[skip] Skipping {} test: {} not set",
                config.name, config.api_key_env
            );
            return;
        }

        println!("[info] Testing {} provider...", config.name);
        let summary = test_provider_integration(config).await;
        assert_provider_summary_clean(&summary);
    }

    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_openai_integration() {
        run_single_provider_integration(0).await;
    }

    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_anthropic_integration() {
        run_single_provider_integration(1).await;
    }

    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_gemini_integration() {
        run_single_provider_integration(2).await;
    }

    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_deepseek_integration() {
        run_single_provider_integration(3).await;
    }

    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_openrouter_integration() {
        run_single_provider_integration(4).await;
    }

    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_groq_integration() {
        run_single_provider_integration(5).await;
    }

    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_xai_integration() {
        run_single_provider_integration(6).await;
    }

    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_ollama_integration() {
        run_single_provider_integration(7).await;
    }

    /// Run all available provider tests
    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_all_available_providers() {
        println!(" Running integration tests for all available providers...\n");

        let configs = get_provider_configs();
        let mut summaries = Vec::new();
        let mut skipped_providers = Vec::new();

        for config in &configs {
            if is_provider_available(config) {
                println!("[info] Testing {} provider...", config.name);
                let summary = test_provider_integration(config).await;
                summary.print_summary();
                summaries.push(summary);
            } else {
                skipped_providers.push(config.name);
                println!("[skip] Skipping {} (no API key)", config.name);
            }
        }

        print_manual_suite_summary("provider integration", &summaries, &skipped_providers);

        let failures = summaries
            .iter()
            .filter(|summary| summary.has_failures())
            .map(|summary| format!("{} => {}", summary.provider, summary.failure_summary()))
            .collect::<Vec<_>>();

        assert!(
            failures.is_empty(),
            "manual provider integration suite had failing checks: {}",
            failures.join(" || ")
        );
    }

    /// Test model listing capability for all available providers
    #[tokio::test]
    #[ignore = "manual extended live suite; prefer provider_env_smoke_test for env wiring regressions"]
    async fn test_model_listing_all_providers() {
        println!(" Testing model listing for all available providers...");

        let configs = get_provider_configs();
        let mut summaries = Vec::new();
        let mut skipped_providers = Vec::new();

        for config in &configs {
            if is_provider_available(config) {
                println!("\n[info] Testing model listing for {}...", config.name);
                let summary = test_provider_model_listing(config).await;
                summary.print_summary();
                summaries.push(summary);
            } else {
                println!("[skip] Skipping {} (no API key found)", config.name);
                skipped_providers.push(config.name);
            }
        }

        print_manual_suite_summary("model listing", &summaries, &skipped_providers);

        let failures = summaries
            .iter()
            .filter(|summary| summary.has_failures())
            .map(|summary| format!("{} => {}", summary.provider, summary.failure_summary()))
            .collect::<Vec<_>>();

        assert!(
            failures.is_empty(),
            "manual model listing suite had failing checks: {}",
            failures.join(" || ")
        );
    }
}
