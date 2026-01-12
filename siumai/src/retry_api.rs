//! Unified retry facade for `siumai`.
//!
//! This module re-exports the provider-agnostic retry API from `siumai-core`, and adds
//! provider-aware convenience defaults at the facade layer (not in `siumai-core`).

pub use siumai_core::retry_api::*;

use backoff::{ExponentialBackoff, ExponentialBackoffBuilder};
use siumai_core::error::LlmError;
use siumai_core::types::ProviderType;
use std::time::Duration;

/// Build a provider-tuned backoff executor.
///
/// Note: This is intentionally implemented in the `siumai` facade crate so that
/// `siumai-core` stays provider-agnostic.
pub fn backoff_executor_for_provider(provider: &ProviderType) -> BackoffRetryExecutor {
    let backoff = match provider {
        ProviderType::OpenAi | ProviderType::XAI | ProviderType::Groq | ProviderType::MiniMaxi => {
            openai_compat_backoff()
        }
        ProviderType::Anthropic => anthropic_backoff(),
        ProviderType::Gemini => google_backoff(),
        ProviderType::Ollama => ollama_backoff(),
        ProviderType::Custom(_) => {
            return BackoffRetryExecutor::new();
        }
    };

    BackoffRetryExecutor::with_backoff(backoff)
}

/// Provider-aware backoff options.
pub fn backoff_options_for_provider(provider: &ProviderType) -> RetryOptions {
    RetryOptions::backoff().with_backoff_executor(backoff_executor_for_provider(provider))
}

/// Provider-aware retry (backoff-based).
pub async fn retry_for_provider<F, Fut, T>(
    provider: &ProviderType,
    operation: F,
) -> Result<T, LlmError>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
    T: Send,
{
    retry_with(operation, backoff_options_for_provider(provider)).await
}

fn openai_compat_backoff() -> ExponentialBackoff {
    ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_millis(1000))
        .with_max_interval(Duration::from_secs(60))
        .with_multiplier(2.0)
        .with_max_elapsed_time(Some(Duration::from_secs(300)))
        .build()
}

fn anthropic_backoff() -> ExponentialBackoff {
    ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_millis(1000))
        .with_max_interval(Duration::from_secs(60))
        .with_multiplier(1.5)
        .with_max_elapsed_time(Some(Duration::from_secs(300)))
        .build()
}

fn google_backoff() -> ExponentialBackoff {
    ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_millis(1000))
        .with_max_interval(Duration::from_secs(60))
        .with_multiplier(1.5)
        .with_max_elapsed_time(Some(Duration::from_secs(300)))
        .build()
}

fn ollama_backoff() -> ExponentialBackoff {
    ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_millis(500))
        .with_max_interval(Duration::from_secs(30))
        .with_multiplier(1.5)
        .with_max_elapsed_time(Some(Duration::from_secs(180)))
        .build()
}
