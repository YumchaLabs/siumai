//! Professional Retry Mechanism using backoff crate
//!
//! This module provides a professional retry implementation using the `backoff` crate,
//! which is more robust and feature-complete than our custom implementation.

use backoff::{ExponentialBackoff, ExponentialBackoffBuilder};
use std::time::Duration;

use crate::error::LlmError;
use crate::types::ProviderType;

/// Professional retry executor using backoff crate
pub struct BackoffRetryExecutor {
    backoff: ExponentialBackoff,
}

impl BackoffRetryExecutor {
    /// Create a new retry executor with default exponential backoff
    pub fn new() -> Self {
        Self {
            backoff: ExponentialBackoff::default(),
        }
    }

    /// Create a retry executor with custom backoff configuration
    pub fn with_backoff(backoff: ExponentialBackoff) -> Self {
        Self { backoff }
    }

    /// Create a retry executor for a specific provider
    pub fn for_provider(provider: &ProviderType) -> Self {
        let backoff = match provider {
            ProviderType::OpenAi => Self::openai_backoff(),
            ProviderType::Anthropic => Self::anthropic_backoff(),
            ProviderType::Gemini => Self::google_backoff(),
            ProviderType::Ollama => Self::ollama_backoff(),
            ProviderType::XAI => Self::openai_backoff(), // xAI uses OpenAI-compatible API
            ProviderType::Groq => Self::openai_backoff(), // Groq uses OpenAI-compatible API
            ProviderType::MiniMaxi => Self::openai_backoff(), // MiniMaxi uses OpenAI-compatible API
            ProviderType::Custom(_) => Self::default_backoff(),
        };

        Self { backoff }
    }

    /// Execute an operation with retry logic
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T, LlmError>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
        T: Send,
    {
        backoff::future::retry(self.backoff.clone(), || async {
            match operation().await {
                Ok(result) => Ok(result),
                Err(error) => {
                    if Self::is_retryable(&error) {
                        Err(backoff::Error::Transient {
                            err: error,
                            retry_after: None,
                        })
                    } else {
                        Err(backoff::Error::Permanent(error))
                    }
                }
            }
        })
        .await
    }

    /// Check if an error is retryable
    fn is_retryable(error: &LlmError) -> bool {
        match error {
            LlmError::ApiError { code, .. } => {
                // Retry on rate limits and server errors
                matches!(*code, 429 | 500..=599)
            }
            LlmError::RateLimitError(_) => true,
            LlmError::TimeoutError(_) => true,
            LlmError::ConnectionError(_) => true,
            LlmError::HttpError(_) => true,
            _ => false,
        }
    }

    /// OpenAI-specific backoff configuration
    fn openai_backoff() -> ExponentialBackoff {
        ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_millis(1000))
            .with_max_interval(Duration::from_secs(60))
            .with_multiplier(2.0)
            .with_max_elapsed_time(Some(Duration::from_secs(300))) // 5 minutes total
            .build()
    }

    /// Anthropic-specific backoff configuration
    fn anthropic_backoff() -> ExponentialBackoff {
        ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_millis(1000))
            .with_max_interval(Duration::from_secs(60))
            .with_multiplier(1.5)
            .with_max_elapsed_time(Some(Duration::from_secs(300)))
            .build()
    }

    /// Google-specific backoff configuration
    fn google_backoff() -> ExponentialBackoff {
        ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_millis(1000))
            .with_max_interval(Duration::from_secs(60))
            .with_multiplier(1.5)
            .with_max_elapsed_time(Some(Duration::from_secs(300)))
            .build()
    }

    /// Ollama-specific backoff configuration
    fn ollama_backoff() -> ExponentialBackoff {
        ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_millis(500))
            .with_max_interval(Duration::from_secs(30))
            .with_multiplier(1.5)
            .with_max_elapsed_time(Some(Duration::from_secs(180))) // 3 minutes for local
            .build()
    }

    /// Default backoff configuration
    fn default_backoff() -> ExponentialBackoff {
        ExponentialBackoff::default()
    }
}

impl Default for BackoffRetryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to retry with default backoff
pub async fn retry_with_backoff<F, Fut, T>(operation: F) -> Result<T, LlmError>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
    T: Send,
{
    let executor = BackoffRetryExecutor::new();
    executor.execute(operation).await
}

/// Convenience function to retry with provider-specific backoff
pub async fn retry_for_provider_backoff<F, Fut, T>(
    provider: &ProviderType,
    operation: F,
) -> Result<T, LlmError>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
    T: Send,
{
    let executor = BackoffRetryExecutor::for_provider(provider);
    executor.execute(operation).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_retry_success_on_second_attempt() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let executor = BackoffRetryExecutor::new();

        let result: Result<String, LlmError> = executor
            .execute(|| async {
                let count = counter_clone.fetch_add(1, Ordering::SeqCst);
                if count == 0 {
                    Err(LlmError::RateLimitError("Rate limited".to_string()))
                } else {
                    Ok("Success".to_string())
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Success");
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_permanent_error_no_retry() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let executor = BackoffRetryExecutor::new();

        let result: Result<String, LlmError> = executor
            .execute(|| async {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Err(LlmError::InvalidInput("Bad input".to_string()))
            })
            .await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 1); // No retry for permanent error
    }

    #[tokio::test]
    async fn test_provider_specific_backoff() {
        let executor = BackoffRetryExecutor::for_provider(&ProviderType::OpenAi);

        // Just test that it creates without panicking
        let result: Result<String, LlmError> = executor
            .execute(|| async { Ok("Success".to_string()) })
            .await;

        assert!(result.is_ok());
    }
}
