//! Public Retry API Facade
//!
//! This module provides a unified, recommended retry API.
//!
//! - Simple defaults: `retry` and `retry_for_provider` use backoff-based executor
//! - Opt-in control: use `RetryOptions` to select backend and configuration
//!
//! Example
//! ```rust,no_run
//! use siumai::retry_api::{retry, retry_for_provider, retry_with, RetryOptions, RetryBackend};
//! use siumai::types::ProviderType;
//!
//! # async fn do_work() -> Result<String, siumai::LlmError> { Ok("ok".into()) }
//! # async fn example() -> Result<(), siumai::LlmError> {
//! // Recommended default (backoff-based)
//! let result = retry(|| do_work()).await?;
//!
//! // Provider-aware backoff
//! let result = retry_for_provider(&ProviderType::OpenAi, || do_work()).await?;
//!
//! // Explicit backend selection (policy-based)
//! let options = RetryOptions::policy_default().with_max_attempts(5);
//! let result = retry_with(|| do_work(), options).await?;
//! # Ok(())
//! # }
//! ```

use crate::error::LlmError;
use crate::types::ProviderType;
use reqwest::header::HeaderMap;

// Re-export core types for convenience
pub use crate::retry::RetryPolicy;
pub use crate::retry_backoff::BackoffRetryExecutor;

/// Retry backend selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryBackend {
    /// Backoff crate-based executor (recommended default)
    Backoff,
    /// Simple policy-based executor (`retry.rs`)
    Policy,
}

impl Default for RetryBackend {
    fn default() -> Self {
        Self::Backoff
    }
}

/// Unified retry options
#[derive(Debug, Clone)]
pub struct RetryOptions {
    pub backend: RetryBackend,
    pub provider: Option<ProviderType>,
    // Policy-based options
    pub policy: Option<RetryPolicy>,
}

impl Default for RetryOptions {
    fn default() -> Self {
        Self {
            backend: RetryBackend::Backoff,
            provider: None,
            policy: None,
        }
    }
}

impl RetryOptions {
    /// Use default backoff backend
    pub fn backoff() -> Self {
        Self::default()
    }

    /// Use provider-aware backoff backend
    pub fn backoff_for_provider(provider: ProviderType) -> Self {
        Self {
            backend: RetryBackend::Backoff,
            provider: Some(provider),
            ..Default::default()
        }
    }

    /// Use policy-based backend with default policy
    pub fn policy_default() -> Self {
        Self {
            backend: RetryBackend::Policy,
            policy: Some(crate::retry::RetryPolicy::default()),
            ..Default::default()
        }
    }

    /// Set max attempts for policy backend
    pub fn with_max_attempts(mut self, attempts: u32) -> Self {
        if let Some(policy) = self.policy.take() {
            self.policy = Some(policy.with_max_attempts(attempts));
        }
        self
    }
}

/// Recommended default retry (backoff-based)
pub async fn retry<F, Fut, T>(operation: F) -> Result<T, LlmError>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
    T: Send,
{
    crate::retry_backoff::retry_with_backoff(operation).await
}

/// Recommended provider-aware retry (backoff-based)
pub async fn retry_for_provider<F, Fut, T>(
    provider: &ProviderType,
    operation: F,
) -> Result<T, LlmError>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
    T: Send,
{
    crate::retry_backoff::retry_for_provider_backoff(provider, operation).await
}

/// Retry with explicit options (backend selection)
pub async fn retry_with<F, Fut, T>(operation: F, options: RetryOptions) -> Result<T, LlmError>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
    T: Send,
{
    match options.backend {
        RetryBackend::Backoff => {
            if let Some(provider) = options.provider.as_ref() {
                crate::retry_backoff::retry_for_provider_backoff(provider, operation).await
            } else {
                crate::retry_backoff::retry_with_backoff(operation).await
            }
        }
        RetryBackend::Policy => {
            let policy = options.policy.unwrap_or_default();
            let executor = crate::retry::RetryExecutor::new(policy);
            executor.execute(operation).await
        }
    }
}

/// Classify an HTTP failure into a more specific error type with retry hints.
///
/// This helper inspects the HTTP status code, response body and headers to
/// derive a better-typed LlmError (e.g., RateLimitError / QuotaExceededError)
/// rather than a generic ApiError. It is provider-agnostic with light-weight
/// heuristics, but includes common Vertex/Google patterns.
pub fn classify_http_error(
    provider_id: &str,
    status: u16,
    body_text: &str,
    headers: &HeaderMap,
    fallback_message: Option<&str>,
) -> LlmError {
    let lower = body_text.to_lowercase();

    // 429 Too Many Requests → RateLimit with optional Retry-After hint
    if status == 429 {
        let retry_after = headers
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        return LlmError::RateLimitError(format!(
            "provider={} http=429 retry_after={}",
            provider_id, retry_after
        ));
    }

    // 401 → Authentication
    if status == 401 {
        return LlmError::AuthenticationError(format!("provider={} unauthorized", provider_id));
    }

    // 403/400 with quota/rate patterns → QuotaExceeded or RateLimit
    if status == 403 || status == 400 {
        let quota_like = lower.contains("quota") || lower.contains("exceed");
        let rate_like = lower.contains("rate limit")
            || lower.contains("ratelimit")
            || lower.contains("resource_exhausted")
            || lower.contains("rate_limit_exceeded")
            || lower.contains("ratelimitexceeded")
            || lower.contains("ratelimit exceeded");
        if quota_like {
            return LlmError::QuotaExceededError(format!(
                "provider={} quota exceeded",
                provider_id
            ));
        }
        if rate_like {
            return LlmError::RateLimitError(format!("provider={} rate limited", provider_id));
        }
    }

    // 5xx → Server error (retryable via is_retryable())
    if (500..=599).contains(&status) {
        return LlmError::api_error(status, fallback_message.unwrap_or("server error"));
    }

    // Fallback to ApiError with original status and body snippet
    let msg = if body_text.is_empty() {
        fallback_message.unwrap_or("api error")
    } else {
        body_text
    };
    LlmError::api_error(status, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    // no extra imports

    #[tokio::test]
    async fn retry_with_policy_backend_works() {
        use std::sync::{
            Arc,
            atomic::{AtomicU32, Ordering},
        };
        // Create a policy with small attempts
        let opts = RetryOptions::policy_default().with_max_attempts(2);
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_for_call = attempts.clone();
        let res: Result<(), LlmError> = retry_with(
            || {
                let attempts = attempts_for_call.clone();
                async move {
                    let prev = attempts.fetch_add(1, Ordering::Relaxed);
                    if prev < 1 {
                        Err(LlmError::ApiError {
                            code: 500,
                            message: "server".into(),
                            details: None,
                        })
                    } else {
                        Ok(())
                    }
                }
            },
            opts,
        )
        .await;
        assert!(res.is_ok());
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
    }
}
