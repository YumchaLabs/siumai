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
