//! Public Retry API Facade
//!
//! This module provides a unified, recommended retry API while keeping
//! existing internal implementations (`retry`, `retry_backoff`, `retry_strategy`).
//!
//! - Simple defaults: `retry` and `retry_for_provider` use backoff-based executor
//! - Opt-in control: use `RetryOptions` to select backend and configuration
//! - Backwards compatibility: original modules remain available
//!
//! Example
//! ```rust,no_run
//! use siumai::retry_api::{retry, retry_for_provider, RetryOptions, RetryBackend};
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
//! let result = retry_api::retry_with(|| do_work(), options).await?;
//! # Ok(())
//! # }
//! ```

use crate::error::LlmError;
use crate::types::ProviderType;

// Re-export core types for convenience
pub use crate::retry::RetryPolicy;
pub use crate::retry_backoff::BackoffRetryExecutor;
#[allow(deprecated)]
pub use crate::retry_strategy::{FailoverConfig, FailoverManager, RateLimitConfig, RetryStrategy};

/// Retry backend selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryBackend {
    /// Backoff crate-based executor (recommended default)
    Backoff,
    /// Simple policy-based executor (`retry.rs`)
    Policy,
    /// Advanced strategy executor with rate limit/failover support (`retry_strategy.rs`)
    Strategy,
}

impl Default for RetryBackend {
    fn default() -> Self {
        Self::Backoff
    }
}

/// Unified retry options
#[allow(deprecated)]
#[derive(Debug, Clone)]
pub struct RetryOptions {
    pub backend: RetryBackend,
    pub provider: Option<ProviderType>,
    // Policy-based options
    pub policy: Option<RetryPolicy>,
    // Strategy-based options
    pub strategy: Option<RetryStrategy>,
    pub rate_limit: Option<RateLimitConfig>,
}

impl Default for RetryOptions {
    fn default() -> Self {
        Self {
            backend: RetryBackend::Backoff,
            provider: None,
            policy: None,
            strategy: None,
            rate_limit: None,
        }
    }
}

#[allow(deprecated)]
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

    /// Use strategy-based backend with provided strategy
    pub fn with_strategy(strategy: RetryStrategy) -> Self {
        Self {
            backend: RetryBackend::Strategy,
            strategy: Some(strategy),
            ..Default::default()
        }
    }

    /// Attach a rate limit config (strategy backend)
    pub fn with_rate_limit(mut self, cfg: RateLimitConfig) -> Self {
        self.rate_limit = Some(cfg);
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
#[allow(deprecated)]
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
        RetryBackend::Strategy => {
            let mut executor =
                crate::retry_strategy::RetryExecutor::new(options.strategy.unwrap_or_default());
            if let Some(cfg) = options.rate_limit {
                executor = executor.with_rate_limit_handler(cfg);
            }
            executor.execute(operation).await
        }
    }
}
