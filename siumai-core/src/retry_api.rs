//! Public Retry API Facade
//!
//! This module provides a unified, recommended retry API.
//!
//! - Simple defaults: `retry` uses backoff-based executor
//! - Opt-in control: use `RetryOptions` to select backend and configuration
//! - Builder/registry integration: `RetryOptions` can be attached to providers via
//!   `Siumai::builder().with_retry(...)` or `RegistryOptions.retry_options`
//!
//! Example
//! ```rust,no_run
//! use siumai::retry_api::{retry, retry_with, RetryOptions, RetryBackend};
//!
//! # async fn do_work() -> Result<String, siumai::LlmError> { Ok("ok".into()) }
//! # async fn example() -> Result<(), siumai::LlmError> {
//! // Recommended default (backoff-based)
//! let result = retry(|| do_work()).await?;
//!
//! // Explicit backend selection (policy-based)
//! let options = RetryOptions::policy_default().with_max_attempts(5);
//! let result = retry_with(|| do_work(), options).await?;
//! # Ok(())
//! # }
//! ```

use crate::error::LlmError;
use reqwest::header::HeaderMap;

// Re-export core types for convenience
pub use crate::retry::BackoffRetryExecutor;
pub use crate::retry::RetryPolicy;

/// Retry backend selector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetryBackend {
    /// Backoff crate-based executor (recommended default)
    #[default]
    Backoff,
    /// Simple policy-based executor (`retry.rs`)
    Policy,
}

/// Unified retry options
#[derive(Debug, Clone)]
pub struct RetryOptions {
    pub backend: RetryBackend,
    /// Optional backoff executor override (Backoff backend only).
    pub backoff_executor: Option<BackoffRetryExecutor>,
    // Policy-based options
    pub policy: Option<RetryPolicy>,
    /// Whether to retry 401 Unauthorized errors (useful for token refresh scenarios)
    pub retry_401: bool,
    /// Whether the request is idempotent (safe to retry without side effects)
    pub idempotent: bool,
}

impl Default for RetryOptions {
    fn default() -> Self {
        Self {
            backend: RetryBackend::Backoff,
            backoff_executor: None,
            policy: None,
            retry_401: true,  // Default to true for backward compatibility
            idempotent: true, // Default to true (most LLM requests are idempotent)
        }
    }
}

impl RetryOptions {
    /// Use default backoff backend
    pub fn backoff() -> Self {
        Self::default()
    }

    /// Use backoff backend with a custom executor.
    pub fn with_backoff_executor(mut self, executor: BackoffRetryExecutor) -> Self {
        self.backoff_executor = Some(executor);
        self
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

    /// Set whether to retry 401 errors
    pub fn with_retry_401(mut self, retry_401: bool) -> Self {
        self.retry_401 = retry_401;
        self
    }

    /// Set whether the request is idempotent
    pub fn with_idempotent(mut self, idempotent: bool) -> Self {
        self.idempotent = idempotent;
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
    crate::retry::retry_with_backoff(operation).await
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
            if let Some(executor) = options.backoff_executor.as_ref() {
                executor.execute(operation).await
            } else {
                crate::retry::retry_with_backoff(operation).await
            }
        }
        RetryBackend::Policy => {
            let policy = options.policy.unwrap_or_default();
            let executor = crate::retry::RetryExecutor::new(policy);
            executor.execute(operation).await
        }
    }
}

/// Retry only when options are provided.
///
/// This is a small helper to keep call sites consistent when retry is optional
/// (e.g. per-client or per-request policy injection).
pub async fn maybe_retry<F, Fut, T>(
    options: Option<RetryOptions>,
    operation: F,
) -> Result<T, LlmError>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<T, LlmError>> + Send,
    T: Send,
{
    if let Some(opts) = options {
        retry_with(operation, opts).await
    } else {
        operation().await
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

    // Extract common request/trace identifiers to aid debugging (best-effort)
    fn header_val(headers: &HeaderMap, name: &str) -> Option<String> {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    }
    let id_keys = [
        "x-request-id",
        "x-response-id",
        "x-openai-request-id",
        "x-trace-id",
        "traceparent",
        "x-correlation-id",
        "x-goog-request-id",
    ];
    let request_ids_vec: Vec<String> = id_keys
        .iter()
        .filter_map(|k| header_val(headers, k).map(|v| format!("{}={}", k, v)))
        .collect();
    let ids_suffix = if request_ids_vec.is_empty() {
        String::new()
    } else {
        format!(" ids=[{}]", request_ids_vec.join(","))
    };
    // Limit body sample size to avoid noisy logs
    let body_sample = body_text.chars().take(200).collect::<String>();

    // 429 Too Many Requests → RateLimit with optional Retry-After hint
    if status == 429 {
        let retry_after = headers
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        return LlmError::RateLimitError(format!(
            "provider={} http=429 retry_after={}{} body_sample={}",
            provider_id, retry_after, ids_suffix, body_sample
        ));
    }

    // 401 → Authentication
    if status == 401 {
        return LlmError::AuthenticationError(format!(
            "provider={} unauthorized{} body_sample={}",
            provider_id, ids_suffix, body_sample
        ));
    }

    // 404 → NotFound
    if status == 404 {
        return LlmError::NotFound(format!(
            "provider={} http=404{} body_sample={}",
            provider_id, ids_suffix, body_sample
        ));
    }

    // 413/415 → InvalidInput (common for file/image/multipart APIs)
    if status == 413 {
        return LlmError::InvalidInput(format!(
            "provider={} http=413 payload too large{} body_sample={}",
            provider_id, ids_suffix, body_sample
        ));
    }
    if status == 415 {
        return LlmError::InvalidInput(format!(
            "provider={} http=415 unsupported media type{} body_sample={}",
            provider_id, ids_suffix, body_sample
        ));
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

    // Generic 403/400 mapping when not a quota/rate envelope.
    if status == 403 {
        return LlmError::AuthenticationError(format!(
            "provider={} forbidden{} body_sample={}",
            provider_id, ids_suffix, body_sample
        ));
    }
    if status == 400 {
        return LlmError::InvalidInput(format!(
            "provider={} bad request{} body_sample={}",
            provider_id, ids_suffix, body_sample
        ));
    }

    // 5xx → Server error (retryable via is_retryable())
    if (500..=599).contains(&status) {
        return LlmError::api_error(status, fallback_message.unwrap_or("server error"));
    }

    // Fallback to ApiError with original status and body snippet
    let msg = if let Some(fallback) = fallback_message {
        fallback.to_string()
    } else if body_text.trim().is_empty() {
        "api error".to_string()
    } else {
        body_sample.clone()
    };
    // Build structured details for ApiError
    let details = match serde_json::from_str::<serde_json::Value>(body_text) {
        Ok(json) => serde_json::json!({
            "status": status,
            "provider": provider_id,
            "response": json,
            "request_ids": request_ids_vec,
        }),
        Err(_) => serde_json::json!({
            "status": status,
            "provider": provider_id,
            "raw": body_text,
            "request_ids": request_ids_vec,
        }),
    };
    LlmError::api_error_with_details(status, msg, details)
}

#[cfg(test)]
mod tests {
    use super::*;
    // no extra imports
    use reqwest::header::HeaderMap;

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

    #[test]
    fn classify_http_error_uses_fallback_message_for_non_json_body() {
        let headers = HeaderMap::new();
        let err = classify_http_error(
            "openai",
            502,
            "<html>bad gateway</html>",
            &headers,
            Some("Bad Gateway"),
        );
        match err {
            LlmError::ApiError {
                code,
                message,
                details: _,
            } => {
                assert_eq!(code, 502);
                assert_eq!(message, "Bad Gateway");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
