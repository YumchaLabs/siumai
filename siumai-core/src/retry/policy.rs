//! Retry Mechanism Module
//!
//! This module provides comprehensive retry functionality for LLM API calls,
//! including exponential backoff and jitter.

use rand::Rng;
use std::time::Duration;
use tokio::time::sleep;

use crate::error::LlmError;

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier (for exponential backoff)
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delays
    pub use_jitter: bool,
    /// Maximum jitter percentage (0.0 to 1.0)
    pub jitter_factor: f64,
    /// Custom retry condition function
    pub retry_condition: Option<fn(&LlmError) -> bool>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            use_jitter: true,
            jitter_factor: 0.1,
            retry_condition: None,
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum attempts
    pub const fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Set initial delay
    pub const fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set maximum delay
    pub const fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set backoff multiplier
    pub const fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Enable or disable jitter
    pub const fn with_jitter(mut self, use_jitter: bool) -> Self {
        self.use_jitter = use_jitter;
        self
    }

    /// Set jitter factor
    pub const fn with_jitter_factor(mut self, factor: f64) -> Self {
        self.jitter_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Set custom retry condition
    pub fn with_retry_condition(mut self, condition: fn(&LlmError) -> bool) -> Self {
        self.retry_condition = Some(condition);
        self
    }

    /// Check if an error should be retried
    pub fn should_retry(&self, error: &LlmError) -> bool {
        if let Some(condition) = self.retry_condition {
            condition(error)
        } else {
            error.is_retryable()
        }
    }

    /// Calculate delay for a given attempt
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let base_delay =
            self.initial_delay.as_millis() as f64 * self.backoff_multiplier.powi(attempt as i32);

        let delay = Duration::from_millis(base_delay as u64).min(self.max_delay);

        if self.use_jitter {
            self.add_jitter(delay)
        } else {
            delay
        }
    }

    /// Add jitter to a delay
    fn add_jitter(&self, delay: Duration) -> Duration {
        let mut rng = rand::thread_rng();
        let jitter_range = delay.as_millis() as f64 * self.jitter_factor;
        let jitter = rng.gen_range(-jitter_range..=jitter_range);

        let new_delay = delay.as_millis() as f64 + jitter;
        Duration::from_millis(new_delay.max(0.0) as u64)
    }
}

/// Retry executor that handles the actual retry logic
pub struct RetryExecutor {
    policy: RetryPolicy,
}

impl RetryExecutor {
    /// Create a new retry executor
    pub const fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }

    /// Execute a function with retry logic
    pub async fn execute<F, Fut, T>(&self, mut operation: F) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, LlmError>>,
    {
        let mut last_error = None;

        for attempt in 0..self.policy.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error.clone());

                    // Check if we should retry
                    if !self.policy.should_retry(&error) {
                        return Err(error);
                    }

                    // If this is the last attempt, don't wait
                    if attempt == self.policy.max_attempts - 1 {
                        break;
                    }

                    // Calculate and apply delay
                    let delay = self.policy.calculate_delay(attempt);
                    sleep(delay).await;
                }
            }
        }

        // Return the last error if all attempts failed
        Err(last_error.unwrap_or_else(|| {
            LlmError::InternalError("Retry executor failed without error".to_string())
        }))
    }

    /// Execute with custom error handling
    pub async fn execute_with_handler<F, Fut, T, H>(
        &self,
        mut operation: F,
        mut error_handler: H,
    ) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, LlmError>>,
        H: FnMut(&LlmError, u32) -> bool, // Returns true to continue retrying
    {
        let mut last_error = None;

        for attempt in 0..self.policy.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error.clone());

                    // Check with custom error handler
                    if !error_handler(&error, attempt) {
                        return Err(error);
                    }

                    // Check if we should retry according to policy
                    if !self.policy.should_retry(&error) {
                        return Err(error);
                    }

                    // If this is the last attempt, don't wait
                    if attempt == self.policy.max_attempts - 1 {
                        break;
                    }

                    // Calculate and apply delay
                    let delay = self.policy.calculate_delay(attempt);
                    sleep(delay).await;
                }
            }
        }

        // Return the last error if all attempts failed
        Err(last_error.unwrap_or_else(|| {
            LlmError::InternalError("Retry executor failed without error".to_string())
        }))
    }
}

/// Convenience function to retry an operation with default policy
pub async fn retry_with_default<F, Fut, T>(operation: F) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, LlmError>>,
{
    let executor = RetryExecutor::new(RetryPolicy::default());
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

        let policy = RetryPolicy::new().with_max_attempts(3);
        let executor = RetryExecutor::new(policy);

        let result = executor
            .execute(|| {
                let counter = counter_clone.clone();
                async move {
                    let count = counter.fetch_add(1, Ordering::SeqCst);
                    if count == 0 {
                        Err(LlmError::ApiError {
                            code: 500,
                            message: "Server error".to_string(),
                            details: None,
                        })
                    } else {
                        Ok("success")
                    }
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retry_exhaustion() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let policy = RetryPolicy::new().with_max_attempts(2);
        let executor = RetryExecutor::new(policy);

        let result: Result<(), LlmError> = executor
            .execute(|| {
                let counter = counter_clone.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err(LlmError::ApiError {
                        code: 500,
                        message: "Server error".to_string(),
                        details: None,
                    })
                }
            })
            .await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_delay_calculation() {
        let policy = RetryPolicy::new()
            .with_initial_delay(Duration::from_millis(100))
            .with_backoff_multiplier(2.0)
            .with_jitter(false);

        assert_eq!(policy.calculate_delay(0), Duration::from_millis(100));
        assert_eq!(policy.calculate_delay(1), Duration::from_millis(200));
        assert_eq!(policy.calculate_delay(2), Duration::from_millis(400));
    }
}
