//! Advanced Retry Strategy and Error Handling
//!
//! Deprecated: use `retry_api` facade instead. This module will be removed in 0.11.
//!
//! This module provides intelligent retry mechanisms, rate limit handling,
//! and provider failover capabilities for robust AI service integration.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::time::sleep;

use crate::error::LlmError;

/// Retry strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryStrategy {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff strategy
    pub backoff: BackoffStrategy,
    /// Jitter configuration
    pub jitter: JitterConfig,
    /// Retryable error types
    pub retryable_errors: Vec<RetryableErrorType>,
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(60),
            backoff: BackoffStrategy::Exponential { multiplier: 2.0 },
            jitter: JitterConfig::Full,
            retryable_errors: vec![
                RetryableErrorType::NetworkError,
                RetryableErrorType::RateLimitError,
                RetryableErrorType::ServerError,
                RetryableErrorType::TimeoutError,
            ],
        }
    }
}

impl RetryStrategy {
    /// Create a new retry strategy
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum attempts
    pub const fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Set base delay
    pub const fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Set maximum delay
    pub const fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set backoff strategy
    pub const fn with_backoff(mut self, backoff: BackoffStrategy) -> Self {
        self.backoff = backoff;
        self
    }

    /// Set jitter configuration
    pub const fn with_jitter(mut self, jitter: JitterConfig) -> Self {
        self.jitter = jitter;
        self
    }

    /// Add retryable error type
    pub fn with_retryable_error(mut self, error_type: RetryableErrorType) -> Self {
        if !self.retryable_errors.contains(&error_type) {
            self.retryable_errors.push(error_type);
        }
        self
    }

    /// Calculate delay for a given attempt
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let base_delay = match self.backoff {
            BackoffStrategy::Fixed => self.base_delay,
            BackoffStrategy::Linear { increment } => {
                self.base_delay + Duration::from_millis((increment * attempt as f64) as u64)
            }
            BackoffStrategy::Exponential { multiplier } => {
                let delay_ms = self.base_delay.as_millis() as f64 * multiplier.powi(attempt as i32);
                Duration::from_millis(delay_ms as u64)
            }
        };

        let delay = base_delay.min(self.max_delay);
        self.apply_jitter(delay)
    }

    /// Apply jitter to delay
    fn apply_jitter(&self, delay: Duration) -> Duration {
        match self.jitter {
            JitterConfig::None => delay,
            JitterConfig::Full => {
                let jitter_ms = (delay.as_millis() as f64 * rand::random::<f64>()) as u64;
                Duration::from_millis(jitter_ms)
            }
            JitterConfig::Equal => {
                let half_delay = delay.as_millis() / 2;
                let jitter_ms = half_delay + (half_delay as f64 * rand::random::<f64>()) as u128;
                Duration::from_millis(jitter_ms as u64)
            }
            JitterConfig::Decorrelated => {
                // Decorrelated jitter: delay = random(base_delay, delay * 3)
                let min_delay = self.base_delay.as_millis();
                let max_delay = (delay.as_millis() * 3).min(self.max_delay.as_millis());
                let jitter_ms =
                    min_delay + ((max_delay - min_delay) as f64 * rand::random::<f64>()) as u128;
                Duration::from_millis(jitter_ms as u64)
            }
        }
    }

    /// Check if an error is retryable
    pub fn is_retryable(&self, error: &LlmError) -> bool {
        let error_type = RetryableErrorType::from_error(error);
        self.retryable_errors.contains(&error_type)
    }
}

/// Backoff strategy for retries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,
    /// Linear increase in delay
    Linear { increment: f64 },
    /// Exponential backoff
    Exponential { multiplier: f64 },
}

/// Jitter configuration to avoid thundering herd
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JitterConfig {
    /// No jitter
    None,
    /// Full jitter: delay = random(0, delay)
    Full,
    /// Equal jitter: delay = delay/2 + random(0, delay/2)
    Equal,
    /// Decorrelated jitter
    Decorrelated,
}

/// Types of errors that can be retried
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RetryableErrorType {
    /// Network connectivity errors
    NetworkError,
    /// Rate limit errors
    RateLimitError,
    /// Server errors (5xx)
    ServerError,
    /// Request timeout errors
    TimeoutError,
    /// Authentication errors (sometimes retryable)
    AuthenticationError,
    /// Quota exceeded errors
    QuotaExceededError,
    /// Client errors (4xx, not retryable)
    ClientError,
}

impl RetryableErrorType {
    /// Determine error type from `LlmError`
    pub const fn from_error(error: &LlmError) -> Self {
        match error {
            LlmError::HttpError(_) | LlmError::ConnectionError(_) => Self::NetworkError,
            LlmError::RateLimitError(_) => Self::RateLimitError,
            LlmError::TimeoutError(_) => Self::TimeoutError,
            LlmError::AuthenticationError(_) => Self::AuthenticationError,
            LlmError::QuotaExceededError(_) => Self::QuotaExceededError,
            LlmError::InvalidParameter(_) | LlmError::InvalidInput(_) => Self::ClientError,
            LlmError::ApiError { code, .. } => {
                if *code >= 500 {
                    Self::ServerError
                } else if *code == 429 {
                    Self::RateLimitError
                } else if *code == 401 || *code == 403 {
                    Self::AuthenticationError
                } else if *code >= 400 {
                    Self::ClientError
                } else {
                    Self::NetworkError // Default for other HTTP errors
                }
            }
            _ => Self::ClientError, // Default for unknown errors (usually client-side issues)
        }
    }
}

/// Rate limit handler
#[derive(Debug, Clone)]
pub struct RateLimitHandler {
    /// Rate limit configuration
    config: RateLimitConfig,
    /// Current rate limit state
    state: RateLimitState,
}

impl RateLimitHandler {
    /// Create a new rate limit handler
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            state: RateLimitState::default(),
        }
    }

    /// Handle rate limit response
    pub async fn handle_rate_limit(&mut self, error: &LlmError) -> Result<(), LlmError> {
        match error {
            LlmError::RateLimitError(message) => {
                // Extract retry-after header if available
                let retry_after = self.extract_retry_after(message);
                let delay = retry_after.unwrap_or(self.config.default_delay);

                self.state.last_rate_limit = Some(Instant::now());
                self.state.consecutive_rate_limits += 1;

                // Apply exponential backoff for consecutive rate limits
                let backoff_delay = delay * 2_u32.pow(self.state.consecutive_rate_limits.min(5));
                let final_delay = backoff_delay.min(self.config.max_delay);

                sleep(final_delay).await;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Extract retry-after duration from error message
    fn extract_retry_after(&self, message: &str) -> Option<Duration> {
        // Try to parse "retry after X seconds" patterns
        if let Some(seconds_str) = message.split("retry after ").nth(1)
            && let Some(seconds_str) = seconds_str.split(' ').next()
            && let Ok(seconds) = seconds_str.parse::<u64>()
        {
            return Some(Duration::from_secs(seconds));
        }

        // Try to parse "Retry-After: X" patterns
        if let Some(seconds_str) = message.split("Retry-After: ").nth(1)
            && let Some(seconds_str) = seconds_str.split('\n').next()
            && let Ok(seconds) = seconds_str.trim().parse::<u64>()
        {
            return Some(Duration::from_secs(seconds));
        }

        None
    }

    /// Reset rate limit state on successful request
    pub const fn reset_on_success(&mut self) {
        self.state.consecutive_rate_limits = 0;
    }
}

/// Rate limit configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Default delay when no retry-after is specified
    pub default_delay: Duration,
    /// Maximum delay for rate limit backoff
    pub max_delay: Duration,
    /// Whether to respect retry-after headers
    pub respect_retry_after: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            default_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(300), // 5 minutes
            respect_retry_after: true,
        }
    }
}

/// Rate limit state
#[derive(Debug, Clone, Default)]
pub struct RateLimitState {
    /// Last time a rate limit was encountered
    pub last_rate_limit: Option<Instant>,
    /// Number of consecutive rate limits
    pub consecutive_rate_limits: u32,
}

/// Retry executor
pub struct RetryExecutor {
    /// Retry strategy
    strategy: RetryStrategy,
    /// Rate limit handler
    rate_limit_handler: Option<RateLimitHandler>,
}

impl RetryExecutor {
    /// Create a new retry executor
    pub const fn new(strategy: RetryStrategy) -> Self {
        Self {
            strategy,
            rate_limit_handler: None,
        }
    }

    /// Enable rate limit handling
    pub fn with_rate_limit_handler(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit_handler = Some(RateLimitHandler::new(config));
        self
    }

    /// Execute a function with retry logic
    pub async fn execute<F, Fut, T>(&mut self, mut operation: F) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, LlmError>>,
    {
        let mut last_error = None;

        for attempt in 0..self.strategy.max_attempts {
            match operation().await {
                Ok(result) => {
                    // Reset rate limit state on success
                    if let Some(ref mut handler) = self.rate_limit_handler {
                        handler.reset_on_success();
                    }
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error.clone());

                    // Check if error is retryable
                    if !self.strategy.is_retryable(&error) {
                        return Err(error);
                    }

                    // Handle rate limits
                    if let Some(ref mut handler) = self.rate_limit_handler {
                        handler.handle_rate_limit(&error).await?;
                    }

                    // Don't delay after the last attempt
                    if attempt < self.strategy.max_attempts - 1 {
                        let delay = self.strategy.calculate_delay(attempt);
                        sleep(delay).await;
                    }
                }
            }
        }

        // Return the last error if all attempts failed
        Err(last_error
            .unwrap_or_else(|| LlmError::InternalError("All retry attempts failed".to_string())))
    }
}

/// Provider failover configuration
#[derive(Debug, Clone)]
pub struct FailoverConfig {
    /// List of provider priorities (higher number = higher priority)
    pub provider_priorities: std::collections::HashMap<String, u32>,
    /// Maximum failures before marking a provider as unhealthy
    pub max_failures: u32,
    /// Time window for failure counting
    pub failure_window: Duration,
    /// Cooldown period before retrying a failed provider
    pub cooldown_period: Duration,
    /// Whether to enable automatic failover
    pub auto_failover: bool,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            provider_priorities: std::collections::HashMap::new(),
            max_failures: 3,
            failure_window: Duration::from_secs(300), // 5 minutes
            cooldown_period: Duration::from_secs(60), // 1 minute
            auto_failover: true,
        }
    }
}

/// Provider health tracker
#[derive(Debug, Clone)]
pub struct ProviderHealth {
    /// Provider name
    pub name: String,
    /// Number of recent failures
    pub failure_count: u32,
    /// Last failure time
    pub last_failure: Option<Instant>,
    /// Whether the provider is currently healthy
    pub is_healthy: bool,
    /// Last successful request time
    pub last_success: Option<Instant>,
}

impl ProviderHealth {
    pub const fn new(name: String) -> Self {
        Self {
            name,
            failure_count: 0,
            last_failure: None,
            is_healthy: true,
            last_success: None,
        }
    }

    /// Record a failure
    pub fn record_failure(&mut self, config: &FailoverConfig) {
        self.failure_count += 1;
        self.last_failure = Some(Instant::now());

        if self.failure_count >= config.max_failures {
            self.is_healthy = false;
        }
    }

    /// Record a success
    pub fn record_success(&mut self) {
        self.failure_count = 0;
        self.last_success = Some(Instant::now());
        self.is_healthy = true;
    }

    /// Check if provider should be retried
    pub fn should_retry(&self, config: &FailoverConfig) -> bool {
        if self.is_healthy {
            return true;
        }

        if let Some(last_failure) = self.last_failure {
            last_failure.elapsed() >= config.cooldown_period
        } else {
            true
        }
    }
}

/// Failover manager
pub struct FailoverManager {
    /// Failover configuration
    config: FailoverConfig,
    /// Provider health tracking
    provider_health: std::collections::HashMap<String, ProviderHealth>,
}

impl FailoverManager {
    /// Create a new failover manager
    pub fn new(config: FailoverConfig) -> Self {
        Self {
            config,
            provider_health: std::collections::HashMap::new(),
        }
    }

    /// Get the next available provider
    pub fn get_next_provider(&mut self, providers: &[String]) -> Option<String> {
        if !self.config.auto_failover {
            return providers.first().cloned();
        }

        // Sort providers by priority and health
        let mut available_providers: Vec<_> = providers
            .iter()
            .filter_map(|name| {
                let health = self
                    .provider_health
                    .entry(name.clone())
                    .or_insert_with(|| ProviderHealth::new(name.clone()));

                if health.should_retry(&self.config) {
                    let priority = self
                        .config
                        .provider_priorities
                        .get(name)
                        .copied()
                        .unwrap_or(0);
                    Some((name.clone(), priority, health.is_healthy))
                } else {
                    None
                }
            })
            .collect();

        // Sort by: healthy first, then by priority (descending)
        available_providers.sort_by(|a, b| {
            b.2.cmp(&a.2) // Healthy first
                .then_with(|| b.1.cmp(&a.1)) // Then by priority
        });

        available_providers.first().map(|(name, _, _)| name.clone())
    }

    /// Record a provider failure
    pub fn record_failure(&mut self, provider: &str) {
        let health = self
            .provider_health
            .entry(provider.to_string())
            .or_insert_with(|| ProviderHealth::new(provider.to_string()));

        health.record_failure(&self.config);
    }

    /// Record a provider success
    pub fn record_success(&mut self, provider: &str) {
        let health = self
            .provider_health
            .entry(provider.to_string())
            .or_insert_with(|| ProviderHealth::new(provider.to_string()));

        health.record_success();
    }

    /// Get provider health status
    pub fn get_provider_health(&self, provider: &str) -> Option<&ProviderHealth> {
        self.provider_health.get(provider)
    }

    /// Get all provider health statuses
    pub const fn get_all_health(&self) -> &std::collections::HashMap<String, ProviderHealth> {
        &self.provider_health
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_strategy_delay_calculation() {
        let strategy = RetryStrategy::new()
            .with_base_delay(Duration::from_millis(100))
            .with_backoff(BackoffStrategy::Exponential { multiplier: 2.0 })
            .with_jitter(JitterConfig::None);

        let delay1 = strategy.calculate_delay(0);
        let delay2 = strategy.calculate_delay(1);
        let delay3 = strategy.calculate_delay(2);

        assert_eq!(delay1, Duration::from_millis(100));
        assert_eq!(delay2, Duration::from_millis(200));
        assert_eq!(delay3, Duration::from_millis(400));
    }

    #[test]
    fn test_retryable_error_detection() {
        let strategy = RetryStrategy::default();

        assert!(strategy.is_retryable(&LlmError::HttpError("Connection failed".to_string())));
        assert!(strategy.is_retryable(&LlmError::RateLimitError("Rate limited".to_string())));
        assert!(strategy.is_retryable(&LlmError::TimeoutError("Request timeout".to_string())));
        assert!(!strategy.is_retryable(&LlmError::InvalidParameter("Bad param".to_string())));
    }

    #[test]
    fn test_rate_limit_retry_after_extraction() {
        let handler = RateLimitHandler::new(RateLimitConfig::default());

        let delay1 = handler.extract_retry_after("Rate limited. Please retry after 30 seconds.");
        assert_eq!(delay1, Some(Duration::from_secs(30)));

        let delay2 = handler.extract_retry_after("HTTP 429: Retry-After: 60");
        assert_eq!(delay2, Some(Duration::from_secs(60)));

        let delay3 = handler.extract_retry_after("No retry info");
        assert_eq!(delay3, None);
    }

    #[test]
    fn test_provider_health_tracking() {
        let config = FailoverConfig::default();
        let mut health = ProviderHealth::new("test-provider".to_string());

        assert!(health.is_healthy);
        assert_eq!(health.failure_count, 0);

        // Record failures
        health.record_failure(&config);
        health.record_failure(&config);
        assert!(health.is_healthy); // Still healthy

        health.record_failure(&config);
        assert!(!health.is_healthy); // Now unhealthy

        // Record success should reset
        health.record_success();
        assert!(health.is_healthy);
        assert_eq!(health.failure_count, 0);
    }

    #[test]
    fn test_failover_manager() {
        let mut config = FailoverConfig::default();
        config
            .provider_priorities
            .insert("provider1".to_string(), 10);
        config
            .provider_priorities
            .insert("provider2".to_string(), 5);

        let mut manager = FailoverManager::new(config);
        let providers = vec!["provider1".to_string(), "provider2".to_string()];

        // Should return highest priority provider first
        let next = manager.get_next_provider(&providers);
        assert_eq!(next, Some("provider1".to_string()));

        // Mark provider1 as failed multiple times
        manager.record_failure("provider1");
        manager.record_failure("provider1");
        manager.record_failure("provider1");

        // Should now return provider2
        let next = manager.get_next_provider(&providers);
        assert_eq!(next, Some("provider2".to_string()));
    }
}
