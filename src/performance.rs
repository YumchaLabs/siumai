//! Performance Optimization and Monitoring
//!
//! This module provides performance monitoring, optimization utilities,
//! and benchmarking tools for the siumai library.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Performance metrics collector
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Request latency metrics
    pub latency: LatencyMetrics,
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// Error rate metrics
    pub error_rate: ErrorRateMetrics,
    /// Memory usage metrics
    pub memory: MemoryMetrics,
    /// Provider-specific metrics
    pub provider_metrics: HashMap<String, ProviderMetrics>,
}

/// Latency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Average latency
    pub avg_latency: Duration,
    /// Median latency (P50)
    pub p50_latency: Duration,
    /// 95th percentile latency
    pub p95_latency: Duration,
    /// 99th percentile latency
    pub p99_latency: Duration,
    /// Maximum latency
    pub max_latency: Duration,
    /// Minimum latency
    pub min_latency: Duration,
    /// Total number of requests
    pub total_requests: u64,
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            avg_latency: Duration::ZERO,
            p50_latency: Duration::ZERO,
            p95_latency: Duration::ZERO,
            p99_latency: Duration::ZERO,
            max_latency: Duration::ZERO,
            min_latency: Duration::MAX,
            total_requests: 0,
        }
    }
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Requests per second
    pub requests_per_second: f64,
    /// Tokens per second (input)
    pub input_tokens_per_second: f64,
    /// Tokens per second (output)
    pub output_tokens_per_second: f64,
    /// Total requests processed
    pub total_requests: u64,
    /// Total tokens processed (input)
    pub total_input_tokens: u64,
    /// Total tokens processed (output)
    pub total_output_tokens: u64,
    /// Measurement window start time (as timestamp)
    #[serde(with = "instant_serde")]
    pub window_start: Instant,
}

mod instant_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(_instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert to duration since a reference point
        let duration_since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        duration_since_epoch.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let _duration = Duration::deserialize(deserializer)?;
        // Return current instant as we can't reconstruct the original
        Ok(Instant::now())
    }
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            requests_per_second: 0.0,
            input_tokens_per_second: 0.0,
            output_tokens_per_second: 0.0,
            total_requests: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            window_start: Instant::now(),
        }
    }
}

/// Error rate metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRateMetrics {
    /// Total number of errors
    pub total_errors: u64,
    /// Total number of requests
    pub total_requests: u64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Error breakdown by type
    pub error_breakdown: HashMap<String, u64>,
}

impl Default for ErrorRateMetrics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            total_requests: 0,
            error_rate: 0.0,
            error_breakdown: HashMap::new(),
        }
    }
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryMetrics {
    /// Current memory usage in bytes
    pub current_usage: u64,
    /// Peak memory usage in bytes
    pub peak_usage: u64,
    /// Average memory usage in bytes
    pub avg_usage: u64,
    /// Number of allocations
    pub allocations: u64,
    /// Number of deallocations
    pub deallocations: u64,
}

/// Provider-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderMetrics {
    /// Provider name
    pub provider: String,
    /// Request count
    pub request_count: u64,
    /// Success count
    pub success_count: u64,
    /// Error count
    pub error_count: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Rate limit hits
    pub rate_limit_hits: u64,
    /// Cache hits (if applicable)
    pub cache_hits: u64,
    /// Cache misses (if applicable)
    pub cache_misses: u64,
}

impl ProviderMetrics {
    pub const fn new(provider: String) -> Self {
        Self {
            provider,
            request_count: 0,
            success_count: 0,
            error_count: 0,
            avg_response_time: Duration::ZERO,
            rate_limit_hits: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.success_count as f64 / self.request_count as f64
        }
    }

    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total_cache_requests = self.cache_hits + self.cache_misses;
        if total_cache_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_cache_requests as f64
        }
    }
}

/// Performance monitor
#[derive(Clone)]
pub struct PerformanceMonitor {
    /// Metrics storage
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Request timing storage
    request_timings: Arc<RwLock<Vec<Duration>>>,
    /// Configuration (used by extended reporting; kept for future hooks)
    #[allow(dead_code)]
    config: MonitorConfig,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            request_timings: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Record a request start
    pub async fn start_request(&self) -> RequestTimer {
        RequestTimer::new(self.metrics.clone(), self.request_timings.clone())
    }

    /// Record an error
    pub async fn record_error(&self, error_type: &str, provider: Option<&str>) {
        let mut metrics = self.metrics.write().await;
        metrics.error_rate.total_errors += 1;
        metrics.error_rate.total_requests += 1;

        *metrics
            .error_rate
            .error_breakdown
            .entry(error_type.to_string())
            .or_insert(0) += 1;

        metrics.error_rate.error_rate =
            metrics.error_rate.total_errors as f64 / metrics.error_rate.total_requests as f64;

        if let Some(provider) = provider {
            let provider_metrics = metrics
                .provider_metrics
                .entry(provider.to_string())
                .or_insert_with(|| ProviderMetrics::new(provider.to_string()));
            provider_metrics.error_count += 1;
            provider_metrics.request_count += 1;
        }
    }

    /// Record a successful request
    pub async fn record_success(&self, provider: Option<&str>, response_time: Duration) {
        let mut metrics = self.metrics.write().await;
        metrics.error_rate.total_requests += 1;

        metrics.error_rate.error_rate =
            metrics.error_rate.total_errors as f64 / metrics.error_rate.total_requests as f64;

        if let Some(provider) = provider {
            let provider_metrics = metrics
                .provider_metrics
                .entry(provider.to_string())
                .or_insert_with(|| ProviderMetrics::new(provider.to_string()));
            provider_metrics.success_count += 1;
            provider_metrics.request_count += 1;

            // Update average response time
            let total_time = provider_metrics.avg_response_time.as_millis() as u64
                * (provider_metrics.success_count - 1)
                + response_time.as_millis() as u64;
            provider_metrics.avg_response_time =
                Duration::from_millis(total_time / provider_metrics.success_count);
        }
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    /// Update latency metrics (currently used by extended reporting; keep for future hook)
    #[allow(dead_code)]
    async fn update_latency_metrics(&self) {
        let timings = self.request_timings.read().await;
        if timings.is_empty() {
            return;
        }

        let mut sorted_timings = timings.clone();
        sorted_timings.sort();

        let mut metrics = self.metrics.write().await;

        // Calculate percentiles
        let len = sorted_timings.len();
        metrics.latency.p50_latency = sorted_timings[len / 2];
        metrics.latency.p95_latency = sorted_timings[(len * 95) / 100];
        metrics.latency.p99_latency = sorted_timings[(len * 99) / 100];

        // Calculate average
        let total: Duration = sorted_timings.iter().sum();
        metrics.latency.avg_latency = total / len as u32;

        // Min and max
        metrics.latency.min_latency = sorted_timings[0];
        metrics.latency.max_latency = sorted_timings[len - 1];
        metrics.latency.total_requests = len as u64;
    }
}

/// Request timer for measuring individual request performance
pub struct RequestTimer {
    start_time: Instant,
    #[allow(dead_code)] // reserved for live updates in extended reporting
    metrics: Arc<RwLock<PerformanceMetrics>>,
    timings: Arc<RwLock<Vec<Duration>>>,
}

impl RequestTimer {
    fn new(metrics: Arc<RwLock<PerformanceMetrics>>, timings: Arc<RwLock<Vec<Duration>>>) -> Self {
        Self {
            start_time: Instant::now(),
            metrics,
            timings,
        }
    }

    /// Finish timing and record the duration
    pub async fn finish(self) -> Duration {
        let duration = self.start_time.elapsed();

        // Store timing for percentile calculations
        let mut timings = self.timings.write().await;
        timings.push(duration);

        // Keep only recent timings to avoid memory growth
        if timings.len() > 10000 {
            timings.drain(0..5000);
        }

        duration
    }
}

/// Performance monitor configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Whether to enable detailed metrics collection
    pub detailed_metrics: bool,
    /// Maximum number of timing samples to keep
    pub max_timing_samples: usize,
    /// Metrics update interval
    pub update_interval: Duration,
    /// Whether to enable memory tracking
    pub memory_tracking: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            detailed_metrics: true,
            max_timing_samples: 10000,
            update_interval: Duration::from_secs(60),
            memory_tracking: false, // Disabled by default due to overhead
        }
    }
}

// Re-export commonly used types at module level
pub use optimization::{CacheStats, ResponseCache};

/// Performance optimization utilities
pub mod optimization {
    use super::*;

    /// Connection pool configuration for HTTP clients
    #[derive(Debug, Clone)]
    pub struct ConnectionPoolConfig {
        /// Maximum number of idle connections per host
        pub max_idle_per_host: usize,
        /// Maximum total idle connections
        pub max_idle_total: usize,
        /// Connection timeout
        pub connect_timeout: Duration,
        /// Request timeout
        pub request_timeout: Duration,
        /// Keep-alive timeout
        pub keep_alive_timeout: Duration,
    }

    impl Default for ConnectionPoolConfig {
        fn default() -> Self {
            Self {
                max_idle_per_host: crate::defaults::http::MAX_IDLE_PER_HOST,
                max_idle_total: crate::defaults::http::MAX_IDLE_TOTAL,
                connect_timeout: crate::defaults::http::CONNECT_TIMEOUT,
                request_timeout: crate::defaults::http::REQUEST_TIMEOUT,
                keep_alive_timeout: crate::defaults::http::KEEP_ALIVE_TIMEOUT,
            }
        }
    }

    /// Create an optimized HTTP client
    pub fn create_optimized_client(
        config: ConnectionPoolConfig,
    ) -> Result<reqwest::Client, Box<dyn std::error::Error>> {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(config.max_idle_per_host)
            .pool_idle_timeout(config.keep_alive_timeout)
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .tcp_keepalive(crate::defaults::http::TCP_KEEP_ALIVE)
            .tcp_nodelay(true)
            .build()?;

        Ok(client)
    }

    /// Memory-efficient string interning for common values
    ///
    /// Note: This is a simplified implementation that uses a HashMap for deduplication.
    /// For production use, consider using a proper string interner library like `string-interner`.
    #[allow(dead_code)]
    pub struct StringInterner {
        strings: std::collections::HashMap<String, String>,
    }

    impl Default for StringInterner {
        fn default() -> Self {
            Self::new()
        }
    }

    impl StringInterner {
        pub fn new() -> Self {
            Self {
                strings: std::collections::HashMap::new(),
            }
        }

        /// Intern a string (safe implementation without memory leaks)
        ///
        /// Returns a reference to the interned string. The string will be kept
        /// alive as long as this StringInterner instance exists.
        pub fn intern(&mut self, s: String) -> &str {
            // Use entry API to avoid cloning if the string already exists
            self.strings.entry(s.clone()).or_insert(s).as_str()
        }

        /// Get the number of interned strings
        pub fn len(&self) -> usize {
            self.strings.len()
        }

        /// Check if the interner is empty
        pub fn is_empty(&self) -> bool {
            self.strings.is_empty()
        }

        /// Clear all interned strings
        pub fn clear(&mut self) {
            self.strings.clear();
        }
    }

    /// High-performance LRU cache for chat responses
    pub struct ResponseCache {
        cache: std::collections::HashMap<String, CachedResponse>,
        access_order: std::collections::VecDeque<String>,
        max_size: usize,
        hit_count: u64,
        miss_count: u64,
    }

    #[derive(Clone)]
    struct CachedResponse {
        response: crate::types::ChatResponse,
        timestamp: std::time::Instant,
        access_count: u32,
    }

    impl ResponseCache {
        /// Create a new response cache with specified capacity
        pub fn new(max_size: usize) -> Self {
            Self {
                cache: std::collections::HashMap::with_capacity(max_size),
                access_order: std::collections::VecDeque::with_capacity(max_size),
                max_size,
                hit_count: 0,
                miss_count: 0,
            }
        }

        /// Generate cache key from messages (optimized for performance)
        pub fn cache_key(messages: &[crate::types::ChatMessage]) -> String {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            for msg in messages {
                msg.role.hash(&mut hasher);
                if let Some(text) = msg.content_text() {
                    text.hash(&mut hasher);
                }
            }
            format!("chat_{:x}", hasher.finish())
        }

        /// Get cached response if available
        pub fn get(&mut self, key: &str) -> Option<crate::types::ChatResponse> {
            if let Some(cached) = self.cache.get_mut(key) {
                // Update access statistics
                cached.access_count += 1;
                self.hit_count += 1;

                // Move to front of access order
                if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                    self.access_order.remove(pos);
                }
                self.access_order.push_front(key.to_string());

                Some(cached.response.clone())
            } else {
                self.miss_count += 1;
                None
            }
        }

        /// Store response in cache
        pub fn put(&mut self, key: String, response: crate::types::ChatResponse) {
            // Remove oldest entry if at capacity
            if self.cache.len() >= self.max_size
                && let Some(oldest_key) = self.access_order.pop_back()
            {
                self.cache.remove(&oldest_key);
            }

            let cached = CachedResponse {
                response,
                timestamp: std::time::Instant::now(),
                access_count: 1,
            };

            self.cache.insert(key.clone(), cached);
            self.access_order.push_front(key);
        }

        /// Get cache hit rate
        pub fn hit_rate(&self) -> f64 {
            let total = self.hit_count + self.miss_count;
            if total == 0 {
                0.0
            } else {
                self.hit_count as f64 / total as f64
            }
        }

        /// Clear expired entries
        pub fn cleanup_expired(&mut self, max_age: std::time::Duration) {
            let now = std::time::Instant::now();
            let mut expired_keys = Vec::new();

            for (key, cached) in &self.cache {
                if now.duration_since(cached.timestamp) > max_age {
                    expired_keys.push(key.clone());
                }
            }

            for key in expired_keys {
                self.cache.remove(&key);
                if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                    self.access_order.remove(pos);
                }
            }
        }

        /// Get cache statistics
        pub fn stats(&self) -> CacheStats {
            CacheStats {
                size: self.cache.len(),
                capacity: self.max_size,
                hit_count: self.hit_count,
                miss_count: self.miss_count,
                hit_rate: self.hit_rate(),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct CacheStats {
        pub size: usize,
        pub capacity: usize,
        pub hit_count: u64,
        pub miss_count: u64,
        pub hit_rate: f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_metrics() {
        let mut metrics = ProviderMetrics::new("test-provider".to_string());

        assert_eq!(metrics.success_rate(), 0.0);
        assert_eq!(metrics.cache_hit_rate(), 0.0);

        metrics.request_count = 10;
        metrics.success_count = 8;
        assert_eq!(metrics.success_rate(), 0.8);

        metrics.cache_hits = 7;
        metrics.cache_misses = 3;
        assert_eq!(metrics.cache_hit_rate(), 0.7);
    }

    #[tokio::test]
    async fn test_performance_monitor() {
        let config = MonitorConfig::default();
        let monitor = PerformanceMonitor::new(config);

        // Test error recording
        monitor.record_error("network_error", Some("openai")).await;

        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.error_rate.total_errors, 1);
        assert_eq!(metrics.error_rate.total_requests, 1);
        assert_eq!(metrics.error_rate.error_rate, 1.0);

        // Test success recording
        monitor
            .record_success(Some("openai"), Duration::from_millis(100))
            .await;

        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.error_rate.total_requests, 2);
        assert_eq!(metrics.error_rate.error_rate, 0.5);
    }

    #[tokio::test]
    async fn test_request_timer() {
        let config = MonitorConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let timer = monitor.start_request().await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        let duration = timer.finish().await;

        assert!(duration >= Duration::from_millis(10));
    }
}
