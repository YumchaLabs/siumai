//! Performance Tracing
//!
//! This module provides performance monitoring and tracing capabilities.

use super::events::{PerformanceEvent, PerformanceMetricType, TracingEvent};
use super::{SpanId, TraceId};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Performance tracer for monitoring system performance
#[derive(Debug, Clone)]
pub struct PerformanceTracer {
    /// Metrics storage
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Configuration
    config: PerformanceConfig,
}

/// Performance metrics storage
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    /// Request latencies
    pub latencies: Vec<Duration>,
    /// Memory usage samples
    pub memory_usage: Vec<u64>,
    /// CPU usage samples
    pub cpu_usage: Vec<f64>,
    /// Connection pool metrics
    pub connection_pool_size: Vec<usize>,
    /// Cache hit rates
    pub cache_hit_rates: Vec<f64>,
    /// Token generation rates
    pub token_generation_rates: Vec<f64>,
    /// Request counts by provider
    pub request_counts: HashMap<String, u64>,
    /// Error counts by provider
    pub error_counts: HashMap<String, u64>,
}

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Maximum number of samples to keep
    pub max_samples: usize,
    /// Whether to collect memory metrics
    pub collect_memory: bool,
    /// Whether to collect CPU metrics
    pub collect_cpu: bool,
    /// Sampling interval for system metrics
    pub sampling_interval: Duration,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_samples: 10000,
            collect_memory: false, // Can be expensive
            collect_cpu: false,    // Can be expensive
            sampling_interval: Duration::from_secs(60),
        }
    }
}

impl PerformanceTracer {
    /// Create a new performance tracer
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            config,
        }
    }

    /// Record request latency
    pub async fn record_latency(
        &self,
        trace_id: TraceId,
        provider: &str,
        latency: Duration,
    ) -> TracingEvent {
        let mut metrics = self.metrics.write().await;

        // Add latency sample
        metrics.latencies.push(latency);
        if metrics.latencies.len() > self.config.max_samples {
            metrics.latencies.remove(0);
        }

        // Update request count
        *metrics
            .request_counts
            .entry(provider.to_string())
            .or_insert(0) += 1;

        debug!(
            trace_id = %trace_id,
            provider = provider,
            latency_ms = latency.as_millis(),
            "Latency recorded"
        );

        TracingEvent::Performance(PerformanceEvent {
            timestamp: SystemTime::now(),
            metric_type: PerformanceMetricType::Latency,
            value: latency.as_secs_f64(),
            unit: "seconds".to_string(),
            context: {
                let mut ctx = HashMap::new();
                ctx.insert("provider".to_string(), provider.to_string());
                ctx.insert("trace_id".to_string(), trace_id.to_string());
                ctx
            },
        })
    }

    /// Record throughput
    pub async fn record_throughput(
        &self,
        provider: &str,
        requests_per_second: f64,
    ) -> TracingEvent {
        info!(
            provider = provider,
            rps = requests_per_second,
            "Throughput recorded"
        );

        TracingEvent::Performance(PerformanceEvent {
            timestamp: SystemTime::now(),
            metric_type: PerformanceMetricType::Throughput,
            value: requests_per_second,
            unit: "requests/second".to_string(),
            context: {
                let mut ctx = HashMap::new();
                ctx.insert("provider".to_string(), provider.to_string());
                ctx
            },
        })
    }

    /// Record memory usage
    pub async fn record_memory_usage(&self, bytes: u64) -> Option<TracingEvent> {
        if !self.config.collect_memory {
            return None;
        }

        let mut metrics = self.metrics.write().await;
        metrics.memory_usage.push(bytes);
        if metrics.memory_usage.len() > self.config.max_samples {
            metrics.memory_usage.remove(0);
        }

        debug!(memory_bytes = bytes, "Memory usage recorded");

        Some(TracingEvent::Performance(PerformanceEvent {
            timestamp: SystemTime::now(),
            metric_type: PerformanceMetricType::MemoryUsage,
            value: bytes as f64,
            unit: "bytes".to_string(),
            context: HashMap::new(),
        }))
    }

    /// Record CPU usage
    pub async fn record_cpu_usage(&self, percentage: f64) -> Option<TracingEvent> {
        if !self.config.collect_cpu {
            return None;
        }

        let mut metrics = self.metrics.write().await;
        metrics.cpu_usage.push(percentage);
        if metrics.cpu_usage.len() > self.config.max_samples {
            metrics.cpu_usage.remove(0);
        }

        debug!(cpu_percentage = percentage, "CPU usage recorded");

        Some(TracingEvent::Performance(PerformanceEvent {
            timestamp: SystemTime::now(),
            metric_type: PerformanceMetricType::CpuUsage,
            value: percentage,
            unit: "percentage".to_string(),
            context: HashMap::new(),
        }))
    }

    /// Record connection pool size
    pub async fn record_connection_pool_size(&self, size: usize) -> TracingEvent {
        let mut metrics = self.metrics.write().await;
        metrics.connection_pool_size.push(size);
        if metrics.connection_pool_size.len() > self.config.max_samples {
            metrics.connection_pool_size.remove(0);
        }

        debug!(pool_size = size, "Connection pool size recorded");

        TracingEvent::Performance(PerformanceEvent {
            timestamp: SystemTime::now(),
            metric_type: PerformanceMetricType::ConnectionPoolSize,
            value: size as f64,
            unit: "connections".to_string(),
            context: HashMap::new(),
        })
    }

    /// Record cache hit rate
    pub async fn record_cache_hit_rate(&self, rate: f64) -> TracingEvent {
        let mut metrics = self.metrics.write().await;
        metrics.cache_hit_rates.push(rate);
        if metrics.cache_hit_rates.len() > self.config.max_samples {
            metrics.cache_hit_rates.remove(0);
        }

        debug!(hit_rate = rate, "Cache hit rate recorded");

        TracingEvent::Performance(PerformanceEvent {
            timestamp: SystemTime::now(),
            metric_type: PerformanceMetricType::CacheHitRate,
            value: rate,
            unit: "percentage".to_string(),
            context: HashMap::new(),
        })
    }

    /// Record token generation rate
    pub async fn record_token_generation_rate(
        &self,
        trace_id: TraceId,
        provider: &str,
        tokens_per_second: f64,
    ) -> TracingEvent {
        let mut metrics = self.metrics.write().await;
        metrics.token_generation_rates.push(tokens_per_second);
        if metrics.token_generation_rates.len() > self.config.max_samples {
            metrics.token_generation_rates.remove(0);
        }

        info!(
            trace_id = %trace_id,
            provider = provider,
            tokens_per_second = tokens_per_second,
            "Token generation rate recorded"
        );

        TracingEvent::Performance(PerformanceEvent {
            timestamp: SystemTime::now(),
            metric_type: PerformanceMetricType::TokenGenerationRate,
            value: tokens_per_second,
            unit: "tokens/second".to_string(),
            context: {
                let mut ctx = HashMap::new();
                ctx.insert("provider".to_string(), provider.to_string());
                ctx.insert("trace_id".to_string(), trace_id.to_string());
                ctx
            },
        })
    }

    /// Record error
    pub async fn record_error(&self, provider: &str) {
        let mut metrics = self.metrics.write().await;
        *metrics
            .error_counts
            .entry(provider.to_string())
            .or_insert(0) += 1;

        debug!(provider = provider, "Error recorded");
    }

    /// Get performance statistics
    pub async fn get_stats(&self) -> PerformanceStats {
        let metrics = self.metrics.read().await;

        PerformanceStats {
            avg_latency: Self::calculate_average(&metrics.latencies),
            p95_latency: Self::calculate_percentile(&metrics.latencies, 0.95),
            p99_latency: Self::calculate_percentile(&metrics.latencies, 0.99),
            total_requests: metrics.request_counts.values().sum(),
            total_errors: metrics.error_counts.values().sum(),
            error_rate: {
                let total_requests: u64 = metrics.request_counts.values().sum();
                let total_errors: u64 = metrics.error_counts.values().sum();
                if total_requests > 0 {
                    total_errors as f64 / total_requests as f64
                } else {
                    0.0
                }
            },
            avg_memory_usage: metrics.memory_usage.iter().map(|&x| x as f64).sum::<f64>()
                / metrics.memory_usage.len().max(1) as f64,
            avg_cpu_usage: metrics.cpu_usage.iter().sum::<f64>()
                / metrics.cpu_usage.len().max(1) as f64,
            request_counts_by_provider: metrics.request_counts.clone(),
            error_counts_by_provider: metrics.error_counts.clone(),
        }
    }

    /// Calculate average duration
    fn calculate_average(durations: &[Duration]) -> Duration {
        if durations.is_empty() {
            return Duration::ZERO;
        }

        let total: Duration = durations.iter().sum();
        total / durations.len() as u32
    }

    /// Calculate percentile duration
    fn calculate_percentile(durations: &[Duration], percentile: f64) -> Duration {
        if durations.is_empty() {
            return Duration::ZERO;
        }

        let mut sorted = durations.to_vec();
        sorted.sort();

        let index = ((durations.len() as f64 - 1.0) * percentile) as usize;
        sorted[index]
    }
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub avg_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub total_requests: u64,
    pub total_errors: u64,
    pub error_rate: f64,
    pub avg_memory_usage: f64,
    pub avg_cpu_usage: f64,
    pub request_counts_by_provider: HashMap<String, u64>,
    pub error_counts_by_provider: HashMap<String, u64>,
}

/// Timing context for measuring operation duration
#[derive(Debug)]
pub struct TimingContext {
    start_time: Instant,
    trace_id: TraceId,
    span_id: SpanId,
    operation: String,
}

impl TimingContext {
    /// Create a new timing context
    pub fn new(trace_id: TraceId, operation: String) -> Self {
        Self {
            start_time: Instant::now(),
            trace_id,
            span_id: SpanId::new(),
            operation,
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Finish timing and return duration
    pub fn finish(self) -> Duration {
        let duration = self.elapsed();

        debug!(
            trace_id = %self.trace_id,
            span_id = %self.span_id,
            operation = %self.operation,
            duration_ms = duration.as_millis(),
            "Operation completed"
        );

        duration
    }
}
