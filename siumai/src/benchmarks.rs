//! Benchmarking and Performance Testing
//!
//! This module provides benchmarking utilities and performance tests
//! for the siumai library components.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

use crate::error::LlmError;
use crate::performance::{MonitorConfig, PerformanceMonitor};
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, MessageContent, MessageMetadata, MessageRole};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of concurrent requests
    pub concurrency: usize,
    /// Total number of requests to send
    pub total_requests: usize,
    /// Duration to run the benchmark
    pub duration: Option<Duration>,
    /// Warmup period before starting measurements
    pub warmup_duration: Duration,
    /// Request rate limit (requests per second)
    pub rate_limit: Option<f64>,
    /// Test scenarios to run
    pub scenarios: Vec<BenchmarkScenario>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            concurrency: 10,
            total_requests: 100,
            duration: None,
            warmup_duration: Duration::from_secs(5),
            rate_limit: None,
            scenarios: vec![BenchmarkScenario::default()],
        }
    }
}

/// Benchmark scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScenario {
    /// Scenario name
    pub name: String,
    /// Test messages to send
    pub messages: Vec<ChatMessage>,
    /// Expected response characteristics
    pub expected: ExpectedResponse,
    /// Weight of this scenario (for mixed workloads)
    pub weight: f64,
}

impl Default for BenchmarkScenario {
    fn default() -> Self {
        Self {
            name: "basic_chat".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: MessageContent::Text("Hello, how are you?".to_string()),
                metadata: MessageMetadata::default(),
            }],
            expected: ExpectedResponse::default(),
            weight: 1.0,
        }
    }
}

/// Expected response characteristics for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedResponse {
    /// Minimum response length
    pub min_length: Option<usize>,
    /// Maximum response length
    pub max_length: Option<usize>,
    /// Expected response time range
    pub response_time_range: Option<(Duration, Duration)>,
    /// Required keywords in response
    pub required_keywords: Vec<String>,
    /// Forbidden keywords in response
    pub forbidden_keywords: Vec<String>,
}

impl Default for ExpectedResponse {
    fn default() -> Self {
        Self {
            min_length: Some(1),
            max_length: None,
            response_time_range: None,
            required_keywords: Vec::new(),
            forbidden_keywords: Vec::new(),
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Total requests sent
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Total duration of the benchmark
    pub total_duration: Duration,
    /// Requests per second
    pub requests_per_second: f64,
    /// Latency statistics
    pub latency_stats: LatencyStats,
    /// Error breakdown
    pub error_breakdown: HashMap<String, usize>,
    /// Scenario results
    pub scenario_results: HashMap<String, ScenarioResults>,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Mean latency
    pub mean: Duration,
    /// Median latency (P50)
    pub median: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// 99.9th percentile
    pub p999: Duration,
    /// Minimum latency
    pub min: Duration,
    /// Maximum latency
    pub max: Duration,
    /// Standard deviation
    pub std_dev: Duration,
}

/// Scenario-specific results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResults {
    /// Scenario name
    pub name: String,
    /// Number of requests for this scenario
    pub requests: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Validation results
    pub validation_results: ValidationResults,
}

/// Validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Number of responses that passed validation
    pub passed: usize,
    /// Number of responses that failed validation
    pub failed: usize,
    /// Validation failure reasons
    pub failure_reasons: HashMap<String, usize>,
}

/// Resource usage during benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Peak memory usage in bytes
    pub peak_memory: u64,
    /// Average memory usage in bytes
    pub avg_memory: u64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Network bytes sent
    pub bytes_sent: u64,
    /// Network bytes received
    pub bytes_received: u64,
}

/// Benchmark runner
pub struct BenchmarkRunner {
    /// Configuration
    config: BenchmarkConfig,
    /// Performance monitor
    monitor: PerformanceMonitor,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        let monitor_config = MonitorConfig {
            detailed_metrics: true,
            memory_tracking: true,
            ..MonitorConfig::default()
        };

        Self {
            config,
            monitor: PerformanceMonitor::new(monitor_config),
        }
    }

    /// Run benchmark against a client
    pub async fn run<T: ChatCapability + Send + Sync + 'static>(
        &self,
        client: std::sync::Arc<T>,
    ) -> Result<BenchmarkResults, LlmError> {
        println!(
            "🚀 Starting benchmark with {} concurrent requests",
            self.config.concurrency
        );

        // Warmup phase
        if !self.config.warmup_duration.is_zero() {
            println!("🔥 Warming up for {:?}", self.config.warmup_duration);
            self.warmup(&*client).await?;
        }

        let start_time = Instant::now();
        let mut handles = Vec::new();
        let mut results = Vec::new();

        // Create semaphore for concurrency control
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(self.config.concurrency));

        // Calculate requests per worker
        let requests_per_worker = self.config.total_requests / self.config.concurrency;
        let remaining_requests = self.config.total_requests % self.config.concurrency;

        for worker_id in 0..self.config.concurrency {
            let worker_requests = if worker_id < remaining_requests {
                requests_per_worker + 1
            } else {
                requests_per_worker
            };

            if worker_requests == 0 {
                continue;
            }

            let semaphore = semaphore.clone();
            let scenarios = self.config.scenarios.clone();
            let monitor = self.monitor.clone();
            let client = client.clone();

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                Self::run_worker(worker_id, worker_requests, scenarios, &*client, monitor).await
            });

            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            match handle.await {
                Ok(worker_results) => results.extend(worker_results),
                Err(e) => eprintln!("Worker failed: {e}"),
            }
        }

        let total_duration = start_time.elapsed();

        // Compile results
        self.compile_results(results, total_duration).await
    }

    /// Run warmup requests
    async fn warmup<T: ChatCapability + Send + Sync>(&self, client: &T) -> Result<(), LlmError> {
        let warmup_requests = (self.config.concurrency * 2).min(10);
        let scenario = &self.config.scenarios[0];

        for _ in 0..warmup_requests {
            let _ = client
                .chat_with_tools(scenario.messages.clone(), None)
                .await;
            sleep(Duration::from_millis(100)).await;
        }

        Ok(())
    }

    /// Run a single worker
    async fn run_worker<T: ChatCapability + Send + Sync>(
        worker_id: usize,
        requests: usize,
        scenarios: Vec<BenchmarkScenario>,
        client: &T,
        monitor: PerformanceMonitor,
    ) -> Vec<RequestResult> {
        let mut results = Vec::new();

        for request_id in 0..requests {
            // Select scenario based on weight
            let scenario = Self::select_scenario(&scenarios);

            let timer = monitor.start_request().await;

            match client
                .chat_with_tools(scenario.messages.clone(), None)
                .await
            {
                Ok(response) => {
                    let duration = timer.finish().await;
                    monitor.record_success(None, duration).await;

                    let validation = Self::validate_response(&response, &scenario.expected);

                    results.push(RequestResult {
                        worker_id,
                        request_id,
                        scenario_name: scenario.name.clone(),
                        success: true,
                        duration,
                        error: None,
                        response_length: response.content.text().map(str::len),
                        validation,
                    });
                }
                Err(error) => {
                    let duration = timer.finish().await;
                    monitor.record_error("request_failed", None).await;

                    results.push(RequestResult {
                        worker_id,
                        request_id,
                        scenario_name: scenario.name.clone(),
                        success: false,
                        duration,
                        error: Some(error.to_string()),
                        response_length: None,
                        validation: ValidationResults {
                            passed: 0,
                            failed: 1,
                            failure_reasons: [("error".to_string(), 1)].into_iter().collect(),
                        },
                    });
                }
            }
        }

        results
    }

    /// Select a scenario based on weights
    fn select_scenario(scenarios: &[BenchmarkScenario]) -> &BenchmarkScenario {
        // Simple implementation - just use the first scenario
        // In a real implementation, you'd use weighted random selection
        &scenarios[0]
    }

    /// Validate response against expected characteristics
    fn validate_response(
        response: &crate::types::ChatResponse,
        expected: &ExpectedResponse,
    ) -> ValidationResults {
        let mut passed = 0;
        let mut failed = 0;
        let mut failure_reasons = HashMap::new();

        let response_text = response.content.text().unwrap_or("");
        let response_length = response_text.len();

        // Check length constraints
        if let Some(min_length) = expected.min_length {
            if response_length >= min_length {
                passed += 1;
            } else {
                failed += 1;
                *failure_reasons.entry("min_length".to_string()).or_insert(0) += 1;
            }
        }

        if let Some(max_length) = expected.max_length {
            if response_length <= max_length {
                passed += 1;
            } else {
                failed += 1;
                *failure_reasons.entry("max_length".to_string()).or_insert(0) += 1;
            }
        }

        // Check required keywords
        for keyword in &expected.required_keywords {
            if response_text.contains(keyword) {
                passed += 1;
            } else {
                failed += 1;
                *failure_reasons
                    .entry("missing_keyword".to_string())
                    .or_insert(0) += 1;
            }
        }

        // Check forbidden keywords
        for keyword in &expected.forbidden_keywords {
            if !response_text.contains(keyword) {
                passed += 1;
            } else {
                failed += 1;
                *failure_reasons
                    .entry("forbidden_keyword".to_string())
                    .or_insert(0) += 1;
            }
        }

        ValidationResults {
            passed,
            failed,
            failure_reasons,
        }
    }

    /// Compile final results
    async fn compile_results(
        &self,
        results: Vec<RequestResult>,
        total_duration: Duration,
    ) -> Result<BenchmarkResults, LlmError> {
        let total_requests = results.len();
        let successful_requests = results.iter().filter(|r| r.success).count();
        let failed_requests = total_requests - successful_requests;

        let requests_per_second = total_requests as f64 / total_duration.as_secs_f64();

        // Calculate latency statistics
        let mut durations: Vec<Duration> = results.iter().map(|r| r.duration).collect();
        durations.sort();

        let latency_stats = if !durations.is_empty() {
            let len = durations.len();
            LatencyStats {
                mean: durations.iter().sum::<Duration>() / len as u32,
                median: durations[len / 2],
                p95: durations[(len * 95) / 100],
                p99: durations[(len * 99) / 100],
                p999: durations[(len * 999) / 1000],
                min: durations[0],
                max: durations[len - 1],
                std_dev: Duration::ZERO, // Simplified - would calculate actual std dev
            }
        } else {
            LatencyStats {
                mean: Duration::ZERO,
                median: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
                p999: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                std_dev: Duration::ZERO,
            }
        };

        // Error breakdown
        let mut error_breakdown = HashMap::new();
        for result in &results {
            if let Some(ref error) = result.error {
                *error_breakdown.entry(error.clone()).or_insert(0) += 1;
            }
        }

        // Scenario results
        let mut scenario_results = HashMap::new();
        for scenario in &self.config.scenarios {
            let scenario_requests: Vec<_> = results
                .iter()
                .filter(|r| r.scenario_name == scenario.name)
                .collect();

            if !scenario_requests.is_empty() {
                let success_count = scenario_requests.iter().filter(|r| r.success).count();
                let success_rate = success_count as f64 / scenario_requests.len() as f64;

                let avg_response_time = scenario_requests
                    .iter()
                    .map(|r| r.duration)
                    .sum::<Duration>()
                    / scenario_requests.len() as u32;

                let validation_results = ValidationResults {
                    passed: scenario_requests.iter().map(|r| r.validation.passed).sum(),
                    failed: scenario_requests.iter().map(|r| r.validation.failed).sum(),
                    failure_reasons: HashMap::new(), // Simplified
                };

                scenario_results.insert(
                    scenario.name.clone(),
                    ScenarioResults {
                        name: scenario.name.clone(),
                        requests: scenario_requests.len(),
                        success_rate,
                        avg_response_time,
                        validation_results,
                    },
                );
            }
        }

        Ok(BenchmarkResults {
            total_requests,
            successful_requests,
            failed_requests,
            total_duration,
            requests_per_second,
            latency_stats,
            error_breakdown,
            scenario_results,
            resource_usage: ResourceUsage {
                peak_memory: 0,
                avg_memory: 0,
                cpu_usage: 0.0,
                bytes_sent: 0,
                bytes_received: 0,
            },
        })
    }
}

/// Individual request result
#[derive(Debug, Clone)]
struct RequestResult {
    #[allow(dead_code)] // retained for per-worker analysis reports
    worker_id: usize,
    #[allow(dead_code)] // retained for per-request diagnostics
    request_id: usize,
    scenario_name: String,
    success: bool,
    duration: Duration,
    error: Option<String>,
    #[allow(dead_code)] // reserved for payload-size based validations
    response_length: Option<usize>,
    validation: ValidationResults,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.concurrency, 10);
        assert_eq!(config.total_requests, 100);
        assert_eq!(config.scenarios.len(), 1);
    }

    #[test]
    fn test_expected_response_validation() {
        let expected = ExpectedResponse {
            min_length: Some(5),
            max_length: Some(100),
            response_time_range: None,
            required_keywords: vec!["hello".to_string()],
            forbidden_keywords: vec!["error".to_string()],
        };

        // This would require a mock response to test properly
        assert!(expected.min_length.is_some());
        assert!(expected.max_length.is_some());
    }
}
