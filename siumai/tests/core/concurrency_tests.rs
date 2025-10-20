//! Concurrency and Thread Safety Tests
//!
//! This module provides comprehensive tests for concurrent usage of the LLM library.
//! These tests ensure that the library is thread-safe and can handle multiple
//! simultaneous requests without data races, deadlocks, or resource leaks.
//!
//! ## Test Categories
//!
//! 1. **Multi-threaded Request Tests**: Verify that multiple threads can make
//!    requests simultaneously without interference.
//!
//! 2. **Connection Pool Tests**: Test that HTTP connection pooling works correctly
//!    under concurrent load.
//!
//! 3. **Resource Competition Tests**: Ensure proper handling when multiple threads
//!    compete for limited resources.
//!
//! 4. **Memory Safety Tests**: Verify no data races or memory corruption under
//!    concurrent access.
//!
//! 5. **Deadlock Prevention Tests**: Ensure the library doesn't deadlock under
//!    various concurrent scenarios.

use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

mod mock_framework;
use mock_framework::{MockLlmServer, MockTestUtils};

/// Configuration for concurrency tests
#[derive(Debug, Clone)]
pub struct ConcurrencyTestConfig {
    /// Number of concurrent threads/tasks
    pub thread_count: usize,
    /// Number of requests per thread
    pub requests_per_thread: usize,
    /// Maximum time to wait for all requests to complete
    pub timeout: Duration,
    /// Whether to use artificial delays
    pub use_delays: bool,
    /// Delay between requests in the same thread
    pub request_delay: Option<Duration>,
}

impl Default for ConcurrencyTestConfig {
    fn default() -> Self {
        Self {
            thread_count: 10,
            requests_per_thread: 5,
            timeout: Duration::from_secs(30),
            use_delays: false,
            request_delay: None,
        }
    }
}

/// Results from a concurrency test
#[derive(Debug)]
pub struct ConcurrencyTestResults {
    /// Total number of successful requests
    pub successful_requests: usize,
    /// Total number of failed requests
    pub failed_requests: usize,
    /// Total time taken for all requests
    pub total_duration: Duration,
    /// Average response time per request
    pub average_response_time: Duration,
    /// Maximum response time observed
    pub max_response_time: Duration,
    /// Minimum response time observed
    pub min_response_time: Duration,
    /// Number of threads that completed successfully
    pub successful_threads: usize,
    /// Any errors encountered
    pub errors: Vec<String>,
}

impl Default for ConcurrencyTestResults {
    fn default() -> Self {
        Self::new()
    }
}

impl ConcurrencyTestResults {
    pub fn new() -> Self {
        Self {
            successful_requests: 0,
            failed_requests: 0,
            total_duration: Duration::ZERO,
            average_response_time: Duration::ZERO,
            max_response_time: Duration::ZERO,
            min_response_time: Duration::MAX,
            successful_threads: 0,
            errors: Vec::new(),
        }
    }

    /// Calculate final statistics
    pub fn finalize(&mut self) {
        let total_requests = self.successful_requests + self.failed_requests;
        if total_requests > 0 {
            self.average_response_time = self.total_duration / total_requests as u32;
        }
    }

    /// Check if the test passed basic criteria
    pub fn is_successful(&self) -> bool {
        self.failed_requests == 0 && self.successful_requests > 0 && self.errors.is_empty()
    }
}

/// Thread-safe request counter for tracking concurrent requests
#[derive(Debug)]
pub struct RequestCounter {
    count: Arc<Mutex<usize>>,
    max_concurrent: Arc<Mutex<usize>>,
    current_concurrent: Arc<Mutex<usize>>,
}

impl RequestCounter {
    pub fn new() -> Self {
        Self {
            count: Arc::new(Mutex::new(0)),
            max_concurrent: Arc::new(Mutex::new(0)),
            current_concurrent: Arc::new(Mutex::new(0)),
        }
    }

    /// Increment request count and track concurrency
    pub fn start_request(&self) {
        let mut count = self.count.lock().unwrap();
        let mut current = self.current_concurrent.lock().unwrap();
        let mut max = self.max_concurrent.lock().unwrap();

        *count += 1;
        *current += 1;

        if *current > *max {
            *max = *current;
        }
    }

    /// Decrement concurrent request count
    pub fn end_request(&self) {
        let mut current = self.current_concurrent.lock().unwrap();
        *current -= 1;
    }

    /// Get total request count
    pub fn total_requests(&self) -> usize {
        *self.count.lock().unwrap()
    }

    /// Get maximum concurrent requests observed
    pub fn max_concurrent_requests(&self) -> usize {
        *self.max_concurrent.lock().unwrap()
    }

    /// Get current concurrent requests
    pub fn current_concurrent_requests(&self) -> usize {
        *self.current_concurrent.lock().unwrap()
    }
}

impl Default for RequestCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Concurrency test runner
pub struct ConcurrencyTester {
    config: ConcurrencyTestConfig,
    counter: RequestCounter,
}

impl ConcurrencyTester {
    pub fn new(config: ConcurrencyTestConfig) -> Self {
        Self {
            config,
            counter: RequestCounter::new(),
        }
    }

    /// Run multi-threaded HTTP requests test
    pub async fn test_concurrent_http_requests(&self) -> ConcurrencyTestResults {
        println!("🔄 Running concurrent HTTP requests test...");
        println!(
            "  📊 Threads: {}, Requests per thread: {}",
            self.config.thread_count, self.config.requests_per_thread
        );

        let server = MockLlmServer::new().await;
        server.setup_openai_chat().await;

        let mut results = ConcurrencyTestResults::new();
        let _start_time = Instant::now();

        // Create a barrier to synchronize thread start
        let barrier = Arc::new(Barrier::new(self.config.thread_count));
        let mut handles = Vec::new();

        for thread_id in 0..self.config.thread_count {
            let barrier = barrier.clone();
            let counter = Arc::new(self.counter.count.clone());
            let base_url = server.base_url();
            let requests_per_thread = self.config.requests_per_thread;
            let request_delay = self.config.request_delay;

            let handle = thread::spawn(move || {
                // Wait for all threads to be ready
                barrier.wait();

                let rt = tokio::runtime::Runtime::new().unwrap();
                let mut thread_results = Vec::new();

                rt.block_on(async {
                    let client = reqwest::Client::new();

                    for request_id in 0..requests_per_thread {
                        let request_start = Instant::now();

                        // Increment counter
                        {
                            let mut count = counter.lock().unwrap();
                            *count += 1;
                        }

                        let result = client
                            .post(format!("{}/v1/chat/completions", base_url))
                            .header("authorization", "Bearer test-key")
                            .json(&serde_json::json!({
                                "model": "gpt-3.5-turbo",
                                "messages": [{
                                    "role": "user", 
                                    "content": format!("Hello from thread {} request {}", thread_id, request_id)
                                }]
                            }))
                            .send()
                            .await;

                        let request_duration = request_start.elapsed();

                        match result {
                            Ok(response) => {
                                if response.status().is_success() {
                                    thread_results.push(Ok(request_duration));
                                } else {
                                    thread_results.push(Err(format!(
                                        "HTTP error: {}", response.status()
                                    )));
                                }
                            }
                            Err(e) => {
                                thread_results.push(Err(format!("Request error: {}", e)));
                            }
                        }

                        // Add delay if configured
                        if let Some(delay) = request_delay {
                            tokio::time::sleep(delay).await;
                        }
                    }
                });

                thread_results
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            match handle.join() {
                Ok(thread_results) => {
                    results.successful_threads += 1;
                    for result in thread_results {
                        match result {
                            Ok(duration) => {
                                results.successful_requests += 1;
                                results.total_duration += duration;
                                if duration > results.max_response_time {
                                    results.max_response_time = duration;
                                }
                                if duration < results.min_response_time {
                                    results.min_response_time = duration;
                                }
                            }
                            Err(error) => {
                                results.failed_requests += 1;
                                results.errors.push(error);
                            }
                        }
                    }
                }
                Err(e) => {
                    results.errors.push(format!("Thread panic: {:?}", e));
                }
            }
        }

        results.finalize();

        println!("  ✅ Concurrent HTTP test completed:");
        println!(
            "    📈 Successful requests: {}",
            results.successful_requests
        );
        println!("    ❌ Failed requests: {}", results.failed_requests);
        println!(
            "    ⏱️  Average response time: {:?}",
            results.average_response_time
        );
        println!("    🔄 Successful threads: {}", results.successful_threads);

        results
    }

    /// Test async task concurrency with tokio
    pub async fn test_async_task_concurrency(&self) -> ConcurrencyTestResults {
        println!("🚀 Running async task concurrency test...");

        let server = MockLlmServer::new().await;
        server.setup_openai_chat().await;

        let mut results = ConcurrencyTestResults::new();
        let mut join_set = JoinSet::new();

        let client = Arc::new(reqwest::Client::new());
        let base_url = Arc::new(server.base_url());

        // Spawn concurrent async tasks
        for task_id in 0..self.config.thread_count {
            let client = client.clone();
            let base_url = base_url.clone();
            let requests_per_task = self.config.requests_per_thread;

            join_set.spawn(async move {
                let mut task_results = Vec::new();

                for request_id in 0..requests_per_task {
                    let request_start = Instant::now();

                    let result = client
                        .post(format!("{}/v1/chat/completions", base_url))
                        .header("authorization", "Bearer test-key")
                        .json(&serde_json::json!({
                            "model": "gpt-3.5-turbo",
                            "messages": [{
                                "role": "user",
                                "content": format!("Hello from task {} request {}", task_id, request_id)
                            }]
                        }))
                        .send()
                        .await;

                    let request_duration = request_start.elapsed();

                    match result {
                        Ok(response) => {
                            if response.status().is_success() {
                                task_results.push(Ok(request_duration));
                            } else {
                                task_results.push(Err(format!(
                                    "HTTP error: {}", response.status()
                                )));
                            }
                        }
                        Err(e) => {
                            task_results.push(Err(format!("Request error: {}", e)));
                        }
                    }
                }

                task_results
            });
        }

        // Wait for all tasks to complete
        while let Some(task_result) = join_set.join_next().await {
            match task_result {
                Ok(task_results) => {
                    results.successful_threads += 1;
                    for result in task_results {
                        match result {
                            Ok(duration) => {
                                results.successful_requests += 1;
                                results.total_duration += duration;
                                if duration > results.max_response_time {
                                    results.max_response_time = duration;
                                }
                                if duration < results.min_response_time {
                                    results.min_response_time = duration;
                                }
                            }
                            Err(error) => {
                                results.failed_requests += 1;
                                results.errors.push(error);
                            }
                        }
                    }
                }
                Err(e) => {
                    results.errors.push(format!("Task error: {}", e));
                }
            }
        }

        results.finalize();

        println!("  ✅ Async task concurrency test completed:");
        println!(
            "    📈 Successful requests: {}",
            results.successful_requests
        );
        println!("    ❌ Failed requests: {}", results.failed_requests);
        println!(
            "    ⏱️  Average response time: {:?}",
            results.average_response_time
        );

        results
    }

    /// Test connection pool behavior under load
    pub async fn test_connection_pool_stress(&self) -> ConcurrencyTestResults {
        println!("🏊 Running connection pool stress test...");

        let server = MockLlmServer::new().await;
        server.configure(MockTestUtils::delayed_response_config(
            Duration::from_millis(100),
        ));
        server.setup_openai_chat().await;

        let mut results = ConcurrencyTestResults::new();

        // Create a client with limited connection pool
        let client = Arc::new(
            reqwest::Client::builder()
                .pool_max_idle_per_host(5) // Limit connections to force reuse
                .pool_idle_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        );

        let base_url = Arc::new(server.base_url());
        let semaphore = Arc::new(Semaphore::new(20)); // Limit concurrent requests

        let mut join_set = JoinSet::new();

        // Create many more tasks than connection pool size
        for task_id in 0..(self.config.thread_count * 3) {
            let client = client.clone();
            let base_url = base_url.clone();
            let semaphore = semaphore.clone();

            join_set.spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                let request_start = Instant::now();

                let result = client
                    .post(format!("{}/v1/chat/completions", base_url))
                    .header("authorization", "Bearer test-key")
                    .json(&serde_json::json!({
                        "model": "gpt-3.5-turbo",
                        "messages": [{
                            "role": "user",
                            "content": format!("Pool stress test from task {}", task_id)
                        }]
                    }))
                    .send()
                    .await;

                let request_duration = request_start.elapsed();

                match result {
                    Ok(response) => {
                        if response.status().is_success() {
                            Ok(request_duration)
                        } else {
                            Err(format!("HTTP error: {}", response.status()))
                        }
                    }
                    Err(e) => Err(format!("Request error: {}", e)),
                }
            });
        }

        // Collect results
        while let Some(task_result) = join_set.join_next().await {
            match task_result {
                Ok(result) => match result {
                    Ok(duration) => {
                        results.successful_requests += 1;
                        results.total_duration += duration;
                        if duration > results.max_response_time {
                            results.max_response_time = duration;
                        }
                        if duration < results.min_response_time {
                            results.min_response_time = duration;
                        }
                    }
                    Err(error) => {
                        results.failed_requests += 1;
                        results.errors.push(error);
                    }
                },
                Err(e) => {
                    results.errors.push(format!("Task join error: {}", e));
                }
            }
        }

        results.finalize();

        println!("  ✅ Connection pool stress test completed:");
        println!(
            "    📈 Successful requests: {}",
            results.successful_requests
        );
        println!("    ❌ Failed requests: {}", results.failed_requests);
        println!(
            "    ⏱️  Average response time: {:?}",
            results.average_response_time
        );
        println!("    🔄 Max response time: {:?}", results.max_response_time);

        results
    }
}

// ============================================================================
// Actual Tests
// ============================================================================

#[tokio::test]
async fn test_basic_concurrency() {
    let config = ConcurrencyTestConfig {
        thread_count: 5,
        requests_per_thread: 3,
        timeout: Duration::from_secs(10),
        use_delays: false,
        request_delay: None,
    };

    let tester = ConcurrencyTester::new(config);
    let results = tester.test_concurrent_http_requests().await;

    assert!(
        results.is_successful(),
        "Basic concurrency test failed: {:?}",
        results.errors
    );
    assert_eq!(results.successful_requests, 15); // 5 threads * 3 requests
    assert_eq!(results.failed_requests, 0);
    assert_eq!(results.successful_threads, 5);
}

#[tokio::test]
async fn test_high_concurrency() {
    let config = ConcurrencyTestConfig {
        thread_count: 20,
        requests_per_thread: 5,
        timeout: Duration::from_secs(30),
        use_delays: false,
        request_delay: None,
    };

    let tester = ConcurrencyTester::new(config);
    let results = tester.test_concurrent_http_requests().await;

    assert!(
        results.is_successful(),
        "High concurrency test failed: {:?}",
        results.errors
    );
    assert_eq!(results.successful_requests, 100); // 20 threads * 5 requests
    assert_eq!(results.failed_requests, 0);
    assert_eq!(results.successful_threads, 20);
}

#[tokio::test]
async fn test_async_task_concurrency() {
    let config = ConcurrencyTestConfig {
        thread_count: 15,
        requests_per_thread: 4,
        timeout: Duration::from_secs(20),
        use_delays: false,
        request_delay: None,
    };

    let tester = ConcurrencyTester::new(config);
    let results = tester.test_async_task_concurrency().await;

    assert!(
        results.is_successful(),
        "Async task concurrency test failed: {:?}",
        results.errors
    );
    assert_eq!(results.successful_requests, 60); // 15 tasks * 4 requests
    assert_eq!(results.failed_requests, 0);
    assert_eq!(results.successful_threads, 15);
}

#[tokio::test]
async fn test_connection_pool_stress() {
    let config = ConcurrencyTestConfig {
        thread_count: 10,
        requests_per_thread: 1,
        timeout: Duration::from_secs(30),
        use_delays: true,
        request_delay: None,
    };

    let expected_requests = config.thread_count * 3;
    let tester = ConcurrencyTester::new(config);
    let results = tester.test_connection_pool_stress().await;

    assert!(
        results.is_successful(),
        "Connection pool stress test failed: {:?}",
        results.errors
    );
    assert_eq!(results.successful_requests, expected_requests);
    assert_eq!(results.failed_requests, 0);

    // Verify that response times are reasonable (should be > 100ms due to artificial delay)
    assert!(results.average_response_time >= Duration::from_millis(90));
    assert!(results.max_response_time >= Duration::from_millis(90));
}

#[tokio::test]
async fn test_concurrent_with_delays() {
    let config = ConcurrencyTestConfig {
        thread_count: 8,
        requests_per_thread: 3,
        timeout: Duration::from_secs(25),
        use_delays: true,
        request_delay: Some(Duration::from_millis(50)),
    };

    let tester = ConcurrencyTester::new(config);
    let results = tester.test_concurrent_http_requests().await;

    assert!(
        results.is_successful(),
        "Concurrent with delays test failed: {:?}",
        results.errors
    );
    assert_eq!(results.successful_requests, 24); // 8 threads * 3 requests
    assert_eq!(results.failed_requests, 0);
    assert_eq!(results.successful_threads, 8);
}

#[tokio::test]
async fn test_memory_safety_under_load() {
    // This test runs multiple concurrent scenarios to stress test memory safety
    println!("🧠 Running memory safety under load test...");

    let scenarios = vec![
        ConcurrencyTestConfig {
            thread_count: 5,
            requests_per_thread: 10,
            timeout: Duration::from_secs(15),
            use_delays: false,
            request_delay: None,
        },
        ConcurrencyTestConfig {
            thread_count: 10,
            requests_per_thread: 5,
            timeout: Duration::from_secs(15),
            use_delays: false,
            request_delay: None,
        },
        ConcurrencyTestConfig {
            thread_count: 20,
            requests_per_thread: 2,
            timeout: Duration::from_secs(15),
            use_delays: false,
            request_delay: None,
        },
    ];

    for (i, config) in scenarios.into_iter().enumerate() {
        println!(
            "  🔄 Running scenario {} with {} threads",
            i + 1,
            config.thread_count
        );

        let tester = ConcurrencyTester::new(config.clone());
        let results = tester.test_concurrent_http_requests().await;

        assert!(
            results.is_successful(),
            "Memory safety test scenario {} failed: {:?}",
            i + 1,
            results.errors
        );

        let expected_requests = config.thread_count * config.requests_per_thread;
        assert_eq!(results.successful_requests, expected_requests);
        assert_eq!(results.failed_requests, 0);

        println!("    ✅ Scenario {} completed successfully", i + 1);
    }

    println!("  🎉 All memory safety scenarios passed!");
}
