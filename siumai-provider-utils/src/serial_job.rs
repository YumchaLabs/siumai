//! AI SDK-style serial async job executor.

use std::future::Future;
use std::sync::Arc;

use tokio::sync::Mutex;

/// Executes async jobs one at a time in submission order.
///
/// This is the Rust equivalent of AI SDK `SerialJobExecutor`: callers can enqueue concurrent
/// async work through [`run`](Self::run), but only one job future is polled at a time.
#[derive(Debug, Clone, Default)]
pub struct SerialJobExecutor {
    lock: Arc<Mutex<()>>,
}

impl SerialJobExecutor {
    /// Create an empty serial job executor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run a job after all previously submitted jobs have completed.
    pub async fn run<F, Fut, T>(&self, job: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let _guard = self.lock.lock().await;
        job().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::Mutex;

    use super::*;

    #[tokio::test]
    async fn serial_job_executor_runs_jobs_in_submission_order() {
        let executor = SerialJobExecutor::new();
        let observed = Arc::new(Mutex::new(Vec::new()));

        let first_observed = Arc::clone(&observed);
        let first_executor = executor.clone();
        let first = tokio::spawn(async move {
            first_executor
                .run(|| async move {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    first_observed.lock().await.push(1);
                    "first"
                })
                .await
        });

        let second_observed = Arc::clone(&observed);
        let second_executor = executor.clone();
        let second = tokio::spawn(async move {
            second_executor
                .run(|| async move {
                    second_observed.lock().await.push(2);
                    "second"
                })
                .await
        });

        assert_eq!(first.await.expect("first task"), "first");
        assert_eq!(second.await.expect("second task"), "second");
        assert_eq!(*observed.lock().await, vec![1, 2]);
    }

    #[tokio::test]
    async fn serial_job_executor_returns_job_errors() {
        let executor = SerialJobExecutor::new();
        let result = executor
            .run(|| async { Result::<(), &str>::Err("job failed") })
            .await;

        assert_eq!(result, Err("job failed"));
    }
}
