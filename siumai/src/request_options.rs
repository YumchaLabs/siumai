//! Internal adapter for AI SDK-style `RequestOptions` on facade helper calls.

use crate::retry_api::{RetryOptions, retry_with};
use futures_util::StreamExt;
use siumai_core::error::LlmError;
use siumai_core::streaming::ChatStream;
use siumai_core::types::{CancelHandle, RequestOptions};
use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;

/// AI SDK defaults `maxRetries` to 2.
pub(crate) const AI_SDK_DEFAULT_MAX_RETRIES: u32 = 2;

#[derive(Debug, Clone, Default)]
pub(crate) struct EffectiveRequestOptions {
    pub retry: Option<RetryOptions>,
    pub timeout: Option<Duration>,
    pub headers: HashMap<String, String>,
    pub abort_signal: Option<CancelHandle>,
}

impl EffectiveRequestOptions {
    pub fn from_parts(
        request_options: Option<RequestOptions>,
        retry: Option<RetryOptions>,
        timeout: Option<Duration>,
        headers: HashMap<String, String>,
    ) -> Self {
        let request_options = request_options.as_ref();
        let request_headers = request_options
            .map(RequestOptions::effective_headers)
            .unwrap_or_default();

        let mut effective_headers = request_headers;
        effective_headers.extend(headers);

        Self {
            retry: retry.or_else(|| request_options.map(retry_options_from_request_options)),
            timeout: timeout.or_else(|| request_options.and_then(RequestOptions::total_timeout)),
            headers: effective_headers,
            abort_signal: request_options.and_then(|options| options.abort_signal.clone()),
        }
    }

    pub fn retry(&self) -> Option<RetryOptions> {
        self.retry.clone()
    }

    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    pub fn headers(&self) -> HashMap<String, String> {
        self.headers.clone()
    }

    pub fn abort_signal(&self) -> Option<CancelHandle> {
        self.abort_signal.clone()
    }
}

pub(crate) fn retry_options_from_request_options(options: &RequestOptions) -> RetryOptions {
    let max_retries = options.max_retries.unwrap_or(AI_SDK_DEFAULT_MAX_RETRIES);
    RetryOptions::policy_default().with_max_attempts(max_retries.saturating_add(1).max(1))
}

pub(crate) async fn retry_or_call_with_abort<F, Fut, T>(
    retry: Option<RetryOptions>,
    abort_signal: Option<CancelHandle>,
    operation: F,
) -> Result<T, LlmError>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = Result<T, LlmError>> + Send,
    T: Send,
{
    run_with_abort(abort_signal, async move {
        if let Some(retry) = retry {
            retry_with(operation, retry).await
        } else {
            operation().await
        }
    })
    .await
}

pub(crate) async fn run_with_abort<F, T>(
    abort_signal: Option<CancelHandle>,
    future: F,
) -> Result<T, LlmError>
where
    F: Future<Output = Result<T, LlmError>>,
{
    if let Some(abort_signal) = abort_signal {
        tokio::select! {
            _ = abort_signal.cancelled() => Err(request_aborted_error()),
            result = future => result,
        }
    } else {
        future.await
    }
}

pub(crate) fn wrap_stream_with_abort(
    stream: ChatStream,
    abort_signal: Option<CancelHandle>,
) -> ChatStream {
    let Some(abort_signal) = abort_signal else {
        return stream;
    };

    let token = abort_signal.token();
    let mut inner = stream;
    Box::pin(async_stream::stream! {
        loop {
            tokio::select! {
                _ = token.cancelled() => break,
                item = inner.next() => {
                    let Some(item) = item else { break };
                    yield item;
                }
            }
        }
    })
}

pub(crate) fn link_stream_handle_abort(
    handle: siumai_core::streaming::ChatStreamHandle,
    abort_signal: Option<CancelHandle>,
) -> siumai_core::streaming::ChatStreamHandle {
    let Some(abort_signal) = abort_signal else {
        return handle;
    };

    let cancel = handle.cancel.clone();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        tokio::select! {
            _ = abort_signal.cancelled() => cancel.cancel(),
            _ = done_rx => {}
        }
    });

    struct DoneOnDrop(Option<tokio::sync::oneshot::Sender<()>>);

    impl Drop for DoneOnDrop {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
        }
    }

    let mut inner = handle.stream;
    let stream = Box::pin(async_stream::stream! {
        let _done = DoneOnDrop(Some(done_tx));
        while let Some(item) = inner.next().await {
            yield item;
        }
    });

    siumai_core::streaming::ChatStreamHandle {
        stream,
        cancel: handle.cancel,
    }
}

fn request_aborted_error() -> LlmError {
    LlmError::TimeoutError("Request aborted by RequestOptions.abort_signal".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use siumai_core::types::TimeoutConfiguration;

    #[test]
    fn request_options_default_retries_match_ai_sdk() {
        let request_options = RequestOptions::new();
        let retry = retry_options_from_request_options(&request_options);
        let policy = retry.policy.expect("policy retry");
        assert_eq!(policy.max_attempts, 3);
    }

    #[test]
    fn explicit_max_retries_zero_means_one_attempt() {
        let request_options = RequestOptions::new().with_max_retries(0);
        let retry = retry_options_from_request_options(&request_options);
        let policy = retry.policy.expect("policy retry");
        assert_eq!(policy.max_attempts, 1);
    }

    #[test]
    fn legacy_headers_and_timeout_override_request_options() {
        let request_options = RequestOptions::new()
            .with_header("x-source", "request-options")
            .with_header("x-only-request", "true")
            .with_timeout(TimeoutConfiguration::from_millis(500));

        let mut headers = HashMap::new();
        headers.insert("x-source".to_string(), "legacy".to_string());

        let effective = EffectiveRequestOptions::from_parts(
            Some(request_options),
            None,
            Some(Duration::from_secs(2)),
            headers,
        );

        assert_eq!(effective.timeout(), Some(Duration::from_secs(2)));
        assert_eq!(
            effective.headers.get("x-source"),
            Some(&"legacy".to_string())
        );
        assert_eq!(
            effective.headers.get("x-only-request"),
            Some(&"true".to_string())
        );
    }

    #[tokio::test]
    async fn run_with_abort_returns_early() {
        let abort = CancelHandle::new();
        abort.cancel();

        let err = run_with_abort(Some(abort), async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok::<_, LlmError>(())
        })
        .await
        .expect_err("aborted");

        assert!(matches!(err, LlmError::TimeoutError(_)));
    }

    #[tokio::test]
    async fn wrapped_stream_stops_after_abort() {
        let abort = CancelHandle::new();
        let stream: ChatStream = Box::pin(futures_util::stream::pending());
        let mut stream = wrap_stream_with_abort(stream, Some(abort.clone()));

        let waiter = tokio::spawn(async move { stream.next().await });
        tokio::task::yield_now().await;
        abort.cancel();

        let item = tokio::time::timeout(Duration::from_millis(200), waiter)
            .await
            .expect("abort should wake stream")
            .expect("task should not panic");
        assert!(item.is_none());
    }

    #[tokio::test]
    async fn linked_stream_handle_propagates_abort() {
        let abort = CancelHandle::new();
        let inner_cancel = CancelHandle::new();
        let handle = siumai_core::streaming::ChatStreamHandle {
            stream: Box::pin(futures_util::stream::pending()),
            cancel: inner_cancel.clone(),
        };

        let linked = link_stream_handle_abort(handle, Some(abort.clone()));
        abort.cancel();

        tokio::time::timeout(Duration::from_millis(200), async {
            while !linked.cancel.is_cancelled() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("abort should propagate to the stream handle");
    }
}
