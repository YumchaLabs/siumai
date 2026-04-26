//! Cancellation utilities
//!
//! Provides first-class cancellation handles for streams and long-running operations.

use crate::error::LlmError;
pub use crate::types::CancelHandle;
use std::future::Future;
use std::time::Duration;

// Stream-based cancellation is implemented via async_stream to avoid pin projection.

/// Make a ChatStream cancellable and return its cancel handle.
pub fn make_cancellable_stream(
    stream: crate::streaming::ChatStream,
) -> (crate::streaming::ChatStream, CancelHandle) {
    let handle = CancelHandle::new();
    let token = handle.token();
    let mut inner = stream;
    let s = async_stream::stream! {
        use futures::StreamExt;
        loop {
            tokio::select! {
                _ = token.cancelled() => break,
                item = inner.next() => {
                    let Some(item) = item else { break };
                    yield item;
                }
            }
        }
    };
    (Box::pin(s), handle)
}

/// Create a `ChatStreamHandle` whose cancellation can abort both:
/// - the streaming request handshake (connect/send/headers), and
/// - the subsequent stream consumption.
///
/// This requires the caller to supply a `'static` future (typically by cloning an `Arc`-based
/// client/executor into the future), because `ChatStream` is stored as a `'static` trait object.
pub fn make_cancellable_stream_handle_from_future<F>(
    future: F,
) -> crate::streaming::ChatStreamHandle
where
    F: Future<Output = Result<crate::streaming::ChatStream, crate::error::LlmError>>
        + Send
        + 'static,
{
    let cancel = CancelHandle::new();
    let token = cancel.token();
    let future = std::sync::Mutex::new(Some(future));

    let s = async_stream::stream! {
        use futures::StreamExt;

        let res = tokio::select! {
            _ = token.cancelled() => return,
            res = async {
                let fut = {
                    let mut guard = future.lock().expect("handshake future mutex poisoned");
                    guard
                        .take()
                        .expect("handshake future should only be awaited once")
                };
                fut.await
            } => res,
        };

        let mut inner = match res {
            Ok(s) => s,
            Err(e) => {
                yield Err(e);
                return;
            }
        };

        loop {
            tokio::select! {
                _ = token.cancelled() => break,
                item = inner.next() => {
                    let Some(item) = item else { break };
                    yield item;
                }
            }
        }
    };

    crate::streaming::ChatStreamHandle {
        stream: Box::pin(s),
        cancel,
    }
}

/// Create a `ChatStreamHandle` whose cancellation can abort both:
/// - the streaming request handshake (connect/send/headers), and
/// - the subsequent stream consumption,
///
/// while also delegating cancellation to an inner `ChatStreamHandle` returned by the future.
///
/// This is useful for wrappers (registry/unified clients) that want to preserve provider-specific
/// cancellation behavior (e.g. best-effort remote cancellation) while still being able to abort
/// their own handshake/retry logic.
pub fn make_cancellable_stream_handle_from_handle_future<F>(
    future: F,
) -> crate::streaming::ChatStreamHandle
where
    F: Future<Output = Result<crate::streaming::ChatStreamHandle, crate::error::LlmError>>
        + Send
        + 'static,
{
    let cancel = CancelHandle::new();
    let token = cancel.token();
    let future = std::sync::Mutex::new(Some(future));

    let s = async_stream::stream! {
        use futures::StreamExt;

        let res = tokio::select! {
            _ = token.cancelled() => return,
            res = async {
                let fut = {
                    let mut guard = future.lock().expect("handshake future mutex poisoned");
                    guard
                        .take()
                        .expect("handshake future should only be awaited once")
                };
                fut.await
            } => res,
        };

        let inner = match res {
            Ok(h) => h,
            Err(e) => {
                yield Err(e);
                return;
            }
        };

        let crate::streaming::ChatStreamHandle { stream: inner_stream, cancel: inner_cancel } = inner;

        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        struct DoneOnDrop(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for DoneOnDrop {
            fn drop(&mut self) {
                if let Some(tx) = self.0.take() {
                    let _ = tx.send(());
                }
            }
        }

        let cancel_for_task = inner_cancel.clone();
        let token_for_task = token.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = token_for_task.cancelled() => cancel_for_task.cancel(),
                _ = done_rx => {
                    // If the stream was dropped because the caller cancelled, still propagate to the inner handle.
                    if token_for_task.is_cancelled() {
                        cancel_for_task.cancel();
                    }
                }
            }
        });

        let mut inner = inner_stream;
        let _done = DoneOnDrop(Some(done_tx));
        loop {
            tokio::select! {
                _ = token.cancelled() => break,
                item = inner.next() => {
                    let Some(item) = item else { break };
                    yield item;
                }
            }
        }
    };

    crate::streaming::ChatStreamHandle {
        stream: Box::pin(s),
        cancel,
    }
}

/// Create a standalone cancel handle that can be shared across tasks.
/// Useful for orchestrating complex pipelines which need a single abort signal.
pub fn new_cancel_handle() -> CancelHandle {
    CancelHandle::new()
}

/// Wait for the requested delay, optionally observing a cancellation handle.
///
/// This is the Rust equivalent of provider-utils `delay(...)`: missing delays resolve
/// immediately, and cancellation returns an abort-compatible timeout error.
pub async fn delay(
    delay_in_ms: Option<u64>,
    abort_signal: Option<&CancelHandle>,
) -> Result<(), LlmError> {
    let Some(delay_in_ms) = delay_in_ms else {
        return Ok(());
    };

    if let Some(abort_signal) = abort_signal {
        if abort_signal.is_cancelled() {
            return Err(delay_abort_error());
        }

        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(delay_in_ms)) => Ok(()),
            _ = abort_signal.cancelled() => Err(delay_abort_error()),
        }
    } else {
        tokio::time::sleep(Duration::from_millis(delay_in_ms)).await;
        Ok(())
    }
}

/// Check whether an error is abort-compatible.
///
/// AI SDK treats `AbortError`, `ResponseAborted`, and `TimeoutError` as abort errors. Rust does
/// not have browser `DOMException.name`, so this helper classifies native timeout errors directly
/// and checks textual/provider error names for the same stable markers.
pub fn is_abort_error(error: &LlmError) -> bool {
    match error {
        LlmError::TimeoutError(_) => true,
        LlmError::HttpError(message)
        | LlmError::StreamError(message)
        | LlmError::ConnectionError(message)
        | LlmError::Other(message) => contains_abort_error_name(message),
        LlmError::ProviderError {
            message,
            error_code,
            ..
        } => {
            error_code.as_deref().is_some_and(is_abort_error_name)
                || contains_abort_error_name(message)
        }
        LlmError::ContextualError {
            source_error: Some(source),
            ..
        } => is_abort_error(source),
        _ => false,
    }
}

fn delay_abort_error() -> LlmError {
    LlmError::TimeoutError("Delay was aborted".to_string())
}

fn contains_abort_error_name(message: &str) -> bool {
    message
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(is_abort_error_name)
}

fn is_abort_error_name(name: &str) -> bool {
    matches!(name, "AbortError" | "ResponseAborted" | "TimeoutError")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn cancel_wakes_pending_next_immediately() {
        // A stream that never yields and never ends.
        let pending: crate::streaming::ChatStream = Box::pin(futures_util::stream::pending());
        let (mut s, cancel) = make_cancellable_stream(pending);

        let waiter = tokio::spawn(async move { s.next().await });

        // Give the task a chance to poll and block on `next()`.
        tokio::task::yield_now().await;

        cancel.cancel();

        let out = tokio::time::timeout(std::time::Duration::from_millis(200), waiter)
            .await
            .expect("cancel should wake the waiting task")
            .expect("task ok");

        assert!(out.is_none());
    }

    #[tokio::test]
    async fn delay_resolves_immediately_when_missing() {
        delay(None, None)
            .await
            .expect("missing delay should resolve");
    }

    #[tokio::test]
    async fn delay_waits_for_duration() {
        let started = tokio::time::Instant::now();
        delay(Some(5), None).await.expect("delay should resolve");
        assert!(started.elapsed() >= std::time::Duration::from_millis(5));
    }

    #[tokio::test]
    async fn delay_observes_cancel_handle() {
        let cancel = new_cancel_handle();
        cancel.cancel();

        let err = delay(Some(1_000), Some(&cancel))
            .await
            .expect_err("cancelled delay should fail");

        assert!(is_abort_error(&err));
    }

    #[test]
    fn abort_error_detection_matches_ai_sdk_names() {
        assert!(is_abort_error(&LlmError::TimeoutError(
            "operation timed out".to_string(),
        )));
        assert!(is_abort_error(&LlmError::Other(
            "DOMException AbortError: aborted".to_string(),
        )));
        assert!(is_abort_error(&LlmError::provider_error_with_code(
            "next",
            "response aborted",
            "ResponseAborted",
        )));
        assert!(!is_abort_error(&LlmError::InvalidInput(
            "AbortError is not an input error name".to_string(),
        )));
    }
}
