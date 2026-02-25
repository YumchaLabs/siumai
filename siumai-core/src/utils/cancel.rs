//! Cancellation utilities
//!
//! Provides first-class cancellation handles for streams and long-running operations.

use std::future::Future;
use tokio_util::sync::CancellationToken;

/// A handle that can be used to request cancellation.
#[derive(Clone, Debug)]
pub struct CancelHandle {
    token: CancellationToken,
}

impl CancelHandle {
    /// Create a new cancel handle.
    fn new() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }

    /// Request cancellation. Any wrapped streams/futures observing this handle
    /// will stop as soon as possible. Dropping the cancelled stream will close
    /// the underlying HTTP connection so providers stop generating tokens.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Check if cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// A future that resolves when cancellation is requested.
    pub fn cancelled(&self) -> tokio_util::sync::WaitForCancellationFuture<'_> {
        self.token.cancelled()
    }
}

// Stream-based cancellation is implemented via async_stream to avoid pin projection.

/// Make a ChatStream cancellable and return its cancel handle.
pub fn make_cancellable_stream(
    stream: crate::streaming::ChatStream,
) -> (crate::streaming::ChatStream, CancelHandle) {
    let handle = CancelHandle::new();
    let token = handle.token.clone();
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
    let token = cancel.token.clone();
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
    let token = cancel.token.clone();
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
}
