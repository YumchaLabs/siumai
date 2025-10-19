//! Cancellation utilities
//!
//! Provides first-class cancellation handles for streams and long-running operations.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// A handle that can be used to request cancellation.
#[derive(Clone, Debug)]
pub struct CancelHandle {
    flag: Arc<AtomicBool>,
}

impl CancelHandle {
    /// Create a new cancel handle with a shared flag.
    fn new(flag: Arc<AtomicBool>) -> Self {
        Self { flag }
    }

    /// Request cancellation. Any wrapped streams/futures observing this handle
    /// will stop as soon as possible. Dropping the cancelled stream will close
    /// the underlying HTTP connection so providers stop generating tokens.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

// Stream-based cancellation is implemented via async_stream to avoid pin projection.

/// Make a ChatStream cancellable and return its cancel handle.
pub fn make_cancellable_stream(
    stream: crate::stream::ChatStream,
) -> (crate::stream::ChatStream, CancelHandle) {
    let flag = Arc::new(AtomicBool::new(false));
    let handle = CancelHandle::new(flag.clone());
    // Implement the wrapper as a manual stream using async_stream! to avoid pin gymnastics
    let wrapped_flag = flag.clone();
    let mut inner = stream;
    let s = async_stream::stream! {
        use futures::StreamExt;
        while let Some(item) = inner.next().await {
            if wrapped_flag.load(Ordering::SeqCst) { break; }
            yield item;
        }
    };
    (Box::pin(s), handle)
}

/// Create a standalone cancel handle that can be shared across tasks.
/// Useful for orchestrating complex pipelines which need a single abort signal.
pub fn new_cancel_handle() -> CancelHandle {
    CancelHandle::new(Arc::new(AtomicBool::new(false)))
}
