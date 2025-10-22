use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Stream state tracker for managing stream lifecycle events
///
/// This utility helps track whether a stream has started and ended, ensuring that
/// StreamStart and StreamEnd events are only emitted once per stream.
///
/// **Design Note**: Uses `AtomicBool` instead of `Mutex` because:
/// 1. Streams are processed sequentially (not concurrently), so we don't need locking
/// 2. `AtomicBool` is `Sync`, which is required for `Send + Sync` futures
/// 3. No async overhead - all operations are synchronous
/// 4. `Arc` is needed for sharing state across cloned converters
#[derive(Clone)]
pub struct StreamStateTracker {
    started: Arc<AtomicBool>,
    ended: Arc<AtomicBool>,
}

impl StreamStateTracker {
    /// Create a new stream state tracker
    pub fn new() -> Self {
        Self {
            started: Arc::new(AtomicBool::new(false)),
            ended: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if StreamStart event needs to be emitted
    ///
    /// Returns `true` on the first call, `false` on subsequent calls.
    /// This ensures StreamStart is only emitted once per stream.
    ///
    /// **Note**: This is a synchronous method using atomic operations.
    /// Uses `Relaxed` ordering because we only care about the final state,
    /// not the ordering of operations across threads.
    pub fn needs_stream_start(&self) -> bool {
        !self
            .started
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    }

    /// Mark that StreamEnd has been emitted
    ///
    /// This should be called when a StreamEnd event is emitted from convert_event.
    /// It prevents handle_stream_end from emitting a duplicate StreamEnd.
    ///
    /// **Note**: This is a synchronous method using atomic operations.
    pub fn mark_stream_ended(&self) {
        self.ended.store(true, Ordering::Relaxed);
    }

    /// Check if StreamEnd event needs to be emitted
    ///
    /// Returns `true` if StreamEnd has not been emitted yet, `false` otherwise.
    /// This ensures StreamEnd is only emitted once per stream.
    ///
    /// **Note**: This is a synchronous method using atomic operations.
    /// Uses `Relaxed` ordering because we only care about the final state,
    /// not the ordering of operations across threads.
    pub fn needs_stream_end(&self) -> bool {
        !self
            .ended
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    }

    /// Reset the tracker (useful for testing)
    #[cfg(test)]
    pub fn reset(&self) {
        self.started.store(false, Ordering::Relaxed);
        self.ended.store(false, Ordering::Relaxed);
    }
}

impl Default for StreamStateTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_needs_stream_start() {
        let tracker = StreamStateTracker::new();

        // First call should return true
        assert!(tracker.needs_stream_start());

        // Subsequent calls should return false
        assert!(!tracker.needs_stream_start());
        assert!(!tracker.needs_stream_start());
    }

    #[test]
    fn test_reset() {
        let tracker = StreamStateTracker::new();

        // First call
        assert!(tracker.needs_stream_start());
        assert!(!tracker.needs_stream_start());

        // Reset
        tracker.reset();

        // Should return true again
        assert!(tracker.needs_stream_start());
        assert!(!tracker.needs_stream_start());
    }

    #[test]
    fn test_clone_shares_state() {
        let tracker1 = StreamStateTracker::new();
        let tracker2 = tracker1.clone();

        // First call on tracker1
        assert!(tracker1.needs_stream_start());

        // tracker2 should share the same state
        assert!(!tracker2.needs_stream_start());
    }

    #[test]
    fn test_default() {
        let tracker = StreamStateTracker::default();
        assert!(tracker.needs_stream_start());
        assert!(!tracker.needs_stream_start());
    }

    #[test]
    fn test_mark_stream_ended() {
        let tracker = StreamStateTracker::new();

        // Initially should need stream end
        assert!(tracker.needs_stream_end());

        // After marking, should not need stream end
        tracker.mark_stream_ended();
        assert!(!tracker.needs_stream_end());
    }

    #[test]
    fn test_needs_stream_end() {
        let tracker = StreamStateTracker::new();

        // First call should return true
        assert!(tracker.needs_stream_end());

        // Subsequent calls should return false
        assert!(!tracker.needs_stream_end());
        assert!(!tracker.needs_stream_end());
    }
}
