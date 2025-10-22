use std::sync::Arc;
use tokio::sync::Mutex;

/// Stream state tracker for managing stream lifecycle events
///
/// This utility helps track whether a stream has started, ensuring that
/// StreamStart events are only emitted once per stream.
#[derive(Clone)]
pub struct StreamStateTracker {
    started: Arc<Mutex<bool>>,
}

impl StreamStateTracker {
    /// Create a new stream state tracker
    pub fn new() -> Self {
        Self {
            started: Arc::new(Mutex::new(false)),
        }
    }

    /// Check if StreamStart event needs to be emitted
    ///
    /// Returns `true` on the first call, `false` on subsequent calls.
    /// This ensures StreamStart is only emitted once per stream.
    pub async fn needs_stream_start(&self) -> bool {
        let mut started = self.started.lock().await;
        if !*started {
            *started = true;
            true
        } else {
            false
        }
    }

    /// Reset the tracker (useful for testing)
    #[cfg(test)]
    pub async fn reset(&self) {
        let mut started = self.started.lock().await;
        *started = false;
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

    #[tokio::test]
    async fn test_needs_stream_start() {
        let tracker = StreamStateTracker::new();

        // First call should return true
        assert!(tracker.needs_stream_start().await);

        // Subsequent calls should return false
        assert!(!tracker.needs_stream_start().await);
        assert!(!tracker.needs_stream_start().await);
    }

    #[tokio::test]
    async fn test_reset() {
        let tracker = StreamStateTracker::new();

        // First call
        assert!(tracker.needs_stream_start().await);
        assert!(!tracker.needs_stream_start().await);

        // Reset
        tracker.reset().await;

        // Should return true again
        assert!(tracker.needs_stream_start().await);
        assert!(!tracker.needs_stream_start().await);
    }

    #[tokio::test]
    async fn test_clone_shares_state() {
        let tracker1 = StreamStateTracker::new();
        let tracker2 = tracker1.clone();

        // First call on tracker1
        assert!(tracker1.needs_stream_start().await);

        // tracker2 should share the same state
        assert!(!tracker2.needs_stream_start().await);
    }

    #[tokio::test]
    async fn test_default() {
        let tracker = StreamStateTracker::default();
        assert!(tracker.needs_stream_start().await);
        assert!(!tracker.needs_stream_start().await);
    }
}

