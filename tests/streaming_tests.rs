/// Comprehensive streaming tests for the siumai library
///
/// This module includes all streaming-related tests to ensure the multi-event
/// emission architecture works correctly across all providers.

// Include all streaming test modules
mod streaming {
    pub mod complete_stream_events_test;
    pub mod first_event_content_preservation_test;
    pub mod stream_start_event_test;
    pub mod streaming_integration_test;
    pub mod tool_call_streaming_integration_test;
}

// Re-export all tests for easy access
pub use streaming::*;
