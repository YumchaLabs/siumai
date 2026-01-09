//! Aggregator for streaming tests under tests/streaming/.
//! This ensures subdirectory tests are picked up by cargo as a single test target.

// Generic streaming behavior (no provider features)
#[path = "streaming/chat_stream_cancel_test.rs"]
mod chat_stream_cancel_test;
#[path = "streaming/chat_stream_connect_retry_test.rs"]
mod chat_stream_connect_retry_test;
#[path = "streaming/factory_injection_tests.rs"]
mod factory_injection_tests;
#[cfg(feature = "openai")]
#[path = "streaming/factory_non_sse_fallback_test.rs"]
mod factory_non_sse_fallback_test;
#[cfg(feature = "openai")]
#[path = "streaming/http_connect_timeout_retry_test.rs"]
mod http_connect_timeout_retry_test;
#[cfg(feature = "openai")]
#[path = "streaming/http_partial_disconnect_cancel_interaction_test.rs"]
mod http_partial_disconnect_cancel_interaction_test;
#[cfg(feature = "openai")]
#[path = "streaming/http_partial_disconnect_no_done_test.rs"]
mod http_partial_disconnect_no_done_test;
#[path = "streaming/siumai_interceptor_request_assert_test.rs"]
mod siumai_interceptor_request_assert_test;

// OpenAI streaming tests
// Provider-specific streaming request header tests were moved into provider crates
// to keep the facade integration test suite lighter and reduce cross-crate coupling.

// Cross-provider converter invariants
#[cfg(all(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama"
))]
#[path = "streaming/first_event_content_preservation_test.rs"]
mod first_event_content_preservation_test;

#[cfg(all(feature = "openai", feature = "anthropic", feature = "ollama"))]
#[path = "streaming/complete_stream_events_test.rs"]
mod complete_stream_events_test;

#[cfg(all(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama"
))]
#[path = "streaming/streaming_integration_test.rs"]
mod streaming_integration_test;

#[cfg(feature = "openai")]
#[path = "streaming/tool_call_streaming_integration_test.rs"]
mod tool_call_streaming_integration_test;

#[cfg(all(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama"
))]
#[path = "streaming/stream_start_event_test.rs"]
mod stream_start_event_test;
