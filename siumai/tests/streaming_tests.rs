//! Aggregator for streaming tests under tests/streaming/.
//! This ensures subdirectory tests are picked up by cargo as a single test target.

// Generic streaming behavior (no provider features)
#[path = "streaming/chat_stream_cancel_test.rs"]
mod chat_stream_cancel_test;
#[path = "streaming/chat_stream_connect_retry_test.rs"]
mod chat_stream_connect_retry_test;
#[path = "streaming/factory_injection_tests.rs"]
mod factory_injection_tests;
#[path = "streaming/factory_non_sse_fallback_test.rs"]
mod factory_non_sse_fallback_test;
#[path = "streaming/http_connect_timeout_retry_test.rs"]
mod http_connect_timeout_retry_test;
#[path = "streaming/http_partial_disconnect_cancel_interaction_test.rs"]
mod http_partial_disconnect_cancel_interaction_test;
#[path = "streaming/http_partial_disconnect_no_done_test.rs"]
mod http_partial_disconnect_no_done_test;
#[path = "streaming/interceptor_request_assert_test.rs"]
mod interceptor_request_assert_test;
#[path = "streaming/siumai_interceptor_request_assert_test.rs"]
mod siumai_interceptor_request_assert_test;

// OpenAI streaming tests
#[cfg(feature = "openai")]
#[path = "streaming/openai_chat_fixtures_test.rs"]
mod openai_chat_fixtures_test;

#[cfg(feature = "openai")]
#[path = "streaming/openai_streaming_request_headers_test.rs"]
mod openai_streaming_request_headers_test;

#[cfg(feature = "openai")]
#[path = "streaming/openai_responses_fixtures_test.rs"]
mod openai_responses_fixtures_test;

#[cfg(feature = "openai")]
#[path = "streaming/openai_responses_output_text_delta_test.rs"]
mod openai_responses_output_text_delta_test;

#[cfg(feature = "openai")]
#[path = "streaming/openai_responses_streaming_request_test.rs"]
mod openai_responses_streaming_request_test;

#[cfg(feature = "openai")]
#[path = "streaming/openai_compatible_request_headers_test.rs"]
mod openai_compatible_request_headers_test;

#[cfg(feature = "openai")]
#[path = "streaming/openai_compatible_end_to_end_sse_test.rs"]
mod openai_compatible_end_to_end_sse_test;

#[cfg(feature = "openai")]
#[path = "streaming/openai_compatible_finish_reason_no_done_test.rs"]
mod openai_compatible_finish_reason_no_done_test;

#[cfg(feature = "openai")]
#[path = "streaming/openai_compatible_multi_event_test.rs"]
mod openai_compatible_multi_event_test;

#[cfg(feature = "openai")]
#[path = "streaming/openai_compatible_responses_shape_on_chat_test.rs"]
mod openai_compatible_responses_shape_on_chat_test;

// Anthropic streaming tests
#[cfg(feature = "anthropic")]
#[path = "streaming/anthropic_fixtures_test.rs"]
mod anthropic_fixtures_test;

#[cfg(feature = "anthropic")]
#[path = "streaming/anthropic_streaming_request_headers_test.rs"]
mod anthropic_streaming_request_headers_test;

#[cfg(feature = "anthropic")]
#[path = "streaming/vertex_streaming_request_headers_test.rs"]
mod vertex_streaming_request_headers_test;

// Gemini streaming tests
#[cfg(feature = "google")]
#[path = "streaming/gemini_fixtures_test.rs"]
mod gemini_fixtures_test;

#[cfg(feature = "google")]
#[path = "streaming/gemini_streaming_request_headers_test.rs"]
mod gemini_streaming_request_headers_test;

// Ollama streaming tests
#[cfg(feature = "ollama")]
#[path = "streaming/ollama_fixtures_test.rs"]
mod ollama_fixtures_test;

// Groq streaming tests
#[cfg(all(feature = "openai", feature = "groq"))]
#[path = "streaming/groq_fixtures_test.rs"]
mod groq_fixtures_test;

#[cfg(all(feature = "openai", feature = "groq"))]
#[path = "streaming/groq_streaming_request_headers_test.rs"]
mod groq_streaming_request_headers_test;

// xAI streaming tests
#[cfg(all(feature = "openai", feature = "xai"))]
#[path = "streaming/xai_fixtures_test.rs"]
mod xai_fixtures_test;

#[cfg(all(feature = "openai", feature = "xai"))]
#[path = "streaming/xai_streaming_request_headers_test.rs"]
mod xai_streaming_request_headers_test;

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

// Shared streaming test support utilities (loaded once for this crate)
#[path = "support/stream_fixture.rs"]
mod support;
