//! Aggregator for standards layer tests under tests/standards/.

// OpenAI standard layer tests
#[cfg(feature = "openai")]
#[path = "standards/openai_adapter_sse_test.rs"]
mod openai_adapter_sse_test;

// Anthropic standard layer tests
#[cfg(feature = "anthropic")]
#[path = "standards/anthropic_adapter_sse_test.rs"]
mod anthropic_adapter_sse_test;
