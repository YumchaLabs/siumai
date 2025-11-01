//! Aggregator for streaming tests under tests/streaming/.
//! This ensures subdirectory tests are picked up by cargo as a single test target.

// OpenAI streaming tests
#[cfg(feature = "openai")]
#[path = "streaming/openai_chat_fixtures_test.rs"]
mod openai_chat_fixtures_test;

// Anthropic streaming tests
#[cfg(feature = "anthropic")]
#[path = "streaming/anthropic_fixtures_test.rs"]
mod anthropic_fixtures_test;

// Gemini streaming tests
#[cfg(feature = "google")]
#[path = "streaming/gemini_fixtures_test.rs"]
mod gemini_fixtures_test;

// Ollama streaming tests
#[cfg(feature = "ollama")]
#[path = "streaming/ollama_fixtures_test.rs"]
mod ollama_fixtures_test;

// Groq streaming tests
#[cfg(feature = "groq")]
#[path = "streaming/groq_fixtures_test.rs"]
mod groq_fixtures_test;

// xAI streaming tests
#[cfg(feature = "xai")]
#[path = "streaming/xai_fixtures_test.rs"]
mod xai_fixtures_test;

// Shared streaming test support utilities (loaded once for this crate)
#[path = "support/stream_fixture.rs"]
mod support;
