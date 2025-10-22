//! Aggregator for streaming tests under tests/streaming/.
//! This ensures subdirectory tests are picked up by cargo as a single test target.

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
