//! Aggregator for mock API tests under tests/mock_api/.
//!
//! These tests simulate provider APIs with local mock servers and validate request/response wiring.

#[cfg(feature = "openai")]
#[path = "mock_api/openai_mock_api_test.rs"]
mod openai_mock_api_test;

#[cfg(feature = "anthropic")]
#[path = "mock_api/anthropic_mock_api_test.rs"]
mod anthropic_mock_api_test;

#[cfg(feature = "google")]
#[path = "mock_api/gemini_mock_api_test.rs"]
mod gemini_mock_api_test;

#[cfg(feature = "groq")]
#[path = "mock_api/groq_mock_api_test.rs"]
mod groq_mock_api_test;

#[cfg(feature = "ollama")]
#[path = "mock_api/ollama_mock_api_test.rs"]
mod ollama_mock_api_test;

#[cfg(feature = "xai")]
#[path = "mock_api/xai_mock_api_test.rs"]
mod xai_mock_api_test;

#[cfg(feature = "minimaxi")]
#[path = "mock_api/minimaxi_mock_api_test.rs"]
mod minimaxi_mock_api_test;
