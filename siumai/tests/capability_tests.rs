//! Aggregator for capability tests under `tests/capabilities/`.
//!
//! We keep these tests in a subfolder for organization, and include them here so that
//! `cargo test` (and scripts like `scripts/run_integration_tests.sh`) can discover them.

#[cfg(all(feature = "openai", feature = "groq"))]
#[path = "capabilities/audio_capability_test.rs"]
mod audio_capability_test;

#[cfg(all(feature = "openai", feature = "anthropic", feature = "google", feature = "xai"))]
#[path = "capabilities/vision_capability_test.rs"]
mod vision_capability_test;

#[cfg(all(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "xai",
    feature = "ollama"
))]
#[path = "capabilities/tool_capability_test.rs"]
mod tool_capability_test;

#[cfg(feature = "openai")]
#[path = "capabilities/image_generation_test.rs"]
mod image_generation_test;

#[cfg(all(feature = "openai", feature = "google", feature = "ollama"))]
#[path = "capabilities/embedding_integration_tests.rs"]
mod embedding_integration_tests;

