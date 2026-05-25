//! Compatibility surface for legacy, method-style APIs.
//!
//! This module exists to keep older code building while the recommended invocation style
//! moves to the Rust-first model-family APIs:
//! - `siumai::text::*`
//! - `siumai::embedding::*`
//! - `siumai::image::*`
//! - `siumai::rerank::*`
//! - `siumai::speech::*`
//! - `siumai::transcription::*`
//!
//! Recommended construction for new code is registry/config-first:
//! - `registry::global().language_model("openai:gpt-4o-mini")?`
//! - `OpenAiClient::from_config(OpenAiConfig { .. })?`
//!
//! Builder-style construction (`Siumai::builder()` / `Provider::<provider>()`) remains available
//! as a compatibility convenience.
//!
//! ## Time-bounded compatibility promise
//!
//! The `siumai::compat` module is intended to be **temporary**.
//!
//! Planned removal target: **no earlier than `0.12.0`**.
//!
//! Until then, new code should prefer:
//!
//! - `siumai::prelude::unified::registry::global().language_model("provider:model")?`
//! - config-first provider clients (`*Client::from_config(...)`)

/// Legacy unified interface entry type.
pub use siumai_registry::provider::Siumai;

/// Legacy unified builder (provider-agnostic construction).
pub use siumai_registry::provider::SiumaiBuilder;

/// Legacy provider-specific builder convenience entry type.
///
/// Prefer registry model handles or config-first provider clients for stable construction.
mod provider;
pub use provider::Provider;

/// Legacy streaming tool-call compatibility helpers.
///
/// Stable application code should consume model-family streams instead of constructing indexed
/// provider streaming deltas manually.
pub use siumai_core::utils::{
    StreamingToolCallDelta, StreamingToolCallFunctionDelta, StreamingToolCallTracker,
    StreamingToolCallTrackerOptions, StreamingToolCallTypeValidation,
};

/// Deprecated AI SDK compatibility aliases and helper spellings.
///
/// Prefer the stable non-experimental names from `siumai::prelude::unified::*` where available.
#[allow(deprecated)]
pub use siumai_core::types::{
    CallSettings, Experimental_GenerateImageResult, Experimental_GeneratedImage,
    Experimental_LanguageModelStreamPart, Experimental_SpeechResult,
    Experimental_TranscriptionResult, ExperimentalLanguageModelStreamPart,
    experimental_filter_active_tools, step_count_is,
};

/// Narrow legacy type namespace for migration-only imports.
///
/// Prefer `siumai::prelude::unified::*`, `siumai::prelude::extensions::*`, or the owning
/// family/provider module for new code.
pub mod types {
    pub use siumai_core::types::{ChatMessage, StopCondition, Tool, Warning};

    /// Historical catch-all type namespace for last-resort migrations.
    ///
    /// Prefer the curated `siumai::compat::types::*` surface or stable owning modules before
    /// importing this broad mirror.
    pub mod legacy_all {
        pub use siumai_core::types::*;
    }
}

/// Legacy builder base types (provider builder internals).
///
/// Prefer using provider builders directly (e.g. `Provider::openai()`) unless you are
/// implementing a custom provider.
pub mod builder {
    pub use siumai_core::builder::*;
}

/// Legacy generic-client dynamic-dispatch types.
///
/// New code should prefer model-family traits and registry handles. This module is the explicit
/// migration path for code that still needs `LlmClient` or `ClientWrapper`.
pub mod client {
    pub use siumai_core::compat::client::{ClientWrapper, LlmClient};
}

/// Legacy chat content payloads.
///
/// Prefer `siumai::prelude::unified::{ModelMessage, UserContentPart, AssistantContentPart,
/// ToolContentPart}` for request input and `GenerateTextContentPart` / output-part carriers for
/// response output. Use this module when migration code intentionally needs the serde-compatible
/// legacy `ContentPart` carrier.
pub mod content {
    pub use siumai_core::compat::content::*;
}
