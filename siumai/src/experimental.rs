//! Experimental low-level APIs (advanced use only).
//!
//! This module exposes lower-level building blocks from `siumai-core` (executors, middleware,
//! auth providers, etc.) without making them part of the stable facade surface.
//!
//! Prefer `siumai::prelude::unified::*`, `siumai::hosted_tools::*`, and
//! `siumai::provider_ext::*` unless you are building integrations or custom providers.

pub mod core {
    pub use siumai_core::core::*;
}

/// Protocol bridge contracts (advanced API).
///
/// This exposes bridge decision/report types used by cross-protocol request, response, and stream
/// adapters.
pub mod bridge {
    pub use siumai_bridge::*;
}

/// Non-streaming response encoding utilities (advanced API).
///
/// This exposes protocol-level JSON encoders used by gateways/proxies to re-serialize unified
/// `ChatResponse` objects into provider-native JSON response bodies.
pub mod encoding {
    pub use siumai_core::encoding::{
        JsonEncodeOptions, JsonResponseConverter, encode_chat_response_as_json,
    };
}

/// Streaming utilities (advanced API).
///
/// This exposes low-level building blocks from `siumai-core` that are useful when building
/// gateways/proxies that need to re-serialize streams into provider-native wire formats.
pub mod streaming {
    #[cfg(feature = "openai")]
    pub use siumai_bridge::stream::OpenAiResponsesStreamPartsBridge;
    pub use siumai_core::streaming::*;
}

/// Custom provider API (advanced).
pub mod custom_provider {
    pub use siumai_core::custom_provider::*;
}

/// Authentication helpers (advanced API).
///
/// Generic token provider contracts remain core-owned. Provider-specific auth helpers are
/// re-exported from the provider crate that owns them.
pub mod auth {
    pub use siumai_core::auth::{StaticTokenProvider, TokenProvider};

    #[cfg(feature = "gcp")]
    pub use siumai_provider_google_vertex::auth::{adc, service_account};

    #[cfg(feature = "google-vertex")]
    pub use siumai_provider_google_vertex::auth::vertex;
}

/// Provider implementation internals (advanced API).
///
/// If you find yourself importing from here, consider depending on the relevant provider crate
/// directly instead of going through the facade.
pub mod providers {
    pub use siumai_core as core;

    #[cfg(feature = "bedrock")]
    pub use siumai_provider_amazon_bedrock as amazon_bedrock;
    #[cfg(feature = "anthropic")]
    pub use siumai_provider_anthropic as anthropic;
    #[cfg(feature = "azure")]
    pub use siumai_provider_azure as azure;
    #[cfg(feature = "cohere")]
    pub use siumai_provider_cohere as cohere;
    #[cfg(feature = "deepseek")]
    pub use siumai_provider_deepseek as deepseek;
    #[cfg(feature = "google")]
    pub use siumai_provider_gemini as gemini;
    #[cfg(feature = "google-vertex")]
    pub use siumai_provider_google_vertex as google_vertex;
    #[cfg(feature = "groq")]
    pub use siumai_provider_groq as groq;
    #[cfg(feature = "minimax")]
    pub use siumai_provider_minimax as minimax;
    #[cfg(feature = "ollama")]
    pub use siumai_provider_ollama as ollama;
    #[cfg(feature = "openai")]
    pub use siumai_provider_openai as openai;
    #[cfg(feature = "togetherai")]
    pub use siumai_provider_togetherai as togetherai;
    #[cfg(feature = "xai")]
    pub use siumai_provider_xai as xai;
}

/// Protocol mapping and adapter helpers (advanced API).
///
/// Prefer `siumai::prelude::unified::*` unless you are building integrations or custom providers.
pub mod standards {
    #[cfg(feature = "bedrock")]
    pub use siumai_provider_amazon_bedrock::standards::bedrock;
    #[cfg(feature = "anthropic")]
    pub use siumai_provider_anthropic::standards::anthropic;
    #[cfg(feature = "cohere")]
    pub use siumai_provider_cohere::standards::cohere;
    #[cfg(feature = "google")]
    pub use siumai_provider_gemini::standards::gemini;
    #[cfg(feature = "ollama")]
    pub use siumai_provider_ollama::standards::ollama;
    #[cfg(feature = "openai")]
    pub use siumai_provider_openai::standards::openai;
    #[cfg(feature = "togetherai")]
    pub use siumai_provider_togetherai::standards::togetherai;
}

/// Legacy generic-client dynamic-dispatch types.
///
/// Prefer `siumai::compat::client` for migration-oriented imports.
pub mod client {
    pub use crate::compat::client::{ClientWrapper, LlmClient};
}

/// Runtime default configuration values.
pub mod defaults {
    pub use siumai_core::defaults::*;
}

/// Provider-agnostic execution building blocks.
pub mod execution {
    pub use siumai_core::execution::*;
}

/// Observability contracts and event emitters.
pub mod observability {
    pub use siumai_core::observability::*;
}

/// Provider-agnostic parameter validation helpers.
pub mod params {
    pub use siumai_core::params::*;
}

/// Low-level retry primitives.
pub mod retry {
    pub use siumai_core::retry::*;
}

/// Core runtime and provider-utils compatibility utility modules.
pub mod utils {
    pub use siumai_core::utils::{
        StreamingToolCallDelta, StreamingToolCallFunctionDelta, StreamingToolCallTracker,
        StreamingToolCallTrackerOptions, StreamingToolCallTypeValidation, cancel, delay,
        is_abort_error, streaming_tool_call,
    };
    pub use siumai_provider_utils::*;
}
