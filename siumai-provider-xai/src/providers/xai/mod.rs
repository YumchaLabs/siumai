//! `xAI` Provider Module
//!
//! Thin wrapper around the OpenAI-compatible vendor implementation with provider-owned entry types.
//!
//! # Architecture
//! - `builder.rs` - Builder that delegates to `openai().compatible("xai")`
//! - `config.rs`  - Provider-owned config-first surface
//! - `client.rs`  - Provider-owned client wrapper
//! - `files.rs`   - Provider-owned files helper for upload/manage routes
//! - `models.rs`  - Built-in model catalog (fallback)
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::models;
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Provider::xai()
//!         .api_key("your-api-key")
//!         .model(models::xai::GROK_3_LATEST)
//!         .build()
//!         .await?;
//!
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!
//!     Ok(())
//! }
//! ```

mod audio;
pub mod builder;
mod client;
pub mod config;
/// xAI extension APIs (non-unified surface)
pub mod ext;
mod files;
mod http;
mod image;
pub mod models;
pub mod settings;
mod video;

pub use builder::XaiBuilder;
pub use client::XaiClient;
pub use config::XaiConfig;
pub use settings::XaiProviderSettings;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// AI SDK-aligned xAI error envelope.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct XaiErrorData {
    pub error: XaiErrorPayload,
}

/// AI SDK-aligned xAI error payload.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct XaiErrorPayload {
    pub message: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<serde_json::Value>,
}

/// AI SDK-style video model id alias.
///
/// Rust keeps model ids as plain strings on the stable provider surface.
pub type XaiVideoModelId = String;

// Provider-owned typed options live at the crate root; re-export them under the provider path.
#[allow(deprecated)]
pub use crate::provider_options::{
    NewsSearchSource, RssSearchSource, SearchMode, SearchSource, WebSearchSource, XSearchSource,
    XaiChatOptions, XaiChatReasoningEffort, XaiFilesOptions, XaiImageModelOptions, XaiImageOptions,
    XaiImageProviderOptions, XaiImageQuality, XaiImageResolution, XaiLanguageModelChatOptions,
    XaiLanguageModelResponsesOptions, XaiOptions, XaiProviderOptions, XaiReasoningSummary,
    XaiResponseInclude, XaiResponsesOptions, XaiResponsesProviderOptions,
    XaiResponsesReasoningEffort, XaiSearchParameters, XaiTtsOptions, XaiVideoMode,
    XaiVideoModelOptions, XaiVideoOptions, XaiVideoProviderOptions, XaiVideoResolution,
};
