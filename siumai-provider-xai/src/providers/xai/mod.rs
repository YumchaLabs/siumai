//! Rust-first xAI language-provider surface.

mod language;
pub mod models;
mod provider;

pub use provider::{
    XaiConfigError, XaiCredential, XaiLanguageApi, XaiLanguageModel, XaiProvider,
    XaiProviderBuilder,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// xAI error envelope returned by OpenAI-compatible HTTP endpoints.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct XaiErrorData {
    pub error: XaiErrorPayload,
}

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
