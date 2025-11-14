//! Core provider specification traits.
//!
//! This module defines provider-agnostic specification traits that can be
//! implemented by standards and provider crates without depending on the
//! aggregator `siumai` crate. It is intentionally minimal and only uses
//! core-level types.

use crate::error::LlmError;
use crate::execution::chat::{ChatInput, ChatRequestTransformer, ChatResponseTransformer};
use crate::execution::streaming::{ChatStreamEventConverterCore, ChatStreamEventCore};
use crate::traits::ProviderCapabilities;
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Core provider execution context used for header construction and routing.
///
/// This context is provider-agnostic and mirrors the minimal data set needed
/// by standards and provider crates. Aggregator crates are expected to map
/// their own context types into this structure.
#[derive(Debug, Clone)]
pub struct CoreProviderContext {
    /// Provider identifier (e.g., "openai", "anthropic", "minimaxi").
    pub provider_id: String,
    /// Base URL for the provider (without trailing slash).
    pub base_url: String,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Provider-specific HTTP extra headers (user supplied or derived).
    pub http_extra_headers: HashMap<String, String>,
    /// Optional organization identifier (for providers that support it).
    pub organization: Option<String>,
    /// Optional project identifier (for providers that support it).
    pub project: Option<String>,
    /// Extra hints for provider-specific toggles
    /// (e.g., responses API flags, previous response id, response format).
    pub extras: HashMap<String, serde_json::Value>,
}

impl CoreProviderContext {
    /// Create a new context with the required fields.
    pub fn new(
        provider_id: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
        http_extra_headers: HashMap<String, String>,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            base_url: base_url.into(),
            api_key,
            http_extra_headers,
            organization: None,
            project: None,
            extras: HashMap::new(),
        }
    }

    /// Attach organization and project identifiers.
    pub fn with_org_project(mut self, org: Option<String>, project: Option<String>) -> Self {
        self.organization = org;
        self.project = project;
        self
    }

    /// Attach provider-specific extras.
    pub fn with_extras(mut self, extras: HashMap<String, serde_json::Value>) -> Self {
        self.extras = extras;
        self
    }
}

/// Transformers bundle required by chat executors at the core level.
///
/// Standards and provider crates use this to wire core request/response and
/// streaming transformers without depending on the aggregator.
#[derive(Clone)]
pub struct CoreChatTransformers {
    pub request: Arc<dyn ChatRequestTransformer>,
    pub response: Arc<dyn ChatResponseTransformer>,
    pub stream: Option<Arc<dyn ChatStreamEventConverterCore>>,
}

/// Core provider specification trait.
///
/// This trait is designed for implementation in standards and provider crates.
/// Aggregator crates should wrap it with their own higher-level ProviderSpec
/// that operates on aggregator-specific request/response types.
pub trait CoreProviderSpec: Send + Sync {
    /// Provider identifier (e.g., "openai", "anthropic", "minimaxi").
    fn id(&self) -> &'static str;

    /// Capability declaration (metadata/debugging).
    fn capabilities(&self) -> ProviderCapabilities;

    /// Build JSON headers (auth + custom). Tracing headers are injected by caller.
    fn build_headers(&self, ctx: &CoreProviderContext) -> Result<HeaderMap, LlmError>;

    /// Compute chat route URL. Implementations may use information from
    /// `ctx.extras` or standard-specific flags to influence routing.
    fn chat_url(&self, ctx: &CoreProviderContext) -> String;

    /// Choose chat transformers (request/response/streaming) for a given input.
    fn choose_chat_transformers(
        &self,
        input: &ChatInput,
        ctx: &CoreProviderContext,
    ) -> CoreChatTransformers;

    /// Optional: map a core streaming event into a provider-specific variant.
    ///
    /// Implementations may override this when they need to add provider-specific
    /// metadata or debugging events. Default behavior forwards the event as-is.
    fn map_core_stream_event(&self, event: ChatStreamEventCore) -> ChatStreamEventCore {
        event
    }
}
