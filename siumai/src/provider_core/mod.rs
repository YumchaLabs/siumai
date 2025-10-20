//! Provider Core Abstractions
//!
//! This module defines provider-agnostic specifications for
//! - building headers (auth + custom headers)
//! - computing request routes (URLs) per capability
//! - choosing request/response/stream transformers
//! - optional pre-send JSON body mutation
//!
//! By concentrating common HTTP/route/header logic here, provider clients can
//! remain thin and focus on feature toggles and parameter surfaces.

use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::types::{ChatRequest, EmbeddingRequest, ImageGenerationRequest};
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Capability kinds (routing/transformer selection)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityKind {
    Chat,
    Embedding,
    Image,
    Files,
    Audio,
    Moderation,
    Rerank,
}

/// Provider execution context (for header construction and routing)
#[derive(Debug, Clone)]
pub struct ProviderContext {
    pub provider_id: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub http_extra_headers: HashMap<String, String>,
    pub organization: Option<String>,
    pub project: Option<String>,
    /// Extra hints for provider-specific toggles
    /// (e.g., openai.responses_api / previous_response_id / response_format)
    pub extras: HashMap<String, serde_json::Value>,
}

impl ProviderContext {
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

    pub fn with_org_project(mut self, org: Option<String>, project: Option<String>) -> Self {
        self.organization = org;
        self.project = project;
        self
    }

    pub fn with_extras(mut self, extras: HashMap<String, serde_json::Value>) -> Self {
        self.extras = extras;
        self
    }
}

/// Transformers bundle required by chat executors
#[derive(Clone)]
pub struct ChatTransformers {
    pub request: Arc<dyn RequestTransformer>,
    pub response: Arc<dyn ResponseTransformer>,
    pub stream: Option<Arc<dyn StreamChunkTransformer>>,
    pub json: Option<Arc<dyn crate::utils::streaming::JsonEventConverter>>,
}

/// Transformers bundle required by embedding executors
#[derive(Clone)]
pub struct EmbeddingTransformers {
    pub request: Arc<dyn RequestTransformer>,
    pub response: Arc<dyn ResponseTransformer>,
}

/// Transformers bundle required by image executors
#[derive(Clone)]
pub struct ImageTransformers {
    pub request: Arc<dyn RequestTransformer>,
    pub response: Arc<dyn ResponseTransformer>,
}

/// Provider Specification: unified header building, routing, and transformer selection
pub trait ProviderSpec: Send + Sync {
    /// Provider identifier (e.g., "openai", "anthropic")
    fn id(&self) -> &'static str;
    /// Capability declaration (metadata/debugging)
    fn capabilities(&self) -> ProviderCapabilities;

    /// Build JSON headers (auth + custom). Tracing headers are injected by caller.
    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError>;

    /// Compute chat route URL (may depend on request/provider params/extras)
    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String;

    /// Choose chat transformers (non-stream/stream)
    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers;

    /// Optional: mutate JSON body before sending (e.g., merge built-in tools)
    fn chat_before_send(
        &self,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::executors::BeforeSendHook> {
        None
    }

    /// Compute embedding route URL (default OpenAI-style)
    fn embedding_url(&self, _req: &EmbeddingRequest, ctx: &ProviderContext) -> String {
        format!("{}/embeddings", ctx.base_url.trim_end_matches('/'))
    }

    /// Choose embedding transformers (default: unimplemented; implement per provider)
    fn choose_embedding_transformers(
        &self,
        _req: &EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        panic!(
            "embedding transformers not implemented for provider {}",
            self.id()
        )
    }

    /// Compute image generation route URL (default OpenAI-compatible)
    fn image_url(&self, _req: &ImageGenerationRequest, ctx: &ProviderContext) -> String {
        format!("{}/images/generations", ctx.base_url.trim_end_matches('/'))
    }

    /// Choose image transformers (default: unimplemented; implement per provider)
    fn choose_image_transformers(
        &self,
        _req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        panic!(
            "image transformers not implemented for provider {}",
            self.id()
        )
    }
}
