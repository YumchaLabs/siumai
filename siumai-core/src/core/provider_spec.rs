//! Provider Specification
//!
//! This module defines the ProviderSpec trait and related types that form the core
//! of the siumai provider architecture.
//!
//! ## ProviderSpec Trait
//!
//! The ProviderSpec trait defines how a provider:
//! - Builds HTTP headers (authentication + custom headers)
//! - Computes request routes (URLs) for different capabilities
//! - Chooses request/response/stream transformers
//! - Optionally mutates JSON body before sending
//!
//! ## Design Philosophy
//!
//! By concentrating common HTTP/route/header logic in ProviderSpec implementations,
//! provider clients can remain thin and focus on provider-specific features.
//!
//! ## Example
//!
//! ```rust,ignore
//! use siumai_core::core::{ProviderSpec, ProviderContext, ChatTransformers};
//!
//! struct MyProviderSpec;
//!
//! impl ProviderSpec for MyProviderSpec {
//!     fn id(&self) -> &'static str { "my-provider" }
//!
//!     fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
//!         // Build authentication headers
//!     }
//!
//!     fn try_chat_url(
//!         &self,
//!         stream: bool,
//!         req: &ChatRequest,
//!         ctx: &ProviderContext,
//!     ) -> Result<String, LlmError> {
//!         // Return provider-owned chat endpoint URL
//!     }
//!
//!     fn choose_chat_transformers(&self, req: &ChatRequest, ctx: &ProviderContext) -> ChatTransformers {
//!         // Return transformers for request/response conversion
//!     }
//! }
//! ```

use crate::error::LlmError;
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::traits::ProviderCapabilities;
use crate::types::{
    ChatRequest, EmbeddingRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
    RerankRequest, Warning,
};
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use std::sync::Arc;

// -----------------------------------------------------------------------------
// Internal fallback transformers that return UnsupportedOperation instead of panic
// -----------------------------------------------------------------------------

struct UnsupportedRequestTx {
    provider: &'static str,
    capability: &'static str,
}

impl crate::execution::transformers::request::RequestTransformer for UnsupportedRequestTx {
    fn provider_id(&self) -> &str {
        self.provider
    }

    fn transform_chat(
        &self,
        _req: &crate::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support chat ({})",
            self.provider, self.capability
        )))
    }
}

struct UnsupportedResponseTx {
    provider: &'static str,
}

impl crate::execution::transformers::response::ResponseTransformer for UnsupportedResponseTx {
    fn provider_id(&self) -> &str {
        self.provider
    }
}

struct UnsupportedAudioTx {
    provider: &'static str,
}

impl crate::execution::transformers::audio::AudioTransformer for UnsupportedAudioTx {
    fn provider_id(&self) -> &str {
        self.provider
    }

    fn build_tts_body(
        &self,
        _req: &crate::types::TtsRequest,
    ) -> Result<crate::execution::transformers::audio::AudioHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support TTS",
            self.provider
        )))
    }

    fn build_stt_body(
        &self,
        _req: &crate::types::SttRequest,
    ) -> Result<crate::execution::transformers::audio::AudioHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support STT",
            self.provider
        )))
    }

    fn tts_endpoint(&self) -> &str {
        ""
    }

    fn stt_endpoint(&self) -> &str {
        ""
    }

    fn parse_stt_response(&self, _json: &serde_json::Value) -> Result<String, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support STT",
            self.provider
        )))
    }
}

struct UnsupportedFilesTx {
    provider: &'static str,
}

impl crate::execution::transformers::files::FilesTransformer for UnsupportedFilesTx {
    fn provider_id(&self) -> &str {
        self.provider
    }

    fn build_upload_body(
        &self,
        _req: &crate::types::FileUploadRequest,
    ) -> Result<crate::execution::transformers::files::FilesHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support file management",
            self.provider
        )))
    }

    fn list_endpoint(&self, _query: &Option<crate::types::FileListQuery>) -> String {
        "/files".to_string()
    }

    fn retrieve_endpoint(&self, _file_id: &str) -> String {
        "/files".to_string()
    }

    fn delete_endpoint(&self, _file_id: &str) -> String {
        "/files".to_string()
    }

    fn transform_file_object(
        &self,
        _raw: &serde_json::Value,
    ) -> Result<crate::types::FileObject, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support file management",
            self.provider
        )))
    }

    fn transform_list_response(
        &self,
        _raw: &serde_json::Value,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support file management",
            self.provider
        )))
    }
}

struct UnsupportedRerankRequestTx {
    provider: &'static str,
}

impl crate::execution::transformers::rerank_request::RerankRequestTransformer
    for UnsupportedRerankRequestTx
{
    fn transform(&self, _req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support rerank",
            self.provider
        )))
    }
}

struct UnsupportedRerankResponseTx {
    provider: &'static str,
}

impl crate::execution::transformers::rerank_response::RerankResponseTransformer
    for UnsupportedRerankResponseTx
{
    fn transform(&self, _raw: serde_json::Value) -> Result<crate::types::RerankResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support rerank",
            self.provider
        )))
    }
}

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
    /// (e.g., provider-a.responses_api / previous_response_id / response_format)
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
    pub json: Option<Arc<dyn crate::streaming::JsonEventConverter>>,
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

/// Audio transformer (for TTS/STT)
#[derive(Clone)]
pub struct AudioTransformer {
    pub transformer: Arc<dyn crate::execution::transformers::audio::AudioTransformer>,
}

/// Files transformer (for file operations)
#[derive(Clone)]
pub struct FilesTransformer {
    pub transformer: Arc<dyn crate::execution::transformers::files::FilesTransformer>,
}

/// Transformers bundle required by rerank executors
#[derive(Clone)]
pub struct RerankTransformers {
    pub request: Arc<dyn crate::execution::transformers::rerank_request::RerankRequestTransformer>,
    pub response:
        Arc<dyn crate::execution::transformers::rerank_response::RerankResponseTransformer>,
}

fn unsupported_route_error(provider: &str, capability: &str) -> LlmError {
    LlmError::UnsupportedOperation(format!(
        "{provider} does not provide provider-owned route resolution for {capability}"
    ))
}

/// Provider Specification: unified header building, routing, and transformer selection
pub trait ProviderSpec: Send + Sync {
    /// Provider identifier (e.g., "provider-a", "provider-b")
    fn id(&self) -> &'static str;
    /// Capability declaration (metadata/debugging)
    fn capabilities(&self) -> ProviderCapabilities;

    /// Build JSON headers (auth + custom). Tracing headers are injected by caller.
    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError>;

    /// Optional: provider-derived per-request header overrides for chat.
    ///
    /// This hook is evaluated per request, before the HTTP layer merges headers.
    /// It enables providers to conditionally enable beta flags (e.g. `provider-beta`)
    /// based on request features such as MCP servers, without requiring global client config.
    ///
    /// Returned headers are merged *before* user-provided `ChatRequest.http_config.headers`.
    fn chat_request_headers(
        &self,
        _stream: bool,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Merge per-request headers into the base headers produced by `build_headers`.
    ///
    /// Default behavior is "last write wins" (per-request overrides base).
    /// Providers may override this to implement additive semantics for specific headers.
    fn merge_request_headers(&self, base: HeaderMap, extra: &HashMap<String, String>) -> HeaderMap {
        crate::execution::http::headers::merge_headers(base, extra)
    }

    /// Optional: provider/standard-specific HTTP error classification.
    ///
    /// When returning `Some`, executors will use the returned error instead of the
    /// generic `retry_api::classify_http_error` heuristics.
    fn classify_http_error(
        &self,
        _status: u16,
        _body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        None
    }

    /// Fallible chat route resolution used by core executors.
    fn try_chat_url(
        &self,
        stream: bool,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let _ = (stream, req);
        Err(unsupported_route_error(&ctx.provider_id, "chat"))
    }

    /// Choose chat transformers (non-stream/stream)
    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let provider = self.id();
        ChatTransformers {
            request: Arc::new(UnsupportedRequestTx {
                provider,
                capability: "chat",
            }),
            response: Arc::new(UnsupportedResponseTx { provider }),
            stream: None,
            json: None,
        }
    }

    /// Optional: mutate JSON body before sending (e.g., merge built-in tools)
    /// Providers can override this to add provider-specific logic.
    fn chat_before_send(
        &self,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        None
    }

    /// Fallible embedding route resolution used by core executors.
    fn try_embedding_url(
        &self,
        req: &EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let _ = req;
        Err(unsupported_route_error(&ctx.provider_id, "embedding"))
    }

    /// Choose embedding transformers (default: unimplemented; implement per provider)
    fn choose_embedding_transformers(
        &self,
        _req: &EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        let provider = self.id();
        EmbeddingTransformers {
            request: Arc::new(UnsupportedRequestTx {
                provider,
                capability: "embedding",
            }),
            response: Arc::new(UnsupportedResponseTx { provider }),
        }
    }

    /// Optional: mutate embedding JSON body before sending.
    fn embedding_before_send(
        &self,
        _req: &EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        None
    }

    /// Fallible image generation route resolution used by core executors.
    fn try_image_url(
        &self,
        req: &ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let _ = req;
        Err(unsupported_route_error(
            &ctx.provider_id,
            "image generation",
        ))
    }

    /// Optional: warnings for image generation requests.
    fn image_warnings(
        &self,
        _req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        None
    }

    /// Fallible image edit route resolution used by core executors.
    fn try_image_edit_url(
        &self,
        req: &ImageEditRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let _ = req;
        Err(unsupported_route_error(&ctx.provider_id, "image edit"))
    }

    /// Optional: warnings for image edit requests.
    fn image_edit_warnings(
        &self,
        _req: &ImageEditRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        None
    }

    /// Whether the executor should materialize URL-backed edit inputs into inline bytes first.
    ///
    /// Default is `true` for backward compatibility with providers that only accept direct
    /// file payloads on edit endpoints. Providers that natively support URL references should
    /// override this and return `false`.
    fn materialize_image_edit_urls(&self, _req: &ImageEditRequest, _ctx: &ProviderContext) -> bool {
        true
    }

    /// Fallible image variation route resolution used by core executors.
    fn try_image_variation_url(
        &self,
        req: &ImageVariationRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let _ = req;
        Err(unsupported_route_error(&ctx.provider_id, "image variation"))
    }

    /// Optional: warnings for image variation requests.
    fn image_variation_warnings(
        &self,
        _req: &ImageVariationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        None
    }

    /// Whether the executor should materialize URL-backed variation inputs into inline bytes first.
    ///
    /// Default is `true` for backward compatibility with providers that only accept direct
    /// file payloads on variation endpoints. Providers that natively support URL references
    /// should override this and return `false`.
    fn materialize_image_variation_urls(
        &self,
        _req: &ImageVariationRequest,
        _ctx: &ProviderContext,
    ) -> bool {
        true
    }

    /// Choose image transformers (default: unimplemented; implement per provider)
    fn choose_image_transformers(
        &self,
        _req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        let provider = self.id();
        ImageTransformers {
            request: Arc::new(UnsupportedRequestTx {
                provider,
                capability: "image",
            }),
            response: Arc::new(UnsupportedResponseTx { provider }),
        }
    }

    /// Compute base URL for audio operations.
    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    /// Choose audio transformer (default: unimplemented; implement per provider)
    fn choose_audio_transformer(&self, _ctx: &ProviderContext) -> AudioTransformer {
        let provider = self.id();
        AudioTransformer {
            transformer: Arc::new(UnsupportedAudioTx { provider }),
        }
    }

    /// Compute base URL for files operations.
    fn files_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    /// Choose files transformer (default: unimplemented; implement per provider)
    fn choose_files_transformer(&self, _ctx: &ProviderContext) -> FilesTransformer {
        let provider = self.id();
        FilesTransformer {
            transformer: Arc::new(UnsupportedFilesTx { provider }),
        }
    }

    /// Fallible rerank route resolution used by core executors.
    fn try_rerank_url(
        &self,
        req: &RerankRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let _ = req;
        Err(unsupported_route_error(&ctx.provider_id, "rerank"))
    }

    /// Choose rerank transformers (default: unimplemented; implement per provider)
    fn choose_rerank_transformers(
        &self,
        _req: &RerankRequest,
        _ctx: &ProviderContext,
    ) -> RerankTransformers {
        let provider = self.id();
        RerankTransformers {
            request: Arc::new(UnsupportedRerankRequestTx { provider }),
            response: Arc::new(UnsupportedRerankResponseTx { provider }),
        }
    }

    /// Optional: mutate rerank JSON body before sending.
    fn rerank_before_send(
        &self,
        _req: &RerankRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        None
    }

    /// Fallible models listing route resolution.
    fn try_models_url(&self, ctx: &ProviderContext) -> Result<String, LlmError> {
        Err(unsupported_route_error(&ctx.provider_id, "models list"))
    }

    /// Fallible model retrieve route resolution.
    fn try_model_url(&self, model_id: &str, ctx: &ProviderContext) -> Result<String, LlmError> {
        let _ = model_id;
        Err(unsupported_route_error(&ctx.provider_id, "model retrieve"))
    }
}
