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
//! use siumai::core::{ProviderSpec, ProviderContext, ChatTransformers};
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
//!     fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
//!         // Return chat endpoint URL
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
use crate::types::{ChatRequest, EmbeddingRequest, ImageGenerationRequest, ProviderOptions};
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
    ///
    /// Default implementation handles CustomProviderOptions injection.
    /// Providers can override this to add provider-specific logic.
    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // Default: handle CustomProviderOptions
        default_custom_options_hook(self.id(), req)
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
    ///
    /// Default implementation injects `ProviderOptions::Custom` when the
    /// provider IDs match (same behavior as chat for custom options).
    fn embedding_before_send(
        &self,
        req: &EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        default_custom_options_hook_embedding(self.id(), req)
    }

    /// Compute image generation route URL (default OpenAI-compatible)
    fn image_url(&self, _req: &ImageGenerationRequest, ctx: &ProviderContext) -> String {
        format!("{}/images/generations", ctx.base_url.trim_end_matches('/'))
    }

    /// Compute image edit route URL (default OpenAI-compatible)
    fn image_edit_url(
        &self,
        _req: &crate::types::ImageEditRequest,
        ctx: &ProviderContext,
    ) -> String {
        format!("{}/images/edits", ctx.base_url.trim_end_matches('/'))
    }

    /// Compute image variation route URL (default OpenAI-compatible)
    fn image_variation_url(
        &self,
        _req: &crate::types::ImageVariationRequest,
        ctx: &ProviderContext,
    ) -> String {
        format!("{}/images/variations", ctx.base_url.trim_end_matches('/'))
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

    /// Compute base URL for audio operations (default OpenAI-compatible)
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

    /// Compute base URL for files operations (default OpenAI-compatible)
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
}

/// Default hook for CustomProviderOptions injection
///
/// This function provides a standard implementation for injecting custom provider options
/// into the request JSON body. All providers automatically support CustomProviderOptions
/// through this default implementation.
///
/// # Example
///
/// ```rust,ignore
/// // In a provider implementation:
/// fn chat_before_send(&self, req: &ChatRequest, ctx: &ProviderContext)
///     -> Option<BeforeSendHook> {
///     // 1. First check for CustomProviderOptions
///     if let Some(hook) = default_custom_options_hook(self.id(), req) {
///         return Some(hook);
///     }
///
///     // 2. Then handle provider-specific logic
///     // ... your custom logic
/// }
/// ```
pub fn default_custom_options_hook(
    provider_id: &str,
    req: &ChatRequest,
) -> Option<crate::execution::executors::BeforeSendHook> {
    if let ProviderOptions::Custom {
        provider_id: custom_provider_id,
        options,
    } = &req.provider_options
        && crate::registry::helpers::matches_provider_id(provider_id, custom_provider_id)
    {
        let custom_options = options.clone();
        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();
            if let Some(obj) = out.as_object_mut() {
                // Merge custom options into the request body
                for (k, v) in &custom_options {
                    obj.insert(k.clone(), v.clone());
                }
            }
            Ok(out)
        };
        return Some(Arc::new(hook));
    }
    None
}

/// Default hook for injecting Custom provider options into embedding requests.
///
/// Mirrors `default_custom_options_hook` for chat, but reads from
/// `EmbeddingRequest::provider_options`. This enables user-defined provider
/// options to be merged into the outbound JSON body for embeddings.
pub fn default_custom_options_hook_embedding(
    provider_id: &str,
    req: &EmbeddingRequest,
) -> Option<crate::execution::executors::BeforeSendHook> {
    if let ProviderOptions::Custom {
        provider_id: custom_provider_id,
        options,
    } = &req.provider_options
        && crate::registry::helpers::matches_provider_id(provider_id, custom_provider_id)
    {
        let custom_options = options.clone();
        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();
            if let Some(obj) = out.as_object_mut() {
                for (k, v) in &custom_options {
                    obj.insert(k.clone(), v.clone());
                }
            }
            Ok(out)
        };
        return Some(Arc::new(hook));
    }
    None
}
