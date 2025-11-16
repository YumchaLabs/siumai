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

/// Core provider context from the `siumai-core` crate.
///
/// This is the minimal context shape that standards/provider crates depend on.
/// The aggregator-level `ProviderContext` can be converted into this type when
/// delegating behavior to external provider crates.
use siumai_core::provider_spec::CoreProviderContext;

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

    /// Convert this aggregator-level context into the core `CoreProviderContext`.
    ///
    /// This helper is used by provider specs when delegating header/routing and
    /// transformer selection logic to external provider crates that operate on
    /// core-only types.
    pub fn to_core_context(&self) -> CoreProviderContext {
        CoreProviderContext {
            provider_id: self.provider_id.clone(),
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            http_extra_headers: self.http_extra_headers.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            extras: self.extras.clone(),
        }
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

/// Bridge a core-level chat transformers bundle into aggregator-level transformers.
///
/// This helper reduces boilerplate in provider specs that delegate chat
/// behavior to `siumai-core` / external provider crates. Callers provide:
/// - `to_core_input`: mapping from aggregator `ChatRequest` to core `ChatInput`
/// - `map_stream_event`: mapping from core `ChatStreamEventCore` to aggregator `ChatStreamEvent`
///
/// The returned `ChatTransformers` can be passed to executors in the
/// aggregator crate.
pub fn bridge_core_chat_transformers<F, G>(
    core_txs: siumai_core::provider_spec::CoreChatTransformers,
    to_core_input: F,
    map_stream_event: G,
) -> ChatTransformers
where
    F: Fn(&ChatRequest) -> siumai_core::execution::chat::ChatInput + Send + Sync + 'static,
    G: Fn(
            siumai_core::execution::streaming::ChatStreamEventCore,
        ) -> crate::streaming::ChatStreamEvent
        + Send
        + Sync
        + 'static,
{
    use siumai_core::execution::chat::{
        ChatInput, ChatRequestTransformer, ChatResponseTransformer,
    };

    struct ChatRequestBridge<F> {
        inner: Arc<dyn ChatRequestTransformer>,
        to_core: F,
    }

    impl<F> RequestTransformer for ChatRequestBridge<F>
    where
        F: Fn(&ChatRequest) -> ChatInput + Send + Sync + 'static,
    {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
            let input = (self.to_core)(req);
            self.inner.transform_chat(&input)
        }
    }

    struct ChatResponseBridge {
        inner: Arc<dyn ChatResponseTransformer>,
    }

    impl ResponseTransformer for ChatResponseBridge {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn transform_chat_response(
            &self,
            raw: &serde_json::Value,
        ) -> Result<crate::types::ChatResponse, LlmError> {
            use crate::types::{FinishReason, MessageContent, Usage};
            use siumai_core::types::FinishReasonCore;

            let core_res = self.inner.transform_chat_response(raw)?;
            let content = MessageContent::Text(core_res.content);

            let usage = core_res
                .usage
                .map(|u| Usage::new(u.prompt_tokens, u.completion_tokens));

            let finish_reason = core_res.finish_reason.map(|fr| match fr {
                FinishReasonCore::Stop => FinishReason::Stop,
                FinishReasonCore::Length => FinishReason::Length,
                FinishReasonCore::ContentFilter => FinishReason::ContentFilter,
                FinishReasonCore::ToolCalls => FinishReason::ToolCalls,
                FinishReasonCore::Other(s) => FinishReason::Other(s),
            });

            Ok(crate::types::ChatResponse {
                id: None,
                model: None,
                content,
                usage,
                finish_reason,
                system_fingerprint: None,
                service_tier: None,
                audio: None,
                warnings: None,
                provider_metadata: None,
            })
        }
    }

    struct StreamBridge<G> {
        inner: Arc<dyn siumai_core::execution::streaming::ChatStreamEventConverterCore>,
        map_evt: G,
    }

    impl<G> crate::execution::transformers::stream::StreamChunkTransformer for StreamBridge<G>
    where
        G: Fn(
                siumai_core::execution::streaming::ChatStreamEventCore,
            ) -> crate::streaming::ChatStreamEvent
            + Send
            + Sync
            + 'static,
    {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn convert_event(
            &self,
            event: eventsource_stream::Event,
        ) -> crate::execution::transformers::stream::StreamEventFuture<'_> {
            let inner = Arc::clone(&self.inner);
            let map_evt = &self.map_evt;
            Box::pin(async move {
                inner
                    .convert_event(event)
                    .into_iter()
                    .map(|res| res.map(|e| map_evt(e)))
                    .collect()
            })
        }

        fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
            self.inner
                .handle_stream_end()
                .map(|res| res.map(|e| (self.map_evt)(e)))
        }
    }

    let request = Arc::new(ChatRequestBridge {
        inner: core_txs.request,
        to_core: to_core_input,
    });

    let response = Arc::new(ChatResponseBridge {
        inner: core_txs.response,
    });

    let stream = core_txs.stream.map(|inner| {
        Arc::new(StreamBridge {
            inner,
            map_evt: map_stream_event,
        }) as Arc<dyn crate::execution::transformers::stream::StreamChunkTransformer>
    });

    ChatTransformers {
        request,
        response,
        stream,
        json: None,
    }
}

/// Helper: map an aggregator-level `ChatRequest` into a minimal
/// `siumai-core` `ChatInput` (OpenAI-style).
///
/// This is shared by OpenAI and OpenAI-compatible providers, and any
/// other providers that reuse the OpenAI Chat Completions standard.
pub fn openai_like_chat_request_to_core_input(
    req: &ChatRequest,
) -> siumai_core::execution::chat::ChatInput {
    use siumai_core::execution::chat::{ChatInput, ChatMessageInput, ChatRole};

    let messages = req
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                crate::types::MessageRole::System => ChatRole::System,
                crate::types::MessageRole::User => ChatRole::User,
                crate::types::MessageRole::Assistant => ChatRole::Assistant,
                _ => ChatRole::User,
            };
            let content = m.content.all_text();
            ChatMessageInput { role, content }
        })
        .collect::<Vec<_>>();

    ChatInput {
        messages,
        model: Some(req.common_params.model.clone()),
        max_tokens: req.common_params.max_tokens,
        temperature: req.common_params.temperature,
        top_p: req.common_params.top_p,
        presence_penalty: None,
        frequency_penalty: None,
        stop: req.common_params.stop_sequences.clone(),
        extra: Default::default(),
    }
}

/// Helper: map an aggregator-level `ChatRequest` into a minimal
/// `siumai-core` `ChatInput` (Gemini-style).
///
/// Currently reuses the OpenAI-style base field mapping and carries only
/// Gemini-specific ProviderOptions in `extra`. JSON details are injected
/// by the std-gemini adapter.
pub fn gemini_like_chat_request_to_core_input(
    req: &ChatRequest,
) -> siumai_core::execution::chat::ChatInput {
    use siumai_core::execution::chat::{ChatInput, ChatMessageInput, ChatRole};
    use std::collections::HashMap;

    let messages = req
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                crate::types::MessageRole::System => ChatRole::System,
                crate::types::MessageRole::User => ChatRole::User,
                crate::types::MessageRole::Assistant => ChatRole::Assistant,
                _ => ChatRole::User,
            };
            let content = m.content.all_text();
            ChatMessageInput { role, content }
        })
        .collect::<Vec<_>>();

    let mut extra: HashMap<String, serde_json::Value> = HashMap::new();
    if let crate::types::ProviderOptions::Gemini(ref options) = req.provider_options {
        if let Some(ref code) = options.code_execution {
            if let Ok(v) = serde_json::to_value(code) {
                extra.insert("gemini_code_execution".to_string(), v);
            }
        }
        if let Some(ref search) = options.search_grounding {
            if let Ok(v) = serde_json::to_value(search) {
                extra.insert("gemini_search_grounding".to_string(), v);
            }
        }
        if let Some(ref fs) = options.file_search {
            if let Ok(v) = serde_json::to_value(fs) {
                extra.insert("gemini_file_search".to_string(), v);
            }
        }
        if let Some(ref mime) = options.response_mime_type {
            extra.insert(
                "gemini_response_mime_type".to_string(),
                serde_json::json!(mime),
            );
        }
    }

    ChatInput {
        messages,
        model: Some(req.common_params.model.clone()),
        max_tokens: req.common_params.max_tokens,
        temperature: req.common_params.temperature,
        top_p: req.common_params.top_p,
        presence_penalty: None,
        frequency_penalty: None,
        stop: req.common_params.stop_sequences.clone(),
        extra,
    }
}

/// Helper: map an aggregator-level `ChatRequest` into the minimal
/// `siumai-core` `ChatInput` used by Anthropic-style standards
/// (Messages API).
///
/// This is shared by Anthropic and MiniMaxi providers, and any other
/// providers that reuse the Anthropic Messages standard.
pub fn anthropic_like_chat_request_to_core_input(
    req: &ChatRequest,
) -> siumai_core::execution::chat::ChatInput {
    use siumai_core::execution::chat::{ChatInput, ChatMessageInput, ChatRole};
    use std::collections::HashMap;

    let messages = req
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                crate::types::MessageRole::System => ChatRole::System,
                crate::types::MessageRole::User => ChatRole::User,
                crate::types::MessageRole::Assistant => ChatRole::Assistant,
                _ => ChatRole::User,
            };
            let content = m.content.all_text();
            ChatMessageInput { role, content }
        })
        .collect::<Vec<_>>();

    // Map typed Anthropic options into core-level `extra` payload.
    //
    // NOTE: We intentionally construct the final protocol JSON shapes here,
    // and keep only a light renaming responsibility in the std layer
    // (`AnthropicDefaultChatAdapter`). This follows the ProviderOptions
    // standard: typed options → serde_json::Value → `ChatInput::extra`.
    let mut extra: HashMap<String, serde_json::Value> = HashMap::new();
    if let crate::types::ProviderOptions::Anthropic(ref options) = req.provider_options {
        // Thinking mode configuration
        if let Some(ref thinking) = options.thinking_mode {
            if thinking.enabled {
                let mut thinking_config = serde_json::json!({ "type": "enabled" });
                if let Some(budget) = thinking.thinking_budget {
                    thinking_config["budget_tokens"] = serde_json::json!(budget);
                }
                extra.insert("anthropic_thinking".to_string(), thinking_config);
            }
        }

        // Structured output configuration
        if let Some(ref rf) = options.response_format {
            let value = match rf {
                crate::types::AnthropicResponseFormat::JsonObject => {
                    serde_json::json!({ "type": "json_object" })
                }
                crate::types::AnthropicResponseFormat::JsonSchema {
                    name,
                    schema,
                    strict,
                } => serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": strict,
                        "schema": schema,
                    }
                }),
            };
            extra.insert("anthropic_response_format".to_string(), value);
        }

        // Prompt caching configuration (v1: message-level caching by message_index).
        //
        // Currently we only apply cache control to user/assistant messages, and
        // the cache type only supports "ephemeral". More fine-grained TTL/cache_key
        // settings may be added in future versions.
        if let Some(ref pc) = options.prompt_caching {
            if pc.enabled && !pc.cache_control.is_empty() {
                let entries: Vec<serde_json::Value> = pc
                    .cache_control
                    .iter()
                    .map(|ctrl| {
                        // `AnthropicCacheType` already implements Serialize (lowercase enum),
                        // so we construct the final protocol shape directly here.
                        let cache_type = serde_json::to_value(&ctrl.cache_type)
                            .unwrap_or_else(|_| serde_json::json!("ephemeral"));
                        serde_json::json!({
                            "index": ctrl.message_index,
                            "cache_control": {
                                "type": cache_type
                            }
                        })
                    })
                    .collect();

                if !entries.is_empty() {
                    extra.insert(
                        "anthropic_prompt_caching".to_string(),
                        serde_json::Value::Array(entries),
                    );
                }
            }
        }
    }

    ChatInput {
        messages,
        model: Some(req.common_params.model.clone()),
        max_tokens: req.common_params.max_tokens,
        temperature: req.common_params.temperature,
        top_p: req.common_params.top_p,
        presence_penalty: None,
        frequency_penalty: None,
        stop: req.common_params.stop_sequences.clone(),
        extra,
    }
}

/// Helper: map a core-level stream event into the aggregator's
/// `ChatStreamEvent`, injecting the given provider id into the
/// StreamStart metadata.
///
/// This is the generic mapping shared by all providers that reuse the
/// core streaming event model.
pub fn map_core_stream_event_with_provider(
    provider: &str,
    evt: siumai_core::execution::streaming::ChatStreamEventCore,
) -> crate::streaming::ChatStreamEvent {
    use crate::streaming::ChatStreamEvent;
    use crate::types::{ChatResponse, FinishReason};
    use siumai_core::execution::streaming::ChatStreamEventCore;
    use siumai_core::types::FinishReasonCore;

    match evt {
        ChatStreamEventCore::ContentDelta { delta, index } => {
            ChatStreamEvent::ContentDelta { delta, index }
        }
        ChatStreamEventCore::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            index,
        } => ChatStreamEvent::ToolCallDelta {
            id: id.unwrap_or_default(),
            function_name,
            arguments_delta,
            index,
        },
        ChatStreamEventCore::ThinkingDelta { delta } => ChatStreamEvent::ThinkingDelta { delta },
        ChatStreamEventCore::UsageUpdate {
            prompt_tokens,
            completion_tokens,
            ..
        } => {
            let usage = crate::types::Usage::new(prompt_tokens, completion_tokens);
            ChatStreamEvent::UsageUpdate { usage }
        }
        ChatStreamEventCore::StreamStart {} => ChatStreamEvent::StreamStart {
            metadata: crate::types::ResponseMetadata {
                id: None,
                model: None,
                created: None,
                provider: provider.to_string(),
                request_id: None,
            },
        },
        ChatStreamEventCore::StreamEnd { finish_reason } => {
            let mapped = match finish_reason {
                Some(FinishReasonCore::Stop) => FinishReason::Stop,
                Some(FinishReasonCore::Length) => FinishReason::Length,
                Some(FinishReasonCore::ContentFilter) => FinishReason::ContentFilter,
                Some(FinishReasonCore::ToolCalls) => FinishReason::ToolCalls,
                Some(FinishReasonCore::Other(s)) => FinishReason::Other(s),
                None => FinishReason::Unknown,
            };
            let response = ChatResponse::empty_with_finish_reason(mapped);
            ChatStreamEvent::StreamEnd { response }
        }
        ChatStreamEventCore::Custom { event_type, data } => {
            ChatStreamEvent::Custom { event_type, data }
        }
        ChatStreamEventCore::Error { error } => ChatStreamEvent::Error { error },
    }
}

/// Helper: map a core-level Anthropic-style stream event into the
/// aggregator's `ChatStreamEvent`, injecting the given provider id
/// into the StreamStart metadata.
///
/// Kept for backward-compatibility; internally delegates to the
/// generic `map_core_stream_event_with_provider`.
pub fn anthropic_like_map_core_stream_event(
    provider: &'static str,
    evt: siumai_core::execution::streaming::ChatStreamEventCore,
) -> crate::streaming::ChatStreamEvent {
    map_core_stream_event_with_provider(provider, evt)
}
