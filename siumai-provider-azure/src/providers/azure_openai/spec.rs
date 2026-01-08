use crate::core::{
    AudioTransformer, ChatTransformers, EmbeddingTransformers, FilesTransformer, ImageTransformers,
    ProviderContext, ProviderSpec,
};
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, EmbeddingRequest, ImageGenerationRequest};
use reqwest::header::HeaderMap;
use std::borrow::Cow;
use std::sync::Arc;

/// URL-related configuration for Azure OpenAI.
#[derive(Debug, Clone)]
pub struct AzureUrlConfig {
    /// Azure API version query param. Vercel AI SDK defaults to `v1`.
    pub api_version: String,
    /// Use legacy deployment-based URLs:
    /// `{baseURL}/deployments/{deploymentId}{path}?api-version={apiVersion}`.
    pub use_deployment_based_urls: bool,
}

impl Default for AzureUrlConfig {
    fn default() -> Self {
        Self {
            api_version: "v1".to_string(),
            use_deployment_based_urls: false,
        }
    }
}

/// Which chat API to use for `ChatCapability`.
///
/// This mirrors Vercel's `azure('deployment')` (Responses API) vs `azure.chat('deployment')`
/// (Chat Completions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AzureChatMode {
    Responses,
    ChatCompletions,
}

impl Default for AzureChatMode {
    fn default() -> Self {
        // Keep the crate compiling with `azure-standard` only (no Responses mapping).
        if cfg!(feature = "azure") {
            Self::Responses
        } else {
            Self::ChatCompletions
        }
    }
}

/// Azure OpenAI ProviderSpec (Responses API by default).
///
/// This spec focuses on Azure's OpenAI-compatible endpoints:
/// - auth: `api-key` header
/// - routing: `{baseURL}/v1{path}?api-version={apiVersion}` (or deployment-based URLs)
#[derive(Debug, Clone)]
pub struct AzureOpenAiSpec {
    url_config: AzureUrlConfig,
    chat_mode: AzureChatMode,
    /// Provider metadata key used by OpenAI Responses mapping (`providerMetadata.azure`).
    provider_metadata_key: &'static str,
}

impl Default for AzureOpenAiSpec {
    fn default() -> Self {
        Self::new(AzureUrlConfig::default())
    }
}

impl AzureOpenAiSpec {
    pub fn new(url_config: AzureUrlConfig) -> Self {
        Self {
            url_config,
            chat_mode: AzureChatMode::default(),
            provider_metadata_key: "azure",
        }
    }

    pub fn with_chat_mode(mut self, mode: AzureChatMode) -> Self {
        self.chat_mode = mode;
        self
    }

    pub fn with_provider_metadata_key(mut self, key: &'static str) -> Self {
        self.provider_metadata_key = key;
        self
    }

    fn build_url(&self, ctx: &ProviderContext, model_id: &str, path: &str) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let path = if self.url_config.use_deployment_based_urls {
            format!("/deployments/{}{}", model_id.trim().trim_matches('/'), path)
        } else {
            format!("/v1{path}")
        };

        let mut url = format!("{base}{path}");
        if url.contains('?') {
            url.push('&');
        } else {
            url.push('?');
        }
        url.push_str("api-version=");
        url.push_str(self.url_config.api_version.trim());
        url
    }

    fn deployment_id_from_context(&self, ctx: &ProviderContext) -> Option<String> {
        ctx.extras
            .get("azureDeploymentId")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}

fn build_azure_openai_json_headers(ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
    let api_key = ctx
        .api_key
        .as_deref()
        .ok_or_else(|| LlmError::MissingApiKey("Azure OpenAI API key not provided".into()))?;

    let mut builder = HttpHeaderBuilder::new().with_json_content_type();
    builder = builder.with_custom_headers(&ctx.http_extra_headers)?;

    let mut headers = builder.build();
    headers.insert(
        "api-key",
        api_key.parse().map_err(|e| {
            LlmError::InvalidParameter(format!("Invalid Azure api-key header: {e}"))
        })?,
    );
    Ok(headers)
}

#[derive(Debug, Clone)]
struct AzureOpenAiFilesTransformer {
    inner: siumai_protocol_openai::standards::openai::files::OpenAiFilesTransformerWithProviderId,
    api_version: Cow<'static, str>,
}

impl AzureOpenAiFilesTransformer {
    fn new(api_version: Cow<'static, str>) -> Self {
        Self {
            inner:
                siumai_protocol_openai::standards::openai::files::OpenAiFilesTransformerWithProviderId::new(
                    "azure",
                ),
            api_version,
        }
    }

    fn prefix_path(&self, endpoint: &str) -> String {
        let trimmed = endpoint.trim_start_matches('/');
        format!("v1/{trimmed}")
    }

    fn append_api_version(&self, endpoint: &str) -> String {
        if endpoint.contains('?') {
            format!("{endpoint}&api-version={}", self.api_version)
        } else {
            format!("{endpoint}?api-version={}", self.api_version)
        }
    }

    fn with_api_version(&self, endpoint: &str) -> String {
        self.append_api_version(&self.prefix_path(endpoint))
    }
}

impl crate::execution::transformers::files::FilesTransformer for AzureOpenAiFilesTransformer {
    fn provider_id(&self) -> &str {
        self.inner.provider_id()
    }

    fn build_upload_body(
        &self,
        req: &crate::types::FileUploadRequest,
    ) -> Result<crate::execution::transformers::files::FilesHttpBody, LlmError> {
        self.inner.build_upload_body(req)
    }

    fn upload_endpoint(&self, req: &crate::types::FileUploadRequest) -> String {
        self.with_api_version(&self.inner.upload_endpoint(req))
    }

    fn list_endpoint(&self, query: &Option<crate::types::FileListQuery>) -> String {
        self.with_api_version(&self.inner.list_endpoint(query))
    }

    fn retrieve_endpoint(&self, file_id: &str) -> String {
        self.with_api_version(&self.inner.retrieve_endpoint(file_id))
    }

    fn delete_endpoint(&self, file_id: &str) -> String {
        self.with_api_version(&self.inner.delete_endpoint(file_id))
    }

    fn transform_file_object(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileObject, LlmError> {
        self.inner.transform_file_object(raw)
    }

    fn transform_list_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        self.inner.transform_list_response(raw)
    }

    fn content_endpoint(&self, file_id: &str) -> Option<String> {
        self.inner
            .content_endpoint(file_id)
            .map(|e| self.with_api_version(&e))
    }
}

#[derive(Debug, Clone)]
struct AzureOpenAiAudioTransformer {
    inner: siumai_protocol_openai::standards::openai::audio::OpenAiAudioTransformerWithProviderId,
    tts_endpoint: String,
    stt_endpoint: String,
}

impl AzureOpenAiAudioTransformer {
    fn new(api_version: Cow<'static, str>, use_deployment_based_urls: bool) -> Self {
        let prefix = if use_deployment_based_urls { "" } else { "/v1" };
        let tts = format!("{prefix}/audio/speech?api-version={api_version}");
        let stt = format!("{prefix}/audio/transcriptions?api-version={api_version}");
        Self {
            inner: siumai_protocol_openai::standards::openai::audio::OpenAiAudioTransformerWithProviderId::new(
                "azure",
            ),
            tts_endpoint: tts,
            stt_endpoint: stt,
        }
    }
}

impl crate::execution::transformers::audio::AudioTransformer for AzureOpenAiAudioTransformer {
    fn provider_id(&self) -> &str {
        self.inner.provider_id()
    }

    fn build_tts_body(
        &self,
        req: &crate::types::TtsRequest,
    ) -> Result<crate::execution::transformers::audio::AudioHttpBody, LlmError> {
        self.inner.build_tts_body(req)
    }

    fn build_stt_body(
        &self,
        req: &crate::types::SttRequest,
    ) -> Result<crate::execution::transformers::audio::AudioHttpBody, LlmError> {
        self.inner.build_stt_body(req)
    }

    fn tts_endpoint(&self) -> &str {
        &self.tts_endpoint
    }

    fn stt_endpoint(&self) -> &str {
        &self.stt_endpoint
    }

    fn parse_tts_response(&self, bytes: Vec<u8>) -> Result<Vec<u8>, LlmError> {
        self.inner.parse_tts_response(bytes)
    }

    fn tts_response_is_json(&self) -> bool {
        self.inner.tts_response_is_json()
    }

    fn parse_tts_metadata(
        &self,
        json: &serde_json::Value,
    ) -> Result<(Option<f32>, Option<u32>), LlmError> {
        self.inner.parse_tts_metadata(json)
    }

    fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError> {
        self.inner.parse_stt_response(json)
    }
}

impl ProviderSpec for AzureOpenAiSpec {
    fn id(&self) -> &'static str {
        "azure"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
            .with_audio()
            .with_file_management()
            .with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_azure_openai_json_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        siumai_protocol_openai::standards::openai::errors::classify_openai_compatible_http_error(
            self.id(),
            status,
            body_text,
        )
    }

    fn chat_url(&self, _stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        // Azure OpenAI uses the deployment id as the model id for routing.
        match self.chat_mode {
            AzureChatMode::Responses => self.build_url(ctx, &req.common_params.model, "/responses"),
            AzureChatMode::ChatCompletions => {
                self.build_url(ctx, &req.common_params.model, "/chat/completions")
            }
        }
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        match self.chat_mode {
            AzureChatMode::Responses => {
                #[cfg(not(feature = "azure"))]
                {
                    // Responses mapping requires `siumai-protocol-openai/openai-responses`.
                    // Fall back to Chat Completions when the feature is disabled.
                    let spec =
                        siumai_protocol_openai::standards::openai::chat::OpenAiChatStandard::new()
                            .create_spec("azure");
                    return spec.choose_chat_transformers(req, ctx);
                }

                #[cfg(feature = "azure")]
                {
                    let provider_metadata_key = self.provider_metadata_key;

                    let req_tx = siumai_protocol_openai::standards::openai::transformers::OpenAiResponsesRequestTransformer;
                    let resp_tx = siumai_protocol_openai::standards::openai::transformers::OpenAiResponsesResponseTransformer::new()
                    .with_provider_metadata_key(provider_metadata_key);
                    let converter = siumai_protocol_openai::standards::openai::responses_sse::OpenAiResponsesEventConverter::new()
                    .with_provider_metadata_key(provider_metadata_key)
                    .with_request_tools(req.tools.as_deref().unwrap_or(&[]));
                    let stream_tx =
                    siumai_protocol_openai::standards::openai::transformers::OpenAiResponsesStreamChunkTransformer {
                        provider_id: "azure_responses".to_string(),
                        inner: converter,
                    };

                    ChatTransformers {
                        request: Arc::new(req_tx),
                        response: Arc::new(resp_tx),
                        stream: Some(Arc::new(stream_tx)),
                        json: None,
                    }
                }
            }
            AzureChatMode::ChatCompletions => {
                let spec =
                    siumai_protocol_openai::standards::openai::chat::OpenAiChatStandard::new()
                        .create_spec("azure");
                spec.choose_chat_transformers(req, ctx)
            }
        }
    }

    fn embedding_url(&self, req: &EmbeddingRequest, ctx: &ProviderContext) -> String {
        self.build_url(ctx, req.model.as_deref().unwrap_or(""), "/embeddings")
    }

    fn choose_embedding_transformers(
        &self,
        req: &EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        let spec =
            siumai_protocol_openai::standards::openai::embedding::OpenAiEmbeddingStandard::new()
                .create_spec("azure");
        spec.choose_embedding_transformers(req, ctx)
    }

    fn image_url(&self, req: &ImageGenerationRequest, ctx: &ProviderContext) -> String {
        self.build_url(
            ctx,
            req.model.as_deref().unwrap_or(""),
            "/images/generations",
        )
    }

    fn choose_image_transformers(
        &self,
        req: &ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> ImageTransformers {
        let spec = siumai_protocol_openai::standards::openai::image::OpenAiImageStandard::new()
            .create_spec("azure");
        spec.choose_image_transformers(req, ctx)
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/').to_string();
        if !self.url_config.use_deployment_based_urls {
            return base;
        }

        // In deployment-based mode, the URL format is:
        // `{baseURL}/deployments/{deploymentId}{path}?api-version={apiVersion}`.
        //
        // The generic AudioExecutor does not pass the model id to `audio_base_url`,
        // so we carry it via ProviderContext.extras (Vercel-aligned: the model object
        // is created with a deployment id).
        let Some(deployment) = self.deployment_id_from_context(ctx) else {
            return base;
        };
        format!("{base}/deployments/{}", deployment.trim().trim_matches('/'))
    }

    fn choose_audio_transformer(&self, _ctx: &ProviderContext) -> AudioTransformer {
        let api_version = Cow::Owned(self.url_config.api_version.clone());
        AudioTransformer {
            transformer: Arc::new(AzureOpenAiAudioTransformer::new(
                api_version,
                self.url_config.use_deployment_based_urls,
            )),
        }
    }

    fn files_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn choose_files_transformer(&self, _ctx: &ProviderContext) -> FilesTransformer {
        let api_version = Cow::Owned(self.url_config.api_version.clone());
        FilesTransformer {
            transformer: Arc::new(AzureOpenAiFilesTransformer::new(api_version)),
        }
    }
}
