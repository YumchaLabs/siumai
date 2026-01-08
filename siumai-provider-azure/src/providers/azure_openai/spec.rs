use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use reqwest::header::HeaderMap;
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

/// Azure OpenAI ProviderSpec (Responses API by default).
///
/// This spec focuses on Azure's OpenAI-compatible endpoints:
/// - auth: `api-key` header
/// - routing: `{baseURL}/v1{path}?api-version={apiVersion}` (or deployment-based URLs)
#[derive(Debug, Clone)]
pub struct AzureOpenAiSpec {
    url_config: AzureUrlConfig,
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
            provider_metadata_key: "azure",
        }
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

impl ProviderSpec for AzureOpenAiSpec {
    fn id(&self) -> &'static str {
        "azure"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
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
        self.build_url(ctx, &req.common_params.model, "/responses")
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let provider_metadata_key = self.provider_metadata_key;

        let req_tx =
            siumai_protocol_openai::standards::openai::transformers::OpenAiResponsesRequestTransformer;
        let resp_tx = siumai_protocol_openai::standards::openai::transformers::OpenAiResponsesResponseTransformer::new()
            .with_provider_metadata_key(provider_metadata_key);
        let converter = siumai_protocol_openai::standards::openai::responses_sse::OpenAiResponsesEventConverter::new()
            .with_provider_metadata_key(provider_metadata_key)
            .with_request_tools(req.tools.as_deref().unwrap_or(&[]));
        let stream_tx = siumai_protocol_openai::standards::openai::transformers::OpenAiResponsesStreamChunkTransformer {
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
