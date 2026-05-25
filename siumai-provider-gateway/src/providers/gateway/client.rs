use super::GatewayConfig;
use crate::core::{ProviderContext, ProviderSpec};
use crate::core_compat::client::LlmClient;
use crate::embedding::EmbeddingModel;
use crate::error::LlmError;
use crate::execution::executors::common::{
    HttpBody, HttpExecutionConfig, execute_json_request, execute_json_request_streaming_response,
};
use crate::execution::http::headers::headermap_to_hashmap;
use crate::retry_api::RetryOptions;
use crate::text::{LanguageModelV4, LanguageModelV4DoStreamResult};
use crate::traits::{
    ChatCapability, EmbeddingCapability, EmbeddingExtensions, ModelMetadata, ProviderCapabilities,
};
use crate::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, ChatMessage, ChatResponse, ContentPart,
    EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, FilePartSource, HttpConfig,
    HttpRequestInfo, HttpResponseInfo, LanguageModelV4CallOptions, LanguageModelV4Content,
    LanguageModelV4GenerateResponseMetadata, LanguageModelV4GenerateResult,
    LanguageModelV4GeneratedFileData, LanguageModelV4RequestMetadata,
    LanguageModelV4StreamResponseMetadata, LanguageModelV4StreamResult, MessageContent,
    ModelMessage, ProviderOptionsMap, Tool, ToolResultOutput, Usage, UsageInputTokens,
    UsageOutputTokens,
};
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use secrecy::ExposeSecret;
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct GatewayProviderSpec {
    team_id_or_slug: Option<String>,
}

impl GatewayProviderSpec {
    fn insert_header(
        headers: &mut HeaderMap,
        name: impl AsRef<str>,
        value: impl AsRef<str>,
    ) -> Result<(), LlmError> {
        let name = name.as_ref();
        let value = value.as_ref();
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid Gateway header name '{name}': {e}"))
        })?;
        let header_value = HeaderValue::from_str(value).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid Gateway header value for '{name}': {e}"))
        })?;
        headers.insert(header_name, header_value);
        Ok(())
    }
}

impl ProviderSpec for GatewayProviderSpec {
    fn id(&self) -> &'static str {
        "gateway"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_embedding()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().ok_or_else(|| {
            LlmError::MissingApiKey("Vercel AI Gateway API key not provided".to_string())
        })?;

        let mut headers = HeaderMap::new();
        Self::insert_header(&mut headers, "authorization", format!("Bearer {api_key}"))?;
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Self::insert_header(&mut headers, "ai-gateway-protocol-version", "0.0.1")?;
        Self::insert_header(&mut headers, "ai-gateway-auth-method", "api-key")?;

        if let Some(team_id_or_slug) = &self.team_id_or_slug {
            Self::insert_header(&mut headers, "x-vercel-ai-gateway-team", team_id_or_slug)?;
        }

        for (name, value) in &ctx.http_extra_headers {
            Self::insert_header(&mut headers, name, value)?;
        }

        Ok(headers)
    }
}

#[derive(Clone)]
pub struct GatewayClient {
    config: GatewayConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for GatewayClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayClient")
            .field("config", &self.config)
            .field("retry_options", &self.retry_options)
            .finish()
    }
}

impl GatewayClient {
    pub fn from_config(config: GatewayConfig) -> Result<Self, LlmError> {
        config.validate()?;
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    pub fn with_http_client(
        config: GatewayConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        Ok(Self {
            config,
            http_client,
            retry_options: None,
        })
    }

    pub fn with_retry_options(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    pub fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    pub fn http_transport(
        &self,
    ) -> Option<Arc<dyn crate::execution::http::transport::HttpTransport>> {
        self.config.http_transport.clone()
    }

    fn endpoint_url(&self, endpoint: &str) -> String {
        format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint
        )
    }

    fn execution_config(&self) -> HttpExecutionConfig {
        let provider_context = ProviderContext::new(
            "gateway",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );

        HttpExecutionConfig {
            provider_id: "gateway".to_string(),
            http_client: self.http_client.clone(),
            transport: self.config.http_transport.clone(),
            provider_spec: Arc::new(GatewayProviderSpec {
                team_id_or_slug: self.config.team_id_or_slug.clone(),
            }),
            provider_context,
            interceptors: self.config.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
        }
    }

    fn language_http_config(
        &self,
        options: &LanguageModelV4CallOptions,
        streaming: bool,
    ) -> HttpConfig {
        let mut http_config = HttpConfig::empty();
        http_config.headers.extend(options.effective_headers());
        http_config.headers.insert(
            "ai-language-model-specification-version".to_string(),
            "4".to_string(),
        );
        http_config.headers.insert(
            "ai-language-model-id".to_string(),
            self.config.common_params.model.clone(),
        );
        http_config.headers.insert(
            "ai-language-model-streaming".to_string(),
            streaming.to_string(),
        );
        http_config
    }

    fn embedding_http_config(&self, request: &EmbeddingRequest, model: &str) -> HttpConfig {
        let mut http_config = request
            .http_config
            .clone()
            .unwrap_or_else(HttpConfig::empty);
        http_config.headers.insert(
            "ai-embedding-model-specification-version".to_string(),
            "4".to_string(),
        );
        http_config
            .headers
            .insert("ai-model-id".to_string(), model.to_string());
        http_config
    }

    fn embedding_model_id(&self, request: &EmbeddingRequest) -> Result<String, LlmError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(&self.config.common_params.model)
            .trim();

        if model.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Gateway embedding requires an explicit model id".to_string(),
            ));
        }

        Ok(model.to_string())
    }

    fn chat_request_to_v4_options(
        &self,
        request: crate::types::ChatRequest,
    ) -> Result<LanguageModelV4CallOptions, LlmError> {
        let requested_model = request.common_params.model.trim();
        if !requested_model.is_empty() && requested_model != self.config.common_params.model {
            return Err(LlmError::InvalidInput(format!(
                "Gateway chat_request model override '{requested_model}' does not match configured model '{}'",
                self.config.common_params.model
            )));
        }

        let prompt = request
            .messages
            .into_iter()
            .map(ModelMessage::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                LlmError::InvalidInput(format!(
                    "Gateway chat_request cannot be projected to AI SDK V4 prompt: {e}"
                ))
            })?;

        let mut options = LanguageModelV4CallOptions::from_model_messages(prompt);
        options.max_output_tokens = request
            .common_params
            .max_completion_tokens
            .or(request.common_params.max_tokens)
            .map(u64::from);
        options.temperature = request.common_params.temperature;
        options.stop_sequences = request.common_params.stop_sequences;
        options.top_p = request.common_params.top_p;
        options.top_k = request.common_params.top_k;
        options.presence_penalty = request.common_params.presence_penalty;
        options.frequency_penalty = request.common_params.frequency_penalty;
        options.response_format = request.response_format;
        options.seed = request.common_params.seed;
        options.tools = request
            .tools
            .map(|tools| tools.into_iter().map(Into::into).collect());
        options.tool_choice = request.tool_choice.map(Into::into);
        options.include_raw_chunks = request.stream_options.include_raw_chunks.then_some(true);
        if !request.provider_options_map.is_empty() {
            options.provider_options = Some(request.provider_options_map);
        }
        if let Some(http_config) = request.http_config {
            options.headers.extend(
                http_config
                    .headers
                    .into_iter()
                    .map(|(name, value)| (name, Some(value))),
            );
        }

        Ok(options)
    }
}

#[async_trait]
impl LanguageModelV4 for GatewayClient {
    async fn do_generate(
        &self,
        options: LanguageModelV4CallOptions,
    ) -> Result<LanguageModelV4GenerateResult, LlmError> {
        let mut body = language_model_v4_call_body(&options)?;
        encode_inline_file_data(&mut body);
        let http_config = self.language_http_config(&options, false);
        let url = self.endpoint_url("language-model");
        let execution = execute_json_request(
            &self.execution_config(),
            &url,
            HttpBody::Json(body.clone()),
            Some(&http_config),
            false,
        )
        .await?;

        let mut result: LanguageModelV4GenerateResult =
            serde_json::from_value(execution.json.clone()).map_err(|e| {
                LlmError::JsonError(format!("Invalid Gateway language response: {e}"))
            })?;

        if result.request.is_none() {
            result.request = Some(LanguageModelV4RequestMetadata { body: Some(body) });
        }

        if result.response.is_none() {
            result.response = Some(LanguageModelV4GenerateResponseMetadata {
                headers: Some(headermap_to_hashmap(&execution.headers)),
                body: Some(execution.json),
                ..Default::default()
            });
        }

        Ok(result)
    }

    async fn do_stream(
        &self,
        options: LanguageModelV4CallOptions,
    ) -> Result<LanguageModelV4DoStreamResult, LlmError> {
        let include_raw_chunks = options.include_raw_chunks.unwrap_or(false);
        let mut body = language_model_v4_call_body(&options)?;
        encode_inline_file_data(&mut body);
        let http_config = self.language_http_config(&options, true);
        let url = self.endpoint_url("language-model");
        let response = execute_json_request_streaming_response(
            &self.execution_config(),
            &url,
            body.clone(),
            Some(&http_config),
        )
        .await?;

        let headers = headermap_to_hashmap(response.headers());
        let byte_stream = response.bytes_stream().map(|chunk| {
            chunk
                .map(|bytes| bytes.to_vec())
                .map_err(|e| LlmError::StreamError(e.to_string()))
        });
        let stream = gateway_v4_sse_stream(byte_stream, include_raw_chunks);

        Ok(LanguageModelV4StreamResult::new(stream)
            .with_request(LanguageModelV4RequestMetadata { body: Some(body) })
            .with_response(LanguageModelV4StreamResponseMetadata {
                headers: Some(headers),
            }))
    }
}

#[async_trait]
impl ChatCapability for GatewayClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut request = crate::types::ChatRequest::new(messages);
        request.tools = tools;
        self.chat_request(request).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        let mut request = crate::types::ChatRequest::new(messages);
        request.tools = tools;
        self.chat_stream_request(request).await
    }

    async fn chat_request(
        &self,
        request: crate::types::ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        let options = self.chat_request_to_v4_options(request)?;
        let result = self.do_generate(options).await?;
        gateway_chat_response_from_v4(result, &self.config.common_params.model)
    }

    async fn chat_stream_request(
        &self,
        request: crate::types::ChatRequest,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        let options = self.chat_request_to_v4_options(request)?;
        let result = self.do_stream(options).await?;
        let model = self.config.common_params.model.clone();
        let response_headers = result.response.and_then(|response| response.headers);
        let mut stream = result.stream;

        Ok(Box::pin(async_stream::stream! {
            let mut text = String::new();
            let mut usage: Option<Usage> = None;
            let mut finish_reason: Option<crate::types::FinishReason> = None;
            let mut raw_finish_reason: Option<String> = None;
            let mut provider_metadata = None;

            while let Some(part) = stream.next().await {
                let part = match part {
                    Ok(part) => part,
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                };

                let runtime_part = part.to_runtime_part();
                if let crate::types::ChatStreamPart::TextDelta { delta, .. } = &runtime_part {
                    text.push_str(delta);
                }
                if let crate::types::ChatStreamPart::Finish {
                    usage: part_usage,
                    finish_reason: part_finish_reason,
                    provider_metadata: part_provider_metadata,
                } = &runtime_part
                {
                    usage = Some(part_usage.clone());
                    finish_reason = Some(part_finish_reason.unified.clone());
                    raw_finish_reason = part_finish_reason.raw.clone();
                    provider_metadata = part_provider_metadata.clone();
                }

                yield Ok(crate::types::ChatStreamEvent::Part { part: runtime_part });
            }

            let mut response = ChatResponse::new(MessageContent::Text(text));
            response.model = Some(model.clone());
            response.usage = usage.or_else(|| Some(Usage::unknown()));
            response.finish_reason = finish_reason.or(Some(crate::types::FinishReason::Unknown));
            response.raw_finish_reason = raw_finish_reason;
            response.provider_metadata = provider_metadata;
            response.response = Some(HttpResponseInfo {
                timestamp: chrono::Utc::now(),
                model_id: Some(model),
                headers: response_headers.unwrap_or_default(),
                body: None,
            });

            yield Ok(crate::types::ChatStreamEvent::StreamEnd { response });
        }))
    }
}

fn gateway_chat_response_from_v4(
    result: LanguageModelV4GenerateResult,
    configured_model: &str,
) -> Result<ChatResponse, LlmError> {
    let mut response = ChatResponse::new(gateway_message_content_from_v4(result.content)?);
    response.finish_reason = Some(result.finish_reason.unified);
    response.raw_finish_reason = result.finish_reason.raw;
    response.usage = Some(gateway_usage_from_v4(result.usage)?);
    response.provider_metadata = result.provider_metadata;
    response.warnings = (!result.warnings.is_empty())
        .then(|| result.warnings.into_iter().map(Into::into).collect());

    if let Some(request) = result.request
        && let Some(body) = request.body
    {
        response.request = Some(HttpRequestInfo {
            body: Some(serde_json::to_string(&body).map_err(|e| {
                LlmError::JsonError(format!("Invalid Gateway request metadata body: {e}"))
            })?),
        });
    }

    if let Some(metadata) = result.response {
        response.id = metadata.id.clone();
        response.model = Some(
            metadata
                .model_id
                .clone()
                .unwrap_or_else(|| configured_model.to_string()),
        );
        response.response = Some(HttpResponseInfo {
            timestamp: metadata.timestamp.unwrap_or_else(chrono::Utc::now),
            model_id: metadata
                .model_id
                .or_else(|| Some(configured_model.to_string())),
            headers: metadata.headers.unwrap_or_default(),
            body: metadata.body,
        });
    } else {
        response.model = Some(configured_model.to_string());
    }

    Ok(response)
}

fn gateway_message_content_from_v4(
    content: Vec<LanguageModelV4Content>,
) -> Result<MessageContent, LlmError> {
    let mut parts = Vec::new();
    for part in content {
        parts.push(gateway_content_part_from_v4(part)?);
    }

    if parts.is_empty() {
        return Ok(MessageContent::Text(String::new()));
    }

    if parts.len() == 1 {
        let part = parts.into_iter().next().expect("single content part");
        return match part {
            ContentPart::Text {
                text,
                provider_options,
                provider_metadata,
            } if provider_options.is_empty() && provider_metadata.is_none() => {
                Ok(MessageContent::Text(text))
            }
            other => Ok(MessageContent::MultiModal(vec![other])),
        };
    }

    Ok(MessageContent::MultiModal(parts))
}

fn gateway_content_part_from_v4(part: LanguageModelV4Content) -> Result<ContentPart, LlmError> {
    Ok(match part {
        LanguageModelV4Content::Text(part) => ContentPart::Text {
            text: part.text,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: part.provider_metadata,
        },
        LanguageModelV4Content::Reasoning(part) => ContentPart::Reasoning {
            text: part.text,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: part.provider_metadata,
        },
        LanguageModelV4Content::Custom(part) => ContentPart::Custom {
            kind: part.kind,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: part.provider_metadata,
        },
        LanguageModelV4Content::ReasoningFile(part) => ContentPart::ReasoningFile {
            source: gateway_generated_file_data_to_media_source(part.data),
            media_type: part.media_type,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: part.provider_metadata,
        },
        LanguageModelV4Content::File(part) => ContentPart::File {
            source: gateway_generated_file_data_to_file_source(part.data),
            media_type: part.media_type,
            filename: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: part.provider_metadata,
        },
        LanguageModelV4Content::ToolApprovalRequest(part) => ContentPart::ToolApprovalRequest {
            approval_id: part.approval_id,
            tool_call_id: part.tool_call_id,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: part.provider_metadata,
        },
        LanguageModelV4Content::Source(part) => ContentPart::Source {
            id: part.id,
            source: part.source,
            provider_metadata: part.provider_metadata,
        },
        LanguageModelV4Content::ToolCall(part) => ContentPart::ToolCall {
            tool_call_id: part.tool_call_id,
            tool_name: part.tool_name,
            arguments: serde_json::from_str(&part.input)
                .unwrap_or_else(|_| serde_json::Value::String(part.input)),
            provider_executed: part.provider_executed,
            dynamic: part.dynamic,
            invalid: None,
            error: None,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: part.provider_metadata,
        },
        LanguageModelV4Content::ToolResult(part) => ContentPart::ToolResult {
            tool_call_id: part.tool_call_id,
            tool_name: part.tool_name,
            output: if part.is_error.unwrap_or(false) {
                ToolResultOutput::error_json(part.result)
            } else {
                ToolResultOutput::json(part.result)
            },
            input: None,
            provider_executed: None,
            dynamic: part.dynamic,
            preliminary: part.preliminary,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: part.provider_metadata,
        },
    })
}

fn gateway_generated_file_data_to_file_source(
    data: LanguageModelV4GeneratedFileData,
) -> FilePartSource {
    match data {
        LanguageModelV4GeneratedFileData::String(data) => FilePartSource::base64(data),
        LanguageModelV4GeneratedFileData::Bytes(data) => FilePartSource::binary(data),
    }
}

fn gateway_generated_file_data_to_media_source(
    data: LanguageModelV4GeneratedFileData,
) -> crate::types::MediaSource {
    match data {
        LanguageModelV4GeneratedFileData::String(data) => crate::types::MediaSource::base64(data),
        LanguageModelV4GeneratedFileData::Bytes(data) => crate::types::MediaSource::binary(data),
    }
}

fn gateway_usage_from_v4(usage: crate::types::LanguageModelV4Usage) -> Result<Usage, LlmError> {
    let mut builder = Usage::builder().with_input_tokens(UsageInputTokens {
        total: gateway_u64_to_u32(usage.input_tokens.total, "inputTokens.total")?,
        no_cache: gateway_u64_to_u32(usage.input_tokens.no_cache, "inputTokens.noCache")?,
        cache_read: gateway_u64_to_u32(usage.input_tokens.cache_read, "inputTokens.cacheRead")?,
        cache_write: gateway_u64_to_u32(usage.input_tokens.cache_write, "inputTokens.cacheWrite")?,
    });
    builder = builder.with_output_tokens(UsageOutputTokens {
        total: gateway_u64_to_u32(usage.output_tokens.total, "outputTokens.total")?,
        text: gateway_u64_to_u32(usage.output_tokens.text, "outputTokens.text")?,
        reasoning: gateway_u64_to_u32(usage.output_tokens.reasoning, "outputTokens.reasoning")?,
    });
    if let Some(raw) = usage.raw {
        builder = builder.with_raw_usage(raw);
    }
    Ok(builder.build())
}

fn gateway_u64_to_u32(value: Option<u64>, field: &str) -> Result<Option<u32>, LlmError> {
    value
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                LlmError::ParseError(format!("Gateway usage field '{field}' exceeds u32"))
            })
        })
        .transpose()
}

#[async_trait]
impl EmbeddingCapability for GatewayClient {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        let request =
            EmbeddingRequest::new(input).with_model(self.config.common_params.model.clone());
        self.embed_with_config(request).await
    }

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }

    fn embedding_dimension(&self) -> usize {
        1536
    }
}

#[async_trait]
impl EmbeddingExtensions for GatewayClient {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        let model = self.embedding_model_id(&request)?;
        let body = gateway_embedding_body(&request)?;
        let http_config = self.embedding_http_config(&request, &model);
        let url = self.endpoint_url("embedding-model");
        let execution = execute_json_request(
            &self.execution_config(),
            &url,
            HttpBody::Json(body),
            Some(&http_config),
            false,
        )
        .await?;

        gateway_embedding_response(execution.json, execution.headers, model)
    }

    async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
        let mut responses = Vec::new();
        for request in requests.requests {
            let result = self
                .embed_with_config(request)
                .await
                .map_err(|e| e.to_string());
            responses.push(result);
        }
        Ok(BatchEmbeddingResponse {
            responses,
            metadata: Default::default(),
        })
    }
}

impl ModelMetadata for GatewayClient {
    fn provider_id(&self) -> &str {
        "gateway"
    }

    fn model_id(&self) -> &str {
        &self.config.common_params.model
    }
}

impl LlmClient for GatewayClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("gateway")
    }

    fn supported_models(&self) -> Vec<String> {
        let model = self.config.common_params.model.trim();
        if model.is_empty() {
            Vec::new()
        } else {
            vec![model.to_string()]
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        Some(self)
    }

    fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
        Some(self)
    }
}

fn _assert_family_traits(client: &GatewayClient) {
    fn assert_language<M: crate::text::LanguageModel + ?Sized>(_model: &M) {}
    fn assert_embedding<M: EmbeddingModel + ?Sized>(_model: &M) {}
    assert_language(client);
    assert_embedding(client);
}

fn language_model_v4_call_body(
    options: &LanguageModelV4CallOptions,
) -> Result<serde_json::Value, LlmError> {
    let mut body = serde_json::Map::new();
    body.insert(
        "prompt".to_string(),
        serde_json::to_value(&options.prompt)
            .map_err(|e| LlmError::JsonError(format!("Invalid Gateway prompt: {e}")))?,
    );

    insert_optional(&mut body, "maxOutputTokens", &options.max_output_tokens)?;
    insert_optional(&mut body, "temperature", &options.temperature)?;
    insert_optional(&mut body, "stopSequences", &options.stop_sequences)?;
    insert_optional(&mut body, "topP", &options.top_p)?;
    insert_optional(&mut body, "topK", &options.top_k)?;
    insert_optional(&mut body, "presencePenalty", &options.presence_penalty)?;
    insert_optional(&mut body, "frequencyPenalty", &options.frequency_penalty)?;
    insert_optional(&mut body, "responseFormat", &options.response_format)?;
    insert_optional(&mut body, "seed", &options.seed)?;
    insert_optional(&mut body, "tools", &options.tools)?;
    insert_optional(&mut body, "toolChoice", &options.tool_choice)?;
    insert_optional(&mut body, "includeRawChunks", &options.include_raw_chunks)?;
    insert_optional(&mut body, "reasoning", &options.reasoning)?;

    let effective_headers = options.effective_headers();
    if !effective_headers.is_empty() {
        body.insert(
            "headers".to_string(),
            serde_json::to_value(effective_headers)
                .map_err(|e| LlmError::JsonError(format!("Invalid Gateway headers: {e}")))?,
        );
    }

    if let Some(provider_options) = &options.provider_options
        && !provider_options.is_empty()
    {
        body.insert(
            "providerOptions".to_string(),
            serde_json::to_value(provider_options).map_err(|e| {
                LlmError::JsonError(format!("Invalid Gateway provider options: {e}"))
            })?,
        );
    }

    Ok(serde_json::Value::Object(body))
}

fn insert_optional<T: serde::Serialize>(
    body: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: &Option<T>,
) -> Result<(), LlmError> {
    if let Some(value) = value {
        body.insert(
            key.to_string(),
            serde_json::to_value(value).map_err(|e| {
                LlmError::JsonError(format!("Invalid Gateway request field '{key}': {e}"))
            })?,
        );
    }
    Ok(())
}

fn encode_inline_file_data(body: &mut serde_json::Value) {
    let Some(prompt) = body
        .get_mut("prompt")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };

    for message in prompt {
        let Some(content) = message
            .get_mut("content")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };

        for part in content {
            if part
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| value == "file")
            {
                encode_inline_data_field(part);
            }
        }
    }
}

fn encode_inline_data_field(part: &mut serde_json::Value) {
    let Some(data) = part.get_mut("data") else {
        return;
    };
    let Some(bytes) = data
        .as_array()
        .and_then(|values| json_array_to_bytes(values))
    else {
        return;
    };

    *data = serde_json::json!({
        "type": "data",
        "data": STANDARD.encode(bytes),
    });
}

fn json_array_to_bytes(values: &[serde_json::Value]) -> Option<Vec<u8>> {
    values
        .iter()
        .map(|value| value.as_u64().and_then(|byte| u8::try_from(byte).ok()))
        .collect()
}

fn gateway_embedding_body(request: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
    let mut body = serde_json::Map::new();
    body.insert(
        "values".to_string(),
        serde_json::to_value(&request.input)
            .map_err(|e| LlmError::JsonError(format!("Invalid Gateway embedding values: {e}")))?,
    );

    if !request.provider_options_map.is_empty() {
        body.insert(
            "providerOptions".to_string(),
            serde_json::to_value(&request.provider_options_map).map_err(|e| {
                LlmError::JsonError(format!("Invalid Gateway embedding provider options: {e}"))
            })?,
        );
    }

    Ok(serde_json::Value::Object(body))
}

fn gateway_embedding_response(
    json: serde_json::Value,
    headers: HeaderMap,
    model: String,
) -> Result<EmbeddingResponse, LlmError> {
    let embeddings: Vec<Vec<f32>> =
        serde_json::from_value(json.get("embeddings").cloned().ok_or_else(|| {
            LlmError::ParseError("Gateway embedding response missing embeddings".to_string())
        })?)
        .map_err(|e| LlmError::JsonError(format!("Invalid Gateway embedding vectors: {e}")))?;

    let mut response =
        EmbeddingResponse::new(embeddings, model.clone()).with_response(HttpResponseInfo {
            timestamp: chrono::Utc::now(),
            model_id: Some(model),
            headers: headermap_to_hashmap(&headers),
            body: Some(json.clone()),
        });

    if let Some(tokens) = json
        .get("usage")
        .and_then(|usage| usage.get("tokens"))
        .and_then(serde_json::Value::as_u64)
    {
        let tokens = u32::try_from(tokens).map_err(|_| {
            LlmError::ParseError("Gateway embedding usage tokens exceed u32".to_string())
        })?;
        response = response.with_usage(EmbeddingUsage::new(tokens, tokens));
    }

    if let Some(metadata) = json
        .get("providerMetadata")
        .and_then(serde_json::Value::as_object)
    {
        response.metadata = metadata
            .iter()
            .map(|(provider, value)| (provider.clone(), value.clone()))
            .collect();
    }

    Ok(response)
}

fn gateway_v4_sse_stream<S>(
    mut byte_stream: S,
    include_raw_chunks: bool,
) -> crate::text::LanguageModelV4Stream
where
    S: futures_util::Stream<Item = Result<Vec<u8>, LlmError>> + Send + Unpin + 'static,
{
    Box::pin(async_stream::stream! {
        let mut buffer = String::new();

        while let Some(chunk) = byte_stream.next().await {
            match chunk {
                Ok(bytes) => buffer.push_str(&String::from_utf8_lossy(&bytes)),
                Err(error) => {
                    yield Err(error);
                    return;
                }
            }

            while let Some((index, separator_len)) = next_sse_frame(&buffer) {
                let frame = buffer[..index].to_string();
                buffer.drain(..index + separator_len);
                if let Some(part) = parse_gateway_sse_frame(&frame, include_raw_chunks) {
                    yield part;
                }
            }
        }

        if !buffer.trim().is_empty()
            && let Some(part) = parse_gateway_sse_frame(&buffer, include_raw_chunks)
        {
            yield part;
        }
    })
}

fn next_sse_frame(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|index| (index, 2));
    let crlf = buffer.find("\r\n\r\n").map(|index| (index, 4));

    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(frame), None) | (None, Some(frame)) => Some(frame),
        (None, None) => None,
    }
}

fn parse_gateway_sse_frame(
    frame: &str,
    include_raw_chunks: bool,
) -> Option<Result<crate::streaming::LanguageModelV4StreamPart, LlmError>> {
    let mut data = String::new();
    for line in frame.lines() {
        let line = line.trim_end_matches('\r');
        if line.starts_with(':') {
            continue;
        }
        let Some(value) = line.strip_prefix("data:") else {
            continue;
        };
        if !data.is_empty() {
            data.push('\n');
        }
        data.push_str(value.strip_prefix(' ').unwrap_or(value));
    }

    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return None;
    }

    let value: serde_json::Value = match serde_json::from_str(data) {
        Ok(value) => value,
        Err(error) => {
            return Some(Err(LlmError::ParseError(format!(
                "Failed to parse Gateway language stream part: {error}"
            ))));
        }
    };

    let Some(part) = crate::streaming::LanguageModelV4StreamPart::parse_loose_json(&value) else {
        return Some(Err(LlmError::ParseError(format!(
            "Invalid Gateway language stream part: {value}"
        ))));
    };

    if matches!(
        part,
        crate::streaming::LanguageModelV4StreamPart::Raw { .. }
    ) && !include_raw_chunks
    {
        return None;
    }

    Some(Ok(part))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::types::{
        LanguageModelV4InputTokens, LanguageModelV4OutputTokens, LanguageModelV4Usage,
        ModelMessage, UserContent, UserModelMessage,
    };
    use futures_util::StreamExt;
    use reqwest::header::HeaderMap;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone)]
    struct CapturedRequest {
        url: String,
        headers: HeaderMap,
        body: serde_json::Value,
    }

    #[derive(Clone)]
    struct CaptureTransport {
        calls: Arc<Mutex<Vec<CapturedRequest>>>,
        body: serde_json::Value,
        stream_body: Option<Vec<u8>>,
    }

    impl CaptureTransport {
        fn new(body: serde_json::Value) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                body,
                stream_body: None,
            }
        }

        fn with_stream_body(mut self, stream_body: Vec<u8>) -> Self {
            self.stream_body = Some(stream_body);
            self
        }

        fn only_call(&self) -> CapturedRequest {
            self.calls
                .lock()
                .expect("calls")
                .first()
                .expect("call")
                .clone()
        }
    }

    #[async_trait::async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.calls.lock().expect("calls").push(CapturedRequest {
                url: request.url,
                headers: request.headers,
                body: request.body,
            });

            Ok(HttpTransportResponse {
                status: 200,
                headers: HeaderMap::new(),
                body: serde_json::to_vec(&self.body).expect("response body"),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            self.calls.lock().expect("calls").push(CapturedRequest {
                url: request.url,
                headers: request.headers,
                body: request.body,
            });

            Ok(HttpTransportStreamResponse {
                status: 200,
                headers: HeaderMap::new(),
                body: HttpTransportStreamBody::from_bytes(
                    self.stream_body.clone().expect("stream body"),
                ),
            })
        }
    }

    #[tokio::test]
    async fn language_generate_posts_v4_body_and_gateway_headers() {
        let response = LanguageModelV4GenerateResult::new(
            vec![crate::types::LanguageModelV4Text::new("hello").into()],
            crate::types::FinishReason::Stop,
            LanguageModelV4Usage::new(
                LanguageModelV4InputTokens::with_total(3),
                LanguageModelV4OutputTokens::with_total(4),
            ),
        );
        let transport = Arc::new(CaptureTransport::new(
            serde_json::to_value(response).expect("response json"),
        ));
        let client = GatewayClient::from_config(
            GatewayConfig::new("test-key")
                .with_base_url("https://gateway.test/v4/ai/")
                .with_model("openai/gpt-5-mini")
                .with_team_id_or_slug("team_123")
                .with_header("x-config", "config")
                .with_http_transport(transport.clone()),
        )
        .expect("client");

        let mut provider_options = crate::types::ProviderOptionsMap::new();
        provider_options.insert("gateway", json!({ "order": ["bedrock", "anthropic"] }));
        let mut options =
            LanguageModelV4CallOptions::from_model_messages(vec![ModelMessage::User(
                UserModelMessage::new(UserContent::text("Hello")),
            )])
            .with_header("x-request", "request");
        options.provider_options = Some(provider_options);

        let result = client.do_generate(options).await.expect("generate");

        assert_eq!(result.content.len(), 1);
        let call = transport.only_call();
        assert_eq!(call.url, "https://gateway.test/v4/ai/language-model");
        assert_eq!(
            call.headers
                .get("authorization")
                .and_then(|v| v.to_str().ok()),
            Some("Bearer test-key")
        );
        assert_eq!(
            call.headers
                .get("ai-gateway-protocol-version")
                .and_then(|v| v.to_str().ok()),
            Some("0.0.1")
        );
        assert_eq!(
            call.headers
                .get("ai-gateway-auth-method")
                .and_then(|v| v.to_str().ok()),
            Some("api-key")
        );
        assert_eq!(
            call.headers
                .get("x-vercel-ai-gateway-team")
                .and_then(|v| v.to_str().ok()),
            Some("team_123")
        );
        assert_eq!(
            call.headers
                .get("ai-language-model-specification-version")
                .and_then(|v| v.to_str().ok()),
            Some("4")
        );
        assert_eq!(
            call.headers
                .get("ai-language-model-id")
                .and_then(|v| v.to_str().ok()),
            Some("openai/gpt-5-mini")
        );
        assert_eq!(
            call.headers
                .get("ai-language-model-streaming")
                .and_then(|v| v.to_str().ok()),
            Some("false")
        );
        assert_eq!(call.body["prompt"][0]["role"], json!("user"));
        assert_eq!(call.body["prompt"][0]["content"][0]["text"], json!("Hello"));
        assert_eq!(
            call.body["providerOptions"]["gateway"]["order"],
            json!(["bedrock", "anthropic"])
        );
        assert!(call.body.get("abortSignal").is_none());
    }

    #[tokio::test]
    async fn language_stream_posts_v4_body_and_gateway_headers_and_skips_raw_by_default() {
        let transport = Arc::new(
            CaptureTransport::new(json!({})).with_stream_body(
                concat!(
                    "data: {\"type\":\"text-start\",\"id\":\"0\"}\n\n",
                    "data: {\"type\":\"raw\",\"rawValue\":{\"ignored\":true}}\n\n",
                    "data: {\"type\":\"text-delta\",\"id\":\"0\",\"delta\":\"hi\"}\n\n"
                )
                .as_bytes()
                .to_vec(),
            ),
        );
        let client = GatewayClient::from_config(
            GatewayConfig::new("test-key")
                .with_base_url("https://gateway.test/v4/ai/")
                .with_model("openai/gpt-5-mini")
                .with_http_transport(transport.clone()),
        )
        .expect("client");

        let options = LanguageModelV4CallOptions::from_model_messages(vec![ModelMessage::User(
            UserModelMessage::new(UserContent::text("Hello")),
        )]);

        let result = client.do_stream(options).await.expect("stream");
        let parts = result
            .stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("stream parts");

        assert_eq!(parts.len(), 2);
        assert!(matches!(
            parts[0],
            crate::streaming::LanguageModelV4StreamPart::TextStart { ref id, .. } if id == "0"
        ));
        assert!(matches!(
            parts[1],
            crate::streaming::LanguageModelV4StreamPart::TextDelta {
                ref id,
                ref delta,
                ..
            } if id == "0" && delta == "hi"
        ));

        let call = transport.only_call();
        assert_eq!(call.url, "https://gateway.test/v4/ai/language-model");
        assert_eq!(
            call.headers
                .get("ai-language-model-streaming")
                .and_then(|v| v.to_str().ok()),
            Some("true")
        );
        assert_eq!(call.body["prompt"][0]["role"], json!("user"));
    }

    #[tokio::test]
    async fn chat_request_bridges_to_v4_generate_without_network() {
        let response = LanguageModelV4GenerateResult::new(
            vec![crate::types::LanguageModelV4Text::new("hello from gateway").into()],
            crate::types::FinishReason::Stop,
            LanguageModelV4Usage::new(
                LanguageModelV4InputTokens::with_total(5),
                LanguageModelV4OutputTokens::with_total(6),
            ),
        );
        let transport = Arc::new(CaptureTransport::new(
            serde_json::to_value(response).expect("response json"),
        ));
        let client = GatewayClient::from_config(
            GatewayConfig::new("test-key")
                .with_base_url("https://gateway.test/v4/ai/")
                .with_model("openai/gpt-5-mini")
                .with_http_transport(transport.clone()),
        )
        .expect("client");

        let request = crate::types::ChatRequest::builder()
            .message(crate::types::ChatMessage::user("Hello").build())
            .max_tokens(64)
            .temperature(0.4)
            .provider_option("gateway", json!({ "only": ["openai"] }))
            .build();

        let response = client.chat_request(request).await.expect("chat response");

        assert_eq!(response.content_text(), Some("hello from gateway"));
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );
        assert_eq!(
            response
                .usage
                .as_ref()
                .and_then(|usage| usage.prompt_tokens()),
            Some(5)
        );
        assert_eq!(
            response
                .usage
                .as_ref()
                .and_then(|usage| usage.completion_tokens()),
            Some(6)
        );

        let call = transport.only_call();
        assert_eq!(call.url, "https://gateway.test/v4/ai/language-model");
        assert_eq!(
            call.headers
                .get("ai-language-model-streaming")
                .and_then(|v| v.to_str().ok()),
            Some("false")
        );
        assert_eq!(call.body["prompt"][0]["role"], json!("user"));
        assert_eq!(call.body["maxOutputTokens"], json!(64));
        assert_eq!(call.body["temperature"], json!(0.4));
        assert_eq!(
            call.body["providerOptions"]["gateway"]["only"],
            json!(["openai"])
        );
    }

    #[tokio::test]
    async fn chat_stream_request_bridges_to_v4_stream_without_network() {
        let transport = Arc::new(CaptureTransport::new(json!({})).with_stream_body(
            concat!(
                "data: {\"type\":\"text-start\",\"id\":\"0\"}\n\n",
                "data: {\"type\":\"text-delta\",\"id\":\"0\",\"delta\":\"hi\"}\n\n",
                "data: {\"type\":\"finish\",\"usage\":{\"inputTokens\":{\"total\":2},\"outputTokens\":{\"total\":3}},\"finishReason\":{\"unified\":\"stop\"}}\n\n"
            )
            .as_bytes()
            .to_vec(),
        ));
        let client = GatewayClient::from_config(
            GatewayConfig::new("test-key")
                .with_base_url("https://gateway.test/v4/ai/")
                .with_model("openai/gpt-5-mini")
                .with_http_transport(transport.clone()),
        )
        .expect("client");

        let request = crate::types::ChatRequest::builder()
            .message(crate::types::ChatMessage::user("Hello").build())
            .provider_option("gateway", json!({ "only": ["openai"] }))
            .build();

        let events = client
            .chat_stream_request(request)
            .await
            .expect("chat stream")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("stream events");

        let text = events
            .iter()
            .filter_map(crate::types::ChatStreamEvent::text_delta)
            .collect::<String>();
        assert_eq!(text, "hi");

        let end = events
            .iter()
            .find_map(|event| match event {
                crate::types::ChatStreamEvent::StreamEnd { response } => Some(response),
                _ => None,
            })
            .expect("stream end");
        assert_eq!(end.content_text(), Some("hi"));
        assert_eq!(end.finish_reason, Some(crate::types::FinishReason::Stop));
        assert_eq!(
            end.usage.as_ref().and_then(|usage| usage.prompt_tokens()),
            Some(2)
        );
        assert_eq!(
            end.usage
                .as_ref()
                .and_then(|usage| usage.completion_tokens()),
            Some(3)
        );

        let call = transport.only_call();
        assert_eq!(call.url, "https://gateway.test/v4/ai/language-model");
        assert_eq!(
            call.headers
                .get("ai-language-model-streaming")
                .and_then(|v| v.to_str().ok()),
            Some("true")
        );
        assert_eq!(call.body["prompt"][0]["role"], json!("user"));
        assert_eq!(
            call.body["providerOptions"]["gateway"]["only"],
            json!(["openai"])
        );
    }

    #[tokio::test]
    async fn embedding_posts_values_provider_options_and_gateway_headers() {
        let transport = Arc::new(CaptureTransport::new(json!({
            "embeddings": [[0.1, 0.2]],
            "usage": { "tokens": 7 },
            "providerMetadata": {
                "gateway": { "selectedProvider": "openai" }
            }
        })));
        let client = GatewayClient::from_config(
            GatewayConfig::new("test-key")
                .with_base_url("https://gateway.test/v4/ai/")
                .with_model("openai/text-embedding-3-small")
                .with_http_transport(transport.clone()),
        )
        .expect("client");

        let request = EmbeddingRequest::new(vec!["hello".to_string()])
            .with_model("openai/text-embedding-3-small")
            .with_provider_option("gateway", json!({ "only": ["openai"] }))
            .with_header("x-request", "embed");

        let response = client.embed_with_config(request).await.expect("embedding");

        assert_eq!(response.embeddings, vec![vec![0.1, 0.2]]);
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.total_tokens),
            Some(7)
        );
        assert_eq!(
            response.metadata["gateway"]["selectedProvider"],
            json!("openai")
        );

        let call = transport.only_call();
        assert_eq!(call.url, "https://gateway.test/v4/ai/embedding-model");
        assert_eq!(
            call.headers
                .get("ai-embedding-model-specification-version")
                .and_then(|v| v.to_str().ok()),
            Some("4")
        );
        assert_eq!(
            call.headers
                .get("ai-model-id")
                .and_then(|v| v.to_str().ok()),
            Some("openai/text-embedding-3-small")
        );
        assert_eq!(
            call.headers.get("x-request").and_then(|v| v.to_str().ok()),
            Some("embed")
        );
        assert_eq!(call.body["values"], json!(["hello"]));
        assert_eq!(
            call.body["providerOptions"]["gateway"]["only"],
            json!(["openai"])
        );
    }
}
