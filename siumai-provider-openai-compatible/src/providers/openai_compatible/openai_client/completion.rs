use super::{DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING, OpenAiCompatibleClient};
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, HttpExecutionConfig};
use crate::standards::openai::completion_metadata::{
    completion_created_at, completion_response_metadata, completion_stream_response_metadata,
    extract_completion_provider_metadata, flatten_completion_stream_provider_metadata,
    merge_completion_provider_metadata,
};
use crate::standards::openai::completion_request::{self, CompletionBodyOptions};
use crate::standards::openai::utils::{
    parse_provider_openai_finish_reason, parse_provider_openai_usage_value,
};
use crate::streaming::{ChatStream, ChatStreamEvent};
use crate::traits::CompletionCapability;
use crate::types::{
    ChatResponse, ChatStreamFinishInfo, ChatStreamPart, CompletionRequest, CompletionResponse,
    FinishReason, MessageContent, ProviderMetadataMap, ResponseMetadata, Usage, Warning,
};
use async_trait::async_trait;
use std::sync::Arc;

fn completion_provider_options_key(provider_id: &str) -> String {
    siumai_protocol_openai::standards::openai::compat::metadata::provider_options_key(provider_id)
}

#[derive(Debug, Clone)]
struct CompletionStreamState {
    text: String,
    id: Option<String>,
    model: Option<String>,
    created: Option<chrono::DateTime<chrono::Utc>>,
    usage: Option<Usage>,
    finish_reason: Option<FinishReason>,
    finish_reason_raw: Option<String>,
    warnings: Vec<Warning>,
    provider_metadata: Option<ProviderMetadataMap>,
    stream_start_emitted: bool,
    response_metadata_emitted: bool,
    text_started: bool,
}

impl CompletionStreamState {
    fn response_metadata(&self, provider: &str) -> ResponseMetadata {
        completion_stream_response_metadata(
            provider,
            self.id.as_deref(),
            self.model.as_deref(),
            self.created,
        )
    }

    fn finish_usage(&self) -> Usage {
        self.usage.clone().unwrap_or_default()
    }

    fn finish_part_provider_metadata(&self) -> Option<ProviderMetadataMap> {
        flatten_completion_stream_provider_metadata(&self.provider_metadata)
    }

    fn final_response(&self) -> ChatResponse {
        let mut response = ChatResponse::new(MessageContent::Text(self.text.clone()));
        response.id = self.id.clone();
        response.model = self.model.clone();
        response.usage = self.usage.clone();
        response.finish_reason = Some(self.finish_reason.clone().unwrap_or(FinishReason::Unknown));
        response.raw_finish_reason = self.finish_reason_raw.clone();
        if !self.warnings.is_empty() {
            response.warnings = Some(self.warnings.clone());
        }
        response.provider_metadata = self.provider_metadata.clone();
        response
    }
}

#[derive(Clone)]
struct CompletionSseConverter {
    provider_id: String,
    provider_metadata_key: String,
    include_raw_chunks: bool,
    state: Arc<std::sync::Mutex<CompletionStreamState>>,
}

impl CompletionSseConverter {
    fn new(
        provider_id: impl Into<String>,
        provider_metadata_key: impl Into<String>,
        warnings: Vec<Warning>,
        include_raw_chunks: bool,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            provider_metadata_key: provider_metadata_key.into(),
            include_raw_chunks,
            state: Arc::new(std::sync::Mutex::new(CompletionStreamState {
                text: String::new(),
                id: None,
                model: None,
                created: None,
                usage: None,
                finish_reason: None,
                finish_reason_raw: None,
                warnings,
                provider_metadata: None,
                stream_start_emitted: false,
                response_metadata_emitted: false,
                text_started: false,
            })),
        }
    }
}

impl crate::streaming::SseEventConverter for CompletionSseConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> crate::streaming::SseEventFuture<'_> {
        let provider_id = self.provider_id.clone();
        let provider_metadata_key = self.provider_metadata_key.clone();
        let include_raw_chunks = self.include_raw_chunks;
        let state = self.state.clone();
        Box::pin(async move {
            let raw: serde_json::Value = match serde_json::from_str(&event.data) {
                Ok(raw) => raw,
                Err(err) => {
                    let mut events = Vec::new();
                    if include_raw_chunks {
                        let mut state = state.lock().expect("completion stream state");
                        if !state.stream_start_emitted {
                            let metadata = state.response_metadata(&provider_id);
                            events.push(Ok(ChatStreamEvent::StreamStart {
                                metadata: metadata.clone(),
                            }));
                            events.push(Ok(ChatStreamEvent::Part {
                                part: ChatStreamPart::StreamStart {
                                    warnings: state.warnings.clone(),
                                },
                            }));
                            state.stream_start_emitted = true;
                        }
                        events.push(Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::Raw {
                                raw_value: serde_json::Value::String(event.data.clone()),
                            },
                        }));
                    }
                    events.push(Err(LlmError::ParseError(format!(
                        "Failed to parse completion stream event: {err}"
                    ))));
                    return events;
                }
            };

            let delta = raw
                .get("choices")
                .and_then(|value| value.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("text"))
                .and_then(|value| value.as_str())
                .map(ToString::to_string);
            let finish_reason_raw = raw
                .get("choices")
                .and_then(|value| value.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(|value| value.as_str())
                .map(ToString::to_string);
            let finish_reason = finish_reason_raw.as_deref().and_then(|value| {
                parse_provider_openai_finish_reason(provider_id.as_str(), Some(value))
            });
            let usage = raw
                .get("usage")
                .and_then(|usage| parse_provider_openai_usage_value(provider_id.as_str(), usage));
            let provider_metadata =
                extract_completion_provider_metadata(&provider_metadata_key, &raw);
            let created = completion_created_at(&raw);

            let mut events = Vec::new();
            let (metadata, warnings, emit_stream_start, emit_response_metadata, emit_text_start) = {
                let mut state = state.lock().expect("completion stream state");
                if let Some(id) = raw.get("id").and_then(|value| value.as_str()) {
                    state.id = Some(id.to_string());
                }
                if let Some(model) = raw.get("model").and_then(|value| value.as_str()) {
                    state.model = Some(model.to_string());
                }
                if let Some(created) = created {
                    state.created = Some(created);
                }
                if let Some(usage) = usage {
                    state.usage = Some(usage);
                }
                if let Some(finish_reason) = finish_reason {
                    state.finish_reason = Some(finish_reason);
                }
                if let Some(raw_reason) = finish_reason_raw.clone() {
                    state.finish_reason_raw = Some(raw_reason);
                }
                merge_completion_provider_metadata(&mut state.provider_metadata, provider_metadata);
                if let Some(delta) = delta.as_deref() {
                    state.text.push_str(delta);
                }

                let metadata = state.response_metadata(&provider_id);
                let warnings = state.warnings.clone();
                let emit_stream_start = !state.stream_start_emitted;
                if emit_stream_start {
                    state.stream_start_emitted = true;
                }
                let emit_response_metadata = !state.response_metadata_emitted;
                if emit_response_metadata {
                    state.response_metadata_emitted = true;
                }
                let emit_text_start = delta.is_some() && !state.text_started;
                if emit_text_start {
                    state.text_started = true;
                }

                (
                    metadata,
                    warnings,
                    emit_stream_start,
                    emit_response_metadata,
                    emit_text_start,
                )
            };

            if emit_stream_start {
                events.push(Ok(ChatStreamEvent::StreamStart {
                    metadata: metadata.clone(),
                }));
                events.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::StreamStart { warnings },
                }));
            }

            if include_raw_chunks {
                events.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::Raw { raw_value: raw },
                }));
            }

            if emit_response_metadata {
                events.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::ResponseMetadata(metadata),
                }));
            }

            if emit_text_start {
                events.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::TextStart {
                        id: "0".to_string(),
                        provider_metadata: None,
                    },
                }));
            }

            if let Some(delta) = delta {
                events.push(Ok(ChatStreamEvent::text_delta_part("0", delta)));
            }

            events
        })
    }

    fn is_stream_end_event(&self, event: &eventsource_stream::Event) -> bool {
        event.data.trim() == "[DONE]"
    }

    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        let state = self.state.lock().expect("completion stream state");
        let mut events = Vec::new();

        if state.text_started {
            events.push(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::TextEnd {
                    id: "0".to_string(),
                    provider_metadata: None,
                },
            }));
        }

        events.push(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Finish {
                usage: state.finish_usage(),
                finish_reason: ChatStreamFinishInfo {
                    unified: state.finish_reason.clone().unwrap_or(FinishReason::Unknown),
                    raw: state.finish_reason_raw.clone(),
                },
                provider_metadata: state.finish_part_provider_metadata(),
            },
        }));
        events.push(Ok(ChatStreamEvent::StreamEnd {
            response: state.final_response(),
        }));

        events
    }
}

impl OpenAiCompatibleClient {
    fn prepare_completion_request(
        &self,
        mut request: CompletionRequest,
    ) -> Result<CompletionRequest, LlmError> {
        self.ensure_completion_surface(false)?;
        request.common_params = crate::utils::chat_request::merge_common_params(
            &self.config.common_params,
            request.common_params,
        );
        if request.http_config.is_none() {
            request.http_config = Some(self.config.http_config.clone());
        }
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "OpenAI-compatible completion request requires a model".to_string(),
            ));
        }

        Ok(request)
    }

    fn completion_execution_config(
        &self,
        spec: Arc<dyn ProviderSpec>,
        ctx: ProviderContext,
    ) -> HttpExecutionConfig {
        HttpExecutionConfig {
            provider_id: self.config.provider_id.clone(),
            http_client: self.http_client.clone(),
            transport: self.config.http_transport.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
        }
    }

    fn completion_url(&self) -> String {
        let base_url = self.config.adapter.url_for(
            &self.config.base_url,
            crate::providers::openai_compatible::RequestType::Completion,
        );
        crate::utils::url::with_query_params(&base_url, &self.config.query_params)
    }

    fn completion_provider_options(
        &self,
        request: &CompletionRequest,
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut merged = serde_json::Map::new();

        for options in [Some("openai-compatible"), Some("openaiCompatible")]
            .into_iter()
            .flatten()
            .filter_map(|key| request.provider_options_map.get_object(key))
            .chain(
                siumai_protocol_openai::standards::openai::compat::metadata::provider_options_keys(
                    &self.config.provider_id,
                )
                .into_iter()
                .filter_map(|key| request.provider_options_map.get_object(&key)),
            )
        {
            for (key, value) in options {
                merged.insert(key.clone(), value.clone());
            }
        }

        if let Some(logit_bias) = merged.remove("logitBias") {
            merged.entry("logit_bias".to_string()).or_insert(logit_bias);
        }

        merged
    }

    fn build_completion_body(
        &self,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<(serde_json::Value, Vec<Warning>), LlmError> {
        let (mut body, warnings) = completion_request::build_completion_body(
            request,
            CompletionBodyOptions::new(stream)
                .with_include_usage(stream && self.config.include_usage == Some(true))
                .with_deprecated_openai_compatible_key_warning(Some(
                    DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING,
                ))
                .with_provider_options(self.completion_provider_options(request)),
        )?;
        self.config.adapter.transform_request_params(
            &mut body,
            &request.common_params.model,
            crate::providers::openai_compatible::RequestType::Completion,
        )?;
        if let Some(transformer) = self.request_settings().request_body_transformer {
            transformer.transform_request_body(
                &mut body,
                &request.common_params.model,
                crate::providers::openai_compatible::RequestType::Completion,
            )?;
        }

        Ok((body, warnings))
    }

    fn build_completion_response(
        &self,
        raw: serde_json::Value,
        headers: &reqwest::header::HeaderMap,
        warnings: Vec<Warning>,
    ) -> CompletionResponse {
        let provider_metadata_key = completion_provider_options_key(&self.config.provider_id);
        let text = raw
            .get("choices")
            .and_then(|value| value.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("text"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
            .unwrap_or_default();
        let raw_finish_reason = raw
            .get("choices")
            .and_then(|value| value.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let finish_reason = raw_finish_reason.as_deref().and_then(|value| {
            parse_provider_openai_finish_reason(self.config.provider_id.as_str(), Some(value))
        });

        CompletionResponse {
            text,
            finish_reason,
            raw_finish_reason,
            usage: raw.get("usage").and_then(|usage| {
                parse_provider_openai_usage_value(self.config.provider_id.as_str(), usage)
            }),
            response_metadata: Some(completion_response_metadata(
                self.config.provider_id.clone(),
                &raw,
                headers,
                true,
            )),
            warnings: (!warnings.is_empty()).then_some(warnings),
            provider_metadata: extract_completion_provider_metadata(&provider_metadata_key, &raw),
        }
    }

    async fn completion_request_via_spec(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError> {
        let request = self.prepare_completion_request(request)?;
        let (body, warnings) = self.build_completion_body(&request, false)?;
        let ctx = self.build_context().await?;
        let spec = Arc::new(self.compat_spec());
        let config = self.completion_execution_config(spec.clone(), ctx);
        let url = self.completion_url();

        let result = crate::execution::executors::http_request::execute_json_request(
            &config,
            &url,
            HttpBody::Json(body),
            request.http_config.as_ref(),
            false,
        )
        .await?;

        Ok(self.build_completion_response(result.json, &result.headers, warnings))
    }

    async fn completion_stream_request_via_spec(
        &self,
        request: CompletionRequest,
    ) -> Result<ChatStream, LlmError> {
        self.ensure_completion_surface(true)?;
        let request = self.prepare_completion_request(request)?;
        let (body, warnings) = self.build_completion_body(&request, true)?;
        let disable_compression = request
            .http_config
            .as_ref()
            .map(|config| config.stream_disable_compression)
            .unwrap_or(false);
        let ctx = self.build_context().await?;
        let spec = Arc::new(self.compat_spec());
        let headers_base = spec.build_headers(&ctx)?;
        let url = self.completion_url();
        let request_id = crate::execution::http::interceptor::generate_request_id();
        let converter = crate::streaming::InterceptingConverter {
            interceptors: self.http_interceptors.clone(),
            ctx: crate::execution::http::interceptor::HttpRequestContext {
                request_id: request_id.clone(),
                provider_id: self.config.provider_id.clone(),
                url: url.clone(),
                stream: true,
            },
            convert: CompletionSseConverter::new(
                self.config.provider_id.clone(),
                completion_provider_options_key(&self.config.provider_id),
                warnings,
                request.stream_options.include_raw_chunks,
            ),
        };

        crate::execution::executors::stream_sse::execute_sse_stream_request_with_headers(
            &self.http_client,
            &self.config.provider_id,
            Some(spec.as_ref()),
            &url,
            request_id,
            headers_base,
            body,
            &self.http_interceptors,
            self.retry_options.clone(),
            request.http_config,
            converter,
            disable_compression,
            self.config.http_transport.clone(),
        )
        .await
    }
}

#[async_trait]
impl CompletionCapability for OpenAiCompatibleClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.completion_request_via_spec(request).await
    }

    async fn complete_stream(&self, request: CompletionRequest) -> Result<ChatStream, LlmError> {
        self.completion_stream_request_via_spec(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::providers::openai_compatible::OpenAiCompatibleConfig;
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use crate::types::{ChatMessage, ResponseFormat, Tool, ToolChoice};
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

    fn make_text_streaming_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "compat-chat".to_string(),
            name: "Compat Chat".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: Some("compat-default-model".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_completion_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "openrouter".to_string(),
            name: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["completion".to_string(), "streaming".to_string()],
            default_model: Some("openai/gpt-3.5-turbo-instruct".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    #[derive(Clone)]
    struct JsonResponseTransport {
        response_body: Arc<Vec<u8>>,
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl JsonResponseTransport {
        fn new(response: serde_json::Value) -> Self {
            Self {
                response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
                last: Arc::new(Mutex::new(None)),
            }
        }

        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for JsonResponseTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.response_body.as_ref().clone(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportStreamResponse {
                status: 501,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"error":{"message":"stream unsupported in test","type":"test_error","code":"unsupported"}}"#
                        .to_vec(),
                ),
            })
        }
    }

    #[derive(Clone)]
    struct SseResponseTransport {
        response_body: Arc<Vec<u8>>,
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl SseResponseTransport {
        fn new(body: impl Into<Vec<u8>>) -> Self {
            Self {
                response_body: Arc::new(body.into()),
                last_stream: Arc::new(Mutex::new(None)),
            }
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for SseResponseTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 501,
                headers,
                body: br#"{"error":{"message":"json unsupported in test","type":"test_error","code":"unsupported"}}"#
                    .to_vec(),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            *self.last_stream.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes(self.response_body.as_ref().clone()),
            })
        }
    }

    #[tokio::test]
    async fn completion_request_runtime_routes_to_completions_and_materializes_prompt() {
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "cmpl_1",
            "model": "compat-model",
            "created": 1718345013u64,
            "choices": [{
                "text": "done",
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9
            }
        }));

        let config = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_text_streaming_adapter(),
        )
        .with_model("compat-model")
        .with_http_transport(Arc::new(transport.clone()));
        let client = OpenAiCompatibleClient::new(config).await.unwrap();

        let request = CompletionRequest::from_prompt(vec![
            ChatMessage::system("Be terse.").build(),
            ChatMessage::user("Hello").build(),
            ChatMessage::assistant("Hi").build(),
            ChatMessage::user("Continue").build(),
        ])
        .with_model("compat-model");

        let response = crate::traits::CompletionCapability::complete(&client, request)
            .await
            .unwrap();

        assert_eq!(response.text(), "done");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.raw_finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            response
                .usage
                .as_ref()
                .and_then(|usage| usage.total_tokens()),
            Some(9)
        );
        assert_eq!(
            response
                .response_metadata
                .as_ref()
                .and_then(|metadata| metadata.model.as_deref()),
            Some("compat-model")
        );

        let captured = transport.take().expect("captured completion request");
        assert_eq!(captured.url, "https://api.test.com/v1/completions");
        assert_eq!(captured.body["model"], serde_json::json!("compat-model"));
        assert_eq!(
            captured.body["prompt"],
            serde_json::json!(
                "Be terse.\n\nuser:\nHello\n\nassistant:\nHi\n\nuser:\nContinue\n\nassistant:\n"
            )
        );
        assert_eq!(captured.body["stop"], serde_json::json!(["\nuser:"]));
    }

    #[tokio::test]
    async fn completion_request_runtime_emits_alignment_warnings_and_merges_provider_options() {
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "cmpl_2",
            "model": "compat-model",
            "created": 1718345013u64,
            "choices": [{
                "text": "ok",
                "finish_reason": "stop"
            }]
        }));

        let config = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_text_streaming_adapter(),
        )
        .with_model("compat-model")
        .with_http_transport(Arc::new(transport.clone()));
        let client = OpenAiCompatibleClient::new(config).await.unwrap();

        let request = CompletionRequest::new("hi")
            .with_model("compat-model")
            .with_top_k(20.0)
            .with_tools(vec![Tool::function(
                "lookup",
                "lookup",
                serde_json::json!({ "type": "object" }),
            )])
            .with_tool_choice(ToolChoice::Required)
            .with_response_format(ResponseFormat::json_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                }
            })))
            .with_provider_option(
                "openaiCompatible",
                serde_json::json!({
                    "echo": true,
                    "suffix": " after"
                }),
            )
            .with_provider_option(
                "compat-chat",
                serde_json::json!({
                    "user": "provider-user",
                    "logitBias": {
                        "42": 1.5
                    }
                }),
            );

        let response = crate::traits::CompletionCapability::complete(&client, request)
            .await
            .unwrap();
        let warnings = response.warnings.expect("completion warnings");
        let unsupported_features = warnings
            .into_iter()
            .filter_map(|warning| match warning {
                Warning::Unsupported { feature, .. } => Some(feature),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            unsupported_features,
            vec![
                "topK".to_string(),
                "tools".to_string(),
                "toolChoice".to_string(),
                "responseFormat".to_string()
            ]
        );

        let captured = transport.take().expect("captured completion request");
        assert_eq!(captured.body["echo"], serde_json::json!(true));
        assert_eq!(captured.body["suffix"], serde_json::json!(" after"));
        assert_eq!(captured.body["user"], serde_json::json!("provider-user"));
        assert_eq!(captured.body["logit_bias"]["42"], serde_json::json!(1.5));
        assert!(captured.body.get("tools").is_none());
        assert!(captured.body.get("tool_choice").is_none());
        assert!(captured.body.get("response_format").is_none());
        assert!(captured.body.get("top_k").is_none());
    }

    #[tokio::test]
    async fn completion_stream_request_runtime_routes_to_completions_and_emits_stream_end() {
        let transport = SseResponseTransport::new(
            br#"data: {"id":"cmpl_3","model":"compat-model","choices":[{"text":"Hel","finish_reason":null}]}

data: {"id":"cmpl_3","model":"compat-model","choices":[{"text":"lo","finish_reason":"stop","logprobs":{"tokens":["lo"],"token_logprobs":[-0.3],"top_logprobs":[{"lo":-0.3}]}}],"sources":[{"url":"https://example.com/stream-source"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}

data: [DONE]

"#,
        );

        let config = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_text_streaming_adapter(),
        )
        .with_model("compat-model")
        .with_include_usage(true)
        .with_http_transport(Arc::new(transport.clone()));
        let client = OpenAiCompatibleClient::new(config).await.unwrap();

        let request = CompletionRequest::new("hi").with_model("compat-model");
        let mut stream = crate::traits::CompletionCapability::complete_stream(&client, request)
            .await
            .unwrap();

        let mut text = String::new();
        let mut end = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                event if event.text_delta().is_some() => {
                    text.push_str(event.text_delta().expect("text delta"));
                }
                ChatStreamEvent::StreamEnd { response } => end = Some(response),
                _ => {}
            }
        }

        assert_eq!(text, "Hello");
        let end = end.expect("stream end response");
        assert_eq!(end.id.as_deref(), Some("cmpl_3"));
        assert_eq!(end.model.as_deref(), Some("compat-model"));
        assert_eq!(end.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            end.usage.as_ref().and_then(|usage| usage.total_tokens()),
            Some(3)
        );
        assert_eq!(end.content_text(), Some("Hello"));
        assert_eq!(
            end.provider_metadata
                .as_ref()
                .and_then(|root| root.get("compat-chat"))
                .and_then(|meta| meta.get("logprobs")),
            Some(&serde_json::json!({
                "tokens": ["lo"],
                "token_logprobs": [-0.3],
                "top_logprobs": [{ "lo": -0.3 }]
            }))
        );
        assert_eq!(
            end.provider_metadata
                .as_ref()
                .and_then(|root| root.get("compat-chat"))
                .and_then(|meta| meta.get("sources")),
            Some(&serde_json::json!([{ "url": "https://example.com/stream-source" }]))
        );

        let captured = transport.take_stream().expect("captured completion stream");
        assert_eq!(captured.url, "https://api.test.com/v1/completions");
        assert_eq!(captured.body["stream"], serde_json::json!(true));
        assert_eq!(
            captured.body["stream_options"],
            serde_json::json!({ "include_usage": true })
        );
    }

    #[tokio::test]
    async fn completion_stream_request_runtime_preserves_empty_and_whitespace_text_deltas() {
        let transport = SseResponseTransport::new(
            br#"data: {"id":"cmpl_lossless","model":"compat-model","choices":[{"text":"A","finish_reason":null}]}

data: {"id":"cmpl_lossless","model":"compat-model","choices":[{"text":"\n","finish_reason":null}]}

data: {"id":"cmpl_lossless","model":"compat-model","choices":[{"text":"","finish_reason":null}]}

data: {"id":"cmpl_lossless","model":"compat-model","choices":[{"text":"B","finish_reason":"stop"}]}

data: [DONE]

"#,
        );

        let config = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_completion_adapter(),
        )
        .with_model("compat-model")
        .with_http_transport(Arc::new(transport.clone()));
        let client = OpenAiCompatibleClient::new(config).await.unwrap();

        let request = CompletionRequest::new("hi").with_model("compat-model");
        let mut stream = crate::traits::CompletionCapability::complete_stream(&client, request)
            .await
            .unwrap();

        let mut deltas = Vec::new();
        while let Some(event) = stream.next().await {
            let event = event.unwrap();
            if let Some(delta) = event.text_delta() {
                deltas.push(delta.to_string());
            }
        }

        assert_eq!(deltas, vec!["A", "\n", "", "B"]);
    }

    #[tokio::test]
    async fn completion_stream_request_runtime_emits_raw_chunks_on_part_lane() {
        let transport = SseResponseTransport::new(
            br#"data: {"id":"cmpl_4","model":"compat-model","created":1718345013,"choices":[{"text":"Hel","finish_reason":null}]}

data: {"id":"cmpl_4","model":"compat-model","created":1718345013,"choices":[{"text":"lo","finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}

data: [DONE]

"#,
        );

        let config = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_text_streaming_adapter(),
        )
        .with_model("compat-model")
        .with_include_usage(true)
        .with_http_transport(Arc::new(transport.clone()));
        let client = OpenAiCompatibleClient::new(config).await.unwrap();

        let request = CompletionRequest::new("hi")
            .with_model("compat-model")
            .with_include_raw_chunks(true);
        let mut stream = crate::traits::CompletionCapability::complete_stream(&client, request)
            .await
            .unwrap();

        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.unwrap());
        }

        assert!(matches!(
            events.first(),
            Some(ChatStreamEvent::StreamStart { metadata })
                if metadata.id.as_deref() == Some("cmpl_4")
        ));

        let parts = events
            .iter()
            .filter_map(|event| match event {
                ChatStreamEvent::Part { part } => Some(part),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert!(matches!(
            parts.first(),
            Some(ChatStreamPart::StreamStart { .. })
        ));
        assert!(matches!(
            parts.get(1),
            Some(ChatStreamPart::Raw { raw_value })
                if raw_value["id"] == serde_json::json!("cmpl_4")
        ));
        assert!(matches!(
            parts.get(2),
            Some(ChatStreamPart::ResponseMetadata(metadata))
                if metadata.id.as_deref() == Some("cmpl_4")
        ));
        assert!(matches!(
            parts.get(3),
            Some(ChatStreamPart::TextStart { id, .. }) if id == "0"
        ));
        assert!(matches!(
            parts.get(4),
            Some(ChatStreamPart::TextDelta { id, delta, .. })
                if id == "0" && delta == "Hel"
        ));
        assert!(parts.iter().any(|part| {
            matches!(
                part,
                ChatStreamPart::TextEnd { id, .. } if id == "0"
            )
        }));
        assert!(parts.iter().any(|part| {
            matches!(
                part,
                ChatStreamPart::Finish { finish_reason, .. }
                    if finish_reason.unified == FinishReason::Stop
            )
        }));
    }

    #[tokio::test]
    async fn completion_response_preserves_raw_logprobs_metadata() {
        let client = OpenAiCompatibleClient::new(
            OpenAiCompatibleConfig::new(
                "openrouter",
                "test-key",
                "https://openrouter.ai/api/v1",
                make_completion_adapter(),
            )
            .with_model("openai/gpt-3.5-turbo-instruct"),
        )
        .await
        .expect("build completion client");

        let mut headers = HeaderMap::new();
        headers.insert("request-id", "req_compat_completion".parse().unwrap());

        let response = client.build_completion_response(
            serde_json::json!({
                "id": "cmpl_compat_1",
                "object": "text_completion",
                "created": 1_718_345_013,
                "model": "openai/gpt-3.5-turbo-instruct",
                "choices": [
                    {
                        "text": "hello",
                        "index": 0,
                        "finish_reason": "stop",
                        "logprobs": {
                            "tokens": ["hello"],
                            "token_logprobs": [-0.2],
                            "top_logprobs": [{"hello": -0.2}]
                        }
                    }
                ],
                "sources": [{ "url": "https://example.com/source" }]
            }),
            &headers,
            Vec::new(),
        );

        assert_eq!(response.text(), "hello");
        let response_body = response
            .response_metadata
            .as_ref()
            .and_then(|metadata| metadata.body.as_ref())
            .expect("compatible completion response body");
        assert_eq!(
            response
                .response_metadata
                .as_ref()
                .and_then(|metadata| metadata.request_id.as_deref()),
            Some("req_compat_completion")
        );
        assert_eq!(response_body["id"], serde_json::json!("cmpl_compat_1"));
        assert_eq!(
            response_body["choices"][0]["text"],
            serde_json::json!("hello")
        );
        assert_eq!(
            response
                .provider_metadata
                .as_ref()
                .and_then(|root| root.get("openrouter"))
                .and_then(|meta| meta.get("logprobs")),
            Some(&serde_json::json!({
                "tokens": ["hello"],
                "token_logprobs": [-0.2],
                "top_logprobs": [{"hello": -0.2}]
            }))
        );
        assert_eq!(
            response
                .provider_metadata
                .as_ref()
                .and_then(|root| root.get("openrouter"))
                .and_then(|meta| meta.get("sources")),
            Some(&serde_json::json!([{ "url": "https://example.com/source" }]))
        );
    }

    #[test]
    fn completion_logic_stays_out_of_monolithic_client_module() {
        let source = include_str!("../openai_client.rs");
        for forbidden in [
            "struct CompletionSseConverter",
            "fn build_completion_body(",
            "fn build_completion_response(",
            "impl CompletionCapability for OpenAiCompatibleClient",
        ] {
            assert!(
                !source.contains(forbidden),
                "OpenAI-compatible completion logic should live in openai_client/completion.rs"
            );
        }
    }
}
