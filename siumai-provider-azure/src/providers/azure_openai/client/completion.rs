use super::AzureOpenAiClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, HttpExecutionConfig, execute_json_request};
use crate::standards::openai::utils::{parse_finish_reason, parse_openai_usage_value};
use crate::streaming::{ChatStream, ChatStreamEvent};
use crate::traits::CompletionCapability;
use crate::types::{
    ChatMessage, ChatResponse, ChatStreamFinishInfo, ChatStreamPart, CompletionRequest,
    CompletionResponse, ContentPart, FinishReason, MessageContent, MessageRole, ResponseMetadata,
    Usage, Warning,
};
use async_trait::async_trait;
use chrono::TimeZone;
use std::collections::HashMap;
use std::sync::Arc;

#[allow(unreachable_patterns)]
fn completion_message_text(content: &MessageContent, role_name: &str) -> Result<String, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::MultiModal(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    ContentPart::Text {
                        text: part_text, ..
                    } => text.push_str(part_text),
                    ContentPart::ToolCall { .. } => {
                        return Err(LlmError::UnsupportedOperation(format!(
                            "Completion prompts do not support tool-call parts in {role_name} messages"
                        )));
                    }
                    ContentPart::ToolResult { .. } => {
                        return Err(LlmError::UnsupportedOperation(
                            "Completion prompts do not support tool messages".to_string(),
                        ));
                    }
                    _ => {
                        return Err(LlmError::UnsupportedOperation(format!(
                            "Completion prompts only support text content in {role_name} messages"
                        )));
                    }
                }
            }
            Ok(text)
        }
        _ => Err(LlmError::UnsupportedOperation(format!(
            "Completion prompts do not support structured JSON content in {role_name} messages"
        ))),
    }
}

#[derive(Debug, Clone)]
struct CompletionPromptMaterialization {
    prompt: String,
    stop_sequences: Vec<String>,
}

fn materialize_completion_prompt(
    prompt: &[ChatMessage],
) -> Result<CompletionPromptMaterialization, LlmError> {
    if prompt.is_empty() {
        return Err(LlmError::InvalidParameter(
            "Completion prompt cannot be empty".to_string(),
        ));
    }

    let mut text = String::new();
    let mut remaining = prompt;

    if let Some(first) = prompt.first()
        && first.role == MessageRole::System
    {
        text.push_str(&completion_message_text(&first.content, "system")?);
        text.push_str("\n\n");
        remaining = &prompt[1..];
    }

    for message in remaining {
        match message.role {
            MessageRole::System => {
                return Err(LlmError::InvalidParameter(
                    "Unexpected system message in completion prompt".to_string(),
                ));
            }
            MessageRole::Developer => {
                return Err(LlmError::UnsupportedOperation(
                    "Completion prompts do not support developer messages".to_string(),
                ));
            }
            MessageRole::Tool => {
                return Err(LlmError::UnsupportedOperation(
                    "Completion prompts do not support tool messages".to_string(),
                ));
            }
            MessageRole::User => {
                text.push_str("user:\n");
                text.push_str(&completion_message_text(&message.content, "user")?);
                text.push_str("\n\n");
            }
            MessageRole::Assistant => {
                text.push_str("assistant:\n");
                text.push_str(&completion_message_text(&message.content, "assistant")?);
                text.push_str("\n\n");
            }
        }
    }

    text.push_str("assistant:\n");

    Ok(CompletionPromptMaterialization {
        prompt: text,
        stop_sequences: vec!["\nuser:".to_string()],
    })
}

fn completion_request_id_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    for key in ["x-request-id", "request-id"] {
        if let Some(value) = headers.get(key)
            && let Ok(value) = value.to_str()
            && !value.trim().is_empty()
        {
            return Some(value.to_string());
        }
    }

    None
}

fn completion_created_at(raw: &serde_json::Value) -> Option<chrono::DateTime<chrono::Utc>> {
    let created = raw
        .get("created")
        .and_then(|value| value.as_i64().or_else(|| value.as_u64().map(|v| v as i64)))?;
    chrono::Utc.timestamp_opt(created, 0).single()
}

fn completion_provider_metadata(
    provider_id: &str,
    raw: &serde_json::Value,
) -> Option<HashMap<String, serde_json::Value>> {
    let mut metadata = HashMap::new();

    if let Some(logprobs) = raw
        .get("choices")
        .and_then(|value| value.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("logprobs"))
        .filter(|value| !value.is_null())
    {
        metadata.insert("logprobs".to_string(), logprobs.clone());
    }

    if metadata.is_empty() {
        None
    } else {
        Some(crate::types::provider_metadata::provider_metadata_from_object(provider_id, metadata))
    }
}

fn merge_completion_provider_metadata(
    target: &mut Option<HashMap<String, serde_json::Value>>,
    source: Option<HashMap<String, serde_json::Value>>,
) {
    let Some(source) = source else {
        return;
    };

    let target = target.get_or_insert_with(HashMap::new);
    crate::types::provider_metadata::merge_provider_metadata(target, source);
}

fn flatten_completion_stream_provider_metadata(
    nested: &Option<HashMap<String, serde_json::Value>>,
) -> Option<HashMap<String, serde_json::Value>> {
    nested.clone()
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
    provider_metadata: Option<HashMap<String, serde_json::Value>>,
    stream_start_emitted: bool,
    response_metadata_emitted: bool,
    text_started: bool,
}

impl CompletionStreamState {
    fn response_metadata(&self, provider: &str) -> ResponseMetadata {
        ResponseMetadata {
            id: self.id.clone(),
            model: self.model.clone(),
            created: self.created,
            provider: provider.to_string(),
            request_id: None,
            headers: None,
            body: None,
        }
    }

    fn finish_usage(&self) -> Usage {
        self.usage.clone().unwrap_or_default()
    }

    fn finish_part_provider_metadata(&self) -> Option<HashMap<String, serde_json::Value>> {
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
    include_raw_chunks: bool,
    state: Arc<std::sync::Mutex<CompletionStreamState>>,
}

impl CompletionSseConverter {
    fn new(
        provider_id: impl Into<String>,
        warnings: Vec<Warning>,
        include_raw_chunks: bool,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
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
                .map(ToString::to_string)
                .unwrap_or_default();
            let finish_reason_raw = raw
                .get("choices")
                .and_then(|value| value.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(|value| value.as_str())
                .map(ToString::to_string);
            let finish_reason = finish_reason_raw
                .as_deref()
                .and_then(|value| parse_finish_reason(Some(value)));
            let usage = raw.get("usage").and_then(parse_openai_usage_value);
            let provider_metadata = completion_provider_metadata(&provider_id, &raw);
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
                if !delta.is_empty() {
                    state.text.push_str(&delta);
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
                let emit_text_start = !delta.is_empty() && !state.text_started;
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

            if !delta.is_empty() {
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

fn normalize_completion_option_key(key: &str) -> &str {
    match key {
        "logitBias" => "logit_bias",
        "reasoningEffort" => "reasoning_effort",
        "strictJsonSchema" => "strict_json_schema",
        "forceReasoning" => "force_reasoning",
        "responsesApi" => "responses_api",
        _ => key,
    }
}

fn normalize_completion_logprobs(value: serde_json::Value) -> Option<serde_json::Value> {
    match value {
        serde_json::Value::Bool(true) => Some(serde_json::json!(0)),
        serde_json::Value::Bool(false) | serde_json::Value::Null => None,
        other => Some(other),
    }
}

impl AzureOpenAiClient {
    fn completion_spec(&self) -> crate::providers::azure_openai::AzureOpenAiSpec {
        crate::providers::azure_openai::AzureOpenAiSpec::new(self.config.url_config.clone())
            .with_chat_mode(self.config.chat_mode)
            .with_provider_metadata_key(self.config.provider_metadata_key)
    }

    fn completion_execution_config(
        &self,
        spec: Arc<dyn ProviderSpec>,
        ctx: ProviderContext,
    ) -> HttpExecutionConfig {
        HttpExecutionConfig {
            provider_id: "azure".to_string(),
            http_client: self.http_client.clone(),
            transport: self.config.http_transport.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
        }
    }

    fn completion_url(&self, model: &str, ctx: &ProviderContext) -> String {
        self.completion_spec().build_url(ctx, model, "/completions")
    }

    fn prepare_completion_request(
        &self,
        mut request: CompletionRequest,
    ) -> Result<CompletionRequest, LlmError> {
        request.common_params = crate::utils::chat_request::merge_common_params(
            &self.config.common_params,
            request.common_params,
        );
        if !self.config.provider_options_map.is_empty() {
            let mut merged = self.config.provider_options_map.clone();
            merged.merge_overrides(std::mem::take(&mut request.provider_options_map));
            request.provider_options_map = merged;
        }
        if request.http_config.is_none() {
            request.http_config = Some(self.config.http_config.clone());
        }
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "Azure OpenAI completion request requires a model (deployment id)".to_string(),
            ));
        }

        Ok(request)
    }

    fn completion_provider_options(
        &self,
        request: &CompletionRequest,
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut merged = serde_json::Map::new();

        for options in [
            request.provider_options_map.get_object("openai"),
            request.provider_options_map.get_object("azure"),
        ]
        .into_iter()
        .flatten()
        {
            for (key, value) in options {
                merged.insert(
                    normalize_completion_option_key(key).to_string(),
                    value.clone(),
                );
            }
        }

        if let Some(logprobs) = merged
            .remove("logprobs")
            .or_else(|| merged.remove("top_logprobs"))
            .and_then(normalize_completion_logprobs)
        {
            merged.insert("logprobs".to_string(), logprobs);
        }

        merged
    }

    fn build_completion_body(
        &self,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<(serde_json::Value, Vec<Warning>), LlmError> {
        let prompt = materialize_completion_prompt(&request.prompt)?;
        let mut warnings = Vec::new();

        if request.common_params.top_k.is_some() {
            warnings.push(Warning::unsupported("topK", None::<String>));
        }
        if request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
        {
            warnings.push(Warning::unsupported("tools", None::<String>));
        }
        if request.tool_choice.is_some() {
            warnings.push(Warning::unsupported("toolChoice", None::<String>));
        }
        if request.response_format.is_some() {
            warnings.push(Warning::unsupported(
                "responseFormat",
                Some("JSON response format is not supported."),
            ));
        }

        let stop = prompt
            .stop_sequences
            .into_iter()
            .chain(
                request
                    .common_params
                    .stop_sequences
                    .clone()
                    .unwrap_or_default(),
            )
            .collect::<Vec<_>>();

        let mut body = serde_json::Map::new();
        body.insert(
            "model".to_string(),
            serde_json::Value::String(request.common_params.model.clone()),
        );

        if let Some(max_tokens) = request
            .common_params
            .max_completion_tokens
            .or(request.common_params.max_tokens)
        {
            body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
        }
        if let Some(temperature) = request.common_params.temperature {
            body.insert("temperature".to_string(), serde_json::json!(temperature));
        }
        if let Some(top_p) = request.common_params.top_p {
            body.insert("top_p".to_string(), serde_json::json!(top_p));
        }
        if let Some(frequency_penalty) = request.common_params.frequency_penalty {
            body.insert(
                "frequency_penalty".to_string(),
                serde_json::json!(frequency_penalty),
            );
        }
        if let Some(presence_penalty) = request.common_params.presence_penalty {
            body.insert(
                "presence_penalty".to_string(),
                serde_json::json!(presence_penalty),
            );
        }
        if let Some(seed) = request.common_params.seed {
            body.insert("seed".to_string(), serde_json::json!(seed));
        }

        body.extend(self.completion_provider_options(request));
        body.insert(
            "prompt".to_string(),
            serde_json::Value::String(prompt.prompt),
        );
        if !stop.is_empty() {
            body.insert("stop".to_string(), serde_json::json!(stop));
        }
        if stream {
            body.insert("stream".to_string(), serde_json::json!(true));
            body.insert(
                "stream_options".to_string(),
                serde_json::json!({ "include_usage": true }),
            );
        }

        Ok((serde_json::Value::Object(body), warnings))
    }

    fn build_completion_response(
        &self,
        raw: serde_json::Value,
        headers: &reqwest::header::HeaderMap,
        warnings: Vec<Warning>,
    ) -> CompletionResponse {
        let provider_key = self.config.provider_metadata_key;
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
        let finish_reason = raw_finish_reason
            .as_deref()
            .and_then(|value| parse_finish_reason(Some(value)));

        CompletionResponse {
            text,
            finish_reason,
            raw_finish_reason,
            usage: raw.get("usage").and_then(parse_openai_usage_value),
            response_metadata: Some(ResponseMetadata {
                id: raw
                    .get("id")
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string),
                model: raw
                    .get("model")
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string),
                created: completion_created_at(&raw),
                provider: "azure".to_string(),
                request_id: completion_request_id_from_headers(headers),
                headers: {
                    let headers = crate::execution::http::headers::headermap_to_hashmap(headers);
                    (!headers.is_empty()).then_some(headers)
                },
                body: Some(raw.clone()),
            }),
            warnings: (!warnings.is_empty()).then_some(warnings),
            provider_metadata: completion_provider_metadata(provider_key, &raw),
        }
    }

    async fn completion_request_via_spec(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError> {
        let request = self.prepare_completion_request(request)?;
        let (body, warnings) = self.build_completion_body(&request, false)?;
        let ctx = self.build_context();
        let spec: Arc<dyn ProviderSpec> = Arc::new(self.completion_spec());
        let config = self.completion_execution_config(spec, ctx.clone());
        let result = execute_json_request(
            &config,
            &self.completion_url(&request.common_params.model, &ctx),
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
        let request = self.prepare_completion_request(request)?;
        let (body, warnings) = self.build_completion_body(&request, true)?;
        let disable_compression = request
            .http_config
            .as_ref()
            .map(|config| config.stream_disable_compression)
            .unwrap_or(false);
        let ctx = self.build_context();
        let spec: Arc<dyn ProviderSpec> = Arc::new(self.completion_spec());
        let headers_base = spec.build_headers(&ctx)?;
        let url = self.completion_url(&request.common_params.model, &ctx);
        let request_id = crate::execution::http::interceptor::generate_request_id();
        let converter = crate::streaming::InterceptingConverter {
            interceptors: self.http_interceptors.clone(),
            ctx: crate::execution::http::interceptor::HttpRequestContext {
                request_id: request_id.clone(),
                provider_id: "azure".to_string(),
                url: url.clone(),
                stream: true,
            },
            convert: CompletionSseConverter::new(
                self.config.provider_metadata_key,
                warnings,
                request.stream_options.include_raw_chunks,
            ),
        };

        crate::execution::executors::stream_sse::execute_sse_stream_request_with_headers(
            &self.http_client,
            "azure",
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
impl CompletionCapability for AzureOpenAiClient {
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
    use crate::traits::CompletionCapability;
    use crate::types::ChatStreamPart;
    use futures_util::StreamExt;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_client(base_url: &str) -> AzureOpenAiClient {
        AzureOpenAiClient::from_config(
            crate::providers::azure_openai::AzureOpenAiConfig::new("test-key")
                .with_base_url(base_url)
                .with_model("deployment-id"),
        )
        .expect("azure client")
    }

    #[tokio::test]
    async fn azure_completion_non_stream_uses_completions_body_shape() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/openai/v1/completions"))
            .and(query_param("api-version", "v1"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-request-id", "req_azure_completion")
                    .set_body_json(serde_json::json!({
                        "id": "cmpl-azure-1",
                        "object": "text_completion",
                        "created": 1_718_345_013,
                        "model": "deployment-id",
                        "choices": [
                            {
                                "text": "world",
                                "index": 0,
                                "finish_reason": "stop",
                                "logprobs": {
                                    "tokens": ["world"],
                                    "token_logprobs": [-0.1],
                                    "top_logprobs": [{"world": -0.1}]
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 1,
                            "total_tokens": 12
                        }
                    })),
            )
            .mount(&server)
            .await;

        let client = make_client(&format!("{}/openai", server.uri()));
        let response = client
            .complete(
                CompletionRequest::new("hello")
                    .with_top_k(3.0)
                    .with_response_format(crate::types::ResponseFormat::json_schema(
                        serde_json::json!({ "type": "object" }),
                    ))
                    .with_provider_option("openai", serde_json::json!({ "logprobs": true }))
                    .with_provider_option("azure", serde_json::json!({ "logitBias": { "7": 1 } })),
            )
            .await
            .expect("azure completion response");

        assert_eq!(response.text(), "world");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.raw_finish_reason.as_deref(), Some("stop"));
        assert_eq!(response.id(), Some("cmpl-azure-1"));
        let response_body = response
            .response_metadata
            .as_ref()
            .and_then(|metadata| metadata.body.as_ref())
            .expect("azure completion response body");
        assert_eq!(response_body["id"], serde_json::json!("cmpl-azure-1"));
        assert_eq!(
            response_body["choices"][0]["text"],
            serde_json::json!("world")
        );
        assert_eq!(
            response
                .provider_metadata
                .as_ref()
                .and_then(|root| root.get("azure"))
                .and_then(|meta| meta.get("logprobs")),
            Some(&serde_json::json!({
                "tokens": ["world"],
                "token_logprobs": [-0.1],
                "top_logprobs": [{"world": -0.1}]
            }))
        );
        assert_eq!(response.warnings.as_ref().map(Vec::len), Some(2));

        let requests = server.received_requests().await.expect("received requests");
        let request = requests.first().expect("request");
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("azure completion request body");

        assert_eq!(body["model"], serde_json::json!("deployment-id"));
        assert_eq!(
            body["prompt"],
            serde_json::json!("user:\nhello\n\nassistant:\n")
        );
        assert_eq!(body["logprobs"], serde_json::json!(0));
        assert_eq!(body["logit_bias"], serde_json::json!({ "7": 1 }));
        assert!(
            request
                .headers
                .get("api-key")
                .is_some_and(|value| value.to_str().ok() == Some("test-key"))
        );
    }

    #[tokio::test]
    async fn azure_completion_stream_uses_completions_sse_shape() {
        let server = MockServer::start().await;

        let sse = concat!(
            "data: {\"id\":\"cmpl-azure-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"deployment-id\",\"choices\":[{\"text\":\"Hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"cmpl-azure-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"deployment-id\",\"choices\":[{\"text\":\" world\",\"index\":0,\"finish_reason\":\"stop\",\"logprobs\":{\"tokens\":[\"world\"],\"token_logprobs\":[-0.2],\"top_logprobs\":[{\"world\":-0.2}]}}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
            "data: [DONE]\n\n"
        );

        Mock::given(method("POST"))
            .and(path("/openai/v1/completions"))
            .and(query_param("api-version", "v1"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(sse, "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = make_client(&format!("{}/openai", server.uri()));
        let mut stream = client
            .complete_stream(CompletionRequest::new("hello"))
            .await
            .expect("azure completion stream");

        let mut deltas = Vec::new();
        let mut end = None;
        while let Some(item) = stream.next().await {
            match item.expect("stream event") {
                event if event.text_delta().is_some() => {
                    deltas.push(event.text_delta().expect("text delta").to_string());
                }
                ChatStreamEvent::StreamEnd { response } => {
                    end = Some(response);
                    break;
                }
                ChatStreamEvent::StreamStart { .. } | ChatStreamEvent::Part { .. } => {}
                other => panic!("unexpected azure completion stream event: {other:?}"),
            }
        }

        assert_eq!(deltas, vec!["Hello".to_string(), " world".to_string()]);
        let response = end.expect("stream end");
        assert_eq!(response.content_text(), Some("Hello world"));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            response
                .provider_metadata
                .as_ref()
                .and_then(|root| root.get("azure"))
                .and_then(|meta| meta.get("logprobs")),
            Some(&serde_json::json!({
                "tokens": ["world"],
                "token_logprobs": [-0.2],
                "top_logprobs": [{"world": -0.2}]
            }))
        );

        let requests = server.received_requests().await.expect("received requests");
        let request = requests.first().expect("request");
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("azure completion stream request body");

        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(
            body["stream_options"],
            serde_json::json!({ "include_usage": true })
        );
    }

    #[tokio::test]
    async fn azure_completion_stream_raw_chunks_follow_stream_start_before_response_metadata() {
        let server = MockServer::start().await;

        let sse = concat!(
            "data: {\"id\":\"cmpl-azure-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"deployment-id\",\"choices\":[{\"text\":\"Hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"cmpl-azure-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"deployment-id\",\"choices\":[{\"text\":\" world\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
            "data: [DONE]\n\n"
        );

        Mock::given(method("POST"))
            .and(path("/openai/v1/completions"))
            .and(query_param("api-version", "v1"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(sse, "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = make_client(&format!("{}/openai", server.uri()));
        let mut stream = client
            .complete_stream(CompletionRequest::new("hello").with_include_raw_chunks(true))
            .await
            .expect("azure completion stream");

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.expect("stream event"));
        }

        assert!(matches!(
            events.first(),
            Some(ChatStreamEvent::StreamStart { metadata })
                if metadata.id.as_deref() == Some("cmpl-azure-stream")
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
                if raw_value["id"] == serde_json::json!("cmpl-azure-stream")
        ));
        assert!(matches!(
            parts.get(2),
            Some(ChatStreamPart::ResponseMetadata(metadata))
                if metadata.id.as_deref() == Some("cmpl-azure-stream")
        ));
        assert!(matches!(
            parts.get(3),
            Some(ChatStreamPart::TextStart { id, .. }) if id == "0"
        ));
        assert!(matches!(
            parts.get(4),
            Some(ChatStreamPart::TextDelta { id, delta, .. })
                if id == "0" && delta == "Hello"
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
}
