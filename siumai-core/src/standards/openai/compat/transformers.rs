//! Transformers for OpenAI-compatible protocol
//!
//! Centralizes request/response/stream transformation to remove duplication
//! across non-streaming and streaming paths.

use super::adapter::ProviderAdapter;
use super::metadata::{
    ensure_provider_metadata_namespace, nested_provider_metadata_to_map, provider_options_key,
    resolve_provider_metadata_key,
};
use super::openai_config::OpenAiCompatibleConfig;
use super::types::RequestType;
use crate::error::LlmError;
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::standards::openai::utils::convert_messages;
use crate::streaming::ChatStreamEvent;
use crate::streaming::SseEventConverter;
use crate::types::{
    ChatRequest, ChatResponse, ContentPart, EmbeddingRequest, EmbeddingResponse,
    ImageGenerationRequest, ImageGenerationResponse, MessageContent, SourcePart,
};
use eventsource_stream::Event;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Request transformer using OpenAI-compatible adapter
#[derive(Clone)]
pub struct CompatRequestTransformer {
    pub config: OpenAiCompatibleConfig,
    pub adapter: Arc<dyn ProviderAdapter>,
}

impl RequestTransformer for CompatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.config.provider_id
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Convert messages into OpenAI-like format
        let openai_messages = convert_messages(&req.messages)?;
        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": openai_messages,
        });

        // Map common params
        if let Some(temp) = req.common_params.temperature {
            body["temperature"] = temp.into();
        }
        if let Some(max_tokens) = req.common_params.max_tokens {
            body["max_tokens"] = max_tokens.into();
        }
        if let Some(max_completion_tokens) = req.common_params.max_completion_tokens {
            body["max_completion_tokens"] = max_completion_tokens.into();
        }
        if let Some(top_p) = req.common_params.top_p {
            body["top_p"] = top_p.into();
        }
        if let Some(seed) = req.common_params.seed {
            body["seed"] = (seed as i64).into();
        }
        if let Some(stops) = &req.common_params.stop_sequences {
            body["stop"] = serde_json::json!(stops);
        }

        // Tools
        if let Some(tools) = &req.tools {
            body["tools"] = serde_json::to_value(tools)?;
        }

        // Structured output is now handled via provider_options in ProviderSpec::chat_before_send()

        // Let adapter transform
        self.adapter
            .transform_request_params(&mut body, &self.config.model, RequestType::Chat)?;
        Ok(body)
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::json!({
            "model": req.model.clone().unwrap_or_else(|| self.config.model.clone()),
            "input": req.input,
        });
        if let Some(dim) = req.dimensions {
            body["dimensions"] = serde_json::json!(dim);
        }
        if let Some(fmt) = &req.encoding_format {
            body["encoding_format"] = serde_json::to_value(fmt).unwrap_or(serde_json::Value::Null);
        }
        if let Some(user) = &req.user {
            body["user"] = serde_json::json!(user);
        }
        // Provider-specific behavior should be configured via the open `providerOptions` map.
        self.adapter.transform_request_params(
            &mut body,
            &self.config.model,
            RequestType::Embedding,
        )?;
        Ok(body)
    }

    fn transform_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        let mut r = request.clone();
        self.adapter.transform_image_request(&mut r)?;
        serde_json::to_value(r)
            .map_err(|e| LlmError::ParseError(format!("Serialize image request failed: {e}")))
    }

    fn transform_rerank(
        &self,
        req: &crate::types::RerankRequest,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::json!({
            "model": req.model,
            "query": req.query,
            "documents": req.documents.to_strings_lossy(),
        });
        if let Some(n) = req.top_n {
            body["top_n"] = serde_json::json!(n);
        }
        if let Some(instr) = &req.instruction {
            body["instruction"] = serde_json::json!(instr);
        }
        if let Some(rd) = req.return_documents {
            body["return_documents"] = serde_json::json!(rd);
        }
        if let Some(maxc) = req.max_chunks_per_doc {
            body["max_chunks_per_doc"] = serde_json::json!(maxc);
        }
        if let Some(over) = req.overlap_tokens {
            body["overlap_tokens"] = serde_json::json!(over);
        }

        // Let adapter specialize
        self.adapter.transform_request_params(
            &mut body,
            &self.config.model,
            RequestType::Rerank,
        )?;
        Ok(body)
    }
}

/// Response transformer for OpenAI-compatible non-streaming responses
#[derive(Clone)]
pub struct CompatResponseTransformer {
    pub config: OpenAiCompatibleConfig,
    pub adapter: Arc<dyn ProviderAdapter>,
    pub provider_metadata_key: Option<String>,
}

impl CompatResponseTransformer {
    fn resolved_provider_metadata_key(&self) -> String {
        self.provider_metadata_key
            .clone()
            .unwrap_or_else(|| resolve_provider_metadata_key(&self.config.provider_id, None))
    }

    fn raw_provider_metadata_key(&self) -> String {
        provider_options_key(self.adapter.provider_id().as_ref())
    }
}

impl ResponseTransformer for CompatResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.config.provider_id
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        // Map the OpenAI-compatible schema to ChatResponse
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatFunction {
            name: String,
            arguments: String,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatToolCall {
            id: String,
            r#type: String,
            function: Option<CompatFunction>,
            extra_content: Option<CompatToolCallExtraContent>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatToolCallExtraContent {
            google: Option<CompatToolCallGoogleExtraContent>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatToolCallGoogleExtraContent {
            thought_signature: Option<String>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatMessage {
            role: String,
            content: Option<serde_json::Value>,
            tool_calls: Option<Vec<CompatToolCall>>,
            function_call: Option<CompatFunction>,
            annotations: Option<Vec<CompatAnnotation>>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatAnnotation {
            #[serde(default, rename = "type")]
            annotation_type: Option<String>,
            url_citation: Option<CompatUrlCitation>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatUrlCitation {
            url: String,
            title: Option<String>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatChoice {
            index: u32,
            message: CompatMessage,
            finish_reason: Option<String>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize, serde::Serialize)]
        struct CompatPromptTokensDetails {
            cached_tokens: Option<u32>,
            audio_tokens: Option<u32>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize, serde::Serialize)]
        struct CompatCompletionTokensDetails {
            reasoning_tokens: Option<u32>,
            audio_tokens: Option<u32>,
            accepted_prediction_tokens: Option<u32>,
            rejected_prediction_tokens: Option<u32>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize, serde::Serialize)]
        struct CompatUsage {
            prompt_tokens: Option<u32>,
            completion_tokens: Option<u32>,
            total_tokens: Option<u32>,
            reasoning_tokens: Option<u32>,
            prompt_tokens_details: Option<CompatPromptTokensDetails>,
            completion_tokens_details: Option<CompatCompletionTokensDetails>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct Compat {
            id: String,
            model: String,
            choices: Vec<CompatChoice>,
            usage: Option<CompatUsage>,
            system_fingerprint: Option<String>,
            service_tier: Option<String>,
        }

        let resp: Compat =
            serde_json::from_value(raw.clone()).map_err(|e| LlmError::ParseError(e.to_string()))?;
        let response_id = resp.id.clone();

        // Extract thinking content using adapter field mappings directly from raw JSON
        let mappings = self.adapter.get_field_mappings(&self.config.model);
        let accessor = self.adapter.get_field_accessor();
        let thinking_content = accessor.extract_thinking_content(raw, &mappings);

        let choice = resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ApiError {
                code: 500,
                message: "No choices in response".into(),
                details: None,
            })?;

        let content = if let Some(c) = choice.message.content {
            match c {
                serde_json::Value::String(text) => MessageContent::Text(text),
                serde_json::Value::Array(parts) => {
                    let mut out = vec![];
                    for p in parts {
                        if let Some(text) = p.get("text").and_then(|t| t.as_str()) {
                            out.push(crate::types::ContentPart::Text {
                                text: text.to_string(),
                                provider_options: crate::types::ProviderOptionsMap::default(),
                                provider_metadata: None,
                            });
                        }
                    }
                    MessageContent::MultiModal(out)
                }
                _ => MessageContent::Text(String::new()),
            }
        } else {
            MessageContent::Text(String::new())
        };

        // Add tool calls and thinking to content if present
        let mut final_content = content;
        let mut parts = match final_content {
            MessageContent::Text(ref text) if !text.is_empty() => vec![ContentPart::text(text)],
            MessageContent::MultiModal(ref parts) => parts.clone(),
            _ => Vec::new(),
        };

        // Add tool calls
        let provider_metadata_key = self.resolved_provider_metadata_key();

        if let Some(calls) = choice.message.tool_calls {
            for call in calls {
                if let Some(function) = call.function {
                    // Parse arguments string to JSON Value
                    let arguments = serde_json::from_str(&function.arguments)
                        .unwrap_or_else(|_| serde_json::Value::String(function.arguments.clone()));
                    let thought_signature = call
                        .extra_content
                        .as_ref()
                        .and_then(|extra| extra.google.as_ref())
                        .and_then(|google| google.thought_signature.as_deref())
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(ToString::to_string);
                    let mut part = ContentPart::tool_call(call.id, function.name, arguments, None);
                    if let Some(thought_signature) = thought_signature
                        && let ContentPart::ToolCall {
                            provider_metadata, ..
                        } = &mut part
                    {
                        *provider_metadata = Some(std::collections::HashMap::from([(
                            provider_metadata_key.clone(),
                            serde_json::json!({ "thoughtSignature": thought_signature }),
                        )]));
                    }
                    parts.push(part);
                }
            }
        } else if let Some(function) = choice.message.function_call {
            // Legacy OpenAI-compatible field: `message.function_call`.
            let arguments = serde_json::from_str(&function.arguments)
                .unwrap_or_else(|_| serde_json::Value::String(function.arguments.clone()));
            parts.push(ContentPart::tool_call(
                "call_0".to_string(),
                function.name,
                arguments,
                None,
            ));
        } else if matches!(
            choice.finish_reason.as_deref(),
            Some("tool_calls" | "function_call")
        ) || provider_allows_tool_call_json_in_text_fallback(&self.config.provider_id)
        {
            // Compatibility: some OpenAI-compatible providers return tool-call payloads as
            // plain JSON text while still reporting `finish_reason: "tool_calls"`.
            //
            // Additionally, a handful of vendors (e.g. SiliconFlow) may return the tool-call JSON
            // in `message.content` while using `finish_reason: "stop"`. We keep the Vercel-aligned
            // behavior by default, but enable the fallback for known providers that behave this way.
            if parts.len() == 1
                && let Some(text) = parts[0].as_text()
                && let Ok(v) = serde_json::from_str::<serde_json::Value>(text)
                && let Some(obj) = v.as_object()
                && let Some(name) = obj.get("name").and_then(|n| n.as_str())
            {
                let args = obj
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));

                // Replace the raw JSON text with a structured tool call.
                parts.clear();
                parts.push(ContentPart::tool_call(
                    "call_0".to_string(),
                    name.to_string(),
                    args,
                    None,
                ));
            }
        }

        if let Some(annotations) = choice.message.annotations {
            for (annotation_index, annotation) in annotations.into_iter().enumerate() {
                let annotation_type = annotation
                    .annotation_type
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty());
                if annotation_type.is_some_and(|value| !value.eq_ignore_ascii_case("url_citation"))
                {
                    continue;
                }

                let Some(url_citation) = annotation.url_citation else {
                    continue;
                };
                if url_citation.url.trim().is_empty() {
                    continue;
                }

                parts.push(ContentPart::Source {
                    id: compat_source_part_id(Some(response_id.as_str()), annotation_index),
                    source: SourcePart::Url {
                        url: url_citation.url,
                        title: url_citation
                            .title
                            .map(|title| title.trim().to_string())
                            .filter(|title| !title.is_empty()),
                    },
                    provider_metadata: None,
                });
            }
        }

        // Add thinking/reasoning
        if let Some(thinking) = thinking_content
            && !thinking.is_empty()
        {
            parts.push(ContentPart::reasoning(&thinking));
        }

        final_content = if parts.len() == 1 && parts[0].is_text() {
            MessageContent::Text(parts[0].as_text().unwrap_or_default().to_string())
        } else if !parts.is_empty() {
            MessageContent::MultiModal(parts)
        } else {
            MessageContent::Text(String::new())
        };

        let usage = raw.get("usage").and_then(|value| {
            crate::standards::openai::utils::parse_provider_openai_usage_value(
                self.adapter.provider_id().as_ref(),
                value,
            )
        });

        let raw_finish_reason = choice.finish_reason.clone();
        let finish_reason = raw_finish_reason.as_deref().and_then(|reason| {
            crate::standards::openai::utils::parse_provider_openai_finish_reason(
                self.adapter.provider_id().as_ref(),
                Some(reason),
            )
        });

        // Extract audio output if present (OpenAI audio modality)
        let audio = raw
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("audio"))
            .and_then(|aud| {
                let id = aud.get("id")?.as_str()?.to_string();
                let expires_at = aud.get("expires_at")?.as_i64()?;
                let data = aud.get("data")?.as_str()?.to_string();
                let transcript = aud.get("transcript")?.as_str()?.to_string();
                Some(crate::types::AudioOutput {
                    id,
                    expires_at,
                    data,
                    transcript,
                })
            });

        let provider_metadata = Some(nested_provider_metadata_to_map(
            ensure_provider_metadata_namespace(
                self.adapter.extract_response_provider_metadata(raw),
                &provider_metadata_key,
                &self.raw_provider_metadata_key(),
            ),
        ));

        Ok(ChatResponse {
            id: Some(resp.id),
            content: final_content,
            model: Some(resp.model),
            usage,
            finish_reason,
            raw_finish_reason,
            audio,
            system_fingerprint: resp.system_fingerprint,
            service_tier: resp.service_tier,
            warnings: None,
            provider_metadata,
        })
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResponse, LlmError> {
        #[derive(serde::Deserialize)]
        struct CompatEmbeddingData {
            embedding: Vec<f32>,
        }
        #[derive(serde::Deserialize)]
        struct CompatEmbeddingResp {
            data: Vec<CompatEmbeddingData>,
            model: String,
        }
        let r: CompatEmbeddingResp = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid embedding response: {e}")))?;
        let vectors = r.data.into_iter().map(|d| d.embedding).collect();
        Ok(EmbeddingResponse::new(vectors, r.model))
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ImageGenerationResponse, LlmError> {
        serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid image response: {e}")))
    }
}

/// Stream chunk transformer that delegates to the unified converter
#[derive(Clone)]
pub struct CompatStreamChunkTransformer {
    pub provider_id: String,
    pub inner: super::streaming::OpenAiCompatibleEventConverter,
}

impl StreamChunkTransformer for CompatStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        self.inner.convert_event(event)
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end()
    }

    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end_events()
    }
}

fn provider_allows_tool_call_json_in_text_fallback(provider_id: &str) -> bool {
    provider_id.eq_ignore_ascii_case("siliconflow")
        || provider_id.eq_ignore_ascii_case("siliconcloud")
}

fn compat_source_part_id(response_id: Option<&str>, index: usize) -> String {
    match response_id.map(str::trim).filter(|value| !value.is_empty()) {
        Some(response_id) => format!("source_{response_id}_{index}"),
        None => format!("source_{index}"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::adapter::ProviderAdapter;
    use super::super::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use super::super::types as compat_types;
    use super::super::types::RequestType;
    use super::*;
    use crate::types::FinishReason;

    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    struct DummyAdapter;
    impl ProviderAdapter for DummyAdapter {
        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("dummy")
        }
        fn transform_request_params(
            &self,
            _params: &mut serde_json::Value,
            _model: &str,
            _request_type: RequestType,
        ) -> Result<(), LlmError> {
            Ok(())
        }
        fn get_field_mappings(&self, _model: &str) -> compat_types::FieldMappings {
            Default::default()
        }
        fn get_model_config(&self, _model: &str) -> compat_types::ModelConfig {
            Default::default()
        }
        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            Default::default()
        }
        fn base_url(&self) -> &str {
            "https://example.com/v1"
        }
        fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
            Box::new(self.clone())
        }
    }

    // Test for structured_output via provider_params has been removed
    // as this functionality is now handled via provider_options

    #[test]
    fn perplexity_extra_fields_are_exposed_as_provider_metadata() {
        let adapter: Arc<dyn ProviderAdapter> =
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "perplexity".to_string(),
                name: "Perplexity".to_string(),
                base_url: "https://api.perplexity.ai".to_string(),
                field_mappings: ProviderFieldMappings::default(),
                capabilities: vec!["chat".to_string(), "streaming".to_string()],
                default_model: Some("sonar-pro".to_string()),
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            }));
        let config = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter.clone(),
        )
        .with_model("sonar-pro");

        let tx = CompatResponseTransformer {
            config,
            adapter,
            provider_metadata_key: None,
        };

        let raw = serde_json::json!({
            "id": "gen-123",
            "model": "sonar-pro",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hello" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 },
            "search_results": [{ "title": "Example", "url": "https://example.com" }],
            "videos": [{ "title": "Video", "url": "https://example.com/video" }]
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        let meta = resp
            .provider_metadata
            .expect("provider_metadata should exist");
        let perplexity = meta
            .get("perplexity")
            .expect("perplexity namespace should exist");
        let perplexity = perplexity
            .as_object()
            .expect("perplexity provider metadata should be object-shaped");
        assert!(perplexity.contains_key("search_results"));
        assert!(perplexity.contains_key("videos"));
    }

    #[test]
    fn legacy_function_call_is_exposed_as_tool_call_part() {
        let adapter = Arc::new(DummyAdapter);
        let config = OpenAiCompatibleConfig::new(
            "dummy",
            "test-key",
            "https://api.test.com/v1",
            adapter.clone(),
        )
        .with_model("test-model");

        let tx = CompatResponseTransformer {
            config,
            adapter,
            provider_metadata_key: None,
        };

        let raw = serde_json::json!({
            "id": "gen-123",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "function_call": { "name": "weather", "arguments": "{\"location\":\"Rome\"}" }
                },
                "finish_reason": "function_call"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        assert!(resp.has_tool_calls());
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].as_tool_name(), Some("weather"));
    }

    #[test]
    fn tool_call_json_in_text_is_parsed_when_finish_reason_indicates_tool_calls() {
        let adapter = Arc::new(DummyAdapter);
        let config = OpenAiCompatibleConfig::new(
            "dummy",
            "test-key",
            "https://api.test.com/v1",
            adapter.clone(),
        )
        .with_model("test-model");

        let tx = CompatResponseTransformer {
            config,
            adapter,
            provider_metadata_key: None,
        };

        let raw = serde_json::json!({
            "id": "gen-123",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "{\"name\":\"weather$weather#get_weather\",\"arguments\":{\"addr\":\"Guangzhou\",\"date\":\"2026-01-09\"}}"
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        assert!(resp.has_tool_calls());
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].as_tool_name(), Some("weather$weather#get_weather"));
    }

    #[test]
    fn siliconflow_tool_call_json_in_text_is_parsed_even_when_finish_reason_is_stop() {
        let adapter = Arc::new(DummyAdapter);
        let config = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            adapter.clone(),
        )
        .with_model("test-model");

        let tx = CompatResponseTransformer {
            config,
            adapter,
            provider_metadata_key: None,
        };

        let raw = serde_json::json!( {
            "id": "gen-123",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "{\"name\":\"weather$weather#get_weather\",\"arguments\":{\"addr\":\"Guangzhou\",\"date\":\"2026-01-09\"}}"
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        assert!(resp.has_tool_calls());
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].as_tool_name(), Some("weather$weather#get_weather"));
    }

    #[test]
    fn openai_compatible_transformer_preserves_top_level_chat_response_fields() {
        let adapter: Arc<dyn ProviderAdapter> =
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "openai".to_string(),
                name: "OpenAI".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                field_mappings: ProviderFieldMappings::default(),
                capabilities: vec!["chat".to_string(), "streaming".to_string()],
                default_model: Some("gpt-4.1-mini".to_string()),
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            }));
        let config = OpenAiCompatibleConfig::new(
            "openai",
            "test-key",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-4.1-mini");

        let tx = CompatResponseTransformer {
            config,
            adapter,
            provider_metadata_key: None,
        };

        let raw = serde_json::json!({
            "id": "chatcmpl_123",
            "model": "gpt-4.1-mini",
            "system_fingerprint": "fp_123",
            "service_tier": "priority",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_tokens_details": {
                    "cached_tokens": 3,
                    "audio_tokens": 2
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 4,
                    "audio_tokens": 1,
                    "accepted_prediction_tokens": 5,
                    "rejected_prediction_tokens": 6
                }
            }
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        assert_eq!(resp.id.as_deref(), Some("chatcmpl_123"));
        assert_eq!(resp.model.as_deref(), Some("gpt-4.1-mini"));
        assert_eq!(resp.system_fingerprint.as_deref(), Some("fp_123"));
        assert_eq!(resp.service_tier.as_deref(), Some("priority"));
        assert_eq!(resp.finish_reason, Some(FinishReason::Stop));
        assert_eq!(resp.raw_finish_reason.as_deref(), Some("stop"));

        let usage = resp.usage.expect("usage");
        assert_eq!(usage.prompt_tokens(), Some(11));
        assert_eq!(usage.completion_tokens(), Some(7));
        assert_eq!(usage.total_tokens(), Some(18));
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens),
            Some(3)
        );
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.audio_tokens),
            Some(2)
        );
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens),
            Some(4)
        );
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.audio_tokens),
            Some(1)
        );
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.accepted_prediction_tokens),
            Some(5)
        );
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.rejected_prediction_tokens),
            Some(6)
        );

        let provider_metadata = resp.provider_metadata.expect("provider metadata");
        let openai = provider_metadata.get("openai").expect("openai metadata");
        assert_eq!(
            openai.get("acceptedPredictionTokens"),
            Some(&serde_json::json!(5))
        );
        assert_eq!(
            openai.get("rejectedPredictionTokens"),
            Some(&serde_json::json!(6))
        );
    }

    #[test]
    fn generic_openai_compatible_provider_keeps_empty_provider_metadata_root_by_default() {
        let adapter: Arc<dyn ProviderAdapter> =
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "test-provider".to_string(),
                name: "Test Provider".to_string(),
                base_url: "https://api.example.com/v1".to_string(),
                field_mappings: ProviderFieldMappings::default(),
                capabilities: vec!["chat".to_string(), "streaming".to_string()],
                default_model: Some("test-model".to_string()),
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            }));
        let config = OpenAiCompatibleConfig::new(
            "test-provider",
            "test-key",
            "https://api.example.com/v1",
            adapter.clone(),
        )
        .with_model("test-model");

        let tx = CompatResponseTransformer {
            config,
            adapter,
            provider_metadata_key: None,
        };
        let raw = serde_json::json!({
            "id": "chatcmpl_123",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [{
                        "token": "hello",
                        "logprob": -0.1,
                        "bytes": [104, 101, 108, 108, 111],
                        "top_logprobs": []
                    }]
                }
            }],
            "sources": [{
                "url": "https://example.com"
            }],
            "usage": {
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 5,
                    "rejected_prediction_tokens": 6
                }
            }
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        let provider_metadata = resp.provider_metadata.expect("provider metadata root");
        assert_eq!(provider_metadata.len(), 1);
        assert_eq!(
            provider_metadata.get("test-provider"),
            Some(&serde_json::json!({}))
        );
    }

    #[test]
    fn openai_compatible_transformer_uses_requested_metadata_key_for_tool_call_signatures() {
        let adapter: Arc<dyn ProviderAdapter> =
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "test-provider".to_string(),
                name: "Test Provider".to_string(),
                base_url: "https://api.example.com/v1".to_string(),
                field_mappings: ProviderFieldMappings::default(),
                capabilities: vec!["chat".to_string(), "streaming".to_string()],
                default_model: Some("test-model".to_string()),
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            }));
        let config = OpenAiCompatibleConfig::new(
            "test-provider",
            "test-key",
            "https://api.example.com/v1",
            adapter.clone(),
        )
        .with_model("test-model");

        let tx = CompatResponseTransformer {
            config,
            adapter,
            provider_metadata_key: Some("testProvider".to_string()),
        };
        let raw = serde_json::json!({
            "id": "chatcmpl_123",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"q\":\"rust\"}"
                        },
                        "extra_content": {
                            "google": {
                                "thought_signature": "<Sig>"
                            }
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        let provider_metadata = resp.provider_metadata.expect("provider metadata");
        assert!(provider_metadata.get("testProvider").is_some());
        assert!(provider_metadata.get("test-provider").is_none());

        let parts = resp.content.as_multimodal().expect("multimodal content");
        let tool_call = parts
            .iter()
            .find_map(|part| match part {
                ContentPart::ToolCall {
                    provider_metadata, ..
                } => provider_metadata.as_ref(),
                _ => None,
            })
            .expect("tool call provider metadata");
        assert_eq!(
            tool_call
                .get("testProvider")
                .and_then(|value| value.get("thoughtSignature")),
            Some(&serde_json::json!("<Sig>"))
        );
    }

    #[test]
    fn openai_compatible_transformer_exposes_annotations_as_source_parts() {
        let adapter = Arc::new(DummyAdapter);
        let config = OpenAiCompatibleConfig::new(
            "openai",
            "test-key",
            "https://api.openai.com/v1",
            adapter.clone(),
        )
        .with_model("gpt-4.1-mini");

        let tx = CompatResponseTransformer {
            config,
            adapter,
            provider_metadata_key: None,
        };

        let raw = serde_json::json!({
            "id": "chatcmpl_123",
            "model": "gpt-4.1-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com/rust",
                                "title": "Rust"
                            }
                        }
                    ]
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        let parts = resp.content.as_multimodal().expect("multimodal content");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].as_text(), Some("hello"));

        let Some((source_id, source)) = parts[1].as_source() else {
            panic!("expected source part");
        };
        assert_eq!(source_id, "source_chatcmpl_123_0");
        assert_eq!(
            source,
            &SourcePart::Url {
                url: "https://example.com/rust".to_string(),
                title: Some("Rust".to_string()),
            }
        );
    }
}
