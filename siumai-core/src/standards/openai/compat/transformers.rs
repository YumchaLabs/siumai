//! Transformers for OpenAI-compatible protocol
//!
//! Centralizes request/response/stream transformation to remove duplication
//! across non-streaming and streaming paths.

use super::adapter::ProviderAdapter;
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
    ChatRequest, ChatResponse, ContentPart, EmbeddingRequest, EmbeddingResponse, FinishReason,
    ImageGenerationRequest, ImageGenerationResponse, MessageContent, Usage,
};
use eventsource_stream::Event;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

fn extract_provider_metadata(
    provider_id: &str,
    raw: &serde_json::Value,
) -> Option<HashMap<String, HashMap<String, serde_json::Value>>> {
    match provider_id {
        // Perplexity extends the OpenAI-like response schema with extra fields such as
        // `search_results` and `videos` (see Perplexity OpenAPI spec). These are intentionally
        // exposed as provider metadata instead of being added to the unified surface.
        "perplexity" => {
            let mut meta = HashMap::<String, serde_json::Value>::new();
            for key in ["search_results", "videos", "citations"] {
                if let Some(v) = raw.get(key) {
                    meta.insert(key.to_string(), v.clone());
                }
            }
            if meta.is_empty() {
                None
            } else {
                Some(HashMap::from([(provider_id.to_string(), meta)]))
            }
        }
        _ => None,
    }
}

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
            "documents": req.documents,
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
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatMessage {
            role: String,
            content: Option<serde_json::Value>,
            tool_calls: Option<Vec<CompatToolCall>>,
            function_call: Option<CompatFunction>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatChoice {
            index: u32,
            message: CompatMessage,
            finish_reason: Option<String>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatPromptTokensDetails {
            cached_tokens: Option<u32>,
            audio_tokens: Option<u32>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct CompatCompletionTokensDetails {
            reasoning_tokens: Option<u32>,
            audio_tokens: Option<u32>,
            accepted_prediction_tokens: Option<u32>,
            rejected_prediction_tokens: Option<u32>,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
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
        }

        let resp: Compat =
            serde_json::from_value(raw.clone()).map_err(|e| LlmError::ParseError(e.to_string()))?;

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
        if let Some(calls) = choice.message.tool_calls {
            for call in calls {
                if let Some(function) = call.function {
                    // Parse arguments string to JSON Value
                    let arguments = serde_json::from_str(&function.arguments)
                        .unwrap_or_else(|_| serde_json::Value::String(function.arguments.clone()));

                    parts.push(ContentPart::tool_call(
                        call.id,
                        function.name,
                        arguments,
                        None,
                    ));
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
        ) {
            // Compatibility: some providers return tool-call payloads as plain JSON text
            // while still reporting `finish_reason: "tool_calls"`.
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

        let usage = resp.usage.map(|u| {
            let mut builder = Usage::builder()
                .prompt_tokens(u.prompt_tokens.unwrap_or(0))
                .completion_tokens(u.completion_tokens.unwrap_or(0))
                .total_tokens(u.total_tokens.unwrap_or(0));

            // Prefer explicit completion_tokens_details.reasoning_tokens when available,
            // otherwise fall back to legacy top-level reasoning_tokens.
            if let Some(details) = &u.prompt_tokens_details {
                if let Some(cached) = details.cached_tokens {
                    builder = builder.with_cached_tokens(cached);
                }
                if let Some(audio) = details.audio_tokens {
                    builder = builder.with_prompt_audio_tokens(audio);
                }
            }

            if let Some(details) = &u.completion_tokens_details {
                if let Some(reasoning) = details.reasoning_tokens {
                    builder = builder.with_reasoning_tokens(reasoning);
                } else if let Some(reasoning) = u.reasoning_tokens {
                    builder = builder.with_reasoning_tokens(reasoning);
                }
                if let Some(audio) = details.audio_tokens {
                    builder = builder.with_completion_audio_tokens(audio);
                }
                if let Some(accepted) = details.accepted_prediction_tokens {
                    builder = builder.with_accepted_prediction_tokens(accepted);
                }
                if let Some(rejected) = details.rejected_prediction_tokens {
                    builder = builder.with_rejected_prediction_tokens(rejected);
                }
            } else if let Some(reasoning) = u.reasoning_tokens {
                builder = builder.with_reasoning_tokens(reasoning);
            }

            builder.build()
        });

        let finish_reason = choice.finish_reason.map(|r| match r.as_str() {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "function_call" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            _ => FinishReason::Other(r),
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

        let provider_metadata = extract_provider_metadata(&self.config.provider_id, raw);

        Ok(ChatResponse {
            id: Some(resp.id),
            content: final_content,
            model: Some(resp.model),
            usage,
            finish_reason,
            audio,
            system_fingerprint: None,
            service_tier: None,
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
}

#[cfg(test)]
mod tests {
    use super::super::adapter::ProviderAdapter;
    use super::super::types as compat_types;
    use super::super::types::RequestType;
    use super::*;

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
        let adapter = Arc::new(DummyAdapter);
        let config = OpenAiCompatibleConfig::new(
            "perplexity",
            "test-key",
            "https://api.perplexity.ai",
            adapter.clone(),
        )
        .with_model("sonar-pro");

        let tx = CompatResponseTransformer { config, adapter };

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

        let tx = CompatResponseTransformer { config, adapter };

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

        let tx = CompatResponseTransformer { config, adapter };

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
}
