//! Transformers for OpenAI-compatible providers
//!
//! Centralizes request/response/stream transformation to remove duplication
//! across non-streaming and streaming paths.

use super::adapter::ProviderAdapter;
use super::openai_config::OpenAiCompatibleConfig;
use super::types::RequestType;
use crate::error::LlmError;
use crate::providers::openai::utils::convert_messages;
use crate::stream::ChatStreamEvent;
use crate::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::types::{
    ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, FinishReason, FunctionCall,
    ImageGenerationRequest, ImageGenerationResponse, MessageContent, ToolCall, Usage,
};
use crate::utils::streaming::SseEventConverter;
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

        // Structured output hint → response_format (json_object/json_schema + strict)
        if let Some(pp) = &req.provider_params {
            if let Some(so) = pp.params.get("structured_output").and_then(|v| v.as_object()) {
                let mut rf: Option<serde_json::Value> = None;
                if let Some(schema_v) = so.get("schema") {
                    if let Some(n) = so.get("name").and_then(|v| v.as_str()) {
                        rf = Some(serde_json::json!({
                            "type": "json_schema",
                            "json_schema": {"name": n, "schema": schema_v, "strict": true}
                        }));
                    } else {
                        rf = Some(serde_json::json!({
                            "type": "json_schema",
                            "json_schema": {"schema": schema_v, "strict": true}
                        }));
                    }
                } else if let Some(t) = so.get("type").and_then(|v| v.as_str()) {
                    if t.eq_ignore_ascii_case("json") || t.eq_ignore_ascii_case("json_object") {
                        rf = Some(serde_json::json!({"type": "json_object"}));
                    }
                }
                if let Some(val) = rf { body["response_format"] = val; }
            }
        }

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
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.provider_params {
                obj.insert(k.clone(), v.clone());
            }
        }
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
        struct CompatUsage {
            prompt_tokens: Option<u32>,
            completion_tokens: Option<u32>,
            total_tokens: Option<u32>,
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

        let tool_calls = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|call| ToolCall {
                    id: call.id,
                    r#type: call.r#type,
                    function: call.function.map(|f| FunctionCall {
                        name: f.name,
                        arguments: f.arguments,
                    }),
                })
                .collect()
        });

        let usage = resp.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens.unwrap_or(0),
            completion_tokens: u.completion_tokens.unwrap_or(0),
            total_tokens: u.total_tokens.unwrap_or(0),
            cached_tokens: None,
            reasoning_tokens: None,
        });

        let finish_reason = choice.finish_reason.map(|r| match r.as_str() {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            _ => FinishReason::Other(r),
        });

        Ok(ChatResponse {
            id: None,
            content,
            model: Some(self.config.model.clone()),
            usage,
            finish_reason,
            tool_calls,
            thinking: thinking_content,
            metadata: std::collections::HashMap::new(),
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
    use super::*;
    use crate::providers::openai_compatible::{self as compat_mod, types as compat_types};
    use crate::providers::openai_compatible::adapter::ProviderAdapter;
    use crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
    use crate::providers::openai_compatible::types::RequestType;

    #[derive(Debug, Clone)]
    struct DummyAdapter;
    impl ProviderAdapter for DummyAdapter {
        fn provider_id(&self) -> &'static str { "dummy" }
        fn transform_request_params(
            &self,
            _params: &mut serde_json::Value,
            _model: &str,
            _request_type: RequestType,
        ) -> Result<(), LlmError> { Ok(()) }
        fn get_field_mappings(&self, _model: &str) -> compat_types::FieldMappings { Default::default() }
        fn get_model_config(&self, _model: &str) -> compat_types::ModelConfig { Default::default() }
        fn capabilities(&self) -> crate::traits::ProviderCapabilities { Default::default() }
        fn base_url(&self) -> &str { "https://example.com/v1" }
        fn clone_adapter(&self) -> Box<dyn ProviderAdapter> { Box::new(self.clone()) }
    }

    #[test]
    fn structured_output_maps_to_response_format() {
        let cfg = OpenAiCompatibleConfig::new("dummy", "sk-", "https://example.com/v1", std::sync::Arc::new(DummyAdapter)).with_model("test-model");
        let tx = CompatRequestTransformer { config: cfg, adapter: std::sync::Arc::new(DummyAdapter) };
        let req = ChatRequest {
            messages: vec![crate::types::ChatMessage::user("hi").build()],
            tools: None,
            common_params: crate::types::CommonParams { model: "test-model".into(), ..Default::default() },
            provider_params: Some(crate::types::ProviderParams::new().with_param(
                "structured_output",
                serde_json::json!({"schema": {"type":"object"}}),
            )),
            http_config: None,
            web_search: None,
            stream: false,
            telemetry: None,
        };
        let body = tx.transform_chat(&req).unwrap();
        assert_eq!(body.get("response_format").and_then(|v| v.get("type")).and_then(|v| v.as_str()), Some("json_schema"));
    }
}
