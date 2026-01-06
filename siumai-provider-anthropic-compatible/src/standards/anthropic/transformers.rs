//! Transformers for Anthropic Claude
//!
//! Centralizes request/response/stream chunk transformations to reduce duplication
//! across chat capability and streaming implementations.

use crate::error::LlmError;
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::streaming::ChatStreamEvent;
use crate::streaming::SseEventConverter;
use crate::types::{ChatRequest, ChatResponse, FinishReason, MessageContent, Usage};
use eventsource_stream::Event;

use super::thinking::ThinkingResponseParser;
use super::types::{AnthropicChatResponse, AnthropicSpecificParams};
use super::utils::{
    convert_messages as convert_messages_to_anthropic, convert_tools_to_anthropic_format,
    create_usage_from_response, parse_finish_reason, parse_response_content_and_tools,
};
use crate::execution::transformers::request::{
    GenericRequestTransformer, MappingProfile, ProviderRequestHooks, RangeMode, Rule,
};

/// Request transformer for Anthropic
#[derive(Clone, Default)]
pub struct AnthropicRequestTransformer {
    pub specific: Option<AnthropicSpecificParams>,
}

impl AnthropicRequestTransformer {
    pub fn new(specific: Option<AnthropicSpecificParams>) -> Self {
        Self { specific }
    }
}

impl RequestTransformer for AnthropicRequestTransformer {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        fn default_max_tokens_for_model(model: &str) -> u32 {
            // Vercel-aligned defaults (heuristic).
            // - Most Claude 3.x models default to 4096.
            // - Claude 4.5 family defaults to 64000 in the AI SDK fixtures.
            if model.starts_with("claude-sonnet-4-5")
                || model.starts_with("claude-opus-4-5")
                || model.starts_with("claude-haiku-4-5")
            {
                return 64000;
            }

            4096
        }

        // Minimal stable validation
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        // Hooks: build base body, then apply declarative rules
        struct AnthropicChatHooks<'a> {
            specific: Option<&'a AnthropicSpecificParams>,
        }
        impl<'a> ProviderRequestHooks for AnthropicChatHooks<'a> {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                let (messages, system) = convert_messages_to_anthropic(&req.messages)?;
                let mut body = serde_json::json!({
                    "model": req.common_params.model,
                    "messages": messages,
                    // require max_tokens; default when not provided
                    "max_tokens": req
                        .common_params
                        .max_tokens
                        .unwrap_or_else(|| default_max_tokens_for_model(&req.common_params.model)),
                });
                if let Some(sys) = system {
                    body["system"] = serde_json::json!(sys);
                }
                if let Some(t) = req.common_params.temperature {
                    body["temperature"] = serde_json::json!(t);
                }
                if let Some(tp) = req.common_params.top_p {
                    body["top_p"] = serde_json::json!(tp);
                }
                if let Some(stops) = &req.common_params.stop_sequences {
                    body["stop_sequences"] = serde_json::json!(stops);
                }
                if let Some(tools) = &req.tools {
                    let arr = convert_tools_to_anthropic_format(tools)?;
                    if !arr.is_empty() {
                        body["tools"] = serde_json::Value::Array(arr);

                        // Add tool_choice if specified
                        if let Some(choice) = &req.tool_choice
                            && let Some(anthropic_choice) =
                                super::utils::convert_tool_choice(choice)
                        {
                            body["tool_choice"] = anthropic_choice;
                        }
                        // If None is returned, tools should be removed (handled by caller if needed)
                    }
                }

                // Vercel-aligned: request-level `responseFormat: { type: "json", schema }`
                // is implemented via a reserved `json` tool + tool_choice.
                if let Some(crate::types::chat::ResponseFormat::Json { schema }) =
                    &req.response_format
                {
                    let json_tool = serde_json::json!({
                        "name": "json",
                        "description": "Respond with a JSON object.",
                        "input_schema": schema,
                    });

                    let tools = body.get_mut("tools").and_then(|v| v.as_array_mut());
                    match tools {
                        Some(arr) => {
                            let has_json = arr
                                .iter()
                                .any(|t| t.get("name").and_then(|v| v.as_str()) == Some("json"));
                            if !has_json {
                                arr.push(json_tool);
                            }
                        }
                        None => {
                            body["tools"] = serde_json::Value::Array(vec![json_tool]);
                        }
                    }

                    body["tool_choice"] = serde_json::json!({
                        "type": "any",
                        "disable_parallel_tool_use": true,
                    });
                }
                if let Some(spec) = self.specific {
                    if let Some(thinking) = &spec.thinking_config {
                        body["thinking"] = thinking.to_request_params();
                    }
                    if let Some(meta) = &spec.metadata {
                        body["metadata"] = meta.clone();
                    }
                }
                if req.stream {
                    body["stream"] = serde_json::json!(true);
                }
                Ok(body)
            }

            fn post_process_chat(
                &self,
                _req: &ChatRequest,
                _body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // All provider-specific features are now handled via provider_options
                // in ProviderSpec::chat_before_send()
                Ok(())
            }
        }

        let hooks = AnthropicChatHooks {
            specific: self.specific.as_ref(),
        };
        let profile = MappingProfile {
            provider_id: "anthropic",
            rules: vec![
                // Stable ranges only
                Rule::Range {
                    field: "temperature",
                    min: 0.0,
                    max: 1.0,
                    mode: RangeMode::Error,
                    message: Some("Anthropic temperature must be between 0.0 and 1.0"),
                },
                Rule::Range {
                    field: "top_p",
                    min: 0.0,
                    max: 1.0,
                    mode: RangeMode::Error,
                    message: Some("Anthropic top_p must be between 0.0 and 1.0"),
                },
            ],
            // Not used directly; provider params merged via hooks with filtered keys
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = GenericRequestTransformer { profile, hooks };
        generic.transform_chat(req)
    }
}

/// Response transformer for Anthropic
#[derive(Clone, Default)]
pub struct AnthropicResponseTransformer;

impl ResponseTransformer for AnthropicResponseTransformer {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        use super::provider_metadata::{
            AnthropicCitation, AnthropicCitationsBlock, AnthropicServerToolUse, AnthropicSource,
        };
        use crate::types::ContentPart;

        let response: AnthropicChatResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Anthropic response: {e}")))?;

        // Vercel-aligned: when using `responseFormat: { type: "json" }`, Anthropic returns a
        // reserved `json` tool call with the structured object in `input`. Treat it as the final
        // text content instead of a tool call.
        let json_tool_input = response
            .content
            .first()
            .and_then(|b| {
                if b.r#type == "tool_use" && b.name.as_deref() == Some("json") {
                    b.input.clone()
                } else {
                    None
                }
            })
            .filter(|_| response.content.len() == 1);
        let is_json_tool_response = json_tool_input.is_some();

        let mut content = if let Some(input) = json_tool_input {
            let text = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
            MessageContent::Text(text)
        } else {
            parse_response_content_and_tools(&response.content)
        };

        // Add thinking/reasoning if present (preserve signature/redacted data via provider_metadata).
        let thinking_block = ThinkingResponseParser::extract_thinking(raw);
        let redacted_block = ThinkingResponseParser::extract_redacted_thinking(raw);

        if let Some(tb) = &thinking_block
            && !tb.thinking.is_empty()
        {
            let mut parts = match content {
                MessageContent::Text(ref text) if !text.is_empty() => vec![ContentPart::text(text)],
                MessageContent::MultiModal(ref parts) => parts.clone(),
                _ => Vec::new(),
            };
            parts.push(ContentPart::reasoning(&tb.thinking));
            content = MessageContent::MultiModal(parts);
        }

        let usage: Option<Usage> = create_usage_from_response(response.usage.clone());
        let finish_reason: Option<FinishReason> = if is_json_tool_response {
            Some(FinishReason::Stop)
        } else {
            parse_finish_reason(response.stop_reason.as_deref())
        };

        // Provider metadata (Vercel alignment): expose citations + server tool usage counters.
        let provider_metadata = {
            let mut anthropic_meta: std::collections::HashMap<String, serde_json::Value> =
                std::collections::HashMap::new();

            // Usage-derived metadata
            if let Some(u) = &response.usage {
                if let Some(v) = u.cache_creation_input_tokens {
                    anthropic_meta.insert(
                        "cache_creation_input_tokens".to_string(),
                        serde_json::json!(v),
                    );
                }
                if let Some(v) = u.cache_read_input_tokens {
                    anthropic_meta
                        .insert("cache_read_input_tokens".to_string(), serde_json::json!(v));
                }

                if let Some(tier) = &u.service_tier {
                    anthropic_meta.insert("service_tier".to_string(), serde_json::json!(tier));
                }

                if let Some(stu) = &u.server_tool_use {
                    let server_tool_use = AnthropicServerToolUse {
                        web_search_requests: stu.web_search_requests,
                        web_fetch_requests: stu.web_fetch_requests,
                    };
                    if let Ok(v) = serde_json::to_value(server_tool_use) {
                        anthropic_meta.insert("server_tool_use".to_string(), v);
                    }
                }
            }

            // Content-block citations (best-effort; keep unknown fields)
            let mut blocks: Vec<AnthropicCitationsBlock> = Vec::new();
            for (idx, block) in response.content.iter().enumerate() {
                let Some(citations) = &block.citations else {
                    continue;
                };
                if citations.is_empty() {
                    continue;
                }

                let parsed: Vec<AnthropicCitation> = citations
                    .iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect();
                if parsed.is_empty() {
                    continue;
                }

                blocks.push(AnthropicCitationsBlock {
                    content_block_index: idx as u32,
                    citations: parsed,
                });
            }

            if !blocks.is_empty()
                && let Ok(v) = serde_json::to_value(blocks)
            {
                anthropic_meta.insert("citations".to_string(), v);
            }

            // Vercel-aligned sources: extract from web search tool results.
            let mut sources: Vec<AnthropicSource> = Vec::new();
            for block in response.content.iter() {
                if block.r#type != "web_search_tool_result" {
                    continue;
                }
                let Some(tool_use_id) = &block.tool_use_id else {
                    continue;
                };
                let Some(content) = &block.content else {
                    continue;
                };
                let Some(arr) = content.as_array() else {
                    continue;
                };

                for (i, item) in arr.iter().enumerate() {
                    let Some(obj) = item.as_object() else {
                        continue;
                    };
                    let url = obj
                        .get("url")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if url.is_empty() {
                        continue;
                    }
                    let title = obj
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let encrypted_content = obj
                        .get("encrypted_content")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let page_age = obj
                        .get("page_age")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    sources.push(AnthropicSource {
                        id: format!("{tool_use_id}:{i}"),
                        source_type: "url".to_string(),
                        url: Some(url),
                        title,
                        media_type: None,
                        filename: None,
                        page_age,
                        encrypted_content,
                        tool_call_id: Some(tool_use_id.clone()),
                        provider_metadata: None,
                    });
                }
            }

            if !sources.is_empty()
                && let Ok(v) = serde_json::to_value(sources)
            {
                anthropic_meta.insert("sources".to_string(), v);
            }

            // Thinking metadata needed to replay assistant reasoning blocks (Vercel-aligned).
            if let Some(tb) = thinking_block
                && let Some(sig) = tb.signature
            {
                anthropic_meta.insert("thinking_signature".to_string(), serde_json::json!(sig));
            }
            if let Some(rb) = redacted_block {
                anthropic_meta.insert(
                    "redacted_thinking_data".to_string(),
                    serde_json::json!(rb.data),
                );
            }

            if anthropic_meta.is_empty() {
                None
            } else {
                let mut all = std::collections::HashMap::new();
                all.insert("anthropic".to_string(), anthropic_meta);
                Some(all)
            }
        };

        Ok(ChatResponse {
            id: Some(response.id),
            model: Some(response.model),
            content,
            usage,
            finish_reason,
            audio: None, // Anthropic doesn't support audio output
            system_fingerprint: None,
            service_tier: response.usage.as_ref().and_then(|u| u.service_tier.clone()),
            warnings: None,
            provider_metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MessageContent;

    #[test]
    fn captures_thinking_signature_in_provider_metadata() {
        let tx = AnthropicResponseTransformer;
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-7-sonnet-latest",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 1, "output_tokens": 2 },
            "content": [
                { "type": "thinking", "thinking": "t", "signature": "sig" },
                { "type": "text", "text": "ok" }
            ]
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        assert!(matches!(resp.content, MessageContent::MultiModal(_)));
        let meta = resp.provider_metadata.unwrap();
        assert_eq!(
            meta.get("anthropic")
                .unwrap()
                .get("thinking_signature")
                .unwrap(),
            "sig"
        );
    }

    #[test]
    fn captures_redacted_thinking_data_in_provider_metadata() {
        let tx = AnthropicResponseTransformer;
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-7-sonnet-latest",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 1, "output_tokens": 2 },
            "content": [
                { "type": "redacted_thinking", "data": "abc123" },
                { "type": "text", "text": "ok" }
            ]
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        let meta = resp.provider_metadata.unwrap();
        assert_eq!(
            meta.get("anthropic")
                .unwrap()
                .get("redacted_thinking_data")
                .unwrap(),
            "abc123"
        );
    }
}

/// Stream chunk transformer wrapping the existing AnthropicEventConverter
#[derive(Clone)]
pub struct AnthropicStreamChunkTransformer {
    pub provider_id: String,
    pub inner: super::streaming::AnthropicEventConverter,
}

impl StreamChunkTransformer for AnthropicStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Vec<Result<ChatStreamEvent, LlmError>>>
                + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end()
    }
}

// Tests for structured_output via provider_params have been removed
// as this functionality is now handled via provider_options in ProviderSpec::chat_before_send()
