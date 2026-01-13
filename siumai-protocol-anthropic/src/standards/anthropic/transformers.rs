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
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum StructuredOutputMode {
            Auto,
            OutputFormat,
            JsonTool,
        }

        fn disable_parallel_tool_use(req: &ChatRequest) -> bool {
            req.provider_options_map
                .get("anthropic")
                .and_then(|v| v.as_object())
                .and_then(|o| {
                    o.get("disableParallelToolUse")
                        .or_else(|| o.get("disable_parallel_tool_use"))
                })
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        }

        fn structured_output_mode(req: &ChatRequest) -> StructuredOutputMode {
            let Some(v) = req.provider_options_map.get("anthropic") else {
                return StructuredOutputMode::Auto;
            };
            let Some(obj) = v.as_object() else {
                return StructuredOutputMode::Auto;
            };

            let mode = obj
                .get("structuredOutputMode")
                .or_else(|| obj.get("structured_output_mode"))
                .and_then(|v| v.as_str());

            match mode {
                Some("outputFormat") | Some("output_format") | Some("output-format") => {
                    StructuredOutputMode::OutputFormat
                }
                Some("jsonTool") | Some("json_tool") | Some("json-tool") => {
                    StructuredOutputMode::JsonTool
                }
                _ => StructuredOutputMode::Auto,
            }
        }

        fn send_reasoning(req: &ChatRequest) -> bool {
            req.provider_options_map
                .get("anthropic")
                .and_then(|v| v.as_object())
                .and_then(|o| o.get("sendReasoning").or_else(|| o.get("send_reasoning")))
                .and_then(|v| v.as_bool())
                .unwrap_or(true)
        }

        fn strip_reasoning_inputs(
            messages: &[crate::types::ChatMessage],
        ) -> Vec<crate::types::ChatMessage> {
            messages
                .iter()
                .cloned()
                .map(|mut msg| {
                    if let crate::types::MessageContent::MultiModal(parts) = &mut msg.content {
                        parts.retain(|p| !matches!(p, crate::types::ContentPart::Reasoning { .. }));
                    }
                    msg
                })
                .collect()
        }

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
                let raw_messages = if send_reasoning(req) {
                    req.messages.clone()
                } else {
                    strip_reasoning_inputs(&req.messages)
                };

                let (messages, system) = convert_messages_to_anthropic(&raw_messages)?;
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
                    let clamped = t.max(0.0).min(1.0);
                    body["temperature"] = serde_json::json!(clamped);
                }
                if let Some(tp) = req.common_params.top_p {
                    // Vercel-aligned: `topP` is ignored when `temperature` is set.
                    if req.common_params.temperature.is_none() {
                        body["top_p"] = serde_json::json!(tp);
                    }
                }
                if let Some(tk) = req.common_params.top_k {
                    body["top_k"] = serde_json::json!(tk);
                }
                if let Some(stops) = &req.common_params.stop_sequences {
                    body["stop_sequences"] = serde_json::json!(stops);
                }

                let disable_parallel = disable_parallel_tool_use(req);
                if let Some(tools) = &req.tools
                    && !matches!(req.tool_choice, Some(crate::types::ToolChoice::None))
                {
                    let arr = convert_tools_to_anthropic_format(tools)?;
                    if !arr.is_empty() {
                        body["tools"] = serde_json::Value::Array(arr);

                        // Add tool_choice if specified
                        if let Some(choice) = &req.tool_choice
                            && let Some(anthropic_choice) =
                                super::utils::convert_tool_choice(choice)
                        {
                            let mut tool_choice = anthropic_choice;
                            if disable_parallel && let Some(obj) = tool_choice.as_object_mut() {
                                obj.insert(
                                    "disable_parallel_tool_use".to_string(),
                                    serde_json::json!(true),
                                );
                            }
                            body["tool_choice"] = tool_choice;
                        } else if disable_parallel {
                            // Vercel alignment: `disableParallelToolUse` forces a tool_choice
                            // even when no explicit toolChoice is provided.
                            body["tool_choice"] = serde_json::json!({
                                "type": "auto",
                                "disable_parallel_tool_use": true
                            });
                        }
                    }
                }

                // Vercel-aligned: request-level `responseFormat: { type: "json", schema }`
                // uses `output_format` for supported models, otherwise falls back to a reserved
                // `json` tool + tool_choice.
                if let Some(crate::types::chat::ResponseFormat::Json { schema }) =
                    &req.response_format
                {
                    fn supports_native_output_format(model: &str) -> bool {
                        model.starts_with("claude-sonnet-4-5")
                            || model.starts_with("claude-opus-4-5")
                            || model.starts_with("claude-haiku-4-5")
                    }

                    let mode = structured_output_mode(req);
                    let supports = supports_native_output_format(&req.common_params.model);
                    let use_output_format = match mode {
                        StructuredOutputMode::OutputFormat => true,
                        StructuredOutputMode::JsonTool => false,
                        StructuredOutputMode::Auto => supports,
                    };

                    if use_output_format {
                        body["output_format"] = serde_json::json!({
                            "type": "json_schema",
                            "schema": schema
                        });
                    } else {
                        let json_tool = serde_json::json!({
                            "name": "json",
                            "description": "Respond with a JSON object.",
                            "input_schema": schema,
                        });

                        let tools = body.get_mut("tools").and_then(|v| v.as_array_mut());
                        match tools {
                            Some(arr) => {
                                let has_json = arr.iter().any(|t| {
                                    t.get("name").and_then(|v| v.as_str()) == Some("json")
                                });
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
            AnthropicCitation, AnthropicCitationsBlock, AnthropicSource,
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

        fn canonicalize_json(mut v: serde_json::Value) -> serde_json::Value {
            fn sort_value(v: &mut serde_json::Value) {
                match v {
                    serde_json::Value::Object(map) => {
                        let mut entries: Vec<(String, serde_json::Value)> =
                            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        entries.sort_by(|a, b| a.0.cmp(&b.0));

                        let mut out = serde_json::Map::new();
                        for (k, mut v) in entries {
                            sort_value(&mut v);
                            out.insert(k, v);
                        }
                        *map = out;
                    }
                    serde_json::Value::Array(arr) => {
                        for v in arr.iter_mut() {
                            sort_value(v);
                        }
                    }
                    _ => {}
                }
            }

            sort_value(&mut v);
            v
        }

        let mut content = if let Some(input) = json_tool_input {
            // Vercel-aligned: stable JSON stringification for reserved `json` tool outputs.
            let canonical = canonicalize_json(input);
            let text = serde_json::to_string(&canonical).unwrap_or_else(|_| "{}".to_string());
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

        // Provider metadata (Vercel alignment): match AI SDK `providerMetadata.anthropic` shape.
        let provider_metadata = {
            let mut anthropic_meta: std::collections::HashMap<String, serde_json::Value> =
                std::collections::HashMap::new();

            // Raw envelope metadata: include nulls when provided by the API.
            if let Some(v) = raw.get("container")
                && let Some(mapped) = super::utils::map_container_provider_metadata(v)
            {
                anthropic_meta.insert("container".to_string(), mapped);
            }
            if let Some(v) = raw.get("context_management")
                && let Some(mapped) = super::utils::map_context_management_provider_metadata(v)
            {
                anthropic_meta.insert("contextManagement".to_string(), mapped);
            }
            if let Some(v) = raw.get("stop_sequence") {
                anthropic_meta.insert("stopSequence".to_string(), v.clone());
            }

            // Usage: keep raw usage object (snake_case) and expose derived cache creation tokens.
            if let Some(v) = raw.get("usage") {
                anthropic_meta.insert("usage".to_string(), v.clone());

                let cache_creation = v
                    .get("cache_creation_input_tokens")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                anthropic_meta.insert("cacheCreationInputTokens".to_string(), cache_creation);
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

    #[test]
    fn tool_choice_none_removes_tools_from_request_body() {
        let tx = AnthropicRequestTransformer::default();

        let req = ChatRequest::builder()
            .model("claude-3-7-sonnet-latest")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![crate::types::Tool::function(
                "testFunction",
                "A test function",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(crate::types::ToolChoice::None)
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert!(body.get("tools").is_none(), "tools should be removed");
        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice should be removed"
        );
    }

    #[test]
    fn disable_parallel_tool_use_injects_tool_choice_auto_when_not_specified() {
        let tx = AnthropicRequestTransformer::default();

        let req = ChatRequest::builder()
            .model("claude-3-7-sonnet-latest")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![crate::types::Tool::function(
                "testFunction",
                "A test function",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .provider_option(
                "anthropic",
                serde_json::json!({ "disableParallelToolUse": true }),
            )
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert!(body.get("tools").is_some(), "expected tools in body");
        assert_eq!(
            body.get("tool_choice"),
            Some(&serde_json::json!({
                "type": "auto",
                "disable_parallel_tool_use": true
            }))
        );
    }

    #[test]
    fn disable_parallel_tool_use_is_propagated_for_required_tool_choice() {
        let tx = AnthropicRequestTransformer::default();

        let req = ChatRequest::builder()
            .model("claude-3-7-sonnet-latest")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![crate::types::Tool::function(
                "testFunction",
                "A test function",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(crate::types::ToolChoice::Required)
            .provider_option(
                "anthropic",
                serde_json::json!({ "disableParallelToolUse": true }),
            )
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body.get("tool_choice"),
            Some(&serde_json::json!({
                "type": "any",
                "disable_parallel_tool_use": true
            }))
        );
    }

    #[test]
    fn disable_parallel_tool_use_is_propagated_for_specific_tool_choice() {
        let tx = AnthropicRequestTransformer::default();

        let req = ChatRequest::builder()
            .model("claude-3-7-sonnet-latest")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![crate::types::Tool::function(
                "testFunction",
                "A test function",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(crate::types::ToolChoice::tool("testFunction"))
            .provider_option(
                "anthropic",
                serde_json::json!({ "disableParallelToolUse": true }),
            )
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body.get("tool_choice"),
            Some(&serde_json::json!({
                "type": "tool",
                "name": "testFunction",
                "disable_parallel_tool_use": true
            }))
        );
    }

    #[test]
    fn structured_output_mode_json_tool_forces_json_tool_even_on_supported_model() {
        let tx = AnthropicRequestTransformer::default();

        let req = ChatRequest::builder()
            .model("claude-sonnet-4-5")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(crate::types::chat::ResponseFormat::Json {
                schema: serde_json::json!({"type":"object","properties":{"a":{"type":"string"}}}),
            })
            .provider_option(
                "anthropic",
                serde_json::json!({ "structuredOutputMode": "jsonTool" }),
            )
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert!(
            body.get("output_format").is_none(),
            "expected json tool fallback, got output_format: {:?}",
            body.get("output_format")
        );
        assert!(body.get("tools").is_some(), "expected json tool");
        assert_eq!(
            body.get("tool_choice"),
            Some(&serde_json::json!({"type":"any","disable_parallel_tool_use": true}))
        );
    }

    #[test]
    fn structured_output_auto_uses_output_format_even_with_tools() {
        let tx = AnthropicRequestTransformer::default();

        let req = ChatRequest::builder()
            .model("claude-sonnet-4-5")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![crate::types::Tool::function(
                "get-weather",
                "Get weather",
                serde_json::json!({"type":"object","properties":{"q":{"type":"string"}}}),
            )])
            .response_format(crate::types::chat::ResponseFormat::Json {
                schema: serde_json::json!({"type":"object","properties":{"a":{"type":"string"}}}),
            })
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert!(body.get("tools").is_some(), "expected tools in body");
        assert!(
            body.get("output_format").is_some(),
            "expected output_format when model supports it"
        );
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .expect("tools array");
        assert!(
            !tools
                .iter()
                .any(|t| t.get("name").and_then(|v| v.as_str()) == Some("json")),
            "did not expect json tool when using output_format"
        );
    }

    #[test]
    fn send_reasoning_false_drops_reasoning_inputs() {
        let tx = AnthropicRequestTransformer::default();

        let assistant = crate::types::ChatMessage::assistant_with_content(vec![
            crate::types::ContentPart::reasoning("secret"),
            crate::types::ContentPart::text("ok"),
        ])
        .build();

        let req = ChatRequest::builder()
            .model("claude-3-7-sonnet-latest")
            .messages(vec![
                crate::types::ChatMessage::user("hi").build(),
                assistant,
            ])
            .provider_option("anthropic", serde_json::json!({ "sendReasoning": false }))
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        let msgs = body
            .get("messages")
            .and_then(|v| v.as_array())
            .expect("messages");
        let content = msgs[1]
            .get("content")
            .and_then(|v| v.as_array())
            .expect("content");
        assert!(
            content
                .iter()
                .all(|p| p.get("type").and_then(|v| v.as_str()) != Some("thinking")),
            "expected no thinking blocks when sendReasoning=false: {content:?}"
        );
        assert!(
            !serde_json::to_string(content)
                .unwrap_or_default()
                .contains("<thinking>"),
            "expected no <thinking> wrappers when sendReasoning=false"
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
