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

use super::params::AnthropicParams;
use super::request_options::{
    anthropic_request_body_overlays_needed, apply_anthropic_request_body_overlays,
    cap_max_tokens_for_known_model, finalize_anthropic_thinking_body, set_output_config_field,
};
use super::types::{AnthropicChatResponse, AnthropicSpecificParams};
use super::utils::{
    convert_messages as convert_messages_to_anthropic, convert_tools_to_anthropic_format,
    create_usage_from_json_value, parse_finish_reason,
};
use crate::execution::transformers::request::{
    GenericRequestTransformer, MappingProfile, ProviderRequestHooks, RangeMode, Rule,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuredOutputMode {
    Auto,
    OutputFormat,
    JsonTool,
}

pub(crate) fn anthropic_provider_options_name(provider_id: &str) -> &str {
    provider_id.split('.').next().unwrap_or(provider_id)
}

pub(crate) fn normalize_custom_anthropic_provider_key(key: &str) -> Option<String> {
    let normalized = anthropic_provider_options_name(key).trim();
    if normalized.is_empty() || normalized.eq_ignore_ascii_case("anthropic") {
        None
    } else {
        Some(normalized.to_string())
    }
}

fn anthropic_provider_options_object<'a>(
    req: &'a ChatRequest,
    key: &str,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    req.provider_options_map.get(key)?.as_object()
}

pub(crate) fn used_custom_anthropic_provider_key(
    req: &ChatRequest,
    provider_id: &str,
) -> Option<String> {
    let key = normalize_custom_anthropic_provider_key(provider_id)?;
    anthropic_provider_options_object(req, &key)?;
    Some(key)
}

fn merged_anthropic_provider_options(
    req: &ChatRequest,
    custom_key: Option<&str>,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut merged = serde_json::Map::new();

    if let Some(canonical) = anthropic_provider_options_object(req, "anthropic") {
        merged.extend(canonical.clone());
    }

    if let Some(custom_key) = custom_key.and_then(normalize_custom_anthropic_provider_key)
        && let Some(custom) = anthropic_provider_options_object(req, &custom_key)
    {
        merged.extend(custom.clone());
    }

    if merged.is_empty() {
        None
    } else {
        Some(merged)
    }
}

fn disable_parallel_tool_use(
    provider_options: Option<&serde_json::Map<String, serde_json::Value>>,
) -> bool {
    provider_options
        .and_then(|options| {
            options
                .get("disableParallelToolUse")
                .or_else(|| options.get("disable_parallel_tool_use"))
        })
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn structured_output_mode(
    provider_options: Option<&serde_json::Map<String, serde_json::Value>>,
) -> StructuredOutputMode {
    let Some(options) = provider_options else {
        return StructuredOutputMode::Auto;
    };

    let mode = options
        .get("structuredOutputMode")
        .or_else(|| options.get("structured_output_mode"))
        .and_then(|value| value.as_str());

    match mode {
        Some("outputFormat") | Some("output_format") | Some("output-format") => {
            StructuredOutputMode::OutputFormat
        }
        Some("jsonTool") | Some("json_tool") | Some("json-tool") => StructuredOutputMode::JsonTool,
        _ => StructuredOutputMode::Auto,
    }
}

fn send_reasoning(provider_options: Option<&serde_json::Map<String, serde_json::Value>>) -> bool {
    provider_options
        .and_then(|options| {
            options
                .get("sendReasoning")
                .or_else(|| options.get("send_reasoning"))
        })
        .and_then(|value| value.as_bool())
        .unwrap_or(true)
}

pub(crate) fn anthropic_provider_metadata_map(
    anthropic_meta: std::collections::HashMap<String, serde_json::Value>,
    custom_key: Option<&str>,
) -> crate::types::ProviderMetadataMap {
    anthropic_provider_metadata_map_from_value(
        serde_json::Value::Object(anthropic_meta.into_iter().collect()),
        custom_key,
    )
}

pub(crate) fn anthropic_provider_metadata_map_from_value(
    anthropic_meta: serde_json::Value,
    custom_key: Option<&str>,
) -> crate::types::ProviderMetadataMap {
    let mut provider_metadata = crate::types::ProviderMetadataMap::from([(
        "anthropic".to_string(),
        anthropic_meta.clone(),
    )]);

    if let Some(custom_key) = custom_key.and_then(normalize_custom_anthropic_provider_key) {
        provider_metadata.insert(custom_key, anthropic_meta);
    }

    provider_metadata
}

/// Request transformer for Anthropic
#[derive(Clone, Default)]
pub struct AnthropicRequestTransformer {
    pub specific: Option<AnthropicSpecificParams>,
    pub provider_options_key: Option<String>,
}

impl AnthropicRequestTransformer {
    pub fn new(specific: Option<AnthropicSpecificParams>) -> Self {
        Self {
            specific,
            provider_options_key: None,
        }
    }

    pub fn with_provider_options_key(mut self, key: impl Into<String>) -> Self {
        self.provider_options_key = normalize_custom_anthropic_provider_key(&key.into());
        self
    }
}

impl RequestTransformer for AnthropicRequestTransformer {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
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

        let merged_provider_options =
            merged_anthropic_provider_options(req, self.provider_options_key.as_deref());

        // Hooks: build base body, then apply declarative rules
        struct AnthropicChatHooks<'a> {
            specific: Option<&'a AnthropicSpecificParams>,
            provider_options: Option<&'a serde_json::Map<String, serde_json::Value>>,
        }
        impl<'a> ProviderRequestHooks for AnthropicChatHooks<'a> {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                let raw_messages = if send_reasoning(self.provider_options) {
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
                    let clamped = t.clamp(0.0, 1.0);
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

                let disable_parallel = disable_parallel_tool_use(self.provider_options);
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
                // uses `output_config.format` for supported models, otherwise falls back to a
                // reserved `json` tool + tool_choice.
                if let Some(crate::types::chat::ResponseFormat::Json { schema, .. }) =
                    &req.response_format
                {
                    fn supports_native_structured_output(model: &str) -> bool {
                        model.starts_with("claude-sonnet-4-5")
                            || model.starts_with("claude-opus-4-5")
                            || model.starts_with("claude-haiku-4-5")
                    }

                    let mode = structured_output_mode(self.provider_options);
                    let supports = supports_native_structured_output(&req.common_params.model);
                    let use_native_structured_output = match mode {
                        StructuredOutputMode::OutputFormat => true,
                        StructuredOutputMode::JsonTool => false,
                        StructuredOutputMode::Auto => supports,
                    };

                    if use_native_structured_output {
                        set_output_config_field(
                            &mut body,
                            "format",
                            serde_json::json!({
                                "type": "json_schema",
                                "schema": schema
                            }),
                        );
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
                if anthropic_request_body_overlays_needed(req) {
                    apply_anthropic_request_body_overlays(req, &mut body);
                }
                finalize_anthropic_thinking_body(req, &mut body);
                cap_max_tokens_for_known_model(&req.common_params.model, &mut body);
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
            provider_options: merged_provider_options.as_ref(),
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
pub struct AnthropicResponseTransformer {
    pub provider_metadata_key: Option<String>,
}

impl AnthropicResponseTransformer {
    pub fn with_provider_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.provider_metadata_key = normalize_custom_anthropic_provider_key(&key.into());
        self
    }

    pub(crate) fn transform_chat_response_with_citation_documents(
        &self,
        raw: &serde_json::Value,
        citation_documents: &[crate::standards::anthropic::streaming::AnthropicCitationDocument],
        params: &AnthropicParams,
    ) -> Result<ChatResponse, LlmError> {
        use super::provider_metadata::{AnthropicCitation, AnthropicCitationsBlock};

        let response: AnthropicChatResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Anthropic response: {e}")))?;

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

        let super::utils::ParsedAnthropicResponseContent {
            content,
            sources: parsed_sources,
        } = if let Some(input) = json_tool_input {
            let canonical = canonicalize_json(input);
            let text = serde_json::to_string(&canonical).unwrap_or_else(|_| "{}".to_string());
            super::utils::ParsedAnthropicResponseContent {
                content: MessageContent::Text(text),
                sources: Vec::new(),
            }
        } else {
            super::utils::parse_response_content_and_tools_with_context_and_params(
                &response.content,
                citation_documents,
                params,
            )
        };

        let usage: Option<Usage> = create_usage_from_json_value(raw.get("usage"));
        let finish_reason: Option<FinishReason> = if is_json_tool_response {
            Some(FinishReason::Stop)
        } else {
            parse_finish_reason(response.stop_reason.as_deref())
        };

        let provider_metadata = {
            let mut anthropic_meta: std::collections::HashMap<String, serde_json::Value> =
                std::collections::HashMap::new();

            anthropic_meta.insert(
                "container".to_string(),
                raw.get("container")
                    .and_then(super::utils::map_container_provider_metadata)
                    .unwrap_or(serde_json::Value::Null),
            );
            anthropic_meta.insert(
                "contextManagement".to_string(),
                raw.get("context_management")
                    .and_then(super::utils::map_context_management_provider_metadata)
                    .unwrap_or(serde_json::Value::Null),
            );
            anthropic_meta.insert(
                "stopSequence".to_string(),
                raw.get("stop_sequence")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            );

            let usage_value = raw.get("usage").cloned().unwrap_or(serde_json::Value::Null);
            anthropic_meta.insert("usage".to_string(), usage_value.clone());
            anthropic_meta.insert(
                "cacheCreationInputTokens".to_string(),
                usage_value
                    .get("cache_creation_input_tokens")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            );
            anthropic_meta.insert(
                "iterations".to_string(),
                super::utils::map_usage_iterations_provider_metadata(&usage_value),
            );

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

            if !parsed_sources.is_empty()
                && let Ok(v) = serde_json::to_value(parsed_sources)
            {
                anthropic_meta.insert("sources".to_string(), v);
            }

            Some(anthropic_provider_metadata_map(
                anthropic_meta,
                self.provider_metadata_key.as_deref(),
            ))
        };

        Ok(ChatResponse {
            id: Some(response.id),
            model: Some(response.model.clone()),
            content,
            usage,
            finish_reason,
            raw_finish_reason: response.stop_reason,
            audio: None,
            system_fingerprint: None,
            service_tier: response.usage.as_ref().and_then(|u| u.service_tier.clone()),
            warnings: None,
            request: None,
            response: Some(crate::types::HttpResponseInfo {
                timestamp: chrono::Utc::now(),
                model_id: Some(response.model),
                headers: std::collections::HashMap::new(),
                body: Some(raw.clone()),
            }),
            provider_metadata,
        })
    }
}

impl ResponseTransformer for AnthropicResponseTransformer {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        self.transform_chat_response_with_citation_documents(raw, &[], &AnthropicParams::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_metadata::anthropic::AnthropicChatResponseExt;
    use crate::types::MessageContent;

    #[test]
    fn captures_thinking_signature_in_provider_metadata() {
        let tx = AnthropicResponseTransformer::default();
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
        let MessageContent::MultiModal(parts) = &resp.content else {
            panic!("expected multimodal content");
        };
        let reasoning = parts
            .iter()
            .find(|part| matches!(part, crate::types::ContentPart::Reasoning { .. }))
            .expect("reasoning part");
        let crate::types::ContentPart::Reasoning {
            provider_metadata: Some(provider_metadata),
            ..
        } = reasoning
        else {
            panic!("expected reasoning provider metadata");
        };
        let anthropic_meta = provider_metadata
            .get("anthropic")
            .expect("anthropic reasoning metadata");
        assert_eq!(
            anthropic_meta.get("signature"),
            Some(&serde_json::json!("sig"))
        );
        let meta = resp.anthropic_metadata().expect("anthropic metadata");
        assert_eq!(meta.thinking_signature.as_deref(), Some("sig"));
    }

    #[test]
    fn anthropic_chat_response_preserves_raw_response_body() {
        let tx = AnthropicResponseTransformer::default();
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-7-sonnet-latest",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 1, "output_tokens": 2 },
            "content": [
                { "type": "text", "text": "ok" }
            ]
        });

        let resp = tx.transform_chat_response(&raw).unwrap();
        let response_info = resp.response.expect("response metadata");

        assert_eq!(
            response_info.model_id.as_deref(),
            Some("claude-3-7-sonnet-latest")
        );
        assert!(response_info.headers.is_empty());
        assert_eq!(
            response_info.body.as_ref().expect("raw body")["id"],
            "msg_1"
        );
    }

    #[test]
    fn captures_redacted_thinking_data_in_provider_metadata() {
        let tx = AnthropicResponseTransformer::default();
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
        let MessageContent::MultiModal(parts) = &resp.content else {
            panic!("expected multimodal content");
        };
        let reasoning = parts
            .iter()
            .find(|part| matches!(part, crate::types::ContentPart::Reasoning { .. }))
            .expect("reasoning part");
        let crate::types::ContentPart::Reasoning {
            provider_metadata: Some(provider_metadata),
            ..
        } = reasoning
        else {
            panic!("expected reasoning provider metadata");
        };
        let anthropic_meta = provider_metadata
            .get("anthropic")
            .expect("anthropic reasoning metadata");
        assert_eq!(
            anthropic_meta.get("redactedData"),
            Some(&serde_json::json!("abc123"))
        );
        let meta = resp.anthropic_metadata().expect("anthropic metadata");
        assert_eq!(meta.redacted_thinking_data.as_deref(), Some("abc123"));
    }

    #[test]
    fn response_provider_metadata_exposes_stop_sequence_and_iterations() {
        let tx = AnthropicResponseTransformer::default();
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-7-sonnet-latest",
            "stop_reason": "stop_sequence",
            "stop_sequence": "STOP",
            "usage": {
                "input_tokens": 17,
                "output_tokens": 2,
                "cache_creation_input_tokens": 10,
                "iterations": [
                    {
                        "type": "compaction",
                        "input_tokens": 6,
                        "output_tokens": 1
                    },
                    {
                        "type": "message",
                        "input_tokens": 11,
                        "output_tokens": 1
                    }
                ]
            },
            "content": [{ "type": "text", "text": "done" }]
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let anthropic_meta = resp
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("anthropic"))
            .expect("anthropic metadata");
        assert_eq!(
            anthropic_meta.get("stopSequence"),
            Some(&serde_json::json!("STOP"))
        );
        assert_eq!(
            anthropic_meta
                .get("iterations")
                .and_then(|iterations| iterations.pointer("/0/inputTokens")),
            Some(&serde_json::json!(6))
        );
        assert_eq!(
            anthropic_meta
                .get("iterations")
                .and_then(|iterations| iterations.pointer("/1/outputTokens")),
            Some(&serde_json::json!(1))
        );

        let typed = resp.anthropic_metadata().expect("typed anthropic metadata");
        assert_eq!(typed.stop_sequence.as_deref(), Some("STOP"));
        let iterations = typed.iterations.expect("iterations");
        assert_eq!(iterations.len(), 2);
        assert_eq!(iterations[0].r#type, "compaction");
        assert_eq!(iterations[0].input_tokens, 6);
        assert_eq!(iterations[1].r#type, "message");
        assert_eq!(iterations[1].output_tokens, 1);
    }

    #[test]
    fn response_usage_preserves_full_raw_usage_object() {
        let tx = AnthropicResponseTransformer::default();
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-7-sonnet-latest",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 20,
                "output_tokens": 50,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 5,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 0,
                    "ephemeral_1h_input_tokens": 10
                },
                "inference_geo": "not_available"
            },
            "content": [{ "type": "text", "text": "done" }]
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let usage = resp.usage.expect("usage");

        assert_eq!(usage.raw_usage_value(), raw.get("usage").cloned());
        assert_eq!(usage.normalized_input_tokens().total, Some(35));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(5));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(10));
        assert_eq!(usage.normalized_output_tokens().total, Some(50));
    }

    #[test]
    fn response_provider_metadata_keeps_null_container_and_context_management_fields() {
        let tx = AnthropicResponseTransformer::default();
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-7-sonnet-latest",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 17,
                "output_tokens": 2
            },
            "content": [{ "type": "text", "text": "done" }]
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let anthropic_meta = resp
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("anthropic"))
            .expect("anthropic metadata");
        assert!(
            anthropic_meta
                .get("container")
                .is_some_and(serde_json::Value::is_null)
        );
        assert!(
            anthropic_meta
                .get("contextManagement")
                .is_some_and(serde_json::Value::is_null)
        );
        assert!(
            anthropic_meta
                .get("stopSequence")
                .is_some_and(serde_json::Value::is_null)
        );
    }

    #[test]
    fn response_transformer_with_citation_documents_emits_document_source_parts() {
        let tx = AnthropicResponseTransformer::default();
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 4,
                "output_tokens": 30
            },
            "content": [{
                "type": "text",
                "text": "Based on the document, the results show positive growth.",
                "citations": [{
                    "type": "page_location",
                    "cited_text": "Revenue increased by 25% year over year",
                    "document_index": 0,
                    "document_title": "Financial Report 2023",
                    "start_page_number": 5,
                    "end_page_number": 6
                }]
            }]
        });

        let resp = tx
            .transform_chat_response_with_citation_documents(
                &raw,
                &[
                    crate::standards::anthropic::streaming::AnthropicCitationDocument {
                        title: "Fallback Title".to_string(),
                        filename: Some("financial-report.pdf".to_string()),
                        media_type: "application/pdf".to_string(),
                    },
                ],
                &AnthropicParams::default(),
            )
            .expect("transform");

        let MessageContent::MultiModal(parts) = &resp.content else {
            panic!("expected multimodal content");
        };
        assert_eq!(parts.len(), 2);

        let crate::types::ContentPart::Source {
            id,
            source,
            provider_metadata,
        } = &parts[1]
        else {
            panic!("expected source part");
        };
        assert_eq!(id, "id-0");
        assert_eq!(
            source,
            &crate::types::SourcePart::Document {
                media_type: "application/pdf".to_string(),
                title: "Financial Report 2023".to_string(),
                filename: Some("financial-report.pdf".to_string()),
            }
        );
        assert_eq!(
            provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("anthropic"))
                .and_then(|metadata| metadata.get("startPageNumber")),
            Some(&serde_json::json!(5))
        );

        let meta = resp.anthropic_metadata().expect("anthropic metadata");
        let sources = meta.sources.expect("sources");
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].id, "id-0");
        assert_eq!(sources[0].source_type, "document");
        assert_eq!(sources[0].title.as_deref(), Some("Financial Report 2023"));
    }

    #[test]
    fn anthropic_chat_response_maps_tool_use_blocks_into_content_parts() {
        let tx = AnthropicResponseTransformer::default();
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-7-sonnet-latest",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": { "input_tokens": 1, "output_tokens": 2 },
            "content": [
                { "type": "tool_use", "id": "toolu_1", "name": "weather", "input": { "city": "Tokyo" } }
            ]
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        assert_eq!(resp.finish_reason, Some(FinishReason::ToolCalls));

        assert!(
            matches!(resp.content, MessageContent::MultiModal(_)),
            "expected multimodal content for tool calls"
        );
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);

        let info = calls[0].as_tool_call().expect("tool call info");
        assert_eq!(info.tool_call_id, "toolu_1");
        assert_eq!(info.tool_name, "weather");
        assert_eq!(info.arguments, &serde_json::json!({ "city": "Tokyo" }));
        assert_eq!(info.provider_executed.copied(), None);
    }

    #[test]
    fn reserved_json_tool_response_is_mapped_to_text_and_stop() {
        let tx = AnthropicResponseTransformer::default();
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-7-sonnet-latest",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": { "input_tokens": 1, "output_tokens": 2 },
            "content": [
                { "type": "tool_use", "id": "toolu_1", "name": "json", "input": { "value": "ok", "a": 1 } }
            ]
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        assert_eq!(resp.finish_reason, Some(FinishReason::Stop));
        assert_eq!(resp.content_text(), Some(r#"{"a":1,"value":"ok"}"#));
        assert!(
            resp.tool_calls().is_empty(),
            "reserved json tool should not surface as tool call"
        );
    }

    #[test]
    fn tool_result_message_is_serialized_as_anthropic_tool_result_block() {
        let tx = AnthropicRequestTransformer::default();

        let req = ChatRequest::builder()
            .model("claude-3-7-sonnet-latest")
            .messages(vec![
                crate::types::ChatMessage::tool_result_json(
                    "toolu_1",
                    "weather",
                    serde_json::json!({ "temperature": 18 }),
                )
                .build(),
            ])
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        let msgs = body["messages"].as_array().expect("messages array");
        assert_eq!(msgs.len(), 1);

        assert_eq!(msgs[0]["role"], "user");
        let content = msgs[0]["content"].as_array().expect("content array");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_1");
        assert_eq!(content[0]["is_error"], serde_json::json!(false));
        assert!(
            content[0]["content"].is_string(),
            "Anthropic tool_result content must be string or blocks"
        );
    }

    #[test]
    fn response_format_json_schema_uses_output_config_format_on_supported_models() {
        use crate::types::chat::ResponseFormat;

        let tx = AnthropicRequestTransformer::default();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let req = ChatRequest::builder()
            .model("claude-sonnet-4-5-latest")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()))
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body.pointer("/output_config/format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "schema": schema
            }))
        );
        assert!(
            body.get("tools").is_none(),
            "output_config.format path should not inject the reserved json tool"
        );
    }

    #[test]
    fn response_format_json_schema_falls_back_to_reserved_json_tool_on_unsupported_models() {
        use crate::types::chat::ResponseFormat;

        let tx = AnthropicRequestTransformer::default();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let req = ChatRequest::builder()
            .model("claude-3-7-sonnet-latest")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()))
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert!(
            body.get("output_format").is_none(),
            "unsupported models should not use output_format"
        );
        assert!(
            body.pointer("/output_config/format").is_none(),
            "unsupported models should not use output_config.format"
        );

        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .expect("expected tools array");
        assert!(tools.iter().any(|t| {
            t.get("name").and_then(|v| v.as_str()) == Some("json")
                && t.get("input_schema") == Some(&schema)
        }));

        assert_eq!(
            body.get("tool_choice"),
            Some(&serde_json::json!({
                "type": "any",
                "disable_parallel_tool_use": true
            }))
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
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({"type":"object","properties":{"a":{"type":"string"}}}),
            ))
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
        assert!(
            body.pointer("/output_config/format").is_none(),
            "expected json tool fallback, got output_config.format: {:?}",
            body.pointer("/output_config/format")
        );
        assert!(body.get("tools").is_some(), "expected json tool");
        assert_eq!(
            body.get("tool_choice"),
            Some(&serde_json::json!({"type":"any","disable_parallel_tool_use": true}))
        );
    }

    #[test]
    fn structured_output_auto_uses_output_config_format_even_with_tools() {
        let tx = AnthropicRequestTransformer::default();

        let req = ChatRequest::builder()
            .model("claude-sonnet-4-5")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![crate::types::Tool::function(
                "get-weather",
                "Get weather",
                serde_json::json!({"type":"object","properties":{"q":{"type":"string"}}}),
            )])
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({"type":"object","properties":{"a":{"type":"string"}}}),
            ))
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert!(body.get("tools").is_some(), "expected tools in body");
        assert!(
            body.pointer("/output_config/format").is_some(),
            "expected output_config.format when model supports it"
        );
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .expect("tools array");
        assert!(
            !tools
                .iter()
                .any(|t| t.get("name").and_then(|v| v.as_str()) == Some("json")),
            "did not expect json tool when using output_config.format"
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

    #[test]
    fn legacy_specific_thinking_adjusts_max_tokens_and_strips_sampling_settings() {
        let tx = AnthropicRequestTransformer::new(Some(AnthropicSpecificParams {
            thinking_config: Some(
                crate::standards::anthropic::thinking::ThinkingConfig::enabled(1024),
            ),
            ..Default::default()
        }));

        let mut req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        req.common_params.model = "claude-sonnet-4-5".to_string();
        req.common_params.max_tokens = Some(2000);
        req.common_params.temperature = Some(0.5);
        req.common_params.top_p = Some(0.7);
        req.common_params.top_k = Some(1.0);

        let body = tx.transform_chat(&req).expect("transform");

        assert_eq!(
            body.get("thinking"),
            Some(&serde_json::json!({
                "type": "enabled",
                "budget_tokens": 1024
            }))
        );
        assert_eq!(body.get("max_tokens"), Some(&serde_json::json!(3024)));
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("top_k").is_none());
    }

    #[test]
    fn request_transformer_accepts_custom_provider_key_and_overrides_canonical_options() {
        let tx = AnthropicRequestTransformer::default()
            .with_provider_options_key("my-custom-anthropic.messages");

        let req = ChatRequest::builder()
            .model("claude-sonnet-4-5")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![crate::types::Tool::function(
                "testFunction",
                "A test function",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .provider_option(
                "anthropic",
                serde_json::json!({
                    "disableParallelToolUse": false,
                    "structuredOutputMode": "jsonTool"
                }),
            )
            .provider_option(
                "my-custom-anthropic",
                serde_json::json!({
                    "disableParallelToolUse": true,
                    "structuredOutputMode": "outputFormat"
                }),
            )
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({"type":"object","properties":{"a":{"type":"string"}}}),
            ))
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body.get("tool_choice"),
            Some(&serde_json::json!({
                "type": "auto",
                "disable_parallel_tool_use": true
            }))
        );
        assert!(
            body.pointer("/output_config/format").is_some(),
            "custom provider key should override canonical structuredOutputMode"
        );
    }

    #[test]
    fn response_transformer_duplicates_provider_metadata_for_custom_key() {
        let tx = AnthropicResponseTransformer::default()
            .with_provider_metadata_key("my-custom-anthropic.messages");
        let raw = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-7-sonnet-latest",
            "stop_reason": "end_turn",
            "stop_sequence": "STOP",
            "usage": { "input_tokens": 1, "output_tokens": 2 },
            "content": [{ "type": "text", "text": "ok" }]
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let provider_metadata = resp.provider_metadata.as_ref().expect("provider metadata");
        assert!(provider_metadata.get("anthropic").is_some());
        assert!(provider_metadata.get("my-custom-anthropic").is_some());
        assert_eq!(
            resp.anthropic_metadata_with_key("my-custom-anthropic")
                .and_then(|meta| meta.stop_sequence),
            Some("STOP".to_string())
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
