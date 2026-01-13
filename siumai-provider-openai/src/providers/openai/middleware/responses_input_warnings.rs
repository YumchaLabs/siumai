//! OpenAI Responses input warning parity middleware (Vercel AI SDK aligned).

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, ContentPart, MessageContent, Warning};
use siumai_core::standards::tool_name_mapping::create_tool_name_mapping;

#[derive(Debug, Default)]
pub struct OpenAiResponsesInputWarningsMiddleware;

impl OpenAiResponsesInputWarningsMiddleware {
    pub const fn new() -> Self {
        Self
    }

    fn openai_provider_option_str(req: &ChatRequest, key: &str) -> Option<String> {
        let openai = req.provider_options_map.get_object("openai");
        let direct = openai
            .and_then(|m| m.get(key))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        if direct.is_some() {
            return direct;
        }

        let nested = openai
            .and_then(|m| m.get("responsesApi").or_else(|| m.get("responses_api")))
            .and_then(|v| v.as_object())
            .and_then(|m| m.get(key))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        nested
    }

    fn openai_provider_option_previous_response_id(req: &ChatRequest) -> Option<String> {
        // Vercel providerOptions key is `previousResponseId`, but accept snake_case as well.
        Self::openai_provider_option_str(req, "previousResponseId")
            .or_else(|| Self::openai_provider_option_str(req, "previous_response_id"))
    }

    fn vercel_reasoning_part_json(part: &ContentPart) -> Option<String> {
        let ContentPart::Reasoning {
            text,
            provider_metadata,
        } = part
        else {
            return None;
        };

        let openai = provider_metadata
            .as_ref()
            .and_then(|m| m.get("openai"))
            .and_then(|v| v.as_object());

        let item_id = openai
            .and_then(|m| m.get("itemId").or_else(|| m.get("item_id")))
            .and_then(|v| v.as_str());
        let encrypted = openai.and_then(|m| {
            m.get("reasoningEncryptedContent")
                .or_else(|| m.get("reasoning_encrypted_content"))
        });
        let reasoning = openai.and_then(|m| m.get("reasoning"));

        // Emit a Vercel-shaped JSON.stringify snapshot with stable key order:
        // {"type":"reasoning","text":"...","providerOptions":{"openai":{...}}}
        let text_json = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string());

        if item_id.is_none() && encrypted.is_none() && reasoning.is_none() {
            return Some(format!("{{\"type\":\"reasoning\",\"text\":{text_json}}}"));
        }

        let mut openai_fields: Vec<String> = Vec::new();
        if let Some(id) = item_id {
            let id_json = serde_json::to_string(id).unwrap_or_else(|_| "\"\"".to_string());
            openai_fields.push(format!("\"itemId\":{id_json}"));
        }
        if let Some(enc) = encrypted {
            let enc_json = serde_json::to_string(enc).unwrap_or_else(|_| "null".to_string());
            openai_fields.push(format!("\"reasoningEncryptedContent\":{enc_json}"));
        }
        if let Some(r) = reasoning {
            let r_json = serde_json::to_string(r).unwrap_or_else(|_| "null".to_string());
            openai_fields.push(format!("\"reasoning\":{r_json}"));
        }

        Some(format!(
            "{{\"type\":\"reasoning\",\"text\":{text_json},\"providerOptions\":{{\"openai\":{{{}}}}}}}",
            openai_fields.join(",")
        ))
    }

    fn store_enabled(req: &ChatRequest) -> bool {
        let openai = req.provider_options_map.get_object("openai");
        let store = openai
            .and_then(|m| m.get("store"))
            .and_then(|v| v.as_bool())
            .or_else(|| {
                openai
                    .and_then(|m| m.get("responsesApi").or_else(|| m.get("responses_api")))
                    .and_then(|v| v.as_object())
                    .and_then(|m| m.get("store"))
                    .and_then(|v| v.as_bool())
            });

        store != Some(false)
    }

    fn compute_warnings(req: &ChatRequest) -> Vec<Warning> {
        let store_enabled = Self::store_enabled(req);

        // Vercel parity: unsupported standardized settings for the Responses API.
        // (The request transformer intentionally drops these fields.)
        let mut warnings: Vec<Warning> = Vec::new();
        if req.common_params.top_k.is_some() {
            warnings.push(Warning::unsupported_setting("topK", None::<String>));
        }
        if req.common_params.seed.is_some() {
            warnings.push(Warning::unsupported_setting("seed", None::<String>));
        }
        if req.common_params.presence_penalty.is_some() {
            warnings.push(Warning::unsupported_setting(
                "presencePenalty",
                None::<String>,
            ));
        }
        if req.common_params.frequency_penalty.is_some() {
            warnings.push(Warning::unsupported_setting(
                "frequencyPenalty",
                None::<String>,
            ));
        }
        if req.common_params.stop_sequences.is_some() {
            warnings.push(Warning::unsupported_setting(
                "stopSequences",
                None::<String>,
            ));
        }

        // Vercel parity: `conversation` and `previousResponseId` cannot be used together.
        let has_conversation = Self::openai_provider_option_str(req, "conversation").is_some();
        let has_prev_id = Self::openai_provider_option_previous_response_id(req).is_some();
        if has_conversation && has_prev_id {
            warnings.push(Warning::unsupported_setting(
                "conversation",
                Some("conversation and previousResponseId cannot be used together"),
            ));
        }

        let tool_name_mapping = req.tools.as_deref().map(|tools| {
            create_tool_name_mapping(tools, siumai_core::tools::openai::PROVIDER_TOOL_NAMES)
        });
        let tool_name_mapping = tool_name_mapping.unwrap_or_default();

        // Vercel parity: warnings for `store=false` + provider-executed web_search.
        let mut has_web_search_results = false;

        for message in &req.messages {
            let MessageContent::MultiModal(parts) = &message.content else {
                continue;
            };

            for part in parts {
                match part {
                    ContentPart::ToolCall {
                        tool_name,
                        provider_executed: Some(true),
                        ..
                    } => {
                        if tool_name_mapping.to_provider_tool_name(tool_name) == "web_search" {
                            has_web_search_results = true;
                        }
                    }
                    ContentPart::ToolResult { tool_name, .. } => {
                        if tool_name_mapping.to_provider_tool_name(tool_name) == "web_search" {
                            has_web_search_results = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        if !store_enabled && has_web_search_results {
            warnings.push(Warning::other(
                "Results for OpenAI tool web_search are not sent to the API when store is false",
            ));
        }

        // Vercel parity: reasoning warnings.
        let mut seen_reasoning_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for message in &req.messages {
            let MessageContent::MultiModal(parts) = &message.content else {
                continue;
            };

            for part in parts {
                let ContentPart::Reasoning { text, .. } = part else {
                    continue;
                };

                let openai = match part {
                    ContentPart::Reasoning {
                        provider_metadata, ..
                    } => provider_metadata
                        .as_ref()
                        .and_then(|m| m.get("openai"))
                        .and_then(|v| v.as_object()),
                    _ => None,
                };

                let reasoning_id = openai
                    .and_then(|m| m.get("itemId").or_else(|| m.get("item_id")))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let Some(reasoning_id) = reasoning_id else {
                    let snapshot = Self::vercel_reasoning_part_json(part)
                        .unwrap_or_else(|| "{\"type\":\"reasoning\"}".to_string());
                    warnings.push(Warning::other(format!(
                        "Non-OpenAI reasoning parts are not supported. Skipping reasoning part: {snapshot}.",
                    )));
                    continue;
                };

                if !store_enabled && text.is_empty() && seen_reasoning_ids.contains(&reasoning_id) {
                    let snapshot = Self::vercel_reasoning_part_json(part)
                        .unwrap_or_else(|| "{\"type\":\"reasoning\"}".to_string());
                    warnings.push(Warning::other(format!(
                        "Cannot append empty reasoning part to existing reasoning sequence. Skipping reasoning part: {snapshot}.",
                    )));
                    continue;
                }

                seen_reasoning_ids.insert(reasoning_id);
            }
        }

        warnings
    }

    fn merge_warnings(mut resp: ChatResponse, additional: Vec<Warning>) -> ChatResponse {
        if additional.is_empty() {
            return resp;
        }

        match resp.warnings.as_mut() {
            Some(existing) => existing.extend(additional),
            None => resp.warnings = Some(additional),
        }

        resp
    }
}

impl LanguageModelMiddleware for OpenAiResponsesInputWarningsMiddleware {
    fn post_generate(
        &self,
        req: &ChatRequest,
        resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        Ok(Self::merge_warnings(resp, Self::compute_warnings(req)))
    }

    fn on_stream_event(
        &self,
        req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        match ev {
            ChatStreamEvent::StreamEnd { response } => {
                let response = Self::merge_warnings(response, Self::compute_warnings(req));
                Ok(vec![ChatStreamEvent::StreamEnd { response }])
            }
            other => Ok(vec![other]),
        }
    }
}
