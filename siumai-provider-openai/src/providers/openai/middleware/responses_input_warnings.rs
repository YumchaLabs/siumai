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
        if Self::store_enabled(req) {
            return Vec::new();
        }

        let tool_name_mapping = req.tools.as_deref().map(|tools| {
            create_tool_name_mapping(
                tools,
                &[
                    ("openai.web_search", "web_search"),
                    ("openai.web_search_preview", "web_search_preview"),
                ],
            )
        });
        let tool_name_mapping = tool_name_mapping.unwrap_or_default();

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

        if !has_web_search_results {
            return Vec::new();
        }

        vec![Warning::other(
            "Results for OpenAI tool web_search are not sent to the API when store is false",
        )]
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
