//! OpenAI-compatible tool warning parity middleware (Vercel AI SDK aligned).

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Tool, Warning};

#[derive(Debug, Default)]
pub struct OpenAiCompatibleToolWarningsMiddleware;

impl OpenAiCompatibleToolWarningsMiddleware {
    pub const fn new() -> Self {
        Self
    }

    fn compute_warnings(req: &ChatRequest) -> Vec<Warning> {
        let Some(tools) = req.tools.as_deref() else {
            return Vec::new();
        };
        if tools.is_empty() {
            return Vec::new();
        }

        // Vercel-aligned: OpenAI-compatible Chat Completions only support function tools.
        tools
            .iter()
            .filter_map(|t| match t {
                Tool::ProviderDefined(t) => {
                    Some(Warning::unsupported_tool(t.id.clone(), None::<String>))
                }
                _ => None,
            })
            .collect()
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

impl LanguageModelMiddleware for OpenAiCompatibleToolWarningsMiddleware {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::middleware::LanguageModelMiddleware;
    use crate::types::{ChatMessage, MessageContent};

    fn dummy_resp() -> ChatResponse {
        ChatResponse::new(MessageContent::Text("ok".to_string()))
    }

    #[test]
    fn warns_on_any_provider_defined_tool() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_tools(vec![
            Tool::provider_defined("openai.web_search", "web_search"),
            Tool::provider_defined("google.google_search", "google_search"),
        ]);

        let mw = OpenAiCompatibleToolWarningsMiddleware::new();
        let out = mw.post_generate(&req, dummy_resp()).unwrap();
        let warnings = out.warnings.unwrap_or_default();

        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, .. } if tool_name == "openai.web_search"
        )));
        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, .. } if tool_name == "google.google_search"
        )));
    }
}
