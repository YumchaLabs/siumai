//! OpenAI-compatible tool warning parity middleware (Vercel AI SDK aligned).

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Tool, Warning};
use std::collections::BTreeSet;

#[derive(Debug, Default)]
pub struct OpenAiCompatibleToolWarningsMiddleware {
    allowlist: BTreeSet<String>,
}

impl OpenAiCompatibleToolWarningsMiddleware {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_allowlist<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowlist = ids.into_iter().map(Into::into).collect();
        self
    }

    fn compute_warnings(&self, req: &ChatRequest) -> Vec<Warning> {
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
                Tool::ProviderDefined(t) if !self.allowlist.contains(&t.id) => Some(
                    Warning::unsupported(format!("provider-defined tool {}", t.id), None::<String>),
                ),
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
        Ok(Self::merge_warnings(resp, self.compute_warnings(req)))
    }

    fn on_stream_event(
        &self,
        req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        match ev {
            ChatStreamEvent::StreamEnd { response } => {
                let response = Self::merge_warnings(response, self.compute_warnings(req));
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
            Warning::Unsupported { feature, .. } if feature == "provider-defined tool openai.web_search"
        )));
        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::Unsupported { feature, .. } if feature == "provider-defined tool google.google_search"
        )));
    }

    #[test]
    fn allowlist_skips_expected_provider_defined_tool() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_tools(vec![
            Tool::provider_defined("groq.browser_search", "browser_search"),
            Tool::provider_defined("openai.web_search", "web_search"),
        ]);

        let mw =
            OpenAiCompatibleToolWarningsMiddleware::new().with_allowlist(["groq.browser_search"]);
        let out = mw.post_generate(&req, dummy_resp()).unwrap();
        let warnings = out.warnings.unwrap_or_default();

        assert!(!warnings.iter().any(|w| matches!(
            w,
            Warning::Unsupported { feature, .. } if feature == "provider-defined tool groq.browser_search"
        )));
        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::Unsupported { feature, .. } if feature == "provider-defined tool openai.web_search"
        )));
    }
}
