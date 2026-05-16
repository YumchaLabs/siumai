//! Vercel-aligned warning middleware for `systemMessageMode`.
//!
//! Vercel AI SDK emits a warning when `systemMessageMode: "remove"` is used and
//! the prompt contains system messages. Protocol converters apply the removal
//! behavior, but warnings are surfaced at the middleware layer
//! (post-generate / stream-end).

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, MessageRole, Warning};

#[derive(Debug, Clone)]
pub struct SystemMessageModeWarningMiddleware {
    provider_option_namespace: String,
}

impl SystemMessageModeWarningMiddleware {
    pub fn new(provider_option_namespace: impl Into<String>) -> Self {
        Self {
            provider_option_namespace: provider_option_namespace.into(),
        }
    }

    fn system_message_mode<'a>(&self, req: &'a ChatRequest) -> Option<&'a str> {
        req.provider_option(&self.provider_option_namespace)
            .and_then(|v| v.as_object())
            .and_then(|obj| {
                obj.get("systemMessageMode")
                    .or_else(|| obj.get("system_message_mode"))
                    .and_then(|v| v.as_str())
            })
    }

    fn has_system_messages(req: &ChatRequest) -> bool {
        req.messages
            .iter()
            .any(|m| matches!(m.role, MessageRole::System))
    }

    fn compute_warnings(&self, req: &ChatRequest) -> Vec<Warning> {
        if self.system_message_mode(req) == Some("remove") && Self::has_system_messages(req) {
            vec![Warning::compatibility(
                "systemMessageMode=remove",
                Some("system messages are removed for this model"),
            )]
        } else {
            Vec::new()
        }
    }

    fn merge_warnings(mut resp: ChatResponse, warnings: Vec<Warning>) -> ChatResponse {
        if warnings.is_empty() {
            return resp;
        }

        let mut merged = resp.warnings.take().unwrap_or_default();
        merged.extend(warnings);
        resp.warnings = Some(merged);
        resp
    }
}

impl LanguageModelMiddleware for SystemMessageModeWarningMiddleware {
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
    use crate::types::{ChatMessage, MessageContent};

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn system_message_mode_warning_source_stays_provider_agnostic() {
        let source = include_str!("system_message_mode_warning.rs");
        let production_source = source_section(
            source,
            "pub struct SystemMessageModeWarningMiddleware",
            "#[cfg(test)]",
        );

        let disallowed = [
            "provider_option(\"".to_string(),
            format!("\"{}\"", ["op", "enai"].concat()),
            format!("\"{}\"", ["az", "ure"].concat()),
            format!("\"{}\"", ["an", "thropic"].concat()),
            format!("\"{}\"", ["ge", "mini"].concat()),
            ["Open", "AI"].concat(),
            ["Az", "ure"].concat(),
            ["An", "thropic"].concat(),
            ["Ge", "mini"].concat(),
        ];

        for disallowed in disallowed {
            assert!(
                !production_source.contains(&disallowed),
                "core system-message warning middleware must use injected provider option namespaces"
            );
        }
    }

    #[test]
    fn post_generate_emits_compatibility_warning_for_system_message_removal() {
        let req = ChatRequest::new(vec![
            ChatMessage::system("sys").build(),
            ChatMessage::user("hi").build(),
        ])
        .with_provider_option(
            "provider-a",
            serde_json::json!({ "systemMessageMode": "remove" }),
        );

        let resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
        let resp = SystemMessageModeWarningMiddleware::new("provider-a")
            .post_generate(&req, resp)
            .expect("post-generate");

        assert_eq!(
            resp.warnings,
            Some(vec![Warning::Compatibility {
                feature: "systemMessageMode=remove".to_string(),
                details: Some("system messages are removed for this model".to_string()),
            }])
        );
    }

    #[test]
    fn post_generate_uses_only_the_configured_provider_option_namespace() {
        let req = ChatRequest::new(vec![
            ChatMessage::system("sys").build(),
            ChatMessage::user("hi").build(),
        ])
        .with_provider_option(
            "provider-a",
            serde_json::json!({ "systemMessageMode": "remove" }),
        );

        let resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
        let resp = SystemMessageModeWarningMiddleware::new("provider-b")
            .post_generate(&req, resp)
            .expect("post-generate");

        assert_eq!(resp.warnings, None);
    }
}
