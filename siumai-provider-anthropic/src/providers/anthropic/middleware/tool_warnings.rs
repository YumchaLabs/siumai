//! Anthropic tool warning parity middleware (Vercel AI SDK aligned).

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Tool, Warning};

#[derive(Debug, Default)]
pub struct AnthropicToolWarningsMiddleware;

impl AnthropicToolWarningsMiddleware {
    pub const fn new() -> Self {
        Self
    }

    fn is_supported_provider_defined_tool_id(id: &str) -> bool {
        matches!(
            id,
            crate::tools::anthropic::WEB_SEARCH_20250305_ID
                | crate::tools::anthropic::WEB_FETCH_20250910_ID
                | crate::tools::anthropic::COMPUTER_20250124_ID
                | crate::tools::anthropic::COMPUTER_20241022_ID
                | crate::tools::anthropic::TEXT_EDITOR_20250124_ID
                | crate::tools::anthropic::TEXT_EDITOR_20241022_ID
                | crate::tools::anthropic::TEXT_EDITOR_20250429_ID
                | crate::tools::anthropic::TEXT_EDITOR_20250728_ID
                | crate::tools::anthropic::BASH_20241022_ID
                | crate::tools::anthropic::BASH_20250124_ID
                | crate::tools::anthropic::TOOL_SEARCH_REGEX_20251119_ID
                | crate::tools::anthropic::TOOL_SEARCH_BM25_20251119_ID
                | crate::tools::anthropic::CODE_EXECUTION_20250522_ID
                | crate::tools::anthropic::CODE_EXECUTION_20250825_ID
                | crate::tools::anthropic::MEMORY_20250818_ID
        )
    }

    fn cache_control_breakpoint_count(req: &ChatRequest) -> usize {
        req.tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter_map(|t| match t {
                Tool::Function { function } => function
                    .provider_options_map
                    .get("anthropic")
                    .and_then(|v| v.as_object())
                    .and_then(|o| {
                        o.get("cacheControl")
                            .or_else(|| o.get("cache_control"))
                            .and_then(|v| v.as_object())
                    })
                    .map(|_| ()),
                _ => None,
            })
            .count()
    }

    fn compute_warnings(req: &ChatRequest) -> Vec<Warning> {
        let Some(tools) = req.tools.as_deref() else {
            return Vec::new();
        };
        if tools.is_empty() {
            return Vec::new();
        }

        let mut warnings: Vec<Warning> = Vec::new();

        // Vercel-aligned: warn about provider-defined tools that are not supported by Anthropic.
        for tool in tools {
            let Tool::ProviderDefined(t) = tool else {
                continue;
            };

            if t.provider() != Some("anthropic") {
                warnings.push(Warning::unsupported_tool(t.id.clone(), None::<String>));
                continue;
            }

            if !Self::is_supported_provider_defined_tool_id(&t.id) {
                warnings.push(Warning::unsupported_tool(t.id.clone(), None::<String>));
            }
        }

        // Vercel-aligned: at most 4 cache breakpoints.
        let count = Self::cache_control_breakpoint_count(req);
        if count > 4 {
            warnings.push(Warning::unsupported_setting(
                "cacheControl breakpoint limit",
                Some(format!(
                    "Maximum 4 cache breakpoints exceeded (found {count}). This breakpoint will be ignored."
                )),
            ));
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

impl LanguageModelMiddleware for AnthropicToolWarningsMiddleware {
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
    fn warns_on_unsupported_provider_defined_tools() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_tools(vec![
            Tool::provider_defined("unsupported.tool", "unsupported_tool"),
            Tool::provider_defined("anthropic.unknown_tool", "unknown_tool"),
            crate::tools::anthropic::web_search_20250305(),
        ]);

        let mw = AnthropicToolWarningsMiddleware::new();
        let out = mw.post_generate(&req, dummy_resp()).unwrap();
        let warnings = out.warnings.unwrap_or_default();

        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, .. } if tool_name == "unsupported.tool"
        )));
        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, .. } if tool_name == "anthropic.unknown_tool"
        )));
    }

    #[test]
    fn warns_when_cache_control_breakpoints_exceed_limit() {
        let mut tools: Vec<Tool> = Vec::new();
        for i in 0..5 {
            let mut t = Tool::function(
                format!("tool{i}"),
                format!("tool {i}"),
                serde_json::json!({}),
            );
            if let Tool::Function { function } = &mut t {
                function.provider_options_map.insert(
                    "anthropic",
                    serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
                );
            }
            tools.push(t);
        }

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_tools(tools);

        let mw = AnthropicToolWarningsMiddleware::new();
        let out = mw.post_generate(&req, dummy_resp()).unwrap();
        let warnings = out.warnings.unwrap_or_default();

        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedSetting { setting, details: Some(d) }
                if setting == "cacheControl breakpoint limit"
                    && d == "Maximum 4 cache breakpoints exceeded (found 5). This breakpoint will be ignored."
        )));
    }
}
