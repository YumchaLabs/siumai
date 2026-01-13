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
        let mut warnings: Vec<Warning> = Vec::new();

        // --------------------------------------------------------------------
        // Settings / thinking-mode warnings (Vercel-aligned)
        // --------------------------------------------------------------------
        #[derive(Debug, Clone, Copy)]
        struct ThinkingState {
            enabled: bool,
            budget_tokens: Option<u32>,
        }

        fn thinking_state(req: &ChatRequest) -> ThinkingState {
            let Some(v) = req.provider_options_map.get("anthropic") else {
                return ThinkingState {
                    enabled: false,
                    budget_tokens: None,
                };
            };
            let Some(obj) = v.as_object() else {
                return ThinkingState {
                    enabled: false,
                    budget_tokens: None,
                };
            };

            // Vercel-style: `thinking: { type: "enabled", budgetTokens? }`
            if let Some(t) = obj.get("thinking").and_then(|v| v.as_object()) {
                let enabled = t
                    .get("type")
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| s == "enabled");
                let budget = t
                    .get("budgetTokens")
                    .or_else(|| t.get("budget_tokens"))
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok());
                return ThinkingState {
                    enabled,
                    budget_tokens: budget,
                };
            }

            // Legacy typed options: `thinkingMode: { enabled, thinkingBudget? }`
            if let Some(t) = obj
                .get("thinkingMode")
                .or_else(|| obj.get("thinking_mode"))
                .and_then(|v| v.as_object())
            {
                let enabled = t.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
                let budget = t
                    .get("thinkingBudget")
                    .or_else(|| t.get("thinking_budget"))
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok());
                return ThinkingState {
                    enabled,
                    budget_tokens: budget,
                };
            }

            ThinkingState {
                enabled: false,
                budget_tokens: None,
            }
        }

        let thinking = thinking_state(req);
        let thinking_budget = if thinking.enabled {
            thinking.budget_tokens.or(Some(1024))
        } else {
            None
        };

        // Vercel-aligned: unsupported standardized settings (ignored by Anthropic).
        if req.common_params.frequency_penalty.is_some() {
            warnings.push(Warning::unsupported_setting(
                "frequencyPenalty",
                None::<String>,
            ));
        }
        if req.common_params.presence_penalty.is_some() {
            warnings.push(Warning::unsupported_setting(
                "presencePenalty",
                None::<String>,
            ));
        }
        if req.common_params.seed.is_some() {
            warnings.push(Warning::unsupported_setting("seed", None::<String>));
        }

        // Vercel-aligned: clamp temperature to [0, 1] and warn.
        if let Some(t) = req.common_params.temperature {
            if t > 1.0 {
                warnings.push(Warning::unsupported_setting(
                    "temperature",
                    Some(format!(
                        "{t} exceeds anthropic maximum of 1.0. clamped to 1.0"
                    )),
                ));
            } else if t < 0.0 {
                warnings.push(Warning::unsupported_setting(
                    "temperature",
                    Some(format!("{t} is below anthropic minimum of 0. clamped to 0")),
                ));
            }
        }

        if thinking.enabled {
            if thinking.budget_tokens.is_none() {
                warnings.push(Warning::unsupported_setting(
                    "extended thinking",
                    Some(
                        "thinking budget is required when thinking is enabled. using default budget of 1024 tokens.",
                    ),
                ));
            }

            if req.common_params.temperature.is_some() {
                warnings.push(Warning::unsupported_setting(
                    "temperature",
                    Some("temperature is not supported when thinking is enabled"),
                ));
            }
            if req.common_params.top_k.is_some() {
                warnings.push(Warning::unsupported_setting(
                    "topK",
                    Some("topK is not supported when thinking is enabled"),
                ));
            }
            if req.common_params.top_p.is_some() {
                warnings.push(Warning::unsupported_setting(
                    "topP",
                    Some("topP is not supported when thinking is enabled"),
                ));
            }
        } else if req.common_params.temperature.is_some() && req.common_params.top_p.is_some() {
            warnings.push(Warning::unsupported_setting(
                "topP",
                Some("topP is not supported when temperature is set. topP is ignored."),
            ));
        }

        // Vercel-aligned: cap max_tokens for known models and warn only when maxOutputTokens is provided.
        if let Some(max_out) =
            crate::providers::anthropic::model_constants::try_get_max_output_tokens(
                req.common_params.model.as_str(),
            )
            && let Some(max_tokens) = req.common_params.max_tokens
        {
            let effective = max_tokens.saturating_add(thinking_budget.unwrap_or(0));
            if effective > max_out {
                warnings.push(Warning::unsupported_setting(
                    "maxOutputTokens",
                    Some(format!(
                        "{effective} (maxOutputTokens + thinkingBudget) is greater than {} {max_out} max output tokens. The max output tokens have been limited to {max_out}.",
                        req.common_params.model
                    )),
                ));
            }
        }

        // --------------------------------------------------------------------
        // Tool warnings
        // --------------------------------------------------------------------
        let Some(tools) = req.tools.as_deref() else {
            return warnings;
        };
        if tools.is_empty() {
            return warnings;
        }

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
