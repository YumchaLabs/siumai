use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::provider_options::anthropic::{
    AnthropicOptions, AnthropicResponseFormat, ThinkingModeConfig,
};
use crate::standards::anthropic::chat::AnthropicChatStandard;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use crate::types::Tool;
use reqwest::header::HeaderMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Anthropic ProviderSpec implementation
///
/// This spec uses the Anthropic standard from the standards layer,
/// with additional support for Anthropic-specific features like Prompt Caching and Thinking Mode.
#[derive(Clone, Default)]
pub struct AnthropicSpec {
    /// Standard Anthropic Chat implementation
    chat_standard: AnthropicChatStandard,
}

impl AnthropicSpec {
    pub fn new() -> Self {
        Self {
            chat_standard: AnthropicChatStandard::new(),
        }
    }
}

impl ProviderSpec for AnthropicSpec {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_custom_feature("prompt_caching", true)
            .with_custom_feature("thinking_mode", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("Anthropic API key not provided".into()))?;
        crate::standards::anthropic::utils::build_headers(api_key, &ctx.http_extra_headers)
    }

    fn merge_request_headers(
        &self,
        mut base: HeaderMap,
        extra: &HashMap<String, String>,
    ) -> HeaderMap {
        fn merge_comma_separated_tokens(a: &str, b: &str) -> String {
            let mut seen: HashSet<String> = HashSet::new();
            let mut out: Vec<String> = Vec::new();

            for raw in a.split(',').chain(b.split(',')) {
                let token = raw.trim();
                if token.is_empty() {
                    continue;
                }
                if seen.insert(token.to_string()) {
                    out.push(token.to_string());
                }
            }

            out.join(",")
        }

        for (k, v) in extra {
            // Anthropic beta features are additive; merge values instead of overriding.
            if k.eq_ignore_ascii_case("anthropic-beta") {
                let existing = base
                    .get("anthropic-beta")
                    .and_then(|hv| hv.to_str().ok())
                    .unwrap_or("");
                let merged = merge_comma_separated_tokens(existing, v);
                if let (Ok(name), Ok(val)) = (
                    reqwest::header::HeaderName::from_bytes(b"anthropic-beta"),
                    reqwest::header::HeaderValue::from_str(&merged),
                ) {
                    base.insert(name, val);
                }
                continue;
            }

            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                reqwest::header::HeaderValue::from_str(v),
            ) {
                base.insert(name, val);
            }
        }

        base
    }

    fn chat_request_headers(
        &self,
        _stream: bool,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> HashMap<String, String> {
        let mut out: HashMap<String, String> = HashMap::new();

        fn required_beta_features(req: &ChatRequest) -> Vec<&'static str> {
            let mut out: Vec<&'static str> = Vec::new();

            let model = req.common_params.model.as_str();
            let supports_structured_outputs = model.starts_with("claude-sonnet-4-5")
                || model.starts_with("claude-opus-4-5")
                || model.starts_with("claude-haiku-4-5");

            // Provider-hosted tools -> required betas.
            if let Some(tools) = &req.tools {
                for tool in tools {
                    let Tool::ProviderDefined(t) = tool else {
                        continue;
                    };
                    if t.provider() != Some("anthropic") {
                        continue;
                    }
                    match t.tool_type() {
                        Some("web_fetch_20250910") => out.push("web-fetch-2025-09-10"),
                        Some("code_execution_20250522") => out.push("code-execution-2025-05-22"),
                        Some("code_execution_20250825") => out.push("code-execution-2025-08-25"),
                        Some("computer_20241022")
                        | Some("text_editor_20241022")
                        | Some("bash_20241022") => out.push("computer-use-2024-10-22"),
                        Some("computer_20250124")
                        | Some("text_editor_20250124")
                        | Some("text_editor_20250429")
                        | Some("bash_20250124") => out.push("computer-use-2025-01-24"),
                        Some("tool_search_regex_20251119") | Some("tool_search_bm25_20251119") => {
                            out.push("advanced-tool-use-2025-11-20")
                        }
                        Some("memory_20250818") => out.push("context-management-2025-06-27"),
                        _ => {}
                    }
                }
            }

            // Structured outputs beta (Vercel-aligned):
            // - enabled for supported models when using request-level JSON format, or
            // - enabled for supported models when any function tools are present.
            if supports_structured_outputs
                && (matches!(
                    req.response_format,
                    Some(crate::types::chat::ResponseFormat::Json { .. })
                ) || req
                    .tools
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .any(|t| matches!(t, Tool::Function { .. })))
            {
                out.push("structured-outputs-2025-11-13");
            }

            // Advanced tool use beta is required for tool input examples and allowed_callers.
            if let Some(tools) = req.tools.as_deref() {
                for tool in tools {
                    let Tool::Function { function } = tool else {
                        continue;
                    };
                    if function
                        .input_examples
                        .as_ref()
                        .is_some_and(|arr| !arr.is_empty())
                    {
                        out.push("advanced-tool-use-2025-11-20");
                        break;
                    }

                    let has_allowed_callers = function
                        .provider_options_map
                        .get("anthropic")
                        .and_then(|v| v.as_object())
                        .and_then(|o| {
                            o.get("allowedCallers")
                                .or_else(|| o.get("allowed_callers"))
                                .and_then(|v| v.as_array())
                        })
                        .is_some_and(|arr| !arr.is_empty());
                    if has_allowed_callers {
                        out.push("advanced-tool-use-2025-11-20");
                        break;
                    }
                }
            }

            // PDF documents -> required beta (Vercel-aligned).
            let uses_pdf = req.messages.iter().any(|m| match &m.content {
                crate::types::MessageContent::MultiModal(parts) => parts.iter().any(|p| {
                    matches!(
                        p,
                        crate::types::ContentPart::File { media_type, .. }
                            if media_type == "application/pdf"
                    )
                }),
                _ => false,
            });
            if uses_pdf {
                out.push("pdfs-2024-09-25");
            }

            out
        }

        fn has_agent_skills(req: &ChatRequest) -> bool {
            let Some(v) = req.provider_options_map.get("anthropic") else {
                return false;
            };
            let Some(obj) = v.as_object() else {
                return false;
            };
            let Some(container) = obj.get("container").and_then(|v| v.as_object()) else {
                return false;
            };
            let Some(skills) = container.get("skills").and_then(|v| v.as_array()) else {
                return false;
            };
            !skills.is_empty()
        }

        fn has_context_management(req: &ChatRequest) -> bool {
            let Some(v) = req.provider_options_map.get("anthropic") else {
                return false;
            };
            let Some(obj) = v.as_object() else {
                return false;
            };

            if obj.get("contextManagement").is_some() {
                return true;
            }
            obj.get("context_management").is_some()
        }

        fn has_effort(req: &ChatRequest) -> bool {
            let Some(v) = req.provider_options_map.get("anthropic") else {
                return false;
            };
            let Some(obj) = v.as_object() else {
                return false;
            };
            obj.get("effort").is_some()
        }

        fn fine_grained_tool_streaming_enabled(req: &ChatRequest) -> bool {
            if !req.stream {
                return false;
            }

            let Some(v) = req.provider_options_map.get("anthropic") else {
                return true;
            };
            let Some(obj) = v.as_object() else {
                return true;
            };

            if let Some(b) = obj.get("toolStreaming").and_then(|v| v.as_bool()) {
                return b;
            }
            if let Some(b) = obj.get("tool_streaming").and_then(|v| v.as_bool()) {
                return b;
            }

            true
        }

        let mut tokens: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        let mut push_token = |t: &str| {
            if seen.insert(t.to_string()) {
                tokens.push(t.to_string());
            }
        };

        if self
            .anthropic_mcp_servers_from_provider_options_map(req)
            .is_some()
        {
            push_token("mcp-client-2025-04-04");
        }

        for t in required_beta_features(req) {
            push_token(t);
        }

        if has_context_management(req) {
            push_token("context-management-2025-06-27");
        }

        // Vercel-aligned: only when streaming, enable fine-grained tool streaming by default.
        if fine_grained_tool_streaming_enabled(req) {
            push_token("fine-grained-tool-streaming-2025-05-14");
        }

        if has_effort(req) {
            push_token("effort-2025-11-24");
        }

        if has_agent_skills(req) {
            // Vercel-aligned: skill containers require the code execution runtime beta even if the
            // code execution tool is missing (a warning is emitted by the Anthropic client middleware).
            push_token("code-execution-2025-08-25");
            push_token("skills-2025-10-02");
            push_token("files-api-2025-04-14");
        }

        if !tokens.is_empty() {
            out.insert("anthropic-beta".to_string(), tokens.join(","));
        }

        out
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        crate::standards::anthropic::errors::classify_anthropic_http_error(
            self.id(),
            status,
            body_text,
        )
    }

    fn chat_url(&self, _stream: bool, _req: &ChatRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            format!("{base}/messages")
        } else {
            format!("{base}/v1/messages")
        }
    }

    fn models_url(&self, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            format!("{base}/models")
        } else {
            format!("{base}/v1/models")
        }
    }

    fn model_url(&self, model_id: &str, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            format!("{base}/models/{model_id}")
        } else {
            format!("{base}/v1/models/{model_id}")
        }
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        // Use standard Anthropic Messages API from standards layer
        let spec = self.chat_standard.create_spec("anthropic");
        spec.choose_chat_transformers(req, ctx)
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // Handle Anthropic-specific options (thinking_mode, response_format, etc.).
        let options = self.anthropic_options_from_provider_options_map(req);

        let thinking_mode: Option<ThinkingModeConfig> =
            options.as_ref().and_then(|o| o.thinking_mode.clone());
        let response_format: Option<AnthropicResponseFormat> =
            options.as_ref().and_then(|o| o.response_format.clone());
        let mcp_servers: Option<Vec<serde_json::Value>> =
            self.anthropic_mcp_servers_from_provider_options_map(req);
        let container = options.as_ref().and_then(|o| o.container.clone());
        let context_management = options.as_ref().and_then(|o| o.context_management.clone());
        let effort = options.as_ref().and_then(|o| o.effort);

        // Vercel-aligned: cap max_tokens for known models (warnings handled by middleware).
        let model_id = req.common_params.model.clone();
        let max_output_tokens =
            super::model_constants::try_get_max_output_tokens(model_id.as_str());
        let needs_max_tokens_cap = max_output_tokens.is_some()
            && req
                .common_params
                .max_tokens
                .is_some_and(|mt| mt > max_output_tokens.unwrap_or(mt));

        // If neither thinking nor response format configured, nothing to inject
        if thinking_mode.is_none()
            && response_format.is_none()
            && mcp_servers.is_none()
            && container.is_none()
            && context_management.is_none()
            && effort.is_none()
            && !needs_max_tokens_cap
        {
            return None;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();

            // Inject thinking mode configuration (Vercel-aligned).
            if let Some(ref thinking) = thinking_mode
                && thinking.enabled
            {
                let budget = thinking.thinking_budget.unwrap_or(1024);

                out["thinking"] = serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget
                });

                // Vercel-aligned: temperature/top_p/top_k are not supported when thinking is enabled.
                if let Some(obj) = out.as_object_mut() {
                    obj.remove("temperature");
                    obj.remove("top_p");
                    obj.remove("top_k");
                }

                // Vercel-aligned: adjust max_tokens to account for thinking budget.
                if let Some(mt) = out.get("max_tokens").and_then(|v| v.as_u64()) {
                    out["max_tokens"] = serde_json::json!(mt.saturating_add(budget as u64));
                }
            }

            // 🎯 Inject structured output if configured
            if let Some(ref rf) = response_format {
                match rf {
                    AnthropicResponseFormat::JsonObject => {
                        out["response_format"] = serde_json::json!({ "type": "json_object" });
                    }
                    AnthropicResponseFormat::JsonSchema {
                        name,
                        schema,
                        strict,
                    } => {
                        out["response_format"] = serde_json::json!({
                            "type": "json_schema",
                            "json_schema": {
                                "name": name,
                                "strict": strict,
                                "schema": schema
                            }
                        });
                    }
                }
            }

            if let Some(ref servers) = mcp_servers {
                out["mcp_servers"] = serde_json::Value::Array(servers.clone());
            }

            if let Some(ref container) = container {
                let is_empty = container.id.is_none()
                    && container
                        .skills
                        .as_ref()
                        .map(|s| s.is_empty())
                        .unwrap_or(true);
                if !is_empty {
                    out["container"] = serde_json::to_value(container).map_err(|e| {
                        LlmError::InvalidParameter(format!(
                            "Failed to serialize Anthropic container options: {e}"
                        ))
                    })?;
                }
            }

            if let Some(ref cm) = context_management {
                out["context_management"] = cm.clone();
            }

            if let Some(effort) = effort {
                out["output_config"] = serde_json::json!({
                    "effort": effort,
                });
            }

            // Vercel-aligned: limit max_tokens to the model max for known models.
            if let Some(max_out) = max_output_tokens
                && let Some(mt) = out.get("max_tokens").and_then(|v| v.as_u64())
                && mt > max_out as u64
            {
                out["max_tokens"] = serde_json::json!(max_out);
            }

            Ok(out)
        };
        Some(Arc::new(hook))
    }
}

impl AnthropicSpec {
    fn anthropic_options_from_provider_options_map(
        &self,
        req: &ChatRequest,
    ) -> Option<AnthropicOptions> {
        let value = req.provider_options_map.get("anthropic")?;
        self.anthropic_options_from_provider_options_value(value)
    }

    fn anthropic_options_from_provider_options_value(
        &self,
        value: &serde_json::Value,
    ) -> Option<AnthropicOptions> {
        let normalized = Self::normalize_anthropic_provider_options_json(value);
        serde_json::from_value(normalized).ok()
    }

    fn normalize_anthropic_provider_options_json(value: &serde_json::Value) -> serde_json::Value {
        fn normalize_key(k: &str) -> Option<&'static str> {
            Some(match k {
                // AnthropicOptions
                "promptCaching" => "prompt_caching",
                "thinkingMode" => "thinking_mode",
                "responseFormat" => "response_format",
                "contextManagement" => "context_management",
                "toolStreaming" => "tool_streaming",
                "expiresAt" => "expires_at",
                // PromptCachingConfig
                "cacheControl" => "cache_control",
                // AnthropicCacheControl
                "cacheType" => "cache_type",
                "messageIndex" => "message_index",
                // ThinkingModeConfig
                "thinkingBudget" => "thinking_budget",
                // Agent skills container
                "skillId" => "skill_id",
                // Context management edit fields (Vercel shape -> API snake_case)
                "clearAtLeast" => "clear_at_least",
                "clearToolInputs" => "clear_tool_inputs",
                "excludeTools" => "exclude_tools",
                _ => return None,
            })
        }

        fn inner(value: &serde_json::Value) -> serde_json::Value {
            match value {
                serde_json::Value::Object(map) => {
                    let mut out = serde_json::Map::new();
                    for (k, v) in map {
                        // Vercel-aligned: `providerOptions.anthropic.thinking`
                        // shape -> our `thinking_mode`.
                        if k == "thinking"
                            && let Some(obj) = v.as_object()
                        {
                            let enabled = obj
                                .get("type")
                                .and_then(|t| t.as_str())
                                .map(|t| t == "enabled")
                                .unwrap_or(false);
                            let budget = obj
                                .get("budgetTokens")
                                .or_else(|| obj.get("budget_tokens"))
                                .and_then(|b| b.as_u64())
                                .and_then(|b| u32::try_from(b).ok());

                            let mut thinking_mode = serde_json::Map::new();
                            thinking_mode
                                .insert("enabled".to_string(), serde_json::Value::Bool(enabled));
                            if let Some(b) = budget {
                                thinking_mode
                                    .insert("thinking_budget".to_string(), serde_json::json!(b));
                            }
                            out.insert(
                                "thinking_mode".to_string(),
                                serde_json::Value::Object(thinking_mode),
                            );
                            continue;
                        }

                        let nk = normalize_key(k).unwrap_or(k);
                        out.insert(nk.to_string(), inner(v));
                    }
                    serde_json::Value::Object(out)
                }
                serde_json::Value::Array(arr) => {
                    serde_json::Value::Array(arr.iter().map(inner).collect())
                }
                other => other.clone(),
            }
        }

        inner(value)
    }

    fn anthropic_mcp_servers_from_provider_options_map(
        &self,
        req: &ChatRequest,
    ) -> Option<Vec<serde_json::Value>> {
        let value = req.provider_options_map.get("anthropic")?;
        let obj = value.as_object()?;

        let servers = obj
            .get("mcpServers")
            .or_else(|| obj.get("mcp_servers"))
            .and_then(|v| v.as_array())
            .cloned()?;

        if servers.is_empty() {
            return None;
        }

        let normalized: Vec<serde_json::Value> = servers
            .into_iter()
            .filter_map(|v| v.as_object().cloned())
            .map(|map| {
                let mut out = serde_json::Map::new();

                for (k, v) in map {
                    let nk = match k.as_str() {
                        "serverName" => "name",
                        "serverUrl" => "url",
                        other => other,
                    };
                    out.insert(nk.to_string(), v);
                }

                serde_json::Value::Object(out)
            })
            .collect();

        if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_request_headers_unions_anthropic_beta_features() {
        let mut ctx_headers = HashMap::new();
        ctx_headers.insert(
            "anthropic-beta".to_string(),
            "web-fetch-2025-09-10,advanced-tool-use-2025-11-20".to_string(),
        );

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com",
            Some("k".to_string()),
            ctx_headers,
        );

        let spec = AnthropicSpec::new();
        let base = spec.build_headers(&ctx).unwrap();

        let mut extra = HashMap::new();
        extra.insert(
            "Anthropic-Beta".to_string(),
            "advanced-tool-use-2025-11-20,code-execution-2025-05-22".to_string(),
        );

        let merged = spec.merge_request_headers(base, &extra);
        let value = merged
            .get("anthropic-beta")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        assert_eq!(
            value,
            "web-fetch-2025-09-10,advanced-tool-use-2025-11-20,code-execution-2025-05-22"
        );
    }

    #[test]
    fn chat_request_headers_includes_agent_skills_betas() {
        let spec = AnthropicSpec::new();

        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({
                    "container": {
                        "skills": [
                            { "type": "anthropic", "skillId": "pptx", "version": "latest" }
                        ]
                    }
                }),
            );

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com",
            None,
            HashMap::new(),
        );
        let headers = spec.chat_request_headers(false, &req, &ctx);
        assert_eq!(
            headers.get("anthropic-beta").map(|s| s.as_str()),
            Some("code-execution-2025-08-25,skills-2025-10-02,files-api-2025-04-14")
        );
    }

    #[test]
    fn chat_request_headers_includes_fine_grained_tool_streaming_beta_by_default() {
        let spec = AnthropicSpec::new();

        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_streaming(true)
            .with_provider_option("anthropic", serde_json::json!({}));

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com/v1",
            None,
            HashMap::new(),
        );
        let headers = spec.chat_request_headers(true, &req, &ctx);
        assert!(
            headers.get("anthropic-beta").is_some_and(|v| v
                .split(',')
                .any(|t| t.trim() == "fine-grained-tool-streaming-2025-05-14")),
            "missing fine-grained-tool-streaming beta token"
        );
    }

    #[test]
    fn chat_request_headers_omits_fine_grained_tool_streaming_beta_when_disabled() {
        let spec = AnthropicSpec::new();

        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_streaming(true)
            .with_provider_option("anthropic", serde_json::json!({ "toolStreaming": false }));

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com/v1",
            None,
            HashMap::new(),
        );
        let headers = spec.chat_request_headers(true, &req, &ctx);
        let beta = headers.get("anthropic-beta").cloned().unwrap_or_default();
        assert!(
            !beta
                .split(',')
                .any(|t| t.trim() == "fine-grained-tool-streaming-2025-05-14"),
            "unexpected fine-grained-tool-streaming beta token: {beta}"
        );
    }
}
