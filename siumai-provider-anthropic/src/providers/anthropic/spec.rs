use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::standards::anthropic::chat::AnthropicChatStandard;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use crate::types::Tool;
use reqwest::header::HeaderMap;
use std::collections::{HashMap, HashSet};

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
            // - enabled when using native `output_format` (depends on structuredOutputMode + model support),
            // - enabled for supported models whenever function tools are present.
            let structured_output_mode = req
                .provider_options_map
                .get("anthropic")
                .and_then(|v| v.as_object())
                .and_then(|o| {
                    o.get("structuredOutputMode")
                        .or_else(|| o.get("structured_output_mode"))
                        .and_then(|v| v.as_str())
                })
                .unwrap_or("auto");
            let prefers_output_format = structured_output_mode == "outputFormat"
                || structured_output_mode == "output_format";
            let prefers_json_tool =
                structured_output_mode == "jsonTool" || structured_output_mode == "json_tool";

            let uses_native_output_format = matches!(
                req.response_format,
                Some(crate::types::chat::ResponseFormat::Json { .. })
            ) && (prefers_output_format
                || (!prefers_json_tool && supports_structured_outputs));
            if uses_native_output_format {
                out.push("structured-outputs-2025-11-13");
            }

            let has_function_tools = req
                .tools
                .as_deref()
                .unwrap_or_default()
                .iter()
                .any(|t| matches!(t, Tool::Function { .. }));
            if supports_structured_outputs && has_function_tools {
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

        let has_mcp_servers = req
            .provider_options_map
            .get("anthropic")
            .and_then(|value| value.as_object())
            .and_then(|options| {
                options
                    .get("mcpServers")
                    .or_else(|| options.get("mcp_servers"))
            })
            .and_then(|value| value.as_array())
            .is_some_and(|servers| !servers.is_empty());
        if has_mcp_servers {
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
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {

            // 🎯 Inject structured output if configured


        None
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

    #[test]
    fn chat_before_send_serializes_container_id_only_as_string() {
        let spec = AnthropicSpec::new();
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({ "container": { "id": "c_1" } }),
            );

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("hook");

        let out = hook(&serde_json::json!({"model":"m","messages":[],"max_tokens":1}))
            .expect("apply hook");

        assert_eq!(out.get("container"), Some(&serde_json::json!("c_1")));
    }

    #[test]
    fn chat_before_send_serializes_container_skills_as_object() {
        let spec = AnthropicSpec::new();
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({
                    "container": {
                        "id": "c_1",
                        "skills": [
                            { "type": "anthropic", "skillId": "pptx", "version": "latest" }
                        ]
                    }
                }),
            );

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("hook");

        let out = hook(&serde_json::json!({"model":"m","messages":[],"max_tokens":1}))
            .expect("apply hook");

        let obj = out
            .get("container")
            .and_then(|v| v.as_object())
            .expect("object");
        assert_eq!(obj.get("id").and_then(|v| v.as_str()), Some("c_1"));
        let skills = obj
            .get("skills")
            .and_then(|v| v.as_array())
            .expect("skills array");
        assert_eq!(
            skills[0].get("skill_id").and_then(|v| v.as_str()),
            Some("pptx")
        );
    }

    #[test]
    fn chat_before_send_normalizes_mcp_servers_keys_to_snake_case() {
        let spec = AnthropicSpec::new();
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({
                    "mcpServers": [
                        {
                            "type": "url",
                            "name": "s1",
                            "url": "https://example.com",
                            "authorizationToken": "tok",
                            "toolConfiguration": {
                                "enabled": true,
                                "allowedTools": ["a", "b"]
                            }
                        }
                    ]
                }),
            );

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("hook");

        let out = hook(&serde_json::json!({"model":"m","messages":[],"max_tokens":1}))
            .expect("apply hook");

        let servers = out
            .get("mcp_servers")
            .and_then(|v| v.as_array())
            .expect("mcp_servers array");
        let server = servers[0].as_object().expect("server object");
        assert!(server.contains_key("authorization_token"));
        let tc = server
            .get("tool_configuration")
            .and_then(|v| v.as_object())
            .expect("tool_configuration");
        assert!(tc.contains_key("allowed_tools"));
    }

    #[test]
    fn chat_before_send_injects_context_management_and_effort() {
        let spec = AnthropicSpec::new();
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({
                    "thinkingMode": {
                        "enabled": true,
                        "thinkingBudget": 1000
                    },
                    "contextManagement": {
                        "clear_at_least": 1,
                        "exclude_tools": ["editor"]
                    },
                    "effort": "high"
                }),
            );

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("hook");

        let out = hook(
            &serde_json::json!({"model":"m","messages":[],"max_tokens":100,"temperature":0.5}),
        )
        .expect("apply hook");

        assert_eq!(
            out.get("context_management"),
            Some(&serde_json::json!({
                "clear_at_least": 1,
                "exclude_tools": ["editor"]
            }))
        );
        assert_eq!(
            out.get("output_config"),
            Some(&serde_json::json!({
                "effort": "high"
            }))
        );
    }
}
