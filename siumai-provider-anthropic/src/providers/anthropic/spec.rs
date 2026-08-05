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

fn anthropic_provider_options(
    req: &ChatRequest,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    req.provider_options_map
        .get("anthropic")
        .and_then(|value| value.as_object())
}

fn provider_option_bool(req: &ChatRequest, camel: &str, snake: &str) -> Option<bool> {
    anthropic_provider_options(req)
        .and_then(|options| options.get(camel).or_else(|| options.get(snake)))
        .and_then(|value| value.as_bool())
}

fn provider_option_str<'a>(req: &'a ChatRequest, camel: &str, snake: &str) -> Option<&'a str> {
    anthropic_provider_options(req)
        .and_then(|options| options.get(camel).or_else(|| options.get(snake)))
        .and_then(|value| value.as_str())
}

fn user_requested_beta_tokens(req: &ChatRequest) -> Vec<String> {
    anthropic_provider_options(req)
        .and_then(|options| {
            options
                .get("anthropicBeta")
                .or_else(|| options.get("anthropic_beta"))
        })
        .and_then(|value| value.as_array())
        .map(|tokens| {
            super::sanitize_beta_features(
                tokens
                    .iter()
                    .filter_map(|value| value.as_str())
                    .filter(|value| !value.trim().is_empty())
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap_or_default()
}

fn has_agent_skills(req: &ChatRequest) -> bool {
    anthropic_provider_options(req)
        .and_then(|options| options.get("container"))
        .and_then(|value| value.as_object())
        .and_then(|container| container.get("skills"))
        .and_then(|value| value.as_array())
        .is_some_and(|skills| !skills.is_empty())
}

fn has_context_management(req: &ChatRequest) -> bool {
    anthropic_provider_options(req).is_some_and(|options| {
        options.get("contextManagement").is_some() || options.get("context_management").is_some()
    })
}

fn has_effort(req: &ChatRequest) -> bool {
    anthropic_provider_options(req).is_some_and(|options| options.get("effort").is_some())
}

fn has_task_budget(req: &ChatRequest) -> bool {
    anthropic_provider_options(req).is_some_and(|options| {
        options.get("taskBudget").is_some() || options.get("task_budget").is_some()
    })
}

fn has_mcp_servers(req: &ChatRequest) -> bool {
    anthropic_provider_options(req)
        .and_then(|options| {
            options
                .get("mcpServers")
                .or_else(|| options.get("mcp_servers"))
        })
        .and_then(|value| value.as_array())
        .is_some_and(|servers| !servers.is_empty())
}

fn fine_grained_tool_streaming_enabled(req: &ChatRequest) -> bool {
    if !req.stream {
        return false;
    }

    provider_option_bool(req, "toolStreaming", "tool_streaming").unwrap_or(true)
}

fn apply_known_model_policy(
    req: &ChatRequest,
    body: &mut serde_json::Value,
) -> Result<(), LlmError> {
    use super::model_constants::{
        CapabilitySupport, SamplingSupport, ThinkingDisablePolicy, ThinkingSupport, model_profile,
    };

    let model = req.common_params.model.as_str();
    let Some(profile) = model_profile(model) else {
        return Ok(());
    };

    if matches!(profile.sampling, SamplingSupport::Unsupported) {
        let has_non_default_sampling = req
            .common_params
            .temperature
            .is_some_and(|value| value != 1.0)
            || req.common_params.top_p.is_some_and(|value| value != 1.0)
            || req.common_params.top_k.is_some();
        if has_non_default_sampling {
            return Err(LlmError::InvalidParameter(format!(
                "{model} rejects non-default temperature, top_p, and top_k values; omit sampling controls"
            )));
        }

        // Anthropic accepts default values, but omitting them keeps the request stable across
        // current models and avoids coupling the wire payload to deprecated controls.
        if let Some(object) = body.as_object_mut() {
            object.remove("temperature");
            object.remove("top_p");
            object.remove("top_k");
        }
    }

    let thinking_type = body
        .get("thinking")
        .and_then(|thinking| thinking.get("type"))
        .and_then(serde_json::Value::as_str);

    if matches!(profile.thinking, ThinkingSupport::AdaptiveOnly)
        && matches!(thinking_type, Some(value) if value != "adaptive" && value != "disabled")
    {
        return Err(LlmError::InvalidParameter(format!(
            "{model} does not support manual thinking budgets; use adaptive thinking or disable thinking when the model permits it"
        )));
    }

    if thinking_type == Some("disabled") {
        match profile.thinking_disable {
            ThinkingDisablePolicy::Forbidden => {
                return Err(LlmError::InvalidParameter(format!(
                    "{model} always uses adaptive thinking and cannot disable it"
                )));
            }
            ThinkingDisablePolicy::AllowedThroughHighEffort => {
                let effort = body
                    .get("effort")
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| provider_option_str(req, "effort", "effort"));
                if matches!(effort, Some("xhigh" | "max")) {
                    return Err(LlmError::InvalidParameter(format!(
                        "{model} can disable thinking only through high effort; xhigh and max require thinking"
                    )));
                }
            }
            ThinkingDisablePolicy::Allowed | ThinkingDisablePolicy::Unknown => {}
        }
    }

    if profile.supports_prefill == CapabilitySupport::Unsupported
        && body
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .and_then(|messages| messages.last())
            .and_then(|message| message.get("role"))
            .and_then(serde_json::Value::as_str)
            == Some("assistant")
    {
        return Err(LlmError::InvalidParameter(format!(
            "{model} does not support assistant-message prefill"
        )));
    }

    if let Some(requested) = req.common_params.max_tokens
        && requested > profile.max_output_tokens
    {
        return Err(LlmError::InvalidParameter(format!(
            "{model} accepts at most {} output tokens, but {requested} were requested",
            profile.max_output_tokens
        )));
    }
    if let Some(requested) = req.common_params.max_tokens
        && let Some(object) = body.as_object_mut()
    {
        // Restore a caller value only when the compatibility transformer capped it. A larger
        // mapped value may intentionally include a manual-thinking budget and must be preserved.
        let mapped = object.get("max_tokens").and_then(serde_json::Value::as_u64);
        if mapped.is_some_and(|mapped| mapped < u64::from(requested)) {
            object.insert("max_tokens".to_string(), serde_json::json!(requested));
        }
    }

    Ok(())
}

pub(crate) fn collect_request_beta_tokens(req: &ChatRequest) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    let mut push_token = |token: &str| {
        let token = token.trim();
        if token.is_empty() {
            return;
        }
        if seen.insert(token.to_string()) {
            tokens.push(token.to_string());
        }
    };

    let model = req.common_params.model.as_str();
    let supports_structured_outputs = model.starts_with("claude-sonnet-4-5")
        || model.starts_with("claude-opus-4-5")
        || model.starts_with("claude-haiku-4-5");

    if has_mcp_servers(req) {
        push_token("mcp-client-2025-04-04");
    }

    if let Some(tools) = &req.tools {
        for tool in tools {
            let Tool::ProviderDefined(t) = tool else {
                continue;
            };
            if t.provider() != Some("anthropic") {
                continue;
            }
            match t.tool_type() {
                Some("web_fetch_20250910") => push_token("web-fetch-2025-09-10"),
                Some("web_fetch_20260209") | Some("web_search_20260209") => {
                    push_token("code-execution-web-tools-2026-02-09")
                }
                Some("code_execution_20250522") => push_token("code-execution-2025-05-22"),
                Some("code_execution_20250825") => push_token("code-execution-2025-08-25"),
                Some("computer_20241022")
                | Some("text_editor_20241022")
                | Some("bash_20241022") => push_token("computer-use-2024-10-22"),
                Some("computer_20251124") => push_token("computer-use-2025-11-24"),
                Some("computer_20250124")
                | Some("text_editor_20250124")
                | Some("text_editor_20250429")
                | Some("bash_20250124") => push_token("computer-use-2025-01-24"),
                Some("tool_search_regex_20251119") | Some("tool_search_bm25_20251119") => {
                    push_token("advanced-tool-use-2025-11-20")
                }
                Some("memory_20250818") => push_token("context-management-2025-06-27"),
                _ => {}
            }
        }
    }

    let structured_output_mode =
        provider_option_str(req, "structuredOutputMode", "structured_output_mode")
            .unwrap_or("auto");
    let prefers_native_structured_output =
        structured_output_mode == "outputFormat" || structured_output_mode == "output_format";
    let prefers_json_tool =
        structured_output_mode == "jsonTool" || structured_output_mode == "json_tool";

    let uses_native_structured_output = matches!(
        req.response_format,
        Some(crate::types::chat::ResponseFormat::Json { .. })
    ) && (prefers_native_structured_output
        || (!prefers_json_tool && supports_structured_outputs));
    if uses_native_structured_output {
        push_token("structured-outputs-2025-11-13");
    }

    let has_function_tools = req
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .any(|tool| matches!(tool, Tool::Function { .. }));
    if supports_structured_outputs && has_function_tools {
        push_token("structured-outputs-2025-11-13");
    }

    if let Some(tools) = req.tools.as_deref() {
        for tool in tools {
            let Tool::Function { function } = tool else {
                continue;
            };

            if function
                .input_examples
                .as_ref()
                .is_some_and(|examples| !examples.is_empty())
            {
                push_token("advanced-tool-use-2025-11-20");
                break;
            }

            let has_allowed_callers = function
                .provider_options_map
                .get("anthropic")
                .and_then(|value| value.as_object())
                .and_then(|options| {
                    options
                        .get("allowedCallers")
                        .or_else(|| options.get("allowed_callers"))
                })
                .and_then(|value| value.as_array())
                .is_some_and(|callers| !callers.is_empty());
            if has_allowed_callers {
                push_token("advanced-tool-use-2025-11-20");
                break;
            }
        }
    }

    let uses_pdf = req.messages.iter().any(|message| match &message.content {
        crate::types::MessageContent::MultiModal(parts) => parts.iter().any(|part| {
            matches!(
                part,
                crate::types::ContentPart::File { media_type, .. }
                    if media_type == "application/pdf"
            )
        }),
        _ => false,
    });
    if uses_pdf {
        push_token("pdfs-2024-09-25");
    }

    let uses_anthropic_file_references =
        req.messages.iter().any(|message| match &message.content {
            crate::types::MessageContent::MultiModal(parts) => {
                parts.iter().any(|part| match part {
                    crate::types::ContentPart::Image { source, .. }
                    | crate::types::ContentPart::File { source, .. } => source
                        .as_provider_reference()
                        .and_then(|provider_reference| provider_reference.get("anthropic"))
                        .is_some(),
                    _ => false,
                })
            }
            _ => false,
        });
    if uses_anthropic_file_references {
        push_token("files-api-2025-04-14");
    }

    if has_context_management(req) {
        push_token("context-management-2025-06-27");
    }

    if fine_grained_tool_streaming_enabled(req) {
        push_token("fine-grained-tool-streaming-2025-05-14");
    }

    if has_effort(req) {
        push_token("effort-2025-11-24");
    }

    if has_task_budget(req) {
        push_token("task-budgets-2026-03-13");
    }

    if matches!(provider_option_str(req, "speed", "speed"), Some("fast")) {
        push_token("fast-mode-2026-02-01");
    }

    if has_agent_skills(req) {
        push_token("code-execution-2025-08-25");
        push_token("skills-2025-10-02");
        push_token("files-api-2025-04-14");
    }

    for token in user_requested_beta_tokens(req) {
        push_token(&token);
    }

    tokens
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
            .with_custom_feature("skills", true)
            .with_custom_feature("prompt_caching", true)
            .with_custom_feature("thinking_mode", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().unwrap_or_default();
        let has_authorization = ctx
            .http_extra_headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("authorization"));
        if api_key.is_empty() && !has_authorization {
            return Err(LlmError::MissingApiKey(
                "Anthropic API key or auth token not provided".into(),
            ));
        }
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
        let tokens = collect_request_beta_tokens(req);
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

    fn try_chat_url(
        &self,
        _stream: bool,
        _req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let base = ctx.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            Ok(format!("{base}/messages"))
        } else {
            Ok(format!("{base}/v1/messages"))
        }
    }

    fn try_models_url(&self, ctx: &ProviderContext) -> Result<String, LlmError> {
        let base = ctx.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            Ok(format!("{base}/models"))
        } else {
            Ok(format!("{base}/v1/models"))
        }
    }

    fn try_model_url(&self, model_id: &str, ctx: &ProviderContext) -> Result<String, LlmError> {
        let base = ctx.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            Ok(format!("{base}/models/{model_id}"))
        } else {
            Ok(format!("{base}/v1/models/{model_id}"))
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
        let needs_overlays =
            crate::standards::anthropic::request_options::anthropic_request_body_overlays_needed(
                req,
            );
        let needs_model_policy =
            super::model_constants::model_profile(&req.common_params.model).is_some();
        if !needs_overlays && !needs_model_policy {
            return None;
        }

        let req = req.clone();
        Some(std::sync::Arc::new(move |body: &serde_json::Value| {
            let mut out = body.clone();
            if needs_overlays {
                crate::standards::anthropic::request_options::apply_anthropic_request_body_overlays(
                    &req, &mut out,
                );
            }
            apply_known_model_policy(&req, &mut out)?;
            Ok(out)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::anthropic::{
        AnthropicContainerConfig, AnthropicContainerSkill, AnthropicOptions,
    };
    use crate::providers::anthropic::ext::request_options::AnthropicChatRequestExt;

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn anthropic_spec_request_option_routing_does_not_read_response_metadata() {
        let source = include_str!("spec.rs");
        let request_routing =
            source_section(source, "fn anthropic_provider_options", "impl ProviderSpec");

        assert!(
            request_routing.contains("provider_options_map"),
            "Anthropic spec should keep request beta/feature routing on request provider options"
        );

        for forbidden in ["provider_metadata", "ProviderMetadata", "providerMetadata"] {
            assert!(
                !request_routing.contains(forbidden),
                "Anthropic spec request routing must not read response metadata"
            );
        }
    }

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
    fn chat_request_headers_include_code_execution_web_tools_beta_for_2026_web_tools() {
        let spec = AnthropicSpec::new();

        let req =
            ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]).with_tools(vec![
                siumai_protocol_anthropic::tool_catalog::anthropic::web_search_20260209(),
            ]);

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com",
            None,
            HashMap::new(),
        );
        let headers = spec.chat_request_headers(false, &req, &ctx);
        assert!(headers.get("anthropic-beta").is_some_and(|value| {
            value
                .split(',')
                .any(|token| token.trim() == "code-execution-web-tools-2026-02-09")
        }));
    }

    #[test]
    fn chat_request_headers_include_computer_use_2025_11_24_beta() {
        let spec = AnthropicSpec::new();

        let req =
            ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]).with_tools(vec![
                siumai_protocol_anthropic::tool_catalog::anthropic::computer_20251124(),
            ]);

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com",
            None,
            HashMap::new(),
        );
        let headers = spec.chat_request_headers(false, &req, &ctx);
        assert!(headers.get("anthropic-beta").is_some_and(|value| {
            value
                .split(',')
                .any(|token| token.trim() == "computer-use-2025-11-24")
        }));
    }

    #[test]
    fn chat_request_headers_do_not_add_beta_for_code_execution_20260120() {
        let spec = AnthropicSpec::new();

        let req =
            ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]).with_tools(vec![
                siumai_protocol_anthropic::tool_catalog::anthropic::code_execution_20260120(),
            ]);

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com",
            None,
            HashMap::new(),
        );
        let headers = spec.chat_request_headers(false, &req, &ctx);
        let beta = headers.get("anthropic-beta").cloned().unwrap_or_default();

        assert!(
            !beta
                .split(',')
                .any(|token| token.trim() == "code-execution-2025-05-22"
                    || token.trim() == "code-execution-2025-08-25"),
            "unexpected code-execution beta token: {beta}"
        );
    }

    #[test]
    fn chat_request_headers_include_files_api_beta_for_provider_references() {
        let spec = AnthropicSpec::new();

        let req = ChatRequest::new(vec![
            crate::types::ChatMessage::user("hi")
                .with_file_provider_reference(
                    crate::types::ProviderReference::single("anthropic", "file-1"),
                    "application/pdf",
                    Some("doc.pdf".to_string()),
                )
                .build(),
        ]);

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com",
            None,
            HashMap::new(),
        );
        let headers = spec.chat_request_headers(false, &req, &ctx);
        assert!(headers.get("anthropic-beta").is_some_and(|value| {
            value
                .as_str()
                .split(',')
                .any(|token| token.trim() == "files-api-2025-04-14")
        }));
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
    fn chat_request_headers_include_fast_mode_and_user_requested_betas() {
        let spec = AnthropicSpec::new();

        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({
                    "speed": "fast",
                    "anthropicBeta": ["custom-beta-1", "custom-beta-2"]
                }),
            );

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com/v1",
            None,
            HashMap::new(),
        );
        let headers = spec.chat_request_headers(false, &req, &ctx);
        let beta = headers.get("anthropic-beta").cloned().unwrap_or_default();

        assert!(
            beta.split(',').any(|t| t.trim() == "fast-mode-2026-02-01"),
            "missing fast-mode beta token: {beta}"
        );
        assert!(
            beta.split(',').any(|t| t.trim() == "custom-beta-1"),
            "missing custom-beta-1 token: {beta}"
        );
        assert!(
            beta.split(',').any(|t| t.trim() == "custom-beta-2"),
            "missing custom-beta-2 token: {beta}"
        );
    }

    #[test]
    fn chat_request_headers_omit_retired_prompt_caching_beta() {
        let spec = AnthropicSpec::new();
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({
                    "anthropicBeta": [
                        "prompt-caching-2024-07-31",
                        "custom-beta-1"
                    ]
                }),
            );
        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com/v1",
            None,
            HashMap::new(),
        );

        let headers = spec.chat_request_headers(false, &req, &ctx);
        let beta = headers.get("anthropic-beta").cloned().unwrap_or_default();

        assert!(!beta.contains("prompt-caching-2024-07-31"));
        assert!(beta.contains("custom-beta-1"));
    }

    #[test]
    fn chat_request_headers_include_task_budget_beta() {
        let spec = AnthropicSpec::new();

        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({
                    "taskBudget": {
                        "type": "tokens",
                        "total": 400000,
                        "remaining": 215000
                    }
                }),
            );

        let ctx = ProviderContext::new(
            "anthropic",
            "https://api.anthropic.com/v1",
            None,
            HashMap::new(),
        );
        let headers = spec.chat_request_headers(false, &req, &ctx);
        let beta = headers.get("anthropic-beta").cloned().unwrap_or_default();

        assert!(
            beta.split(',')
                .any(|t| t.trim() == "task-budgets-2026-03-13"),
            "missing task budget beta token: {beta}"
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
    fn current_model_policy_restores_requested_output_limit_after_protocol_mapping() {
        let spec = AnthropicSpec::new();
        let mut req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        req.common_params.model = "claude-opus-5".to_string();
        req.common_params.max_tokens = Some(100_000);

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("model policy hook");
        let out = hook(&serde_json::json!({
            "model": "claude-opus-5",
            "messages": [{"role":"user","content":"hi"}],
            "max_tokens": 8192
        }))
        .expect("apply model policy");

        assert_eq!(out.get("max_tokens"), Some(&serde_json::json!(100_000)));
    }

    #[test]
    fn current_model_policy_rejects_sampling_controls() {
        let spec = AnthropicSpec::new();
        let mut req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        req.common_params.model = "claude-sonnet-5".to_string();
        req.common_params.temperature = Some(0.2);

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("model policy hook");
        let error = hook(&serde_json::json!({
            "model": "claude-sonnet-5",
            "messages": [{"role":"user","content":"hi"}],
            "max_tokens": 4096,
            "temperature": 0.2
        }))
        .expect_err("sampling controls must be rejected before sending");

        assert!(matches!(error, LlmError::InvalidParameter(_)));
    }

    #[test]
    fn current_model_policy_omits_explicit_default_sampling_controls() {
        let spec = AnthropicSpec::new();
        let mut req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        req.common_params.model = "claude-sonnet-5".to_string();
        req.common_params.temperature = Some(1.0);
        req.common_params.top_p = Some(1.0);

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("model policy hook");
        let out = hook(&serde_json::json!({
            "model": "claude-sonnet-5",
            "messages": [{"role":"user","content":"hi"}],
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 1.0
        }))
        .expect("default sampling values remain callable");

        assert!(out.get("temperature").is_none());
        assert!(out.get("top_p").is_none());
    }

    #[test]
    fn current_model_policy_rejects_manual_thinking_and_prefill() {
        let spec = AnthropicSpec::new();
        let mut req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({"thinking": {"type": "enabled", "budgetTokens": 2048}}),
            );
        req.common_params.model = "claude-fable-5".to_string();

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("model policy hook");
        let error = hook(&serde_json::json!({
            "model": "claude-fable-5",
            "messages": [{"role":"user","content":"hi"}],
            "max_tokens": 4096
        }))
        .expect_err("manual thinking must be rejected");
        assert!(matches!(error, LlmError::InvalidParameter(_)));

        let mut prefill_req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        prefill_req.common_params.model = "claude-fable-5".to_string();
        let prefill_hook = spec
            .chat_before_send(
                &prefill_req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("model policy hook");
        let error = prefill_hook(&serde_json::json!({
            "model": "claude-fable-5",
            "messages": [{"role":"assistant","content":"prefix"}],
            "max_tokens": 4096
        }))
        .expect_err("assistant prefill must be rejected");
        assert!(matches!(error, LlmError::InvalidParameter(_)));
    }

    #[test]
    fn current_model_policy_distinguishes_thinking_disable_rules() {
        let spec = AnthropicSpec::new();
        let context = ProviderContext::new("anthropic", "", None, HashMap::new());

        let mut sonnet = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        sonnet.common_params.model = "claude-sonnet-5".to_string();
        let sonnet_hook = spec
            .chat_before_send(&sonnet, &context)
            .expect("Sonnet policy hook");
        sonnet_hook(&serde_json::json!({
            "model": "claude-sonnet-5",
            "messages": [{"role":"user","content":"hi"}],
            "thinking": {"type":"disabled"},
            "max_tokens": 4096
        }))
        .expect("Sonnet 5 permits disabled thinking");

        let mut fable = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        fable.common_params.model = "claude-fable-5".to_string();
        let fable_hook = spec
            .chat_before_send(&fable, &context)
            .expect("Fable policy hook");
        assert!(
            fable_hook(&serde_json::json!({
                "model": "claude-fable-5",
                "messages": [{"role":"user","content":"hi"}],
                "thinking": {"type":"disabled"},
                "max_tokens": 4096
            }))
            .is_err()
        );

        let mut opus = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        opus.common_params.model = "claude-opus-5".to_string();
        let opus_hook = spec
            .chat_before_send(&opus, &context)
            .expect("Opus policy hook");
        assert!(
            opus_hook(&serde_json::json!({
                "model": "claude-opus-5",
                "messages": [{"role":"user","content":"hi"}],
                "thinking": {"type":"disabled"},
                "effort": "max",
                "max_tokens": 4096
            }))
            .is_err()
        );
        opus_hook(&serde_json::json!({
            "model": "claude-opus-5",
            "messages": [{"role":"user","content":"hi"}],
            "thinking": {"type":"disabled"},
            "effort": "high",
            "max_tokens": 4096
        }))
        .expect("Opus 5 permits disabled thinking through high effort");
    }

    #[test]
    fn current_model_policy_rejects_output_limit_overflow() {
        let spec = AnthropicSpec::new();
        let mut req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        req.common_params.model = "claude-opus-5".to_string();
        req.common_params.max_tokens = Some(128_001);

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("model policy hook");
        assert!(
            hook(&serde_json::json!({
                "model": "claude-opus-5",
                "messages": [{"role":"user","content":"hi"}],
                "max_tokens": 8192
            }))
            .is_err()
        );
    }

    #[test]
    fn unknown_model_without_overlays_remains_callable_without_policy_hook() {
        let spec = AnthropicSpec::new();
        let mut req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        req.common_params.model = "claude-future-custom-model".to_string();

        assert!(
            spec.chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .is_none()
        );
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
    fn chat_before_send_resolves_custom_container_skill_provider_reference() {
        let spec = AnthropicSpec::new();
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_anthropic_options(AnthropicOptions::new().with_container(
                AnthropicContainerConfig {
                    id: Some("c_1".to_string()),
                    skills: Some(vec![AnthropicContainerSkill::custom(
                        crate::types::ProviderReference::single("anthropic", "skill_custom_1"),
                    )
                    .with_version("latest")]),
                },
            ));

        let hook = spec
            .chat_before_send(
                &req,
                &ProviderContext::new("anthropic", "", None, HashMap::new()),
            )
            .expect("hook");

        let out = hook(&serde_json::json!({"model":"m","messages":[],"max_tokens":1}))
            .expect("apply hook");

        let skills = out["container"]["skills"].as_array().expect("skills array");
        assert_eq!(skills[0]["type"], serde_json::json!("custom"));
        assert_eq!(skills[0]["skill_id"], serde_json::json!("skill_custom_1"));
        assert_eq!(skills[0]["version"], serde_json::json!("latest"));
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

    #[test]
    fn chat_before_send_injects_speed_cache_control_and_metadata() {
        let spec = AnthropicSpec::new();
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({
                    "thinking": {
                        "type": "adaptive"
                    },
                    "cacheControl": {
                        "type": "ephemeral",
                        "ttl": "1h"
                    },
                    "metadata": {
                        "userId": "user-1"
                    },
                    "speed": "fast"
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
            out.get("thinking"),
            Some(&serde_json::json!({ "type": "adaptive" }))
        );
        assert_eq!(
            out.get("cache_control"),
            Some(&serde_json::json!({
                "type": "ephemeral",
                "ttl": "1h"
            }))
        );
        assert_eq!(
            out.get("metadata"),
            Some(&serde_json::json!({
                "user_id": "user-1"
            }))
        );
        assert_eq!(out.get("speed"), Some(&serde_json::json!("fast")));
        assert!(out.get("temperature").is_none());
    }
}
