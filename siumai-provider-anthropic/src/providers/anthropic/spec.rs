use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::provider_options::anthropic::{
    AnthropicOptions, AnthropicResponseFormat, ThinkingModeConfig,
};
use crate::standards::anthropic::chat::AnthropicChatStandard;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
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
        // Handle Anthropic-specific options (thinking_mode, response_format).
        let options = self.anthropic_options_from_provider_options_map(req)?;

        let thinking_mode: Option<ThinkingModeConfig> = options.thinking_mode.clone();
        let response_format: Option<AnthropicResponseFormat> = options.response_format.clone();

        // If neither thinking nor response format configured, nothing to inject
        if thinking_mode.is_none() && response_format.is_none() {
            return None;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();

            // 🎯 Inject thinking mode configuration
            if let Some(ref thinking) = thinking_mode
                && thinking.enabled
            {
                let mut thinking_config = serde_json::json!({ "type": "enabled" });
                if let Some(budget) = thinking.thinking_budget {
                    thinking_config["budget_tokens"] = serde_json::json!(budget);
                }
                out["thinking"] = thinking_config;
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

            Ok(out)
        };
        Some(Arc::new(hook))
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
                // PromptCachingConfig
                "cacheControl" => "cache_control",
                // AnthropicCacheControl
                "cacheType" => "cache_type",
                "messageIndex" => "message_index",
                // ThinkingModeConfig
                "thinkingBudget" => "thinking_budget",
                _ => return None,
            })
        }

        fn inner(value: &serde_json::Value) -> serde_json::Value {
            match value {
                serde_json::Value::Object(map) => {
                    let mut out = serde_json::Map::new();
                    for (k, v) in map {
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
}
