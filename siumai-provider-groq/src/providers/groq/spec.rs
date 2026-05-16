use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::provider_options::GroqOptions;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use reqwest::header::HeaderMap;
use std::sync::Arc;

#[derive(Clone, Copy, Default)]
struct GroqOpenAiChatAdapter;

impl crate::standards::openai::chat::OpenAiChatAdapter for GroqOpenAiChatAdapter {
    fn build_headers(
        &self,
        api_key: &str,
        base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        use reqwest::header::{CONTENT_TYPE, HeaderValue, USER_AGENT};

        if api_key.is_empty() {
            return Err(LlmError::MissingApiKey("Groq API key not provided".into()));
        }

        // Keep Groq behavior aligned with the legacy implementation: always send JSON content type.
        base_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Prefer the shared HTTP client user agent, but ensure a stable header in case a custom
        // reqwest client doesn't set one.
        let version = env!("CARGO_PKG_VERSION");
        let ua = HeaderValue::from_str(&format!("siumai/{version} (groq)")).map_err(|e| {
            LlmError::InvalidParameter(format!("Invalid Groq user-agent header: {e}"))
        })?;
        base_headers.insert(USER_AGENT, ua);

        Ok(())
    }

    fn transform_request(
        &self,
        req: &ChatRequest,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        // Groq does not accept the "developer" role; treat it as "system".
        if let Some(msgs) = body.get_mut("messages").and_then(|v| v.as_array_mut()) {
            for m in msgs {
                if m.get("role").and_then(|v| v.as_str()) == Some("developer") {
                    m["role"] = serde_json::Value::String("system".to_string());
                }
            }
        }

        // Groq does not document stream_options; keep behavior aligned with the previous
        // implementation by omitting it.
        if req.stream
            && let Some(obj) = body.as_object_mut()
        {
            obj.remove("stream_options");
        }

        // Groq uses `max_tokens` (OpenAI-style chat completions) rather than
        // `max_completion_tokens`.
        if let Some(obj) = body.as_object_mut()
            && let Some(v) = obj.remove("max_completion_tokens")
        {
            obj.entry("max_tokens".to_string()).or_insert(v);
        }

        Ok(())
    }
}

/// Groq ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct GroqSpec;

fn take_any(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<serde_json::Value> {
    for key in keys {
        if let Some(value) = obj.remove(*key) {
            return Some(value);
        }
    }
    None
}

fn normalize_groq_body_option_keys(obj: &mut serde_json::Map<String, serde_json::Value>) {
    let structured_outputs = take_any(obj, &["structuredOutputs", "structured_outputs"])
        .and_then(|value| value.as_bool());
    let strict_json_schema = take_any(obj, &["strictJsonSchema", "strict_json_schema"])
        .and_then(|value| value.as_bool());

    if let Some(top_logprobs) = take_any(obj, &["topLogprobs", "top_logprobs"]) {
        obj.insert("top_logprobs".to_string(), top_logprobs);
    }
    if let Some(service_tier) = take_any(obj, &["serviceTier", "service_tier"]) {
        obj.insert("service_tier".to_string(), service_tier);
    }
    if let Some(reasoning_effort) = take_any(obj, &["reasoningEffort", "reasoning_effort"]) {
        obj.insert("reasoning_effort".to_string(), reasoning_effort);
    }
    if let Some(reasoning_format) = take_any(obj, &["reasoningFormat", "reasoning_format"]) {
        obj.insert("reasoning_format".to_string(), reasoning_format);
    }
    if let Some(parallel_tool_calls) = take_any(obj, &["parallelToolCalls", "parallel_tool_calls"])
    {
        obj.insert("parallel_tool_calls".to_string(), parallel_tool_calls);
    }

    if let Some(false) = structured_outputs
        && obj
            .get("response_format")
            .and_then(|value| value.get("type"))
            .and_then(|value| value.as_str())
            == Some("json_schema")
    {
        obj.insert(
            "response_format".to_string(),
            serde_json::json!({ "type": "json_object" }),
        );
    }

    if let Some(strict) = strict_json_schema
        && let Some(serde_json::Value::Object(response_format)) = obj.get_mut("response_format")
        && response_format.get("type").and_then(|value| value.as_str()) == Some("json_schema")
        && let Some(serde_json::Value::Object(json_schema)) = response_format.get_mut("json_schema")
    {
        json_schema.insert("strict".to_string(), serde_json::Value::Bool(strict));
    }
}

fn remove_known_aliases_for_request_override(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
) {
    match key {
        "topLogprobs" | "top_logprobs" => {
            obj.remove("topLogprobs");
            obj.remove("top_logprobs");
        }
        "serviceTier" | "service_tier" => {
            obj.remove("serviceTier");
            obj.remove("service_tier");
        }
        "reasoningEffort" | "reasoning_effort" => {
            obj.remove("reasoningEffort");
            obj.remove("reasoning_effort");
        }
        "reasoningFormat" | "reasoning_format" => {
            obj.remove("reasoningFormat");
            obj.remove("reasoning_format");
        }
        "parallelToolCalls" | "parallel_tool_calls" => {
            obj.remove("parallelToolCalls");
            obj.remove("parallel_tool_calls");
        }
        "structuredOutputs" | "structured_outputs" => {
            obj.remove("structuredOutputs");
            obj.remove("structured_outputs");
        }
        "strictJsonSchema" | "strict_json_schema" => {
            obj.remove("strictJsonSchema");
            obj.remove("strict_json_schema");
        }
        _ => {}
    }
}

impl ProviderSpec for GroqSpec {
    fn id(&self) -> &'static str {
        "groq"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_audio()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        self.chat_spec().build_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        crate::standards::openai::errors::classify_openai_compatible_http_error(
            self.id(),
            status,
            body_text,
        )
    }

    fn try_chat_url(
        &self,
        stream: bool,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        self.chat_spec().try_chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        self.chat_spec().choose_chat_transformers(req, ctx)
    }

    fn chat_before_send(
        &self,
        req: &crate::types::ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // Groq typed options: merge request-level extras, normalize AI SDK option keys, and
        // validate. The hook is always installed because config-level defaults may already be
        // present on the body via the compat adapter even when the request itself has no
        // `providerOptions.groq`.
        let extra = req
            .provider_options_map
            .get("groq")
            .cloned()
            .and_then(|val| serde_json::from_value::<GroqOptions>(val).ok())
            .map(|opts| opts.extra_params)
            .unwrap_or_default();

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();
            if let Some(obj) = out.as_object_mut() {
                for (k, v) in &extra {
                    remove_known_aliases_for_request_override(obj, k);
                    if matches!(k.as_str(), "response_format" | "tool_choice")
                        && obj.contains_key(k)
                    {
                        continue;
                    }
                    obj.insert(k.clone(), v.clone());
                }
                normalize_groq_body_option_keys(obj);
            }
            crate::providers::groq::utils::validate_groq_params(&out)?;
            Ok(out)
        };

        Some(Arc::new(hook))
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn choose_audio_transformer(&self, _ctx: &ProviderContext) -> crate::core::AudioTransformer {
        crate::core::AudioTransformer {
            transformer: Arc::new(crate::providers::groq::transformers::GroqAudioTransformer),
        }
    }
}

impl GroqSpec {
    fn chat_spec(&self) -> crate::standards::openai::chat::OpenAiChatSpec {
        let provider_adapter: Arc<dyn crate::standards::openai::compat::adapter::ProviderAdapter> =
            Arc::new(
                crate::standards::openai::compat::adapter::MetadataExtractingAdapter::new(
                    Box::new(
                        crate::standards::openai::compat::adapter::OpenAiStandardAdapter {
                            base_url: String::new(),
                        },
                    ),
                    Arc::new(crate::providers::groq::utils::extract_groq_response_metadata),
                ),
            );

        crate::standards::openai::chat::OpenAiChatStandard::with_adapters(
            Arc::new(GroqOpenAiChatAdapter),
            provider_adapter,
        )
        .create_spec("groq")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProviderContext;
    use crate::core::ProviderSpec;
    use crate::types::{ChatMessage, ChatRequest, CommonParams, FinishReason, Tool, ToolChoice};

    #[test]
    fn groq_spec_declares_audio_capability() {
        let caps = GroqSpec.capabilities();
        assert!(
            caps.supports("audio"),
            "GroqSpec must declare audio=true to pass HttpAudioExecutor capability guards"
        );
    }

    #[test]
    fn groq_adapter_preserves_response_format_json_schema() {
        use crate::types::chat::ResponseFormat;

        let spec = GroqSpec;
        let ctx = ProviderContext::new(
            "groq",
            crate::providers::groq::config::GroqConfig::DEFAULT_BASE_URL.to_string(),
            None,
            Default::default(),
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                model: "llama-3.3-70b-versatile".to_string(),
                ..Default::default()
            })
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            }))
        );
    }

    #[test]
    fn groq_spec_merges_typed_provider_options_before_send() {
        use crate::provider_options::{
            GroqOptions, GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier,
        };
        use crate::providers::groq::ext::GroqChatRequestExt;

        let spec = GroqSpec;
        let ctx = ProviderContext::new(
            "groq",
            crate::providers::groq::config::GroqConfig::DEFAULT_BASE_URL.to_string(),
            None,
            Default::default(),
        );

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                model: "llama-3.3-70b-versatile".to_string(),
                ..Default::default()
            })
            .build()
            .with_groq_options(
                GroqOptions::new()
                    .with_logprobs(true)
                    .with_top_logprobs(2)
                    .with_service_tier(GroqServiceTier::Performance)
                    .with_reasoning_effort(GroqReasoningEffort::High)
                    .with_reasoning_format(GroqReasoningFormat::Parsed),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        let hook = spec.chat_before_send(&req, &ctx).expect("before-send hook");
        let body = hook(&body).expect("hook should produce body");

        assert_eq!(body["logprobs"], serde_json::json!(true));
        assert_eq!(body["top_logprobs"], serde_json::json!(2));
        assert_eq!(body["service_tier"], serde_json::json!("performance"));
        assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(body["reasoning_format"], serde_json::json!("parsed"));
        assert!(body.get("topLogprobs").is_none());
        assert!(body.get("serviceTier").is_none());
        assert!(body.get("reasoningEffort").is_none());
        assert!(body.get("reasoningFormat").is_none());
    }

    #[test]
    fn groq_spec_keeps_stable_response_format_while_merging_typed_provider_options() {
        use crate::provider_options::{
            GroqOptions, GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier,
        };
        use crate::providers::groq::ext::GroqChatRequestExt;
        use crate::types::chat::ResponseFormat;

        let spec = GroqSpec;
        let ctx = ProviderContext::new(
            "groq",
            crate::providers::groq::config::GroqConfig::DEFAULT_BASE_URL.to_string(),
            None,
            Default::default(),
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                model: "llama-3.3-70b-versatile".to_string(),
                ..Default::default()
            })
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build()
            .with_groq_options(
                GroqOptions::new()
                    .with_logprobs(true)
                    .with_top_logprobs(2)
                    .with_service_tier(GroqServiceTier::Performance)
                    .with_reasoning_effort(GroqReasoningEffort::High)
                    .with_reasoning_format(GroqReasoningFormat::Parsed)
                    .with_parallel_tool_calls(false)
                    .with_user("groq-user-1")
                    .with_param(
                        "response_format",
                        serde_json::json!({
                            "type": "json_object"
                        }),
                    ),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        let hook = spec.chat_before_send(&req, &ctx).expect("before-send hook");
        let body = hook(&body).expect("hook should produce body");

        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            }))
        );
        assert_eq!(body["logprobs"], serde_json::json!(true));
        assert_eq!(body["top_logprobs"], serde_json::json!(2));
        assert_eq!(body["service_tier"], serde_json::json!("performance"));
        assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(body["reasoning_format"], serde_json::json!("parsed"));
        assert_eq!(body["parallel_tool_calls"], serde_json::json!(false));
        assert_eq!(body["user"], serde_json::json!("groq-user-1"));
        assert!(body.get("parallelToolCalls").is_none());
    }

    #[test]
    fn groq_adapter_preserves_tool_choice_none_with_tools() {
        let spec = GroqSpec;
        let ctx = ProviderContext::new(
            "groq",
            crate::providers::groq::config::GroqConfig::DEFAULT_BASE_URL.to_string(),
            None,
            Default::default(),
        );

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                model: "llama-3.3-70b-versatile".to_string(),
                ..Default::default()
            })
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .build()
            .with_provider_option(
                "groq",
                serde_json::json!({
                    "tool_choice": "auto"
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        let hook = spec.chat_before_send(&req, &ctx).expect("before-send hook");
        let body = hook(&body).expect("hook should produce body");
        assert_eq!(body.get("tool_choice"), Some(&serde_json::json!("none")));
    }

    #[test]
    fn groq_adapter_maps_tool_call_responses_like_openai() {
        let spec = GroqSpec;
        let ctx = ProviderContext::new(
            "groq",
            crate::providers::groq::config::GroqConfig::DEFAULT_BASE_URL.to_string(),
            None,
            Default::default(),
        );

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                model: "llama-3.3-70b-versatile".to_string(),
                ..Default::default()
            })
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let raw = serde_json::json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "llama-3.3-70b-versatile",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Tokyo\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        });

        let resp = bundle.response.transform_chat_response(&raw).unwrap();
        assert_eq!(resp.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(resp.tool_calls().len(), 1);
        let call = resp.tool_calls()[0].as_tool_call().expect("tool call");
        assert_eq!(call.tool_call_id, "call_1");
        assert_eq!(call.tool_name, "get_weather");
        assert_eq!(call.arguments, &serde_json::json!({ "city": "Tokyo" }));
    }
}
