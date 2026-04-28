use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, ResponseFormat};
use reqwest::header::HeaderMap;
use serde_json::{Map, Value};
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleRequestSettings;
use std::sync::Arc;

fn rename_field(obj: &mut Map<String, Value>, from: &str, to: &str) {
    if let Some(value) = obj.remove(from) {
        obj.entry(to.to_string()).or_insert(value);
    }
}

fn normalize_deepseek_options(value: &Value) -> Option<Map<String, Value>> {
    let mut obj = value.as_object()?.clone();

    let legacy_enable = obj
        .remove("enableReasoning")
        .or_else(|| obj.remove("enable_reasoning"))
        .and_then(|value| value.as_bool());

    obj.remove("reasoningBudget");
    obj.remove("reasoning_budget");

    if let Some(thinking) = obj
        .get_mut("thinking")
        .and_then(|value| value.as_object_mut())
    {
        rename_field(thinking, "thinkingType", "type");
        rename_field(thinking, "thinking_type", "type");
        thinking.remove("budgetTokens");
        thinking.remove("budget_tokens");
    } else if let Some(enable) = legacy_enable {
        obj.insert(
            "thinking".to_string(),
            serde_json::json!({
                "type": if enable { "enabled" } else { "disabled" }
            }),
        );
    }

    Some(obj)
}

fn custom_deepseek_provider_options(
    req: &ChatRequest,
    provider_id: &str,
) -> Option<Map<String, Value>> {
    let key = provider_id.split('.').next().unwrap_or(provider_id).trim();
    if key.is_empty() || key.eq_ignore_ascii_case("deepseek") {
        return None;
    }

    normalize_deepseek_options(req.provider_options_map.get(key)?)
}

fn merge_custom_deepseek_options(
    body_obj: &mut Map<String, Value>,
    options: &Option<Map<String, Value>>,
) {
    let Some(options) = options.as_ref() else {
        return;
    };

    body_obj.remove("enableReasoning");
    body_obj.remove("enable_reasoning");
    body_obj.remove("reasoningBudget");
    body_obj.remove("reasoning_budget");

    for (key, value) in options {
        if matches!(key.as_str(), "response_format" | "tool_choice") && body_obj.contains_key(key) {
            continue;
        }
        body_obj.insert(key.clone(), value.clone());
    }
}

fn deepseek_json_response_prompt(response_format: &ResponseFormat) -> String {
    match response_format {
        ResponseFormat::JsonObject { .. } => "Return JSON.".to_string(),
        ResponseFormat::Json { schema, .. } => {
            let schema = serde_json::to_string(schema).unwrap_or_else(|_| schema.to_string());
            format!("Return JSON that conforms to the following schema: {schema}")
        }
    }
}

fn apply_deepseek_response_format_body(
    body: &mut serde_json::Value,
    response_format: Option<&ResponseFormat>,
) {
    let Some(response_format) = response_format else {
        return;
    };
    let Some(body_obj) = body.as_object_mut() else {
        return;
    };

    body_obj.insert(
        "response_format".to_string(),
        serde_json::json!({ "type": "json_object" }),
    );

    if let Some(messages) = body_obj
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
    {
        messages.insert(
            0,
            serde_json::json!({
                "role": "system",
                "content": deepseek_json_response_prompt(response_format),
            }),
        );
    }
}

/// `DeepSeek` ProviderSpec implementation.
#[derive(Clone)]
pub struct DeepSeekSpec {
    inner: siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter,
}

impl DeepSeekSpec {
    pub fn new(
        adapter: Arc<
            dyn siumai_provider_openai_compatible::providers::openai_compatible::ProviderAdapter,
        >,
    ) -> Self {
        Self {
            inner: siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter::with_settings(
                adapter,
                OpenAiCompatibleRequestSettings {
                    include_usage: Some(true),
                    ..OpenAiCompatibleRequestSettings::default()
                },
            ),
        }
    }
}

impl ProviderSpec for DeepSeekSpec {
    fn id(&self) -> &'static str {
        "deepseek"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        self.inner.build_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        headers: &HeaderMap,
    ) -> Option<LlmError> {
        self.inner.classify_http_error(status, body_text, headers)
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        self.inner.chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        self.inner.choose_chat_transformers(req, ctx)
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        let inner_hook = self.inner.chat_before_send(req, ctx);
        let custom_options = custom_deepseek_provider_options(req, &ctx.provider_id);
        let Some(response_format) = req.response_format.clone() else {
            let Some(custom_options) = custom_options else {
                return inner_hook;
            };
            return Some(Arc::new(
                move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                    let mut out = match inner_hook.as_ref() {
                        Some(hook) => hook(body)?,
                        None => body.clone(),
                    };
                    if let Some(body_obj) = out.as_object_mut() {
                        merge_custom_deepseek_options(body_obj, &Some(custom_options.clone()));
                    }
                    Ok(out)
                },
            ));
        };

        Some(Arc::new(
            move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                let mut out = match inner_hook.as_ref() {
                    Some(hook) => hook(body)?,
                    None => body.clone(),
                };
                if let Some(body_obj) = out.as_object_mut() {
                    merge_custom_deepseek_options(body_obj, &custom_options);
                }
                apply_deepseek_response_format_body(&mut out, Some(&response_format));
                Ok(out)
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_metadata::deepseek::DeepSeekChatResponseExt;
    use crate::types::{ChatMessage, Tool, ToolChoice, chat::ResponseFormat};
    use siumai_provider_openai_compatible::providers::openai_compatible::{
        ConfigurableAdapter, get_provider_config,
    };
    use std::collections::HashMap;

    #[test]
    fn deepseek_spec_normalizes_camel_case_provider_options() {
        let provider = get_provider_config("deepseek").expect("deepseek config");
        let spec = DeepSeekSpec::new(Arc::new(ConfigurableAdapter::new(provider)));
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_provider_option(
            "deepseek",
            serde_json::json!({
                "enableReasoning": true,
                "reasoningBudget": 2048,
                "foo": "bar"
            }),
        );
        let ctx = ProviderContext::new(
            "deepseek",
            "https://api.deepseek.com",
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let hook = spec
            .chat_before_send(&request, &ctx)
            .expect("before_send hook");
        let body = hook(&serde_json::json!({ "messages": [] })).expect("hook output");

        assert_eq!(
            body["thinking"],
            serde_json::json!({
                "type": "enabled"
            })
        );
        assert_eq!(body["foo"], serde_json::json!("bar"));
        assert!(body.get("enableReasoning").is_none());
        assert!(body.get("enable_reasoning").is_none());
        assert!(body.get("reasoningBudget").is_none());
        assert!(body.get("reasoning_budget").is_none());
    }

    #[test]
    fn deepseek_spec_keeps_stable_response_format_over_raw_provider_options() {
        let provider = get_provider_config("deepseek").expect("deepseek config");
        let spec = DeepSeekSpec::new(Arc::new(ConfigurableAdapter::new(provider)));
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });
        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "thinking": {
                        "type": "enabled",
                        "budgetTokens": 2048
                    }
                }),
            );
        let ctx = ProviderContext::new(
            "deepseek",
            "https://api.deepseek.com",
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let body = bundle.request.transform_chat(&request).expect("transform");
        let hook = spec
            .chat_before_send(&request, &ctx)
            .expect("before_send hook");
        let body = hook(&body).expect("hook output");

        assert_eq!(
            body["thinking"],
            serde_json::json!({
                "type": "enabled"
            })
        );
        assert!(body["thinking"].get("budgetTokens").is_none());
        assert!(body["thinking"].get("budget_tokens").is_none());
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({ "type": "json_object" }))
        );
        assert_eq!(
            body["messages"][0],
            serde_json::json!({
                "role": "system",
                "content": format!(
                    "Return JSON that conforms to the following schema: {}",
                    serde_json::to_string(&schema).expect("schema string")
                )
            })
        );
    }

    #[test]
    fn deepseek_spec_keeps_stable_tool_choice_over_raw_provider_options() {
        let provider = get_provider_config("deepseek").expect("deepseek config");
        let spec = DeepSeekSpec::new(Arc::new(ConfigurableAdapter::new(provider)));
        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "tool_choice": "auto"
                }),
            );
        let ctx = ProviderContext::new(
            "deepseek",
            "https://api.deepseek.com",
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let body = bundle.request.transform_chat(&request).expect("transform");
        let hook = spec
            .chat_before_send(&request, &ctx)
            .expect("before_send hook");
        let body = hook(&body).expect("hook output");

        assert_eq!(body.get("tool_choice"), Some(&serde_json::json!("none")));
    }

    #[test]
    fn deepseek_spec_injects_logprobs_into_provider_metadata() {
        let provider = get_provider_config("deepseek").expect("deepseek config");
        let spec = DeepSeekSpec::new(Arc::new(ConfigurableAdapter::new(provider)));
        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();
        let ctx = ProviderContext::new(
            "deepseek",
            "https://api.deepseek.com",
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let response = bundle
            .response
            .transform_chat_response(&serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_741_392_000,
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello"
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [{
                            "token": "hello",
                            "logprob": -0.1,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }]
                    }
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2
                }
            }))
            .expect("transform response");

        assert_eq!(
            response.deepseek_metadata().and_then(|meta| meta.logprobs),
            Some(serde_json::json!([
                {
                    "token": "hello",
                    "logprob": -0.1,
                    "bytes": [104, 101, 108, 108, 111],
                    "top_logprobs": []
                }
            ]))
        );
    }

    #[test]
    fn deepseek_spec_reads_runtime_custom_provider_options_key() {
        let provider = get_provider_config("deepseek").expect("deepseek config");
        let spec = DeepSeekSpec::new(Arc::new(ConfigurableAdapter::new(provider)));
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_provider_option(
            "my-custom-deepseek",
            serde_json::json!({
                "enableReasoning": true,
                "reasoningBudget": 2048,
                "foo": "bar"
            }),
        );
        let ctx = ProviderContext::new(
            "my-custom-deepseek.chat",
            "https://api.deepseek.com",
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let hook = spec
            .chat_before_send(&request, &ctx)
            .expect("before_send hook");
        let body = hook(&serde_json::json!({ "messages": [] })).expect("hook output");

        assert_eq!(
            body["thinking"],
            serde_json::json!({
                "type": "enabled"
            })
        );
        assert_eq!(body["foo"], serde_json::json!("bar"));
    }

    #[test]
    fn deepseek_spec_uses_runtime_custom_provider_metadata_key() {
        let provider = get_provider_config("deepseek").expect("deepseek config");
        let spec = DeepSeekSpec::new(Arc::new(ConfigurableAdapter::new(provider)));
        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();
        let ctx = ProviderContext::new(
            "my-custom-deepseek.chat",
            "https://api.deepseek.com",
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let response = bundle
            .response
            .transform_chat_response(&serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_741_392_000,
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello"
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [{
                            "token": "hello",
                            "logprob": -0.1,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }]
                    }
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2
                }
            }))
            .expect("transform response");

        let root = response
            .provider_metadata
            .as_ref()
            .expect("provider metadata");
        assert!(root.get("my-custom-deepseek").is_some());
        assert!(root.get("deepseek").is_none());
        assert!(response.deepseek_metadata().is_none());
        assert!(
            response
                .deepseek_metadata_with_key("my-custom-deepseek")
                .and_then(|meta| meta.logprobs)
                .is_some()
        );
    }
}
