use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::provider_options::DeepSeekOptions;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use reqwest::header::HeaderMap;
use serde_json::{Map, Value};
use std::sync::Arc;

fn rename_field(obj: &mut Map<String, Value>, from: &str, to: &str) {
    if let Some(value) = obj.remove(from) {
        obj.entry(to.to_string()).or_insert(value);
    }
}

fn normalize_deepseek_options(value: &Value) -> Option<Map<String, Value>> {
    let mut obj = value.as_object()?.clone();
    rename_field(&mut obj, "enableReasoning", "enable_reasoning");
    rename_field(&mut obj, "reasoningBudget", "reasoning_budget");

    let normalized = serde_json::from_value::<DeepSeekOptions>(Value::Object(obj.clone()))
        .ok()
        .and_then(|options| serde_json::to_value(options).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or(obj);

    Some(normalized)
}

fn deepseek_provider_options_hook(
    req: &ChatRequest,
) -> Option<crate::execution::executors::BeforeSendHook> {
    let normalized = normalize_deepseek_options(req.provider_options_map.get("deepseek")?)?;
    let hook = move |body: &Value| -> Result<Value, LlmError> {
        let mut out = body.clone();
        if let Some(body_obj) = out.as_object_mut() {
            for (key, value) in &normalized {
                if matches!(key.as_str(), "response_format" | "tool_choice")
                    && body_obj.contains_key(key)
                {
                    continue;
                }
                body_obj.insert(key.clone(), value.clone());
            }
        }
        Ok(out)
    };
    Some(Arc::new(hook))
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
            inner: siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter::new(adapter),
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
        deepseek_provider_options_hook(req).or_else(|| self.inner.chat_before_send(req, ctx))
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

        assert_eq!(body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(body["reasoning_budget"], serde_json::json!(2048));
        assert_eq!(body["foo"], serde_json::json!("bar"));
        assert!(body.get("enableReasoning").is_none());
        assert!(body.get("reasoningBudget").is_none());
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
                    "reasoningBudget": 2048
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

        assert_eq!(body["reasoning_budget"], serde_json::json!(2048));
        assert!(body.get("reasoningBudget").is_none());
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
}
