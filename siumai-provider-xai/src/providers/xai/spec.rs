use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, ProviderOptions};
use reqwest::header::HeaderMap;
use std::borrow::Cow;
use std::sync::Arc;

/// xAI ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct XaiSpec;

#[derive(Clone, Copy, Default)]
struct XaiOpenAiChatAdapter;

impl crate::standards::openai::chat::OpenAiChatAdapter for XaiOpenAiChatAdapter {
    fn build_headers(
        &self,
        api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        if api_key.is_empty() {
            return Err(LlmError::MissingApiKey("xAI API key not provided".into()));
        }
        Ok(())
    }

    fn transform_request(
        &self,
        req: &ChatRequest,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        // xAI does not consistently support the "developer" role; treat it as "system".
        if let Some(msgs) = body.get_mut("messages").and_then(|v| v.as_array_mut()) {
            for m in msgs {
                if m.get("role").and_then(|v| v.as_str()) == Some("developer") {
                    m["role"] = serde_json::Value::String("system".to_string());
                }
            }
        }

        // Keep behavior aligned with the legacy xAI implementation: do not send stream_options.
        if req.stream && let Some(obj) = body.as_object_mut() {
            obj.remove("stream_options");
        }

        // xAI uses `max_tokens` for Chat Completions; map `max_completion_tokens` when present.
        if let Some(obj) = body.as_object_mut() && let Some(v) = obj.remove("max_completion_tokens")
        {
            obj.entry("max_tokens".to_string()).or_insert(v);
        }

        // xAI provider options (Vercel-aligned):
        // - search parameters are sent as `search_parameters` (snake_case) in the request body
        // - reasoning effort is sent as `reasoning_effort`
        if let ProviderOptions::Xai(options) = &req.provider_options {
            if let Some(params) = &options.search_parameters
                && let Ok(val) = serde_json::to_value(params)
            {
                body["search_parameters"] = val;
            }

            if let Some(effort) = &options.reasoning_effort {
                body["reasoning_effort"] = serde_json::Value::String(effort.clone());
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct XaiOpenAiCompatAdapter {
    base_url: String,
}

impl crate::standards::openai::compat::adapter::ProviderAdapter for XaiOpenAiCompatAdapter {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("xai")
    }

    fn transform_request_params(
        &self,
        _params: &mut serde_json::Value,
        _model: &str,
        _request_type: crate::standards::openai::compat::types::RequestType,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn get_field_mappings(
        &self,
        _model: &str,
    ) -> crate::standards::openai::compat::types::FieldMappings {
        // xAI returns reasoning in `reasoning_content` (OpenAI-compatible extensions).
        crate::standards::openai::compat::types::FieldMappings::deepseek()
    }

    fn get_model_config(
        &self,
        _model: &str,
    ) -> crate::standards::openai::compat::types::ModelConfig {
        crate::standards::openai::compat::types::ModelConfig::default()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn compatibility(&self) -> crate::standards::openai::compat::adapter::ProviderCompatibility {
        crate::standards::openai::compat::adapter::ProviderCompatibility {
            supports_array_content: true,
            supports_stream_options: false,
            supports_developer_role: false,
            supports_enable_thinking: false,
            supports_service_tier: false,
            force_streaming_models: vec![],
            custom_flags: Default::default(),
        }
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn clone_adapter(&self) -> Box<dyn crate::standards::openai::compat::adapter::ProviderAdapter> {
        Box::new(self.clone())
    }
}

impl ProviderSpec for XaiSpec {
    fn id(&self) -> &'static str {
        "xai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        self.chat_spec(ctx).build_headers(ctx)
    }

    fn chat_url(
        &self,
        stream: bool,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        self.chat_spec(ctx).chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        self.chat_spec(ctx).choose_chat_transformers(req, ctx)
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        self.chat_spec(ctx).chat_before_send(req, ctx)
    }
}

impl XaiSpec {
    fn chat_spec(&self, ctx: &ProviderContext) -> crate::standards::openai::chat::OpenAiChatSpec {
        let provider_adapter = Arc::new(XaiOpenAiCompatAdapter {
            base_url: ctx.base_url.clone(),
        });
        crate::standards::openai::chat::OpenAiChatStandard::with_adapters(
            Arc::new(XaiOpenAiChatAdapter),
            provider_adapter,
        )
        .create_spec("xai")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_xai_chat_transformer_maps_developer_role_to_system() {
        let spec = XaiSpec;
        let ctx = ProviderContext::new(
            "xai",
            "https://api.x.ai/v1".to_string(),
            None,
            Default::default(),
        );

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::developer("dev-msg").build()])
            .common_params(CommonParams {
                model: "grok-3-latest".to_string(),
                ..Default::default()
            })
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        assert_eq!(body["messages"][0]["role"], "system");
    }

    #[test]
    fn test_xai_chat_transformer_omits_stream_options() {
        let spec = XaiSpec;
        let ctx = ProviderContext::new(
            "xai",
            "https://api.x.ai/v1".to_string(),
            None,
            Default::default(),
        );

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                model: "grok-3-latest".to_string(),
                ..Default::default()
            })
            .stream(true)
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        assert!(body.get("stream_options").is_none());
        assert_eq!(body.get("stream").and_then(|v| v.as_bool()), Some(true));
    }

    #[test]
    fn test_xai_chat_transformer_uses_max_tokens() {
        let spec = XaiSpec;
        let ctx = ProviderContext::new(
            "xai",
            "https://api.x.ai/v1".to_string(),
            None,
            Default::default(),
        );

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                model: "grok-3-latest".to_string(),
                max_completion_tokens: Some(123),
                ..Default::default()
            })
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        assert!(body.get("max_completion_tokens").is_none());
        assert_eq!(body.get("max_tokens").and_then(|v| v.as_u64()), Some(123));
    }

    #[test]
    fn test_xai_injects_search_parameters_via_adapter() {
        let spec = XaiSpec;
        let ctx = ProviderContext::new(
            "xai",
            "https://api.x.ai/v1".to_string(),
            None,
            Default::default(),
        );

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                model: "grok-3-latest".to_string(),
                ..Default::default()
            })
            .xai_options(XaiOptions::new().with_search(XaiSearchParameters::default()))
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        assert!(body.get("search_parameters").is_some());
    }

    #[test]
    fn test_xai_chat_response_extracts_reasoning_content() {
        let spec = XaiSpec;
        let ctx = ProviderContext::new(
            "xai",
            "https://api.x.ai/v1".to_string(),
            None,
            Default::default(),
        );

        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                model: "grok-3-latest".to_string(),
                ..Default::default()
            })
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);

        let raw = serde_json::json!({
            "id": "chatcmpl_123",
            "object": "chat.completion",
            "created": 0,
            "model": "grok-3-latest",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning_content": "thinking..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "completion_tokens_details": { "reasoning_tokens": 4 }
            }
        });

        let resp = bundle.response.transform_chat_response(&raw).unwrap();
        assert_eq!(resp.id.as_deref(), Some("chatcmpl_123"));
        assert_eq!(resp.model.as_deref(), Some("grok-3-latest"));

        let parts = resp
            .content
            .as_multimodal()
            .expect("expected multimodal content");
        assert!(
            parts
                .iter()
                .any(|p| matches!(p, ContentPart::Text { text } if text == "hello"))
        );
        assert!(
            parts
                .iter()
                .any(|p| matches!(p, ContentPart::Reasoning { text } if text == "thinking..."))
        );
        assert_eq!(
            resp.usage
                .and_then(|u| u.completion_tokens_details)
                .and_then(|d| d.reasoning_tokens),
            Some(4)
        );
    }
}
