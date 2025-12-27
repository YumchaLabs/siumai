use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
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
        if req.stream && let Some(obj) = body.as_object_mut() {
            obj.remove("stream_options");
        }

        // Groq uses `max_tokens` (OpenAI-style chat completions) rather than
        // `max_completion_tokens`.
        if let Some(obj) = body.as_object_mut() && let Some(v) = obj.remove("max_completion_tokens")
        {
            obj.entry("max_tokens".to_string()).or_insert(v);
        }

        Ok(())
    }
}

/// Groq ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct GroqSpec;

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

    fn chat_url(
        &self,
        stream: bool,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        self.chat_spec().chat_url(stream, req, ctx)
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
        use crate::types::ProviderOptions;

        // Preserve Custom provider options behavior.
        if let Some(custom_hook) = crate::core::default_custom_options_hook(self.id(), req) {
            let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                let out = custom_hook(body)?;
                crate::providers::groq::utils::validate_groq_params(&out)?;
                Ok(out)
            };
            return Some(Arc::new(hook));
        }

        // Groq typed options: merge extra params and validate.
        if let ProviderOptions::Groq(opts) = &req.provider_options {
            let extra = opts.extra_params.clone();
            let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                let mut out = body.clone();
                if let Some(obj) = out.as_object_mut() {
                    for (k, v) in &extra {
                        obj.insert(k.clone(), v.clone());
                    }
                }
                crate::providers::groq::utils::validate_groq_params(&out)?;
                Ok(out)
            };
            return Some(Arc::new(hook));
        }

        None
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
        crate::standards::openai::chat::OpenAiChatStandard::with_adapter(Arc::new(
            GroqOpenAiChatAdapter,
        ))
        .create_spec("groq")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groq_spec_declares_audio_capability() {
        let caps = GroqSpec.capabilities();
        assert!(
            caps.supports("audio"),
            "GroqSpec must declare audio=true to pass HttpAudioExecutor capability guards"
        );
    }
}
