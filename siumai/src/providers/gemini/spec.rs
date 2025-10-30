use crate::core::{
    ChatTransformers, EmbeddingTransformers, ImageTransformers, ProviderContext, ProviderSpec,
};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, ProviderOptions};
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Gemini ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct GeminiSpec;

impl ProviderSpec for GeminiSpec {
    fn id(&self) -> &'static str {
        "gemini"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
            .with_vision()
            .with_custom_feature("image_generation", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        // Delegate to standard headers with adapter hook capability
        let spec = crate::standards::gemini::GeminiChatStandard::new().create_spec("gemini");
        spec.build_headers(ctx)
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        // Delegate to standard spec for URL decision
        let spec = crate::standards::gemini::GeminiChatStandard::new().create_spec("gemini");
        spec.chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        crate::standards::gemini::GeminiChatStandard::new().create_transformers(&ctx.provider_id)
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // 1. First check for CustomProviderOptions (using default implementation)
        if let Some(hook) = crate::core::default_custom_options_hook(self.id(), req) {
            return Some(hook);
        }

        // 2. Handle Gemini-specific options (code_execution, search_grounding)
        // 🎯 Extract Gemini-specific options from provider_options
        let (code_execution, search_grounding, response_mime_type) =
            if let ProviderOptions::Gemini(ref options) = req.provider_options {
                (
                    options.code_execution.clone(),
                    options.search_grounding.clone(),
                    options.response_mime_type.clone(),
                )
            } else {
                return None;
            };

        // Check if we have anything to inject
        if code_execution.is_none() && search_grounding.is_none() && response_mime_type.is_none() {
            return None;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();

            // 🎯 Inject code execution tool
            // According to Gemini API, code execution is enabled via tools array
            if let Some(ref code_exec) = code_execution
                && code_exec.enabled
            {
                let mut tools = out
                    .get("tools")
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();

                // Add code execution tool
                tools.push(serde_json::json!({
                    "code_execution": {}
                }));

                out["tools"] = serde_json::Value::Array(tools);
            }

            // 🎯 Inject search grounding (Google Search)
            // According to Gemini API, search grounding is enabled via tools array
            if let Some(ref search) = search_grounding
                && search.enabled
            {
                let mut tools = out
                    .get("tools")
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();

                let mut google_search_tool = serde_json::json!({
                    "google_search": {}
                });

                // Add dynamic retrieval config if specified
                if let Some(ref dynamic_config) = search.dynamic_retrieval_config
                    && let Ok(config_json) = serde_json::to_value(dynamic_config)
                {
                    google_search_tool["google_search"]["dynamic_retrieval_config"] = config_json;
                }

                tools.push(google_search_tool);
                out["tools"] = serde_json::Value::Array(tools);
            }

            // 🎯 Inject response MIME type into generation_config
            if let Some(ref mime) = response_mime_type {
                if let Some(obj) = out
                    .get_mut("generation_config")
                    .and_then(|v| v.as_object_mut())
                {
                    obj.insert("response_mime_type".to_string(), serde_json::json!(mime));
                } else {
                    out["generation_config"] = serde_json::json!({
                        "response_mime_type": mime
                    });
                }
            }

            Ok(out)
        };
        Some(Arc::new(hook))
    }

    fn embedding_url(&self, req: &crate::types::EmbeddingRequest, ctx: &ProviderContext) -> String {
        let spec = crate::standards::gemini::GeminiEmbeddingStandard::new().create_spec("gemini");
        spec.embedding_url(req, ctx)
    }

    fn choose_embedding_transformers(
        &self,
        _req: &crate::types::EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        crate::standards::gemini::GeminiEmbeddingStandard::new().create_transformers()
    }

    fn image_url(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        let spec = crate::standards::gemini::GeminiImageStandard::new().create_spec("gemini");
        spec.image_url(req, ctx)
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        crate::standards::gemini::GeminiImageStandard::new().create_transformers()
    }

    fn files_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn choose_files_transformer(&self, _ctx: &ProviderContext) -> crate::core::FilesTransformer {
        crate::core::FilesTransformer {
            transformer: Arc::new(
                crate::providers::gemini::transformers::GeminiFilesTransformer {
                    config: crate::providers::gemini::types::GeminiConfig::default(),
                },
            ),
        }
    }
}

// Lightweight ProviderSpec wrappers with explicit URL decisions for embedding/image

struct GeminiEmbeddingSpecWrapper {
    base_url: String,
    model: String,
}

impl ProviderSpec for GeminiEmbeddingSpecWrapper {
    fn id(&self) -> &'static str {
        "gemini"
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_embedding()
    }
    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        crate::execution::http::headers::ProviderHeaders::gemini(api_key, &ctx.http_extra_headers)
    }
    fn chat_url(&self, _s: bool, _r: &ChatRequest, _c: &ProviderContext) -> String {
        self.base_url.clone()
    }
    fn choose_chat_transformers(&self, _r: &ChatRequest, _c: &ProviderContext) -> ChatTransformers {
        struct UnsupportedReq;
        impl crate::execution::transformers::request::RequestTransformer for UnsupportedReq {
            fn provider_id(&self) -> &str {
                "gemini"
            }
            fn transform_chat(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, crate::error::LlmError> {
                Err(crate::error::LlmError::UnsupportedOperation(
                    "gemini embedding wrapper does not support chat".into(),
                ))
            }
        }
        struct UnsupportedResp;
        impl crate::execution::transformers::response::ResponseTransformer for UnsupportedResp {
            fn provider_id(&self) -> &str {
                "gemini"
            }
        }
        ChatTransformers {
            request: std::sync::Arc::new(UnsupportedReq),
            response: std::sync::Arc::new(UnsupportedResp),
            stream: None,
            json: None,
        }
    }
    fn embedding_url(
        &self,
        req: &crate::types::EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> String {
        if req.input.len() == 1 {
            crate::utils::url::join_url(
                &self.base_url,
                &format!("models/{}:embedContent", self.model),
            )
        } else {
            crate::utils::url::join_url(
                &self.base_url,
                &format!("models/{}:batchEmbedContents", self.model),
            )
        }
    }
    fn choose_embedding_transformers(
        &self,
        _r: &crate::types::EmbeddingRequest,
        _c: &ProviderContext,
    ) -> EmbeddingTransformers {
        struct UnsupportedReq;
        impl crate::execution::transformers::request::RequestTransformer for UnsupportedReq {
            fn provider_id(&self) -> &str {
                "gemini"
            }
            fn transform_embedding(
                &self,
                _req: &crate::types::EmbeddingRequest,
            ) -> Result<serde_json::Value, crate::error::LlmError> {
                Err(crate::error::LlmError::UnsupportedOperation(
                    "gemini wrapper does not expose embedding transformers here".into(),
                ))
            }
            fn transform_chat(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, crate::error::LlmError> {
                Err(crate::error::LlmError::UnsupportedOperation(
                    "gemini embedding wrapper does not support chat".into(),
                ))
            }
        }
        struct UnsupportedResp;
        impl crate::execution::transformers::response::ResponseTransformer for UnsupportedResp {
            fn provider_id(&self) -> &str {
                "gemini"
            }
        }
        EmbeddingTransformers {
            request: std::sync::Arc::new(UnsupportedReq),
            response: std::sync::Arc::new(UnsupportedResp),
        }
    }
}

struct GeminiImageSpecWrapper {
    base_url: String,
    model: String,
}

impl ProviderSpec for GeminiImageSpecWrapper {
    fn id(&self) -> &'static str {
        "gemini"
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_vision()
    }
    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        crate::execution::http::headers::ProviderHeaders::gemini(api_key, &ctx.http_extra_headers)
    }
    fn chat_url(&self, _s: bool, _r: &ChatRequest, _c: &ProviderContext) -> String {
        self.base_url.clone()
    }
    fn choose_chat_transformers(&self, _r: &ChatRequest, _c: &ProviderContext) -> ChatTransformers {
        struct UnsupportedReq;
        impl crate::execution::transformers::request::RequestTransformer for UnsupportedReq {
            fn provider_id(&self) -> &str {
                "gemini"
            }
            fn transform_chat(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, crate::error::LlmError> {
                Err(crate::error::LlmError::UnsupportedOperation(
                    "gemini image wrapper does not support chat".into(),
                ))
            }
        }
        struct UnsupportedResp;
        impl crate::execution::transformers::response::ResponseTransformer for UnsupportedResp {
            fn provider_id(&self) -> &str {
                "gemini"
            }
        }
        ChatTransformers {
            request: std::sync::Arc::new(UnsupportedReq),
            response: std::sync::Arc::new(UnsupportedResp),
            stream: None,
            json: None,
        }
    }
    fn image_url(
        &self,
        _r: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> String {
        crate::utils::url::join_url(
            &self.base_url,
            &format!("models/{}:generateContent", self.model),
        )
    }
    fn choose_image_transformers(
        &self,
        _r: &crate::types::ImageGenerationRequest,
        _c: &ProviderContext,
    ) -> ImageTransformers {
        struct UnsupportedReq;
        impl crate::execution::transformers::request::RequestTransformer for UnsupportedReq {
            fn provider_id(&self) -> &str {
                "gemini"
            }
            fn transform_image(
                &self,
                _req: &crate::types::ImageGenerationRequest,
            ) -> Result<serde_json::Value, crate::error::LlmError> {
                Err(crate::error::LlmError::UnsupportedOperation(
                    "gemini wrapper does not expose image transformers here".into(),
                ))
            }
            fn transform_chat(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, crate::error::LlmError> {
                Err(crate::error::LlmError::UnsupportedOperation(
                    "gemini image wrapper does not support chat".into(),
                ))
            }
        }
        struct UnsupportedResp;
        impl crate::execution::transformers::response::ResponseTransformer for UnsupportedResp {
            fn provider_id(&self) -> &str {
                "gemini"
            }
        }
        ImageTransformers {
            request: std::sync::Arc::new(UnsupportedReq),
            response: std::sync::Arc::new(UnsupportedResp),
        }
    }
}

/// Create an embedding wrapper spec with explicit URL behavior.
pub fn create_embedding_wrapper(base_url: String, model: String) -> Arc<dyn ProviderSpec> {
    Arc::new(GeminiEmbeddingSpecWrapper { base_url, model })
}

/// Create an image wrapper spec with explicit URL behavior.
pub fn create_image_wrapper(base_url: String, model: String) -> Arc<dyn ProviderSpec> {
    Arc::new(GeminiImageSpecWrapper { base_url, model })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn embedding_wrapper_selects_single_vs_batch_url() {
        let base = "https://example/v1".to_string();
        let model = "gemini-embedding-001".to_string();
        let spec = create_embedding_wrapper(base.clone(), model.clone());

        let ctx = ProviderContext::new("gemini", base.clone(), Some("KEY".into()), HashMap::new());

        let single = crate::types::EmbeddingRequest::new(vec!["hello".to_string()])
            .with_model(model.clone());
        let url_single = spec.embedding_url(&single, &ctx);
        assert!(url_single.ends_with(&format!("models/{}:embedContent", model)));

        let batch = crate::types::EmbeddingRequest::new(vec!["a".into(), "b".into()])
            .with_model(model.clone());
        let url_batch = spec.embedding_url(&batch, &ctx);
        assert!(url_batch.ends_with(&format!("models/{}:batchEmbedContents", model)));
    }

    #[test]
    fn image_wrapper_uses_generate_content_url() {
        let base = "https://example/v1".to_string();
        let model = "gemini-1.5-flash".to_string();
        let spec = create_image_wrapper(base.clone(), model.clone());
        let ctx = ProviderContext::new("gemini", base.clone(), Some("KEY".into()), HashMap::new());

        let req = crate::types::ImageGenerationRequest::default();
        let url = spec.image_url(&req, &ctx);
        assert!(url.ends_with(&format!("models/{}:generateContent", model)));
    }

    #[test]
    fn gemini_headers_use_api_key_without_bearer() {
        let base = "https://example".to_string();
        let model = "gemini-1.5-flash".to_string();
        let spec = create_image_wrapper(base.clone(), model.clone());
        let ctx = ProviderContext::new("gemini", base, Some("APIKEY".into()), HashMap::new());
        let headers = spec.build_headers(&ctx).unwrap();
        assert_eq!(headers.get("x-goog-api-key").unwrap(), "APIKEY");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn gemini_headers_skip_api_key_with_bearer() {
        let base = "https://example".to_string();
        let model = "gemini-1.5-flash".to_string();
        let spec = create_embedding_wrapper(base.clone(), model.clone());
        let mut extra = HashMap::new();
        extra.insert("Authorization".into(), "Bearer token".into());
        let ctx = ProviderContext::new("gemini", base, Some("APIKEY".into()), extra);
        let headers = spec.build_headers(&ctx).unwrap();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer token");
        assert!(headers.get("x-goog-api-key").is_none());
    }
}
