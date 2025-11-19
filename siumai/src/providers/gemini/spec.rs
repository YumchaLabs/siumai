#[cfg(feature = "std-gemini-external")]
use crate::core::provider_spec::{
    bridge_core_chat_transformers, bridge_core_embedding_transformers,
    bridge_core_image_transformers, map_core_stream_event_with_provider,
};
use crate::core::{
    ChatTransformers, EmbeddingTransformers, ImageTransformers, ProviderContext, ProviderSpec,
};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
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
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        crate::execution::http::headers::ProviderHeaders::gemini(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = &req.common_params.model;
        if stream {
            format!("{}/models/{}:streamGenerateContent?alt=sse", base, model)
        } else {
            format!("{}/models/{}:generateContent", base, model)
        }
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        #[cfg(feature = "std-gemini-external")]
        {
            use siumai_core::provider_spec::CoreChatTransformers;
            use siumai_std_gemini::gemini::chat::{GeminiChatStandard, GeminiDefaultChatAdapter};

            let std =
                GeminiChatStandard::with_adapter(Arc::new(GeminiDefaultChatAdapter::default()));
            let core_txs: CoreChatTransformers = CoreChatTransformers {
                request: std.create_request_transformer(&ctx.provider_id),
                response: std.create_response_transformer(&ctx.provider_id),
                stream: Some(std.create_stream_converter(&ctx.provider_id)),
            };

            return bridge_core_chat_transformers(
                core_txs,
                crate::core::provider_spec::gemini_like_chat_request_to_core_input,
                |evt| map_core_stream_event_with_provider("gemini", evt),
            );
        }

        #[cfg(not(feature = "std-gemini-external"))]
        {
            // Without std-gemini, Gemini provider is not fully supported.
            // Keep a minimal unsupported transformer to avoid panics.
            struct UnsupportedReq;
            impl crate::execution::transformers::request::RequestTransformer for UnsupportedReq {
                fn provider_id(&self) -> &str {
                    "gemini"
                }
                fn transform_chat(
                    &self,
                    _req: &crate::types::ChatRequest,
                ) -> Result<serde_json::Value, LlmError> {
                    Err(LlmError::UnsupportedOperation(
                        "Gemini chat requires std-gemini-external feature".into(),
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
                request: Arc::new(UnsupportedReq),
                response: Arc::new(UnsupportedResp),
                stream: None,
                json: None,
            }
        }
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

        // Gemini-specific typed options are now mapped via `gemini_like_chat_request_to_core_input`
        // → `ChatInput::extra` → `GeminiDefaultChatAdapter`. Only CustomProviderOptions
        // are handled here.
        None
    }

    fn embedding_url(&self, req: &crate::types::EmbeddingRequest, ctx: &ProviderContext) -> String {
        // Default behavior remains: choose between embedContent and batchEmbedContents
        // based on the number of inputs.
        let base = ctx.base_url.trim_end_matches('/');
        let model = req.model.as_deref().unwrap_or("");
        if req.input.len() == 1 {
            format!("{}/models/{}:embedContent", base, model)
        } else {
            format!("{}/models/{}:batchEmbedContents", base, model)
        }
    }

    fn choose_embedding_transformers(
        &self,
        _req: &crate::types::EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        #[cfg(feature = "std-gemini-external")]
        {
            let std = siumai_std_gemini::gemini::embedding::GeminiEmbeddingStandard::new();
            let req_tx = std.create_request_transformer("gemini");
            let resp_tx = std.create_response_transformer("gemini");
            return bridge_core_embedding_transformers(req_tx, resp_tx);
        }

        #[cfg(not(feature = "std-gemini-external"))]
        {
            crate::standards::gemini::GeminiEmbeddingStandard::new().create_transformers()
        }
    }

    fn image_url(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = req.model.as_deref().unwrap_or("");
        format!("{}/models/{}:generateContent", base, model)
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        #[cfg(feature = "std-gemini-external")]
        {
            let std = siumai_std_gemini::gemini::image::GeminiImageStandard::new();
            let req_tx = std.create_request_transformer("gemini");
            let resp_tx = std.create_response_transformer("gemini");
            return bridge_core_image_transformers(req_tx, resp_tx);
        }

        #[cfg(not(feature = "std-gemini-external"))]
        {
            crate::standards::gemini::GeminiImageStandard::new().create_transformers()
        }
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
    use crate::types::{ChatMessage, GeminiOptions};

    #[test]
    fn chat_before_send_injects_file_search_tool() {
        // Build a ChatRequest with GeminiOptions.file_search configured
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_gemini_options(
            GeminiOptions::new().with_file_search_store_names(vec![
                "stores/foo".to_string(),
                "stores/bar".to_string(),
            ]),
        );

        // Minimal provider context
        let ctx = crate::core::ProviderContext::new(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta".to_string(),
            None,
            std::collections::HashMap::new(),
        );

        let spec = GeminiSpec;
        let hook = spec
            .chat_before_send(&req, &ctx)
            .expect("expected before_send hook");

        let base = serde_json::json!({
            "model": "gemini-2.0-flash-exp",
            "contents": []
        });

        let out = hook(&base).expect("hook apply ok");
        let tools = out
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .expect("tools array present");

        let fs = tools
            .iter()
            .find_map(|t| t.get("file_search"))
            .cloned()
            .expect("file_search tool present");

        let names = fs
            .get("file_search_store_names")
            .and_then(|v| v.as_array())
            .cloned()
            .expect("store names present");

        let names: Vec<String> = names
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        assert_eq!(
            names,
            vec!["stores/foo".to_string(), "stores/bar".to_string()]
        );
    }

    #[test]
    fn embedding_wrapper_selects_single_vs_batch_url() {
        let base = "https://example/v1".to_string();
        let model = "gemini-embedding-001".to_string();
        let spec = create_embedding_wrapper(base.clone(), model.clone());

        let ctx = ProviderContext::new(
            "gemini",
            base.clone(),
            Some("KEY".into()),
            std::collections::HashMap::new(),
        );

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
        let ctx = ProviderContext::new(
            "gemini",
            base.clone(),
            Some("KEY".into()),
            std::collections::HashMap::new(),
        );

        let req = crate::types::ImageGenerationRequest::default();
        let url = spec.image_url(&req, &ctx);
        assert!(url.ends_with(&format!("models/{}:generateContent", model)));
    }

    #[test]
    fn gemini_headers_use_api_key_without_bearer() {
        let base = "https://example".to_string();
        let model = "gemini-1.5-flash".to_string();
        let spec = create_image_wrapper(base.clone(), model.clone());
        let ctx = ProviderContext::new(
            "gemini",
            base,
            Some("APIKEY".into()),
            std::collections::HashMap::new(),
        );
        let headers = spec.build_headers(&ctx).unwrap();
        assert_eq!(headers.get("x-goog-api-key").unwrap(), "APIKEY");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn gemini_headers_skip_api_key_with_bearer() {
        let base = "https://example".to_string();
        let model = "gemini-1.5-flash".to_string();
        let spec = create_embedding_wrapper(base.clone(), model.clone());
        let mut extra = std::collections::HashMap::new();
        extra.insert("Authorization".into(), "Bearer token".into());
        let ctx = ProviderContext::new("gemini", base, Some("APIKEY".into()), extra);
        let headers = spec.build_headers(&ctx).unwrap();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer token");
        assert!(headers.get("x-goog-api-key").is_none());
    }
}
