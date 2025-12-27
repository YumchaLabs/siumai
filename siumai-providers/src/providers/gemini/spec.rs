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
            .with_file_management()
            .with_image_generation()
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
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        crate::standards::gemini::GeminiChatStandard::new()
            .create_transformers_with_model(&ctx.provider_id, Some(&req.common_params.model))
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // Provider-specific behavior is implemented in the protocol-layer transformers (siumai-core),
        // so providers only need to support custom provider options injection here.
        crate::core::default_custom_options_hook(self.id(), req)
    }

    fn embedding_url(&self, req: &crate::types::EmbeddingRequest, ctx: &ProviderContext) -> String {
        let spec = crate::standards::gemini::GeminiEmbeddingStandard::new().create_spec("gemini");
        spec.embedding_url(req, ctx)
    }

    fn choose_embedding_transformers(
        &self,
        req: &crate::types::EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        crate::standards::gemini::GeminiEmbeddingStandard::new()
            .create_spec("gemini")
            .choose_embedding_transformers(req, ctx)
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
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> ImageTransformers {
        crate::standards::gemini::GeminiImageStandard::new()
            .create_spec("gemini")
            .choose_image_transformers(req, ctx)
    }

    fn files_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn model_url(&self, model_id: &str, ctx: &ProviderContext) -> String {
        // Gemini's model APIs accept resource-style names (e.g. "models/gemini-1.5-pro").
        // Keep behavior compatible with `GeminiModels::get_model`.
        let trimmed = model_id.trim();
        let resource = if trimmed.is_empty() {
            "models".to_string()
        } else if trimmed.starts_with("models/") || trimmed.contains('/') {
            trimmed.to_string()
        } else {
            format!("models/{trimmed}")
        };
        crate::utils::url::join_url(ctx.base_url.trim_end_matches('/'), &resource)
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

/// Gemini ProviderSpec that carries protocol config (model/generation_config/etc).
///
/// Use this when building embedding/image executors to avoid duplicating transformer wiring in
/// provider clients and to ensure request-level model overrides are reflected in the JSON body.
#[derive(Clone)]
pub struct GeminiSpecWithConfig {
    config: crate::providers::gemini::types::GeminiConfig,
}

impl GeminiSpecWithConfig {
    pub fn new(config: crate::providers::gemini::types::GeminiConfig) -> Self {
        Self { config }
    }

    fn config_with_request_model(
        &self,
        model: Option<&str>,
    ) -> crate::providers::gemini::types::GeminiConfig {
        let mut cfg = self.config.clone();
        if let Some(m) = model
            && !m.is_empty()
        {
            cfg.model = m.to_string();
            cfg.common_params.model = m.to_string();
        }
        cfg
    }
}

impl ProviderSpec for GeminiSpecWithConfig {
    fn id(&self) -> &'static str {
        "gemini"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        GeminiSpec.capabilities()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        GeminiSpec.build_headers(ctx)
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        GeminiSpec.chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        let cfg = self.config_with_request_model(Some(&req.common_params.model));

        let request_tx = Arc::new(
            crate::providers::gemini::transformers::GeminiRequestTransformer {
                config: cfg.clone(),
            },
        );
        let response_tx = Arc::new(
            crate::providers::gemini::transformers::GeminiResponseTransformer {
                config: cfg.clone(),
            },
        );

        let stream_tx = if req.stream {
            Some(Arc::new(
                crate::providers::gemini::transformers::GeminiStreamChunkTransformer {
                    provider_id: ctx.provider_id.clone(),
                    inner: crate::providers::gemini::streaming::GeminiEventConverter::new(cfg),
                },
            )
                as Arc<
                    dyn crate::execution::transformers::stream::StreamChunkTransformer,
                >)
        } else {
            None
        };

        ChatTransformers {
            request: request_tx,
            response: response_tx,
            stream: stream_tx,
            json: None,
        }
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        GeminiSpec.chat_before_send(req, ctx)
    }

    fn embedding_url(&self, req: &crate::types::EmbeddingRequest, ctx: &ProviderContext) -> String {
        GeminiSpec.embedding_url(req, ctx)
    }

    fn choose_embedding_transformers(
        &self,
        req: &crate::types::EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        let cfg = self.config_with_request_model(req.model.as_deref());
        EmbeddingTransformers {
            request: Arc::new(
                crate::providers::gemini::transformers::GeminiRequestTransformer {
                    config: cfg.clone(),
                },
            ),
            response: Arc::new(
                crate::providers::gemini::transformers::GeminiResponseTransformer { config: cfg },
            ),
        }
    }

    fn image_url(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        GeminiSpec.image_url(req, ctx)
    }

    fn choose_image_transformers(
        &self,
        req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        let cfg = self.config_with_request_model(req.model.as_deref());
        ImageTransformers {
            request: Arc::new(
                crate::providers::gemini::transformers::GeminiRequestTransformer {
                    config: cfg.clone(),
                },
            ),
            response: Arc::new(
                crate::providers::gemini::transformers::GeminiResponseTransformer { config: cfg },
            ),
        }
    }

    fn files_base_url(&self, ctx: &ProviderContext) -> String {
        GeminiSpec.files_base_url(ctx)
    }

    fn model_url(&self, model_id: &str, ctx: &ProviderContext) -> String {
        GeminiSpec.model_url(model_id, ctx)
    }

    fn choose_files_transformer(&self, _ctx: &ProviderContext) -> crate::core::FilesTransformer {
        crate::core::FilesTransformer {
            transformer: Arc::new(
                crate::providers::gemini::transformers::GeminiFilesTransformer {
                    config: self.config.clone(),
                },
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_spec_declares_file_management() {
        let caps = GeminiSpec.capabilities();
        assert!(
            caps.supports("file_management"),
            "GeminiSpec must declare file_management=true to pass HttpFilesExecutor capability guards"
        );
    }

    #[test]
    fn embedding_wrapper_selects_single_vs_batch_url() {
        let base = "https://example/v1".to_string();
        let model = "gemini-embedding-001".to_string();
        let spec = crate::standards::gemini::GeminiEmbeddingStandard::new().create_spec("gemini");

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
        let spec = crate::standards::gemini::GeminiImageStandard::new().create_spec("gemini");
        let ctx = ProviderContext::new(
            "gemini",
            base.clone(),
            Some("KEY".into()),
            std::collections::HashMap::new(),
        );

        let req = crate::types::ImageGenerationRequest {
            model: Some(model.clone()),
            ..Default::default()
        };
        let url = spec.image_url(&req, &ctx);
        assert!(url.ends_with(&format!("models/{}:generateContent", model)));
    }

    #[test]
    fn gemini_headers_use_api_key_without_bearer() {
        let base = "https://example".to_string();
        let spec = crate::standards::gemini::GeminiImageStandard::new().create_spec("gemini");
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
        let spec = crate::standards::gemini::GeminiEmbeddingStandard::new().create_spec("gemini");
        let mut extra = std::collections::HashMap::new();
        extra.insert("Authorization".into(), "Bearer token".into());
        let ctx = ProviderContext::new("gemini", base, Some("APIKEY".into()), extra);
        let headers = spec.build_headers(&ctx).unwrap();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer token");
        assert!(headers.get("x-goog-api-key").is_none());
    }
}
