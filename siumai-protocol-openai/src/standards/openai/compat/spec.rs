use crate::core::{
    ChatTransformers, EmbeddingTransformers, ImageTransformers, ProviderContext, ProviderSpec,
    RerankTransformers,
};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, RerankRequest};
use reqwest::header::HeaderMap;
use std::sync::Arc;

fn provider_options_map_merge_hook(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
) -> Option<crate::execution::executors::BeforeSendHook> {
    let value = map.get(provider_id)?;
    let mut obj = value.as_object()?.clone();

    fn rename_field(obj: &mut serde_json::Map<String, serde_json::Value>, from: &str, to: &str) {
        if let Some(v) = obj.remove(from) {
            obj.entry(to.to_string()).or_insert(v);
        }
    }

    fn normalize_xai_search_parameters(v: &mut serde_json::Value) {
        let Some(obj) = v.as_object_mut() else {
            return;
        };

        rename_field(obj, "returnCitations", "return_citations");
        rename_field(obj, "maxSearchResults", "max_search_results");
        rename_field(obj, "fromDate", "from_date");
        rename_field(obj, "toDate", "to_date");

        if let Some(arr) = obj.get_mut("sources").and_then(|v| v.as_array_mut()) {
            for src in arr {
                let Some(src_obj) = src.as_object_mut() else {
                    continue;
                };

                rename_field(src_obj, "allowedWebsites", "allowed_websites");
                rename_field(src_obj, "excludedWebsites", "excluded_websites");
                rename_field(src_obj, "safeSearch", "safe_search");

                rename_field(src_obj, "excludedXHandles", "excluded_x_handles");
                rename_field(src_obj, "includedXHandles", "included_x_handles");
                rename_field(src_obj, "postFavoriteCount", "post_favorite_count");
                rename_field(src_obj, "postViewCount", "post_view_count");
                rename_field(src_obj, "xHandles", "x_handles");
            }
        }
    }

    if provider_id == "xai" {
        rename_field(&mut obj, "reasoningEffort", "reasoning_effort");
        rename_field(&mut obj, "searchParameters", "search_parameters");
        if let Some(v) = obj.get_mut("search_parameters") {
            normalize_xai_search_parameters(v);
        }
    }
    let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
        let mut out = body.clone();
        if let Some(body_obj) = out.as_object_mut() {
            for (k, v) in &obj {
                body_obj.insert(k.clone(), v.clone());
            }
        }
        Ok(out)
    };
    Some(Arc::new(hook))
}

/// OpenAI-Compatible ProviderSpec implementation with an injected adapter.
///
/// This is used by OpenAI-compatible clients to avoid runtime global registry lookups.
#[derive(Clone)]
pub struct OpenAiCompatibleSpecWithAdapter {
    adapter: Arc<dyn super::adapter::ProviderAdapter>,
}

impl OpenAiCompatibleSpecWithAdapter {
    pub fn new(adapter: Arc<dyn super::adapter::ProviderAdapter>) -> Self {
        Self { adapter }
    }
}

impl ProviderSpec for OpenAiCompatibleSpecWithAdapter {
    fn id(&self) -> &'static str {
        "openai_compatible"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Prefer adapter-declared capabilities.
        self.adapter.capabilities()
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
        let provider_id = self.adapter.provider_id();
        crate::standards::openai::errors::classify_openai_compatible_http_error(
            provider_id.as_ref(),
            status,
            body_text,
        )
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        self.chat_spec().chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        self.chat_spec().choose_chat_transformers(req, ctx)
    }

    fn chat_before_send(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // For OpenAI-compatible vendors, provider options are keyed by the runtime provider id
        // (e.g. "deepseek") and are merged directly into the OpenAI-compatible request body.
        provider_options_map_merge_hook(&ctx.provider_id, &req.provider_options_map)
    }

    fn choose_embedding_transformers(
        &self,
        req: &crate::types::EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        self.embedding_spec()
            .choose_embedding_transformers(req, ctx)
    }

    fn embedding_before_send(
        &self,
        req: &crate::types::EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        provider_options_map_merge_hook(&ctx.provider_id, &req.provider_options_map)
    }

    fn embedding_url(&self, req: &crate::types::EmbeddingRequest, ctx: &ProviderContext) -> String {
        self.embedding_spec().embedding_url(req, ctx)
    }

    fn choose_image_transformers(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> ImageTransformers {
        self.image_spec().choose_image_transformers(req, ctx)
    }

    fn image_url(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        self.image_spec().image_url(req, ctx)
    }

    fn rerank_url(&self, req: &RerankRequest, ctx: &ProviderContext) -> String {
        self.rerank_spec().rerank_url(req, ctx)
    }

    fn choose_rerank_transformers(
        &self,
        req: &RerankRequest,
        ctx: &ProviderContext,
    ) -> RerankTransformers {
        self.rerank_spec().choose_rerank_transformers(req, ctx)
    }
}

impl OpenAiCompatibleSpecWithAdapter {
    fn chat_spec(&self) -> crate::standards::openai::chat::OpenAiChatSpec {
        #[derive(Debug)]
        struct CompatToOpenAiChatAdapter {
            adapter: Arc<dyn super::adapter::ProviderAdapter>,
            chat_endpoint: String,
        }

        impl CompatToOpenAiChatAdapter {
            fn new(adapter: Arc<dyn super::adapter::ProviderAdapter>) -> Self {
                let path = adapter.route_for(super::types::RequestType::Chat);
                let endpoint = format!("/{}", path.trim_start_matches('/'));
                Self {
                    adapter,
                    chat_endpoint: endpoint,
                }
            }
        }

        impl crate::standards::openai::chat::OpenAiChatAdapter for CompatToOpenAiChatAdapter {
            fn build_headers(
                &self,
                api_key: &str,
                base_headers: &mut reqwest::header::HeaderMap,
            ) -> Result<(), LlmError> {
                if api_key.is_empty() {
                    return Err(LlmError::MissingApiKey(
                        "OpenAI-Compatible API key not provided".into(),
                    ));
                }
                let _ = base_headers;
                Ok(())
            }

            fn transform_request(
                &self,
                req: &ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.adapter.transform_request_params(
                    body,
                    &req.common_params.model,
                    super::types::RequestType::Chat,
                )
            }

            fn chat_endpoint(&self) -> &str {
                &self.chat_endpoint
            }
        }

        crate::standards::openai::chat::OpenAiChatStandard::with_adapters(
            Arc::new(CompatToOpenAiChatAdapter::new(self.adapter.clone())),
            self.adapter.clone(),
        )
        .create_spec("openai_compatible")
    }

    fn embedding_spec(&self) -> crate::standards::openai::embedding::OpenAiEmbeddingSpec {
        #[derive(Clone)]
        struct CompatEmbeddingAdapter {
            adapter: Arc<dyn super::adapter::ProviderAdapter>,
            embedding_endpoint: String,
        }

        impl crate::standards::openai::embedding::OpenAiEmbeddingAdapter for CompatEmbeddingAdapter {
            fn transform_request(
                &self,
                req: &crate::types::EmbeddingRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.adapter.transform_request_params(
                    body,
                    req.model.as_deref().unwrap_or(""),
                    super::types::RequestType::Embedding,
                )
            }

            fn embedding_endpoint(&self) -> &str {
                &self.embedding_endpoint
            }
        }

        let endpoint = format!(
            "/{}",
            self.adapter
                .route_for(super::types::RequestType::Embedding)
                .trim_start_matches('/')
        );

        crate::standards::openai::embedding::OpenAiEmbeddingStandard::with_adapter(Arc::new(
            CompatEmbeddingAdapter {
                adapter: self.adapter.clone(),
                embedding_endpoint: endpoint,
            },
        ))
        .create_spec("openai_compatible")
    }

    fn image_spec(&self) -> crate::standards::openai::image::OpenAiImageSpec {
        #[derive(Clone)]
        struct CompatImageAdapter {
            adapter: Arc<dyn super::adapter::ProviderAdapter>,
            generation_endpoint: String,
        }

        impl crate::standards::openai::image::OpenAiImageAdapter for CompatImageAdapter {
            fn transform_generation_request(
                &self,
                _req: &crate::types::ImageGenerationRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                let model_s = body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                self.adapter.transform_request_params(
                    body,
                    &model_s,
                    super::types::RequestType::ImageGeneration,
                )
            }

            fn generation_endpoint(&self) -> &str {
                &self.generation_endpoint
            }
        }

        let endpoint = format!(
            "/{}",
            self.adapter
                .route_for(super::types::RequestType::ImageGeneration)
                .trim_start_matches('/')
        );

        crate::standards::openai::image::OpenAiImageStandard::with_adapter(Arc::new(
            CompatImageAdapter {
                adapter: self.adapter.clone(),
                generation_endpoint: endpoint,
            },
        ))
        .create_spec("openai_compatible")
    }

    fn rerank_spec(&self) -> crate::standards::openai::rerank::OpenAiRerankSpec {
        #[derive(Clone)]
        struct CompatRerankAdapter {
            adapter: Arc<dyn super::adapter::ProviderAdapter>,
            rerank_endpoint: String,
        }

        impl crate::standards::openai::rerank::OpenAiRerankAdapter for CompatRerankAdapter {
            fn transform_request(
                &self,
                req: &RerankRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.adapter.transform_request_params(
                    body,
                    &req.model,
                    super::types::RequestType::Rerank,
                )
            }

            fn rerank_endpoint(&self) -> &str {
                &self.rerank_endpoint
            }
        }

        let endpoint = format!(
            "/{}",
            self.adapter
                .route_for(super::types::RequestType::Rerank)
                .trim_start_matches('/')
        );

        crate::standards::openai::rerank::OpenAiRerankStandard::with_adapter(Arc::new(
            CompatRerankAdapter {
                adapter: self.adapter.clone(),
                rerank_endpoint: endpoint,
            },
        ))
        .create_spec("openai_compatible")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig,
    };

    #[test]
    fn openai_compatible_custom_provider_options_are_keyed_by_runtime_provider_id() {
        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::default().with_provider_option(
            "deepseek",
            serde_json::json!({
                "some_vendor_param": true
            }),
        );

        let hook = spec
            .chat_before_send(&req, &ctx)
            .expect("should install before_send for matching custom provider options");

        let body = serde_json::json!({
            "model": "deepseek-chat",
        });
        let out = hook(&body).unwrap();
        assert_eq!(
            out.get("some_vendor_param"),
            Some(&serde_json::Value::Bool(true))
        );
    }
}
