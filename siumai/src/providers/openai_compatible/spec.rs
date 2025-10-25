use crate::core::{
    ChatTransformers, EmbeddingTransformers, ImageTransformers, ProviderContext, ProviderSpec,
};
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// OpenAI-Compatible ProviderSpec implementation
///
/// This Spec delegates request/response/stream mapping to the configured
/// ProviderAdapter resolved from the global registry by `ctx.provider_id`.
#[derive(Clone, Copy, Default)]
pub struct OpenAiCompatibleSpec;

impl ProviderSpec for OpenAiCompatibleSpec {
    fn id(&self) -> &'static str {
        "openai_compatible"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Baseline capabilities; concrete provider capabilities are tracked in registry.
        // Declare the set of endpoints this spec can drive via transformers.
        // Actual per-provider enablement is enforced by client/adapters and higher-level handles.
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
            .with_custom_feature("image_generation", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_ref().ok_or_else(|| {
            LlmError::MissingApiKey("OpenAI-Compatible API key not provided".into())
        })?;
        // Use OpenAI-style Bearer with pass-through custom headers; adapter-level custom
        // headers should have been injected into `ctx.http_extra_headers` by the builder.
        ProviderHeaders::openai(
            api_key,
            ctx.organization.as_deref(),
            ctx.project.as_deref(),
            &ctx.http_extra_headers,
        )
    }

    fn chat_url(&self, _stream: bool, _req: &ChatRequest, ctx: &ProviderContext) -> String {
        // Resolve adapter to support providers with divergent chat routes
        let adapter: std::sync::Arc<
            dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
        > = {
            #[cfg(feature = "openai")]
            {
                crate::registry::get_provider_adapter(&ctx.provider_id).unwrap_or_else(|_| {
                    std::sync::Arc::new(
                        crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                            crate::providers::openai_compatible::registry::ProviderConfig {
                                id: ctx.provider_id.clone(),
                                name: ctx.provider_id.clone(),
                                base_url: ctx.base_url.clone(),
                                field_mappings: Default::default(),
                                capabilities: vec!["chat".into(), "streaming".into()],
                                default_model: None,
                                supports_reasoning: false,
                            },
                        ),
                    )
                })
            }
            #[cfg(not(feature = "openai"))]
            {
                std::sync::Arc::new(
                    crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                        crate::providers::openai_compatible::registry::ProviderConfig {
                            id: ctx.provider_id.clone(),
                            name: ctx.provider_id.clone(),
                            base_url: ctx.base_url.clone(),
                            field_mappings: Default::default(),
                            capabilities: vec!["chat".into(), "streaming".into()],
                            default_model: None,
                            supports_reasoning: false,
                        },
                    ),
                )
            }
        };
        let path = adapter.route_for(crate::providers::openai_compatible::types::RequestType::Chat);
        format!("{}/{}", ctx.base_url.trim_end_matches('/'), path)
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        // Resolve provider adapter (OpenAI-compatible)
        let adapter: Arc<dyn crate::providers::openai_compatible::adapter::ProviderAdapter> = {
            #[cfg(feature = "openai")]
            {
                crate::registry::get_provider_adapter(&ctx.provider_id).unwrap_or_else(|_| {
                    Arc::new(
                        crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                            crate::providers::openai_compatible::registry::ProviderConfig {
                                id: ctx.provider_id.clone(),
                                name: ctx.provider_id.clone(),
                                base_url: ctx.base_url.clone(),
                                field_mappings: Default::default(),
                                capabilities: vec!["chat".into(), "streaming".into()],
                                default_model: None,
                                supports_reasoning: false,
                            },
                        ),
                    )
                })
            }
            #[cfg(not(feature = "openai"))]
            {
                Arc::new(
                    crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                        crate::providers::openai_compatible::registry::ProviderConfig {
                            id: ctx.provider_id.clone(),
                            name: ctx.provider_id.clone(),
                            base_url: ctx.base_url.clone(),
                            field_mappings: Default::default(),
                            capabilities: vec!["chat".into(), "streaming".into()],
                            default_model: None,
                            supports_reasoning: false,
                        },
                    ),
                )
            }
        };

        // Bridge ProviderAdapter -> OpenAiChatAdapter for standards layer
        struct CompatToOpenAiChatAdapter(
            Arc<dyn crate::providers::openai_compatible::adapter::ProviderAdapter>,
        );
        impl crate::standards::openai::chat::OpenAiChatAdapter for CompatToOpenAiChatAdapter {
            fn transform_request(
                &self,
                req: &ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.0.transform_request_params(
                    body,
                    &req.common_params.model,
                    crate::providers::openai_compatible::types::RequestType::Chat,
                )
            }
            fn transform_response(&self, resp: &mut serde_json::Value) -> Result<(), LlmError> {
                if let Some(choices) = resp.get_mut("choices").and_then(|v| v.as_array_mut()) {
                    for ch in choices {
                        if let Some(msg) = ch.get_mut("message") {
                            let has_thinking = msg.get("thinking").is_some();
                            if !has_thinking {
                                let t = msg
                                    .get("reasoning_content")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .or_else(|| {
                                        msg.get("reasoning")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string())
                                    });
                                if let Some(text) = t {
                                    if let Some(obj) = msg.as_object_mut() {
                                        obj.insert("thinking".to_string(), serde_json::json!(text));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            fn transform_sse_event(&self, event: &mut serde_json::Value) -> Result<(), LlmError> {
                if let Some(choices) = event.get_mut("choices").and_then(|v| v.as_array_mut()) {
                    for ch in choices {
                        if let Some(delta) = ch.get_mut("delta") {
                            let has_thinking = delta.get("thinking").is_some();
                            if !has_thinking {
                                let t = delta
                                    .get("reasoning_content")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .or_else(|| {
                                        delta
                                            .get("reasoning")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string())
                                    });
                                if let Some(text) = t {
                                    if let Some(obj) = delta.as_object_mut() {
                                        obj.insert("thinking".to_string(), serde_json::json!(text));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
        }

        let std = crate::standards::openai::chat::OpenAiChatStandard::with_adapter(Arc::new(
            CompatToOpenAiChatAdapter(adapter),
        ));
        std.create_transformers(&ctx.provider_id)
    }

    fn choose_embedding_transformers(
        &self,
        _req: &crate::types::EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        // Resolve adapter via registry
        let adapter: Arc<dyn crate::providers::openai_compatible::adapter::ProviderAdapter> = {
            #[cfg(feature = "openai")]
            {
                crate::registry::get_provider_adapter(&ctx.provider_id).unwrap_or_else(|_| {
                    Arc::new(
                        crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                            crate::providers::openai_compatible::registry::ProviderConfig {
                                id: ctx.provider_id.clone(),
                                name: ctx.provider_id.clone(),
                                base_url: ctx.base_url.clone(),
                                field_mappings: Default::default(),
                                capabilities: vec!["chat".into(), "embedding".into()],
                                default_model: None,
                                supports_reasoning: false,
                            },
                        ),
                    )
                })
            }
            #[cfg(not(feature = "openai"))]
            {
                Arc::new(
                    crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                        crate::providers::openai_compatible::registry::ProviderConfig {
                            id: ctx.provider_id.clone(),
                            name: ctx.provider_id.clone(),
                            base_url: ctx.base_url.clone(),
                            field_mappings: Default::default(),
                            capabilities: vec!["chat".into(), "embedding".into()],
                            default_model: None,
                            supports_reasoning: false,
                        },
                    ),
                )
            }
        };
        // Bridge ProviderAdapter -> OpenAiEmbeddingAdapter
        struct CompatToOpenAiEmbeddingAdapter(
            Arc<dyn crate::providers::openai_compatible::adapter::ProviderAdapter>,
        );
        impl crate::standards::openai::embedding::OpenAiEmbeddingAdapter
            for CompatToOpenAiEmbeddingAdapter
        {
            fn transform_request(
                &self,
                req: &crate::types::EmbeddingRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.0.transform_request_params(
                    body,
                    req.model.as_deref().unwrap_or(""),
                    crate::providers::openai_compatible::types::RequestType::Embedding,
                )
            }
        }

        let std = crate::standards::openai::embedding::OpenAiEmbeddingStandard::with_adapter(
            Arc::new(CompatToOpenAiEmbeddingAdapter(adapter)),
        );

        EmbeddingTransformers {
            request: std.create_request_transformer(&ctx.provider_id),
            response: std.create_response_transformer(&ctx.provider_id),
        }
    }

    fn embedding_url(
        &self,
        _req: &crate::types::EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> String {
        let adapter: std::sync::Arc<
            dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
        > = {
            #[cfg(feature = "openai")]
            {
                crate::registry::get_provider_adapter(&ctx.provider_id).unwrap_or_else(|_| {
                    std::sync::Arc::new(
                        crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                            crate::providers::openai_compatible::registry::ProviderConfig {
                                id: ctx.provider_id.clone(),
                                name: ctx.provider_id.clone(),
                                base_url: ctx.base_url.clone(),
                                field_mappings: Default::default(),
                                capabilities: vec!["embedding".into()],
                                default_model: None,
                                supports_reasoning: false,
                            },
                        ),
                    )
                })
            }
            #[cfg(not(feature = "openai"))]
            {
                std::sync::Arc::new(
                    crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                        crate::providers::openai_compatible::registry::ProviderConfig {
                            id: ctx.provider_id.clone(),
                            name: ctx.provider_id.clone(),
                            base_url: ctx.base_url.clone(),
                            field_mappings: Default::default(),
                            capabilities: vec!["embedding".into()],
                            default_model: None,
                            supports_reasoning: false,
                        },
                    ),
                )
            }
        };
        let path =
            adapter.route_for(crate::providers::openai_compatible::types::RequestType::Embedding);
        format!("{}/{}", ctx.base_url.trim_end_matches('/'), path)
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> ImageTransformers {
        let adapter: Arc<dyn crate::providers::openai_compatible::adapter::ProviderAdapter> = {
            #[cfg(feature = "openai")]
            {
                crate::registry::get_provider_adapter(&ctx.provider_id).unwrap_or_else(|_| {
                    Arc::new(
                        crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                            crate::providers::openai_compatible::registry::ProviderConfig {
                                id: ctx.provider_id.clone(),
                                name: ctx.provider_id.clone(),
                                base_url: ctx.base_url.clone(),
                                field_mappings: Default::default(),
                                capabilities: vec!["chat".into(), "image_generation".into()],
                                default_model: None,
                                supports_reasoning: false,
                            },
                        ),
                    )
                })
            }
            #[cfg(not(feature = "openai"))]
            {
                Arc::new(
                    crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                        crate::providers::openai_compatible::registry::ProviderConfig {
                            id: ctx.provider_id.clone(),
                            name: ctx.provider_id.clone(),
                            base_url: ctx.base_url.clone(),
                            field_mappings: Default::default(),
                            capabilities: vec!["chat".into(), "image_generation".into()],
                            default_model: None,
                            supports_reasoning: false,
                        },
                    ),
                )
            }
        };
        // Bridge ProviderAdapter -> OpenAiImageAdapter
        struct CompatToOpenAiImageAdapter(
            Arc<dyn crate::providers::openai_compatible::adapter::ProviderAdapter>,
        );
        impl crate::standards::openai::image::OpenAiImageAdapter for CompatToOpenAiImageAdapter {
            fn transform_generation_request(
                &self,
                _req: &crate::types::ImageGenerationRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // Reuse provider adapter param mapping for image generation
                // Extract model first to avoid borrow conflict
                let model_s = body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                self.0.transform_request_params(
                    body,
                    &model_s,
                    crate::providers::openai_compatible::types::RequestType::ImageGeneration,
                )
            }
        }

        let std = crate::standards::openai::image::OpenAiImageStandard::with_adapter(Arc::new(
            CompatToOpenAiImageAdapter(adapter),
        ));
        let t = std.create_transformers(&ctx.provider_id);
        ImageTransformers {
            request: t.request,
            response: t.response,
        }
    }

    fn image_url(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        let adapter: std::sync::Arc<
            dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
        > = {
            #[cfg(feature = "openai")]
            {
                crate::registry::get_provider_adapter(&ctx.provider_id).unwrap_or_else(|_| {
                    std::sync::Arc::new(
                        crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                            crate::providers::openai_compatible::registry::ProviderConfig {
                                id: ctx.provider_id.clone(),
                                name: ctx.provider_id.clone(),
                                base_url: ctx.base_url.clone(),
                                field_mappings: Default::default(),
                                capabilities: vec!["image_generation".into()],
                                default_model: None,
                                supports_reasoning: false,
                            },
                        ),
                    )
                })
            }
            #[cfg(not(feature = "openai"))]
            {
                std::sync::Arc::new(
                    crate::providers::openai_compatible::registry::ConfigurableAdapter::new(
                        crate::providers::openai_compatible::registry::ProviderConfig {
                            id: ctx.provider_id.clone(),
                            name: ctx.provider_id.clone(),
                            base_url: ctx.base_url.clone(),
                            field_mappings: Default::default(),
                            capabilities: vec!["image_generation".into()],
                            default_model: None,
                            supports_reasoning: false,
                        },
                    ),
                )
            }
        };
        let path = adapter
            .route_for(crate::providers::openai_compatible::types::RequestType::ImageGeneration);
        format!("{}/{}", ctx.base_url.trim_end_matches('/'), path)
    }
}
