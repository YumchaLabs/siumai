#[cfg(all(feature = "openai", feature = "std-openai-external"))]
use crate::core::provider_spec::{bridge_core_chat_transformers, map_core_stream_event_with_provider};
use crate::core::{
    ChatTransformers, EmbeddingTransformers, ImageTransformers, ProviderContext, ProviderSpec,
};
use crate::error::LlmError;
#[cfg(not(feature = "provider-openai-compatible-external"))]
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
        #[cfg(feature = "provider-openai-compatible-external")]
        {
            // Build headers via external helpers, allowing provider-specific policies
            // (e.g., OpenRouter HTTP-Referer behavior).
            use reqwest::header::HeaderMap as Hm;
            return siumai_provider_openai_compatible::helpers::build_json_headers_with_provider(
                &ctx.provider_id,
                api_key,
                &ctx.http_extra_headers,
                &Hm::new(),
                &Hm::new(),
            );
        }
        #[cfg(not(feature = "provider-openai-compatible-external"))]
        {
            // Default: OpenAI-style headers and passthrough http_extra_headers.
            return ProviderHeaders::openai(
                api_key,
                ctx.organization.as_deref(),
                ctx.project.as_deref(),
                &ctx.http_extra_headers,
            );
        }
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
        #[cfg(feature = "provider-openai-compatible-external")]
        {
            return siumai_provider_openai_compatible::helpers::build_url(
                &ctx.base_url,
                adapter.as_ref(),
                siumai_provider_openai_compatible::types::RequestType::Chat,
            );
        }
        #[cfg(not(feature = "provider-openai-compatible-external"))]
        {
            let path =
                adapter.route_for(crate::providers::openai_compatible::types::RequestType::Chat);
            return format!("{}/{}", ctx.base_url.trim_end_matches('/'), path);
        }
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
        #[cfg(not(feature = "std-openai-external"))]
        impl crate::std_openai::openai::chat::OpenAiChatAdapter for CompatToOpenAiChatAdapter {
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
                                if let Some(text) = t
                                    && let Some(obj) = msg.as_object_mut()
                                {
                                    obj.insert("thinking".to_string(), serde_json::json!(text));
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
                                if let Some(text) = t
                                    && let Some(obj) = delta.as_object_mut()
                                {
                                    obj.insert("thinking".to_string(), serde_json::json!(text));
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
        }

        let std = crate::std_openai::openai::chat::OpenAiChatStandard::with_adapter(Arc::new(
            CompatToOpenAiChatAdapter(adapter.clone()),
        ));

        #[cfg(feature = "std-openai-external")]
        impl crate::std_openai::openai::chat::OpenAiChatAdapter for CompatToOpenAiChatAdapter {
            fn transform_request(
                &self,
                req: &siumai_core::execution::chat::ChatInput,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.0.transform_request_params(
                    body,
                    req.model.as_deref().unwrap_or(""),
                    crate::providers::openai_compatible::types::RequestType::Chat,
                )
            }
            fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
                Ok(())
            }
            fn chat_endpoint(&self) -> &str {
                "/chat/completions"
            }
        }
        #[cfg(not(feature = "std-openai-external"))]
        {
            return std.create_transformers(&ctx.provider_id);
        }

        #[cfg(all(feature = "std-openai-external", feature = "openai"))]
        {
            // Bridge external core-only chat transformers to aggregator ChatTransformers
            use siumai_core::provider_spec::CoreChatTransformers;

            let core_txs: CoreChatTransformers = CoreChatTransformers {
                request: std.create_request_transformer(&ctx.provider_id),
                response: std.create_response_transformer(&ctx.provider_id),
                stream: Some(std.create_stream_converter(&ctx.provider_id)),
            };

            let provider_id = ctx.provider_id.clone();
            return bridge_core_chat_transformers(
                core_txs,
                crate::core::provider_spec::openai_chat_request_to_core_input,
                move |evt| map_core_stream_event_with_provider(&provider_id, evt),
            );
        }

        #[cfg(not(all(feature = "std-openai-external", feature = "openai")))]
        {
            // Internal compat transformers path
            let cfg =
                crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
                    &ctx.provider_id,
                    ctx.api_key.as_deref().unwrap_or(""),
                    &ctx.base_url,
                    adapter.clone(),
                )
                .with_model(&req.common_params.model)
                .with_common_params(req.common_params.clone());

            let req_tx =
                crate::providers::openai_compatible::transformers::CompatRequestTransformer {
                    config: cfg.clone(),
                    adapter: adapter.clone(),
                };
            let resp_tx =
                crate::providers::openai_compatible::transformers::CompatResponseTransformer {
                    config: cfg,
                    adapter,
                };
            return ChatTransformers {
                request: Arc::new(req_tx),
                response: Arc::new(resp_tx),
                stream: None,
                json: None,
            };
        }
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
        // For internal standard (not external), the adapter receives aggregator EmbeddingRequest
        #[cfg(not(feature = "std-openai-external"))]
        impl crate::std_openai::openai::embedding::OpenAiEmbeddingAdapter
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
        // For external standard, the adapter receives core EmbeddingInput
        #[cfg(feature = "std-openai-external")]
        impl crate::std_openai::openai::embedding::OpenAiEmbeddingAdapter
            for CompatToOpenAiEmbeddingAdapter
        {
            fn transform_request(
                &self,
                req: &siumai_core::execution::embedding::EmbeddingInput,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.0.transform_request_params(
                    body,
                    req.model.as_deref().unwrap_or(""),
                    crate::providers::openai_compatible::types::RequestType::Embedding,
                )
            }
        }

        let std = crate::std_openai::openai::embedding::OpenAiEmbeddingStandard::with_adapter(
            Arc::new(CompatToOpenAiEmbeddingAdapter(adapter)),
        );

        #[cfg(feature = "std-openai-external")]
        {
            let req_tx = std.create_request_transformer(&ctx.provider_id);
            let resp_tx = std.create_response_transformer(&ctx.provider_id);
            crate::core::provider_spec::bridge_core_embedding_transformers(req_tx, resp_tx)
        }
        #[cfg(not(feature = "std-openai-external"))]
        {
            EmbeddingTransformers {
                request: std.create_request_transformer(&ctx.provider_id),
                response: std.create_response_transformer(&ctx.provider_id),
            }
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
        #[cfg(feature = "provider-openai-compatible-external")]
        {
            return siumai_provider_openai_compatible::helpers::build_url(
                &ctx.base_url,
                adapter.as_ref(),
                siumai_provider_openai_compatible::types::RequestType::Embedding,
            );
        }
        #[cfg(not(feature = "provider-openai-compatible-external"))]
        {
            let path = adapter
                .route_for(crate::providers::openai_compatible::types::RequestType::Embedding);
            return format!("{}/{}", ctx.base_url.trim_end_matches('/'), path);
        }
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
        impl crate::std_openai::openai::image::OpenAiImageAdapter for CompatToOpenAiImageAdapter {
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

        let std = crate::std_openai::openai::image::OpenAiImageStandard::with_adapter(Arc::new(
            CompatToOpenAiImageAdapter(adapter),
        ));
        let t = std.create_transformers(&ctx.provider_id);
        #[cfg(feature = "std-openai-external")]
        {
            crate::core::provider_spec::bridge_core_image_transformers(t.request, t.response)
        }
        #[cfg(not(feature = "std-openai-external"))]
        {
            ImageTransformers {
                request: t.request,
                response: t.response,
            }
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
        #[cfg(feature = "provider-openai-compatible-external")]
        {
            return siumai_provider_openai_compatible::helpers::build_url(
                &ctx.base_url,
                adapter.as_ref(),
                siumai_provider_openai_compatible::types::RequestType::ImageGeneration,
            );
        }
        #[cfg(not(feature = "provider-openai-compatible-external"))]
        {
            let path = adapter.route_for(
                crate::providers::openai_compatible::types::RequestType::ImageGeneration,
            );
            return format!("{}/{}", ctx.base_url.trim_end_matches('/'), path);
        }
    }
}
