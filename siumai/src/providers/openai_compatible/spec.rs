use crate::error::LlmError;
use crate::provider_core::{
    ChatTransformers, EmbeddingTransformers, ImageTransformers, ProviderContext, ProviderSpec,
};
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use crate::utils::http_headers::{ProviderHeaders, inject_tracing_headers};
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
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_ref().ok_or_else(|| {
            LlmError::MissingApiKey("OpenAI-Compatible API key not provided".into())
        })?;
        // Use OpenAI-style Bearer with pass-through custom headers; adapter-level custom
        // headers should have been injected into `ctx.http_extra_headers` by the builder.
        let mut headers = ProviderHeaders::openai(
            api_key,
            ctx.organization.as_deref(),
            ctx.project.as_deref(),
            &ctx.http_extra_headers,
        )?;
        inject_tracing_headers(&mut headers);
        Ok(headers)
    }

    fn chat_url(&self, _stream: bool, _req: &ChatRequest, ctx: &ProviderContext) -> String {
        // Standard OpenAI-compatible route for chat
        format!("{}/chat/completions", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        // Resolve adapter from global registry using the provider id in context.
        // Fallback to a minimal ConfigurableAdapter if not found.
        #[allow(unused_mut)]
        let mut adapter: Arc<
            dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
        > = {
            #[cfg(feature = "openai")]
            {
                if let Ok(a) = crate::registry::get_provider_adapter(&ctx.provider_id) {
                    a
                } else {
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

        let compat_cfg =
            crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
                &ctx.provider_id,
                ctx.api_key.as_deref().unwrap_or_default(),
                &ctx.base_url,
                adapter.clone(),
            )
            .with_model(&req.common_params.model);

        let request_tx =
            crate::providers::openai_compatible::transformers::CompatRequestTransformer {
                config: compat_cfg.clone(),
                adapter: adapter.clone(),
            };
        let response_tx =
            crate::providers::openai_compatible::transformers::CompatResponseTransformer {
                config: compat_cfg.clone(),
                adapter: adapter.clone(),
            };
        let converter =
            crate::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter::new(
                compat_cfg, adapter,
            );
        let stream_tx =
            crate::providers::openai_compatible::transformers::CompatStreamChunkTransformer {
                provider_id: ctx.provider_id.clone(),
                inner: converter,
            };
        ChatTransformers {
            request: Arc::new(request_tx),
            response: Arc::new(response_tx),
            stream: Some(Arc::new(stream_tx)),
            json: None,
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
        let compat_cfg =
            crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
                &ctx.provider_id,
                ctx.api_key.as_deref().unwrap_or_default(),
                &ctx.base_url,
                adapter.clone(),
            );
        let req_tx = crate::providers::openai_compatible::transformers::CompatRequestTransformer {
            config: compat_cfg.clone(),
            adapter: adapter.clone(),
        };
        let resp_tx =
            crate::providers::openai_compatible::transformers::CompatResponseTransformer {
                config: compat_cfg,
                adapter,
            };
        EmbeddingTransformers {
            request: Arc::new(req_tx),
            response: Arc::new(resp_tx),
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
        let compat_cfg =
            crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
                &ctx.provider_id,
                ctx.api_key.as_deref().unwrap_or_default(),
                &ctx.base_url,
                adapter.clone(),
            );
        let req_tx = crate::providers::openai_compatible::transformers::CompatRequestTransformer {
            config: compat_cfg.clone(),
            adapter: adapter.clone(),
        };
        let resp_tx =
            crate::providers::openai_compatible::transformers::CompatResponseTransformer {
                config: compat_cfg,
                adapter,
            };
        ImageTransformers {
            request: Arc::new(req_tx),
            response: Arc::new(resp_tx),
        }
    }
}
