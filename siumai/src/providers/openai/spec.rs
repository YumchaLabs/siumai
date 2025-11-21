#[cfg(all(feature = "openai-compatible", feature = "std-openai-external"))]
use crate::core::provider_spec::{
    bridge_core_chat_transformers, bridge_core_embedding_transformers,
    bridge_core_image_transformers, map_core_stream_event_with_provider,
    openai_chat_request_to_core_input,
};
use crate::core::{ChatTransformers, EmbeddingTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
#[cfg(not(feature = "provider-openai-external"))]
use crate::execution::http::headers::ProviderHeaders;
#[cfg(feature = "openai-compatible")]
use crate::std_openai::openai::chat::{OpenAiChatStandard, OpenAiDefaultChatAdapter};
#[cfg(feature = "openai-compatible")]
use crate::std_openai::openai::embedding::OpenAiEmbeddingStandard;
use crate::std_openai::openai::image::OpenAiImageStandard;
#[cfg(feature = "std-openai-external")]
use crate::std_openai::openai::responses::OpenAiResponsesStandard;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, EmbeddingRequest, ProviderOptions};
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// OpenAI ProviderSpec implementation
///
/// In non-external mode this spec talks directly to the OpenAI HTTP API
/// using in-crate standards and helpers.
///
/// When `provider-openai-external` is enabled, this spec becomes a thin
/// bridge that maps `ProviderContext` / `ChatRequest` into the core-level
/// `CoreProviderContext` used by the `siumai-provider-openai` crate, and
/// delegates headers/routing/transformers selection to that crate.
#[derive(Clone, Default)]
pub struct OpenAiSpec {
    /// Standard OpenAI Chat implementation
    #[cfg(feature = "openai-compatible")]
    chat_standard: OpenAiChatStandard,
    /// Standard OpenAI Embedding implementation
    #[cfg(feature = "openai-compatible")]
    embedding_standard: OpenAiEmbeddingStandard,
    /// Standard OpenAI Image implementation
    image_standard: OpenAiImageStandard,
}

impl OpenAiSpec {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "openai-compatible")]
            chat_standard: OpenAiChatStandard::with_adapter(Arc::new(
                OpenAiDefaultChatAdapter::default(),
            )),
            #[cfg(feature = "openai-compatible")]
            embedding_standard: OpenAiEmbeddingStandard::new(),
            image_standard: OpenAiImageStandard::new(),
        }
    }

    fn use_responses_api(&self, req: &ChatRequest, _ctx: &ProviderContext) -> bool {
        // Check if Responses API is configured in provider_options
        if let ProviderOptions::OpenAi(ref options) = req.provider_options {
            options.responses_api.is_some()
        } else {
            false
        }
    }
}

impl ProviderSpec for OpenAiSpec {
    fn id(&self) -> &'static str {
        "openai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        #[cfg(feature = "provider-openai-external")]
        {
            use siumai_core::provider_spec::CoreProviderSpec;

            // Forward all core context fields directly to the provider crate.
            let core_ctx = ctx.to_core_context();

            let core_spec = siumai_provider_openai::spec::OpenAiCoreSpec::new();
            return core_spec.build_headers(&core_ctx);
        }

        #[cfg(not(feature = "provider-openai-external"))]
        {
            let api_key = ctx
                .api_key
                .as_ref()
                .ok_or_else(|| LlmError::MissingApiKey("OpenAI API key not provided".into()))?;
            return ProviderHeaders::openai(
                api_key,
                ctx.organization.as_deref(),
                ctx.project.as_deref(),
                &ctx.http_extra_headers,
            );
        }
    }

    fn chat_url(&self, _stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        #[cfg(feature = "provider-openai-external")]
        {
            use siumai_core::provider_spec::CoreProviderSpec;

            // Inject Responses API toggle into core extras so that the core spec
            // can decide which endpoint to use.
            let mut core_ctx = ctx.to_core_context();
            let enabled = self.use_responses_api(req, ctx);
            core_ctx
                .extras
                .entry("openai.responses_api".to_string())
                .or_insert(serde_json::json!(enabled));

            let core_spec = siumai_provider_openai::spec::OpenAiCoreSpec::new();
            return core_spec.chat_url(&core_ctx);
        }

        #[cfg(not(feature = "provider-openai-external"))]
        {
            let use_responses = self.use_responses_api(req, ctx);
            let suffix = if use_responses {
                "/responses"
            } else {
                "/chat/completions"
            };
            return format!("{}{}", ctx.base_url.trim_end_matches('/'), suffix);
        }
    }

    fn embedding_url(&self, _req: &EmbeddingRequest, ctx: &ProviderContext) -> String {
        #[cfg(feature = "provider-openai-external")]
        {
            let suffix = siumai_provider_openai::helpers::embedding_path();
            return format!("{}{}", ctx.base_url.trim_end_matches('/'), suffix);
        }
        #[cfg(not(feature = "provider-openai-external"))]
        {
            return format!("{}/embeddings", ctx.base_url.trim_end_matches('/'));
        }
    }

    fn image_url(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        #[cfg(feature = "provider-openai-external")]
        {
            let suffix = siumai_provider_openai::helpers::image_generation_path();
            return format!("{}{}", ctx.base_url.trim_end_matches('/'), suffix);
        }
        #[cfg(not(feature = "provider-openai-external"))]
        {
            return format!("{}/images/generations", ctx.base_url.trim_end_matches('/'));
        }
    }

    fn image_edit_url(
        &self,
        _req: &crate::types::ImageEditRequest,
        ctx: &ProviderContext,
    ) -> String {
        #[cfg(feature = "provider-openai-external")]
        {
            let suffix = siumai_provider_openai::helpers::image_edit_path();
            return format!("{}{}", ctx.base_url.trim_end_matches('/'), suffix);
        }
        #[cfg(not(feature = "provider-openai-external"))]
        {
            return format!("{}/images/edits", ctx.base_url.trim_end_matches('/'));
        }
    }

    fn image_variation_url(
        &self,
        _req: &crate::types::ImageVariationRequest,
        ctx: &ProviderContext,
    ) -> String {
        #[cfg(feature = "provider-openai-external")]
        {
            let suffix = siumai_provider_openai::helpers::image_variation_path();
            return format!("{}{}", ctx.base_url.trim_end_matches('/'), suffix);
        }
        #[cfg(not(feature = "provider-openai-external"))]
        {
            return format!("{}/images/variations", ctx.base_url.trim_end_matches('/'));
        }
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        if self.use_responses_api(req, ctx) {
            // Responses API transformers
            #[cfg(feature = "std-openai-external")]
            {
                // Bridge from aggregator-level ChatRequest into core ResponsesInput
                // and then into the std-openai Responses standard.
                #[derive(Clone)]
                struct ResponsesRequestBridge {
                    standard: OpenAiResponsesStandard,
                }

                impl crate::execution::transformers::request::RequestTransformer for ResponsesRequestBridge {
                    fn provider_id(&self) -> &str {
                        "openai_responses"
                    }

                    fn transform_chat(
                        &self,
                        req: &crate::types::ChatRequest,
                    ) -> Result<serde_json::Value, LlmError> {
                        let core_req =
                            crate::providers::openai::responses_bridge::build_responses_input(req)?;
                        let tx = self.standard.create_request_transformer("openai_responses");
                        tx.transform_responses(&core_req)
                    }
                }

                let req_tx = ResponsesRequestBridge {
                    standard: OpenAiResponsesStandard::new(),
                };
                let resp_tx =
                    crate::providers::openai::transformers::OpenAiResponsesResponseTransformer;
                #[cfg(feature = "openai-compatible")]
                let stream_tx = {
                    use crate::core::provider_spec::map_core_stream_event_with_provider;
                    use siumai_core::execution::streaming::ChatStreamEventConverterCore;

                    #[derive(Clone)]
                    struct OpenAiResponsesCoreStreamBridge {
                        inner: std::sync::Arc<dyn ChatStreamEventConverterCore>,
                    }

                    impl crate::execution::transformers::stream::StreamChunkTransformer
                        for OpenAiResponsesCoreStreamBridge
                    {
                        fn provider_id(&self) -> &str {
                            self.inner.provider_id()
                        }

                        fn convert_event(
                            &self,
                            event: eventsource_stream::Event,
                        ) -> crate::execution::transformers::stream::StreamEventFuture<'_>
                        {
                            let inner = self.inner.clone();
                            Box::pin(async move {
                                inner
                                    .convert_event(event)
                                    .into_iter()
                                    .map(|res| {
                                        res.map(|e| {
                                            map_core_stream_event_with_provider("openai", e)
                                        })
                                    })
                                    .collect()
                            })
                        }

                        fn handle_stream_end(
                            &self,
                        ) -> Option<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>>
                        {
                            let inner = self.inner.clone();
                            inner.handle_stream_end().map(|res| {
                                res.map(|e| map_core_stream_event_with_provider("openai", e))
                            })
                        }
                    }

                    let core_stream =
                        OpenAiResponsesStandard::new().create_stream_converter("openai_responses");

                    OpenAiResponsesCoreStreamBridge { inner: core_stream }
                };

                return ChatTransformers {
                    request: Arc::new(req_tx),
                    response: Arc::new(resp_tx),
                    stream: {
                        #[cfg(feature = "openai-compatible")]
                        {
                            Some(Arc::new(stream_tx))
                        }
                        #[cfg(not(feature = "openai-compatible"))]
                        {
                            None
                        }
                    },
                    json: None,
                };
            }

            #[cfg(not(feature = "std-openai-external"))]
            {
                let req_tx =
                    crate::providers::openai::transformers::OpenAiResponsesRequestTransformer;
                let resp_tx =
                    crate::providers::openai::transformers::OpenAiResponsesResponseTransformer;
                #[cfg(feature = "openai-compatible")]
                let stream_tx = {
                    let converter =
                        crate::providers::openai::responses::OpenAiResponsesEventConverter::new();
                    crate::providers::openai::transformers::OpenAiResponsesStreamChunkTransformer {
                        provider_id: "openai_responses".to_string(),
                        inner: converter,
                    }
                };
                return ChatTransformers {
                    request: Arc::new(req_tx),
                    response: Arc::new(resp_tx),
                    stream: {
                        #[cfg(feature = "openai-compatible")]
                        {
                            Some(Arc::new(stream_tx))
                        }
                        #[cfg(not(feature = "openai-compatible"))]
                        {
                            None
                        }
                    },
                    json: None,
                };
            }
        } else {
            #[cfg(feature = "openai-compatible")]
            {
                // Use standard OpenAI Chat API from standards layer
                #[cfg(feature = "std-openai-external")]
                {
                    // Bridge external core-only chat transformers into aggregator traits
                    use siumai_core::provider_spec::CoreChatTransformers;

                    let core_txs: CoreChatTransformers = CoreChatTransformers {
                        request: self.chat_standard.create_request_transformer("openai"),
                        response: self.chat_standard.create_response_transformer("openai"),
                        stream: Some(self.chat_standard.create_stream_converter("openai")),
                    };

                    return bridge_core_chat_transformers(
                        core_txs,
                        openai_chat_request_to_core_input,
                        |evt| map_core_stream_event_with_provider("openai", evt),
                    );
                }
                #[cfg(not(feature = "std-openai-external"))]
                {
                    let spec = self.chat_standard.create_spec("openai");
                    return spec.choose_chat_transformers(req, ctx);
                }
            }
            #[cfg(not(feature = "openai-compatible"))]
            {
                // Fallback: provider-native transformers without streaming when compat is disabled
                let request_tx =
                    Arc::new(crate::providers::openai::transformers::OpenAiRequestTransformer);
                let response_tx =
                    Arc::new(crate::providers::openai::transformers::OpenAiResponseTransformer);
                return ChatTransformers {
                    request: request_tx,
                    response: response_tx,
                    stream: None,
                    json: None,
                };
            }
        }
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // All OpenAI-specific Chat/Responses configuration is injected via
        // ProviderOptions → ChatInput::extra / ResponsesInput::extra → std/provider
        // adapters. Only CustomProviderOptions are handled here to avoid
        // overlapping responsibilities with the standards layer.
        crate::core::default_custom_options_hook(self.id(), req)
    }

    fn choose_embedding_transformers(
        &self,
        _req: &EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        #[cfg(feature = "openai-compatible")]
        {
            // Use standard OpenAI Embedding API from standards layer
            #[cfg(feature = "std-openai-external")]
            {
                // Bridge external core-only embedding transformers to aggregator traits
                let req_tx = self.embedding_standard.create_request_transformer("openai");
                let resp_tx = self
                    .embedding_standard
                    .create_response_transformer("openai");

                return bridge_core_embedding_transformers(req_tx, resp_tx);
            }
            #[cfg(not(feature = "std-openai-external"))]
            {
                return EmbeddingTransformers {
                    request: self.embedding_standard.create_request_transformer("openai"),
                    response: self
                        .embedding_standard
                        .create_response_transformer("openai"),
                };
            }
        }
        #[cfg(not(feature = "openai-compatible"))]
        {
            // Fallback: provider-native transformers
            return EmbeddingTransformers {
                request: Arc::new(crate::providers::openai::transformers::OpenAiRequestTransformer),
                response: Arc::new(
                    crate::providers::openai::transformers::OpenAiResponseTransformer,
                ),
            };
        }
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> crate::core::ImageTransformers {
        // Use standard OpenAI Image API from standards layer
        let transformers = self.image_standard.create_transformers("openai");
        #[cfg(feature = "std-openai-external")]
        {
            return bridge_core_image_transformers(transformers.request, transformers.response);
        }
        #[cfg(not(feature = "std-openai-external"))]
        {
            crate::core::ImageTransformers {
                request: transformers.request,
                response: transformers.response,
            }
        }
    }

    fn choose_audio_transformer(&self, _ctx: &ProviderContext) -> crate::core::AudioTransformer {
        crate::core::AudioTransformer {
            transformer: Arc::new(crate::providers::openai::transformers::OpenAiAudioTransformer),
        }
    }

    fn choose_files_transformer(&self, _ctx: &ProviderContext) -> crate::core::FilesTransformer {
        crate::core::FilesTransformer {
            transformer: Arc::new(crate::providers::openai::transformers::OpenAiFilesTransformer),
        }
    }
}
