#[cfg(all(feature = "openai-compatible", feature = "std-openai-external"))]
use crate::core::provider_spec::{
    bridge_core_chat_transformers, map_core_stream_event_with_provider,
    openai_like_chat_request_to_core_input,
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
                use crate::types::ProviderOptions;
                use siumai_core::execution::responses::ResponsesInput;
                use std::collections::HashMap;

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
                        // Reuse existing message → input-item mapping logic.
                        let mut input_items = Vec::with_capacity(req.messages.len());
                        for m in &req.messages {
                            let item = crate::providers::openai::transformers::OpenAiResponsesRequestTransformer::convert_message(m)?;
                            input_items.push(item);
                        }

                        let mut extra: HashMap<String, serde_json::Value> = HashMap::new();

                        // Core Responses fields that used to live in the base body.
                        extra.insert("stream".to_string(), serde_json::Value::Bool(req.stream));

                        // tools/tool_choice in Responses format
                        if let Some(tools) = &req.tools {
                            let openai_tools =
                                crate::providers::openai::utils::convert_tools_to_responses_format(
                                    tools,
                                )?;
                            if !openai_tools.is_empty() {
                                extra.insert(
                                    "tools".to_string(),
                                    serde_json::Value::Array(openai_tools),
                                );

                                if let Some(choice) = &req.tool_choice {
                                    let tc = crate::providers::openai::utils::convert_tool_choice(
                                        choice,
                                    );
                                    extra.insert("tool_choice".to_string(), tc);
                                }
                            }
                        }

                        // stream_options
                        if req.stream {
                            extra.insert(
                                "stream_options".to_string(),
                                serde_json::json!({ "include_usage": true }),
                            );
                        }

                        // temperature range-checked by GenericRequestTransformer previously;
                        // we keep the value here and let upstream validation handle ranges.
                        if let Some(temp) = req.common_params.temperature {
                            extra.insert("temperature".to_string(), serde_json::json!(temp));
                        }

                        // max_output_tokens (prefer max_completion_tokens, fallback to max_tokens)
                        if let Some(max_tokens) = req.common_params.max_completion_tokens {
                            extra.insert(
                                "max_output_tokens".to_string(),
                                serde_json::json!(max_tokens),
                            );
                        } else if let Some(max_tokens) = req.common_params.max_tokens {
                            extra.insert(
                                "max_output_tokens".to_string(),
                                serde_json::json!(max_tokens),
                            );
                        }

                        // seed
                        if let Some(seed) = req.common_params.seed {
                            extra.insert("seed".to_string(), serde_json::json!(seed));
                        }

                        // Inject OpenAI-specific ProviderOptions into ResponsesInput::extra
                        if let ProviderOptions::OpenAi(ref options) = req.provider_options {
                            // Responses API configuration
                            if let Some(ref cfg) = options.responses_api {
                                if let Some(ref pid) = cfg.previous_response_id {
                                    extra.insert(
                                        "previous_response_id".to_string(),
                                        serde_json::json!(pid),
                                    );
                                }
                                if let Some(ref fmt) = cfg.response_format {
                                    extra.insert("response_format".to_string(), fmt.clone());
                                }
                                if let Some(bg) = cfg.background {
                                    extra.insert("background".to_string(), serde_json::json!(bg));
                                }
                                if let Some(ref inc) = cfg.include {
                                    extra.insert("include".to_string(), serde_json::json!(inc));
                                }
                                if let Some(ref instr) = cfg.instructions {
                                    extra.insert(
                                        "instructions".to_string(),
                                        serde_json::json!(instr),
                                    );
                                }
                                if let Some(mtc) = cfg.max_tool_calls {
                                    extra.insert(
                                        "max_tool_calls".to_string(),
                                        serde_json::json!(mtc),
                                    );
                                }
                                if let Some(st) = cfg.store {
                                    extra.insert("store".to_string(), serde_json::json!(st));
                                }
                                if let Some(ref trunc) = cfg.truncation
                                    && let Ok(val) = serde_json::to_value(trunc)
                                {
                                    extra.insert("truncation".to_string(), val);
                                }
                                if let Some(ref verb) = cfg.text_verbosity
                                    && let Ok(val) = serde_json::to_value(verb)
                                {
                                    // text_verbosity is nested under "text.verbosity"
                                    let existing_text = extra
                                        .remove("text")
                                        .unwrap_or_else(|| serde_json::json!({}));
                                    let mut map =
                                        existing_text.as_object().cloned().unwrap_or_default();
                                    map.insert("verbosity".to_string(), val);
                                    extra
                                        .insert("text".to_string(), serde_json::Value::Object(map));
                                }
                                if let Some(ref meta) = cfg.metadata {
                                    extra.insert("metadata".to_string(), serde_json::json!(meta));
                                }
                                if let Some(ptc) = cfg.parallel_tool_calls {
                                    extra.insert(
                                        "parallel_tool_calls".to_string(),
                                        serde_json::json!(ptc),
                                    );
                                }
                            }

                            // Reasoning effort / service tier
                            if let Some(effort) = options.reasoning_effort
                                && let Ok(val) = serde_json::to_value(effort)
                            {
                                extra.insert("reasoning_effort".to_string(), val);
                            }
                            if let Some(tier) = options.service_tier
                                && let Ok(val) = serde_json::to_value(tier)
                            {
                                extra.insert("service_tier".to_string(), val);
                            }

                            // Modalities (for multimodal/audio output)
                            if let Some(ref mods) = options.modalities
                                && let Ok(val) = serde_json::to_value(mods)
                            {
                                extra.insert("modalities".to_string(), val);
                            }

                            // Audio configuration
                            if let Some(ref aud) = options.audio
                                && let Ok(val) = serde_json::to_value(aud)
                            {
                                extra.insert("audio".to_string(), val);
                            }

                            // Prediction (Predicted Outputs)
                            if let Some(ref pred) = options.prediction
                                && let Ok(val) = serde_json::to_value(pred)
                            {
                                extra.insert("prediction".to_string(), val);
                            }

                            // Web search options
                            if let Some(ref ws) = options.web_search_options
                                && let Ok(val) = serde_json::to_value(ws)
                            {
                                extra.insert("web_search_options".to_string(), val);
                            }
                        }

                        let core_req = ResponsesInput {
                            model: req.common_params.model.clone(),
                            input: input_items,
                            extra,
                        };

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

                    // Map aggregator-level ChatRequest into core ChatInput, injecting
                    // a subset of OpenAI-specific ProviderOptions into ChatInput::extra.
                    fn openai_chat_request_to_core_input(
                        req: &ChatRequest,
                    ) -> siumai_core::execution::chat::ChatInput {
                        let mut input = openai_like_chat_request_to_core_input(req);

                        if let ProviderOptions::OpenAi(ref options) = req.provider_options {
                            // Reasoning effort (o1/o3 models)
                            if let Some(effort) = options.reasoning_effort {
                                if let Ok(v) = serde_json::to_value(effort) {
                                    input.extra.insert("openai_reasoning_effort".to_string(), v);
                                }
                            }

                            // Service tier preference
                            if let Some(tier) = options.service_tier {
                                if let Ok(v) = serde_json::to_value(tier) {
                                    input.extra.insert("openai_service_tier".to_string(), v);
                                }
                            }

                            // Modalities (e.g., ["text","audio"])
                            if let Some(ref mods) = options.modalities
                                && let Ok(v) = serde_json::to_value(mods)
                            {
                                input.extra.insert("openai_modalities".to_string(), v);
                            }

                            // Audio configuration
                            if let Some(ref aud) = options.audio
                                && let Ok(v) = serde_json::to_value(aud)
                            {
                                input.extra.insert("openai_audio".to_string(), v);
                            }

                            // Prediction content
                            if let Some(ref pred) = options.prediction
                                && let Ok(v) = serde_json::to_value(pred)
                            {
                                input.extra.insert("openai_prediction".to_string(), v);
                            }

                            // Web search options
                            if let Some(ref ws) = options.web_search_options
                                && let Ok(v) = serde_json::to_value(ws)
                            {
                                input
                                    .extra
                                    .insert("openai_web_search_options".to_string(), v);
                            }
                        }

                        input
                    }

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

                struct EmbRequestBridge(
                    std::sync::Arc<
                        dyn siumai_core::execution::embedding::EmbeddingRequestTransformer,
                    >,
                );
                impl crate::execution::transformers::request::RequestTransformer for EmbRequestBridge {
                    fn provider_id(&self) -> &str {
                        self.0.provider_id()
                    }
                    fn transform_chat(
                        &self,
                        _req: &crate::types::ChatRequest,
                    ) -> Result<serde_json::Value, LlmError> {
                        Err(LlmError::UnsupportedOperation(
                            "Chat is not supported by embedding transformer".to_string(),
                        ))
                    }
                    fn transform_embedding(
                        &self,
                        req: &crate::types::EmbeddingRequest,
                    ) -> Result<serde_json::Value, LlmError> {
                        let fmt = req.encoding_format.as_ref().map(|f| match f {
                            crate::types::embedding::EmbeddingFormat::Float => "float".to_string(),
                            crate::types::embedding::EmbeddingFormat::Base64 => {
                                "base64".to_string()
                            }
                        });
                        let input = siumai_core::execution::embedding::EmbeddingInput {
                            input: req.input.clone(),
                            model: req.model.clone(),
                            dimensions: req.dimensions,
                            encoding_format: fmt,
                            user: req.user.clone(),
                            title: req.title.clone(),
                        };
                        self.0.transform_embedding(&input)
                    }
                }

                struct EmbResponseBridge(
                    std::sync::Arc<
                        dyn siumai_core::execution::embedding::EmbeddingResponseTransformer,
                    >,
                );
                impl crate::execution::transformers::response::ResponseTransformer for EmbResponseBridge {
                    fn provider_id(&self) -> &str {
                        self.0.provider_id()
                    }
                    fn transform_embedding_response(
                        &self,
                        raw: &serde_json::Value,
                    ) -> Result<crate::types::EmbeddingResponse, LlmError> {
                        let r = self.0.transform_embedding_response(raw)?;
                        let mut out = crate::types::EmbeddingResponse::new(r.embeddings, r.model);
                        if let Some(u) = r.usage {
                            out = out.with_usage(crate::types::embedding::EmbeddingUsage::new(
                                u.prompt_tokens,
                                u.total_tokens,
                            ));
                        }
                        Ok(out)
                    }
                }

                return EmbeddingTransformers {
                    request: std::sync::Arc::new(EmbRequestBridge(req_tx)),
                    response: std::sync::Arc::new(EmbResponseBridge(resp_tx)),
                };
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
            struct ImageOnlyRequestTransformerBridge(
                std::sync::Arc<dyn siumai_core::execution::image::ImageRequestTransformer>,
            );
            impl crate::execution::transformers::request::RequestTransformer
                for ImageOnlyRequestTransformerBridge
            {
                fn provider_id(&self) -> &str {
                    self.0.provider_id()
                }
                fn transform_chat(
                    &self,
                    _req: &crate::types::ChatRequest,
                ) -> Result<serde_json::Value, LlmError> {
                    Err(LlmError::UnsupportedOperation(
                        "Chat is not supported by image transformer".to_string(),
                    ))
                }
                fn transform_image(
                    &self,
                    req: &crate::types::ImageGenerationRequest,
                ) -> Result<serde_json::Value, LlmError> {
                    self.0.transform_image(req)
                }
                fn transform_image_edit(
                    &self,
                    req: &crate::types::ImageEditRequest,
                ) -> Result<crate::execution::transformers::request::ImageHttpBody, LlmError>
                {
                    match self.0.transform_image_edit(req)? {
                        siumai_core::execution::image::ImageHttpBody::Json(v) => {
                            Ok(crate::execution::transformers::request::ImageHttpBody::Json(v))
                        }
                        siumai_core::execution::image::ImageHttpBody::Multipart(f) => Ok(
                            crate::execution::transformers::request::ImageHttpBody::Multipart(f),
                        ),
                    }
                }
                fn transform_image_variation(
                    &self,
                    req: &crate::types::ImageVariationRequest,
                ) -> Result<crate::execution::transformers::request::ImageHttpBody, LlmError>
                {
                    match self.0.transform_image_variation(req)? {
                        siumai_core::execution::image::ImageHttpBody::Json(v) => {
                            Ok(crate::execution::transformers::request::ImageHttpBody::Json(v))
                        }
                        siumai_core::execution::image::ImageHttpBody::Multipart(f) => Ok(
                            crate::execution::transformers::request::ImageHttpBody::Multipart(f),
                        ),
                    }
                }
            }

            struct ImageOnlyResponseTransformerBridge(
                std::sync::Arc<dyn siumai_core::execution::image::ImageResponseTransformer>,
            );
            impl crate::execution::transformers::response::ResponseTransformer
                for ImageOnlyResponseTransformerBridge
            {
                fn provider_id(&self) -> &str {
                    self.0.provider_id()
                }
                fn transform_image_response(
                    &self,
                    raw: &serde_json::Value,
                ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
                    self.0.transform_image_response(raw)
                }
            }

            crate::core::ImageTransformers {
                request: std::sync::Arc::new(ImageOnlyRequestTransformerBridge(
                    transformers.request,
                )),
                response: std::sync::Arc::new(ImageOnlyResponseTransformerBridge(
                    transformers.response,
                )),
            }
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
