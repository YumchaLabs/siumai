use crate::core::{ChatTransformers, EmbeddingTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
#[cfg(not(feature = "provider-openai-external"))]
use crate::execution::http::headers::ProviderHeaders;
#[cfg(feature = "openai-compatible")]
use crate::std_openai::openai::chat::OpenAiChatStandard;
#[cfg(feature = "openai-compatible")]
use crate::std_openai::openai::embedding::OpenAiEmbeddingStandard;
use crate::std_openai::openai::image::OpenAiImageStandard;
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
            chat_standard: OpenAiChatStandard::new(),
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
            let req_tx = crate::providers::openai::transformers::OpenAiResponsesRequestTransformer;
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
            ChatTransformers {
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
            }
        } else {
            #[cfg(feature = "openai-compatible")]
            {
                // Use standard OpenAI Chat API from standards layer
                #[cfg(feature = "std-openai-external")]
                {
                    // Bridge external core-only chat transformers into aggregator traits
                    let req_tx = self.chat_standard.create_request_transformer("openai");
                    let resp_tx = self.chat_standard.create_response_transformer("openai");
                    let sse_tx = self.chat_standard.create_stream_converter("openai");

                    struct ChatRequestBridge(
                        std::sync::Arc<dyn siumai_core::execution::chat::ChatRequestTransformer>,
                    );
                    impl crate::execution::transformers::request::RequestTransformer for ChatRequestBridge {
                        fn provider_id(&self) -> &str {
                            self.0.provider_id()
                        }
                        fn transform_chat(
                            &self,
                            req: &crate::types::ChatRequest,
                        ) -> Result<serde_json::Value, LlmError> {
                            // Minimal mapping from aggregator ChatRequest to core ChatInput
                            let msgs = req
                                .messages
                                .iter()
                                .map(|m| {
                                    let role = match m.role {
                                        crate::types::MessageRole::System => {
                                            siumai_core::execution::chat::ChatRole::System
                                        }
                                        crate::types::MessageRole::User => {
                                            siumai_core::execution::chat::ChatRole::User
                                        }
                                        crate::types::MessageRole::Assistant => {
                                            siumai_core::execution::chat::ChatRole::Assistant
                                        }
                                        _ => siumai_core::execution::chat::ChatRole::User,
                                    };
                                    let content = m.content.all_text();
                                    siumai_core::execution::chat::ChatMessageInput { role, content }
                                })
                                .collect::<Vec<_>>();
                            let input = siumai_core::execution::chat::ChatInput {
                                messages: msgs,
                                model: Some(req.common_params.model.clone()),
                                max_tokens: req.common_params.max_tokens,
                                temperature: req.common_params.temperature,
                                top_p: req.common_params.top_p,
                                presence_penalty: None,
                                frequency_penalty: None,
                                stop: req.common_params.stop_sequences.clone(),
                                extra: Default::default(),
                            };
                            self.0.transform_chat(&input)
                        }
                    }

                    struct ChatResponseBridge(
                        std::sync::Arc<dyn siumai_core::execution::chat::ChatResponseTransformer>,
                    );
                    impl crate::execution::transformers::response::ResponseTransformer for ChatResponseBridge {
                        fn provider_id(&self) -> &str {
                            self.0.provider_id()
                        }
                        fn transform_chat_response(
                            &self,
                            raw: &serde_json::Value,
                        ) -> Result<crate::types::ChatResponse, LlmError> {
                            let r = self.0.transform_chat_response(raw)?;
                            // Map ChatResult → ChatResponse
                            let mut resp = crate::types::ChatResponse::new(
                                crate::types::MessageContent::Text(r.content),
                            );
                            // finish_reason mapping
                            if let Some(fr) = r.finish_reason.as_deref() {
                                use crate::types::FinishReason as FR;
                                let mapped = match fr {
                                    "stop" => FR::Stop,
                                    "length" => FR::Length,
                                    "content_filter" => FR::ContentFilter,
                                    "tool_calls" => FR::ToolCalls,
                                    other => FR::Other(other.to_string()),
                                };
                                resp.finish_reason = Some(mapped);
                            }
                            if let Some(u) = r.usage {
                                resp.usage = Some(crate::types::Usage::new(
                                    u.prompt_tokens,
                                    u.completion_tokens,
                                ));
                            }
                            Ok(resp)
                        }
                    }

                    // Stream bridge
                    struct ChatStreamBridge(
                        std::sync::Arc<
                            dyn siumai_core::execution::streaming::ChatStreamEventConverterCore,
                        >,
                    );
                    impl crate::execution::transformers::stream::StreamChunkTransformer for ChatStreamBridge {
                        fn provider_id(&self) -> &str {
                            self.0.provider_id()
                        }
                        fn convert_event(
                            &self,
                            event: eventsource_stream::Event,
                        ) -> crate::execution::transformers::stream::StreamEventFuture<'_>
                        {
                            let events = self.0.convert_event(event)
                                .into_iter()
                                .map(|r| r.map(|e| match e {
                                    siumai_core::execution::streaming::ChatStreamEventCore::ContentDelta { delta, index } => crate::streaming::ChatStreamEvent::ContentDelta { delta, index },
                                    siumai_core::execution::streaming::ChatStreamEventCore::ToolCallDelta { id, function_name, arguments_delta, index } => crate::streaming::ChatStreamEvent::ToolCallDelta {
                                        id: id.unwrap_or_default(),
                                        function_name,
                                        arguments_delta,
                                        index,
                                    },
                                    siumai_core::execution::streaming::ChatStreamEventCore::ThinkingDelta { delta } => crate::streaming::ChatStreamEvent::ThinkingDelta { delta },
                                    siumai_core::execution::streaming::ChatStreamEventCore::UsageUpdate { prompt_tokens, completion_tokens, .. } => crate::streaming::ChatStreamEvent::UsageUpdate { usage: crate::types::Usage::new(prompt_tokens, completion_tokens) },
                                    siumai_core::execution::streaming::ChatStreamEventCore::StreamStart {} => crate::streaming::ChatStreamEvent::StreamStart { metadata: crate::types::ResponseMetadata { id: None, model: None, created: None, provider: self.provider_id().to_string(), request_id: None } },
                                    siumai_core::execution::streaming::ChatStreamEventCore::Custom { event_type, data } => crate::streaming::ChatStreamEvent::Custom { event_type, data },
                                    siumai_core::execution::streaming::ChatStreamEventCore::Error { error } => crate::streaming::ChatStreamEvent::Error { error },
                                }))
                                .collect::<Vec<_>>();
                            Box::pin(async move { events })
                        }
                    }

                    return ChatTransformers {
                        request: std::sync::Arc::new(ChatRequestBridge(req_tx)),
                        response: std::sync::Arc::new(ChatResponseBridge(resp_tx)),
                        stream: Some(std::sync::Arc::new(ChatStreamBridge(sse_tx))),
                        json: None,
                    };
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
        // 1. First check for CustomProviderOptions (using default implementation)
        if let Some(hook) = crate::core::default_custom_options_hook(self.id(), req) {
            return Some(hook);
        }

        // 2. Handle OpenAI-specific options (built_in_tools, responses_api, modalities, audio, prediction, web_search_options, etc.)
        // Extract options from provider_options
        let (
            builtins,
            responses_api_config,
            reasoning_effort,
            service_tier,
            modalities,
            audio,
            prediction,
            web_search_options,
        ) = if let ProviderOptions::OpenAi(ref options) = req.provider_options {
            // Build built-in tools JSON from provider_tools using appropriate format
            let use_responses = self.use_responses_api(req, _ctx);
            let tools_json: Vec<serde_json::Value> = if use_responses {
                crate::providers::openai::utils::convert_tools_to_responses_format(
                    &options.provider_tools,
                )
                .unwrap_or_default()
            } else {
                crate::providers::openai::utils::convert_tools_to_openai_format(
                    &options.provider_tools,
                )
                .unwrap_or_default()
            };
            let builtins = if tools_json.is_empty() {
                None
            } else {
                Some(serde_json::Value::Array(tools_json))
            };
            (
                builtins,
                options.responses_api.clone(),
                options.reasoning_effort,
                options.service_tier,
                options.modalities.clone(),
                options.audio.clone(),
                options.prediction.clone(),
                options.web_search_options.clone(),
            )
        } else {
            return None;
        };

        let builtins = builtins.unwrap_or(serde_json::Value::Array(vec![]));

        // Extract all Responses API config fields
        let prev_id = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.previous_response_id.clone());
        let response_format = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.response_format.clone())
            .and_then(|fmt| serde_json::to_value(fmt).ok());
        let background = responses_api_config.as_ref().and_then(|cfg| cfg.background);
        let include = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.include.clone());
        let instructions = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.instructions.clone());
        let max_tool_calls = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.max_tool_calls);
        let store = responses_api_config.as_ref().and_then(|cfg| cfg.store);
        let truncation = responses_api_config.as_ref().and_then(|cfg| cfg.truncation);
        let text_verbosity = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.text_verbosity);
        let metadata = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.metadata.clone());
        let parallel_tool_calls = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.parallel_tool_calls);

        // Check if we need to inject anything
        let has_builtins = matches!(&builtins, serde_json::Value::Array(arr) if !arr.is_empty());
        let has_prev_id = prev_id.is_some();
        let has_response_format = response_format.is_some();
        let has_reasoning_effort = reasoning_effort.is_some();
        let has_service_tier = service_tier.is_some();
        let has_background = background.is_some();
        let has_include = include.is_some();
        let has_instructions = instructions.is_some();
        let has_max_tool_calls = max_tool_calls.is_some();
        let has_store = store.is_some();
        let has_truncation = truncation.is_some();
        let has_text_verbosity = text_verbosity.is_some();
        let has_metadata = metadata.is_some();
        let has_parallel_tool_calls = parallel_tool_calls.is_some();
        let has_modalities = modalities.is_some();
        let has_audio = audio.is_some();
        let has_prediction = prediction.is_some();
        let has_web_search_options = web_search_options.is_some();

        // Also capture whether this is a streaming request so that we can
        // inject OpenAI `stream` flags close to the provider layer instead
        // of the generic executor.
        let is_stream = req.stream;

        if !has_builtins
            && !has_prev_id
            && !has_response_format
            && !has_reasoning_effort
            && !has_service_tier
            && !has_background
            && !has_include
            && !has_instructions
            && !has_max_tool_calls
            && !has_store
            && !has_truncation
            && !has_text_verbosity
            && !has_metadata
            && !has_parallel_tool_calls
            && !has_modalities
            && !has_audio
            && !has_prediction
            && !has_web_search_options
        {
            return None;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();

            // Ensure OpenAI streaming flags are present for streaming calls.
            //
            // This mirrors the previous behavior in the chat executor where
            // `stream = true` and `stream_options.include_usage = true` were
            // injected based on `provider_id.starts_with("openai")`, but moves
            // the provider-specific logic into the provider layer.
            if is_stream {
                out["stream"] = serde_json::Value::Bool(true);
                if out.get("stream_options").is_none() {
                    out["stream_options"] = serde_json::json!({
                        "include_usage": true,
                    });
                } else if let Some(obj) = out["stream_options"].as_object_mut() {
                    obj.entry("include_usage")
                        .or_insert(serde_json::Value::Bool(true));
                }
            }

            // 🎯 Inject built-in tools (merge with existing tools)
            if let serde_json::Value::Array(bi) = &builtins
                && !bi.is_empty()
            {
                let mut arr = out
                    .get("tools")
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();
                for t in bi.iter() {
                    arr.push(t.clone());
                }
                // De-dup logic:
                // - Keep all "function" tools (they have unique names)
                // - For built-in tools (file_search, web_search, computer_use):
                //   - Keep all if they have different configurations
                //   - Only de-dup exact duplicates
                let mut dedup = Vec::new();
                let mut seen_builtins: Vec<serde_json::Value> = Vec::new();

                for item in arr.into_iter() {
                    let typ = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

                    if typ == "function" {
                        // Always keep function tools (they have unique names)
                        dedup.push(item);
                    } else {
                        // For built-in tools, check if we've seen an exact duplicate
                        if !seen_builtins.contains(&item) {
                            seen_builtins.push(item.clone());
                            dedup.push(item);
                        }
                    }
                }
                out["tools"] = serde_json::Value::Array(dedup);
            }

            // 🎯 Inject Responses API fields
            if let Some(pid) = &prev_id {
                out["previous_response_id"] = serde_json::Value::String(pid.clone());
            }
            if let Some(fmt) = &response_format {
                out["response_format"] = fmt.clone();
            }
            if let Some(bg) = background {
                out["background"] = serde_json::Value::Bool(bg);
            }
            if let Some(ref inc) = include
                && let Ok(val) = serde_json::to_value(inc)
            {
                out["include"] = val;
            }
            if let Some(ref instr) = instructions {
                out["instructions"] = serde_json::Value::String(instr.clone());
            }
            if let Some(mtc) = max_tool_calls {
                out["max_tool_calls"] = serde_json::Value::Number(mtc.into());
            }
            if let Some(st) = store {
                out["store"] = serde_json::Value::Bool(st);
            }
            if let Some(ref trunc) = truncation
                && let Ok(val) = serde_json::to_value(trunc)
            {
                out["truncation"] = val;
            }
            if let Some(ref verb) = text_verbosity
                && let Ok(val) = serde_json::to_value(verb)
            {
                // text_verbosity should be nested under "text.verbosity" in Responses API
                if let Some(text_obj) = out.get_mut("text") {
                    if let Some(text_map) = text_obj.as_object_mut() {
                        text_map.insert("verbosity".to_string(), val);
                    }
                } else {
                    // If no "text" field exists, create one with verbosity
                    out["text"] = serde_json::json!({
                        "verbosity": val
                    });
                }
            }
            if let Some(ref meta) = metadata
                && let Ok(val) = serde_json::to_value(meta)
            {
                out["metadata"] = val;
            }
            if let Some(ptc) = parallel_tool_calls {
                out["parallel_tool_calls"] = serde_json::Value::Bool(ptc);
            }

            // 🎯 Inject reasoning_effort
            if let Some(ref effort) = reasoning_effort
                && let Ok(val) = serde_json::to_value(effort)
            {
                out["reasoning_effort"] = val;
            }

            // 🎯 Inject service_tier
            if let Some(ref tier) = service_tier
                && let Ok(val) = serde_json::to_value(tier)
            {
                out["service_tier"] = val;
            }

            // 🎯 Inject modalities (for multimodal audio output)
            if let Some(ref mods) = modalities
                && let Ok(val) = serde_json::to_value(mods)
            {
                out["modalities"] = val;
            }

            // 🎯 Inject audio configuration (voice and format for audio output)
            if let Some(ref aud) = audio
                && let Ok(val) = serde_json::to_value(aud)
            {
                out["audio"] = val;
            }

            // 🎯 Inject prediction (Predicted Outputs for faster response times)
            if let Some(ref pred) = prediction
                && let Ok(val) = serde_json::to_value(pred)
            {
                out["prediction"] = val;
            }

            // 🎯 Inject web_search_options (context size and user location)
            if let Some(ref ws_opts) = web_search_options
                && let Ok(val) = serde_json::to_value(ws_opts)
            {
                out["web_search_options"] = val;
            }

            Ok(out)
        };
        Some(Arc::new(hook))
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
