use crate::core::{
    ChatTransformers, EmbeddingTransformers, ProviderContext, ProviderSpec, RerankTransformers,
};
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::standards::openai::chat::OpenAiChatStandard;
use crate::standards::openai::embedding::OpenAiEmbeddingStandard;
use crate::standards::openai::image::OpenAiImageStandard;
use crate::standards::openai::rerank::OpenAiRerankStandard;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, EmbeddingRequest, ProviderOptions, RerankRequest};
use reqwest::header::HeaderMap;
use std::sync::Arc;

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct OpenAiOptionsFromMap {
    #[serde(default)]
    responses_api: Option<crate::provider_options::openai::ResponsesApiConfig>,
    #[serde(default)]
    provider_tools: Vec<crate::types::Tool>,
    #[serde(default)]
    reasoning_effort: Option<crate::provider_options::openai::ReasoningEffort>,
    #[serde(default)]
    service_tier: Option<crate::provider_options::openai::ServiceTier>,
    #[serde(default)]
    modalities: Option<Vec<crate::provider_options::openai::ChatCompletionModalities>>,
    #[serde(default)]
    audio: Option<crate::provider_options::openai::ChatCompletionAudio>,
    #[serde(default)]
    prediction: Option<crate::provider_options::openai::PredictionContent>,
    #[serde(default)]
    web_search_options: Option<crate::provider_options::openai::OpenAiWebSearchOptions>,
}

/// OpenAI ProviderSpec implementation
///
/// This spec uses the OpenAI standards from the standards layer,
/// with additional support for OpenAI-specific features like Responses API.
#[derive(Clone, Default)]
pub struct OpenAiSpec {
    /// Standard OpenAI Chat implementation
    chat_standard: OpenAiChatStandard,
    /// Standard OpenAI Embedding implementation
    embedding_standard: OpenAiEmbeddingStandard,
    /// Standard OpenAI Image implementation
    image_standard: OpenAiImageStandard,
    /// Optional forced Responses API configuration.
    ///
    /// This is primarily used by the provider-specific `OpenAiClient` builder
    /// to route all chat requests through `/responses` without requiring every
    /// request to carry `ProviderOptions::OpenAi`.
    forced_responses_api: Option<crate::provider_options::openai::ResponsesApiConfig>,
}

impl OpenAiSpec {
    pub fn new() -> Self {
        Self {
            chat_standard: OpenAiChatStandard::new(),
            embedding_standard: OpenAiEmbeddingStandard::new(),
            image_standard: OpenAiImageStandard::new(),
            forced_responses_api: None,
        }
    }

    pub fn with_forced_responses_api(
        mut self,
        cfg: crate::provider_options::openai::ResponsesApiConfig,
    ) -> Self {
        self.forced_responses_api = Some(cfg);
        self
    }

    fn use_responses_api(&self, req: &ChatRequest, _ctx: &ProviderContext) -> bool {
        if let Some(cfg) = self.forced_responses_api.as_ref() {
            return cfg.enabled;
        }

        if let Some(cfg) = self.responses_api_config_from_provider_options_map(req) {
            return cfg.enabled;
        }

        // Check if Responses API is configured in provider_options
        if let ProviderOptions::OpenAi(ref value) = req.provider_options {
            if let Some(opts) = self.openai_options_from_provider_options_value(value) {
                return opts
                    .responses_api
                    .as_ref()
                    .map(|cfg| cfg.enabled)
                    .unwrap_or(false);
            }
        }

        false
    }

    fn normalize_openai_provider_options_json(&self, value: &serde_json::Value) -> serde_json::Value {
        fn normalize_key(k: &str) -> Option<&'static str> {
            Some(match k {
                // OpenAiOptions
                "responsesApi" => "responses_api",
                "providerTools" => "provider_tools",
                "reasoningEffort" => "reasoning_effort",
                "serviceTier" => "service_tier",
                "webSearchOptions" => "web_search_options",
                // ResponsesApiConfig
                "previousResponseId" => "previous_response_id",
                "promptCacheKey" => "prompt_cache_key",
                "responseFormat" => "response_format",
                "maxToolCalls" => "max_tool_calls",
                "parallelToolCalls" => "parallel_tool_calls",
                "textVerbosity" => "text_verbosity",
                _ => return None,
            })
        }

        fn inner(
            this: &OpenAiSpec,
            value: &serde_json::Value,
            parent_key: Option<&str>,
        ) -> serde_json::Value {
            match value {
                serde_json::Value::Object(map) => {
                    let mut out = serde_json::Map::new();
                    for (k, v) in map {
                        let nk = normalize_key(k).unwrap_or(k);
                        out.insert(nk.to_string(), inner(this, v, Some(nk)));
                    }
                    if parent_key == Some("responses_api") && !out.contains_key("enabled") {
                        out.insert("enabled".to_string(), serde_json::Value::Bool(true));
                    }
                    serde_json::Value::Object(out)
                }
                serde_json::Value::Array(arr) => serde_json::Value::Array(
                    arr.iter()
                        .map(|v| inner(this, v, parent_key))
                        .collect(),
                ),
                other => other.clone(),
            }
        }

        inner(self, value, None)
    }

    fn openai_options_from_provider_options_map(
        &self,
        req: &ChatRequest,
    ) -> Option<OpenAiOptionsFromMap> {
        let value = req.provider_options_map.get("openai")?;
        let normalized = self.normalize_openai_provider_options_json(value);
        serde_json::from_value(normalized).ok()
    }

    fn openai_options_from_provider_options_value(
        &self,
        value: &serde_json::Value,
    ) -> Option<OpenAiOptionsFromMap> {
        let normalized = self.normalize_openai_provider_options_json(value);
        serde_json::from_value(normalized).ok()
    }

    fn responses_api_config_from_provider_options_map(
        &self,
        req: &ChatRequest,
    ) -> Option<crate::provider_options::openai::ResponsesApiConfig> {
        let value = req.provider_options_map.get("openai")?;
        let normalized = self.normalize_openai_provider_options_json(value);
        let obj = normalized.as_object()?;
        let responses_api = obj.get("responses_api")?;
        serde_json::from_value(responses_api.clone()).ok()
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
            .with_audio()
            .with_file_management()
            .with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("OpenAI API key not provided".into()))?;
        ProviderHeaders::openai(
            api_key,
            ctx.organization.as_deref(),
            ctx.project.as_deref(),
            &ctx.http_extra_headers,
        )
    }

    fn chat_url(&self, _stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        let use_responses = self.use_responses_api(req, ctx);
        let suffix = if use_responses {
            "/responses"
        } else {
            "/chat/completions"
        };
        format!("{}{}", ctx.base_url.trim_end_matches('/'), suffix)
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
            let converter =
                crate::providers::openai::responses::OpenAiResponsesEventConverter::new();
            let stream_tx =
                crate::providers::openai::transformers::OpenAiResponsesStreamChunkTransformer {
                    provider_id: "openai_responses".to_string(),
                    inner: converter,
                };
            ChatTransformers {
                request: Arc::new(req_tx),
                response: Arc::new(resp_tx),
                stream: Some(Arc::new(stream_tx)),
                json: None,
            }
        } else {
            // Use standard OpenAI Chat API from standards layer
            let spec = self.chat_standard.create_spec("openai");
            spec.choose_chat_transformers(req, ctx)
        }
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // 1. First check for CustomProviderOptions (using default implementation)
        if let Some(hook) = crate::core::default_custom_options_hook(self.id(), req) {
            return Some(hook);
        }

        // 2. Handle OpenAI-specific options (built_in_tools, responses_api, modalities, audio, prediction, web_search_options, etc.)
        // Extract options from provider_options (or forced config)
        let use_responses_api = self.use_responses_api(req, ctx);
        let (
            builtins,
            responses_api_config,
            reasoning_effort,
            service_tier,
            modalities,
            audio,
            prediction,
            web_search_options,
        ) = if let ProviderOptions::OpenAi(ref value) = req.provider_options {
            let options = self.openai_options_from_provider_options_value(value)?;
            #[allow(deprecated)]
            {
                // Vercel-aligned: provider-defined tools are supported on the Responses API path.
                // Keep `OpenAiOptions.provider_tools` injection as a compatibility layer.
                let tools_json: Vec<serde_json::Value> = if use_responses_api {
                    crate::providers::openai::utils::convert_tools_to_responses_format(
                        &options.provider_tools,
                    )
                    .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let builtins = if tools_json.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Array(tools_json))
                };
                (
                    builtins,
                    options
                        .responses_api
                        .clone()
                        .filter(|cfg| cfg.enabled)
                        .or_else(|| {
                            if use_responses_api {
                                self.forced_responses_api.clone()
                            } else {
                                None
                            }
                        }),
                    options.reasoning_effort,
                    options.service_tier,
                    options.modalities.clone(),
                    options.audio.clone(),
                    options.prediction.clone(),
                    options.web_search_options.clone(),
                )
            }
        } else if let Some(options) = self.openai_options_from_provider_options_map(req) {
            let tools_json: Vec<serde_json::Value> = if use_responses_api {
                crate::providers::openai::utils::convert_tools_to_responses_format(
                    &options.provider_tools,
                )
                .unwrap_or_default()
            } else {
                Vec::new()
            };
            let builtins = if tools_json.is_empty() {
                None
            } else {
                Some(serde_json::Value::Array(tools_json))
            };

            (
                builtins,
                options
                    .responses_api
                    .clone()
                    .filter(|cfg| cfg.enabled)
                    .or_else(|| {
                        if use_responses_api {
                            self.forced_responses_api.clone()
                        } else {
                            None
                        }
                    }),
                options.reasoning_effort,
                options.service_tier,
                options.modalities.clone(),
                options.audio.clone(),
                options.prediction.clone(),
                options.web_search_options.clone(),
            )
        } else {
            // If the client forces the Responses API, we still allow injecting the
            // Responses API config even without request-level OpenAI options.
            if use_responses_api {
                (
                    None,
                    self.forced_responses_api.clone(),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            } else {
                return None;
            }
        };

        let builtins = builtins.unwrap_or(serde_json::Value::Array(vec![]));

        // Extract all Responses API config fields
        let prev_id = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.previous_response_id.clone());
        let prompt_cache_key = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.prompt_cache_key.clone());
        let response_format = responses_api_config.as_ref().and_then(|cfg| cfg.response_format.clone());
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
        let has_prompt_cache_key = prompt_cache_key.is_some();
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

        if !has_builtins
            && !has_prev_id
            && !has_prompt_cache_key
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
            if let Some(key) = &prompt_cache_key {
                out["prompt_cache_key"] = serde_json::Value::String(key.clone());
            }
            if let Some(fmt) = &response_format {
                // OpenAI Responses API: structured output is configured via `text.format`.
                //
                // Backward-compat: accept the Chat Completions `response_format` JSON schema shape
                // (`{ type, json_schema: {...} }`) and translate into the Responses shape
                // (`{ type, name, schema, strict, ... }`) when possible.
                let normalized_format = if fmt.get("json_schema").is_some() {
                    // Chat Completions shape: { type: "json_schema", json_schema: { name, schema, strict, ... } }
                    // Responses shape: { type: "json_schema", name, schema, strict, ... }
                    let typ = fmt.get("type").cloned().unwrap_or(serde_json::json!("json_schema"));
                    let inner = fmt.get("json_schema").cloned().unwrap_or(serde_json::Value::Null);
                    if let (Some(name), Some(schema)) = (
                        inner.get("name").cloned(),
                        inner.get("schema").cloned(),
                    ) {
                        let strict = inner.get("strict").cloned();
                        let description = inner.get("description").cloned();
                        let mut obj = serde_json::Map::new();
                        obj.insert("type".to_string(), typ);
                        obj.insert("name".to_string(), name);
                        obj.insert("schema".to_string(), schema);
                        if let Some(v) = strict {
                            obj.insert("strict".to_string(), v);
                        }
                        if let Some(v) = description {
                            obj.insert("description".to_string(), v);
                        }
                        serde_json::Value::Object(obj)
                    } else {
                        fmt.clone()
                    }
                } else {
                    fmt.clone()
                };

                if let Some(text_obj) = out.get_mut("text") {
                    if let Some(text_map) = text_obj.as_object_mut() {
                        text_map.insert("format".to_string(), normalized_format);
                    }
                } else {
                    out["text"] = serde_json::json!({
                        "format": normalized_format
                    });
                }
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
                if use_responses_api {
                    // Responses API: configured via `reasoning.effort`.
                    if let Some(reasoning_obj) = out.get_mut("reasoning") {
                        if let Some(map) = reasoning_obj.as_object_mut() {
                            map.insert("effort".to_string(), val);
                        }
                    } else {
                        out["reasoning"] = serde_json::json!({ "effort": val });
                    }
                } else {
                    // Chat Completions API: `reasoning_effort`.
                    out["reasoning_effort"] = val;
                }
            }

            // 🎯 Inject service_tier
            if let Some(ref tier) = service_tier
                && let Ok(val) = serde_json::to_value(tier)
            {
                out["service_tier"] = val;
            }

            // 🎯 Inject modalities (for multimodal audio output)
            if !use_responses_api
                && let Some(ref mods) = modalities
                && let Ok(val) = serde_json::to_value(mods)
            {
                out["modalities"] = val;
            }

            // 🎯 Inject audio configuration (voice and format for audio output)
            if !use_responses_api
                && let Some(ref aud) = audio
                && let Ok(val) = serde_json::to_value(aud)
            {
                out["audio"] = val;
            }

            // 🎯 Inject prediction (Predicted Outputs for faster response times)
            if !use_responses_api
                && let Some(ref pred) = prediction
                && let Ok(val) = serde_json::to_value(pred)
            {
                out["prediction"] = val;
            }

            // 🎯 Inject web_search_options (context size and user location)
            if !use_responses_api
                && let Some(ref ws_opts) = web_search_options
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
        // Use standard OpenAI Embedding API from standards layer
        EmbeddingTransformers {
            request: self.embedding_standard.create_request_transformer("openai"),
            response: self
                .embedding_standard
                .create_response_transformer("openai"),
        }
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> crate::core::ImageTransformers {
        // Use standard OpenAI Image API from standards layer
        let transformers = self.image_standard.create_transformers("openai");
        crate::core::ImageTransformers {
            request: transformers.request,
            response: transformers.response,
        }
    }

    fn choose_rerank_transformers(
        &self,
        _req: &RerankRequest,
        ctx: &ProviderContext,
    ) -> RerankTransformers {
        let std = OpenAiRerankStandard::new();
        let t = std.create_transformers(&ctx.provider_id);
        RerankTransformers {
            request: t.request,
            response: t.response,
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

/// Wrapper spec that enables the rerank capability bit.
///
/// Rationale:
/// - OpenAI itself does not expose a rerank endpoint.
/// - Some OpenAI-compatible providers do (e.g., SiliconFlow), and users may still route
///   through the OpenAI protocol stack via a custom base URL.
/// - The rerank executor uses `ProviderSpec::capabilities()` as a guard, so rerank must be
///   explicitly opted-in for those configurations.
#[derive(Clone)]
pub struct OpenAiSpecWithRerank {
    inner: OpenAiSpec,
}

impl OpenAiSpecWithRerank {
    pub fn new() -> Self {
        Self {
            inner: OpenAiSpec::new(),
        }
    }
}

impl Default for OpenAiSpecWithRerank {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderSpec for OpenAiSpecWithRerank {
    fn id(&self) -> &'static str {
        self.inner.id()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities().with_rerank()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        self.inner.build_headers(ctx)
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        self.inner.chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        self.inner.choose_chat_transformers(req, ctx)
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        self.inner.chat_before_send(req, ctx)
    }

    fn embedding_url(&self, req: &EmbeddingRequest, ctx: &ProviderContext) -> String {
        self.inner.embedding_url(req, ctx)
    }

    fn choose_embedding_transformers(
        &self,
        req: &EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        self.inner.choose_embedding_transformers(req, ctx)
    }

    fn embedding_before_send(
        &self,
        req: &EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        self.inner.embedding_before_send(req, ctx)
    }

    fn image_url(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        self.inner.image_url(req, ctx)
    }

    fn image_edit_url(
        &self,
        req: &crate::types::ImageEditRequest,
        ctx: &ProviderContext,
    ) -> String {
        self.inner.image_edit_url(req, ctx)
    }

    fn image_variation_url(
        &self,
        req: &crate::types::ImageVariationRequest,
        ctx: &ProviderContext,
    ) -> String {
        self.inner.image_variation_url(req, ctx)
    }

    fn choose_image_transformers(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> crate::core::ImageTransformers {
        self.inner.choose_image_transformers(req, ctx)
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        self.inner.audio_base_url(ctx)
    }

    fn choose_audio_transformer(&self, ctx: &ProviderContext) -> crate::core::AudioTransformer {
        self.inner.choose_audio_transformer(ctx)
    }

    fn files_base_url(&self, ctx: &ProviderContext) -> String {
        self.inner.files_base_url(ctx)
    }

    fn choose_files_transformer(&self, ctx: &ProviderContext) -> crate::core::FilesTransformer {
        self.inner.choose_files_transformer(ctx)
    }

    fn rerank_url(&self, req: &RerankRequest, ctx: &ProviderContext) -> String {
        self.inner.rerank_url(req, ctx)
    }

    fn choose_rerank_transformers(
        &self,
        req: &RerankRequest,
        ctx: &ProviderContext,
    ) -> RerankTransformers {
        self.inner.choose_rerank_transformers(req, ctx)
    }

    fn rerank_before_send(
        &self,
        req: &RerankRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        self.inner.rerank_before_send(req, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_spec_declares_image_audio_files_capabilities() {
        let caps = OpenAiSpec::new().capabilities();
        assert!(
            caps.supports("image_generation"),
            "OpenAiSpec must declare image_generation=true to pass HttpImageExecutor capability guards"
        );
        assert!(
            caps.supports("audio"),
            "OpenAiSpec must declare audio=true to pass HttpAudioExecutor capability guards"
        );
        assert!(
            caps.supports("file_management"),
            "OpenAiSpec must declare file_management=true to pass HttpFilesExecutor capability guards"
        );
    }

    #[test]
    fn openai_spec_with_rerank_declares_rerank_capability() {
        let caps = OpenAiSpecWithRerank::new().capabilities();
        assert!(caps.supports("rerank"));
    }

    #[test]
    fn openai_spec_uses_provider_options_map_for_responses_api() {
        let spec = OpenAiSpec::new();
        let ctx = ProviderContext::new(
            "openai",
            "https://api.openai.com/v1",
            Some("KEY".to_string()),
            std::collections::HashMap::new(),
        );

        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option("openai", serde_json::json!({ "responsesApi": { "enabled": true } }));

        let url = spec.chat_url(false, &req, &ctx);
        assert!(url.ends_with("/responses"));
    }

    #[test]
    fn openai_spec_injects_previous_response_id_from_provider_options_map() {
        let spec = OpenAiSpec::new();
        let ctx = ProviderContext::new(
            "openai",
            "https://api.openai.com/v1",
            Some("KEY".to_string()),
            std::collections::HashMap::new(),
        );
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "openai",
                serde_json::json!({ "responsesApi": { "previousResponseId": "resp_123" } }),
            );

        let hook = spec.chat_before_send(&req, &ctx).expect("hook");
        let out = hook(&serde_json::json!({})).expect("hook ok");
        assert_eq!(out["previous_response_id"], "resp_123");
    }

    #[test]
    fn openai_spec_injects_text_format_from_provider_options_map() {
        let spec = OpenAiSpec::new();
        let ctx = ProviderContext::new(
            "openai",
            "https://api.openai.com/v1",
            Some("KEY".to_string()),
            std::collections::HashMap::new(),
        );
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "openai",
                serde_json::json!({
                    "responsesApi": {
                        "responseFormat": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "my_schema",
                                "schema": { "type": "object" },
                                "strict": true
                            }
                        }
                    }
                }),
            );

        let hook = spec.chat_before_send(&req, &ctx).expect("hook");
        let out = hook(&serde_json::json!({})).expect("hook ok");
        assert!(out["text"]["format"].is_object());
        assert_eq!(out["text"]["format"]["type"], "json_schema");
    }
}
