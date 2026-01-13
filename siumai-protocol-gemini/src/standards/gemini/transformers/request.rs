use super::options::LegacyDynamicRetrievalMode;
use super::*;

/// Request transformer for Gemini
#[derive(Clone)]
pub struct GeminiRequestTransformer {
    pub config: GeminiConfig,
}

impl RequestTransformer for GeminiRequestTransformer {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Minimal validation: require model
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        // Hooks + rules via Generic transformer
        struct GeminiChatHooks(types::GeminiConfig);
        impl ProviderRequestHooks for GeminiChatHooks {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                // Start from typed builder (includes content/messages/tools)
                let typed: GenerateContentRequest =
                    convert::build_request_body(&self.0, &req.messages, req.tools.as_deref())?;
                let mut body = serde_json::to_value(typed)
                    .map_err(|e| LlmError::ParseError(format!("Serialize request failed: {e}")))?;

                // Put common params at top-level for rule-based moving into generationConfig
                if let Some(t) = req.common_params.temperature {
                    let v = (t * 1_000_000.0).round() / 1_000_000.0;
                    body["temperature"] = serde_json::json!(v);
                }
                if let Some(tp) = req.common_params.top_p {
                    let v = (tp * 1_000_000.0).round() / 1_000_000.0;
                    body["top_p"] = serde_json::json!(v);
                }
                if let Some(max) = req.common_params.max_tokens {
                    body["max_tokens"] = serde_json::json!(max);
                }
                if let Some(stops) = &req.common_params.stop_sequences {
                    body["stop_sequences"] = serde_json::json!(stops);
                }
                if let Some(top_k) = req.common_params.top_k {
                    // Gemini expects an integer topK.
                    let rounded = top_k.round();
                    if (top_k - rounded).abs() > 1e-9 {
                        return Err(LlmError::InvalidParameter(
                            "top_k must be an integer for Gemini".to_string(),
                        ));
                    }
                    if rounded < 0.0 || rounded > i32::MAX as f64 {
                        return Err(LlmError::InvalidParameter(
                            "top_k must be between 0 and i32::MAX for Gemini".to_string(),
                        ));
                    }
                    body["top_k"] = serde_json::json!(rounded as i32);
                }
                if let Some(seed) = req.common_params.seed {
                    if seed > i32::MAX as u64 {
                        return Err(LlmError::InvalidParameter(
                            "seed must be <= i32::MAX for Gemini".to_string(),
                        ));
                    }
                    body["seed"] = serde_json::json!(seed as i32);
                }
                if let Some(fp) = req.common_params.frequency_penalty {
                    body["frequency_penalty"] = serde_json::json!(fp);
                }
                if let Some(pp) = req.common_params.presence_penalty {
                    body["presence_penalty"] = serde_json::json!(pp);
                }
                Ok(body)
            }

            fn post_process_chat(
                &self,
                req: &ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                fn denormalize_image_config(value: &serde_json::Value) -> serde_json::Value {
                    let Some(obj) = value.as_object() else {
                        return value.clone();
                    };
                    let mut out = serde_json::Map::new();
                    for (k, v) in obj {
                        match k.as_str() {
                            "aspect_ratio" => out.insert("aspectRatio".to_string(), v.clone()),
                            "image_size" => out.insert("imageSize".to_string(), v.clone()),
                            "aspectRatio" | "imageSize" => out.insert(k.clone(), v.clone()),
                            other => out.insert(other.to_string(), v.clone()),
                        };
                    }
                    serde_json::Value::Object(out)
                }

                fn denormalize_retrieval_config(value: &serde_json::Value) -> serde_json::Value {
                    let Some(obj) = value.as_object() else {
                        return value.clone();
                    };
                    let mut out = serde_json::Map::new();
                    for (k, v) in obj {
                        match k.as_str() {
                            "lat_lng" => out.insert("latLng".to_string(), v.clone()),
                            "latLng" => out.insert(k.clone(), v.clone()),
                            other => out.insert(other.to_string(), v.clone()),
                        };
                    }
                    serde_json::Value::Object(out)
                }

                fn validate_thinking_config(value: &serde_json::Value) -> Result<(), LlmError> {
                    let Some(obj) = value.as_object() else {
                        return Ok(());
                    };
                    let has_budget =
                        obj.get("thinkingBudget").is_some() || obj.get("thinking_budget").is_some();
                    let has_level =
                        obj.get("thinkingLevel").is_some() || obj.get("thinking_level").is_some();
                    if has_budget && has_level {
                        return Err(LlmError::InvalidParameter(
                            "thinkingConfig must not include both thinkingBudget and thinkingLevel"
                                .to_string(),
                        ));
                    }
                    Ok(())
                }

                // Add tool_choice if specified (Gemini toolConfig).
                //
                // Vercel AI SDK alignment:
                // - If provider-defined tools are present, toolChoice does not produce toolConfig.
                // - toolConfig only applies to function calling.
                if req.tools.is_some()
                    && req.tool_choice.is_some()
                    && let Some(choice) = &req.tool_choice
                {
                    let has_provider_tools = req
                        .tools
                        .as_deref()
                        .unwrap_or_default()
                        .iter()
                        .any(|t| matches!(t, crate::types::Tool::ProviderDefined(_)));

                    if !has_provider_tools && body.get("tools").is_some() {
                        body["toolConfig"] = convert::convert_tool_choice(choice);
                    }
                }

                // Provider-specific features:
                // - Prefer provider-defined tools (`Tool::ProviderDefined`) for Google hosted tools.
                // - Keep a deprecated compatibility bridge for `GeminiOptions` fields.
                if let Some(opts) = gemini_options_from_request(req) {
                    // response MIME type escape hatch
                    if let Some(mime) = &opts.response_mime_type {
                        if body
                            .get("generationConfig")
                            .and_then(|v| v.as_object())
                            .is_none()
                        {
                            body["generationConfig"] = serde_json::json!({});
                        }
                        if let Some(obj) = body
                            .get_mut("generationConfig")
                            .and_then(|v| v.as_object_mut())
                        {
                            obj.insert("responseMimeType".to_string(), serde_json::json!(mime));
                        }
                    }

                    // cached content escape hatch (Vercel: providerOptions.google.cachedContent)
                    if let Some(cached) = &opts.cached_content {
                        body["cachedContent"] = serde_json::json!(cached);
                    }

                    // generationConfig provider options (Vercel: providerOptions.google.*)
                    if opts.response_modalities.is_some()
                        || opts.thinking_config.is_some()
                        || opts.audio_timestamp.is_some()
                        || opts.media_resolution.is_some()
                        || opts.image_config.is_some()
                        || opts.response_logprobs.is_some()
                        || opts.logprobs.is_some()
                        || opts.response_json_schema.is_some()
                    {
                        if body
                            .get("generationConfig")
                            .and_then(|v| v.as_object())
                            .is_none()
                        {
                            body["generationConfig"] = serde_json::json!({});
                        }
                        if let Some(obj) = body
                            .get_mut("generationConfig")
                            .and_then(|v| v.as_object_mut())
                        {
                            if let Some(modalities) = &opts.response_modalities {
                                obj.insert(
                                    "responseModalities".to_string(),
                                    serde_json::json!(modalities),
                                );
                            }
                            if let Some(thinking) = &opts.thinking_config {
                                validate_thinking_config(thinking)?;
                                obj.insert(
                                    "thinkingConfig".to_string(),
                                    serde_json::json!(thinking),
                                );
                            }
                            if let Some(audio_ts) = opts.audio_timestamp {
                                obj.insert(
                                    "audioTimestamp".to_string(),
                                    serde_json::json!(audio_ts),
                                );
                            }
                            if let Some(resolution) = &opts.media_resolution {
                                obj.insert(
                                    "mediaResolution".to_string(),
                                    serde_json::json!(resolution),
                                );
                            }
                            if let Some(image_cfg) = &opts.image_config {
                                obj.insert(
                                    "imageConfig".to_string(),
                                    denormalize_image_config(image_cfg),
                                );
                            }
                            if let Some(rjs) = &opts.response_json_schema {
                                obj.insert(
                                    "responseJsonSchema".to_string(),
                                    serde_json::json!(rjs),
                                );
                            }
                            if let Some(enabled) = opts.response_logprobs {
                                obj.insert(
                                    "responseLogprobs".to_string(),
                                    serde_json::json!(enabled),
                                );
                            }
                            if let Some(k) = opts.logprobs {
                                obj.insert("logprobs".to_string(), serde_json::json!(k));
                            }
                        }
                    }

                    // safetySettings (top-level)
                    if let Some(settings) = &opts.safety_settings {
                        body["safetySettings"] = serde_json::json!(settings);
                    }

                    // labels (top-level)
                    if let Some(labels) = &opts.labels {
                        body["labels"] = serde_json::json!(labels);
                    }

                    // retrievalConfig is nested under toolConfig (top-level).
                    if let Some(retrieval) = &opts.retrieval_config {
                        if body.get("toolConfig").and_then(|v| v.as_object()).is_none() {
                            body["toolConfig"] = serde_json::json!({});
                        }
                        if let Some(obj) =
                            body.get_mut("toolConfig").and_then(|v| v.as_object_mut())
                        {
                            obj.insert(
                                "retrievalConfig".to_string(),
                                denormalize_retrieval_config(retrieval),
                            );
                        }
                    }

                    #[allow(deprecated)]
                    {
                        let model = req.common_params.model.as_str();

                        fn is_gemini_2_or_newer(model: &str) -> bool {
                            model.contains("gemini-2")
                                || model.contains("gemini-3")
                                || model.ends_with("-latest")
                        }

                        fn supports_dynamic_retrieval(model: &str) -> bool {
                            model.contains("gemini-1.5-flash") && !model.contains("-8b")
                        }

                        fn supports_file_search(model: &str) -> bool {
                            model.contains("gemini-2.5") || model.contains("gemini-3")
                        }

                        let mut tools = body
                            .get("tools")
                            .and_then(|v| v.as_array().cloned())
                            .unwrap_or_default();

                        if let Some(code_exec) = &opts.code_execution
                            && code_exec.enabled
                        {
                            tools.push(serde_json::json!({ "codeExecution": {} }));
                        }

                        if let Some(search) = &opts.search_grounding
                            && search.enabled
                        {
                            if is_gemini_2_or_newer(model) {
                                tools.push(serde_json::json!({ "googleSearch": {} }));
                            } else {
                                let mut entry = serde_json::json!({ "googleSearchRetrieval": {} });

                                if supports_dynamic_retrieval(model)
                                    && let Some(cfg) = &search.dynamic_retrieval_config
                                {
                                    let mode = match cfg.mode {
                                        LegacyDynamicRetrievalMode::ModeDynamic => "MODE_DYNAMIC",
                                        LegacyDynamicRetrievalMode::ModeUnspecified => {
                                            "MODE_UNSPECIFIED"
                                        }
                                    };
                                    let mut drc = serde_json::json!({ "mode": mode });
                                    if let Some(th) = &cfg.dynamic_threshold {
                                        drc["dynamicThreshold"] =
                                            serde_json::Value::Number(th.clone());
                                    }
                                    entry["googleSearchRetrieval"]["dynamicRetrievalConfig"] = drc;
                                }

                                tools.push(entry);
                            }
                        }

                        if let Some(fs) = &opts.file_search
                            && supports_file_search(model)
                            && !fs.file_search_store_names.is_empty()
                        {
                            tools.push(serde_json::json!({
                                "fileSearch": {
                                    "fileSearchStoreNames": fs.file_search_store_names
                                }
                            }));
                        }

                        if !tools.is_empty() {
                            body["tools"] = serde_json::Value::Array(tools);
                        }
                    }
                }

                // responseFormat (Vercel-aligned): map JSON schema into responseMimeType/responseSchema.
                if let Some(fmt) = &req.response_format {
                    let structured_outputs = gemini_options_from_request(req)
                        .and_then(|o| o.structured_outputs)
                        .unwrap_or(true);

                    if body
                        .get("generationConfig")
                        .and_then(|v| v.as_object())
                        .is_none()
                    {
                        body["generationConfig"] = serde_json::json!({});
                    }

                    if let Some(obj) = body
                        .get_mut("generationConfig")
                        .and_then(|v| v.as_object_mut())
                    {
                        match fmt {
                            crate::types::ResponseFormat::Json { schema } => {
                                obj.insert(
                                    "responseMimeType".to_string(),
                                    serde_json::json!("application/json"),
                                );
                                if structured_outputs {
                                    // Prefer Vercel-aligned structured outputs (`responseSchema`) over
                                    // the legacy `responseJsonSchema` escape hatch if both are set.
                                    obj.remove("responseJsonSchema");
                                    if let Some(openapi) =
                                        crate::standards::gemini::convert::convert_json_schema_to_openapi_schema_root(schema)
                                    {
                                        obj.insert("responseSchema".to_string(), openapi);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
        }

        let hooks = GeminiChatHooks(self.config.clone());
        let profile = MappingProfile {
            provider_id: "gemini",
            rules: vec![
                // Stable ranges only
                Rule::Range {
                    field: "temperature",
                    min: 0.0,
                    max: 2.0,
                    mode: RangeMode::Error,
                    message: Some("temperature must be between 0.0 and 2.0"),
                },
                Rule::Range {
                    field: "top_p",
                    min: 0.0,
                    max: 1.0,
                    mode: RangeMode::Error,
                    message: Some("top_p must be between 0.0 and 1.0"),
                },
                Rule::Range {
                    field: "frequency_penalty",
                    min: -2.0,
                    max: 2.0,
                    mode: RangeMode::Error,
                    message: Some("frequency_penalty must be between -2.0 and 2.0"),
                },
                Rule::Range {
                    field: "presence_penalty",
                    min: -2.0,
                    max: 2.0,
                    mode: RangeMode::Error,
                    message: Some("presence_penalty must be between -2.0 and 2.0"),
                },
                // Move top-level common params into generationConfig (camelCase)
                Rule::Move {
                    from: "temperature",
                    to: "generationConfig.temperature",
                },
                Rule::Move {
                    from: "top_p",
                    to: "generationConfig.topP",
                },
                Rule::Move {
                    from: "top_k",
                    to: "generationConfig.topK",
                },
                Rule::Move {
                    from: "max_tokens",
                    to: "generationConfig.maxOutputTokens",
                },
                Rule::Move {
                    from: "stop_sequences",
                    to: "generationConfig.stopSequences",
                },
                Rule::Move {
                    from: "seed",
                    to: "generationConfig.seed",
                },
                Rule::Move {
                    from: "frequency_penalty",
                    to: "generationConfig.frequencyPenalty",
                },
                Rule::Move {
                    from: "presence_penalty",
                    to: "generationConfig.presencePenalty",
                },
            ],
            // Provider options are injected via ProviderSpec::chat_before_send()
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = GenericRequestTransformer { profile, hooks };
        generic.transform_chat(req)
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        // Use Generic transformer hooks to build typed JSON; no extra rules for now
        struct GeminiEmbeddingHooks(types::GeminiConfig);
        impl crate::execution::transformers::request::ProviderRequestHooks for GeminiEmbeddingHooks {
            fn build_base_embedding_body(
                &self,
                req: &EmbeddingRequest,
            ) -> Result<serde_json::Value, LlmError> {
                // Map to Gemini embedContent / batchEmbedContents request model
                #[derive(serde::Serialize)]
                struct GeminiPart {
                    text: String,
                }
                #[derive(serde::Serialize)]
                struct GeminiContent {
                    #[serde(skip_serializing_if = "Option::is_none")]
                    role: Option<String>,
                    parts: Vec<GeminiPart>,
                }
                #[derive(serde::Serialize)]
                struct GeminiEmbeddingRequest {
                    #[serde(skip_serializing_if = "Option::is_none")]
                    model: Option<String>,
                    content: GeminiContent,
                    #[serde(skip_serializing_if = "Option::is_none", rename = "taskType")]
                    task_type: Option<String>,
                    #[serde(skip_serializing_if = "Option::is_none")]
                    title: Option<String>,
                    #[serde(
                        skip_serializing_if = "Option::is_none",
                        rename = "outputDimensionality"
                    )]
                    output_dimensionality: Option<u32>,
                }
                #[derive(serde::Serialize)]
                struct GeminiBatchEmbeddingRequest {
                    requests: Vec<GeminiEmbeddingRequest>,
                }

                let mut task_type = req.task_type.as_ref().map(|tt| match tt {
                    crate::types::EmbeddingTaskType::RetrievalQuery => {
                        "RETRIEVAL_QUERY".to_string()
                    }
                    crate::types::EmbeddingTaskType::RetrievalDocument => {
                        "RETRIEVAL_DOCUMENT".to_string()
                    }
                    crate::types::EmbeddingTaskType::SemanticSimilarity => {
                        "SEMANTIC_SIMILARITY".to_string()
                    }
                    crate::types::EmbeddingTaskType::Classification => "CLASSIFICATION".to_string(),
                    crate::types::EmbeddingTaskType::Clustering => "CLUSTERING".to_string(),
                    crate::types::EmbeddingTaskType::QuestionAnswering => {
                        "QUESTION_ANSWERING".to_string()
                    }
                    crate::types::EmbeddingTaskType::FactVerification => {
                        "FACT_VERIFICATION".to_string()
                    }
                    crate::types::EmbeddingTaskType::CodeRetrievalQuery => {
                        "CODE_RETRIEVAL_QUERY".to_string()
                    }
                    crate::types::EmbeddingTaskType::Unspecified => {
                        "TASK_TYPE_UNSPECIFIED".to_string()
                    }
                });
                let title = req.title.clone();
                let mut output_dimensionality = req.dimensions;

                // Vercel-aligned: allow providerOptions.google.taskType/outputDimensionality.
                if let Some(opts) = req
                    .provider_options_map
                    .get("gemini")
                    .or_else(|| req.provider_options_map.get("google"))
                    .or_else(|| req.provider_options_map.get("vertex"))
                    && let Some(obj) = opts.as_object()
                {
                    if task_type.is_none()
                        && let Some(tt) = obj.get("taskType").and_then(|v| v.as_str())
                    {
                        task_type = Some(tt.to_string());
                    } else if task_type.is_none()
                        && let Some(tt) = obj.get("task_type").and_then(|v| v.as_str())
                    {
                        task_type = Some(tt.to_string());
                    }

                    if output_dimensionality.is_none()
                        && let Some(dim) = obj.get("outputDimensionality").and_then(|v| v.as_u64())
                    {
                        output_dimensionality = Some(dim as u32);
                    } else if output_dimensionality.is_none()
                        && let Some(dim) = obj.get("output_dimensionality").and_then(|v| v.as_u64())
                    {
                        output_dimensionality = Some(dim as u32);
                    }
                }

                if req.input.len() == 1 {
                    let content = GeminiContent {
                        role: None,
                        parts: vec![GeminiPart {
                            text: req.input[0].clone(),
                        }],
                    };
                    let body = GeminiEmbeddingRequest {
                        model: Some(format!("models/{}", self.0.model)),
                        content,
                        task_type,
                        title,
                        output_dimensionality,
                    };
                    serde_json::to_value(body)
                        .map_err(|e| LlmError::ParseError(format!("Serialize request failed: {e}")))
                } else {
                    let requests: Vec<GeminiEmbeddingRequest> = req
                        .input
                        .iter()
                        .map(|text| {
                            let content = GeminiContent {
                                role: Some("user".to_string()),
                                parts: vec![GeminiPart { text: text.clone() }],
                            };
                            GeminiEmbeddingRequest {
                                model: Some(format!("models/{}", self.0.model)),
                                content,
                                task_type: task_type.clone(),
                                title: title.clone(),
                                output_dimensionality,
                            }
                        })
                        .collect();
                    let batch = GeminiBatchEmbeddingRequest { requests };
                    serde_json::to_value(batch)
                        .map_err(|e| LlmError::ParseError(format!("Serialize request failed: {e}")))
                }
            }
        }
        let hooks = GeminiEmbeddingHooks(self.config.clone());
        let profile = crate::execution::transformers::request::MappingProfile {
            provider_id: "gemini",
            rules: vec![], // no generic rules; hook builds typed JSON
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic =
            crate::execution::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_embedding(req)
    }

    fn transform_image(&self, req: &ImageGenerationRequest) -> Result<serde_json::Value, LlmError> {
        // Use Generic hooks to build typed JSON for generateContent (IMAGE)
        struct GeminiImageHooks(types::GeminiConfig);
        impl crate::execution::transformers::request::ProviderRequestHooks for GeminiImageHooks {
            fn build_base_image_body(
                &self,
                req: &ImageGenerationRequest,
            ) -> Result<serde_json::Value, LlmError> {
                use types::{Content, GenerateContentRequest, ImageConfig, Part};
                if self.0.model.is_empty() {
                    return Err(LlmError::InvalidParameter("Model must be specified".into()));
                }

                let is_imagen = self.0.model.trim().starts_with("imagen-");
                if is_imagen {
                    // Imagen uses `models/{model}:predict` with an `instances` + `parameters` body.
                    let sample_count = if req.count == 0 { 1 } else { req.count };
                    if sample_count > 4 {
                        return Err(LlmError::InvalidParameter(
                            "Imagen models support at most 4 images per call".to_string(),
                        ));
                    }

                    fn get_provider_options<'a>(
                        req: &'a ImageGenerationRequest,
                    ) -> Option<&'a serde_json::Value> {
                        req.provider_options_map
                            .get("gemini")
                            .or_else(|| req.provider_options_map.get("google"))
                            .or_else(|| req.provider_options_map.get("vertex"))
                    }

                    fn get_string_opt(obj: &serde_json::Value, key: &str) -> Option<String> {
                        obj.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
                    }

                    let mut parameters = serde_json::Map::new();
                    parameters.insert("sampleCount".to_string(), serde_json::json!(sample_count));

                    // Prefer providerOptions for Vercel parity, then fall back to extra_params.
                    if let Some(opts) = get_provider_options(req) {
                        let aspect_ratio = get_string_opt(opts, "aspectRatio")
                            .or_else(|| get_string_opt(opts, "aspect_ratio"));
                        if let Some(v) = aspect_ratio {
                            parameters.insert("aspectRatio".to_string(), serde_json::json!(v));
                        }

                        let person_generation = get_string_opt(opts, "personGeneration")
                            .or_else(|| get_string_opt(opts, "person_generation"));
                        if let Some(v) = person_generation {
                            parameters.insert("personGeneration".to_string(), serde_json::json!(v));
                        }
                    }

                    if parameters.get("aspectRatio").is_none()
                        && let Some(v) = req.extra_params.get("aspectRatio")
                    {
                        parameters.insert("aspectRatio".to_string(), v.clone());
                    }
                    if parameters.get("personGeneration").is_none()
                        && let Some(v) = req.extra_params.get("personGeneration")
                    {
                        parameters.insert("personGeneration".to_string(), v.clone());
                    }

                    let body = serde_json::json!({
                        "instances": [{ "prompt": req.prompt }],
                        "parameters": serde_json::Value::Object(parameters),
                    });

                    return Ok(body);
                }

                let prompt = req.prompt.clone();
                let contents = vec![Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::Text {
                        text: prompt,
                        thought: None,
                        thought_signature: None,
                    }],
                }];
                let mut gcfg = self.0.generation_config.clone().unwrap_or_default();
                if req.count > 0 {
                    gcfg.candidate_count = Some(req.count as i32);
                }

                // Allow `mediaResolution` / `imageConfig` from providerOptions on image requests.
                if let Some(opts) = req
                    .provider_options_map
                    .get("gemini")
                    .or_else(|| req.provider_options_map.get("google"))
                    .or_else(|| req.provider_options_map.get("vertex"))
                    && let Some(obj) = opts.as_object()
                {
                    if let Some(res) = obj.get("mediaResolution").and_then(|v| v.as_str()) {
                        gcfg.media_resolution = Some(res.to_string());
                    } else if let Some(res) = obj.get("media_resolution").and_then(|v| v.as_str()) {
                        gcfg.media_resolution = Some(res.to_string());
                    }

                    if let Some(ic) = obj.get("imageConfig").and_then(|v| v.as_object()) {
                        let aspect_ratio = ic
                            .get("aspectRatio")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| {
                                ic.get("aspect_ratio")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            });
                        let image_size = ic
                            .get("imageSize")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| {
                                ic.get("image_size")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            });
                        gcfg.image_config = Some(ImageConfig {
                            aspect_ratio,
                            image_size,
                        });
                    } else if let Some(ic) = obj.get("image_config").and_then(|v| v.as_object()) {
                        let aspect_ratio = ic
                            .get("aspect_ratio")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let image_size = ic
                            .get("image_size")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        gcfg.image_config = Some(ImageConfig {
                            aspect_ratio,
                            image_size,
                        });
                    }
                }

                // Default to requesting both TEXT + IMAGE modalities (Vercel cookbook alignment).
                let mut modalities = gcfg.response_modalities.take().unwrap_or_default();
                if !modalities.iter().any(|m| m == "TEXT") {
                    modalities.insert(0, "TEXT".to_string());
                }
                if !modalities.iter().any(|m| m == "IMAGE") {
                    modalities.push("IMAGE".to_string());
                }
                gcfg.response_modalities = Some(modalities);
                let body = GenerateContentRequest {
                    model: self.0.model.clone(),
                    contents,
                    system_instruction: None,
                    tools: None,
                    tool_config: None,
                    safety_settings: self.0.safety_settings.clone(),
                    generation_config: Some(gcfg),
                    cached_content: None,
                };
                serde_json::to_value(body).map_err(|e| {
                    LlmError::ParseError(format!("Serialize image request failed: {e}"))
                })
            }
        }
        let hooks = GeminiImageHooks(self.config.clone());
        let profile = crate::execution::transformers::request::MappingProfile {
            provider_id: "gemini",
            rules: vec![],
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic =
            crate::execution::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_image(req)
    }
}

#[cfg(test)]
mod tests_gemini_rules {
    use super::*;
    use crate::types::{ChatMessage, Tool};

    #[test]
    fn move_common_params_into_generation_config() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-1.5-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };
        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "gemini-1.5-flash".to_string();
        req.common_params.temperature = Some(0.4);
        req.common_params.top_p = Some(0.9);
        req.common_params.max_tokens = Some(1024);
        req.common_params.stop_sequences = Some(vec!["END".into()]);
        let body = tx.transform_chat(&req).expect("transform");
        let got_temp = body["generationConfig"]["temperature"].as_f64().unwrap();
        assert!((got_temp - 0.4).abs() < 1e-6);
        let got_top_p = body["generationConfig"]["topP"].as_f64().unwrap();
        assert!((got_top_p - 0.9).abs() < 1e-6);
        assert_eq!(
            body["generationConfig"]["maxOutputTokens"],
            serde_json::json!(1024)
        );
        assert_eq!(
            body["generationConfig"]["stopSequences"],
            serde_json::json!(["END"])
        );
    }

    #[test]
    fn provider_options_cached_content_is_mapped() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-1.5-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };

        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "gemini-1.5-flash".to_string();
        let req = req.with_provider_option(
            "gemini",
            serde_json::json!({
                "cachedContent": "cachedContents/test-123"
            }),
        );

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body.get("cachedContent").and_then(|v| v.as_str()),
            Some("cachedContents/test-123")
        );
    }

    #[test]
    fn provider_options_generation_config_fields_are_mapped() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-3-flash-preview".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };

        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "gemini-3-flash-preview".to_string();
        let req = req.with_provider_option(
            "gemini",
            serde_json::json!({
                "responseModalities": ["TEXT"],
                "thinkingConfig": { "thinkingBudget": -1 },
                "audioTimestamp": true
            }),
        );

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body["generationConfig"]["responseModalities"],
            serde_json::json!(["TEXT"])
        );
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            serde_json::json!(-1)
        );
        assert_eq!(
            body["generationConfig"]["audioTimestamp"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn provider_options_safety_settings_and_labels_are_mapped() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-1.5-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };

        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "gemini-1.5-flash".to_string();
        let req = req.with_provider_option(
            "gemini",
            serde_json::json!({
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ],
                "labels": { "env": "dev" }
            }),
        );

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body["safetySettings"][0]["category"],
            serde_json::json!("HARM_CATEGORY_DANGEROUS_CONTENT")
        );
        assert_eq!(
            body["safetySettings"][0]["threshold"],
            serde_json::json!("BLOCK_MEDIUM_AND_ABOVE")
        );
        assert_eq!(body["labels"]["env"], serde_json::json!("dev"));
    }

    #[test]
    fn provider_options_response_mime_type_is_mapped_to_generation_config() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.0-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };

        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "gemini-2.0-flash".to_string();
        let req = req.with_provider_option(
            "gemini",
            serde_json::json!({
                "responseMimeType": "application/json"
            }),
        );

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
    }

    #[test]
    fn provider_options_serialization_locations_and_casing_are_stable() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-3-flash-preview".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };

        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "gemini-3-flash-preview".to_string();
        req.common_params.temperature = Some(0.4);
        req.common_params.top_p = Some(0.9);
        req.common_params.max_tokens = Some(256);
        req.common_params.stop_sequences = Some(vec!["END".into()]);
        let req = req.with_provider_option(
            "gemini",
            serde_json::json!({
                "responseMimeType": "application/json",
                "cachedContent": "cachedContents/test-123",
                "responseModalities": ["TEXT"],
                "thinkingConfig": { "thinkingBudget": -1 },
                "audioTimestamp": true,
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ],
                "labels": { "env": "dev" }
            }),
        );

        let body = tx.transform_chat(&req).expect("transform");

        // Locations: cachedContent/safetySettings/labels are top-level.
        assert_eq!(
            body.get("cachedContent").and_then(|v| v.as_str()),
            Some("cachedContents/test-123")
        );
        assert!(body.get("safetySettings").is_some());
        assert_eq!(body["labels"]["env"], serde_json::json!("dev"));

        // Locations: generationConfig holds generation-related settings (camelCase).
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert_eq!(
            body["generationConfig"]["responseModalities"],
            serde_json::json!(["TEXT"])
        );
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            serde_json::json!(-1)
        );
        assert_eq!(
            body["generationConfig"]["audioTimestamp"],
            serde_json::json!(true)
        );
        assert_eq!(
            body["generationConfig"]["temperature"],
            serde_json::json!(0.4)
        );
        assert_eq!(body["generationConfig"]["topP"], serde_json::json!(0.9));
        assert_eq!(
            body["generationConfig"]["maxOutputTokens"],
            serde_json::json!(256)
        );
        assert_eq!(
            body["generationConfig"]["stopSequences"],
            serde_json::json!(["END"])
        );

        // Ensure snake_case compatibility fields are not leaked into the final body.
        assert!(body.get("top_p").is_none());
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("stop_sequences").is_none());
        assert!(body.get("response_mime_type").is_none());
        assert!(body.get("cached_content").is_none());
        assert!(body.get("safety_settings").is_none());
    }

    #[test]
    fn tool_choice_emits_tool_config_for_function_tools() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.5-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };

        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "testFunction",
                "A test function",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(crate::types::ToolChoice::Required)
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert!(
            body.get("tools").is_some(),
            "expected tools in request body"
        );
        assert!(
            body.get("toolConfig").is_some(),
            "expected toolConfig for function tools"
        );
    }

    #[test]
    fn tool_choice_does_not_emit_tool_config_when_provider_defined_tools_are_present() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.5-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };

        let req = ChatRequest::builder()
            .model("gemini-2.5-flash")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![crate::tools::google::google_search()])
            .tool_choice(crate::types::ToolChoice::Required)
            .build();

        let body = tx.transform_chat(&req).expect("transform");
        assert!(
            body.get("tools").is_some(),
            "expected tools in request body"
        );
        assert!(
            body.get("toolConfig").is_none(),
            "toolConfig should not be emitted for provider-defined tools"
        );
    }

    #[test]
    fn response_format_json_sets_response_mime_type_and_schema() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.5-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };

        let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
        req.common_params.model = "gemini-2.5-flash".to_string();
        req.response_format = Some(crate::types::ResponseFormat::Json {
            schema: serde_json::json!({
                "type": "object",
                "properties": { "a": { "type": "string" } },
                "required": ["a"]
            }),
        });

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert!(body["generationConfig"].get("responseSchema").is_some());
    }

    #[test]
    fn response_format_json_respects_structured_outputs_disabled() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.5-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };

        let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
        req.common_params.model = "gemini-2.5-flash".to_string();
        req.response_format = Some(crate::types::ResponseFormat::Json {
            schema: serde_json::json!({
                "type": "object",
                "properties": { "a": { "type": "string" } },
                "required": ["a"]
            }),
        });
        let req = req.with_provider_option(
            "google",
            serde_json::json!({
                "structuredOutputs": false
            }),
        );

        let body = tx.transform_chat(&req).expect("transform");
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert!(body["generationConfig"].get("responseSchema").is_none());
    }
}
