use super::*;

#[cfg(test)]
mod tests_gemini_metadata {
    use super::*;

    #[test]
    fn gemini_response_populates_provider_metadata_for_grounding_and_url_context() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.0-flash-exp".into())
            .with_base_url("https://example".into());
        let tx = GeminiResponseTransformer { config: cfg };

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": { "parts": [ { "text": "hello" } ] },
                    "safetyRatings": [
                        { "category": "HARM_CATEGORY_DEROGATORY", "probability": "NEGLIGIBLE" }
                    ],
                    "groundingMetadata": {
                        "searchEntryPoint": { "renderedContent": "<div/>" },
                        "groundingChunks": [
                            { "web": { "uri": "https://www.rust-lang.org/", "title": "Rust" } },
                            { "retrievedContext": { "uri": "https://example.com/", "title": "Example" } },
                            { "retrievedContext": { "uri": "gs://bucket/a.pdf", "title": "Spec PDF" } },
                            { "retrievedContext": { "fileSearchStore": "projects/p/locations/l/fileSearchStores/store1", "title": "File Search Store" } },
                            { "maps": { "uri": "https://maps.example.com/place/1", "title": "Maps" } }
                        ]
                    },
                    "urlContextMetadata": {
                        "urlMetadata": [
                            { "retrievedUrl": "https://www.rust-lang.org/", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS" }
                        ]
                    }
                }
            ],
            "promptFeedback": {
                "safetyRatings": [
                    { "category": "HARM_CATEGORY_DEROGATORY", "probability": "NEGLIGIBLE" }
                ]
            },
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 5,
                "totalTokenCount": 8
            },
            "modelVersion": "gemini-2.0-flash-exp"
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let meta = resp
            .provider_metadata
            .as_ref()
            .and_then(|m| m.get("google"))
            .expect("google provider metadata");

        assert!(meta.get("groundingMetadata").is_some());
        assert!(meta.get("urlContextMetadata").is_some());
        assert!(meta.get("safetyRatings").is_some());
        assert!(meta.get("promptFeedback").is_some());
        assert!(meta.get("usageMetadata").is_some());
        assert!(meta.get("sources").is_some());

        let sources = meta
            .get("sources")
            .and_then(|v| v.as_array())
            .expect("sources array");
        assert_eq!(sources.len(), 5);

        assert!(sources.iter().any(|s| {
            s.get("source_type").and_then(|v| v.as_str()) == Some("url")
                && s.get("url").and_then(|v| v.as_str()) == Some("https://www.rust-lang.org/")
        }));
        assert!(sources.iter().any(|s| {
            s.get("source_type").and_then(|v| v.as_str()) == Some("document")
                && s.get("media_type").and_then(|v| v.as_str()) == Some("application/pdf")
                && s.get("filename").and_then(|v| v.as_str()) == Some("a.pdf")
        }));
    }

    #[test]
    fn finish_reason_stop_with_tool_calls_maps_to_tool_calls() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.0-flash-exp".into())
            .with_base_url("https://example".into());
        let tx = GeminiResponseTransformer { config: cfg };

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            { "functionCall": { "name": "weather", "args": { "city": "Tokyo" } } }
                        ]
                    },
                    "finishReason": "STOP"
                }
            ],
            "modelVersion": "gemini-2.0-flash-exp"
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        assert_eq!(resp.finish_reason, Some(FinishReason::ToolCalls));
    }

    #[test]
    fn gemini_response_exposes_logprobs_in_provider_metadata() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.5-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiResponseTransformer { config: cfg };

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": { "parts": [ { "text": "hello" } ] },
                    "avgLogprobs": -0.123,
                    "logprobsResult": {
                        "topCandidates": [
                            { "candidates": [ { "token": "h", "tokenId": 1, "logProbability": -0.1 } ] },
                            { "candidates": [ { "token": "ello", "tokenId": 2, "logProbability": -0.2 } ] }
                        ],
                        "chosenCandidates": [
                            { "token": "h", "tokenId": 1, "logProbability": -0.1 },
                            { "token": "ello", "tokenId": 2, "logProbability": -0.2 }
                        ],
                        "logProbabilitySum": -0.3
                    }
                }
            ],
            "modelVersion": "gemini-2.5-flash"
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let meta = resp
            .provider_metadata
            .as_ref()
            .and_then(|m| m.get("google"))
            .expect("google provider metadata");

        assert_eq!(
            meta.get("avgLogprobs").and_then(|v| v.as_f64()),
            Some(-0.123)
        );
        assert!(meta.get("logprobsResult").is_some());

        let chosen = meta
            .get("logprobsResult")
            .and_then(|v| v.get("chosenCandidates"))
            .and_then(|v| v.as_array())
            .expect("chosenCandidates array");
        assert_eq!(chosen.len(), 2);
    }
}

/// Response transformer for Gemini
#[derive(Clone)]
pub struct GeminiResponseTransformer {
    pub config: GeminiConfig,
}

impl ResponseTransformer for GeminiResponseTransformer {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        fn provider_metadata_key(config: &GeminiConfig) -> &'static str {
            if let Some(override_key) = config.provider_metadata_key.as_deref() {
                let key = override_key.trim().to_ascii_lowercase();
                if key.contains("vertex") {
                    return "vertex";
                }
                if key.contains("google") {
                    return "google";
                }
            }

            if config.base_url.contains("aiplatform.googleapis.com")
                || config.base_url.contains("vertex")
            {
                "vertex"
            } else {
                "google"
            }
        }

        fn thought_signature_provider_metadata(
            provider_key: &str,
            sig: Option<&String>,
        ) -> Option<std::collections::HashMap<String, serde_json::Value>> {
            let sig = sig?;
            if sig.trim().is_empty() {
                return None;
            }
            let mut out = std::collections::HashMap::new();
            out.insert(
                provider_key.to_string(),
                serde_json::json!({ "thoughtSignature": sig }),
            );
            Some(out)
        }

        // Parse typed response and convert to unified ChatResponse (mirrors chat::convert_response)
        let response: GenerateContentResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Gemini response: {e}")))?;

        if response.candidates.is_empty() {
            return Err(LlmError::api_error(400, "No candidates in response"));
        }

        let provider_key = provider_metadata_key(&self.config);

        let candidate = &response.candidates[0];
        let content = candidate
            .content
            .as_ref()
            .ok_or_else(|| LlmError::api_error(400, "No content in candidate"))?;

        let mut text_content = String::new();
        let mut content_parts = Vec::new();
        // Track if any non-text parts are present (kept for future heuristics)
        let mut _has_multimodal_content = false;
        // Pair executableCode -> codeExecutionResult (Vercel-aligned tool events).
        let mut pending_code_execution_id: Option<String> = None;

        for part in &content.parts {
            match part {
                Part::Text {
                    text,
                    thought,
                    thought_signature,
                } => {
                    let provider_metadata = thought_signature_provider_metadata(
                        provider_key,
                        thought_signature.as_ref(),
                    );
                    if thought.unwrap_or(false) {
                        // Add reasoning content
                        content_parts.push(ContentPart::Reasoning {
                            text: text.clone(),
                            provider_metadata,
                        });
                    } else {
                        if !text_content.is_empty() {
                            text_content.push('\n');
                        }
                        text_content.push_str(text);
                        content_parts.push(ContentPart::Text {
                            text: text.clone(),
                            provider_metadata,
                        });
                    }
                }
                Part::InlineData { inline_data, .. } => {
                    _has_multimodal_content = true;
                    if inline_data.mime_type.starts_with("image/") {
                        content_parts.push(crate::types::ContentPart::Image {
                            source: crate::types::chat::MediaSource::Base64 {
                                data: inline_data.data.clone(),
                            },
                            detail: None,
                            provider_metadata: None,
                        });
                    } else if inline_data.mime_type.starts_with("audio/") {
                        content_parts.push(crate::types::ContentPart::Audio {
                            source: crate::types::chat::MediaSource::Base64 {
                                data: inline_data.data.clone(),
                            },
                            media_type: Some(inline_data.mime_type.clone()),
                            provider_metadata: None,
                        });
                    } else {
                        // Other file types
                        content_parts.push(crate::types::ContentPart::File {
                            source: crate::types::chat::MediaSource::Base64 {
                                data: inline_data.data.clone(),
                            },
                            media_type: inline_data.mime_type.clone(),
                            filename: None,
                            provider_metadata: None,
                        });
                    }
                }
                Part::FileData { file_data, .. } => {
                    _has_multimodal_content = true;
                    let mime_type = file_data
                        .mime_type
                        .as_deref()
                        .unwrap_or("application/octet-stream");
                    if mime_type.starts_with("image/") {
                        content_parts.push(crate::types::ContentPart::Image {
                            source: crate::types::chat::MediaSource::Url {
                                url: file_data.file_uri.clone(),
                            },
                            detail: None,
                            provider_metadata: None,
                        });
                    } else if mime_type.starts_with("audio/") {
                        content_parts.push(crate::types::ContentPart::Audio {
                            source: crate::types::chat::MediaSource::Url {
                                url: file_data.file_uri.clone(),
                            },
                            media_type: Some(mime_type.to_string()),
                            provider_metadata: None,
                        });
                    } else {
                        // Other file types
                        content_parts.push(crate::types::ContentPart::File {
                            source: crate::types::chat::MediaSource::Url {
                                url: file_data.file_uri.clone(),
                            },
                            media_type: mime_type.to_string(),
                            filename: None,
                            provider_metadata: None,
                        });
                    }
                }
                Part::FunctionCall {
                    function_call,
                    thought_signature,
                } => {
                    let arguments = function_call
                        .args
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({}));
                    let provider_metadata = thought_signature_provider_metadata(
                        provider_key,
                        thought_signature.as_ref(),
                    );
                    content_parts.push(ContentPart::ToolCall {
                        tool_call_id: format!("call_{}", uuid::Uuid::new_v4()),
                        tool_name: function_call.name.clone(),
                        arguments,
                        provider_executed: None,
                        provider_metadata,
                    });
                }
                Part::ExecutableCode {
                    executable_code,
                    thought_signature,
                } => {
                    let id = format!("call_{}", uuid::Uuid::new_v4());
                    pending_code_execution_id = Some(id.clone());

                    let language = match &executable_code.language {
                        types::CodeLanguage::Python => "PYTHON",
                        types::CodeLanguage::Unspecified => "LANGUAGE_UNSPECIFIED",
                    };

                    let provider_metadata = thought_signature_provider_metadata(
                        provider_key,
                        thought_signature.as_ref(),
                    );
                    content_parts.push(ContentPart::ToolCall {
                        tool_call_id: id,
                        tool_name: "code_execution".to_string(),
                        arguments: serde_json::json!({
                            "language": language,
                            "code": executable_code.code.clone()
                        }),
                        provider_executed: Some(true),
                        provider_metadata,
                    });
                }
                Part::CodeExecutionResult {
                    code_execution_result,
                    thought_signature,
                } => {
                    let id = pending_code_execution_id
                        .take()
                        .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));

                    let outcome = match &code_execution_result.outcome {
                        types::CodeExecutionOutcome::Ok => "OUTCOME_OK",
                        types::CodeExecutionOutcome::Failed => "OUTCOME_FAILED",
                        types::CodeExecutionOutcome::DeadlineExceeded => {
                            "OUTCOME_DEADLINE_EXCEEDED"
                        }
                        types::CodeExecutionOutcome::Unspecified => "OUTCOME_UNSPECIFIED",
                    };

                    let provider_metadata = thought_signature_provider_metadata(
                        provider_key,
                        thought_signature.as_ref(),
                    );
                    content_parts.push(ContentPart::ToolResult {
                        tool_call_id: id,
                        tool_name: "code_execution".to_string(),
                        output: crate::types::ToolResultOutput::json(serde_json::json!({
                            "outcome": outcome,
                            "output": code_execution_result.output.clone()
                        })),
                        provider_executed: Some(true),
                        provider_metadata,
                    });
                }
                _ => {}
            }
        }

        let usage = response.usage_metadata.as_ref().map(|m| Usage {
            prompt_tokens: m.prompt_token_count.unwrap_or(0) as u32,
            completion_tokens: m.candidates_token_count.unwrap_or(0) as u32,
            total_tokens: m.total_token_count.unwrap_or(0) as u32,
            #[allow(deprecated)]
            cached_tokens: m.cached_content_token_count.map(|t| t as u32),
            #[allow(deprecated)]
            reasoning_tokens: m.thoughts_token_count.map(|t| t as u32),
            prompt_tokens_details: m.cached_content_token_count.map(|cached| {
                crate::types::PromptTokensDetails {
                    audio_tokens: None,
                    cached_tokens: Some(cached as u32),
                }
            }),
            completion_tokens_details: m.thoughts_token_count.map(|reasoning| {
                crate::types::CompletionTokensDetails {
                    reasoning_tokens: Some(reasoning as u32),
                    audio_tokens: None,
                    accepted_prediction_tokens: None,
                    rejected_prediction_tokens: None,
                }
            }),
        });

        let finish_reason = candidate.finish_reason.as_ref().map(|reason| match reason {
            types::FinishReason::Stop => {
                let has_client_tool_calls = content_parts.iter().any(|p| match p {
                    ContentPart::ToolCall {
                        provider_executed, ..
                    } => provider_executed != &Some(true),
                    _ => false,
                });
                if has_client_tool_calls {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Stop
                }
            }
            types::FinishReason::MaxTokens => FinishReason::Length,
            types::FinishReason::ImageSafety
            | types::FinishReason::Safety
            | types::FinishReason::Recitation
            | types::FinishReason::Blocklist
            | types::FinishReason::ProhibitedContent
            | types::FinishReason::Spii => FinishReason::ContentFilter,
            types::FinishReason::MalformedFunctionCall => FinishReason::Error,
            types::FinishReason::Language => FinishReason::Other("language".to_string()),
            types::FinishReason::Unspecified
            | types::FinishReason::Other
            | types::FinishReason::Unknown => FinishReason::Other("other".to_string()),
        });

        // Provider metadata (Vercel alignment): expose grounding/url_context and safety ratings.
        let provider_metadata = {
            let mut google_meta: std::collections::HashMap<String, serde_json::Value> =
                std::collections::HashMap::new();

            if let Some(m) = &candidate.grounding_metadata
                && let Ok(v) = serde_json::to_value(m)
            {
                google_meta.insert("groundingMetadata".to_string(), v);
            }

            if let Some(m) = &candidate.url_context_metadata
                && let Ok(v) = serde_json::to_value(m)
            {
                google_meta.insert("urlContextMetadata".to_string(), v);
            }

            if !candidate.safety_ratings.is_empty()
                && let Ok(v) = serde_json::to_value(&candidate.safety_ratings)
            {
                google_meta.insert("safetyRatings".to_string(), v);
            }

            if let Some(m) = &response.prompt_feedback
                && let Ok(v) = serde_json::to_value(m)
            {
                google_meta.insert("promptFeedback".to_string(), v);
            }

            if let Some(m) = &response.usage_metadata
                && let Ok(v) = serde_json::to_value(m)
            {
                google_meta.insert("usageMetadata".to_string(), v);
            }

            if let Some(avg) = candidate.avg_logprobs {
                google_meta.insert("avgLogprobs".to_string(), serde_json::json!(avg));
            }

            if let Some(m) = &candidate.logprobs_result
                && let Ok(v) = serde_json::to_value(m)
            {
                google_meta.insert("logprobsResult".to_string(), v);
            }

            // Vercel-aligned: extract normalized sources from grounding chunks.
            let sources = crate::standards::gemini::sources::extract_sources(
                candidate.grounding_metadata.as_ref(),
            );
            if !sources.is_empty()
                && let Ok(v) = serde_json::to_value(sources)
            {
                google_meta.insert("sources".to_string(), v);
            }

            if google_meta.is_empty() {
                None
            } else {
                let mut all = std::collections::HashMap::new();
                all.insert(provider_key.to_string(), google_meta);
                Some(all)
            }
        };

        // If we collected any content parts (text, tool calls, media), prefer MultiModal
        // unless all parts are plain text, in which case return a single Text.
        let content = if !content_parts.is_empty() {
            if content_parts.iter().all(|p| p.is_text()) {
                MessageContent::Text(text_content)
            } else {
                MessageContent::MultiModal(content_parts)
            }
        } else {
            MessageContent::Text(text_content)
        };

        Ok(ChatResponse {
            id: response.response_id.clone(),
            content,
            model: response.model_version.clone(),
            usage,
            finish_reason,
            audio: None, // Gemini doesn't support audio output in this format
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata,
        })
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::EmbeddingResponse, LlmError> {
        fn parse_usage(raw: &serde_json::Value) -> Option<crate::types::EmbeddingUsage> {
            let usage = raw.get("usageMetadata")?;
            let prompt = usage
                .get("promptTokenCount")
                .or_else(|| usage.get("prompt_token_count"))
                .and_then(|v| v.as_u64())
                .or_else(|| usage.get("totalTokenCount").and_then(|v| v.as_u64()))?;
            let total = usage
                .get("totalTokenCount")
                .or_else(|| usage.get("total_token_count"))
                .and_then(|v| v.as_u64())
                .unwrap_or(prompt);
            Some(crate::types::EmbeddingUsage::new(
                prompt as u32,
                total as u32,
            ))
        }

        // Handle both single and batch embeddings
        // Single: { "embedding": { "values": [f32,..] } }
        // Batch:  { "embeddings": [ { "values": [...] }, ... ] }
        let model = self.config.model.clone();
        if let Some(obj) = raw.get("embedding") {
            let vals = obj
                .get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| LlmError::ParseError("missing embedding.values".to_string()))?;
            let mut vec = Vec::with_capacity(vals.len());
            for v in vals {
                vec.push(v.as_f64().unwrap_or(0.0) as f32);
            }
            let mut out = crate::types::EmbeddingResponse::new(vec![vec], model);
            if let Some(usage) = parse_usage(raw) {
                out = out.with_usage(usage);
            }
            return Ok(out);
        }
        if let Some(arr) = raw.get("embeddings").and_then(|v| v.as_array()) {
            let mut all = Vec::with_capacity(arr.len());
            for e in arr {
                let vals = e.get("values").and_then(|v| v.as_array()).ok_or_else(|| {
                    LlmError::ParseError("missing embeddings[i].values".to_string())
                })?;
                let mut vec = Vec::with_capacity(vals.len());
                for v in vals {
                    vec.push(v.as_f64().unwrap_or(0.0) as f32);
                }
                all.push(vec);
            }
            let mut out = crate::types::EmbeddingResponse::new(all, model);
            if let Some(usage) = parse_usage(raw) {
                out = out.with_usage(usage);
            }
            return Ok(out);
        }
        Err(LlmError::ParseError(
            "Unrecognized Gemini embedding response shape".to_string(),
        ))
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
        // Imagen: { predictions: [ { bytesBase64Encoded: "..." }, ... ] }
        if let Some(preds) = raw.get("predictions").and_then(|v| v.as_array()) {
            let mut images = Vec::new();
            for p in preds {
                if let Some(b64) = p.get("bytesBase64Encoded").and_then(|v| v.as_str())
                    && !b64.trim().is_empty()
                {
                    images.push(crate::types::GeneratedImage {
                        url: None,
                        b64_json: Some(b64.to_string()),
                        format: None,
                        width: None,
                        height: None,
                        revised_prompt: None,
                        metadata: std::collections::HashMap::new(),
                    });
                }
            }
            return Ok(crate::types::ImageGenerationResponse {
                images,
                metadata: std::collections::HashMap::new(),
                warnings: None,
                response: None,
            });
        }

        let response: GenerateContentResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Gemini image response: {e}")))?;
        let mut images = Vec::new();
        if let Some(candidate) = response.candidates.first()
            && let Some(content) = &candidate.content
        {
            for part in &content.parts {
                match part {
                    Part::InlineData { inline_data, .. } => {
                        if inline_data.mime_type.starts_with("image/") {
                            images.push(crate::types::GeneratedImage {
                                url: None,
                                b64_json: Some(inline_data.data.clone()),
                                format: Some(inline_data.mime_type.clone()),
                                width: None,
                                height: None,
                                revised_prompt: None,
                                metadata: std::collections::HashMap::new(),
                            });
                        }
                    }
                    Part::FileData { file_data, .. } => {
                        if let Some(m) = &file_data.mime_type
                            && !m.starts_with("image/")
                        {
                            continue;
                        }
                        images.push(crate::types::GeneratedImage {
                            url: Some(file_data.file_uri.clone()),
                            b64_json: None,
                            format: file_data.mime_type.clone(),
                            width: None,
                            height: None,
                            revised_prompt: None,
                            metadata: std::collections::HashMap::new(),
                        });
                    }
                    _ => {}
                }
            }
        }
        Ok(crate::types::ImageGenerationResponse {
            images,
            metadata: std::collections::HashMap::new(),
            warnings: None,
            response: None,
        })
    }
}
