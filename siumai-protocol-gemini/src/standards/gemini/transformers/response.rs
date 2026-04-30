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
        assert_eq!(resp.raw_finish_reason.as_deref(), Some("STOP"));
    }

    #[test]
    fn gemini_response_preserves_raw_finish_reason_for_content_filter() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.0-flash-exp".into())
            .with_base_url("https://example".into());
        let tx = GeminiResponseTransformer { config: cfg };

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": { "parts": [ { "text": "blocked" } ] },
                    "finishReason": "PROHIBITED_CONTENT"
                }
            ],
            "modelVersion": "gemini-2.0-flash-exp"
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");

        assert_eq!(resp.finish_reason, Some(FinishReason::ContentFilter));
        assert_eq!(
            resp.raw_finish_reason.as_deref(),
            Some("PROHIBITED_CONTENT")
        );
    }

    #[test]
    fn gemini_response_maps_unknown_finish_reason_to_raw_other() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.0-flash-exp".into())
            .with_base_url("https://example".into());
        let tx = GeminiResponseTransformer { config: cfg };

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": { "parts": [ { "text": "future" } ] },
                    "finishReason": "FUTURE_STOP_REASON"
                }
            ],
            "modelVersion": "gemini-2.0-flash-exp"
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");

        assert_eq!(
            resp.finish_reason,
            Some(FinishReason::Other("FUTURE_STOP_REASON".to_string()))
        );
        assert_eq!(
            resp.raw_finish_reason.as_deref(),
            Some("FUTURE_STOP_REASON")
        );
    }

    #[test]
    fn gemini_response_usage_counts_reasoning_inside_completion_totals() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.5-pro".into())
            .with_base_url("https://example".into());
        let tx = GeminiResponseTransformer { config: cfg };

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": { "parts": [ { "text": "hello" } ] },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "cachedContentTokenCount": 5,
                "candidatesTokenCount": 6,
                "thoughtsTokenCount": 3,
                "totalTokenCount": 21,
                "trafficType": "ON_DEMAND",
                "promptTokensDetails": [
                    { "modality": "TEXT", "tokenCount": 10 },
                    { "modality": "IMAGE", "tokenCount": 2 }
                ],
                "candidatesTokensDetails": [
                    { "modality": "TEXT", "tokenCount": 6 }
                ]
            },
            "modelVersion": "gemini-2.5-pro"
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let usage = resp.usage.expect("usage");

        assert_eq!(usage.prompt_tokens(), Some(12));
        assert_eq!(usage.completion_tokens(), Some(9));
        assert_eq!(usage.total_tokens(), Some(21));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(7));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(5));
        assert_eq!(usage.normalized_output_tokens().total, Some(9));
        assert_eq!(usage.normalized_output_tokens().text, Some(6));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(3));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["trafficType"],
            serde_json::json!("ON_DEMAND")
        );
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["promptTokensDetails"],
            serde_json::json!([
                { "modality": "TEXT", "tokenCount": 10 },
                { "modality": "IMAGE", "tokenCount": 2 }
            ])
        );
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["candidatesTokensDetails"],
            serde_json::json!([{ "modality": "TEXT", "tokenCount": 6 }])
        );
    }

    #[test]
    fn gemini_response_usage_preserves_unknown_totals() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.5-pro".into())
            .with_base_url("https://example".into());
        let tx = GeminiResponseTransformer { config: cfg };

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": { "parts": [ { "text": "hello" } ] },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "trafficType": "ON_DEMAND"
            },
            "modelVersion": "gemini-2.5-pro"
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let usage = resp.usage.expect("usage");

        assert_eq!(usage.prompt_tokens_value(), None);
        assert_eq!(usage.completion_tokens_value(), None);
        assert_eq!(usage.total_tokens_value(), None);
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["trafficType"],
            serde_json::json!("ON_DEMAND")
        );
    }

    #[test]
    fn gemini_chat_response_maps_function_call_parts_into_tool_call_content_parts() {
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
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);

        let info = calls[0].as_tool_call().expect("tool call info");
        assert!(
            !info.tool_call_id.is_empty(),
            "expected generated tool_call_id"
        );
        assert_eq!(info.tool_name, "weather");
        assert_eq!(info.arguments, &serde_json::json!({ "city": "Tokyo" }));
        assert_eq!(info.provider_executed.copied(), None);
    }

    #[test]
    fn gemini_chat_response_uses_custom_generate_id_for_tool_calls_and_sources() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.0-flash-exp".into())
            .with_base_url("https://example".into())
            .with_generate_id({
                let counter = Arc::clone(&counter);
                move || format!("custom-id-{}", counter.fetch_add(1, Ordering::Relaxed))
            });
        let tx = GeminiResponseTransformer { config: cfg };

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            { "functionCall": { "name": "weather", "args": { "city": "Tokyo" } } }
                        ]
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            { "web": { "uri": "https://example.com", "title": "Example" } }
                        ]
                    },
                    "finishReason": "STOP"
                }
            ],
            "modelVersion": "gemini-2.0-flash-exp"
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let call = resp.tool_calls()[0].as_tool_call().expect("tool call");
        assert_eq!(call.tool_call_id, "custom-id-0");

        let sources = resp
            .provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("google"))
            .and_then(|meta| meta.get("sources"))
            .and_then(|value| value.as_array())
            .expect("sources");
        assert_eq!(
            sources[0].get("id").and_then(|value| value.as_str()),
            Some("custom-id-1")
        );
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

    #[test]
    fn gemini_response_maps_thought_files_to_reasoning_file_parts() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-2.5-pro".into())
            .with_base_url("https://example".into());
        let tx = GeminiResponseTransformer { config: cfg };

        let raw = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "application/pdf",
                                    "data": "aGVsbG8="
                                },
                                "thought": true,
                                "thoughtSignature": "sig_file"
                            }
                        ]
                    }
                }
            ],
            "modelVersion": "gemini-2.5-pro"
        });

        let resp = tx.transform_chat_response(&raw).expect("transform");
        let parts = resp.content.as_multimodal().expect("multimodal content");
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            ContentPart::ReasoningFile {
                media_type,
                provider_metadata,
                ..
            } => {
                assert_eq!(media_type, "application/pdf");
                assert_eq!(
                    provider_metadata
                        .as_ref()
                        .and_then(|meta| meta.get("google"))
                        .and_then(|value| value.get("thoughtSignature"))
                        .and_then(|value| value.as_str()),
                    Some("sig_file")
                );
            }
            other => panic!("expected reasoning file part, got: {other:?}"),
        }
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

        fn raw_candidate_finish_reason(raw: &serde_json::Value) -> Option<String> {
            raw.get("candidates")
                .and_then(|value| value.as_array())
                .and_then(|candidates| candidates.first())
                .and_then(|candidate| {
                    candidate
                        .get("finishReason")
                        .or_else(|| candidate.get("finish_reason"))
                })
                .and_then(|value| value.as_str())
                .map(str::to_string)
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
        let raw_finish_reason = raw_candidate_finish_reason(raw);
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
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
                        });
                    } else {
                        if !text_content.is_empty() {
                            text_content.push('\n');
                        }
                        text_content.push_str(text);
                        content_parts.push(ContentPart::Text {
                            text: text.clone(),
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
                        });
                    }
                }
                Part::InlineData {
                    inline_data,
                    thought,
                    thought_signature,
                } => {
                    _has_multimodal_content = true;
                    let provider_metadata = thought_signature_provider_metadata(
                        provider_key,
                        thought_signature.as_ref(),
                    );
                    if thought.unwrap_or(false) {
                        content_parts.push(crate::types::ContentPart::ReasoningFile {
                            source: crate::types::chat::MediaSource::Base64 {
                                data: inline_data.data.clone(),
                            },
                            media_type: inline_data.mime_type.clone(),
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
                        });
                    } else if inline_data.mime_type.starts_with("image/") {
                        content_parts.push(crate::types::ContentPart::Image {
                            source: crate::types::FilePartSource::base64(inline_data.data.clone()),
                            media_type: Some(inline_data.mime_type.clone()),
                            detail: None,
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
                        });
                    } else if inline_data.mime_type.starts_with("audio/") {
                        content_parts.push(crate::types::ContentPart::Audio {
                            source: crate::types::chat::MediaSource::Base64 {
                                data: inline_data.data.clone(),
                            },
                            media_type: Some(inline_data.mime_type.clone()),
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
                        });
                    } else {
                        // Other file types
                        content_parts.push(crate::types::ContentPart::File {
                            source: crate::types::FilePartSource::base64(inline_data.data.clone()),
                            media_type: inline_data.mime_type.clone(),
                            filename: None,
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
                        });
                    }
                }
                Part::FileData {
                    file_data,
                    thought,
                    thought_signature,
                } => {
                    _has_multimodal_content = true;
                    let mime_type = file_data
                        .mime_type
                        .as_deref()
                        .unwrap_or("application/octet-stream");
                    let provider_metadata = thought_signature_provider_metadata(
                        provider_key,
                        thought_signature.as_ref(),
                    );
                    if thought.unwrap_or(false) {
                        content_parts.push(crate::types::ContentPart::ReasoningFile {
                            source: crate::types::chat::MediaSource::Url {
                                url: file_data.file_uri.clone(),
                            },
                            media_type: mime_type.to_string(),
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
                        });
                    } else if mime_type.starts_with("image/") {
                        content_parts.push(crate::types::ContentPart::Image {
                            source: crate::types::FilePartSource::url(file_data.file_uri.clone()),
                            media_type: Some(mime_type.to_string()),
                            detail: None,
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
                        });
                    } else if mime_type.starts_with("audio/") {
                        content_parts.push(crate::types::ContentPart::Audio {
                            source: crate::types::chat::MediaSource::Url {
                                url: file_data.file_uri.clone(),
                            },
                            media_type: Some(mime_type.to_string()),
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
                        });
                    } else {
                        // Other file types
                        content_parts.push(crate::types::ContentPart::File {
                            source: crate::types::FilePartSource::url(file_data.file_uri.clone()),
                            media_type: mime_type.to_string(),
                            filename: None,
                            provider_options: crate::types::ProviderOptionsMap::default(),
                            provider_metadata,
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
                        tool_call_id: self.config.generate_id(),
                        tool_name: function_call.name.clone(),
                        arguments,
                        provider_executed: None,
                        dynamic: None,
                        invalid: None,
                        error: None,
                        title: None,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                }
                Part::ExecutableCode {
                    executable_code,
                    thought_signature,
                } => {
                    let id = self.config.generate_id();
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
                        dynamic: None,
                        invalid: None,
                        error: None,
                        title: None,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                }
                Part::CodeExecutionResult {
                    code_execution_result,
                    thought_signature,
                } => {
                    let id = pending_code_execution_id
                        .take()
                        .unwrap_or_else(|| self.config.generate_id());

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
                        input: None,
                        provider_executed: Some(true),
                        dynamic: None,
                        preliminary: None,
                        title: None,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                }
                _ => {}
            }
        }

        let usage = response.usage_metadata.as_ref().map(|m| {
            let prompt_tokens = m.prompt_token_count.and_then(|t| u32::try_from(t).ok());
            let text_tokens = m.candidates_token_count.and_then(|t| u32::try_from(t).ok());
            let total_tokens = m.total_token_count.and_then(|t| u32::try_from(t).ok());
            let cached_tokens = m
                .cached_content_token_count
                .and_then(|t| u32::try_from(t).ok());
            let reasoning_tokens = m.thoughts_token_count.and_then(|t| u32::try_from(t).ok());
            let completion_tokens = text_tokens
                .zip(reasoning_tokens)
                .map(|(text, reasoning)| text.saturating_add(reasoning))
                .or(text_tokens)
                .or_else(|| {
                    total_tokens
                        .zip(prompt_tokens)
                        .map(|(total, prompt)| total.saturating_sub(prompt))
                });
            let output_text_tokens = text_tokens.or_else(|| {
                completion_tokens.map(|total| total.saturating_sub(reasoning_tokens.unwrap_or(0)))
            });

            let mut builder = Usage::builder()
                .with_raw_usage_value(serde_json::to_value(m).unwrap_or(serde_json::Value::Null));

            if let Some(prompt_tokens) = prompt_tokens {
                builder = builder
                    .prompt_tokens(prompt_tokens)
                    .with_input_total_tokens(prompt_tokens)
                    .with_input_no_cache_tokens(
                        prompt_tokens.saturating_sub(cached_tokens.unwrap_or(0)),
                    );
            }
            if let Some(completion_tokens) = completion_tokens {
                builder = builder
                    .completion_tokens(completion_tokens)
                    .with_output_total_tokens(completion_tokens);
            }
            if let Some(output_text_tokens) = output_text_tokens {
                builder = builder.with_output_text_tokens(output_text_tokens);
            }
            if let Some(total_tokens) = total_tokens {
                builder = builder.total_tokens(total_tokens);
            }

            if let Some(cached_tokens) = cached_tokens {
                builder = builder
                    .with_cached_tokens(cached_tokens)
                    .with_input_cache_read_tokens(cached_tokens);
            }
            if let Some(reasoning_tokens) = reasoning_tokens {
                builder = builder
                    .with_reasoning_tokens(reasoning_tokens)
                    .with_output_reasoning_tokens(reasoning_tokens);
            }

            builder.build()
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
            types::FinishReason::Language => FinishReason::Other(
                raw_finish_reason
                    .clone()
                    .unwrap_or_else(|| "language".to_string()),
            ),
            types::FinishReason::Unspecified
            | types::FinishReason::Other
            | types::FinishReason::Unknown => FinishReason::Other(
                raw_finish_reason
                    .clone()
                    .unwrap_or_else(|| "other".to_string()),
            ),
        });
        let service_tier = raw
            .get("serviceTier")
            .or_else(|| raw.get("service_tier"))
            .and_then(|value| value.as_str())
            .map(|value| value.to_string());
        let finish_message = raw
            .get("finishMessage")
            .or_else(|| raw.get("finish_message"))
            .and_then(|value| value.as_str())
            .map(|value| value.to_string());

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

            if let Some(message) = &finish_message {
                google_meta.insert("finishMessage".to_string(), serde_json::json!(message));
            }

            if let Some(service_tier) = &service_tier {
                google_meta.insert("serviceTier".to_string(), serde_json::json!(service_tier));
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
            let sources = crate::standards::gemini::sources::extract_sources_with_generate_id(
                candidate.grounding_metadata.as_ref(),
                || self.config.generate_id(),
            );
            if !sources.is_empty()
                && let Ok(v) = serde_json::to_value(sources)
            {
                google_meta.insert("sources".to_string(), v);
            }

            if google_meta.is_empty() {
                None
            } else {
                Some(
                    crate::types::provider_metadata::provider_metadata_from_object(
                        provider_key,
                        google_meta,
                    ),
                )
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
            raw_finish_reason,
            audio: None, // Gemini doesn't support audio output in this format
            system_fingerprint: None,
            service_tier,
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
