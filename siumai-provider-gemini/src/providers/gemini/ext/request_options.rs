use crate::provider_options::gemini::{
    GoogleLanguageModelInteractionsOptions, GoogleLanguageModelOptions,
};
use crate::types::ChatRequest;

/// Gemini request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait GeminiChatRequestExt {
    /// Convenience: attach Gemini-specific options to `provider_options_map["gemini"]`.
    fn with_gemini_options<T: serde::Serialize>(self, options: T) -> Self;
}

/// Google request option helpers for `ChatRequest`.
pub trait GoogleChatRequestExt {
    /// Convenience: attach Google-specific options to `provider_options_map["google"]`.
    fn with_google_options(self, options: GoogleLanguageModelOptions) -> Self;

    /// Convenience: attach Google Interactions options to `provider_options_map["google"]`.
    ///
    /// This is request-shape parity only. The Interactions runtime is a distinct
    /// provider-owned execution lane and is not routed through ordinary Gemini chat.
    fn with_google_interactions_options(
        self,
        options: GoogleLanguageModelInteractionsOptions,
    ) -> Self;
}

fn denormalize_gemini_option_key(key: &str) -> Option<&'static str> {
    Some(match key {
        "response_mime_type" => "responseMimeType",
        "response_json_schema" => "responseJsonSchema",
        "cached_content" => "cachedContent",
        "response_modalities" => "responseModalities",
        "thinking_config" => "thinkingConfig",
        "safety_settings" => "safetySettings",
        "audio_timestamp" => "audioTimestamp",
        "media_resolution" => "mediaResolution",
        "image_config" => "imageConfig",
        "response_logprobs" => "responseLogprobs",
        "retrieval_config" => "retrievalConfig",
        "stream_function_call_arguments" => "streamFunctionCallArguments",
        "structured_outputs" => "structuredOutputs",
        "code_execution" => "codeExecution",
        "search_grounding" => "searchGrounding",
        "file_search" => "fileSearch",
        "service_tier" => "serviceTier",
        "previous_interaction_id" => "previousInteractionId",
        "agent_config" => "agentConfig",
        "thinking_level" => "thinkingLevel",
        "thinking_summaries" => "thinkingSummaries",
        "response_format" => "responseFormat",
        "system_instruction" => "systemInstruction",
        "interaction_id" => "interactionId",
        "polling_timeout_ms" => "pollingTimeoutMs",
        "collaborative_planning" => "collaborativePlanning",
        "dynamic_retrieval_config" => "dynamicRetrievalConfig",
        "dynamic_threshold" => "dynamicThreshold",
        "lat_lng" => "latLng",
        "file_search_store_names" => "fileSearchStoreNames",
        "aspect_ratio" => "aspectRatio",
        "image_size" => "imageSize",
        "output_dimensionality" => "outputDimensionality",
        "inline_data" => "inlineData",
        "mime_type" => "mimeType",
        "display_name" => "displayName",
        "poll_interval_ms" => "pollIntervalMs",
        "poll_timeout_ms" => "pollTimeoutMs",
        "person_generation" => "personGeneration",
        "negative_prompt" => "negativePrompt",
        "reference_images" => "referenceImages",
        "bytes_base64_encoded" => "bytesBase64Encoded",
        "gcs_uri" => "gcsUri",
        _ => return None,
    })
}

pub(crate) fn denormalize_gemini_options_json(value: &serde_json::Value) -> serde_json::Value {
    fn inner(value: &serde_json::Value) -> Option<serde_json::Value> {
        match value {
            serde_json::Value::Null => None,
            serde_json::Value::Object(map) => {
                let mut out = serde_json::Map::new();
                for (key, value) in map {
                    if let Some(value) = inner(value) {
                        let key = denormalize_gemini_option_key(key).unwrap_or(key);
                        out.insert(key.to_string(), value);
                    }
                }
                Some(serde_json::Value::Object(out))
            }
            serde_json::Value::Array(values) => Some(serde_json::Value::Array(
                values
                    .iter()
                    .map(|value| inner(value).unwrap_or(serde_json::Value::Null))
                    .collect(),
            )),
            other => Some(other.clone()),
        }
    }

    inner(value).unwrap_or(serde_json::Value::Null)
}

pub(crate) fn merge_provider_option_object_for(
    provider_id: &str,
    map: &mut crate::types::ProviderOptionsMap,
    value: serde_json::Value,
) {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = map
            .get(provider_id)
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        map.insert(provider_id, serde_json::Value::Object(merged));
    } else {
        map.insert(provider_id, value);
    }
}

pub(crate) fn merge_provider_option_object(
    map: &mut crate::types::ProviderOptionsMap,
    value: serde_json::Value,
) {
    merge_provider_option_object_for("gemini", map, value);
}

impl GeminiChatRequestExt for ChatRequest {
    fn with_gemini_options<T: serde::Serialize>(mut self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

impl GoogleChatRequestExt for ChatRequest {
    fn with_google_options(mut self, options: GoogleLanguageModelOptions) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object_for("google", &mut self.provider_options_map, value);
        self
    }

    fn with_google_interactions_options(
        mut self,
        options: GoogleLanguageModelInteractionsOptions,
    ) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object_for("google", &mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::gemini::{
        GeminiImageConfig, GeminiLatLng, GeminiOptions, GeminiRetrievalConfig,
        GeminiThinkingConfig, GoogleInteractionsAgentConfig, GoogleInteractionsResponseFormatEntry,
        GoogleLanguageModelInteractionsOptions, GoogleLanguageModelOptions,
    };
    use crate::types::{ChatMessage, ChatRequest};

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn gemini_request_option_extension_source_does_not_read_response_metadata() {
        let source = include_str!("request_options.rs");
        let request_source =
            source_section(source, "pub trait GeminiChatRequestExt", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "Gemini request option extension helpers must stay request-only"
            );
        }
    }

    #[test]
    fn with_gemini_options_serializes_logprobs_fields_in_google_shape() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_gemini_options(
            GeminiOptions::new()
                .with_response_logprobs(true)
                .with_logprobs(3),
        );

        let options = req
            .provider_option("gemini")
            .expect("gemini provider options");

        assert_eq!(options["responseLogprobs"], serde_json::json!(true));
        assert_eq!(options["logprobs"], serde_json::json!(3));
        assert!(options.get("response_logprobs").is_none());
        assert_eq!(options.as_object().map(|value| value.len()), Some(2));
    }

    #[test]
    fn with_gemini_options_serializes_complex_fields_in_google_shape() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_gemini_options(
            GeminiOptions::new()
                .with_response_mime_type("application/json")
                .with_response_json_schema(serde_json::json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } }
                }))
                .with_thinking_config(GeminiThinkingConfig::new().with_thinking_budget(-1))
                .with_retrieval_config(GeminiRetrievalConfig {
                    lat_lng: Some(GeminiLatLng {
                        latitude: 22.3193,
                        longitude: 114.1694,
                    }),
                })
                .with_image_config(GeminiImageConfig {
                    aspect_ratio: Some("16:9".to_string()),
                    image_size: Some("1536x1024".to_string()),
                })
                .with_structured_outputs(true),
        );

        let options = req
            .provider_option("gemini")
            .expect("gemini provider options");

        assert_eq!(
            options["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert_eq!(
            options["responseJsonSchema"]["properties"]["answer"]["type"],
            serde_json::json!("string")
        );
        assert_eq!(
            options["thinkingConfig"]["thinkingBudget"],
            serde_json::json!(-1)
        );
        assert_eq!(
            options["retrievalConfig"]["latLng"]["latitude"],
            serde_json::json!(22.3193)
        );
        assert_eq!(
            options["imageConfig"]["aspectRatio"],
            serde_json::json!("16:9")
        );
        assert_eq!(
            options["imageConfig"]["imageSize"],
            serde_json::json!("1536x1024")
        );
        assert_eq!(options["structuredOutputs"], serde_json::json!(true));
        assert!(options.get("response_mime_type").is_none());
        assert!(options.get("retrieval_config").is_none());
        assert!(options.get("image_config").is_none());
    }

    #[test]
    fn with_gemini_options_merges_existing_provider_options() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option(
                "gemini",
                serde_json::json!({
                    "existing": true,
                    "responseMimeType": "text/plain"
                }),
            )
            .with_gemini_options(
                GeminiOptions::new()
                    .with_response_mime_type("application/json")
                    .with_structured_outputs(true),
            );

        let options = req
            .provider_option("gemini")
            .expect("gemini provider options");

        assert_eq!(options["existing"], serde_json::json!(true));
        assert_eq!(
            options["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert_eq!(options["structuredOutputs"], serde_json::json!(true));
    }

    #[test]
    fn with_google_options_serializes_google_runtime_fields() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_google_options(
            GoogleLanguageModelOptions::new()
                .with_service_tier("flex")
                .with_stream_function_call_arguments(true),
        );

        let options = req
            .provider_option("google")
            .expect("google provider options");

        assert_eq!(options["serviceTier"], serde_json::json!("flex"));
        assert_eq!(
            options["streamFunctionCallArguments"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn with_google_interactions_options_serializes_package_fields() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_google_interactions_options(
                GoogleLanguageModelInteractionsOptions::new()
                    .with_previous_interaction_id("iact_123")
                    .with_store(true)
                    .with_agent("deep-research-preview-04-2026")
                    .with_agent_config(
                        GoogleInteractionsAgentConfig::deep_research()
                            .with_thinking_summaries("auto")
                            .with_visualization("auto")
                            .with_collaborative_planning(true),
                    )
                    .with_thinking_level("high")
                    .with_response_format(vec![
                        GoogleInteractionsResponseFormatEntry::json_schema(serde_json::json!({
                            "type": "object"
                        })),
                        GoogleInteractionsResponseFormatEntry::image()
                            .with_mime_type("image/png")
                            .with_aspect_ratio("16:9")
                            .with_image_size("1K"),
                    ])
                    .with_media_resolution("high")
                    .with_response_modalities(["text", "image"])
                    .with_service_tier("priority")
                    .with_system_instruction("be concise")
                    .with_signature("sig_123")
                    .with_interaction_id("iact_456")
                    .with_polling_timeout_ms(30_000),
            );

        let options = req
            .provider_option("google")
            .expect("google provider options");

        assert_eq!(
            options["previousInteractionId"],
            serde_json::json!("iact_123")
        );
        assert_eq!(options["store"], serde_json::json!(true));
        assert_eq!(
            options["agent"],
            serde_json::json!("deep-research-preview-04-2026")
        );
        assert_eq!(
            options["agentConfig"]["type"],
            serde_json::json!("deep-research")
        );
        assert_eq!(
            options["agentConfig"]["collaborativePlanning"],
            serde_json::json!(true)
        );
        assert_eq!(options["thinkingLevel"], serde_json::json!("high"));
        assert_eq!(
            options["responseFormat"][0]["mimeType"],
            serde_json::json!("application/json")
        );
        assert_eq!(
            options["responseFormat"][1]["aspectRatio"],
            serde_json::json!("16:9")
        );
        assert_eq!(options["mediaResolution"], serde_json::json!("high"));
        assert_eq!(
            options["responseModalities"],
            serde_json::json!(["text", "image"])
        );
        assert_eq!(options["serviceTier"], serde_json::json!("priority"));
        assert_eq!(
            options["systemInstruction"],
            serde_json::json!("be concise")
        );
        assert_eq!(options["signature"], serde_json::json!("sig_123"));
        assert_eq!(options["interactionId"], serde_json::json!("iact_456"));
        assert_eq!(options["pollingTimeoutMs"], serde_json::json!(30_000));
    }

    #[test]
    fn denormalize_gemini_options_json_preserves_null_array_entries() {
        let value = denormalize_gemini_options_json(&serde_json::json!({
            "content": [
                [{"inline_data": {"mime_type": "image/png", "data": "Zm9v"}}],
                null
            ]
        }));

        assert_eq!(
            value,
            serde_json::json!({
                "content": [
                    [{"inlineData": {"mimeType": "image/png", "data": "Zm9v"}}],
                    null
                ]
            })
        );
    }
}
