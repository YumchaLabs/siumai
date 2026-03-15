use crate::types::ChatRequest;

/// Gemini request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait GeminiChatRequestExt {
    /// Convenience: attach Gemini-specific options to `provider_options_map["gemini"]`.
    fn with_gemini_options<T: serde::Serialize>(self, options: T) -> Self;
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
        "structured_outputs" => "structuredOutputs",
        "code_execution" => "codeExecution",
        "search_grounding" => "searchGrounding",
        "file_search" => "fileSearch",
        "dynamic_retrieval_config" => "dynamicRetrievalConfig",
        "dynamic_threshold" => "dynamicThreshold",
        "lat_lng" => "latLng",
        "file_search_store_names" => "fileSearchStoreNames",
        "aspect_ratio" => "aspectRatio",
        "image_size" => "imageSize",
        _ => return None,
    })
}

fn denormalize_gemini_options_json(value: &serde_json::Value) -> serde_json::Value {
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
                values.iter().filter_map(inner).collect(),
            )),
            other => Some(other.clone()),
        }
    }

    inner(value).unwrap_or(serde_json::Value::Null)
}

impl GeminiChatRequestExt for ChatRequest {
    fn with_gemini_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        let value = denormalize_gemini_options_json(&value);
        self.with_provider_option("gemini", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::gemini::{
        GeminiImageConfig, GeminiLatLng, GeminiOptions, GeminiRetrievalConfig, GeminiThinkingConfig,
    };
    use crate::types::{ChatMessage, ChatRequest};

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
}
