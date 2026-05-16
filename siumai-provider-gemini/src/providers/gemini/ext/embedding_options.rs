use crate::provider_options::gemini::GoogleEmbeddingModelOptions;
use crate::types::EmbeddingRequest;

use super::request_options::{denormalize_gemini_options_json, merge_provider_option_object_for};

/// Google embedding request helpers for stable embedding request families.
pub trait GoogleEmbeddingRequestExt {
    /// Convenience: attach Google embedding options to `provider_options_map["google"]`.
    fn with_google_embedding_options(self, options: GoogleEmbeddingModelOptions) -> Self;
}

impl GoogleEmbeddingRequestExt for EmbeddingRequest {
    fn with_google_embedding_options(mut self, options: GoogleEmbeddingModelOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GoogleEmbeddingModelOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object_for("google", &mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::gemini::{GoogleEmbeddingContentPart, GoogleEmbeddingInlineData};

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn google_embedding_request_extension_source_does_not_read_response_metadata() {
        let source = include_str!("embedding_options.rs");
        let request_source = source_section(
            source,
            "pub trait GoogleEmbeddingRequestExt",
            "#[cfg(test)]",
        );

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "Google embedding request extension helpers must stay request-only"
            );
        }
    }

    #[test]
    fn embedding_request_ext_attaches_google_embedding_options() {
        let request = EmbeddingRequest::single("hello").with_google_embedding_options(
            GoogleEmbeddingModelOptions::new()
                .with_output_dimensionality(128)
                .with_task_type(crate::types::EmbeddingTaskType::SemanticSimilarity)
                .with_content(vec![Some(vec![GoogleEmbeddingContentPart::InlineData {
                    inline_data: GoogleEmbeddingInlineData {
                        mime_type: "image/png".to_string(),
                        data: "Zm9v".to_string(),
                    },
                }])]),
        );

        let value = request
            .provider_options_map
            .get("google")
            .expect("google embedding options present");

        assert_eq!(value["outputDimensionality"], serde_json::json!(128));
        assert_eq!(value["taskType"], serde_json::json!("SEMANTIC_SIMILARITY"));
        assert_eq!(
            value["content"][0][0]["inlineData"]["mimeType"],
            serde_json::json!("image/png")
        );
    }
}
