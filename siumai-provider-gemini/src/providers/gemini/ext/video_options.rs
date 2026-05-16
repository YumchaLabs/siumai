use crate::provider_options::gemini::GoogleVideoModelOptions;
use crate::types::video::VideoGenerationRequest;

use super::request_options::{denormalize_gemini_options_json, merge_provider_option_object_for};

/// Google video request helpers for stable video request families.
pub trait GoogleVideoRequestExt {
    /// Convenience: attach Google video options to `provider_options_map["google"]`.
    fn with_google_video_options(self, options: GoogleVideoModelOptions) -> Self;
}

impl GoogleVideoRequestExt for VideoGenerationRequest {
    fn with_google_video_options(mut self, options: GoogleVideoModelOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GoogleVideoModelOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object_for("google", &mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::gemini::GoogleReferenceImage;

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn google_video_request_extension_source_does_not_read_response_metadata() {
        let source = include_str!("video_options.rs");
        let request_source =
            source_section(source, "pub trait GoogleVideoRequestExt", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "Google video request extension helpers must stay request-only"
            );
        }
    }

    #[test]
    fn video_request_ext_attaches_google_video_options() {
        let request = VideoGenerationRequest::new("veo-3.1-generate-preview", "hi")
            .with_google_video_options(
                GoogleVideoModelOptions::new()
                    .with_negative_prompt("no cats")
                    .with_person_generation("allow_all")
                    .with_poll_interval_ms(500)
                    .with_reference_images(vec![GoogleReferenceImage {
                        bytes_base64_encoded: Some("Zm9v".to_string()),
                        gcs_uri: None,
                    }]),
            );

        let value = request
            .provider_options_map
            .get("google")
            .expect("google video options present");
        assert_eq!(value["negativePrompt"], serde_json::json!("no cats"));
        assert_eq!(value["personGeneration"], serde_json::json!("allow_all"));
        assert_eq!(value["pollIntervalMs"], serde_json::json!(500));
        assert_eq!(
            value["referenceImages"][0]["bytesBase64Encoded"],
            serde_json::json!("Zm9v")
        );
    }
}
