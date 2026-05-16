use crate::provider_options::gemini::{GeminiImageOptions, GoogleImageModelOptions};
use crate::types::{
    GenerateImageRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
};

use super::request_options::{
    denormalize_gemini_options_json, merge_provider_option_object, merge_provider_option_object_for,
};

/// Gemini image request helpers for stable image request families.
pub trait GeminiImageRequestExt {
    /// Convenience: attach Gemini image options to `provider_options_map["gemini"]`.
    fn with_gemini_image_options(self, options: GeminiImageOptions) -> Self;
}

/// Google image request helpers for stable image request families.
pub trait GoogleImageRequestExt {
    /// Convenience: attach Google image options to `provider_options_map["google"]`.
    fn with_google_image_options(self, options: GoogleImageModelOptions) -> Self;
}

impl GeminiImageRequestExt for ImageGenerationRequest {
    fn with_gemini_image_options(mut self, options: GeminiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GeminiImageOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

impl GoogleImageRequestExt for ImageGenerationRequest {
    fn with_google_image_options(mut self, options: GoogleImageModelOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GoogleImageModelOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object_for("google", &mut self.provider_options_map, value);
        self
    }
}

impl GeminiImageRequestExt for ImageEditRequest {
    fn with_gemini_image_options(mut self, options: GeminiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GeminiImageOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

impl GoogleImageRequestExt for ImageEditRequest {
    fn with_google_image_options(mut self, options: GoogleImageModelOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GoogleImageModelOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object_for("google", &mut self.provider_options_map, value);
        self
    }
}

impl GeminiImageRequestExt for ImageVariationRequest {
    fn with_gemini_image_options(mut self, options: GeminiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GeminiImageOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

impl GoogleImageRequestExt for ImageVariationRequest {
    fn with_google_image_options(mut self, options: GoogleImageModelOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GoogleImageModelOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object_for("google", &mut self.provider_options_map, value);
        self
    }
}

impl GeminiImageRequestExt for GenerateImageRequest {
    fn with_gemini_image_options(mut self, options: GeminiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GeminiImageOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

impl GoogleImageRequestExt for GenerateImageRequest {
    fn with_google_image_options(mut self, options: GoogleImageModelOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GoogleImageModelOptions");
        let value = denormalize_gemini_options_json(&value);
        merge_provider_option_object_for("google", &mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn gemini_image_request_extension_source_does_not_read_response_metadata() {
        let source = include_str!("image_options.rs");
        let request_source =
            source_section(source, "pub trait GeminiImageRequestExt", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "Gemini image request extension helpers must stay request-only"
            );
        }
    }

    #[test]
    fn image_generation_request_ext_attaches_gemini_image_options() {
        let request = ImageGenerationRequest {
            prompt: "draw a robot".to_string(),
            count: 1,
            ..Default::default()
        }
        .with_gemini_image_options(
            GeminiImageOptions::new()
                .with_aspect_ratio("16:9")
                .with_person_generation("allow_all"),
        );

        let value = request
            .provider_options_map
            .get("gemini")
            .expect("gemini image options present");
        assert_eq!(value["aspectRatio"], serde_json::json!("16:9"));
        assert_eq!(value["personGeneration"], serde_json::json!("allow_all"));
    }

    #[test]
    fn generate_image_request_ext_merges_existing_gemini_image_options() {
        let request = GenerateImageRequest::new("draw a robot")
            .with_provider_option(
                "gemini",
                serde_json::json!({
                    "existing": true,
                    "personGeneration": "dont_allow"
                }),
            )
            .with_gemini_image_options(
                GeminiImageOptions::new()
                    .with_aspect_ratio("16:9")
                    .with_person_generation("allow_all"),
            );

        let value = request
            .provider_options_map
            .get("gemini")
            .expect("gemini image options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["aspectRatio"], serde_json::json!("16:9"));
        assert_eq!(value["personGeneration"], serde_json::json!("allow_all"));
    }

    #[test]
    fn generate_image_request_ext_attaches_google_image_options() {
        let request = GenerateImageRequest::new("draw a robot").with_google_image_options(
            GoogleImageModelOptions::new()
                .with_aspect_ratio("16:9")
                .with_person_generation("allow_all"),
        );

        let value = request
            .provider_options_map
            .get("google")
            .expect("google image options present");
        assert_eq!(value["aspectRatio"], serde_json::json!("16:9"));
        assert_eq!(value["personGeneration"], serde_json::json!("allow_all"));
    }
}
