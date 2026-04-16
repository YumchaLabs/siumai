use crate::provider_options::XaiImageOptions;
use crate::types::{
    GenerateImageRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
};

fn merge_provider_option_object(
    map: &mut crate::types::ProviderOptionsMap,
    value: serde_json::Value,
) {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = map
            .get("xai")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        map.insert("xai", serde_json::Value::Object(merged));
    } else {
        map.insert("xai", value);
    }
}

pub trait XaiImageRequestExt {
    /// Convenience: attach xAI image options to `provider_options_map["xai"]`.
    fn with_xai_image_options(self, options: XaiImageOptions) -> Self;
}

impl XaiImageRequestExt for ImageGenerationRequest {
    fn with_xai_image_options(mut self, options: XaiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageOptions");
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

impl XaiImageRequestExt for ImageEditRequest {
    fn with_xai_image_options(mut self, options: XaiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageOptions");
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

impl XaiImageRequestExt for ImageVariationRequest {
    fn with_xai_image_options(mut self, options: XaiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageOptions");
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

impl XaiImageRequestExt for GenerateImageRequest {
    fn with_xai_image_options(mut self, options: XaiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageOptions");
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_generation_request_ext_attaches_xai_image_options() {
        let request = ImageGenerationRequest {
            prompt: "hi".to_string(),
            count: 1,
            ..Default::default()
        }
        .with_xai_image_options(
            XaiImageOptions::new()
                .with_aspect_ratio("16:9")
                .with_resolution("2k"),
        );

        let value = request
            .provider_options_map
            .get("xai")
            .expect("xai image options present");
        assert_eq!(value["aspect_ratio"], serde_json::json!("16:9"));
        assert_eq!(value["resolution"], serde_json::json!("2k"));
    }

    #[test]
    fn generate_image_request_ext_merges_existing_xai_image_options() {
        let request = GenerateImageRequest::new("draw a robot")
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "existing": true,
                    "quality": "low"
                }),
            )
            .with_xai_image_options(
                XaiImageOptions::new()
                    .with_aspect_ratio("16:9")
                    .with_quality("high"),
            );

        let value = request
            .provider_options_map
            .get("xai")
            .expect("xai image options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["aspect_ratio"], serde_json::json!("16:9"));
        assert_eq!(value["quality"], serde_json::json!("high"));
    }
}
