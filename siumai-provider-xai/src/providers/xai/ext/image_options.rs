use crate::provider_options::XaiImageOptions;
use crate::types::{ImageEditRequest, ImageGenerationRequest, ImageVariationRequest};

pub trait XaiImageRequestExt {
    /// Convenience: attach xAI image options to `provider_options_map["xai"]`.
    fn with_xai_image_options(self, options: XaiImageOptions) -> Self;
}

impl XaiImageRequestExt for ImageGenerationRequest {
    fn with_xai_image_options(self, options: XaiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageOptions");
        self.with_provider_option("xai", value)
    }
}

impl XaiImageRequestExt for ImageEditRequest {
    fn with_xai_image_options(self, options: XaiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageOptions");
        self.with_provider_option("xai", value)
    }
}

impl XaiImageRequestExt for ImageVariationRequest {
    fn with_xai_image_options(self, options: XaiImageOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageOptions");
        self.with_provider_option("xai", value)
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
}
