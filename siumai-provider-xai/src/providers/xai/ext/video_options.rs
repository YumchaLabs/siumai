use crate::provider_options::XaiVideoOptions;
use crate::types::video::VideoGenerationRequest;

pub trait XaiVideoRequestExt {
    /// Convenience: attach xAI video options to `provider_options_map["xai"]`.
    fn with_xai_video_options(self, options: XaiVideoOptions) -> Self;
}

impl XaiVideoRequestExt for VideoGenerationRequest {
    fn with_xai_video_options(self, options: XaiVideoOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiVideoOptions");
        self.with_provider_option("xai", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_request_ext_attaches_xai_video_options() {
        let request = VideoGenerationRequest::new("grok-imagine-video", "hi")
            .with_xai_video_options(
                XaiVideoOptions::new()
                    .with_mode("reference-to-video")
                    .with_video_url("https://example.com/video.mp4")
                    .with_reference_image_urls(["https://example.com/ref-1.png"])
                    .with_poll_interval_ms(1500),
            );

        let value = request
            .provider_options_map
            .get("xai")
            .expect("xai video options present");
        assert_eq!(value["mode"], serde_json::json!("reference-to-video"));
        assert_eq!(
            value["videoUrl"],
            serde_json::json!("https://example.com/video.mp4")
        );
        assert_eq!(
            value["referenceImageUrls"],
            serde_json::json!(["https://example.com/ref-1.png"])
        );
        assert_eq!(value["pollIntervalMs"], serde_json::json!(1500));
    }
}
