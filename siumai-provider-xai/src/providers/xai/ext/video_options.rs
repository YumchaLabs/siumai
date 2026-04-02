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
                    .with_video_url("https://example.com/video.mp4")
                    .with_poll_interval_ms(1500),
            );

        let value = request
            .provider_options_map
            .get("xai")
            .expect("xai video options present");
        assert_eq!(
            value["video_url"],
            serde_json::json!("https://example.com/video.mp4")
        );
        assert_eq!(value["poll_interval_ms"], serde_json::json!(1500));
    }
}
