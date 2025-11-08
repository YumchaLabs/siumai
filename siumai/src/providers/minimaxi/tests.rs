//! MiniMaxi Provider Tests

#[cfg(test)]
mod minimaxi_tests {
    use super::super::*;

    #[test]
    fn test_config_creation() {
        let config = MinimaxiConfig::new("test-api-key");
        assert_eq!(config.api_key, "test-api-key");
        assert_eq!(config.base_url, MinimaxiConfig::DEFAULT_BASE_URL);
        assert_eq!(config.common_params.model, MinimaxiConfig::DEFAULT_MODEL);
    }

    #[test]
    fn test_config_validation() {
        let config = MinimaxiConfig::new("test-api-key");
        assert!(config.validate().is_ok());

        let empty_config = MinimaxiConfig::new("");
        assert!(empty_config.validate().is_err());
    }

    #[test]
    fn test_model_constants() {
        assert_eq!(model_constants::text::MINIMAX_M2, "MiniMax-M2");
        assert_eq!(
            model_constants::text::MINIMAX_M2_STABLE,
            "MiniMax-M2-Stable"
        );
        assert_eq!(model_constants::audio::SPEECH_2_6_HD, "speech-2.6-hd");
        assert_eq!(model_constants::audio::SPEECH_2_6_TURBO, "speech-2.6-turbo");
        assert_eq!(model_constants::voice::MALE_QN_QINGSE, "male-qn-qingse");
        assert_eq!(model_constants::voice::FEMALE_SHAONV, "female-shaonv");
        assert_eq!(model_constants::video::HAILUO_2_3, "hailuo-2.3");
        assert_eq!(model_constants::video::HAILUO_2_3_FAST, "hailuo-2.3-fast");
        assert_eq!(model_constants::music::MUSIC_2_0, "music-2.0");
        assert_eq!(model_constants::images::IMAGE_01, "image-01");
        assert_eq!(model_constants::images::IMAGE_01_LIVE, "image-01-live");
    }

    #[test]
    fn test_url_switching_for_audio() {
        use crate::core::{ProviderContext, ProviderSpec};

        let spec = spec::MinimaxiSpec::new();

        // Test with Anthropic base URL
        let ctx_anthropic = ProviderContext {
            provider_id: "minimaxi".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: "https://api.minimaxi.com/anthropic".to_string(),
            http_extra_headers: Default::default(),
            organization: None,
            project: None,
            extras: Default::default(),
        };

        let audio_url = spec.audio_base_url(&ctx_anthropic);
        assert_eq!(audio_url, config::MinimaxiConfig::OPENAI_BASE_URL);

        // Test with OpenAI base URL
        let ctx_openai = ProviderContext {
            provider_id: "minimaxi".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: "https://api.minimaxi.com/v1".to_string(),
            http_extra_headers: Default::default(),
            organization: None,
            project: None,
            extras: Default::default(),
        };

        let audio_url = spec.audio_base_url(&ctx_openai);
        assert_eq!(audio_url, "https://api.minimaxi.com/v1");
    }

    #[test]
    fn test_url_switching_for_image() {
        use crate::core::{ProviderContext, ProviderSpec};
        use crate::types::ImageGenerationRequest;

        let spec = spec::MinimaxiSpec::new();
        let request = ImageGenerationRequest::default();

        // Test with Anthropic base URL
        let ctx_anthropic = ProviderContext {
            provider_id: "minimaxi".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: "https://api.minimaxi.com/anthropic".to_string(),
            http_extra_headers: Default::default(),
            organization: None,
            project: None,
            extras: Default::default(),
        };

        let image_url = spec.image_url(&request, &ctx_anthropic);
        assert_eq!(
            image_url,
            format!(
                "{}/image_generation",
                config::MinimaxiConfig::OPENAI_BASE_URL
            )
        );

        // Test with OpenAI base URL
        let ctx_openai = ProviderContext {
            provider_id: "minimaxi".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: "https://api.minimaxi.com/v1".to_string(),
            http_extra_headers: Default::default(),
            organization: None,
            project: None,
            extras: Default::default(),
        };

        let image_url = spec.image_url(&request, &ctx_openai);
        assert_eq!(image_url, "https://api.minimaxi.com/v1/image_generation");
    }

    #[test]
    fn test_image_capability() {
        use crate::traits::ImageGenerationCapability;

        let config = MinimaxiConfig::new("test-api-key");
        let client = MinimaxiClient::new(config, reqwest::Client::new());

        // Test supported sizes
        let sizes = client.get_supported_sizes();
        assert_eq!(sizes.len(), 8);
        assert!(sizes.contains(&"1024x1024".to_string())); // 1:1
        assert!(sizes.contains(&"1280x720".to_string())); // 16:9

        // Test supported formats
        let formats = client.get_supported_formats();
        assert_eq!(formats.len(), 2);
        assert!(formats.contains(&"url".to_string()));
        assert!(formats.contains(&"b64_json".to_string()));
    }

    #[test]
    fn test_video_capability() {
        use crate::traits::VideoGenerationCapability;

        let config = MinimaxiConfig::new("test-api-key");
        let client = MinimaxiClient::new(config, reqwest::Client::new());

        // Test supported models
        let models = client.get_supported_models();
        assert_eq!(models.len(), 4);
        assert!(models.contains(&"MiniMax-Hailuo-2.3".to_string()));
        assert!(models.contains(&"MiniMax-Hailuo-02".to_string()));
        assert!(models.contains(&"T2V-01-Director".to_string()));
        assert!(models.contains(&"T2V-01".to_string()));

        // Test supported resolutions for Hailuo models
        let resolutions = client.get_supported_resolutions("MiniMax-Hailuo-2.3");
        assert_eq!(resolutions.len(), 2);
        assert!(resolutions.contains(&"768P".to_string()));
        assert!(resolutions.contains(&"1080P".to_string()));

        // Test supported resolutions for T2V models
        let resolutions = client.get_supported_resolutions("T2V-01");
        assert_eq!(resolutions.len(), 1);
        assert!(resolutions.contains(&"720P".to_string()));

        // Test supported durations for Hailuo models
        let durations = client.get_supported_durations("MiniMax-Hailuo-2.3");
        assert_eq!(durations.len(), 2);
        assert!(durations.contains(&6));
        assert!(durations.contains(&10));

        // Test supported durations for T2V models
        let durations = client.get_supported_durations("T2V-01-Director");
        assert_eq!(durations.len(), 1);
        assert!(durations.contains(&6));
    }

    #[test]
    fn test_video_request_builder() {
        use crate::types::video::VideoGenerationRequest;

        let request =
            VideoGenerationRequest::new("MiniMax-Hailuo-2.3", "A beautiful sunset over the ocean")
                .with_duration(6)
                .with_resolution("1080P")
                .with_prompt_optimizer(true)
                .with_watermark(false);

        assert_eq!(request.model, "MiniMax-Hailuo-2.3");
        assert_eq!(request.prompt, "A beautiful sunset over the ocean");
        assert_eq!(request.duration, Some(6));
        assert_eq!(request.resolution, Some("1080P".to_string()));
        assert_eq!(request.prompt_optimizer, Some(true));
        assert_eq!(request.aigc_watermark, Some(false));
    }

    #[test]
    fn test_video_task_status() {
        use crate::types::video::{VideoTaskStatus, VideoTaskStatusResponse};

        let mut response = VideoTaskStatusResponse {
            task_id: "task_123".to_string(),
            status: VideoTaskStatus::Processing,
            file_id: None,
            video_width: None,
            video_height: None,
            base_resp: None,
        };

        assert!(response.is_in_progress());
        assert!(!response.is_complete());

        response.status = VideoTaskStatus::Success;
        response.file_id = Some("file_456".to_string());
        response.video_width = Some(1920);
        response.video_height = Some(1080);

        assert!(response.is_complete());
        assert!(response.is_success());
        assert!(!response.is_failed());
        assert_eq!(response.file_id, Some("file_456".to_string()));
    }

    #[test]
    fn test_music_capability() {
        use crate::traits::MusicGenerationCapability;

        let config = MinimaxiConfig::new("test-api-key");
        let client = MinimaxiClient::new(config, reqwest::Client::new());

        // Test supported models
        let models = client.get_supported_music_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "music-2.0");
    }

    #[test]
    fn test_music_request_builder() {
        use crate::types::music::MusicGenerationRequest;

        let request =
            MusicGenerationRequest::new("music-2.0", "Indie folk, melancholic, introspective")
                .with_lyrics("[verse]\nStreetlights flicker, the night breeze sighs")
                .with_sample_rate(48000)
                .with_bitrate(320000)
                .with_format("wav");

        assert_eq!(request.model, "music-2.0");
        assert_eq!(request.prompt, "Indie folk, melancholic, introspective");
        assert!(
            request
                .lyrics
                .as_ref()
                .unwrap()
                .contains("Streetlights flicker")
        );

        let setting = request.audio_setting.unwrap();
        assert_eq!(setting.sample_rate, Some(48000));
        assert_eq!(setting.bitrate, Some(320000));
        assert_eq!(setting.format, Some("wav".to_string()));
    }

    #[test]
    fn test_music_audio_setting_default() {
        use crate::types::music::MusicAudioSetting;

        let setting = MusicAudioSetting::default();
        assert_eq!(setting.sample_rate, Some(44100));
        assert_eq!(setting.bitrate, Some(256000));
        assert_eq!(setting.format, Some("mp3".to_string()));
    }
}
