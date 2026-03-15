use super::*;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct BridgeMusicClient;

#[async_trait::async_trait]
impl crate::traits::MusicGenerationCapability for BridgeMusicClient {
    async fn generate_music(
        &self,
        request: crate::types::MusicGenerationRequest,
    ) -> Result<crate::types::MusicGenerationResponse, LlmError> {
        Ok(crate::types::MusicGenerationResponse {
            audio_data: request.prompt.into_bytes(),
            metadata: crate::types::MusicMetadata {
                music_duration: Some(12_000),
                music_sample_rate: Some(44_100),
                music_channel: Some(2),
                bitrate: Some(256_000),
                music_size: Some(5),
            },
        })
    }

    fn get_supported_music_models(&self) -> Vec<String> {
        vec!["music-model".to_string()]
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        vec!["mp3".to_string(), "wav".to_string()]
    }

    fn supports_lyrics(&self) -> bool {
        true
    }
}

impl LlmClient for BridgeMusicClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_music")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["music-model".into()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_custom_feature("music", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_music_generation_capability(
        &self,
    ) -> Option<&dyn crate::traits::MusicGenerationCapability> {
        Some(self)
    }
}

struct BridgeMusicFactory;

#[async_trait::async_trait]
impl ProviderFactory for BridgeMusicFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(BridgeMusicClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_music")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_custom_feature("music", true)
    }
}

#[tokio::test]
async fn language_model_handle_delegates_music_generation_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_music".to_string(),
        Arc::new(BridgeMusicFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("testprov_music:music-model").unwrap();

    let response = handle
        .generate_music(crate::types::MusicGenerationRequest::new(
            "music-model",
            "ambient piano",
        ))
        .await
        .unwrap();

    assert_eq!(response.audio_data, b"ambient piano");
    assert_eq!(handle.get_supported_music_models(), vec!["music-model"]);
    assert_eq!(handle.get_supported_audio_formats(), Vec::<String>::new());
    assert!(handle.supports_lyrics());
    assert!(handle.as_music_generation_capability().is_some());
}
