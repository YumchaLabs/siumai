use super::*;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct BridgeVideoClient;

#[async_trait::async_trait]
impl crate::traits::VideoGenerationCapability for BridgeVideoClient {
    async fn create_video_task(
        &self,
        request: crate::types::VideoGenerationRequest,
    ) -> Result<crate::types::VideoGenerationResponse, LlmError> {
        Ok(crate::types::VideoGenerationResponse {
            task_id: format!("task:{}", request.prompt),
            base_resp: None,
        })
    }

    async fn query_video_task(
        &self,
        task_id: &str,
    ) -> Result<crate::types::VideoTaskStatusResponse, LlmError> {
        Ok(crate::types::VideoTaskStatusResponse {
            task_id: task_id.to_string(),
            status: crate::types::VideoTaskStatus::Success,
            file_id: Some("file-123".to_string()),
            video_width: Some(1920),
            video_height: Some(1080),
            base_resp: None,
        })
    }

    fn get_supported_models(&self) -> Vec<String> {
        vec!["video-model".to_string()]
    }

    fn get_supported_resolutions(&self, _model: &str) -> Vec<String> {
        vec!["1080P".to_string()]
    }

    fn get_supported_durations(&self, _model: &str) -> Vec<u32> {
        vec![6, 10]
    }
}

impl LlmClient for BridgeVideoClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_video")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["video-model".into()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_custom_feature("video", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_video_generation_capability(
        &self,
    ) -> Option<&dyn crate::traits::VideoGenerationCapability> {
        Some(self)
    }
}

struct BridgeVideoFactory;

#[async_trait::async_trait]
impl ProviderFactory for BridgeVideoFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(BridgeVideoClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_video")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_custom_feature("video", true)
    }
}

#[tokio::test]
async fn language_model_handle_delegates_video_generation_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_video".to_string(),
        Arc::new(BridgeVideoFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("testprov_video:video-model").unwrap();

    let created = handle
        .create_video_task(crate::types::VideoGenerationRequest::new(
            "video-model",
            "tiny robot",
        ))
        .await
        .unwrap();
    let queried = handle.query_video_task(&created.task_id).await.unwrap();

    assert_eq!(created.task_id, "task:tiny robot");
    assert_eq!(queried.file_id.as_deref(), Some("file-123"));
    assert_eq!(handle.get_supported_models(), vec!["video-model"]);
    assert_eq!(
        handle.get_supported_resolutions("video-model"),
        Vec::<String>::new()
    );
    assert_eq!(
        handle.get_supported_durations("video-model"),
        Vec::<u32>::new()
    );
    assert!(handle.as_video_generation_capability().is_some());
}
