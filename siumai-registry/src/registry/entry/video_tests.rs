use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone)]
struct BridgeVideoClient;

#[async_trait::async_trait]
impl crate::traits::VideoGenerationCapability for BridgeVideoClient {
    async fn create_video_task(
        &self,
        request: crate::types::VideoGenerationRequest,
    ) -> Result<crate::types::VideoGenerationResponse, LlmError> {
        Ok(crate::types::VideoGenerationResponse {
            task_id: format!("task:{}", request.prompt.unwrap_or_default()),
            base_resp: None,
            metadata: HashMap::new(),
            warnings: None,
            response: None,
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
            video_url: Some("https://example.com/video.mp4".to_string()),
            provider_reference: Some(crate::types::ProviderReference::single(
                "testprov_video",
                "file-123",
            )),
            duration: Some(6.0),
            video_width: Some(1920),
            video_height: Some(1080),
            base_resp: None,
            metadata: HashMap::new(),
            response: None,
        })
    }

    fn max_videos_per_call(&self) -> Option<u32> {
        Some(2)
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
        ProviderCapabilities::new()
            .with_chat()
            .with_custom_feature("video", true)
    }
}

#[tokio::test]
async fn video_model_handle_builds_client() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_video".to_string(),
        Arc::new(BridgeVideoFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.video_model("testprov_video:video-model").unwrap();

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
}

#[test]
fn video_model_handle_rejects_provider_without_video_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_chat".to_string(),
        Arc::new(TestProviderFactory::new("testprov_chat")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);

    let err = match reg.video_model("testprov_chat:model") {
        Ok(_) => panic!("video handle should be rejected without video capability"),
        Err(err) => err,
    };

    assert!(
        matches!(err, LlmError::UnsupportedOperation(message) if message.contains("video_model handle"))
    );
}

#[test]
fn video_model_handle_implements_model_metadata() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_video".to_string(),
        Arc::new(BridgeVideoFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.video_model("testprov_video:model").unwrap();

    fn assert_video_model<M>(model: &M)
    where
        M: siumai_core::video::VideoModel + ?Sized,
    {
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model),
            "testprov_video"
        );
        assert_eq!(crate::traits::ModelMetadata::model_id(model), "model");
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model),
            crate::traits::ModelSpecVersion::V1
        );
    }

    assert_video_model(&handle);
}

#[test]
fn video_model_handle_exposes_known_max_videos_per_call_defaults() {
    assert_eq!(
        super::video_model_handle_max_videos_per_call("gemini", "veo-3.1-generate-preview"),
        Some(4)
    );
    assert_eq!(
        super::video_model_handle_max_videos_per_call("vertex", "veo-3.1-generate-preview"),
        Some(4)
    );
    assert_eq!(
        super::video_model_handle_max_videos_per_call("xai", "grok-imagine-video"),
        Some(1)
    );
    assert_eq!(
        super::video_model_handle_max_videos_per_call("minimaxi", "hailuo-2.3"),
        Some(1)
    );
    assert_eq!(
        super::video_model_handle_max_videos_per_call("custom", "video-model"),
        None
    );
}

#[tokio::test]
async fn provider_factory_video_family_bridge_works() {
    let factory = BridgeVideoFactory;
    let model = factory
        .video_model_family("bridged-video-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "testprov_video"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "bridged-video-model"
    );

    let created = model
        .create_task(crate::types::VideoGenerationRequest::new(
            "bridged-video-model",
            "bridge video",
        ))
        .await
        .unwrap();
    let queried = model.query_task(&created.task_id).await.unwrap();

    assert_eq!(created.task_id, "task:bridge video");
    assert_eq!(
        queried.video_url.as_deref(),
        Some("https://example.com/video.mp4")
    );
}

#[tokio::test]
async fn provider_factory_native_video_family_path_works() {
    #[derive(Clone)]
    struct NativeVideoModel;

    impl crate::traits::ModelMetadata for NativeVideoModel {
        fn provider_id(&self) -> &str {
            "native-video"
        }

        fn model_id(&self) -> &str {
            "native-video-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::video::VideoModel for NativeVideoModel {
        async fn create_task(
            &self,
            request: crate::types::VideoGenerationRequest,
        ) -> Result<crate::types::VideoGenerationResponse, LlmError> {
            Ok(crate::types::VideoGenerationResponse {
                task_id: format!("native:{}", request.prompt.unwrap_or_default()),
                base_resp: None,
                metadata: HashMap::new(),
                warnings: None,
                response: None,
            })
        }

        async fn query_task(
            &self,
            task_id: &str,
        ) -> Result<crate::types::VideoTaskStatusResponse, LlmError> {
            Ok(crate::types::VideoTaskStatusResponse {
                task_id: task_id.to_string(),
                status: crate::types::VideoTaskStatus::Success,
                file_id: Some("native-file".to_string()),
                video_url: Some("https://example.com/native.mp4".to_string()),
                provider_reference: Some(crate::types::ProviderReference::single(
                    "native-video",
                    "native-file",
                )),
                duration: Some(4.0),
                video_width: Some(1280),
                video_height: Some(720),
                base_resp: None,
                metadata: HashMap::new(),
                response: None,
            })
        }
    }

    struct NativeOnlyVideoFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeOnlyVideoFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by native video-family test")
        }

        async fn video_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::video::VideoModel>, LlmError> {
            Ok(Arc::new(NativeVideoModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-video")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_custom_feature("video", true)
        }
    }

    let factory = NativeOnlyVideoFactory;
    let model = factory
        .video_model_family("native-video-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "native-video"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "native-video-model"
    );

    let created = model
        .create_task(crate::types::VideoGenerationRequest::new(
            "native-video-model",
            "native video",
        ))
        .await
        .unwrap();
    assert_eq!(created.task_id, "native:native video");
}

#[tokio::test]
async fn video_model_handle_uses_native_family_path_when_available() {
    #[derive(Clone)]
    struct NativeHandleVideoModel;

    impl crate::traits::ModelMetadata for NativeHandleVideoModel {
        fn provider_id(&self) -> &str {
            "native-video-handle"
        }

        fn model_id(&self) -> &str {
            "native-video-handle-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::video::VideoModel for NativeHandleVideoModel {
        async fn create_task(
            &self,
            request: crate::types::VideoGenerationRequest,
        ) -> Result<crate::types::VideoGenerationResponse, LlmError> {
            assert_eq!(request.model, "model");
            Ok(crate::types::VideoGenerationResponse {
                task_id: format!("native:{}", request.prompt.unwrap_or_default()),
                base_resp: None,
                metadata: HashMap::new(),
                warnings: None,
                response: None,
            })
        }

        async fn query_task(
            &self,
            task_id: &str,
        ) -> Result<crate::types::VideoTaskStatusResponse, LlmError> {
            Ok(crate::types::VideoTaskStatusResponse {
                task_id: task_id.to_string(),
                status: crate::types::VideoTaskStatus::Success,
                file_id: Some("native-file".to_string()),
                video_url: Some("https://example.com/native-handle.mp4".to_string()),
                provider_reference: Some(crate::types::ProviderReference::single(
                    "native-video-handle",
                    "native-file",
                )),
                duration: Some(6.0),
                video_width: Some(1920),
                video_height: Some(1080),
                base_resp: None,
                metadata: HashMap::new(),
                response: None,
            })
        }
    }

    struct NativeHandleVideoFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeHandleVideoFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by video handle")
        }

        async fn video_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy video client path should not be used by video handle")
        }

        async fn video_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::video::VideoModel>, LlmError> {
            Ok(Arc::new(NativeHandleVideoModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-video-handle")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_custom_feature("video", true)
        }
    }

    let mut providers = HashMap::new();
    providers.insert(
        "native-video-handle".to_string(),
        Arc::new(NativeHandleVideoFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.video_model("native-video-handle:model").unwrap();

    let response = siumai_core::video::VideoModel::create_task(
        &handle,
        crate::types::VideoGenerationRequest::new("", "family path"),
    )
    .await
    .unwrap();
    assert_eq!(response.task_id, "native:family path");
}

#[tokio::test]
async fn video_model_handle_reuses_cached_family_model() {
    #[derive(Clone)]
    struct CountingVideoModel;

    impl crate::traits::ModelMetadata for CountingVideoModel {
        fn provider_id(&self) -> &str {
            "cached-video"
        }

        fn model_id(&self) -> &str {
            "cached-video-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::video::VideoModel for CountingVideoModel {
        async fn create_task(
            &self,
            request: crate::types::VideoGenerationRequest,
        ) -> Result<crate::types::VideoGenerationResponse, LlmError> {
            Ok(crate::types::VideoGenerationResponse {
                task_id: format!("cached:{}", request.prompt.unwrap_or_default()),
                base_resp: None,
                metadata: HashMap::new(),
                warnings: None,
                response: None,
            })
        }

        async fn query_task(
            &self,
            task_id: &str,
        ) -> Result<crate::types::VideoTaskStatusResponse, LlmError> {
            Ok(crate::types::VideoTaskStatusResponse {
                task_id: task_id.to_string(),
                status: crate::types::VideoTaskStatus::Success,
                file_id: Some("cached-file".to_string()),
                video_url: Some("https://example.com/cached.mp4".to_string()),
                provider_reference: Some(crate::types::ProviderReference::single(
                    "cached-video",
                    "cached-file",
                )),
                duration: None,
                video_width: None,
                video_height: None,
                base_resp: None,
                metadata: HashMap::new(),
                response: None,
            })
        }
    }

    struct CountingVideoFactory {
        builds: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ProviderFactory for CountingVideoFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by video cache test")
        }

        async fn video_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::video::VideoModel>, LlmError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(CountingVideoModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("cached-video")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_custom_feature("video", true)
        }
    }

    let builds = Arc::new(AtomicUsize::new(0));
    let mut providers = HashMap::new();
    providers.insert(
        "cached-video".to_string(),
        Arc::new(CountingVideoFactory {
            builds: builds.clone(),
        }) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.video_model("cached-video:model").unwrap();

    let first = siumai_core::video::VideoModel::create_task(
        &handle,
        crate::types::VideoGenerationRequest::new("model", "once"),
    )
    .await
    .unwrap();
    let second = siumai_core::video::VideoModel::query_task(&handle, &first.task_id)
        .await
        .unwrap();

    assert_eq!(first.task_id, "cached:once");
    assert_eq!(second.file_id.as_deref(), Some("cached-file"));
    assert_eq!(builds.load(Ordering::SeqCst), 1);
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
