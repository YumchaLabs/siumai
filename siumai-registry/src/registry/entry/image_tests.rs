use super::*;

#[tokio::test]
async fn image_model_handle_builds_client() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_image".to_string(),
        Arc::new(TestImageProviderFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.image_model("testprov_image:model").unwrap();

    let out = handle
        .generate_images(ImageGenerationRequest {
            prompt: "cat".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(out.images.len(), 1);
}

#[test]
fn image_model_handle_rejects_provider_without_image_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_chat".to_string(),
        Arc::new(TestProviderFactory::new("testprov_chat")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);

    let err = match reg.image_model("testprov_chat:model") {
        Ok(_) => panic!("image handle should be rejected without image capability"),
        Err(err) => err,
    };

    assert!(
        matches!(err, LlmError::UnsupportedOperation(message) if message.contains("image_model handle"))
    );
}

#[test]
fn image_model_handle_implements_model_metadata() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_image".to_string(),
        Arc::new(TestImageProviderFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.image_model("testprov_image:model").unwrap();

    fn assert_image_model<M>(model: &M)
    where
        M: siumai_core::image::ImageModel + ?Sized,
    {
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model),
            "testprov_image"
        );
        assert_eq!(crate::traits::ModelMetadata::model_id(model), "model");
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model),
            crate::traits::ModelSpecVersion::V1
        );
    }

    assert_image_model(&handle);
}

#[tokio::test]
async fn provider_factory_image_family_bridge_works() {
    let factory = TestImageProviderFactory;
    let model = factory
        .image_model_family("bridged-image-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "testprov_image"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "bridged-image-model"
    );

    let response = model
        .generate(ImageGenerationRequest {
            prompt: "cat".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.images.len(), 1);
}

#[tokio::test]
async fn provider_factory_native_image_family_path_works() {
    #[derive(Clone)]
    struct NativeImageModel;

    impl crate::traits::ModelMetadata for NativeImageModel {
        fn provider_id(&self) -> &str {
            "native-image"
        }

        fn model_id(&self) -> &str {
            "native-image-model"
        }
    }

    #[async_trait::async_trait]
    impl crate::image::ImageModelV3 for NativeImageModel {
        async fn generate(
            &self,
            request: ImageGenerationRequest,
        ) -> Result<ImageGenerationResponse, LlmError> {
            Ok(ImageGenerationResponse {
                images: vec![crate::types::GeneratedImage {
                    url: Some(format!("https://example.com/{}.png", request.prompt)),
                    b64_json: None,
                    format: None,
                    width: None,
                    height: None,
                    revised_prompt: None,
                    metadata: std::collections::HashMap::new(),
                }],
                metadata: std::collections::HashMap::new(),
                warnings: None,
                response: None,
            })
        }
    }

    struct NativeOnlyImageFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeOnlyImageFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by native image-family test")
        }

        async fn image_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::image::ImageModel>, LlmError> {
            Ok(Arc::new(NativeImageModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-image")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_image_generation()
        }
    }

    let factory = NativeOnlyImageFactory;
    let model = factory
        .image_model_family("native-image-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "native-image"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "native-image-model"
    );

    let response = model
        .generate(ImageGenerationRequest {
            prompt: "cat".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.images.len(), 1);
}

#[tokio::test]
async fn image_model_handle_uses_native_family_path_when_available() {
    #[derive(Clone)]
    struct NativeImageModel;

    impl crate::traits::ModelMetadata for NativeImageModel {
        fn provider_id(&self) -> &str {
            "native-image-handle"
        }

        fn model_id(&self) -> &str {
            "native-image-handle-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::image::ImageModelV3 for NativeImageModel {
        async fn generate(
            &self,
            request: ImageGenerationRequest,
        ) -> Result<ImageGenerationResponse, LlmError> {
            Ok(ImageGenerationResponse {
                images: vec![crate::types::GeneratedImage {
                    url: Some(format!("https://example.com/{}.png", request.prompt)),
                    b64_json: None,
                    format: None,
                    width: None,
                    height: None,
                    revised_prompt: None,
                    metadata: std::collections::HashMap::new(),
                }],
                metadata: std::collections::HashMap::new(),
                warnings: None,
                response: None,
            })
        }
    }

    struct NativeImageHandleFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeImageHandleFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by image handle")
        }

        async fn image_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy image client path should not be used by image handle")
        }

        async fn image_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::image::ImageModel>, LlmError> {
            Ok(Arc::new(NativeImageModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-image-handle")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_image_generation()
        }
    }

    let mut providers = HashMap::new();
    providers.insert(
        "native-image-handle".to_string(),
        Arc::new(NativeImageHandleFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.image_model("native-image-handle:model").unwrap();

    let response = handle
        .generate_images(ImageGenerationRequest {
            prompt: "cat".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.images.len(), 1);
}
