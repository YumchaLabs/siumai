use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::traits::{ChatCapability, ProviderCapabilities};

use super::ProviderFactory;

pub static TEST_BUILD_COUNT: AtomicUsize = AtomicUsize::new(0);

static REG_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn reg_test_guard() -> std::sync::MutexGuard<'static, ()> {
    REG_TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

pub struct TestProvClient;

#[async_trait::async_trait]
impl ChatCapability for TestProvClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Ok(crate::types::ChatResponse::new(
            crate::types::MessageContent::Text("ok".to_string()),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation("mock stream".into()))
    }
}

impl LlmClient for TestProvClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvClient)
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
    }
}

pub struct TestProvEmbedClient;

#[async_trait::async_trait]
impl crate::traits::EmbeddingCapability for TestProvEmbedClient {
    async fn embed(&self, input: Vec<String>) -> Result<crate::types::EmbeddingResponse, LlmError> {
        Ok(crate::types::EmbeddingResponse::new(
            vec![vec![input.len() as f32]],
            "test-embed-model".to_string(),
        ))
    }

    fn embedding_dimension(&self) -> usize {
        1
    }
}

impl LlmClient for TestProvEmbedClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_embed")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvEmbedClient)
    }

    fn as_embedding_capability(&self) -> Option<&dyn crate::traits::EmbeddingCapability> {
        Some(self)
    }
}

#[async_trait::async_trait]
impl ChatCapability for TestProvEmbedClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat not supported in TestProvEmbedClient".into(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat stream not supported in TestProvEmbedClient".into(),
        ))
    }
}

pub struct TestProvImageClient;

#[async_trait::async_trait]
impl crate::traits::ImageGenerationCapability for TestProvImageClient {
    async fn generate_images(
        &self,
        request: crate::types::ImageGenerationRequest,
    ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
        Ok(crate::types::ImageGenerationResponse {
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

impl LlmClient for TestProvImageClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_image")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvImageClient)
    }

    fn as_image_generation_capability(
        &self,
    ) -> Option<&dyn crate::traits::ImageGenerationCapability> {
        Some(self)
    }
}

pub struct TestImageProviderFactory;

#[async_trait::async_trait]
impl ProviderFactory for TestImageProviderFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvImageClient))
    }

    async fn image_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvImageClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_image")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }
}

pub struct TestProviderFactory {
    id: &'static str,
}

impl TestProviderFactory {
    pub const fn new(id: &'static str) -> Self {
        Self { id }
    }
}

#[async_trait::async_trait]
impl ProviderFactory for TestProviderFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        TEST_BUILD_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(TestProvClient))
    }

    async fn embedding_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvEmbedClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(self.id)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        match self.id {
            "testprov_embed" => ProviderCapabilities::new().with_embedding(),
            _ => ProviderCapabilities::new().with_chat(),
        }
    }
}
