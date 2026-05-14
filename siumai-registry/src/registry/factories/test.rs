//! Provider factory implementations.

use super::*;

/// Test provider factory (for testing)
#[cfg(test)]
pub struct TestProviderFactory;

#[cfg(test)]
#[async_trait::async_trait]
impl ProviderFactory for TestProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
    }

    async fn compat_language_client(
        &self,
        _model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        use crate::registry::entry::TEST_BUILD_COUNT;
        TEST_BUILD_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(Arc::new(crate::registry::entry::TestProvClient::new(
            "test", _model_id,
        )))
    }

    async fn compat_embedding_client(
        &self,
        _model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(crate::registry::entry::TestProvEmbedClient::new(
            _model_id,
        )))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov")
    }
}
