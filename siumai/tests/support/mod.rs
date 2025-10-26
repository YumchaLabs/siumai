use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Flips Authorization header between attempts to simulate token refresh on 401.
pub struct FlippingAuthSpec {
    pub counter: Arc<AtomicUsize>,
}

impl siumai::core::ProviderSpec for FlippingAuthSpec {
    fn id(&self) -> &'static str {
        "test"
    }
    fn capabilities(&self) -> siumai::traits::ProviderCapabilities {
        siumai::traits::ProviderCapabilities::new().with_audio()
    }
    fn build_headers(
        &self,
        _ctx: &siumai::core::ProviderContext,
    ) -> Result<reqwest::header::HeaderMap, siumai::LlmError> {
        use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        let token = if n == 0 { "bad" } else { "ok" };
        let mut h = HeaderMap::new();
        h.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_str(&format!("Bearer {}", token)).unwrap(),
        );
        Ok(h)
    }
    fn chat_url(
        &self,
        _stream: bool,
        _req: &siumai::types::ChatRequest,
        ctx: &siumai::core::ProviderContext,
    ) -> String {
        format!("{}/never", ctx.base_url)
    }
    fn choose_chat_transformers(
        &self,
        _req: &siumai::types::ChatRequest,
        _ctx: &siumai::core::ProviderContext,
    ) -> siumai::core::ChatTransformers {
        panic!("not used in these tests")
    }
}
