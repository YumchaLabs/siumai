//! Unified entry for test support modules
//!
//! - Provide lightweight wrappers for different mock backends (mockito, etc.)
//! - Reduce boilerplate and soften the impact of API changes

pub mod mockito;

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::experimental::core::{ChatTransformers, ProviderContext, ProviderSpec};
use siumai::prelude::unified::{ChatRequest, LlmError, ProviderCapabilities};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// A minimal ProviderSpec that uses a bad token for the first request and
/// a good token for the next one. This is used to validate single 401 retry
/// with header rebuild behavior.
pub struct FlippingAuthSpec {
    pub counter: Arc<AtomicUsize>,
}

impl ProviderSpec for FlippingAuthSpec {
    fn id(&self) -> &'static str {
        "test"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_audio()
    }

    fn build_headers(&self, _ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        let token = if n == 0 { "bad" } else { "ok" };
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", token)).unwrap(),
        );
        Ok(headers)
    }

    fn chat_url(&self, _stream: bool, _req: &ChatRequest, ctx: &ProviderContext) -> String {
        format!("{}/json", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        // Not used in tests; panic to prevent accidental usage
        unimplemented!("FlippingAuthSpec is only for header build tests")
    }
}
