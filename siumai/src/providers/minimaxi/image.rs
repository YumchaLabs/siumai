//! MiniMaxi Image Generation Helper Functions
//!
//! Internal helper functions for image generation capability implementation.

use crate::core::ProviderContext;
use crate::execution::executors::image::{HttpImageExecutor, ImageExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::providers::minimaxi::spec::MinimaxiSpec;
use crate::retry_api::RetryOptions;
use std::sync::Arc;

/// Build image executor for MiniMaxi
pub(super) fn build_image_executor(
    api_key: &str,
    base_url: &str,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    http_interceptors: &[Arc<dyn HttpInterceptor>],
) -> Arc<HttpImageExecutor> {
    // MiniMaxi image API uses OpenAI-style Bearer token authentication
    // We need to inject the Authorization header into http_extra_headers
    let extra_headers = super::utils::create_openai_auth_headers(api_key);

    let ctx = ProviderContext {
        provider_id: "minimaxi".to_string(),
        api_key: Some(api_key.to_string()),
        base_url: base_url.to_string(),
        http_extra_headers: extra_headers,
        organization: None,
        project: None,
        extras: Default::default(),
    };

    let spec = Arc::new(MinimaxiSpec::new());

    let mut builder = ImageExecutorBuilder::new("minimaxi", http_client.clone())
        .with_spec(spec)
        .with_context(ctx);

    // Note: MiniMaxi image generation currently does not use the unified OpenAI
    // image standard. Image transformers are wired via dedicated transformers
    // when std-openai is enabled. For the minimal anthropic-only feature
    // combination, we keep image unsupported here.

    if !http_interceptors.is_empty() {
        builder = builder.with_interceptors(http_interceptors.to_vec());
    }

    if let Some(retry) = retry_options {
        builder = builder.with_retry_options(retry.clone());
    }

    builder.build()
}
