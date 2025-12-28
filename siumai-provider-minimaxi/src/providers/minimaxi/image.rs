//! MiniMaxi Image Generation Helper Functions
//!
//! Internal helper functions for image generation capability implementation.

use crate::execution::executors::image::{HttpImageExecutor, ImageExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::types::{HttpConfig, ImageGenerationRequest};
use std::sync::Arc;

/// Build image executor for MiniMaxi
pub(super) fn build_image_executor(
    request: &ImageGenerationRequest,
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    http_interceptors: &[Arc<dyn HttpInterceptor>],
) -> Arc<HttpImageExecutor> {
    let ctx = super::utils::build_context(api_key, base_url, http_config);

    let spec = Arc::new(super::spec::MinimaxiImageSpec::new());

    let mut builder = ImageExecutorBuilder::new("minimaxi", http_client.clone())
        .with_spec(spec)
        .with_context(ctx);

    if !http_interceptors.is_empty() {
        builder = builder.with_interceptors(http_interceptors.to_vec());
    }

    if let Some(retry) = retry_options {
        builder = builder.with_retry_options(retry.clone());
    }

    builder.build_for_request(request)
}
