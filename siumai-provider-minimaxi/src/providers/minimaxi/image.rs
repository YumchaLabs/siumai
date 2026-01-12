//! MiniMaxi Image Generation Helper Functions
//!
//! Internal helper functions for image generation capability implementation.

use crate::execution::executors::image::{HttpImageExecutor, ImageExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::wiring::HttpExecutionWiring;
use crate::retry_api::RetryOptions;
use crate::types::{HttpConfig, ImageGenerationRequest};
use std::sync::Arc;

/// Build image executor for MiniMaxi
#[allow(clippy::too_many_arguments)]
pub(super) fn build_image_executor(
    request: &ImageGenerationRequest,
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    http_interceptors: &[Arc<dyn HttpInterceptor>],
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Arc<HttpImageExecutor> {
    let mut wiring = HttpExecutionWiring::new(
        "minimaxi",
        http_client.clone(),
        super::utils::build_context(api_key, base_url, http_config),
    )
    .with_interceptors(http_interceptors.to_vec())
    .with_retry_options(retry_options.cloned());

    if let Some(transport) = http_transport {
        wiring = wiring.with_transport(transport);
    }

    let spec = Arc::new(super::spec::MinimaxiImageSpec::new());

    let mut builder = ImageExecutorBuilder::new("minimaxi", wiring.http_client)
        .with_spec(spec)
        .with_context(wiring.provider_context);

    if !wiring.interceptors.is_empty() {
        builder = builder.with_interceptors(wiring.interceptors);
    }

    if let Some(transport) = wiring.transport {
        builder = builder.with_transport(transport);
    }

    if let Some(retry) = wiring.retry_options {
        builder = builder.with_retry_options(retry);
    }

    builder.build_for_request(request)
}
