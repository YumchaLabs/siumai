//! MiniMaxi Audio Helper Functions
//!
//! Internal helper functions for audio capability implementation.

use std::sync::Arc;

use crate::execution::executors::audio::{AudioExecutorBuilder, HttpAudioExecutor};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::wiring::HttpExecutionWiring;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;

use super::spec::MinimaxiAudioSpec;

/// Build audio executor for MiniMaxi
pub(super) fn build_audio_executor(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    http_interceptors: &[Arc<dyn HttpInterceptor>],
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Arc<HttpAudioExecutor> {
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

    let spec = Arc::new(MinimaxiAudioSpec::new());

    let mut builder = AudioExecutorBuilder::new("minimaxi", wiring.http_client)
        .with_spec(spec)
        .with_context(wiring.provider_context)
        .with_interceptors(wiring.interceptors);

    if let Some(transport) = wiring.transport {
        builder = builder.with_transport(transport);
    }

    if let Some(retry_opts) = wiring.retry_options {
        builder = builder.with_retry_options(retry_opts);
    }

    builder.build()
}
