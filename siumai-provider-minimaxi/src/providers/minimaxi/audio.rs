//! MiniMaxi Audio Helper Functions
//!
//! Internal helper functions for audio capability implementation.

use std::sync::Arc;

use crate::execution::executors::audio::{AudioExecutorBuilder, HttpAudioExecutor};
use crate::execution::http::interceptor::HttpInterceptor;
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
) -> Arc<HttpAudioExecutor> {
    let ctx = super::utils::build_context(api_key, base_url, http_config);

    let spec = Arc::new(MinimaxiAudioSpec::new());

    let mut builder = AudioExecutorBuilder::new("minimaxi", http_client.clone())
        .with_spec(spec)
        .with_context(ctx)
        .with_interceptors(http_interceptors.to_vec());

    if let Some(retry_opts) = retry_options {
        builder = builder.with_retry_options(retry_opts.clone());
    }

    builder.build()
}
