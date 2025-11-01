//! MiniMaxi Audio Helper Functions
//!
//! Internal helper functions for audio capability implementation.

use std::sync::Arc;

use crate::core::ProviderContext;
use crate::execution::executors::audio::{AudioExecutorBuilder, HttpAudioExecutor};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;

use super::spec::MinimaxiSpec;
use super::transformers::audio::MinimaxiAudioTransformer;

/// Build audio executor for MiniMaxi
pub(super) fn build_audio_executor(
    api_key: &str,
    base_url: &str,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    http_interceptors: &[Arc<dyn HttpInterceptor>],
) -> Arc<HttpAudioExecutor> {
    let ctx = ProviderContext {
        provider_id: "minimaxi".to_string(),
        api_key: Some(api_key.to_string()),
        base_url: base_url.to_string(),
        http_extra_headers: Default::default(),
        organization: None,
        project: None,
        extras: Default::default(),
    };

    let spec = Arc::new(MinimaxiSpec::new());
    let transformer = Arc::new(MinimaxiAudioTransformer);

    let mut builder = AudioExecutorBuilder::new("minimaxi", http_client.clone())
        .with_spec(spec)
        .with_context(ctx)
        .with_transformer(transformer)
        .with_interceptors(http_interceptors.to_vec());

    if let Some(retry_opts) = retry_options {
        builder = builder.with_retry_options(retry_opts.clone());
    }

    builder.build()
}
