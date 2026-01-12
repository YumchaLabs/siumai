#![cfg(feature = "openai")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use siumai_protocol_openai::standards::openai::compat::provider_registry::ConfigurableAdapter;
use siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter;
use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;
use std::sync::Arc;

fn make_ctx() -> (ProviderContext, Arc<ConfigurableAdapter>) {
    let provider_config = get_provider_config("perplexity").expect("perplexity provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));
    let ctx = ProviderContext::new(
        "perplexity".to_string(),
        provider_config.base_url,
        Some("pplx-test".to_string()),
        Default::default(),
    );
    (ctx, adapter)
}

#[test]
fn perplexity_chat_url_matches_official_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = "sonar".to_string();

    assert_eq!(
        spec.chat_url(false, &req, &ctx),
        "https://api.perplexity.ai/chat/completions"
    );
}
