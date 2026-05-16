#![cfg(feature = "openai")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use siumai_protocol_openai::standards::openai::compat::provider_registry::ConfigurableAdapter;
use siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter;
use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;
use std::sync::Arc;

fn make_ctx() -> (ProviderContext, Arc<ConfigurableAdapter>) {
    let provider_config = get_provider_config("moonshotai").expect("moonshotai provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));
    let ctx = ProviderContext::new(
        "moonshotai".to_string(),
        provider_config.base_url,
        Some("sk-moonshotai-test".to_string()),
        Default::default(),
    );
    (ctx, adapter)
}

#[test]
fn moonshotai_chat_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = "kimi-k2.5".to_string();

    assert_eq!(
        spec.try_chat_url(false, &req, &ctx).unwrap(),
        "https://api.moonshot.ai/v1/chat/completions"
    );
}

#[test]
fn moonshotai_hidden_alias_resolves_to_same_base_url() {
    let canonical = get_provider_config("moonshotai").expect("moonshotai provider config");
    let alias = get_provider_config("moonshot").expect("moonshot alias config");

    assert_eq!(canonical.id, "moonshotai");
    assert_eq!(alias.id, "moonshotai");
    assert_eq!(canonical.base_url, "https://api.moonshot.ai/v1");
    assert_eq!(alias.base_url, canonical.base_url);
}
