#![cfg(feature = "openai")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use siumai_protocol_openai::standards::openai::compat::provider_registry::ConfigurableAdapter;
use siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter;
use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;
use std::sync::Arc;

fn make_ctx() -> (ProviderContext, Arc<ConfigurableAdapter>) {
    let provider_config = get_provider_config("mistral").expect("mistral provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));
    let ctx = ProviderContext::new(
        "mistral".to_string(),
        provider_config.base_url,
        Some("sk-mistral-test".to_string()),
        Default::default(),
    );
    (ctx, adapter)
}

#[test]
fn mistral_chat_url_matches_official_openapi_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = "mistral-large-latest".to_string();

    assert_eq!(
        spec.chat_url(false, &req, &ctx),
        "https://api.mistral.ai/v1/chat/completions"
    );
}

#[test]
fn mistral_embedding_url_matches_official_openapi_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let req = EmbeddingRequest::new(vec!["hi".into()]).with_model("mistral-embed");
    assert_eq!(
        spec.embedding_url(&req, &ctx),
        "https://api.mistral.ai/v1/embeddings"
    );
}
