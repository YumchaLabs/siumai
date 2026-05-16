#![cfg(feature = "togetherai")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use siumai_protocol_openai::standards::openai::compat::provider_registry::ConfigurableAdapter;
use siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter;
use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;
use std::sync::Arc;

fn make_ctx() -> (ProviderContext, Arc<ConfigurableAdapter>) {
    let provider_config = get_provider_config("togetherai").expect("togetherai provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));
    let ctx = ProviderContext::new(
        "togetherai".to_string(),
        provider_config.base_url,
        Some("sk-togetherai-test".to_string()),
        Default::default(),
    );
    (ctx, adapter)
}

#[test]
fn togetherai_chat_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string();

    assert_eq!(
        spec.try_chat_url(false, &req, &ctx).unwrap(),
        "https://api.together.xyz/v1/chat/completions"
    );
}

#[test]
fn togetherai_embedding_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let req = EmbeddingRequest::new(vec!["hi".into()])
        .with_model("togethercomputer/m2-bert-80M-8k-retrieval");
    assert_eq!(
        spec.try_embedding_url(&req, &ctx).unwrap(),
        "https://api.together.xyz/v1/embeddings"
    );
}

#[test]
fn togetherai_completion_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    assert_eq!(
        spec.completion_url(&ctx),
        "https://api.together.xyz/v1/completions"
    );
}
