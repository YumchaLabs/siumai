#![cfg(feature = "deepinfra")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use siumai_protocol_openai::standards::openai::compat::provider_registry::ConfigurableAdapter;
use siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter;
use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;
use std::sync::Arc;

fn make_ctx() -> (ProviderContext, Arc<ConfigurableAdapter>) {
    let provider_config = get_provider_config("deepinfra").expect("deepinfra provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));
    let ctx = ProviderContext::new(
        "deepinfra".to_string(),
        provider_config.base_url,
        Some("sk-deepinfra-test".to_string()),
        Default::default(),
    );
    (ctx, adapter)
}

#[test]
fn deepinfra_chat_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = "meta-llama/Llama-3.3-70B-Instruct".to_string();

    assert_eq!(
        spec.chat_url(false, &req, &ctx),
        "https://api.deepinfra.com/v1/openai/chat/completions"
    );
}

#[test]
fn deepinfra_embedding_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let req = EmbeddingRequest::new(vec!["hi".into()]).with_model("BAAI/bge-base-en-v1.5");
    assert_eq!(
        spec.embedding_url(&req, &ctx),
        "https://api.deepinfra.com/v1/openai/embeddings"
    );
}

#[test]
fn deepinfra_completion_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    assert_eq!(
        spec.completion_url(&ctx),
        "https://api.deepinfra.com/v1/openai/completions"
    );
}
