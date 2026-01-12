#![cfg(feature = "openai")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use siumai_protocol_openai::standards::openai::compat::provider_registry::ConfigurableAdapter;
use siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter;
use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;
use std::sync::Arc;

fn make_ctx() -> (ProviderContext, Arc<ConfigurableAdapter>) {
    let provider_config = get_provider_config("siliconflow").expect("siliconflow provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));
    let ctx = ProviderContext::new(
        "siliconflow".to_string(),
        provider_config.base_url,
        Some("sk-siliconflow-test".to_string()),
        Default::default(),
    );
    (ctx, adapter)
}

#[test]
fn siliconflow_chat_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = "deepseek-ai/DeepSeek-V3".to_string();

    assert_eq!(
        spec.chat_url(false, &req, &ctx),
        "https://api.siliconflow.cn/v1/chat/completions"
    );
}

#[test]
fn siliconflow_embedding_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let req = EmbeddingRequest::new(vec!["hi".into()]).with_model("text-embedding-3-small");
    assert_eq!(
        spec.embedding_url(&req, &ctx),
        "https://api.siliconflow.cn/v1/embeddings"
    );
}

#[test]
fn siliconflow_image_generation_url_matches_official_openai_compatible_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let req = ImageGenerationRequest {
        prompt: "a cat".to_string(),
        count: 1,
        model: Some("stability-ai/sdxl".to_string()),
        ..Default::default()
    };
    assert_eq!(
        spec.image_url(&req, &ctx),
        "https://api.siliconflow.cn/v1/images/generations"
    );
}

#[test]
fn siliconflow_rerank_url_matches_official_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let req = RerankRequest::new(
        "BAAI/bge-reranker-v2-m3".to_string(),
        "query".to_string(),
        vec!["a".to_string(), "b".to_string()],
    );
    assert_eq!(
        spec.rerank_url(&req, &ctx),
        "https://api.siliconflow.cn/v1/rerank"
    );
}
