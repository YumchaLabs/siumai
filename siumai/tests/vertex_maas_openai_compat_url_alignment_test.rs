#![cfg(feature = "google-vertex")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::*;
use siumai_protocol_openai::standards::openai::compat::provider_registry::ConfigurableAdapter;
use siumai_protocol_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter;
use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;
use std::sync::Arc;

fn make_ctx() -> (ProviderContext, Arc<ConfigurableAdapter>) {
    let provider_config = get_provider_config("vertex-maas").expect("vertex-maas provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config));
    let ctx = ProviderContext::new(
        "vertex-maas".to_string(),
        siumai_core::auth::vertex::google_vertex_maas_base_url("test-project", "us-central1"),
        Some("vertex-maas-test-token".to_string()),
        Default::default(),
    );
    (ctx, adapter)
}

#[test]
fn vertex_maas_chat_url_matches_real_openapi_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = "deepseek-ai/deepseek-v3.2-maas".to_string();

    assert_eq!(
        spec.chat_url(false, &req, &ctx),
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/endpoints/openapi/chat/completions"
    );
}

#[test]
fn vertex_maas_embedding_url_matches_real_openapi_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    let req = EmbeddingRequest::new(vec!["hi".into()]).with_model("text-embedding-005");
    assert_eq!(
        spec.embedding_url(&req, &ctx),
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/endpoints/openapi/embeddings"
    );
}

#[test]
fn vertex_maas_completion_url_matches_real_openapi_endpoint() {
    let (ctx, adapter) = make_ctx();
    let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

    assert_eq!(
        spec.completion_url(&ctx),
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/endpoints/openapi/completions"
    );
}
