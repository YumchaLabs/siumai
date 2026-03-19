#![cfg(feature = "openai")]

use siumai::prelude::unified::{ChatResponse, MessageContent};
use siumai::provider_ext::openrouter::OpenRouterChatResponseExt;
use siumai::provider_ext::perplexity::PerplexityChatResponseExt;
use std::collections::HashMap;

#[test]
fn openrouter_typed_metadata_requires_openrouter_root_namespace() {
    let mut vendor_response = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut vendor_metadata = HashMap::new();
    vendor_metadata.insert(
        "logprobs".to_string(),
        serde_json::json!([{
            "token": "ok",
            "logprob": -0.1,
            "bytes": [111, 107],
            "top_logprobs": []
        }]),
    );
    vendor_metadata.insert(
        "sources".to_string(),
        serde_json::json!([{
            "id": "src_1",
            "source_type": "url",
            "url": "https://openrouter.ai/docs",
            "title": "OpenRouter Docs"
        }]),
    );
    vendor_response.provider_metadata = Some(HashMap::from([(
        "openrouter".to_string(),
        vendor_metadata.clone(),
    )]));

    let vendor_view = vendor_response
        .openrouter_metadata()
        .expect("openrouter metadata should parse under vendor root");
    assert_eq!(vendor_view.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        vendor_view
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://openrouter.ai/docs")
    );

    let mut generic_response = ChatResponse::new(MessageContent::Text("ok".to_string()));
    generic_response.provider_metadata = Some(HashMap::from([(
        "openai_compatible".to_string(),
        vendor_metadata,
    )]));

    assert!(
        generic_response.openrouter_metadata().is_none(),
        "typed OpenRouter metadata must not fall back to a generic compat root"
    );
}

#[test]
fn perplexity_typed_metadata_requires_perplexity_root_namespace() {
    let mut vendor_response = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut vendor_metadata = HashMap::new();
    vendor_metadata.insert(
        "citations".to_string(),
        serde_json::json!(["https://example.com/rust"]),
    );
    vendor_metadata.insert(
        "images".to_string(),
        serde_json::json!([{
            "image_url": "https://images.example.com/rust.png",
            "origin_url": "https://example.com/rust",
            "height": 900,
            "width": 1600
        }]),
    );
    vendor_metadata.insert(
        "usage".to_string(),
        serde_json::json!({
            "citation_tokens": 7,
            "num_search_queries": 2,
            "reasoning_tokens": 3
        }),
    );
    vendor_response.provider_metadata = Some(HashMap::from([(
        "perplexity".to_string(),
        vendor_metadata.clone(),
    )]));

    let vendor_view = vendor_response
        .perplexity_metadata()
        .expect("perplexity metadata should parse under vendor root");
    assert_eq!(
        vendor_view.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(vendor_view.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        vendor_view
            .usage
            .as_ref()
            .and_then(|usage| usage.reasoning_tokens),
        Some(3)
    );

    let mut generic_response = ChatResponse::new(MessageContent::Text("ok".to_string()));
    generic_response.provider_metadata = Some(HashMap::from([(
        "openai_compatible".to_string(),
        vendor_metadata,
    )]));

    assert!(
        generic_response.perplexity_metadata().is_none(),
        "typed Perplexity metadata must not fall back to a generic compat root"
    );
}
