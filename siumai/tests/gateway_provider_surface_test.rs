#![cfg(feature = "gateway")]

use siumai::prelude::unified::{ChatMessage, ChatRequest, EmbeddingRequest, ModelMetadata};
use siumai::provider_ext::gateway::{
    GatewayBuilder, GatewayChatRequestExt, GatewayClient, GatewayConfig,
    GatewayEmbeddingRequestExt, GatewayOptions, GatewayServiceTier, GatewaySort, create_gateway,
};

#[test]
fn gateway_provider_ext_exports_package_surface() {
    let _builder_type: Option<GatewayBuilder> = None;
    let _client_type: Option<GatewayClient> = None;

    let config = GatewayConfig::new("test-key")
        .with_base_url("https://gateway.test/v4/ai/")
        .with_model("openai/gpt-5-mini")
        .with_team_id_or_slug("team_123");

    assert_eq!(config.base_url, "https://gateway.test/v4/ai");
    assert_eq!(config.common_params.model, "openai/gpt-5-mini");
    assert_eq!(config.team_id_or_slug.as_deref(), Some("team_123"));

    let options = GatewayOptions::new()
        .with_only(["openai"])
        .with_sort(GatewaySort::Cost)
        .with_service_tier(GatewayServiceTier::Priority);
    let chat_req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
        .with_gateway_options(options.clone());
    let embed_req = EmbeddingRequest::single("hi").with_gateway_options(options.clone());

    assert!(chat_req.provider_options_map.get("gateway").is_some());
    assert!(embed_req.provider_options_map.get("gateway").is_some());

    let client = create_gateway()
        .api_key("test-key")
        .base_url("https://gateway.test/v4/ai/")
        .model("openai/gpt-5-mini")
        .team_id_or_slug("team_123")
        .build()
        .expect("gateway builder should build");
    assert_eq!(ModelMetadata::provider_id(&client), "gateway");
}

#[test]
fn gateway_provider_alias_and_compat_builder_are_available() {
    let provider_client = siumai::providers::gateway::gateway()
        .api_key("test-key")
        .model("openai/gpt-5-mini")
        .build()
        .expect("gateway provider alias should build");
    assert_eq!(ModelMetadata::provider_id(&provider_client), "gateway");

    let compat_client = siumai::compat::Provider::gateway()
        .api_key("test-key")
        .model("openai/gpt-5-mini")
        .build()
        .expect("gateway compat builder should build");
    assert_eq!(ModelMetadata::provider_id(&compat_client), "gateway");
}
