#![cfg(feature = "openai")]

#[test]
fn cerebras_provider_ext_exports_package_surface() {
    use siumai::provider_ext::cerebras::{
        CerebrasChatModelId, CerebrasClient, CerebrasConfig, CerebrasProviderSettings, VERSION,
        chat, create_cerebras, model_sets,
    };

    let _client_type: Option<CerebrasClient> = None;
    let _config_type: Option<CerebrasConfig> = None;
    let _model_id: CerebrasChatModelId = chat::LLAMA3_1_8B.to_string();

    let config = CerebrasProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model(chat::ZAI_GLM_4_7)
        .expect("settings should build config");

    assert_eq!(config.provider_id, "cerebras");
    assert_eq!(config.base_url, "https://api.cerebras.ai/v1");
    assert_eq!(config.common_params.model, chat::ZAI_GLM_4_7);
    assert_eq!(model_sets::CHAT, chat::LLAMA3_1_8B);
    assert!(!VERSION.is_empty());

    let client = tokio_test_build(
        create_cerebras()
            .api_key("test-key")
            .model(chat::LLAMA3_1_8B),
    );
    assert_eq!(client.metadata().provider_id, "cerebras");
    assert_eq!(
        client.metadata().provider_type,
        siumai::prelude::unified::ProviderType::Cerebras
    );
}

#[test]
fn cerebras_provider_alias_and_compat_builder_are_available() {
    let provider_client = tokio_test_build(
        siumai::providers::cerebras::cerebras()
            .api_key("test-key")
            .model(siumai::providers::cerebras::chat::GPT_OSS_120B),
    );
    assert_eq!(provider_client.metadata().provider_id, "cerebras");

    let compat_client = tokio_test_build(
        siumai::compat::Provider::cerebras()
            .api_key("test-key")
            .model("zai-glm-4.7"),
    );
    assert_eq!(compat_client.metadata().provider_id, "cerebras");
}

fn tokio_test_build(builder: siumai::compat::SiumaiBuilder) -> siumai::compat::Siumai {
    tokio::runtime::Runtime::new()
        .expect("tokio runtime")
        .block_on(async { builder.build().await.expect("builder should build") })
}
