use super::{Siumai, accepts_language_model, error_diagnostics};
use siumai::Model;
use siumai::core::{
    ApiModeId, ApiStability, ModelCatalog, ModelFamily, OfficialSource, PlatformId, ProfileId,
    ProtocolContractId, ProtocolId, ProviderId, ProviderInstanceId, ProviderProfile,
    ReplayDomainId, SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity,
    VerifiedSupportClaim,
};
use siumai::providers::openai_compatible::{
    OpenAiCompatibleApiMode, OpenAiCompatibleCredential, OpenAiCompatibleLanguageModel,
    OpenAiCompatibleProfile, OpenAiCompatibleProvider, OpenAiCompatibleProviderBuilder,
};
use siumai::transport::{EndpointConfig, OfficialOrigin};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID as CHAT_API_MODE_ID, ChatCompletionsDialect, PROTOCOL_ID as CHAT_PROTOCOL_ID,
};

const CREDENTIAL_CANARY: &str = "compatible-facade-secret-canary";

pub(super) fn custom_profile() -> Result<OpenAiCompatibleProfile, Box<dyn std::error::Error>> {
    Ok(OpenAiCompatibleProfile::public_custom(
        ProviderId::new("custom-compatible")?,
        "https://custom-compatible.example/v1",
        ReplayDomainId::new("custom-compatible-replay")?,
        OpenAiCompatibleApiMode::Responses,
    )?)
}

fn verified_profile() -> Result<OpenAiCompatibleProfile, Box<dyn std::error::Error>> {
    let provider = ProviderId::new("verified-compatible")?;
    let scope = SupportScope::new(
        provider,
        PlatformId::new("official-api")?,
        ModelFamily::Language,
        ProtocolId::new(CHAT_PROTOCOL_ID)?,
        ApiModeId::new(CHAT_API_MODE_ID)?,
    );
    let verified_at: VerificationDate = serde_json::from_str("\"2026-08-19\"")?;
    let evidence = VerificationEvidence::new(
        OfficialSource::new("https://verified-compatible.example/docs")?,
        verified_at,
        ProtocolContractId::new("verified-compatible-2026-08")?,
    );
    let profile = ProviderProfile::verified(
        ProfileId::new("verified-compatible")?,
        vec![VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Compatible,
            ApiStability::Stable,
            evidence,
        )],
        ModelCatalog::default(),
    )?;
    let endpoint = EndpointConfig::official(
        "https://verified-compatible.example/v1",
        OfficialOrigin::new("https://verified-compatible.example")?,
    )?;
    Ok(OpenAiCompatibleProfile::verified_chat(
        profile,
        endpoint,
        ChatCompletionsDialect::generic(),
    )?)
}

fn accepts_compatible_provider(provider: &OpenAiCompatibleProvider) {
    let _ = provider;
}

fn accepts_compatible_model(model: &OpenAiCompatibleLanguageModel) {
    let _ = model;
}

#[test]
fn custom_and_verified_profiles_pass_through_the_typed_stages()
-> Result<(), Box<dyn std::error::Error>> {
    let configured_instance = ProviderInstanceId::new();
    let custom = Siumai::builder()
        .openai_compatible()
        .profile(custom_profile()?)
        .api_key("test-api-key")
        .configure_provider(|builder: OpenAiCompatibleProviderBuilder| {
            builder.with_provider_instance(configured_instance.clone())
        })
        .build()?;
    let verified = Siumai::builder()
        .openai_compatible()
        .profile(verified_profile()?)
        .credential(OpenAiCompatibleCredential::api_key("test-api-key"))
        .build()?;

    let custom_model = custom.language("future-compatible-model")?;
    let verified_model = verified.language("future-compatible-model")?;
    accepts_language_model(&custom_model);
    accepts_language_model(&verified_model);
    accepts_compatible_provider(custom.provider());
    accepts_compatible_provider(verified.provider());
    accepts_compatible_model(custom_model.model());
    accepts_compatible_model(verified_model.model());
    assert_eq!(
        custom.provider().profile().provider_profile().id().as_str(),
        "custom-compatible"
    );
    assert_eq!(
        verified
            .provider()
            .profile()
            .provider_profile()
            .id()
            .as_str(),
        "verified-compatible"
    );
    assert_eq!(
        custom_model.descriptor().api_mode(),
        Some(OpenAiCompatibleApiMode::Responses.as_str())
    );
    assert_eq!(
        custom_model.descriptor().instance_id(),
        &configured_instance
    );
    assert_eq!(
        verified_model.descriptor().api_mode(),
        Some(OpenAiCompatibleApiMode::ChatCompletions.as_str())
    );

    Ok(())
}

#[tokio::test]
async fn custom_http_supports_unauthenticated_generation_and_streaming()
-> Result<(), Box<dyn std::error::Error>> {
    use futures::StreamExt;
    use serde_json::json;
    use siumai::{LanguageStreamEvent, StreamTerminal};
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(
            json!({"model": "caller-owned-model", "stream": false}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "local-chat",
            "model": "caller-owned-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "local answer"},
                "finish_reason": "stop"
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": "caller-owned-model", "stream": true})))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            concat!(
                "data: {\"id\":\"local-stream\",\"model\":\"caller-owned-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"local answer\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n"
            ),
            "text/event-stream",
        ))
        .expect(1)
        .mount(&server)
        .await;

    let profile = OpenAiCompatibleProfile::custom(
        ProviderId::new("self-hosted")?,
        format!("{}/v1", server.uri()),
        ReplayDomainId::new("self-hosted")?,
        OpenAiCompatibleApiMode::ChatCompletions,
    )?;
    let ai = Siumai::builder()
        .openai_compatible()
        .profile(profile)
        .unauthenticated()
        .build()?;
    let client = ai.language("caller-owned-model")?;
    assert_eq!(client.descriptor().platform(), Some("custom-endpoint"));
    assert_eq!(
        client.generate("Hello").await?.output_text().as_deref(),
        Some("local answer")
    );

    let events = client.stream("Hello").await?.collect::<Vec<_>>().await;
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LanguageStreamEvent::Terminal(_)))
            .count(),
        1
    );
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.output_text().as_deref() == Some("local answer")
    ));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| !request.headers.contains_key("authorization"))
    );
    server.verify().await;
    Ok(())
}

#[test]
fn compatible_stage_hub_client_and_error_diagnostics_redact_credentials()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder()
        .openai_compatible()
        .profile(custom_profile()?)
        .api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.language("future-compatible-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let error = Siumai::builder()
        .openai_compatible()
        .profile(custom_profile()?)
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

    Ok(())
}
