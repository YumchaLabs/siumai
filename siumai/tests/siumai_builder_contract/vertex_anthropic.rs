use siumai::providers::google_vertex_anthropic::{
    GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE, GoogleVertexAnthropicLanguageModel,
    GoogleVertexAnthropicProvider, GoogleVertexAnthropicProviderBuilder, GoogleVertexCredential,
};
use siumai::{Model, ModelFamily, ReplayDomain, ReplayDomainId, Siumai};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "vertex-anthropic-facade-credential-canary";

fn replay_domain(label: &str) -> ReplayDomain {
    ReplayDomain::official(
        ReplayDomainId::new(GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE)
            .expect("the official replay audience is valid"),
    )
    .with_caller_scope(
        ReplayDomainId::new(label).expect("the test caller scope is a valid identifier"),
    )
}

#[test]
fn ordered_inputs_and_access_token_sugar_build_language_hubs()
-> Result<(), Box<dyn std::error::Error>> {
    let access_token_hub = Siumai::builder()
        .vertex_anthropic()
        .project("test-project")
        .location("global")
        .access_token("test-access-token")
        .configure_provider(|builder: GoogleVertexAnthropicProviderBuilder| {
            builder.with_replay_domain(replay_domain("vertex-access-token-test"))
        })
        .build()?;
    let credential_hub = Siumai::builder()
        .vertex_anthropic()
        .project("test-project")
        .location("global")
        .credential(GoogleVertexCredential::access_token("test-access-token"))
        .configure_provider(|builder| {
            builder.with_replay_domain(replay_domain("vertex-credential-test"))
        })
        .build()?;

    let client = access_token_hub.language("future-vertex-anthropic-model")?;
    let _: &GoogleVertexAnthropicProvider = access_token_hub.provider();
    let _: &GoogleVertexAnthropicLanguageModel = client.model();
    assert!(std::ptr::eq(access_token_hub.provider(), client.provider()));
    assert_eq!(client.family(), ModelFamily::Language);
    assert_eq!(client.descriptor().api_mode(), Some("messages"));
    assert_eq!(
        client.descriptor().model().as_str(),
        "future-vertex-anthropic-model"
    );
    assert_eq!(
        credential_hub
            .language("future-vertex-credential-model")?
            .descriptor()
            .model()
            .as_str(),
        "future-vertex-credential-model"
    );

    Ok(())
}

#[test]
fn stages_hubs_clients_and_error_chains_redact_vertex_credentials()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder()
        .vertex_anthropic()
        .project("private-project-canary")
        .location("global")
        .access_token(CREDENTIAL_CANARY)
        .configure_provider(|builder| {
            builder.with_replay_domain(replay_domain("vertex-canary-test"))
        });
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.language("future-vertex-canary-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let error = Siumai::builder()
        .vertex_anthropic()
        .project("private-project-canary")
        .location("global")
        .access_token(format!("{CREDENTIAL_CANARY}\n"))
        .configure_provider(|builder| {
            builder.with_replay_domain(replay_domain("vertex-error-test"))
        })
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

    Ok(())
}
