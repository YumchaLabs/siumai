use std::sync::Arc;

use async_trait::async_trait;
use chrono::NaiveDate;
use futures_util::StreamExt;
use http::header::{AUTHORIZATION, CONTENT_TYPE, HOST, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::json;
use siumai_core::{
    ApiModeId, ApiStability, CallOptions, ContentAnnotationTarget, ContentAnnotations, Error,
    ErrorKind, LanguageModel, LanguageRequest, LanguageStreamEvent, Message, MessagePart,
    MessageRole, Model, ModelCatalog, ModelFamily, ModelId, ModelLifecycle, ModelOperation,
    ModelProfile, OfficialSource, PartialLanguageOutputPart, PlatformId, ProfileId,
    ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ReplayDomain, ReplayDomainId,
    StreamTerminal, SupportScope, ToolAnnotationTarget, ToolAnnotations, ToolSpec,
    TypedProviderAnnotation, TypedProviderOptions, UsageValue, VerificationDate,
    VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, AnthropicTool, CacheControl, CacheTtl, ContentNodeOptions, InferenceGeo,
    InferenceSpeed, McpToolsetOptions, MessagesAnnotationResolver, MessagesCodecError,
    MessagesContainer, PROTOCOL_ID, TokenTaskBudget, ToolNodeOptions, anthropic_tool_anchor_schema,
};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, EndpointConfig, OfficialOrigin,
    RequestHeaders, RequestTarget,
};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::{
    AnthropicCompatibleCredential, AnthropicCompatibleProfile, AnthropicCompatibleProvider,
    CacheControlWireStyle, MessagesCallOptions, MessagesEncodingRules, MessagesRequestProjection,
    MessagesRequestProjectionContext, MessagesServiceTierPreference, MidConversationSystemEncoding,
    NativeMessagesRequestProjection, ProjectedMessagesRequest, TemperatureEncodingRule,
};

const PROVIDER_ID: &str = "test-compatible";
const PLATFORM_ID: &str = "test-platform";
const API_VERSION: &str = "2023-06-01";

fn local_profile(server: &MockServer) -> AnthropicCompatibleProfile {
    AnthropicCompatibleProfile::local_explicit(
        ProfileId::new("test-compatible-local").unwrap(),
        ProviderId::new(PROVIDER_ID).unwrap(),
        PlatformId::new(PLATFORM_ID).unwrap(),
        format!("{}/v1", server.uri()),
        ReplayDomainId::new("compatible-test").unwrap(),
        API_VERSION,
    )
    .unwrap()
}

fn request(text: &str, max_tokens: u64) -> LanguageRequest {
    let mut request = LanguageRequest::new(vec![Message::text(MessageRole::User, text)]);
    request.generation.max_output_tokens = Some(max_tokens);
    request
}

fn response(model: &str, id: &str, text: &str) -> serde_json::Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
}

fn request_body(model: &str, text: &str, max_tokens: u64, stream: bool) -> serde_json::Value {
    json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }],
        "stream": stream
    })
}

#[test]
fn credentials_and_configured_runtime_debug_are_secret_safe() {
    let api_key = AnthropicCompatibleCredential::api_key("api-key-canary");
    let bearer = AnthropicCompatibleCredential::bearer("bearer-canary");
    assert!(!format!("{api_key:?}").contains("api-key-canary"));
    assert!(!format!("{bearer:?}").contains("bearer-canary"));

    let profile = AnthropicCompatibleProfile::public_custom(
        ProfileId::new("debug-profile").unwrap(),
        ProviderId::new(PROVIDER_ID).unwrap(),
        PlatformId::new(PLATFORM_ID).unwrap(),
        "https://compatible.example/v1",
        ReplayDomainId::new("compatible-debug-test").unwrap(),
        API_VERSION,
    )
    .unwrap();
    assert_eq!(
        profile.encoding_rules().mid_conversation_system(),
        MidConversationSystemEncoding::Unsupported
    );
    let provider = AnthropicCompatibleProvider::builder(profile, api_key)
        .build()
        .unwrap();
    let debug = format!("{provider:?}");
    assert!(!debug.contains("api-key-canary"));
    assert!(!debug.contains("compatible.example"));
}

#[test]
fn configured_instance_identity_is_shared_within_one_build_and_fresh_across_builds() {
    let profile = AnthropicCompatibleProfile::public_custom(
        ProfileId::new("instance-profile").unwrap(),
        ProviderId::new(PROVIDER_ID).unwrap(),
        PlatformId::new(PLATFORM_ID).unwrap(),
        "https://compatible.example/v1",
        ReplayDomainId::new("compatible-instance-test").unwrap(),
        API_VERSION,
    )
    .unwrap();
    let first = AnthropicCompatibleProvider::builder(
        profile.clone(),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let second = AnthropicCompatibleProvider::builder(
        profile,
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();

    let first_direct = first.language("future-model-v9").unwrap();
    let first_registered = first
        .registration()
        .language_model(ModelId::new("future-model-v9").unwrap())
        .unwrap();
    let second_direct = second.language("future-model-v9").unwrap();

    assert_eq!(
        first_direct.descriptor().instance_id(),
        first_registered.descriptor().instance_id()
    );
    assert_ne!(
        first_direct.descriptor().instance_id(),
        second_direct.descriptor().instance_id()
    );
}

#[test]
fn exact_raw_options_reach_only_the_selected_compatible_instance() {
    let profile = AnthropicCompatibleProfile::public_custom(
        ProfileId::new("exact-options-profile").unwrap(),
        ProviderId::new(PROVIDER_ID).unwrap(),
        PlatformId::new(PLATFORM_ID).unwrap(),
        "https://compatible.example/v1",
        ReplayDomainId::new("compatible-exact-options").unwrap(),
        API_VERSION,
    )
    .unwrap();
    let first = AnthropicCompatibleProvider::builder(
        profile.clone(),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let second = AnthropicCompatibleProvider::builder(
        profile,
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let first_model = first.language("future-model-v9").unwrap();
    let second_model = second.language("future-model-v9").unwrap();
    let options = CallOptions::default()
        .with_raw_provider_options_for(&first_model, json!({"future_service_tier": "priority_v2"}))
        .unwrap();

    let merged = first
        .runtime
        .merge_options_for(&first_model, &options)
        .unwrap();
    let mut body = json!({"model": "future-model-v9", "messages": []});
    merged.apply_raw_body_overlay(&mut body).unwrap();
    assert_eq!(body["future_service_tier"], "priority_v2");
    assert!(matches!(
        second.runtime.merge_options_for(&second_model, &options),
        Err(siumai_core::ProviderOptionError::ExactTargetMismatch { .. })
    ));
}

#[tokio::test]
async fn direct_and_erased_models_have_identical_api_key_wire_behavior() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-api-key"))
        .and(header("anthropic-version", API_VERSION))
        .and(header("anthropic-beta", "default-test-beta-2026-08-06"))
        .and(header("accept", "application/json"))
        .and(body_json(request_body(
            "future-model-v9",
            "hello",
            64,
            false,
        )))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "future-model-v9",
            "msg_direct",
            "ok",
        )))
        .expect(2)
        .mount(&server)
        .await;
    let provider = AnthropicCompatibleProvider::builder(
        local_profile(&server)
            .with_beta_feature("default-test-beta-2026-08-06")
            .unwrap(),
        AnthropicCompatibleCredential::api_key("test-api-key"),
    )
    .build()
    .unwrap();
    let direct = provider.language("future-model-v9").unwrap();
    let erased = provider
        .registration()
        .language_model(ModelId::new("future-model-v9").unwrap())
        .unwrap();
    assert_eq!(direct.descriptor(), erased.descriptor());

    let direct_response = direct
        .generate(request("hello", 64), CallOptions::default())
        .await
        .unwrap();
    let erased_response = erased
        .generate(request("hello", 64), CallOptions::default())
        .await
        .unwrap();
    assert_eq!(direct_response.content(), erased_response.content());
    assert!(direct_response.warnings().is_empty());
}

struct BodyVersionProjection;

impl MessagesRequestProjection for BodyVersionProjection {
    fn project(
        &self,
        context: &MessagesRequestProjectionContext<'_>,
        mut body: serde_json::Value,
    ) -> Result<ProjectedMessagesRequest, Error> {
        let object = body.as_object_mut().ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "canonical Messages request body must be an object",
            )
        })?;
        object.remove("model");
        object.insert(
            "protocol_version".to_string(),
            serde_json::Value::String(context.api_version().to_string()),
        );
        let operation = if context.is_streaming() {
            "stream"
        } else {
            "generate"
        };
        ProjectedMessagesRequest::try_new(
            format!("models/{}:{operation}", context.model()),
            body,
            RequestHeaders::new(),
        )
    }
}

#[test]
fn custom_projection_selects_target_by_model_and_stream_and_places_version_in_body() {
    let model = ModelId::new("projected-model").unwrap();
    let default_target = RequestTarget::new("messages").unwrap();
    for (stream, operation) in [(false, "generate"), (true, "stream")] {
        let context = MessagesRequestProjectionContext::new(
            &model,
            stream,
            API_VERSION,
            &default_target,
            Some("test-beta"),
        );
        let projected = BodyVersionProjection
            .project(&context, json!({"model": model.as_str(), "stream": stream}))
            .unwrap();
        assert_eq!(
            projected.target().as_str(),
            format!("models/{model}:{operation}")
        );
        assert_eq!(projected.body()["protocol_version"], API_VERSION);
        assert!(projected.body().get("model").is_none());
        assert!(projected.headers().iter().next().is_none());
    }
}

#[test]
fn native_projection_preserves_target_body_version_and_beta_headers() {
    let model = ModelId::new("native-model").unwrap();
    let default_target = RequestTarget::new("messages").unwrap();
    let body = json!({"model": model.as_str(), "stream": false});
    let context = MessagesRequestProjectionContext::new(
        &model,
        false,
        API_VERSION,
        &default_target,
        Some("beta-a,beta-b"),
    );
    let projected = NativeMessagesRequestProjection
        .project(&context, body.clone())
        .unwrap();
    assert_eq!(projected.target().as_str(), "messages");
    assert_eq!(projected.body(), &body);
    assert_eq!(
        projected
            .headers()
            .get(&HeaderName::from_static("anthropic-version"))
            .unwrap(),
        API_VERSION
    );
    assert_eq!(
        projected
            .headers()
            .get(&HeaderName::from_static("anthropic-beta"))
            .unwrap(),
        "beta-a,beta-b"
    );
}

#[test]
fn projected_requests_reject_protected_headers_and_unsafe_targets() {
    for protected in [AUTHORIZATION, HOST] {
        assert!(
            RequestHeaders::new()
                .try_insert(protected, HeaderValue::from_static("protected"))
                .is_err()
        );
    }

    let content_type = RequestHeaders::new()
        .try_insert(CONTENT_TYPE, HeaderValue::from_static("text/plain"))
        .unwrap();
    let error = ProjectedMessagesRequest::new(
        RequestTarget::new("messages").unwrap(),
        json!({}),
        content_type,
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);

    for target in [
        "https://attacker.invalid/messages",
        "models/../secrets",
        "models/%252e%252e/secrets",
    ] {
        let error = ProjectedMessagesRequest::try_new(target, json!({}), RequestHeaders::new())
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput, "target: {target}");
    }
}

#[tokio::test]
async fn configured_projection_runs_after_canonical_encoding() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/models/projected-model:generate"))
        .and(header("accept", "application/json"))
        .and(header("content-type", "application/json"))
        .and(body_json(json!({
            "protocol_version": API_VERSION,
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }],
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "projected-model",
            "msg_projected",
            "ok",
        )))
        .expect(1)
        .mount(&server)
        .await;
    let profile = local_profile(&server).with_request_projection(Arc::new(BodyVersionProjection));
    let provider = AnthropicCompatibleProvider::builder(
        profile,
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();

    provider
        .language("projected-model")
        .unwrap()
        .generate(request("hello", 64), CallOptions::default())
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].headers.get("anthropic-version").is_none());
}

#[tokio::test]
async fn bearer_and_custom_auth_are_applied_inside_transport() {
    let bearer_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("authorization", "Bearer test-bearer"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "model",
            "msg_bearer",
            "ok",
        )))
        .mount(&bearer_server)
        .await;
    let bearer = AnthropicCompatibleProvider::builder(
        local_profile(&bearer_server),
        AnthropicCompatibleCredential::bearer("test-bearer"),
    )
    .build()
    .unwrap();
    bearer
        .language("model")
        .unwrap()
        .generate(request("hello", 32), CallOptions::default())
        .await
        .unwrap();

    let custom_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-compatible-auth", "custom-canary"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "model",
            "msg_custom",
            "ok",
        )))
        .mount(&custom_server)
        .await;
    let custom = AnthropicCompatibleProvider::builder_with_auth(
        local_profile(&custom_server),
        Arc::new(CustomAuth),
    )
    .build()
    .unwrap();
    custom
        .language("model")
        .unwrap()
        .generate(request("hello", 32), CallOptions::default())
        .await
        .unwrap();
}

struct CustomAuth;

#[async_trait]
impl AuthApplier for CustomAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        CredentialPatch::new()
            .try_insert(
                HeaderName::from_static("x-compatible-auth"),
                HeaderValue::from_static("custom-canary"),
            )
            .map_err(|source| {
                Error::new(ErrorKind::Configuration, "custom auth is invalid").with_source(source)
            })
    }
}

#[derive(Serialize)]
struct TestTypedOptions {
    metadata: serde_json::Value,
    thinking: serde_json::Value,
    custom_level: &'static str,
}

#[derive(Serialize)]
struct TestServiceTierOptions {
    service_tier: Option<MessagesServiceTierPreference>,
}

#[derive(Serialize)]
struct TestCurrentRequestOptions {
    cache_control: CacheControl,
    speed: InferenceSpeed,
    inference_geo: InferenceGeo,
    task_budget: TokenTaskBudget,
    container: MessagesContainer,
    context_management: serde_json::Value,
    mcp_servers: serde_json::Value,
}

impl TypedProviderOptions for TestServiceTierOptions {
    const NAMESPACE: &'static str = PROVIDER_ID;
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

impl TypedProviderOptions for TestCurrentRequestOptions {
    const NAMESPACE: &'static str = PROVIDER_ID;
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

impl TypedProviderOptions for TestTypedOptions {
    const NAMESPACE: &'static str = PROVIDER_ID;
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

#[tokio::test]
async fn typed_and_checked_raw_patches_merge_into_messages_request_options() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "thinking-model",
            "max_tokens": 4096,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "reason"}]
            }],
            "stream": false,
            "metadata": {"user_id": "typed-user"},
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "custom_level": "raw"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "thinking-model",
            "msg_options",
            "done",
        )))
        .mount(&server)
        .await;
    let provider = AnthropicCompatibleProvider::builder(
        local_profile(&server),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let model = provider.language("thinking-model").unwrap();
    let typed = TestTypedOptions {
        metadata: json!({"user_id": "typed-user"}),
        thinking: json!({"type": "enabled", "budget_tokens": 2048}),
        custom_level: "typed",
    };
    let options = CallOptions::default()
        .with_provider_options_for(&model, &typed)
        .unwrap()
        .with_raw_provider_options_for(&model, json!({"custom_level": "raw"}))
        .unwrap();
    model
        .generate(request("reason", 4096), options)
        .await
        .unwrap();
}

#[tokio::test]
async fn typed_service_tier_precedence_and_raw_future_values_reach_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "tier-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "tier"}]
            }],
            "stream": false,
            "service_tier": "standard_only"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "tier-model",
            "msg_tier",
            "ok",
        )))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "tier-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "tier"}]
            }],
            "stream": false,
            "service_tier": "priority_v2"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "tier-model",
            "msg_tier_raw",
            "ok",
        )))
        .expect(1)
        .mount(&server)
        .await;
    let provider = AnthropicCompatibleProvider::builder(
        local_profile(&server),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .with_default_options(
        MessagesCallOptions::new().with_service_tier(MessagesServiceTierPreference::Auto),
    )
    .build()
    .unwrap();
    let model = provider.language("tier-model").unwrap();
    let typed = TestServiceTierOptions {
        service_tier: Some(MessagesServiceTierPreference::StandardOnly),
    };
    let typed_options = CallOptions::default()
        .with_provider_options_for(&model, &typed)
        .unwrap();
    model
        .generate(request("tier", 64), typed_options)
        .await
        .unwrap();

    let raw_options = CallOptions::default()
        .with_raw_provider_options_for(&model, json!({"service_tier": "priority_v2"}))
        .unwrap();
    model
        .generate(request("tier", 64), raw_options)
        .await
        .unwrap();
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn current_typed_request_controls_survive_the_compatible_merge() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "current-options-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "current options"}]
            }],
            "stream": false,
            "output_config": {
                "task_budget": {"type": "tokens", "total": 20000}
            },
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "speed": "fast",
            "inference_geo": "us",
            "container": "container-1",
            "context_management": {
                "edits": [{
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 100000}
                }]
            },
            "mcp_servers": [{
                "type": "url",
                "name": "docs",
                "url": "https://mcp.example.test",
                "authorization_token": "sentinel-secret"
            }],
            "tools": [{
                "type": "mcp_toolset",
                "mcp_server_name": "docs"
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "current-options-model",
            "msg_current_options",
            "ok",
        )))
        .expect(1)
        .mount(&server)
        .await;
    let provider = AnthropicCompatibleProvider::builder(
        local_profile(&server).with_annotation_resolver(Arc::new(TestAnnotationResolver)),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let model = provider.language("current-options-model").unwrap();
    let typed = TestCurrentRequestOptions {
        cache_control: CacheControl::new(CacheTtl::FiveMinutes),
        speed: InferenceSpeed::Fast,
        inference_geo: InferenceGeo::Us,
        task_budget: TokenTaskBudget::new(20_000).unwrap(),
        container: MessagesContainer::existing("container-1").unwrap(),
        context_management: json!({
            "edits": [{
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 100000}
            }]
        }),
        mcp_servers: json!([{
            "type": "url",
            "name": "docs",
            "url": "https://mcp.example.test",
            "authorization_token": "sentinel-secret"
        }]),
    };
    let options = CallOptions::default()
        .with_provider_options_for(&model, &typed)
        .unwrap();
    assert!(!format!("{options:?}").contains("sentinel-secret"));

    let mut current_request = request("current options", 64);
    current_request.tools.push(
        ToolSpec::new("docs", None, anthropic_tool_anchor_schema())
            .unwrap()
            .with_provider_annotation(&TestMcpToolsetAnnotation { enabled: true })
            .unwrap(),
    );
    model.generate(current_request, options).await.unwrap();
}

#[tokio::test]
async fn protected_version_endpoint_and_auth_fields_fail_before_network() {
    let server = MockServer::start().await;
    let provider = AnthropicCompatibleProvider::builder(
        local_profile(&server),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let model = provider.language("model").unwrap();
    for field in ["anthropicVersion", "requestEndpoint", "credentialToken"] {
        let mut value = serde_json::Map::new();
        value.insert(field.to_string(), json!("canary-secret"));
        let options = CallOptions::default()
            .with_raw_provider_options_for(&model, serde_json::Value::Object(value))
            .unwrap();
        let error = model
            .generate(request("hello", 32), options)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(!format!("{error:?}").contains("canary-secret"));
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestCacheAnnotation {
    enabled: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestMcpToolsetAnnotation {
    enabled: bool,
}

impl TypedProviderAnnotation for TestCacheAnnotation {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = PROVIDER_ID;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

impl TypedProviderAnnotation for TestMcpToolsetAnnotation {
    type Target = ToolAnnotationTarget;

    const NAMESPACE: &'static str = PROVIDER_ID;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

struct TestAnnotationResolver;

impl MessagesAnnotationResolver for TestAnnotationResolver {
    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<ContentNodeOptions, MessagesCodecError> {
        let annotation = annotations
            .decode::<TestCacheAnnotation>()
            .map_err(|source| MessagesCodecError::InvalidAnnotation {
                node: "content",
                source,
            })?;
        Ok(if annotation.is_some_and(|annotation| annotation.enabled) {
            ContentNodeOptions::default()
                .with_cache_control(CacheControl::new(CacheTtl::FiveMinutes))
        } else {
            ContentNodeOptions::default()
        })
    }

    fn resolve_tool(
        &self,
        annotations: &ToolAnnotations,
    ) -> Result<ToolNodeOptions, MessagesCodecError> {
        let annotation = annotations
            .decode::<TestMcpToolsetAnnotation>()
            .map_err(|source| MessagesCodecError::InvalidAnnotation {
                node: "tool",
                source,
            })?;
        Ok(if annotation.is_some_and(|annotation| annotation.enabled) {
            ToolNodeOptions::default().with_anthropic_tool(AnthropicTool::mcp_toolset(
                McpToolsetOptions::new("docs").expect("MCP toolset"),
            ))
        } else {
            ToolNodeOptions::default()
        })
    }
}

#[tokio::test]
async fn profile_owned_annotation_resolver_projects_node_local_cache_control() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "cache-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "cache me",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"}
                }]
            }],
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "cache-model",
            "msg_cache",
            "ok",
        )))
        .mount(&server)
        .await;
    let profile = local_profile(&server).with_annotation_resolver(Arc::new(TestAnnotationResolver));
    let provider = AnthropicCompatibleProvider::builder(
        profile,
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let part = MessagePart::text("cache me")
        .with_provider_annotation(&TestCacheAnnotation { enabled: true })
        .unwrap();
    let mut annotated = LanguageRequest::new(vec![Message::new(MessageRole::User, [part])]);
    annotated.generation.max_output_tokens = Some(64);
    provider
        .language("cache-model")
        .unwrap()
        .generate(annotated, CallOptions::default())
        .await
        .unwrap();
}

#[tokio::test]
async fn profile_owned_encoding_rules_apply_compatible_temperature_and_cache_wire_forms() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "dialect-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "cache me",
                    "cache_control": {"type": "ephemeral"}
                }]
            }],
            "stream": false,
            "temperature": 1.5
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "dialect-model",
            "msg_dialect",
            "ok",
        )))
        .mount(&server)
        .await;
    let rules = MessagesEncodingRules::native()
        .with_temperature(TemperatureEncodingRule::new(2.0).unwrap())
        .with_cache_control(CacheControlWireStyle::FiveMinutesImplicit);
    let profile = local_profile(&server)
        .with_annotation_resolver(Arc::new(TestAnnotationResolver))
        .with_encoding_rules(rules);
    let provider = AnthropicCompatibleProvider::builder(
        profile,
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let part = MessagePart::text("cache me")
        .with_provider_annotation(&TestCacheAnnotation { enabled: true })
        .unwrap();
    let mut annotated = LanguageRequest::new(vec![Message::new(MessageRole::User, [part])]);
    annotated.generation.max_output_tokens = Some(64);
    annotated.generation.temperature = Some(1.5);
    provider
        .language("dialect-model")
        .unwrap()
        .generate(annotated, CallOptions::default())
        .await
        .unwrap();
}

#[tokio::test]
async fn streaming_uses_canonical_decoder_and_emits_one_terminal() {
    let server = MockServer::start().await;
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "stream-model",
                "usage": {"input_tokens": 2}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 1}
        }),
        json!({"type": "message_stop"}),
    ];
    let sse = frames
        .into_iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect::<String>();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("accept", "text/event-stream"))
        .and(body_json(request_body("stream-model", "hello", 64, true)))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;
    let provider = AnthropicCompatibleProvider::builder(
        local_profile(&server),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let events = provider
        .language("stream-model")
        .unwrap()
        .stream(request("hello", 64), CallOptions::default())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::TextDelta { delta, .. } if delta == "hello"
    )));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LanguageStreamEvent::Terminal(_)))
            .count(),
        1
    );
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(
            StreamTerminal::Completed { .. }
        ))
    ));
}

#[tokio::test]
async fn streaming_rejects_frames_after_terminal_in_one_sse_batch() {
    let server = MockServer::start().await;
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "stream-model",
                "usage": {"input_tokens": 2}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 1}
        }),
        json!({"type": "message_stop"}),
        json!({"type": "message_stop"}),
    ];
    let sse = frames
        .into_iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect::<String>();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("accept", "text/event-stream"))
        .and(body_json(request_body("stream-model", "hello", 64, true)))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let provider = AnthropicCompatibleProvider::builder(
        local_profile(&server),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let events = provider
        .language("stream-model")
        .unwrap()
        .stream(request("hello", 64), CallOptions::default())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    let terminals = events
        .iter()
        .filter_map(LanguageStreamEvent::terminal)
        .collect::<Vec<_>>();
    assert_eq!(terminals.len(), 1);
    match terminals[0] {
        StreamTerminal::Failed { error, .. } => assert_eq!(error.kind(), ErrorKind::Protocol),
        other => panic!("expected a failed terminal, got {other:?}"),
    }
}

#[tokio::test]
async fn in_band_stream_error_preserves_bounded_partial_output_and_safe_diagnostics() {
    let server = MockServer::start().await;
    let secret = "private-in-band-error-secret";
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_partial_error",
                "type": "message",
                "role": "assistant",
                "model": "stream-model",
                "usage": {"input_tokens": 7}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""}
        }),
        json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "thinking_delta", "thinking": "chain"}
        }),
        json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "signature_delta", "signature": "signature"}
        }),
        json!({"type": "content_block_stop", "index": 1}),
        json!({
            "type": "message_delta",
            "delta": {},
            "usage": {"output_tokens": 3}
        }),
        json!({
            "type": "error",
            "error": {"type": "rate_limit_error", "message": secret},
            "request_id": "request-safe"
        }),
    ];
    let sse = frames
        .into_iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect::<String>();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("accept", "text/event-stream"))
        .and(body_json(request_body("stream-model", "hello", 64, true)))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let provider = AnthropicCompatibleProvider::builder(
        local_profile(&server),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let events = provider
        .language("stream-model")
        .unwrap()
        .stream(request("hello", 64), CallOptions::default())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::TextDelta { delta, .. } if delta == "hello"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::ReasoningDelta { delta, .. } if delta == "chain"
    )));
    let terminals = events
        .iter()
        .filter_map(LanguageStreamEvent::terminal)
        .collect::<Vec<_>>();
    assert_eq!(terminals.len(), 1);
    let StreamTerminal::Failed { error, partial } = terminals[0] else {
        panic!("expected a failed terminal, got {:?}", terminals[0]);
    };
    assert_eq!(error.kind(), ErrorKind::RateLimited);
    let diagnostics = error.diagnostics().expect("safe diagnostics");
    assert_eq!(diagnostics.status(), Some(200));
    assert_eq!(diagnostics.request_id(), Some("request-safe"));
    assert_eq!(diagnostics.provider_type(), Some("rate_limit_error"));
    let partial = partial.as_ref().expect("bounded partial output");
    assert_eq!(
        partial.content(),
        [
            PartialLanguageOutputPart::Text {
                text: "hello".to_string(),
            },
            PartialLanguageOutputPart::Reasoning {
                text: "chain".to_string(),
            },
        ]
    );
    assert_eq!(partial.usage().input_tokens, UsageValue::Known(7));
    assert_eq!(partial.usage().output_tokens, UsageValue::Known(3));
    assert_eq!(partial.usage().total_tokens, UsageValue::Known(10));
    let public = format!("{error:?} {error} {partial:?}");
    assert!(!public.contains(secret));
    assert!(error.sensitive_response().is_some());
}

#[tokio::test]
async fn post_is_not_replayed_and_http_diagnostics_are_sanitized() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("request-id", "request-safe")
                .insert_header("set-cookie", "secret-cookie-canary")
                .set_body_json(json!({
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "private-body-canary"
                    }
                })),
        )
        .expect(1)
        .mount(&server)
        .await;
    let provider = AnthropicCompatibleProvider::builder(
        local_profile(&server),
        AnthropicCompatibleCredential::unauthenticated(),
    )
    .build()
    .unwrap();
    let error = provider
        .language("model")
        .unwrap()
        .generate(request("hello", 32), CallOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RateLimited);
    let diagnostics = error.diagnostics().unwrap();
    assert_eq!(diagnostics.status(), Some(429));
    assert_eq!(diagnostics.request_id(), Some("request-safe"));
    assert_eq!(diagnostics.provider_type(), Some("rate_limit_error"));
    let public = format!("{error:?} {error}");
    assert!(!public.contains("private-body-canary"));
    assert!(!public.contains("secret-cookie-canary"));
    assert!(error.sensitive_response().is_some());
}

#[test]
fn verified_profile_carries_exact_evidence_and_open_model_construction() {
    let scope = SupportScope::new(
        ProviderId::new(PROVIDER_ID).unwrap(),
        PlatformId::new(PLATFORM_ID).unwrap(),
        ModelFamily::Language,
        ProtocolId::new(PROTOCOL_ID).unwrap(),
        ApiModeId::new(API_MODE_ID).unwrap(),
    );
    let evidence = VerificationEvidence::new(
        OfficialSource::new("https://docs.example.com/anthropic-compatible").unwrap(),
        VerificationDate::new(NaiveDate::from_ymd_opt(2026, 8, 6).unwrap()),
        ProtocolContractId::new("anthropic-messages-2023-06-01").unwrap(),
    );
    let catalog = ModelCatalog::new([ModelProfile::new(
        ModelId::new("known-model").unwrap(),
        scope.clone(),
        [ModelOperation::Generate, ModelOperation::Stream],
        ModelLifecycle::Retired { replacement: None },
        evidence.clone(),
    )
    .unwrap()])
    .unwrap();
    let provider_profile = ProviderProfile::verified(
        ProfileId::new("verified-compatible").unwrap(),
        vec![VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Compatible,
            ApiStability::Stable,
            evidence,
        )],
        catalog,
    )
    .unwrap();
    let endpoint = EndpointConfig::official(
        "https://api.compatible.example/v1",
        OfficialOrigin::new("https://api.compatible.example").unwrap(),
    )
    .unwrap();
    let profile =
        AnthropicCompatibleProfile::verified(provider_profile, endpoint, API_VERSION).unwrap();
    assert!(matches!(
        profile.clone().with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("caller-relay").unwrap(),
        )),
        Err(crate::AnthropicCompatibleConfigError::ReplayAudienceMismatch)
    ));
    let provider = AnthropicCompatibleProvider::builder(
        profile,
        AnthropicCompatibleCredential::api_key("test-key"),
    )
    .build()
    .unwrap();
    let registration = provider.registration();
    assert!(matches!(
        provider
            .profile()
            .provider_profile()
            .catalog()
            .unwrap()
            .iter()
            .find(|entry| entry.model().as_str() == "known-model")
            .unwrap()
            .lifecycle(),
        ModelLifecycle::Retired { .. }
    ));
    assert!(
        registration
            .language_model(ModelId::new("known-model").unwrap())
            .is_ok()
    );
    assert!(
        registration
            .language_model(ModelId::new("future-model").unwrap())
            .is_ok()
    );
    assert_eq!(
        provider
            .profile()
            .provider_profile()
            .verified_claims()
            .unwrap()[0]
            .evidence()
            .verified_at()
            .value(),
        NaiveDate::from_ymd_opt(2026, 8, 6).unwrap()
    );
}

#[test]
fn generic_profile_rejects_official_replay_audience() {
    let profile = AnthropicCompatibleProfile::public_custom(
        ProfileId::new("custom-compatible").unwrap(),
        ProviderId::new(PROVIDER_ID).unwrap(),
        PlatformId::new(PLATFORM_ID).unwrap(),
        "https://relay.example/v1",
        ReplayDomainId::new("caller-relay").unwrap(),
        API_VERSION,
    )
    .unwrap();

    assert!(matches!(
        profile.with_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("official").unwrap(),
        )),
        Err(crate::AnthropicCompatibleConfigError::ReplayAudienceMismatch)
    ));
}
