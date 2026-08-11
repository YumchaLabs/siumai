use futures_util::StreamExt;
use serde_json::json;
use siumai_core::{
    ApiStability, CallOptions, ErrorKind, LanguageModel, LanguageRequest, LanguageStreamEvent,
    Message, MessagePart, MessageRole, Model, ModelId, ModelLifecycle, Provider, ReplayDomain,
    ReplayDomainId, StreamTerminal, ToolSpec,
};
use siumai_protocol_anthropic::messages::{
    MessagesCodecError, MessagesRequestOptions, encode_request_with_resolver,
};
use siumai_transport::{EndpointConfig, OfficialOrigin};
use wiremock::matchers::{body_json, body_string_contains, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::resources::{
    AnthropicBatchItem, AnthropicBatchRequest, AnthropicFileUpload, AnthropicSkillFile,
    AnthropicSkillUpload,
};
use crate::{
    AdvisorToolOptions, AnthropicAnnotationResolver, AnthropicCacheTtl, AnthropicContentOptions,
    AnthropicCredential, AnthropicMessageCache, AnthropicMessageFile, AnthropicMessagesOptions,
    AnthropicProvider, AnthropicThinking, AnthropicTokenCountOptions, AnthropicTool,
    AnthropicToolOptions, CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5, CLAUDE_OPUS_4_1_20250805,
    CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7, CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_6,
    CLAUDE_SONNET_5, MessagesMetadata, OutputEffort, ServerFallbacks, ThinkingDisplay,
    TokenTaskBudget, current_models,
};

const API_VERSION: &str = "2023-06-01";

fn request(text: &str, max_tokens: u64) -> LanguageRequest {
    let mut request = LanguageRequest::new(vec![Message::text(MessageRole::User, text)]);
    request.generation.max_output_tokens = Some(max_tokens);
    request
}

#[test]
fn retired_and_future_model_ids_remain_callable() {
    let provider = AnthropicProvider::builder(AnthropicCredential::api_key("offline-test-key"))
        .build()
        .expect("provider");
    assert!(provider.language(CLAUDE_OPUS_4_1_20250805).is_ok());
    assert!(provider.language("claude-future-2030").is_ok());
}

#[tokio::test]
async fn explicit_options_are_encoded_without_model_name_gating() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": CLAUDE_OPUS_4_7,
            "max_tokens": 4_096,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "caller intent"}]
            }],
            "stream": false,
            "temperature": 0.4,
            "top_p": 0.5,
            "top_k": 32,
            "thinking": {"type": "enabled", "budget_tokens": 2_048},
            "output_config": {"effort": "max"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            CLAUDE_OPUS_4_7,
            "msg_caller_intent",
            "ok",
        )))
        .expect(1)
        .mount(&server)
        .await;
    let provider = local_provider(&server, AnthropicCredential::unauthenticated());
    let mut explicit = request("caller intent", 4_096);
    explicit.generation.temperature = Some(0.4);
    explicit.generation.top_p = Some(0.5);
    let model = provider.language(CLAUDE_OPUS_4_7).expect("model");
    let options = AnthropicMessagesOptions::new()
        .with_enabled_thinking(2_048)
        .with_output_effort(OutputEffort::Max)
        .with_top_k(32);
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &options)
        .expect("call options");

    let response = model
        .generate(explicit, call_options)
        .await
        .expect("explicit caller options must reach the wire");
    assert_eq!(response.id(), Some("msg_caller_intent"));
}

#[tokio::test]
async fn future_raw_provider_values_reach_the_request_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "future-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "future values"}]
            }],
            "stream": false,
            "service_tier": "priority_v2",
            "output_config": {"effort": "ultra"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "future-model",
            "msg_future_options",
            "ok",
        )))
        .expect(1)
        .mount(&server)
        .await;
    let provider = local_provider(&server, AnthropicCredential::unauthenticated());
    let model = provider.language("future-model").expect("model");
    let call_options = CallOptions::default()
        .with_raw_provider_options_for(
            &model,
            json!({
                "service_tier": "priority_v2",
                "output_config": {"effort": "ultra"}
            }),
        )
        .expect("bounded raw options");

    model
        .generate(request("future values", 64), call_options)
        .await
        .expect("future raw values must reach the wire");
}

#[tokio::test]
async fn fallback_options_encode_and_add_the_required_beta_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("anthropic-beta", "server-side-fallback-2026-07-01"))
        .and(body_json(json!({
            "model": "future-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "fallback"}]
            }],
            "stream": false,
            "fallbacks": "default"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "future-model",
            "msg_fallback",
            "ok",
        )))
        .mount(&server)
        .await;

    let provider = local_provider(&server, AnthropicCredential::unauthenticated());
    let model = provider.language("future-model").expect("model");
    let options = AnthropicMessagesOptions::new().with_fallbacks(ServerFallbacks::Default);
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &options)
        .expect("call options");
    model
        .generate(request("fallback", 64), call_options)
        .await
        .expect("fallback request");
}

#[tokio::test]
async fn future_model_accepts_low_task_budget_and_deduplicates_the_beta_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("anthropic-beta", "task-budgets-2026-03-13"))
        .and(body_json(json!({
            "model": "future-budget-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "budget"}]
            }],
            "stream": false,
            "output_config": {
                "task_budget": {
                    "type": "tokens",
                    "total": 1_024,
                    "remaining": 512
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "future-budget-model",
            "msg_budget",
            "ok",
        )))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::builder(AnthropicCredential::unauthenticated())
        .with_endpoint(
            EndpointConfig::local_explicit(format!("{}/v1/", server.uri()))
                .expect("local endpoint"),
        )
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("anthropic-budget-test-relay").expect("replay domain"),
        ))
        .with_beta_feature("task-budgets-2026-03-13")
        .build()
        .expect("provider");
    let model = provider
        .language("future-budget-model")
        .expect("future model");
    let budget = TokenTaskBudget::new(1_024)
        .expect("positive task budget")
        .with_remaining(512)
        .expect("remaining budget");
    let options = AnthropicMessagesOptions::new().with_task_budget(budget);
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &options)
        .expect("call options");

    model
        .generate(request("budget", 64), call_options)
        .await
        .expect("future model task budget");
}

#[test]
fn task_budget_requires_positive_total_and_bounded_remaining_tokens() {
    assert!(matches!(
        TokenTaskBudget::new(0),
        Err(MessagesCodecError::InvalidOption {
            field: "output_config.task_budget.total",
            ..
        })
    ));
    assert!(matches!(
        TokenTaskBudget::new(1_024)
            .expect("positive task budget")
            .with_remaining(1_025),
        Err(MessagesCodecError::InvalidOption {
            field: "output_config.task_budget.remaining",
            ..
        })
    ));

    let exhausted = TokenTaskBudget::new(1)
        .expect("positive task budget")
        .with_remaining(0)
        .expect("exhausted budget");
    assert_eq!(exhausted.total(), 1);
    assert_eq!(exhausted.remaining(), Some(0));
}

#[tokio::test]
async fn mid_conversation_system_message_emits_no_retired_beta_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "future-system-model",
            "msg_system",
            "ok",
        )))
        .expect(1)
        .mount(&server)
        .await;

    let provider = local_provider(&server, AnthropicCredential::unauthenticated());
    let model = provider
        .language("future-system-model")
        .expect("future model");
    let mut language_request = request("start", 64);
    language_request.messages.push(Message::text(
        MessageRole::System,
        "updated system guidance",
    ));

    model
        .generate(language_request, CallOptions::default())
        .await
        .expect("mid-conversation system message");

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    assert!(requests[0].headers.get("anthropic-beta").is_none());
}

#[tokio::test]
async fn cache_prewarm_reuses_messages_with_zero_output_and_requires_cache_configuration() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "future-model",
            "max_tokens": 0,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "warm this prompt",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"}
                }]
            }],
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_prewarm",
            "type": "message",
            "role": "assistant",
            "model": "future-model",
            "content": [],
            "stop_reason": "max_tokens",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 8,
                "output_tokens": 0,
                "cache_creation_input_tokens": 8
            }
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "future-model",
            "max_tokens": 0,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "warm automatically"}]
            }],
            "stream": false,
            "cache_control": {"type": "ephemeral", "ttl": "5m"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_automatic_prewarm",
            "type": "message",
            "role": "assistant",
            "model": "future-model",
            "content": [],
            "stop_reason": "max_tokens",
            "stop_sequence": null,
            "usage": {"input_tokens": 8, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = local_provider(&server, AnthropicCredential::unauthenticated());
    let model = provider.language("future-model").expect("model");
    let missing_marker = model
        .prewarm_cache(request("not marked", 64), AnthropicMessagesOptions::new())
        .await
        .expect_err("prewarming without cache configuration must fail");
    assert_eq!(missing_marker.kind(), ErrorKind::InvalidInput);

    let marked = Message::user("warm this prompt")
        .with_provider_annotation(&AnthropicMessageCache::five_minutes())
        .expect("cache marker");
    let response = model
        .prewarm_cache(
            LanguageRequest::new(vec![marked]),
            AnthropicMessagesOptions::new(),
        )
        .await
        .expect("prewarm");
    assert_eq!(response.id(), Some("msg_prewarm"));
    assert_eq!(response.usage().input_tokens.value(), Some(8));
    assert_eq!(response.usage().output_tokens.value(), Some(0));

    let automatic = model
        .prewarm_cache(
            LanguageRequest::new(vec![Message::user("warm automatically")]),
            AnthropicMessagesOptions::new().with_automatic_cache(AnthropicCacheTtl::FiveMinutes),
        )
        .await
        .expect("automatic cache prewarm");
    assert_eq!(automatic.id(), Some("msg_automatic_prewarm"));
}

#[tokio::test]
async fn anthropic_tool_annotations_project_wire_and_beta_contracts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("anthropic-beta", "advisor-tool-2026-03-01"))
        .and(body_json(json!({
            "model": "future-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "use advisor"}]
            }],
            "stream": false,
            "tools": [{
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "claude-sonnet-5"
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "future-model",
            "msg_advisor",
            "ok",
        )))
        .mount(&server)
        .await;

    let anthropic_tool = AnthropicTool::advisor_20260301(
        AdvisorToolOptions::new("claude-sonnet-5").expect("advisor options"),
    );
    let tool = AnthropicToolOptions::for_tool(anthropic_tool)
        .into_tool_spec()
        .expect("Anthropic tool anchor");
    let mut language_request = request("use advisor", 64);
    language_request.tools.push(tool);

    let provider = local_provider(&server, AnthropicCredential::unauthenticated());
    provider
        .language("future-model")
        .expect("model")
        .generate(language_request, CallOptions::default())
        .await
        .expect("Anthropic tool request");
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

fn local_provider(server: &MockServer, credential: AnthropicCredential) -> AnthropicProvider {
    AnthropicProvider::builder(credential)
        .with_endpoint(
            EndpointConfig::local_explicit(format!("{}/v1/", server.uri()))
                .expect("local endpoint"),
        )
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("anthropic-test-relay").expect("replay domain"),
        ))
        .build()
        .expect("provider")
}

fn local_provider_with_caller_scope(
    server: &MockServer,
    credential: AnthropicCredential,
    caller_scope: &str,
) -> AnthropicProvider {
    AnthropicProvider::builder(credential)
        .with_endpoint(
            EndpointConfig::local_explicit(format!("{}/v1/", server.uri()))
                .expect("local endpoint"),
        )
        .with_replay_domain(
            ReplayDomain::custom(
                ReplayDomainId::new("anthropic-test-relay").expect("replay domain"),
            )
            .with_caller_scope(ReplayDomainId::new(caller_scope).expect("caller scope")),
        )
        .build()
        .expect("provider")
}

#[test]
fn official_provider_is_network_free_and_catalog_is_introspection_only() {
    let provider = AnthropicProvider::builder(AnthropicCredential::api_key("offline-test-key"))
        .build()
        .expect("network-free provider construction");

    assert_eq!(provider.provider_id().as_str(), "anthropic");
    let profile_claims = provider
        .profile()
        .provider_profile()
        .verified_claims()
        .expect("official Anthropic profile");
    assert_eq!(profile_claims[0].scope().api_mode().as_str(), "messages");
    assert_eq!(
        current_models(),
        [
            CLAUDE_OPUS_5,
            CLAUDE_SONNET_5,
            CLAUDE_FABLE_5,
            CLAUDE_HAIKU_4_5,
        ]
    );
    let catalog = provider
        .profile()
        .provider_profile()
        .catalog()
        .expect("verified catalog");
    for model in current_models() {
        let id = ModelId::new(model).expect("model id");
        assert!(catalog.iter().any(|entry| entry.model() == &id));
        assert_eq!(
            provider.language(model).expect("known model").model_id(),
            &id
        );
    }
    for model in [
        CLAUDE_OPUS_4_6,
        CLAUDE_SONNET_4_6,
        CLAUDE_OPUS_4_7,
        CLAUDE_OPUS_4_8,
        CLAUDE_OPUS_5,
        CLAUDE_SONNET_5,
        CLAUDE_FABLE_5,
    ] {
        let id = ModelId::new(model).expect("model id");
        let profile = catalog
            .iter()
            .find(|entry| entry.model() == &id)
            .expect("pinned model profile");
        assert_eq!(profile.lifecycle(), &ModelLifecycle::Active);
    }

    let mythos_preview = ModelId::new(crate::CLAUDE_MYTHOS_PREVIEW).expect("model id");
    let mythos_preview = catalog
        .iter()
        .find(|entry| entry.model() == &mythos_preview)
        .expect("Mythos Preview profile");
    assert_eq!(
        mythos_preview.lifecycle(),
        &ModelLifecycle::Deprecated {
            replacement: Some(ModelId::new(crate::CLAUDE_MYTHOS_5).expect("replacement")),
        }
    );
    assert!(provider.language(crate::CLAUDE_MYTHOS_PREVIEW).is_ok());

    let retired = ModelId::new(crate::CLAUDE_OPUS_4_1_20250805).expect("model id");
    let retired_profile = catalog
        .iter()
        .find(|entry| entry.model() == &retired)
        .expect("retired model profile");
    assert!(matches!(
        retired_profile.lifecycle(),
        ModelLifecycle::Retired { .. }
    ));
    assert!(provider.language(crate::CLAUDE_OPUS_4_1_20250805).is_ok());

    let unknown = provider
        .language("claude-future-2030")
        .expect("future model remains callable");
    assert_eq!(unknown.model_id().as_str(), "claude-future-2030");
    assert_eq!(
        provider.registration().provider_id(),
        provider.provider_id()
    );
    let manifest = provider.support_manifest();
    assert_eq!(manifest.profiles().len(), 1);
    assert_eq!(manifest.native_claims().len(), 4);
    assert!(manifest.native_claims().iter().any(|claim| {
        claim
            .scope()
            .binding()
            .surface_id()
            .is_some_and(|surface| surface.as_str() == "files")
            && claim.stability() == ApiStability::Experimental
    }));
}

#[tokio::test]
async fn custom_endpoint_does_not_inherit_anthropic_native_claims() {
    let server = MockServer::start().await;
    let provider = local_provider(&server, AnthropicCredential::unauthenticated());

    assert!(provider.support_manifest().native_claims().is_empty());
    assert!(
        provider.support_manifest().profiles()[0]
            .generic_claims()
            .is_some()
    );
}

#[test]
fn caller_supplied_official_policy_remains_a_custom_endpoint() {
    let endpoint = EndpointConfig::official(
        "https://relay.example/v1/",
        OfficialOrigin::new("https://relay.example").expect("official origin"),
    )
    .expect("policy-bound endpoint");

    let missing_domain = AnthropicProvider::builder(AnthropicCredential::api_key("test-api-key"))
        .with_endpoint(endpoint.clone())
        .build()
        .expect_err("caller endpoint requires a custom replay domain");
    assert!(matches!(
        missing_domain,
        crate::AnthropicConfigError::CustomEndpointRequiresReplayDomain
    ));

    let official_domain = AnthropicProvider::builder(AnthropicCredential::api_key("test-api-key"))
        .with_endpoint(endpoint.clone())
        .with_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("forged-official").expect("replay domain"),
        ))
        .build()
        .expect_err("caller endpoint cannot select an official replay audience");
    assert!(matches!(
        official_domain,
        crate::AnthropicConfigError::ReplayAudienceMismatch
    ));

    let provider = AnthropicProvider::builder(AnthropicCredential::api_key("test-api-key"))
        .with_endpoint(endpoint)
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("caller-relay").expect("replay domain"),
        ))
        .build()
        .expect("custom provider");
    assert!(
        provider
            .profile()
            .provider_profile()
            .verified_claims()
            .is_none()
    );
    assert!(provider.support_manifest().native_claims().is_empty());
}

#[test]
fn credentials_and_debug_surfaces_are_secret_safe() {
    let credential = AnthropicCredential::api_key("anthropic-canary-secret");
    assert!(!format!("{credential:?}").contains("anthropic-canary-secret"));

    let provider = AnthropicProvider::builder(credential)
        .build()
        .expect("provider");
    let debug = format!("{provider:?}");
    assert!(!debug.contains("anthropic-canary-secret"));
    assert!(!debug.contains("api.anthropic.com"));
}

#[test]
fn official_endpoint_rejects_explicit_unauthenticated_mode() {
    assert!(matches!(
        AnthropicProvider::builder(AnthropicCredential::unauthenticated()).build(),
        Err(crate::AnthropicConfigError::OfficialEndpointRequiresAuthentication)
    ));
}

#[test]
fn typed_messages_options_are_inspectable_and_protect_engine_fields() {
    let metadata = MessagesMetadata::new("user-123").expect("metadata");
    let options = AnthropicMessagesOptions::new()
        .with_metadata(metadata)
        .with_enabled_thinking(2_048)
        .try_with_extra("custom_level", json!("standard_only"))
        .expect("safe extra");
    assert_eq!(
        options.metadata().map(MessagesMetadata::user_id),
        Some("user-123")
    );
    assert_eq!(options.thinking(), Some(AnthropicThinking::enabled(2_048)));
    assert_eq!(
        options.extra().get("custom_level"),
        Some(&json!("standard_only"))
    );
    let erased = options.provider_options().expect("typed options");
    assert_eq!(erased.namespace().as_str(), "anthropic");

    for protected in ["api_key", "anthropic_version", "base_url", "headers"] {
        assert!(
            AnthropicMessagesOptions::new()
                .try_with_extra(protected, json!("secret"))
                .is_err()
        );
    }
    assert!(
        AnthropicMessagesOptions::new()
            .with_enabled_thinking(512)
            .provider_options()
            .is_err()
    );
}

#[test]
fn prompt_cache_annotations_project_in_prefix_order_and_enforce_limits() {
    let tool = ToolSpec::new("lookup", None, json!({"type": "object"}))
        .expect("tool")
        .with_provider_annotation(
            &AnthropicToolOptions::new().with_cache_ttl(AnthropicCacheTtl::OneHour),
        )
        .expect("tool annotation");
    let system = Message::text(MessageRole::System, "Stable policy")
        .with_provider_annotation(&AnthropicMessageCache::one_hour())
        .expect("message annotation");
    let content = MessagePart::text("Hello")
        .with_provider_annotation(&AnthropicContentOptions::five_minutes())
        .expect("content annotation");
    let mut annotated =
        LanguageRequest::new(vec![system, Message::new(MessageRole::User, [content])]);
    annotated.generation.max_output_tokens = Some(64);
    annotated.tools.push(tool);
    let encoded = encode_request_with_resolver(
        &ModelId::new("cache-model").expect("model"),
        &annotated,
        &MessagesRequestOptions::default(),
        &AnthropicAnnotationResolver,
    )
    .expect("encoded");
    assert_eq!(encoded["tools"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(encoded["system"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        encoded["messages"][0]["content"][0]["cache_control"]["ttl"],
        "5m"
    );

    let messages = (0..5)
        .map(|index| {
            Message::new(
                MessageRole::User,
                [MessagePart::text(format!("part-{index}"))
                    .with_provider_annotation(&AnthropicContentOptions::five_minutes())
                    .expect("annotation")],
            )
        })
        .collect();
    let mut too_many = LanguageRequest::new(messages);
    too_many.generation.max_output_tokens = Some(64);
    assert!(matches!(
        encode_request_with_resolver(
            &ModelId::new("cache-model").expect("model"),
            &too_many,
            &MessagesRequestOptions::default(),
            &AnthropicAnnotationResolver,
        ),
        Err(MessagesCodecError::TooManyCacheBreakpoints {
            actual: 5,
            maximum: 4
        })
    ));

    let tool = ToolSpec::new("lookup", None, json!({"type": "object"}))
        .expect("tool")
        .with_provider_annotation(
            &AnthropicToolOptions::new().with_cache_ttl(AnthropicCacheTtl::FiveMinutes),
        )
        .expect("tool annotation");
    let system = Message::text(MessageRole::System, "Policy")
        .with_provider_annotation(&AnthropicMessageCache::one_hour())
        .expect("message annotation");
    let mut invalid_order =
        LanguageRequest::new(vec![system, Message::text(MessageRole::User, "Hello")]);
    invalid_order.generation.max_output_tokens = Some(64);
    invalid_order.tools.push(tool);
    assert!(matches!(
        encode_request_with_resolver(
            &ModelId::new("cache-model").expect("model"),
            &invalid_order,
            &MessagesRequestOptions::default(),
            &AnthropicAnnotationResolver,
        ),
        Err(MessagesCodecError::InvalidCacheTtlOrder)
    ));
}

#[tokio::test]
async fn typed_messages_options_use_the_anthropic_wire_shape() {
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
            "metadata": {"user_id": "user-123"},
            "thinking": {
                "type": "enabled",
                "budget_tokens": 2048,
                "display": "omitted"
            },
            "output_config": {"effort": "high"},
            "custom_level": "provider-specific"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "thinking-model",
            "msg_thinking",
            "done",
        )))
        .mount(&server)
        .await;
    let provider = local_provider(&server, AnthropicCredential::unauthenticated());
    let options = AnthropicMessagesOptions::new()
        .with_metadata(MessagesMetadata::new("user-123").expect("metadata"))
        .with_thinking(AnthropicThinking::enabled(2_048).with_display(ThinkingDisplay::Omitted))
        .with_output_effort(OutputEffort::High)
        .try_with_extra("custom_level", json!("provider-specific"))
        .expect("extra");
    let model = provider.language("thinking-model").expect("model");
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &options)
        .expect("call options");
    model
        .generate(request("reason", 4_096), call_options)
        .await
        .expect("generate");
}

#[tokio::test]
async fn generate_and_stream_delegate_with_exact_auth_and_version_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-anthropic-key"))
        .and(header("anthropic-version", API_VERSION))
        .and(header("accept", "application/json"))
        .and(body_json(json!({
            "model": "future-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }],
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "future-model",
            "msg_1",
            "ok",
        )))
        .mount(&server)
        .await;

    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "future-model",
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
        .and(header("x-api-key", "test-anthropic-key"))
        .and(header("anthropic-version", API_VERSION))
        .and(header("accept", "text/event-stream"))
        .and(body_json(json!({
            "model": "future-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }],
            "stream": true
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let provider = local_provider(&server, AnthropicCredential::api_key("test-anthropic-key"));
    let model = provider.language("future-model").expect("model");
    let generated = model
        .generate(request("hello", 64), CallOptions::default())
        .await
        .expect("generate");
    assert_eq!(generated.id(), Some("msg_1"));

    let events = model
        .stream(request("hello", 64), CallOptions::default())
        .await
        .expect("stream")
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::TextDelta { delta, .. } if delta == "hello"
    )));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                LanguageStreamEvent::Terminal(StreamTerminal::Completed { .. })
            ))
            .count(),
        1
    );
}

#[tokio::test]
async fn native_resources_share_auth_transport_and_canonical_message_encoding() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file_123"))
        .and(header("x-api-key", "resource-key"))
        .and(header("anthropic-version", API_VERSION))
        .and(header("anthropic-beta", "files-api-2025-04-14"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file_123",
            "type": "file",
            "filename": "fixture.txt"
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/count_tokens"))
        .and(header("x-api-key", "resource-key"))
        .and(body_json(json!({
            "model": "future-model",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "count me"}]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"input_tokens": 7})))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .and(header("x-api-key", "resource-key"))
        .and(body_json(json!({
            "requests": [{
                "custom_id": "request_1",
                "params": {
                    "model": "future-model",
                    "max_tokens": 64,
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "text": "batch me"}]
                    }]
                }
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msgbatch_123",
            "type": "message_batch",
            "processing_status": "in_progress"
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/skills"))
        .and(query_param("beta", "true"))
        .and(header("x-api-key", "resource-key"))
        .and(header("anthropic-beta", "skills-2025-10-02"))
        .and(body_string_contains(
            "name=\"files[]\"; filename=\"fixture/SKILL.md\"",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "skill_123",
            "type": "skill",
            "display_title": "Fixture",
            "latest_version": "1"
        })))
        .mount(&server)
        .await;

    let provider = local_provider(&server, AnthropicCredential::api_key("resource-key"));
    let file = provider.files().retrieve("file_123").await.expect("file");
    assert_eq!(file.filename.as_deref(), Some("fixture.txt"));

    let count = provider
        .tokens()
        .count(
            "future-model",
            LanguageRequest::new(vec![Message::user("count me")]),
            AnthropicTokenCountOptions::default(),
        )
        .await
        .expect("count");
    assert_eq!(count.input_tokens, 7);

    let batch = AnthropicBatchRequest::new(vec![
        AnthropicBatchItem::new("request_1", "future-model", request("batch me", 64))
            .expect("batch item"),
    ])
    .expect("batch request");
    assert_eq!(
        provider
            .message_batches()
            .create(batch)
            .await
            .expect("batch")
            .id,
        "msgbatch_123"
    );

    let skill = AnthropicSkillUpload::new(vec![
        AnthropicSkillFile::new(
            "fixture/SKILL.md",
            "text/markdown",
            "---\nname: fixture\n---",
        )
        .expect("skill file"),
    ])
    .expect("skill upload")
    .with_display_title("Fixture")
    .expect("title");
    assert_eq!(
        provider
            .skills()
            .upload(skill)
            .await
            .expect("skill")
            .skill
            .id,
        "skill_123"
    );
}

#[tokio::test]
async fn scoped_file_references_encode_messages_blocks_and_files_beta_once() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("anthropic-beta", "files-api-2025-04-14"))
        .and(body_json(json!({
            "model": "future-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "file", "file_id": "file/image?#资源"}
                    },
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": "file_document"},
                        "title": "A title",
                        "context": "A bounded context",
                        "citations": {"enabled": true}
                    }
                ]
            }],
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "future-model",
            "msg_files",
            "file-aware",
        )))
        .expect(1)
        .mount(&server)
        .await;

    let provider = local_provider_with_caller_scope(
        &server,
        AnthropicCredential::unauthenticated(),
        "workspace-a",
    );
    let files = provider.files();
    let image = AnthropicContentOptions::file_part(AnthropicMessageFile::image(
        files
            .reference("file/image?#资源")
            .expect("image reference"),
    ))
    .expect("image annotation");
    let document = AnthropicMessageFile::document(
        files
            .reference("file_document")
            .expect("document reference"),
    )
    .with_title("A title")
    .expect("document title")
    .with_context("A bounded context")
    .expect("document context")
    .with_citations(true)
    .expect("document citations");
    let document = AnthropicContentOptions::file_part(document).expect("document annotation");
    let mut request =
        LanguageRequest::new(vec![Message::new(MessageRole::User, [image, document])]);
    request.generation.max_output_tokens = Some(64);

    let model = provider.language("future-model").expect("model");
    let response = model
        .generate(request, CallOptions::default())
        .await
        .expect("file-aware request");
    assert_eq!(response.id(), Some("msg_files"));
}

#[tokio::test]
async fn file_references_require_caller_scope_and_allow_same_scope_replay_across_instances() {
    let first_server = MockServer::start().await;
    let first_provider = local_provider(&first_server, AnthropicCredential::unauthenticated());
    assert!(
        first_provider
            .files()
            .reference("file_without_workspace")
            .is_err()
    );

    let second_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "future-model",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {"type": "file", "file_id": "file_shared"}
                }]
            }],
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response(
            "future-model",
            "msg_shared_file",
            "ok",
        )))
        .expect(1)
        .mount(&second_server)
        .await;

    let first_provider = local_provider_with_caller_scope(
        &first_server,
        AnthropicCredential::unauthenticated(),
        "workspace-a",
    );
    let second_provider = local_provider_with_caller_scope(
        &second_server,
        AnthropicCredential::unauthenticated(),
        "workspace-a",
    );
    let reference = first_provider
        .files()
        .reference("file_shared")
        .expect("reference");
    let part = AnthropicContentOptions::file_part(AnthropicMessageFile::image(reference))
        .expect("file annotation");
    let mut request = LanguageRequest::new(vec![Message::new(MessageRole::User, [part])]);
    request.generation.max_output_tokens = Some(64);
    second_provider
        .language("future-model")
        .expect("model")
        .generate(request, CallOptions::default())
        .await
        .expect("same durable scope should replay across instances");
}

#[tokio::test]
async fn protected_raw_options_fail_before_network() {
    let server = MockServer::start().await;
    let provider = local_provider(&server, AnthropicCredential::unauthenticated());
    let model = provider.language("future-model").expect("model");
    let call_options = CallOptions::default()
        .with_raw_provider_options_for(&model, json!({"credential_token": "canary-secret"}))
        .expect("checked raw layer");
    let error = model
        .generate(request("hello", 64), call_options)
        .await
        .expect_err("engine-owned protected field");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(!format!("{error:?}").contains("canary-secret"));

    let invalid = AnthropicMessagesOptions::new().with_enabled_thinking(512);
    assert!(invalid.provider_options().is_err());
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );
}

#[tokio::test]
async fn resource_errors_keep_bounded_sensitive_material_off_default_surfaces() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file_error"))
        .respond_with(
            ResponseTemplate::new(400)
                .insert_header("request-id", "req_resource_123")
                .set_body_json(json!({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "private-resource-canary"
                    }
                })),
        )
        .mount(&server)
        .await;
    let provider = local_provider(&server, AnthropicCredential::unauthenticated());
    let error = provider
        .files()
        .retrieve("file_error")
        .await
        .expect_err("resource error");

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    let diagnostics = error.diagnostics().expect("diagnostics");
    assert_eq!(diagnostics.status(), Some(400));
    assert_eq!(diagnostics.provider_type(), Some("invalid_request_error"));
    assert_eq!(diagnostics.request_id(), Some("req_resource_123"));
    assert!(!format!("{error}").contains("private-resource-canary"));
    assert!(!format!("{error:?}").contains("private-resource-canary"));
    assert!(
        !serde_json::to_string(&error)
            .expect("serialize error")
            .contains("private-resource-canary")
    );
    let (_, sensitive_body) = error
        .sensitive_response()
        .expect("explicit sensitive response")
        .expose();
    assert!(
        std::str::from_utf8(sensitive_body)
            .expect("utf8 fixture")
            .contains("private-resource-canary")
    );
}

#[test]
fn upload_inputs_are_bounded_before_transport() {
    assert!(AnthropicFileUpload::new("", "text/plain", "data").is_err());
    assert!(AnthropicSkillFile::new("../secret", "text/plain", "data").is_err());
    assert_eq!(
        AnthropicBatchRequest::new(Vec::new())
            .expect_err("empty batch")
            .kind(),
        ErrorKind::InvalidInput
    );
}
