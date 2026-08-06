use serde_json::{Value, json};
use siumai_core::{
    ApiModeId, ApiStability, CallOptions, ContentPart, ErrorKind, LanguageModel, LanguageRequest,
    MediaData, MediaPart, Message, MessagePart, MessageRole, ModelFamily, Provider,
    ProviderOptions, ToolSpec,
};
use siumai_transport::EndpointConfig;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::{
    MINIMAX_M2_7, MINIMAX_M3, MinimaxChatCompletionsOptions, MinimaxContentCache,
    MinimaxCredential, MinimaxLanguageApi, MinimaxMessageCache, MinimaxMessagesOptions,
    MinimaxProvider, MinimaxReasoningEffort, MinimaxResponsesOptions, MinimaxServiceTier,
    MinimaxThinking, MinimaxToolCache,
};

fn request(parts: impl IntoIterator<Item = ContentPart>) -> LanguageRequest {
    let mut request = LanguageRequest::new(vec![Message::new(MessageRole::User, parts)]);
    request.generation.max_output_tokens = Some(64);
    request
}

fn text_request(text: &str) -> LanguageRequest {
    request([ContentPart::Text {
        text: text.to_string(),
    }])
}

fn provider(server: &MockServer, credential: MinimaxCredential) -> MinimaxProvider {
    MinimaxProvider::builder(credential)
        .with_messages_endpoint(
            EndpointConfig::local_explicit(format!("{}/anthropic/v1/", server.uri()))
                .expect("messages endpoint"),
        )
        .with_openai_endpoint(
            EndpointConfig::local_explicit(format!("{}/v1/", server.uri()))
                .expect("openai endpoint"),
        )
        .with_resource_endpoint(
            EndpointConfig::local_explicit(format!("{}/", server.uri()))
                .expect("resource endpoint"),
        )
        .build()
        .expect("provider")
}

fn messages_response(model: &str) -> Value {
    json!({
        "id": "msg-minimax",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 4, "output_tokens": 2}
    })
}

fn chat_response(model: &str) -> Value {
    json!({
        "id": "chatcmpl-minimax",
        "object": "chat.completion",
        "created": 1_785_811_200_i64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    })
}

fn chat_reasoning_response(model: &str) -> Value {
    json!({
        "id": "chatcmpl-minimax-reasoning",
        "object": "chat.completion",
        "created": 1_785_811_200_i64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "ok",
                "reasoning_content": "inspect the next step",
                "reasoning_details": [{
                    "type": "reasoning.text",
                    "id": "reasoning-text-1",
                    "format": "MiniMax-response-v1",
                    "index": 0,
                    "text": "inspect the next step"
                }]
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    })
}

fn responses_response(model: &str) -> Value {
    json!({
        "id": "resp-minimax",
        "object": "response",
        "created_at": 1_785_811_200_i64,
        "model": model,
        "status": "completed",
        "output": [{
            "id": "msg-minimax",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}]
        }],
        "usage": {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 1},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 6
        },
        "error": null,
        "incomplete_details": null,
        "reasoning": null
    })
}

#[test]
fn provider_identity_modes_and_credentials_are_rust_first() {
    let credential = MinimaxCredential::api_key("canary-secret");
    let debug = format!("{credential:?}");
    assert!(!debug.contains("canary-secret"));
    assert!(debug.contains("REDACTED"));

    let provider = MinimaxProvider::builder(MinimaxCredential::api_key("test-key"))
        .build()
        .expect("provider");
    assert_eq!(provider.provider_id().as_str(), "minimax");
    assert_eq!(
        provider
            .registration()
            .api_mode(ModelFamily::Language)
            .map(ApiModeId::as_str),
        Some("messages")
    );
    assert_eq!(
        provider
            .chat_completions_registration()
            .api_mode(ModelFamily::Language)
            .map(ApiModeId::as_str),
        Some("chat-completions")
    );
    assert_eq!(
        provider
            .responses_registration()
            .api_mode(ModelFamily::Language)
            .map(ApiModeId::as_str),
        Some("responses")
    );
    assert_eq!(
        provider
            .language("future-model")
            .expect("future model")
            .api(),
        MinimaxLanguageApi::Messages
    );
    let manifest = provider.support_manifest();
    assert_eq!(manifest.profiles().len(), 2);
    assert_eq!(manifest.native_claims().len(), 6);
    assert!(manifest.native_claims().iter().any(|claim| {
        claim
            .scope()
            .binding()
            .surface_id()
            .is_some_and(|surface| surface.as_str() == "video-tasks")
            && claim.stability() == ApiStability::Experimental
    }));
}

#[tokio::test]
async fn custom_endpoints_do_not_inherit_official_native_claims() {
    let server = MockServer::start().await;
    let provider = provider(&server, MinimaxCredential::unauthenticated());
    let manifest = provider.support_manifest();

    assert!(manifest.native_claims().is_empty());
    assert!(
        manifest
            .profiles()
            .iter()
            .all(|profile| profile.generic_claims().is_some())
    );
}

#[test]
fn official_endpoints_require_authentication() {
    assert!(matches!(
        MinimaxProvider::builder(MinimaxCredential::unauthenticated()).build(),
        Err(crate::MinimaxConfigError::OfficialEndpointRequiresAuthentication)
    ));
}

#[tokio::test]
async fn messages_is_recommended_and_uses_minimax_dialect_controls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(messages_response(MINIMAX_M3)))
        .expect(1)
        .mount(&server)
        .await;

    let options = MinimaxMessagesOptions::new()
        .with_thinking(MinimaxThinking::Adaptive)
        .with_service_tier(MinimaxServiceTier::Priority);
    let mut request = request([
        ContentPart::Text {
            text: "inspect".to_string(),
        },
        ContentPart::Media(MediaPart {
            media_type: "video/mp4".to_string(),
            data: MediaData::Url("https://example.com/input.mp4".to_string()),
            name: None,
        }),
    ]);
    request.generation.temperature = Some(2.0);
    provider(&server, MinimaxCredential::api_key("test-key"))
        .language(MINIMAX_M3)
        .expect("model")
        .generate(
            request,
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("typed options")),
        )
        .await
        .expect("response");

    let requests = server.received_requests().await.expect("requests");
    let headers = &requests[0].headers;
    assert!(!headers.contains_key("anthropic-version"));
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body");
    assert_eq!(body["temperature"], 2.0);
    assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    assert_eq!(body["service_tier"], "priority");
    assert_eq!(body["messages"][0]["content"][1]["type"], "video");
}

#[tokio::test]
async fn messages_projects_typed_fixed_lifetime_cache_markers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(messages_response(MINIMAX_M3)))
        .expect(1)
        .mount(&server)
        .await;

    let system = Message::text(MessageRole::System, "Stable policy")
        .with_provider_annotation(&MinimaxMessageCache::new())
        .expect("message cache marker");
    let content = MessagePart::text("Hello")
        .with_provider_annotation(&MinimaxContentCache::new())
        .expect("content cache marker");
    let tool = ToolSpec::new("lookup", None, json!({"type": "object"}))
        .expect("tool")
        .with_provider_annotation(&MinimaxToolCache::new())
        .expect("tool cache marker");
    let mut request =
        LanguageRequest::new(vec![system, Message::new(MessageRole::User, [content])]);
    request.generation.max_output_tokens = Some(64);
    request.tools.push(tool);

    provider(&server, MinimaxCredential::unauthenticated())
        .messages(MINIMAX_M3)
        .expect("model")
        .generate(request, CallOptions::default())
        .await
        .expect("response");

    let requests = server.received_requests().await.expect("requests");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body");
    let marker = json!({"type": "ephemeral"});
    assert_eq!(body["system"][0]["cache_control"], marker);
    assert_eq!(body["messages"][0]["content"][0]["cache_control"], marker);
    assert_eq!(body["tools"][0]["cache_control"], marker);
    assert!(body["tools"][0]["cache_control"].get("ttl").is_none());
}

#[tokio::test]
async fn messages_rejects_unverified_mid_conversation_system_before_transport() {
    let server = MockServer::start().await;
    let mut request = LanguageRequest::new(vec![
        Message::text(MessageRole::User, "Start"),
        Message::text(MessageRole::System, "Updated policy"),
    ]);
    request.generation.max_output_tokens = Some(64);

    let error = provider(&server, MinimaxCredential::unauthenticated())
        .messages(MINIMAX_M3)
        .expect("model")
        .generate(request, CallOptions::default())
        .await
        .expect_err("unverified wire shape must fail closed");

    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );
}

#[tokio::test]
async fn known_m2_cannot_claim_disabled_thinking() {
    let server = MockServer::start().await;
    let options = MinimaxMessagesOptions::new().with_thinking(MinimaxThinking::Disabled);
    let error = provider(&server, MinimaxCredential::unauthenticated())
        .messages(MINIMAX_M2_7)
        .expect("model")
        .generate(
            text_request("hello"),
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("typed options")),
        )
        .await
        .expect_err("disabled thinking must fail");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );
}

#[tokio::test]
async fn chat_completions_uses_its_own_defaults_and_wire_contract() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(MINIMAX_M3)))
        .expect(1)
        .mount(&server)
        .await;

    let options = MinimaxChatCompletionsOptions::new()
        .with_thinking(MinimaxThinking::Adaptive)
        .with_service_tier(MinimaxServiceTier::Standard);
    let mut request = text_request("hello");
    request.generation.max_output_tokens = Some(128);
    request.generation.temperature = Some(2.0);
    provider(&server, MinimaxCredential::unauthenticated())
        .chat_completions(MINIMAX_M3)
        .expect("model")
        .generate(
            request,
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("typed options")),
        )
        .await
        .expect("response");

    let requests = server.received_requests().await.expect("requests");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body");
    assert_eq!(body["max_completion_tokens"], 128);
    assert!(body.get("max_tokens").is_none());
    assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    assert_eq!(body["reasoning_split"], true);
    assert_eq!(body["service_tier"], "standard");
}

#[tokio::test]
async fn chat_completions_preserves_and_replays_reasoning_details() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_reasoning_response(MINIMAX_M3)))
        .expect(2)
        .mount(&server)
        .await;

    let provider = provider(&server, MinimaxCredential::unauthenticated());
    let model = provider.chat_completions(MINIMAX_M3).expect("chat model");
    let response = model
        .generate(text_request("first turn"), CallOptions::default())
        .await
        .expect("first response");
    assert!(response.content().iter().any(|part| matches!(
        part,
        ContentPart::ProviderOpaque(item)
            if item.kind()
                == siumai_protocol_openai::chat_completions::REASONING_DETAILS_OPAQUE_KIND
                && item.provenance().provider.as_str() == "minimax"
                && item.provenance().platform.as_deref() == Some("local")
    )));

    let history = LanguageRequest::new(vec![
        Message::text(MessageRole::User, "first turn"),
        Message::new(MessageRole::Assistant, response.content().iter().cloned()),
        Message::text(MessageRole::User, "continue"),
    ]);
    model
        .generate(history, CallOptions::default())
        .await
        .expect("continued response");

    let requests = server.received_requests().await.expect("requests");
    let body: Value = serde_json::from_slice(&requests[1].body).expect("request body");
    assert_eq!(
        body["messages"][1]["reasoning_content"],
        "inspect the next step"
    );
    assert_eq!(
        body["messages"][1]["reasoning_details"][0]["id"],
        "reasoning-text-1"
    );
    assert_eq!(
        body["messages"][1]["reasoning_details"][0]["text"],
        "inspect the next step"
    );
}

#[tokio::test]
async fn responses_encodes_video_and_bounded_typed_options() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_response(MINIMAX_M3)))
        .expect(1)
        .mount(&server)
        .await;

    let options = MinimaxResponsesOptions::new()
        .with_reasoning_effort(MinimaxReasoningEffort::High)
        .with_service_tier(MinimaxServiceTier::Priority)
        .with_prompt_cache_key("conversation-1")
        .with_metadata("tenant", "test");
    let mut request = request([ContentPart::Media(MediaPart {
        media_type: "video/mp4".to_string(),
        data: MediaData::Url("https://example.com/input.mp4".to_string()),
        name: None,
    })]);
    request.generation.temperature = Some(1.0);
    provider(&server, MinimaxCredential::unauthenticated())
        .responses(MINIMAX_M3)
        .expect("model")
        .generate(
            request,
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("typed options")),
        )
        .await
        .expect("response");

    let requests = server.received_requests().await.expect("requests");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body");
    assert_eq!(body["temperature"], 1.0);
    assert_eq!(body["input"][0]["content"][0]["type"], "input_video");
    assert_eq!(body["reasoning"], json!({"effort": "high"}));
    assert_eq!(body["service_tier"], "priority");
    assert_eq!(body["prompt_cache_key"], "conversation-1");
    assert_eq!(body["metadata"], json!({"tenant": "test"}));
}

#[tokio::test]
async fn unknown_models_remain_open_but_receive_no_invented_controls() {
    let server = MockServer::start().await;
    let provider = provider(&server, MinimaxCredential::unauthenticated());
    let options = MinimaxMessagesOptions::new().with_thinking(MinimaxThinking::Adaptive);
    let error = provider
        .messages("future-minimax-model")
        .expect("future model")
        .generate(
            text_request("hello"),
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("typed options")),
        )
        .await
        .expect_err("unverified thinking must fail");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );
}
