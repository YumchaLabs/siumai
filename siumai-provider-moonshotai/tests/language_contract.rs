use futures_util::StreamExt;
use siumai_core::{
    CallOptions, ErrorKind, LanguageModel, LanguageRequest, LanguageStreamEvent, Message,
    MessageRole, ProviderOptions, ReplayDomain, ReplayDomainId, StreamTerminal, UsageValue,
};
use siumai_provider_moonshotai::{
    KIMI_K3, KimiLanguageOptions, KimiReasoningEffort, MoonshotCredential, MoonshotProvider,
};
use siumai_transport::EndpointConfig;

fn provider(base_url: String) -> MoonshotProvider {
    MoonshotProvider::builder(MoonshotCredential::api_key("test-key"))
        .with_endpoint(EndpointConfig::local_explicit(base_url).expect("local endpoint"))
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("moonshot-test-endpoint").expect("replay domain"),
        ))
        .build()
        .expect("provider")
}

fn request() -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
}

#[tokio::test]
async fn typed_kimi_options_and_cache_usage_survive_direct_generation() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer test-key")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex(r#"\"model\":\"kimi-k3\""#.to_string()),
            mockito::Matcher::Regex(r#"\"reasoning_effort\":\"high\""#.to_string()),
            mockito::Matcher::Regex(r#"\"stream\":false"#.to_string()),
        ]))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"id":"chat-1","model":"kimi-k3","choices":[{"index":0,"message":{"role":"assistant","content":"ok","reasoning_content":"why"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cached_tokens":7}}"#,
        )
        .create_async()
        .await;
    let model = provider(format!("{}/v1", server.url()))
        .language(KIMI_K3)
        .expect("model");
    let options = CallOptions::default().with_provider_options(
        ProviderOptions::typed(
            &KimiLanguageOptions::new().with_reasoning_effort(KimiReasoningEffort::High),
        )
        .expect("typed options"),
    );

    let response = model.generate(request(), options).await.expect("response");

    assert_eq!(response.usage().cache_read_tokens, UsageValue::Known(7));
    mock.assert_async().await;
}

#[tokio::test]
async fn trailing_choice_usage_settles_one_completed_stream() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::Regex(r#"\"stream\":true"#.to_string()))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(concat!(
            "data: {\"id\":\"chat-1\",\"model\":\"kimi-k3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\",\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12,\"cached_tokens\":7}}]}\n\n",
            "data: [DONE]\n\n"
        ))
        .create_async()
        .await;
    let model = provider(format!("{}/v1", server.url()))
        .language(KIMI_K3)
        .expect("model");

    let events = model
        .stream(request(), CallOptions::default())
        .await
        .expect("stream")
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        events
            .iter()
            .filter(|event| event.terminal().is_some())
            .count(),
        1
    );
    let response = events.iter().find_map(|event| match event {
        LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => {
            Some(response.as_ref())
        }
        _ => None,
    });
    assert_eq!(
        response
            .expect("completed response")
            .usage()
            .cache_read_tokens,
        UsageValue::Known(7)
    );
    mock.assert_async().await;
}

#[tokio::test]
async fn provider_error_keeps_typed_sanitized_diagnostics() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(400)
        .with_header("content-type", "application/json")
        .with_header("x-request-id", "kimi-request-42")
        .with_header("retry-after", "2")
        .with_body(
            r#"{"error":{"message":"invalid thinking configuration","type":"invalid_request_error","param":"thinking.keep","code":"invalid_parameter"}}"#,
        )
        .create_async()
        .await;
    let model = provider(format!("{}/v1", server.url()))
        .language(KIMI_K3)
        .expect("model");

    let error = model
        .generate(request(), CallOptions::default())
        .await
        .expect_err("provider error");
    let diagnostics = error.diagnostics().expect("diagnostics");

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(diagnostics.status(), Some(400));
    assert_eq!(diagnostics.provider_code(), Some("invalid_parameter"));
    assert_eq!(diagnostics.provider_type(), Some("invalid_request_error"));
    assert_eq!(diagnostics.provider_param(), Some("thinking.keep"));
    assert_eq!(diagnostics.request_id(), Some("kimi-request-42"));
    assert_eq!(
        diagnostics.retry_after(),
        Some(std::time::Duration::from_secs(2))
    );
    mock.assert_async().await;
}
