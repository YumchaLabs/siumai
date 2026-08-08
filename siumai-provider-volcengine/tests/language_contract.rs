use futures_util::StreamExt;
use siumai_core::{
    CallOptions, ContentPart, ErrorKind, LanguageModel, LanguageRequest, LanguageStreamEvent,
    Message, MessageRole, ProviderOptions, ReplayDomain, ReplayDomainId, StreamTerminal,
};
use siumai_protocol_openai::responses::OPENAI_RESPONSES_PROTOCOL;
use siumai_provider_volcengine::models::DOUBAO_SEED_2_1_PRO_260628;
use siumai_provider_volcengine::{
    ARK_BETA_KNOWLEDGE_SEARCH_HEADER, ArkChatOptions, ArkResponsesOptions, ArkResponsesTool,
    ArkThinking, VolcengineCredential, VolcengineProvider,
};
use siumai_transport::EndpointConfig;

const TEST_MODEL: &str = DOUBAO_SEED_2_1_PRO_260628;

fn test_provider(base_url: &str) -> VolcengineProvider {
    VolcengineProvider::builder(VolcengineCredential::unauthenticated())
        .with_endpoint(EndpointConfig::local_explicit(base_url).expect("local endpoint"))
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("volcengine-test-relay").expect("replay domain"),
        ))
        .build()
        .expect("provider")
}

fn user_request(text: &str) -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, text)])
}

#[tokio::test]
async fn chat_sends_ark_thinking_and_decodes_reasoning() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex(format!(r#"\"model\":\"{TEST_MODEL}\""#)),
            mockito::Matcher::Regex(r#"\"thinking\":\{\"type\":\"enabled\"\}"#.to_string()),
            mockito::Matcher::Regex(r#"\"stream\":false"#.to_string()),
        ]))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(format!(
            r#"{{"id":"chat-ark","model":"{TEST_MODEL}","choices":[{{"index":0,"message":{{"role":"assistant","content":"answer","reasoning_content":"thought"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}}}"#
        ))
        .expect(1)
        .create_async()
        .await;
    let provider = test_provider(&format!("{}/v1", server.url()));
    let options = ArkChatOptions::new().with_thinking(ArkThinking::enabled());

    let response = provider
        .chat_completions(TEST_MODEL)
        .expect("model")
        .generate(
            user_request("hello"),
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("options")),
        )
        .await
        .expect("response");

    assert!(
        response
            .content()
            .iter()
            .any(|part| matches!(part, ContentPart::Reasoning { text } if text == "thought"))
    );
    mock.assert_async().await;
}

#[tokio::test]
async fn responses_stream_preserves_ark_citation_and_settles_once() {
    let mut server = mockito::Server::new_async().await;
    let response_id = "resp-ark-stream";
    let completed_item = serde_json::json!({
        "id": "msg-1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{
            "type": "output_text",
            "text": "answer",
            "annotations": [{
                "type": "doc_citation",
                "title": "Doc",
                "url": "https://example.com/doc"
            }]
        }]
    });
    let frames = [
        serde_json::json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": response_id,
                "created_at": 1_785_811_200_i64,
                "model": TEST_MODEL,
                "status": "in_progress",
                "output": [],
                "usage": null,
                "error": null,
                "incomplete_details": null,
                "reasoning": null
            }
        }),
        serde_json::json!({
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": []
            }
        }),
        serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": 2,
            "output_index": 0,
            "item": completed_item.clone()
        }),
        serde_json::json!({
            "type": "response.completed",
            "sequence_number": 3,
            "response": {
                "id": response_id,
                "created_at": 1_785_811_200_i64,
                "model": TEST_MODEL,
                "status": "completed",
                "output": [completed_item],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2
                },
                "error": null,
                "incomplete_details": null,
                "reasoning": null
            }
        }),
    ];
    let stream_body = frames
        .into_iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect::<String>();
    let mock = server
        .mock("POST", "/v1/responses")
        .match_header(ARK_BETA_KNOWLEDGE_SEARCH_HEADER, "true")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex(r#"\"type\":\"knowledge_search\""#.to_string()),
            mockito::Matcher::Regex(r#"\"stream\":true"#.to_string()),
        ]))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(stream_body)
        .expect(1)
        .create_async()
        .await;
    let provider = test_provider(&format!("{}/v1", server.url()));
    let options =
        ArkResponsesOptions::new().with_native_tool(ArkResponsesTool::knowledge_search("kb-1"));

    let events = provider
        .responses(TEST_MODEL)
        .expect("model")
        .stream(
            user_request("search"),
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("options")),
        )
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
    let response = response.unwrap_or_else(|| panic!("unexpected stream events: {events:#?}"));
    assert!(response.content().iter().any(|part| {
        matches!(
            part,
            ContentPart::Citation(citation)
                if citation.title.as_deref() == Some("Doc")
                    && citation.provider[OPENAI_RESPONSES_PROTOCOL]["type"]
                        == serde_json::json!("doc_citation")
        )
    }));
    mock.assert_async().await;
}

#[tokio::test]
async fn responses_preserves_sanitized_http_error_diagnostics() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/responses")
        .with_status(400)
        .with_header("content-type", "application/json")
        .with_header("x-request-id", "ark-request-42")
        .with_body(
            r#"{"error":{"message":"invalid tool configuration","type":"invalid_request_error","param":"tools.0","code":"invalid_parameter"}}"#,
        )
        .expect(1)
        .create_async()
        .await;
    let provider = test_provider(&format!("{}/v1", server.url()));

    let error = provider
        .responses(TEST_MODEL)
        .expect("model")
        .generate(user_request("hello"), CallOptions::default())
        .await
        .expect_err("request should fail");

    let diagnostics = error.diagnostics().expect("diagnostics");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(diagnostics.provider_code(), Some("invalid_parameter"));
    assert_eq!(diagnostics.provider_param(), Some("tools.0"));
    assert_eq!(diagnostics.request_id(), Some("ark-request-42"));
    mock.assert_async().await;
}
