use futures_util::stream;
use serde::Deserialize;
use siumai::prelude::unified::{
    ChatResponse, ChatStreamEvent, ContentPart, FinishReason, LlmError, MessageContent,
};

#[derive(Debug, Deserialize, PartialEq)]
struct PublicTypedStructuredOutput {
    value: String,
}

#[derive(Debug, Deserialize)]
struct PublicWrongTypedStructuredOutput {
    #[serde(rename = "value")]
    _value: u32,
}

#[test]
fn public_facade_extracts_typed_json_from_response() {
    let response = ChatResponse::new(MessageContent::Text("{\"value\":\"ok\"}".to_string()));

    let typed: PublicTypedStructuredOutput =
        siumai::structured_output::extract_json_from_response(&response).expect("typed parse");

    assert_eq!(
        typed,
        PublicTypedStructuredOutput {
            value: "ok".to_string(),
        }
    );
}

#[test]
fn public_facade_reports_target_type_mismatch() {
    let response = ChatResponse::new(MessageContent::Text("{\"value\":\"ok\"}".to_string()));

    let err = siumai::structured_output::extract_json_from_response::<
        PublicWrongTypedStructuredOutput,
    >(&response)
    .expect_err("typed mismatch");

    match err {
        LlmError::ParseError(message) => {
            assert!(message.contains("deserialize structured output JSON into target type"))
        }
        other => panic!("expected ParseError, got {other:?}"),
    }
}

#[test]
fn public_facade_reports_content_filter_when_no_json_was_produced() {
    let mut response = ChatResponse::new(MessageContent::Text(
        "I cannot comply with that request.".to_string(),
    ));
    response.finish_reason = Some(FinishReason::ContentFilter);

    let err = siumai::structured_output::extract_json_value_from_response(&response)
        .expect_err("content filter should fail");

    match err {
        LlmError::ParseError(message) => {
            assert!(message.contains("content filtering/refusal"))
        }
        other => panic!("expected ParseError, got {other:?}"),
    }
}

#[test]
fn public_facade_extracts_reserved_json_tool_response_when_text_is_empty() {
    let response = ChatResponse::new(MessageContent::MultiModal(vec![ContentPart::tool_call(
        "call_1",
        "json",
        serde_json::json!({ "value": "ok" }),
        None,
    )]));

    let value = siumai::structured_output::extract_json_value_from_response(&response)
        .expect("reserved tool json should parse");

    assert_eq!(value["value"], "ok");
}

#[tokio::test]
async fn public_facade_rejects_truncated_stream_without_stream_end() {
    let events = vec![Ok(ChatStreamEvent::ContentDelta {
        delta: "{\"value\":".to_string(),
        index: Some(0),
    })];
    let stream = Box::pin(stream::iter(events));

    let err = siumai::structured_output::extract_json_value_from_stream(stream)
        .await
        .expect_err("truncated stream should fail");

    match err {
        LlmError::ParseError(message) => {
            assert!(message.contains("stream ended before a complete JSON value was produced"))
        }
        other => panic!("expected ParseError, got {other:?}"),
    }
}

#[tokio::test]
async fn public_facade_accepts_stream_end_response_with_best_effort_repair() {
    let events = vec![
        Ok(ChatStreamEvent::ContentDelta {
            delta: "{\"value\":".to_string(),
            index: Some(0),
        }),
        Ok(ChatStreamEvent::StreamEnd {
            response: ChatResponse::new(MessageContent::Text(
                "```json\n{\"value\":\"ok\"}\n```".to_string(),
            )),
        }),
    ];
    let stream = Box::pin(stream::iter(events));

    let value = siumai::structured_output::extract_json_value_from_stream(stream)
        .await
        .expect("stream-end repair should parse");

    assert_eq!(value["value"], "ok");
}

#[tokio::test]
async fn public_facade_accepts_unknown_stream_end_using_accumulated_deltas() {
    let events = vec![
        Ok(ChatStreamEvent::ContentDelta {
            delta: "{\"value\":\"ok\"}".to_string(),
            index: Some(0),
        }),
        Ok(ChatStreamEvent::StreamEnd {
            response: ChatResponse::empty_with_finish_reason(FinishReason::Unknown),
        }),
    ];
    let stream = Box::pin(stream::iter(events));

    let value = siumai::structured_output::extract_json_value_from_stream(stream)
        .await
        .expect("unknown stream-end should still use accumulated complete JSON");

    assert_eq!(value["value"], "ok");
}

#[tokio::test]
async fn public_facade_rejects_truncated_unknown_stream_end_using_accumulated_deltas() {
    let events = vec![
        Ok(ChatStreamEvent::ContentDelta {
            delta: "{\"value\":".to_string(),
            index: Some(0),
        }),
        Ok(ChatStreamEvent::StreamEnd {
            response: ChatResponse::empty_with_finish_reason(FinishReason::Unknown),
        }),
    ];
    let stream = Box::pin(stream::iter(events));

    let err = siumai::structured_output::extract_json_value_from_stream(stream)
        .await
        .expect_err("truncated unknown stream-end should fail");

    match err {
        LlmError::ParseError(message) => {
            assert!(message.contains("stream ended before a complete JSON value was produced"))
        }
        other => panic!("expected ParseError, got {other:?}"),
    }
}

#[tokio::test]
async fn public_facade_rejects_truncated_reserved_tool_stream_without_stream_end() {
    let events = vec![Ok(ChatStreamEvent::ToolCallDelta {
        id: "call_1".to_string(),
        function_name: Some("json".to_string()),
        arguments_delta: Some("{\"value\":".to_string()),
        index: Some(0),
    })];
    let stream = Box::pin(stream::iter(events));

    let err = siumai::structured_output::extract_json_value_from_stream(stream)
        .await
        .expect_err("truncated reserved-tool stream should fail");

    match err {
        LlmError::ParseError(message) => {
            assert!(message.contains("stream ended before a complete JSON value was produced"))
        }
        other => panic!("expected ParseError, got {other:?}"),
    }
}
