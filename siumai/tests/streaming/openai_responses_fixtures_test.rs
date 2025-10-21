//! OpenAI Responses API streaming fixtures tests

use siumai::providers::openai::responses::OpenAiResponsesEventConverter;
use siumai::streaming::ChatStreamEvent;

#[path = "../support/stream_fixture.rs"]
mod support;

#[tokio::test]
async fn responses_output_text_delta_completed_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/responses/output_text_delta_completed.sse",
    )
    .expect("load fixture");

    let converter = OpenAiResponsesEventConverter::new();
    let events = support::collect_sse_events(bytes, converter).await;

    // Concatenate ContentDelta text and receive StreamEnd on completed
    let mut text = String::new();
    let mut saw_end = false;
    for e in events {
        match e {
            ChatStreamEvent::ContentDelta { delta, .. } => text.push_str(&delta),
            ChatStreamEvent::StreamEnd { .. } => saw_end = true,
            _ => {}
        }
    }
    assert_eq!(text, "Hello world");
    assert!(saw_end, "expect stream end on completed");
}

#[tokio::test]
async fn responses_function_call_arguments_sequence_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/responses/function_call_arguments_sequence.sse",
    )
    .expect("load fixture");

    let converter = OpenAiResponsesEventConverter::new();
    let events = support::collect_sse_events(bytes, converter).await;

    // First: init with function name
    let saw_init = events.iter().any(|e| matches!(
        e,
        ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, .. }
        if id == "call_1" && function_name.as_deref() == Some("lookup") && arguments_delta.is_none()
    ));
    // Then: arguments in multiple deltas; concat and compare
    let combined_args = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ToolCallDelta { id, arguments_delta: Some(a), .. } if id == "call_1" => Some(a.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    let saw_end = events.iter().any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. }));

    assert!(saw_init, "expect function call init delta with name only");
    assert_eq!(combined_args, "{\"q\":\"rust\"}");
    assert!(saw_end, "expect stream end on response.completed");
}

#[tokio::test]
async fn responses_usage_mixed_case_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/responses/usage_mixed_case.sse",
    )
    .expect("load fixture");

    let converter = OpenAiResponsesEventConverter::new();
    let events = support::collect_sse_events(bytes, converter).await;

    let saw_usage = events.iter().any(|e| matches!(
        e,
        ChatStreamEvent::UsageUpdate { usage } if usage.prompt_tokens == 7 && usage.completion_tokens == 5 && usage.total_tokens == 12
    ));
    assert!(saw_usage, "expect usage update with mixed-case fields");
}

#[tokio::test]
async fn responses_error_event_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/responses/error_event.sse",
    )
    .expect("load fixture");

    let converter = OpenAiResponsesEventConverter::new();
    let events = support::collect_sse_events(bytes, converter).await;

    let saw_error = events.iter().any(|e| matches!(
        e,
        ChatStreamEvent::Error { error } if error.contains("Rate limit exceeded")
    ));
    assert!(saw_error, "expect an Error event for response.error");
}

#[tokio::test]
async fn responses_partial_without_completed_fixture() {
    // Expect content deltas present, but no StreamEnd since 'response.completed' is absent.
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/responses/partial_without_completed.sse",
    )
    .expect("load fixture");

    let converter = OpenAiResponsesEventConverter::new();
    let events = support::collect_sse_events(bytes, converter).await;
    let saw_content = events.iter().any(|e| matches!(e, ChatStreamEvent::ContentDelta { .. }));
    let saw_end = events.iter().any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. }));
    assert!(saw_content, "expect at least one content delta");
    assert!(!saw_end, "should not see StreamEnd without 'response.completed'");
}

#[tokio::test]
async fn responses_tool_call_then_arguments_then_completed_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/responses/tool_call_then_arguments_then_completed.sse",
    )
    .expect("load fixture");
    let converter = OpenAiResponsesEventConverter::new();
    let events = support::collect_sse_events(bytes, converter).await;

    let saw_init = events.iter().any(|e| matches!(
        e,
        ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, .. }
        if id == "call_1" && function_name.as_deref() == Some("lookup") && arguments_delta.is_none()
    ));
    let args = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ToolCallDelta { id, arguments_delta: Some(a), .. } if id == "call_1" => Some(a.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    let saw_end = events.iter().any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. }));

    assert!(saw_init, "expect tool call init with name");
    assert_eq!(args, "{\"q\":\"hello world\"}");
    assert!(saw_end, "expect StreamEnd on completed");
}

#[tokio::test]
async fn responses_partial_then_error_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/responses/partial_then_error.sse",
    )
    .expect("load fixture");
    let converter = OpenAiResponsesEventConverter::new();
    let events = support::collect_sse_events(bytes, converter).await;
    assert!(events.iter().any(|e| matches!(e, ChatStreamEvent::ContentDelta { delta, .. } if delta.contains("Chunk before error"))));
    assert!(events.iter().any(|e| matches!(e, ChatStreamEvent::Error { error } if error.contains("Upstream failure"))));
    assert!(!events.iter().any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. })), "no end expected after error");
}
