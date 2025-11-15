use siumai_core::execution::chat::{ChatInput, ChatMessageInput, ChatRole};
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::types::FinishReasonCore;
use siumai_std_openai::openai::chat::OpenAiChatStandard;

#[test]
fn openai_chat_request_transformer_builds_expected_json() {
    let messages = vec![
        ChatMessageInput {
            role: ChatRole::System,
            content: "You are a helpful assistant.".to_string(),
        },
        ChatMessageInput {
            role: ChatRole::User,
            content: "Hello".to_string(),
        },
    ];

    let mut input = ChatInput::default();
    input.messages = messages;
    input.model = Some("gpt-4o".to_string());
    input.max_tokens = Some(128);
    input.temperature = Some(0.7);

    let standard = OpenAiChatStandard::new();
    let tx = standard.create_request_transformer("openai");
    let body = tx
        .transform_chat(&input)
        .expect("transform_chat should succeed");

    assert_eq!(body["model"], "gpt-4o");
    assert_eq!(body["max_tokens"], 128);
    let temp = body["temperature"]
        .as_f64()
        .expect("temperature should be a number");
    assert!(temp > 0.69 && temp < 0.71, "unexpected temperature: {temp}");

    let msgs = body["messages"]
        .as_array()
        .expect("messages should be an array");
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "You are a helpful assistant.");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"], "Hello");
}

#[test]
fn openai_chat_response_transformer_parses_minimal_response() {
    let standard = OpenAiChatStandard::new();
    let tx = standard.create_response_transformer("openai");

    let raw = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hi there!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12
        }
    });

    let result = tx
        .transform_chat_response(&raw)
        .expect("transform_chat_response should succeed");

    assert_eq!(result.content, "Hi there!");
    assert!(matches!(result.finish_reason, Some(FinishReasonCore::Stop)));

    let usage = result.usage.expect("usage should be present");
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 7);
    assert_eq!(usage.total_tokens, 12);
}

#[test]
fn openai_chat_stream_converter_handles_basic_chunk_and_usage() {
    let standard = OpenAiChatStandard::new();
    let converter = standard.create_stream_converter("openai");

    let chunk = serde_json::json!({
        "choices": [{
            "delta": {
                "role": "assistant",
                "content": "Hello"
            },
            "index": 0
        }],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7
        }
    });

    let event = eventsource_stream::Event {
        id: String::new(),
        event: String::new(),
        data: chunk.to_string(),
        retry: None,
    };

    let results = converter.convert_event(event);
    assert_eq!(results.len(), 3);

    // First event should be StreamStart
    match &results[0] {
        Ok(ChatStreamEventCore::StreamStart {}) => {}
        other => panic!("expected StreamStart event, got {:?}", other),
    }

    // Second event should be content delta
    match &results[1] {
        Ok(ChatStreamEventCore::ContentDelta { delta, index }) => {
            assert_eq!(delta, "Hello");
            assert_eq!(*index, Some(0));
        }
        other => panic!("expected ContentDelta event, got {:?}", other),
    }

    // Third event should be usage update
    match &results[2] {
        Ok(ChatStreamEventCore::UsageUpdate {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        }) => {
            assert_eq!(*prompt_tokens, 3);
            assert_eq!(*completion_tokens, 4);
            assert_eq!(*total_tokens, 7);
        }
        other => panic!("expected UsageUpdate event, got {:?}", other),
    }
}

#[test]
fn openai_chat_stream_converter_ignores_done_sentinel() {
    let standard = OpenAiChatStandard::new();
    let converter = standard.create_stream_converter("openai");

    let event = eventsource_stream::Event {
        id: String::new(),
        event: String::new(),
        data: "[DONE]".to_string(),
        retry: None,
    };

    let results = converter.convert_event(event);
    assert!(results.is_empty(), "DONE sentinel should yield no events");
}
