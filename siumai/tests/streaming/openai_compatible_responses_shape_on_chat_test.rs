//! Tests for handling Responses-style streaming shapes on the OpenAI-compatible
//! chat path. These verify that our converter accepts delta as a plain string,
//! nested delta.text, and even raw JSON string events, and that finish_reason
//! produces a StreamEnd.

use siumai::streaming::ChatStreamEvent;
use siumai::streaming::SseEventConverter;

#[tokio::test]
async fn delta_plain_string_yields_content() {
    // Build OpenAI-compatible event converter using the OpenAI standard adapter
    let base = "https://api.openai.com/v1".to_string();
    let adapter: std::sync::Arc<
        dyn siumai::providers::openai_compatible::adapter::ProviderAdapter,
    > = std::sync::Arc::new(siumai::providers::openai::adapter::OpenAiStandardAdapter {
        base_url: base.clone(),
    });
    let cfg = siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
        "openai",
        "test-key",
        &base,
        adapter.clone(),
    )
    .with_model("gpt-4o-mini");

    let conv = siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter::new(
        cfg, adapter,
    );

    // Responses-style: delta as a plain string
    let event = eventsource_stream::Event {
        event: "message".to_string(),
        data: r#"{"delta":"Hello"}"#.to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let out = conv.convert_event(event).await;
    assert!(!out.is_empty());
    let content = out
        .into_iter()
        .filter_map(|e| e.ok())
        .find_map(|ev| match ev {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta),
            _ => None,
        })
        .expect("expected at least one ContentDelta");
    assert_eq!(content, "Hello");
}

#[tokio::test]
async fn delta_text_yields_content() {
    let base = "https://api.openai.com/v1".to_string();
    let adapter: std::sync::Arc<
        dyn siumai::providers::openai_compatible::adapter::ProviderAdapter,
    > = std::sync::Arc::new(siumai::providers::openai::adapter::OpenAiStandardAdapter {
        base_url: base.clone(),
    });
    let cfg = siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
        "openai",
        "test-key",
        &base,
        adapter.clone(),
    )
    .with_model("gpt-4o-mini");

    let conv = siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter::new(
        cfg, adapter,
    );

    // Responses-style: nested delta.text
    let event = eventsource_stream::Event {
        event: "message".to_string(),
        data: r#"{"delta":{"text":"World"}}"#.to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let out = conv.convert_event(event).await;
    assert!(!out.is_empty());
    let content = out
        .into_iter()
        .filter_map(|e| e.ok())
        .find_map(|ev| match ev {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta),
            _ => None,
        })
        .expect("expected at least one ContentDelta");
    assert_eq!(content, "World");
}

#[tokio::test]
async fn json_string_event_yields_content() {
    let base = "https://api.openai.com/v1".to_string();
    let adapter: std::sync::Arc<
        dyn siumai::providers::openai_compatible::adapter::ProviderAdapter,
    > = std::sync::Arc::new(siumai::providers::openai::adapter::OpenAiStandardAdapter {
        base_url: base.clone(),
    });
    let cfg = siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
        "openai",
        "test-key",
        &base,
        adapter.clone(),
    )
    .with_model("gpt-4o-mini");

    let conv = siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter::new(
        cfg, adapter,
    );

    // Entire SSE data is a JSON string (e.g., "Hi"); converter should treat it as text delta
    let event = eventsource_stream::Event {
        event: "message".to_string(),
        data: r#""Hi""#.to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let out = conv.convert_event(event).await;
    assert!(!out.is_empty());
    let content = out
        .into_iter()
        .filter_map(|e| e.ok())
        .find_map(|ev| match ev {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta),
            _ => None,
        })
        .expect("expected at least one ContentDelta");
    assert_eq!(content, "Hi");
}

#[tokio::test]
async fn finish_reason_emits_stream_end() {
    let base = "https://api.openai.com/v1".to_string();
    let adapter: std::sync::Arc<
        dyn siumai::providers::openai_compatible::adapter::ProviderAdapter,
    > = std::sync::Arc::new(siumai::providers::openai::adapter::OpenAiStandardAdapter {
        base_url: base.clone(),
    });
    let cfg = siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
        "openai",
        "test-key",
        &base,
        adapter.clone(),
    )
    .with_model("gpt-4o-mini");

    let conv = siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter::new(
        cfg, adapter,
    );

    // Finish chunk with choices[0].finish_reason should yield a StreamEnd
    let event = eventsource_stream::Event {
        event: "message".to_string(),
        data: r#"{"choices":[{"index":0,"finish_reason":"stop"}]}"#.to_string(),
        id: "4".to_string(),
        retry: None,
    };
    let out = conv.convert_event(event).await;
    assert!(!out.is_empty());
    let mut saw_end = false;
    for e in out {
        if let ChatStreamEvent::StreamEnd { .. } = e.unwrap() {
            saw_end = true;
        }
    }
    assert!(saw_end, "expected StreamEnd emitted on finish_reason");
}
