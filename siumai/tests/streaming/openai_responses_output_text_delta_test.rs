//! OpenAI Responses SSE: response.output_text.delta + response.completed

use eventsource_stream::Event;
use futures_util::StreamExt;
use siumai::providers::openai::responses::OpenAiResponsesEventConverter;
use siumai::streaming::ChatStreamEvent;
use siumai::streaming::{SseEventConverter, SseStreamExt};

#[tokio::test]
async fn responses_output_text_and_completed() {
    let converter = OpenAiResponsesEventConverter::new();
    let sse_chunks = vec![
        // Delta as plain string per Responses API
        format!(
            "event: response.output_text.delta\ndata: {}\n\n",
            r#"{"type":"response.output_text.delta","delta":"Hello ","output_index":0}"#
        ),
        format!(
            "event: response.output_text.delta\ndata: {}\n\n",
            r#"{"type":"response.output_text.delta","delta":"world","output_index":0}"#
        ),
        // Completed with a final payload
        format!(
            "event: response.completed\ndata: {}\n\n",
            r#"{"type":"response.completed","response":{"output":[{"content":[{"type":"output_text","text":"Hello world"}]}]}}"#
        ),
    ];

    let bytes: Vec<Result<Vec<u8>, std::io::Error>> =
        sse_chunks.into_iter().map(|s| Ok(s.into_bytes())).collect();
    let byte_stream = futures_util::stream::iter(bytes);
    let mut sse_stream = byte_stream.into_sse_stream();

    let mut content = String::new();
    let mut saw_end = false;
    while let Some(item) = sse_stream.next().await {
        let event: Event = item.expect("valid event");
        let converted = converter.convert_event(event).await;
        for e in converted {
            match e.expect("ok") {
                ChatStreamEvent::ContentDelta { delta, .. } => content.push_str(&delta),
                ChatStreamEvent::StreamEnd { response } => {
                    saw_end = true;
                    assert!(
                        response
                            .content_text()
                            .unwrap_or_default()
                            .contains("Hello")
                    );
                }
                _ => {}
            }
        }
    }
    assert_eq!(content, "Hello world");
    assert!(saw_end, "should emit StreamEnd on response.completed");
}
