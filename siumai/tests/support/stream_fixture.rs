//! Test fixtures utilities: load SSE/JSON streaming chunks and drive converters

use futures_util::StreamExt;
use std::io;

/// Load an `.sse` fixture file and split it into SSE data chunks (separated by blank lines), returning a byte stream
pub fn load_sse_fixture_as_bytes(path: &str) -> io::Result<Vec<Result<Vec<u8>, io::Error>>> {
    let raw = std::fs::read_to_string(path)?;
    // Normalize line endings
    let normalized = raw.replace("\r\n", "\n");
    let mut out = Vec::new();
    for chunk in normalized.split("\n\n") {
        let s = chunk.trim_end_matches('\n');
        if s.is_empty() {
            continue;
        }
        // Restore SSE event blank line terminator
        let mut owned = String::from(s);
        owned.push_str("\n\n");
        out.push(Ok(owned.into_bytes()));
    }
    Ok(out)
}

/// Collect ChatStreamEvent sequence from a byte stream using the provided SSE event converter
pub async fn collect_sse_events<C>(
    bytes: Vec<Result<Vec<u8>, io::Error>>,
    converter: C,
) -> Vec<siumai::stream::ChatStreamEvent>
where
    C: siumai::utils::streaming::SseEventConverter + Clone + 'static,
{
    use siumai::utils::sse_stream::SseStreamExt;
    let byte_stream = futures_util::stream::iter(bytes);
    let mut sse_stream = byte_stream.into_sse_stream();

    let mut events = Vec::new();
    while let Some(item) = sse_stream.next().await {
        let event: eventsource_stream::Event = item.expect("valid SSE event");
        let converted = converter.convert_event(event).await;
        for e in converted {
            events.push(e.expect("convert ok"));
        }
    }
    events
}
