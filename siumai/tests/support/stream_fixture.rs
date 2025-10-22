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

/// Load a `.jsonl` fixture file and split it into JSON lines, returning a byte stream
pub fn load_jsonl_fixture_as_bytes(path: &str) -> io::Result<Vec<Result<Vec<u8>, io::Error>>> {
    let raw = std::fs::read_to_string(path)?;
    // Normalize line endings
    let normalized = raw.replace("\r\n", "\n");
    let mut out = Vec::new();
    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Each line is a complete JSON object
        out.push(Ok(trimmed.as_bytes().to_vec()));
    }
    Ok(out)
}

/// Collect ChatStreamEvent sequence from a byte stream using the provided SSE event converter
pub async fn collect_sse_events<C>(
    bytes: Vec<Result<Vec<u8>, io::Error>>,
    converter: C,
) -> Vec<siumai::streaming::ChatStreamEvent>
where
    C: siumai::streaming::SseEventConverter + Clone + 'static,
{
    use siumai::streaming::SseStreamExt;
    let byte_stream = futures_util::stream::iter(bytes);
    let mut sse_stream = byte_stream.into_sse_stream();

    let mut events = Vec::new();
    while let Some(item) = sse_stream.next().await {
        let event: eventsource_stream::Event = item.expect("valid SSE event");

        // Skip [DONE] marker
        if event.data.trim() == "[DONE]" {
            continue;
        }

        let converted = converter.convert_event(event).await;
        for e in converted {
            events.push(e.expect("convert ok"));
        }
    }

    // Call handle_stream_end to get final event
    if let Some(end_event) = converter.handle_stream_end() {
        events.push(end_event.expect("stream end ok"));
    }

    events
}

/// Collect ChatStreamEvent sequence from a JSON line stream using the provided JSON event converter
/// This is used for providers like Ollama that stream newline-delimited JSON instead of SSE
pub async fn collect_json_events<C>(
    bytes: Vec<Result<Vec<u8>, io::Error>>,
    converter: C,
) -> Vec<siumai::streaming::ChatStreamEvent>
where
    C: siumai::streaming::JsonEventConverter + Clone + 'static,
{
    let mut events = Vec::new();

    for chunk in bytes {
        let chunk_bytes = chunk.expect("valid chunk");
        let chunk_str = String::from_utf8(chunk_bytes).expect("valid UTF-8");

        // Split by newlines to handle JSONL format
        for line in chunk_str.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let converted = converter.convert_json(line).await;
            for e in converted {
                events.push(e.expect("convert ok"));
            }
        }
    }

    events
}
