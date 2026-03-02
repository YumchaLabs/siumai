//! OpenAI TTS SSE streaming (provider extension).
//!
//! This example demonstrates how to consume OpenAI's `POST /audio/speech` SSE stream
//! (`stream_format: "sse"`) via Siumai's provider extension API.
//!
//! Notes:
//! - This is intentionally NOT part of the Vercel-aligned unified `SpeechCapability`.
//! - Use a model that supports SSE streaming, e.g. `gpt-4o-mini-tts`.

use futures_util::StreamExt;
use siumai::prelude::unified::*;
use siumai::provider_ext::openai::{OpenAiClient, OpenAiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let openai = OpenAiClient::from_config(
        OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?).with_model("gpt-4o-mini"),
    )?;

    let req = TtsRequest::new("hello from siumai (SSE)".to_string())
        .with_model("gpt-4o-mini-tts".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());

    let mut stream =
        siumai::provider_ext::openai::ext::speech_streaming::tts_sse_stream(&openai, req).await?;

    let mut total_bytes = 0usize;
    while let Some(item) = stream.next().await {
        match item? {
            AudioStreamEvent::AudioDelta { data, format } => {
                total_bytes += data.len();
                println!("chunk: {} bytes ({})", data.len(), format);
            }
            AudioStreamEvent::Done { metadata, .. } => {
                println!("done: total_bytes={}", total_bytes);
                if let Some(usage) = metadata.get("usage") {
                    println!("usage: {}", usage);
                }
                break;
            }
            AudioStreamEvent::Metadata { .. } => {}
            AudioStreamEvent::Error { error } => return Err(error.into()),
        }
    }

    Ok(())
}
