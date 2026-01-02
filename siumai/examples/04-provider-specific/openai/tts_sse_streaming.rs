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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini") // chat default, not used for TTS below
        .build()
        .await?;

    let openai = client
        .downcast_client::<siumai::provider_ext::openai::OpenAiClient>()
        .expect("this Siumai instance is backed by OpenAiClient");

    let req = TtsRequest::new("hello from siumai (SSE)".to_string())
        .with_model("gpt-4o-mini-tts".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());

    let mut stream =
        siumai::provider_ext::openai::speech_streaming::tts_sse_stream(openai, req).await?;

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
