//! OpenAI STT SSE streaming (provider extension).
//!
//! This example demonstrates how to consume OpenAI's `POST /audio/transcriptions` SSE stream
//! (`stream=true`) via Siumai's provider extension API.
//!
//! Notes:
//! - This is intentionally NOT part of the Vercel-aligned unified `TranscriptionCapability`.
//! - Set `OPENAI_AUDIO_FILE` to an audio file path (e.g. `audio.mp3` / `audio.wav`).
//! - Use a model that supports streaming, e.g. `gpt-4o-mini-transcribe` / `gpt-4o-transcribe`.

use futures_util::StreamExt;
use siumai::prelude::unified::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_file = std::env::var("OPENAI_AUDIO_FILE")
        .expect("set OPENAI_AUDIO_FILE=/path/to/audio.(mp3|wav|...)");

    let client = Siumai::builder()
        .openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini") // chat default, not used for STT below
        .build()
        .await?;

    let openai = client
        .downcast_client::<siumai::provider_ext::openai::OpenAiClient>()
        .expect("this Siumai instance is backed by OpenAiClient");

    let mut req = SttRequest::from_file(audio_file);
    req.model = Some("gpt-4o-mini-transcribe".to_string());

    let mut stream =
        siumai::provider_ext::openai::transcription_streaming::stt_sse_stream(openai, req).await?;

    while let Some(item) = stream.next().await {
        match item? {
            siumai::provider_ext::openai::transcription_streaming::OpenAiTranscriptionStreamEvent::TextDelta { delta, .. } => {
                print!("{delta}");
            }
            siumai::provider_ext::openai::transcription_streaming::OpenAiTranscriptionStreamEvent::Segment { id, start, end, speaker, text } => {
                eprintln!("\nsegment {id} [{start:.2}-{end:.2}] speaker={:?}: {text}", speaker);
            }
            siumai::provider_ext::openai::transcription_streaming::OpenAiTranscriptionStreamEvent::Done { text, usage, .. } => {
                eprintln!("\n\ndone");
                if let Some(t) = text {
                    eprintln!("text_len={}", t.len());
                }
                if let Some(u) = usage {
                    eprintln!("usage: {u}");
                }
                break;
            }
            siumai::provider_ext::openai::transcription_streaming::OpenAiTranscriptionStreamEvent::Custom { event_type, .. } => {
                eprintln!("\n(custom event: {event_type})");
            }
        }
    }

    Ok(())
}
