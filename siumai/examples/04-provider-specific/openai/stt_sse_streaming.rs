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
use siumai::provider_ext::openai::{OpenAiClient, OpenAiConfig};

fn infer_audio_media_type(path: &str) -> Option<&'static str> {
    let extension = std::path::Path::new(path)
        .extension()?
        .to_string_lossy()
        .to_ascii_lowercase();

    match extension.as_str() {
        "mp3" => Some("audio/mpeg"),
        "wav" => Some("audio/wav"),
        "m4a" => Some("audio/mp4"),
        "flac" => Some("audio/flac"),
        "ogg" => Some("audio/ogg"),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_file = std::env::var("OPENAI_AUDIO_FILE")
        .expect("set OPENAI_AUDIO_FILE=/path/to/audio.(mp3|wav|...)");

    let openai = OpenAiClient::from_config(
        OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?).with_model("gpt-4o-mini"),
    )?;

    let audio_bytes = tokio::fs::read(&audio_file).await?;
    let media_type = if let Ok(media_type) = std::env::var("OPENAI_AUDIO_MEDIA_TYPE") {
        media_type
    } else if let Some(media_type) = infer_audio_media_type(&audio_file) {
        media_type.to_string()
    } else {
        return Err("Could not infer OPENAI audio media type; set OPENAI_AUDIO_MEDIA_TYPE".into());
    };
    let mut req = SttRequest::from_audio(audio_bytes, media_type);
    req.model = Some("gpt-4o-mini-transcribe".to_string());

    let mut stream =
        siumai::provider_ext::openai::ext::transcription_streaming::stt_sse_stream(&openai, req)
            .await?;

    while let Some(item) = stream.next().await {
        match item? {
            siumai::provider_ext::openai::ext::transcription_streaming::OpenAiTranscriptionStreamEvent::TextDelta { delta, .. } => {
                print!("{delta}");
            }
            siumai::provider_ext::openai::ext::transcription_streaming::OpenAiTranscriptionStreamEvent::Segment { id, start, end, speaker, text } => {
                eprintln!("\nsegment {id} [{start:.2}-{end:.2}] speaker={:?}: {text}", speaker);
            }
            siumai::provider_ext::openai::ext::transcription_streaming::OpenAiTranscriptionStreamEvent::Done { text, usage, .. } => {
                eprintln!("\n\ndone");
                if let Some(t) = text {
                    eprintln!("text_len={}", t.len());
                }
                if let Some(u) = usage {
                    eprintln!("usage: {u}");
                }
                break;
            }
            siumai::provider_ext::openai::ext::transcription_streaming::OpenAiTranscriptionStreamEvent::Custom { event_type, .. } => {
                eprintln!("\n(custom event: {event_type})");
            }
        }
    }

    Ok(())
}
