//! SiliconFlow speech-to-text on the shared OpenAI-compatible runtime.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - preferred path in this file: config-first built-in compat construction
//! - not a dedicated provider-owned transcription package
//!
//! Credentials:
//! - reads `SILICONFLOW_API_KEY` from the environment
//! - reads `SILICONFLOW_AUDIO_FILE` from the environment
//!
//! Run:
//! ```bash
//! export SILICONFLOW_API_KEY="your-api-key-here"
//! export SILICONFLOW_AUDIO_FILE="/path/to/audio.mp3"
//! cargo run --example siliconflow-transcription --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;

const SILICONFLOW_STT_MODEL: &str = "FunAudioLLM/SenseVoiceSmall";

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
    let audio_file =
        std::env::var("SILICONFLOW_AUDIO_FILE").expect("set SILICONFLOW_AUDIO_FILE=/path/to/audio");

    let client =
        OpenAiCompatibleClient::from_builtin_env("siliconflow", Some(SILICONFLOW_STT_MODEL))
            .await?;

    let mut request = SttRequest::from_file(audio_file.clone()).with_model(SILICONFLOW_STT_MODEL);

    if let Ok(media_type) = std::env::var("SILICONFLOW_AUDIO_MEDIA_TYPE") {
        request = request.with_media_type(media_type);
    } else if let Some(media_type) = infer_audio_media_type(&audio_file) {
        request = request.with_media_type(media_type.to_string());
    }

    let response = transcription::transcribe(
        &client,
        request,
        transcription::TranscribeOptions::default(),
    )
    .await?;

    println!("Text:\n{}", response.text);
    if let Some(language) = response.language {
        println!("language: {language}");
    }
    if let Some(duration) = response.duration {
        println!("duration_seconds: {duration}");
    }

    Ok(())
}
