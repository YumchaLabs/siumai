//! Together text-to-speech on the shared OpenAI-compatible runtime.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - preferred path in this file: config-first built-in compat construction
//! - not a dedicated provider-owned speech package
//!
//! Credentials:
//! - reads `TOGETHER_API_KEY` from the environment
//!
//! Run:
//! ```bash
//! export TOGETHER_API_KEY="your-api-key-here"
//! cargo run --example together-speech --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;

const TOGETHER_TTS_MODEL: &str = "cartesia/sonic-2";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client =
        OpenAiCompatibleClient::from_builtin_env("together", Some(TOGETHER_TTS_MODEL)).await?;

    let request = TtsRequest::new(
        "Give me a short status update about Rust async best practices.".to_string(),
    )
    .with_model(TOGETHER_TTS_MODEL.to_string())
    .with_voice("alloy".to_string())
    .with_format("mp3".to_string());

    let response =
        speech::synthesize(&client, request, speech::SynthesizeOptions::default()).await?;

    std::fs::write("together-tts-sample.mp3", &response.audio_data)?;
    println!("Saved audio to together-tts-sample.mp3");
    println!("format: {}", response.format);
    if let Some(sample_rate) = response.sample_rate {
        println!("sample_rate: {sample_rate}");
    }

    Ok(())
}
