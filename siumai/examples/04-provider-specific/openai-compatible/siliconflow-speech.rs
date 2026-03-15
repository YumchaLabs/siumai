//! SiliconFlow text-to-speech on the shared OpenAI-compatible runtime.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - preferred path in this file: config-first built-in compat construction
//! - not a dedicated provider-owned speech package
//!
//! Credentials:
//! - reads `SILICONFLOW_API_KEY` from the environment
//!
//! Run:
//! ```bash
//! export SILICONFLOW_API_KEY="your-api-key-here"
//! cargo run --example siliconflow-speech --features openai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;

const SILICONFLOW_TTS_MODEL: &str = "FunAudioLLM/CosyVoice2-0.5B";
const SILICONFLOW_TTS_VOICE: &str = "FunAudioLLM/CosyVoice2-0.5B:diana";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client =
        OpenAiCompatibleClient::from_builtin_env("siliconflow", Some(SILICONFLOW_TTS_MODEL))
            .await?;

    let request = TtsRequest::new("请用简洁中文播报一条 Rust 异步开发状态更新。".to_string())
        .with_model(SILICONFLOW_TTS_MODEL.to_string())
        .with_voice(SILICONFLOW_TTS_VOICE.to_string())
        .with_format("mp3".to_string());

    let response =
        speech::synthesize(&client, request, speech::SynthesizeOptions::default()).await?;

    std::fs::write("siliconflow-tts-sample.mp3", &response.audio_data)?;
    println!("Saved audio to siliconflow-tts-sample.mp3");
    println!("format: {}", response.format);
    if let Some(sample_rate) = response.sample_rate {
        println!("sample_rate: {sample_rate}");
    }

    Ok(())
}
