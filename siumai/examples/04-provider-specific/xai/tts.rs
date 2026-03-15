//! xAI text-to-speech on the provider-owned audio path.
//!
//! Package tier:
//! - provider-owned wrapper package
//! - preferred path in this directory: config-first `XaiClient`
//!
//! Credentials:
//! - set `XAI_API_KEY` before running this example
//!
//! Run:
//! ```bash
//! cargo run --example xai-tts --features xai
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::xai::{XaiClient, XaiTtsOptions, XaiTtsRequestExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = XaiClient::from_builtin_env(None).await?;

    let request = TtsRequest::new(
        "Give me a short status update about Rust async best practices.".to_string(),
    )
    .with_voice("aria".to_string())
    .with_format("mp3".to_string())
    .with_xai_tts_options(
        XaiTtsOptions::new()
            .with_sample_rate(44_100)
            .with_bit_rate(192_000),
    );

    let response =
        speech::synthesize(&client, request, speech::SynthesizeOptions::default()).await?;

    std::fs::write("xai-tts-sample.mp3", &response.audio_data)?;
    println!("Saved audio to xai-tts-sample.mp3");
    println!("format: {}", response.format);
    if let Some(sample_rate) = response.sample_rate {
        println!("sample_rate: {sample_rate}");
    }

    Ok(())
}
