//! MiniMax Extensions - TTS vendor parameters (explicit extension API)
//!
//! This example demonstrates how to use `siumai::provider_ext::minimax` helpers
//! to configure MiniMax-specific TTS parameters, while calling the unified
//! `speech::synthesize` family API.
//!
//! ## Run
//! ```bash
//! cargo run --example minimax_tts-ext --features minimax
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::minimax::options::MinimaxTtsRequestBuilder;
use siumai::providers::minimax::{MinimaxClient, MinimaxConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MinimaxConfig::new(std::env::var("MINIMAX_API_KEY")?);
    let client = MinimaxClient::from_config(config)?;

    let request = MinimaxTtsRequestBuilder::new("Today is such a happy day, of course!")
        .model("speech-2.8-hd")
        .voice_id("male-qn-qingse")
        .format("mp3")
        .speed(1.0)
        .emotion("happy")
        .sample_rate(32_000)
        .bitrate(128_000)
        .channel(1)
        .build();

    let response =
        speech::synthesize(&client, request, speech::SynthesizeOptions::default()).await?;
    println!("Audio bytes = {}", response.audio_data.len());
    println!("Format = {}", response.format);
    Ok(())
}
