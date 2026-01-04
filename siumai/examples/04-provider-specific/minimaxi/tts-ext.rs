//! MiniMaxi Extensions - TTS vendor parameters (explicit extension API)
//!
//! This example demonstrates how to use `siumai::provider_ext::minimaxi` helpers
//! to configure MiniMaxi-specific TTS parameters, while calling the unified
//! `SpeechCapability::tts` method.
//!
//! ## Run
//! ```bash
//! cargo run --example minimaxi_tts-ext --features minimaxi
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::minimaxi::options::MinimaxiTtsRequestBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .minimaxi()
        .api_key(&std::env::var("MINIMAXI_API_KEY")?)
        .build()
        .await?;

    let request = MinimaxiTtsRequestBuilder::new("Today is such a happy day, of course!")
        .model("speech-2.6-hd")
        .voice_id("male-qn-qingse")
        .format("mp3")
        .speed(1.0)
        .emotion("happy")
        .sample_rate(32_000)
        .bitrate(128_000)
        .channel(1)
        .build();

    let response = client.tts(request).await?;
    println!("Audio bytes = {}", response.audio_data.len());
    println!("Format = {}", response.format);
    Ok(())
}
