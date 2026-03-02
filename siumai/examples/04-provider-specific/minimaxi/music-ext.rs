//! MiniMaxi Extensions - Music generation (explicit extension API)
//!
//! This example demonstrates how to use `siumai::provider_ext::minimaxi` helpers
//! to build a MiniMaxi-flavored music request, while calling the (non-unified)
//! `MusicGenerationCapability` trait on the unified `Siumai` client.
//!
//! ## Run
//! ```bash
//! cargo run --example minimaxi_music-ext --features minimaxi
//! ```

use siumai::prelude::extensions::*;
use siumai::provider_ext::minimaxi::ext::music::MinimaxiMusicRequestBuilder;
use siumai::providers::minimaxi::{MinimaxiClient, MinimaxiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MinimaxiConfig::new(std::env::var("MINIMAXI_API_KEY")?);
    let client = MinimaxiClient::from_config(config)?;

    let request = MinimaxiMusicRequestBuilder::new("Indie folk, melancholic, acoustic guitar")
        .lyrics_template()
        .format("mp3")
        .sample_rate(44_100)
        .bitrate(256_000)
        .build();

    let resp = client.generate_music(request).await?;
    println!("Audio bytes = {}", resp.audio_data.len());
    Ok(())
}
