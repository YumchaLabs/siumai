//! MiniMax Extensions - Music generation (explicit extension API)
//!
//! This example demonstrates how to use `siumai::provider_ext::minimax` helpers
//! to build a MiniMax-flavored music request, while calling the (non-unified)
//! `MusicGenerationCapability` trait on the unified `Siumai` client.
//!
//! ## Run
//! ```bash
//! cargo run --example minimax_music-ext --features minimax
//! ```

use siumai::prelude::extensions::*;
use siumai::provider_ext::minimax::ext::music::MinimaxMusicRequestBuilder;
use siumai::providers::minimax::{MinimaxClient, MinimaxConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MinimaxConfig::new(std::env::var("MINIMAX_API_KEY")?);
    let client = MinimaxClient::from_config(config)?;

    let request = MinimaxMusicRequestBuilder::new("Indie folk, melancholic, acoustic guitar")
        .lyrics_template()
        .format("mp3")
        .sample_rate(44_100)
        .bitrate(256_000)
        .build();

    let resp = client.generate_music(request).await?;
    println!("Audio bytes = {}", resp.audio_data.len());
    Ok(())
}
