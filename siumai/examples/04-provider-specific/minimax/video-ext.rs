//! MiniMax Extensions - Video task creation (explicit extension API)
//!
//! This example demonstrates how to use `siumai::provider_ext::minimax` helpers
//! to build a MiniMax-flavored video request, while calling the (non-unified)
//! `VideoGenerationCapability` trait on the unified `Siumai` client.
//!
//! ## Run
//! ```bash
//! cargo run --example minimax_video-ext --features minimax
//! ```

use siumai::prelude::extensions::*;
use siumai::provider_ext::minimax::ext::video::MinimaxVideoRequestBuilder;
use siumai::providers::minimax::{MinimaxClient, MinimaxConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MinimaxConfig::new(std::env::var("MINIMAX_API_KEY")?);
    let client = MinimaxClient::from_config(config)?;

    let request = MinimaxVideoRequestBuilder::new(
        "MiniMax-Hailuo-2.3",
        "A cinematic sunset over the ocean, wide shot, gentle camera movement",
    )
    .duration(6)
    .resolution("1080P")
    .prompt_optimizer(true)
    .watermark(false)
    .build();

    let resp = client.create_video_task(request).await?;
    println!("Task ID = {}", resp.task_id);
    Ok(())
}
