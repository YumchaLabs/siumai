//! MiniMaxi Extensions - Video task creation (explicit extension API)
//!
//! This example demonstrates how to use `siumai::provider_ext::minimaxi` helpers
//! to build a MiniMaxi-flavored video request, while calling the (non-unified)
//! `VideoGenerationCapability` trait on the unified `Siumai` client.
//!
//! ## Run
//! ```bash
//! cargo run --example minimaxi_video-ext --features minimaxi
//! ```

use siumai::prelude::extensions::*;
use siumai::provider_ext::minimaxi::ext::video::MinimaxiVideoRequestBuilder;
use siumai::providers::minimaxi::{MinimaxiClient, MinimaxiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MinimaxiConfig::new(std::env::var("MINIMAXI_API_KEY")?);
    let client = MinimaxiClient::from_config(config)?;

    let request = MinimaxiVideoRequestBuilder::new(
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
