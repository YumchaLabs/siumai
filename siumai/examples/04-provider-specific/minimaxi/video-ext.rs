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
use siumai::prelude::unified::*;
use siumai::provider_ext::minimaxi::ext::video::MinimaxiVideoRequestBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .minimaxi()
        .api_key(&std::env::var("MINIMAXI_API_KEY")?)
        .build()
        .await?;

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
