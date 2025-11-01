# MiniMaxi Video Generation Example

This example demonstrates how to use the MiniMaxi provider's video generation capability.

## Basic Usage

```rust
use siumai::prelude::*;
use siumai::types::video::{VideoGenerationRequest, VideoTaskStatus};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create MiniMaxi client
    let client = LlmBuilder::new()
        .minimaxi()
        .api_key("your-api-key")
        .build()
        .await?;

    // Create video generation request
    let request = VideoGenerationRequest::new(
        "MiniMax-Hailuo-2.3",
        "A beautiful sunset over the ocean with waves gently crashing on the shore"
    )
    .with_duration(6)
    .with_resolution("1080P")
    .with_prompt_optimizer(true);

    // Submit video generation task
    println!("Submitting video generation task...");
    let response = client.create_video_task(request).await?;
    println!("Task ID: {}", response.task_id);

    // Poll task status until completion
    loop {
        let status = client.query_video_task(&response.task_id).await?;
        
        match status.status {
            VideoTaskStatus::Success => {
                println!("Video generation completed!");
                println!("File ID: {:?}", status.file_id);
                println!("Video dimensions: {}x{}", 
                    status.video_width.unwrap_or(0),
                    status.video_height.unwrap_or(0)
                );
                break;
            }
            VideoTaskStatus::Fail => {
                println!("Video generation failed!");
                break;
            }
            VideoTaskStatus::Processing => {
                println!("Processing...");
            }
            VideoTaskStatus::Queueing => {
                println!("Queueing...");
            }
            VideoTaskStatus::Preparing => {
                println!("Preparing...");
            }
        }
        
        // Wait before polling again
        sleep(Duration::from_secs(5)).await;
    }

    Ok(())
}
```

## Supported Models

MiniMaxi supports the following video generation models:

| Model | Duration | Resolution | Description |
|-------|----------|------------|-------------|
| `MiniMax-Hailuo-2.3` | 6s, 10s | 768P, 1080P | Latest Hailuo model with high quality |
| `MiniMax-Hailuo-02` | 6s, 10s | 768P, 1080P | Previous Hailuo model |
| `T2V-01-Director` | 6s | 720P | Director mode for T2V |
| `T2V-01` | 6s | 720P | Basic T2V model |

## Query Supported Capabilities

```rust
use siumai::traits::VideoGenerationCapability;

// Get supported models
let models = client.get_supported_models();
println!("Supported models: {:?}", models);

// Get supported resolutions for a model
let resolutions = client.get_supported_resolutions("MiniMax-Hailuo-2.3");
println!("Supported resolutions: {:?}", resolutions);

// Get supported durations for a model
let durations = client.get_supported_durations("MiniMax-Hailuo-2.3");
println!("Supported durations: {:?}", durations);
```

## Advanced Options

```rust
use siumai::types::video::VideoGenerationRequest;

let request = VideoGenerationRequest::new(
    "MiniMax-Hailuo-2.3",
    "A futuristic city at night with neon lights"
)
.with_duration(10)                    // 10 seconds video
.with_resolution("1080P")             // 1080P resolution
.with_prompt_optimizer(true)          // Enable prompt optimization
.with_fast_pretreatment(false)        // Disable fast preprocessing
.with_watermark(false)                // No watermark
.with_callback_url("https://your-callback-url.com"); // Optional callback

let response = client.create_video_task(request).await?;
```

## Task Status Helpers

```rust
use siumai::types::video::VideoTaskStatusResponse;

let status: VideoTaskStatusResponse = client.query_video_task(&task_id).await?;

// Check if task is complete
if status.is_complete() {
    println!("Task completed!");
}

// Check if task succeeded
if status.is_success() {
    println!("Video ready: {:?}", status.file_id);
}

// Check if task failed
if status.is_failed() {
    println!("Task failed!");
}

// Check if task is still in progress
if status.is_in_progress() {
    println!("Still processing...");
}
```

## Error Handling

```rust
use siumai::error::LlmError;

match client.create_video_task(request).await {
    Ok(response) => {
        println!("Task created: {}", response.task_id);
    }
    Err(LlmError::ProviderError { provider, message, .. }) => {
        eprintln!("Provider error from {}: {}", provider, message);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Notes

1. **Asynchronous Processing**: Video generation is an asynchronous task. You need to poll the task status until it completes.

2. **Polling Interval**: It's recommended to poll every 5-10 seconds to avoid excessive API calls.

3. **Timeout**: Consider implementing a timeout mechanism to avoid infinite polling.

4. **Callback URL**: You can provide a callback URL to receive notifications when the task completes, instead of polling.

5. **File Retrieval**: After the task succeeds, you'll receive a `file_id`. You may need to use the file management API to download the actual video file.

## Complete Example with Timeout

```rust
use siumai::prelude::*;
use siumai::types::video::{VideoGenerationRequest, VideoTaskStatus};
use std::time::Duration;
use tokio::time::{sleep, timeout};

async fn generate_video_with_timeout(
    client: &impl VideoGenerationCapability,
    request: VideoGenerationRequest,
    max_wait: Duration,
) -> Result<String, Box<dyn std::error::Error>> {
    // Submit task
    let response = client.create_video_task(request).await?;
    let task_id = response.task_id.clone();
    
    // Poll with timeout
    let result = timeout(max_wait, async {
        loop {
            let status = client.query_video_task(&task_id).await?;
            
            if status.is_success() {
                return Ok(status.file_id.unwrap_or_default());
            } else if status.is_failed() {
                return Err("Video generation failed".into());
            }
            
            sleep(Duration::from_secs(5)).await;
        }
    }).await;
    
    match result {
        Ok(Ok(file_id)) => Ok(file_id),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Timeout waiting for video generation".into()),
    }
}
```

