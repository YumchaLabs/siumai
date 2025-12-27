//! MiniMaxi Basic Example
//!
//! Run with:
//!   cargo run -p siumai --example minimaxi_basic --features minimaxi
//!
//! Env:
//!   set MINIMAXI_API_KEY=your-key

use futures::StreamExt;
use siumai::prelude::unified::*;
use siumai::provider_ext::minimaxi::tts::MinimaxiTtsRequestBuilder;
use siumai::providers::minimaxi::model_constants;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MiniMaxi Basic Chat Example\n");

    let api_key = std::env::var("MINIMAXI_API_KEY")
        .expect("Please set MINIMAXI_API_KEY environment variable");

    // Build MiniMaxi client (now with SiumaiBuilder::minimaxi())
    let client = Siumai::builder()
        .minimaxi()
        .api_key(&api_key)
        .model(model_constants::text::MINIMAX_M2)
        .build()
        .await?;

    // Send a simple message
    let resp = client
        .chat(vec![siumai::user!("Hello MiniMaxi! Give me one fun fact.")])
        .await?;

    // Optionally show extracted reasoning (default: hidden)
    let show_reasoning = std::env::var("SIUMAI_SHOW_THINKING").ok().as_deref() == Some("1");
    if show_reasoning && resp.has_reasoning() {
        println!("Thinking:\n{}\n", resp.reasoning().join("\n"));
    }

    // Print final answer text (without reasoning tags)
    println!("AI: {}", resp.content_text().unwrap_or_default());

    // --- Streaming demo: prints ThinkingDelta and ContentDelta ---
    println!("\nStreaming (answer only):\n");
    let mut stream = client
        .chat_stream(
            vec![siumai::user!(
                "Give me a short fun fact and think step by step."
            )],
            None,
        )
        .await?;

    let mut printed_any_content = false;

    // Toggle to display streamed thinking via env var
    let show_stream_thinking = std::env::var("SIUMAI_SHOW_THINKING").ok().as_deref() == Some("1");

    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ThinkingDelta { delta } => {
                if show_stream_thinking {
                    if !printed_any_content {
                        println!("Thinking: ");
                    }
                    print!("{}", delta);
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
            }
            ChatStreamEvent::ContentDelta { delta, .. } => {
                if !printed_any_content {
                    println!("\n\nAI: ");
                    printed_any_content = true;
                }
                print!("{}", delta);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            ChatStreamEvent::StreamStart { .. }
            | ChatStreamEvent::UsageUpdate { .. }
            | ChatStreamEvent::ToolCallDelta { .. }
            | ChatStreamEvent::Custom { .. } => {}
            ChatStreamEvent::StreamEnd { response } => {
                if !printed_any_content {
                    // If no deltas were printed (rare), show the final content snapshot
                    if let Some(text) = response.content_text() {
                        println!("\nAI: {}", text);
                    }
                } else {
                    println!();
                }
                if let Some(usage) = &response.usage {
                    println!(
                        "\nUsage: total={} prompt={} completion={}",
                        usage.total_tokens, usage.prompt_tokens, usage.completion_tokens
                    );
                }
            }
            ChatStreamEvent::Error { error } => {
                eprintln!("\n[stream error] {}", error);
            }
        }
    }

    if let Some(usage) = &resp.usage {
        println!(
            "\nUsage: total={} prompt={} completion={}",
            usage.total_tokens, usage.prompt_tokens, usage.completion_tokens
        );
    }

    // Test Audio (TTS) functionality
    println!("\n\n=== Testing Audio (TTS) ===");
    let tts_request = MinimaxiTtsRequestBuilder::new("你好，这是一个测试。")
        .model(model_constants::audio::SPEECH_2_6_HD)
        .voice_id(model_constants::voice::MALE_QN_QINGSE)
        .format("mp3")
        .build();

    match client.tts(tts_request).await {
        Ok(response) => {
            println!("✅ TTS Success!");
            println!("Audio data length: {} bytes", response.audio_data.len());
        }
        Err(e) => {
            println!("❌ TTS Error: {:?}", e);
        }
    }

    // Test Image Generation functionality
    println!("\n\n=== Testing Image Generation ===");
    let image_request = ImageGenerationRequest {
        prompt: "一只可爱的猫咪在花园里玩耍".to_string(),
        model: Some(model_constants::images::IMAGE_01.to_string()),
        size: Some("1024x1024".to_string()),
        count: 1,
        ..Default::default()
    };

    match client.generate_images(image_request).await {
        Ok(response) => {
            println!("✅ Image Generation Success!");
            println!("Generated {} image(s)", response.images.len());
            for (i, img) in response.images.iter().enumerate() {
                if let Some(url) = &img.url {
                    println!("  Image {}: {}", i + 1, url);
                }
            }
        }
        Err(e) => {
            println!("❌ Image Generation Error: {:?}", e);
        }
    }

    Ok(())
}
