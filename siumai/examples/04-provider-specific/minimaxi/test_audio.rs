//! MiniMaxi Audio (TTS) Test
//!
//! Run with:
//!   cargo run --example test_audio --features minimaxi
//!
//! Env:
//!   set MINIMAXI_API_KEY=your-key

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MiniMaxi Audio (TTS) Test\n");

    let api_key = std::env::var("MINIMAXI_API_KEY")
        .expect("Please set MINIMAXI_API_KEY environment variable");

    // Build MiniMaxi client
    let client = Siumai::builder()
        .minimaxi()
        .api_key(&api_key)
        .build()
        .await?;

    println!("Testing Text-to-Speech...");
    
    // Test TTS
    let tts_request = TtsRequest::builder()
        .text("你好，这是一个测试。")
        .voice("speech-2.6-hd")
        .build();

    match client.text_to_speech(tts_request).await {
        Ok(response) => {
            println!("✅ TTS Success!");
            println!("Audio data length: {} bytes", response.audio_data.len());
        }
        Err(e) => {
            println!("❌ TTS Error: {:?}", e);
        }
    }

    Ok(())
}

