//! OpenAI Audio Multimodal Example
//!
//! This example demonstrates how to use OpenAI's audio multimodal capabilities:
//! - Audio input (base64-encoded audio data)
//! - Audio output (text-to-speech with different voices)
//! - Modalities configuration (text + audio)
//!
//! ## Prerequisites
//! - Set `OPENAI_API_KEY` environment variable
//! - Use a model that supports audio (e.g., gpt-4o-audio-preview)
//!
//! ## Run
//! ```bash
//! cargo run --example openai-audio-multimodal --features openai
//! ```

use siumai::prelude::*;
use siumai::types::provider_options::openai::{
    ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
    ChatCompletionModalities,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🎵 OpenAI Audio Multimodal Example\n");

    // Example 1: Audio input (speech-to-text)
    example_audio_input().await?;

    // Example 2: Audio output (text-to-speech)
    example_audio_output().await?;

    // Example 3: Both audio input and output
    example_audio_bidirectional().await?;

    Ok(())
}

/// Example 1: Send audio input to the model
async fn example_audio_input() -> Result<(), Box<dyn std::error::Error>> {
    println!("📥 Example 1: Audio Input (Speech-to-Text)\n");

    let client = OpenAiClient::from_env().build()?;

    // Create a sample base64-encoded WAV audio (this is a placeholder)
    // In a real application, you would encode actual audio data
    let audio_data = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";

    let request = ChatRequest::new("gpt-4o-audio-preview")
        .user_message_multimodal(vec![
            ContentPart::text("What does this audio say?"),
            ContentPart::audio_base64(audio_data, "audio/wav"),
        ]);

    match client.generate(&request).await {
        Ok(response) => {
            println!("✅ Response: {}\n", response.content_text().unwrap_or(""));
        }
        Err(e) => {
            println!("❌ Error: {}\n", e);
        }
    }

    Ok(())
}

/// Example 2: Generate audio output from text
async fn example_audio_output() -> Result<(), Box<dyn std::error::Error>> {
    println!("📤 Example 2: Audio Output (Text-to-Speech)\n");

    let client = OpenAiClient::from_env().build()?;

    // Configure audio output with a specific voice
    let audio_config = ChatCompletionAudio {
        voice: ChatCompletionAudioVoice::Alloy,
        format: ChatCompletionAudioFormat::Wav,
    };

    let request = ChatRequest::new("gpt-4o-audio-preview")
        .user_message("Please say 'Hello, world!' in a friendly tone.")
        .with_provider_options(
            OpenAiOptions::default()
                .with_modalities(vec![
                    ChatCompletionModalities::Text,
                    ChatCompletionModalities::Audio,
                ])
                .with_audio(audio_config),
        );

    match client.generate(&request).await {
        Ok(response) => {
            println!("✅ Text response: {}", response.content_text().unwrap_or(""));
            println!("   (Audio data would be in response.audio field if available)\n");
        }
        Err(e) => {
            println!("❌ Error: {}\n", e);
        }
    }

    Ok(())
}

/// Example 3: Bidirectional audio (both input and output)
async fn example_audio_bidirectional() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Example 3: Bidirectional Audio\n");

    let client = OpenAiClient::from_env().build()?;

    // Sample audio input
    let audio_data = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";

    // Configure audio output
    let audio_config = ChatCompletionAudio {
        voice: ChatCompletionAudioVoice::Shimmer,
        format: ChatCompletionAudioFormat::Mp3,
    };

    let request = ChatRequest::new("gpt-4o-audio-preview")
        .user_message_multimodal(vec![
            ContentPart::text("Listen to this audio and respond with a summary."),
            ContentPart::audio_base64(audio_data, "audio/wav"),
        ])
        .with_provider_options(
            OpenAiOptions::default()
                .with_modalities(vec![
                    ChatCompletionModalities::Text,
                    ChatCompletionModalities::Audio,
                ])
                .with_audio(audio_config),
        );

    match client.generate(&request).await {
        Ok(response) => {
            println!("✅ Text response: {}", response.content_text().unwrap_or(""));
            println!("   (Audio response would be available if the model returns it)\n");
        }
        Err(e) => {
            println!("❌ Error: {}\n", e);
        }
    }

    Ok(())
}

/// Example 4: Different voice options
#[allow(dead_code)]
async fn example_voice_options() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎤 Example 4: Different Voice Options\n");

    let client = OpenAiClient::from_env().build()?;

    let voices = vec![
        ("Alloy", ChatCompletionAudioVoice::Alloy),
        ("Ash", ChatCompletionAudioVoice::Ash),
        ("Ballad", ChatCompletionAudioVoice::Ballad),
        ("Coral", ChatCompletionAudioVoice::Coral),
        ("Echo", ChatCompletionAudioVoice::Echo),
        ("Sage", ChatCompletionAudioVoice::Sage),
        ("Shimmer", ChatCompletionAudioVoice::Shimmer),
        ("Verse", ChatCompletionAudioVoice::Verse),
    ];

    for (name, voice) in voices {
        println!("Testing voice: {}", name);

        let audio_config = ChatCompletionAudio {
            voice,
            format: ChatCompletionAudioFormat::Wav,
        };

        let request = ChatRequest::new("gpt-4o-audio-preview")
            .user_message("Say hello!")
            .with_provider_options(
                OpenAiOptions::default()
                    .with_audio_voice(voice) // Convenience method
            );

        match client.generate(&request).await {
            Ok(response) => {
                println!("  ✅ {}: {}", name, response.content_text().unwrap_or(""));
            }
            Err(e) => {
                println!("  ❌ {}: {}", name, e);
            }
        }
    }

    Ok(())
}

/// Example 5: Using convenience method with_audio_voice
#[allow(dead_code)]
async fn example_convenience_method() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Example 5: Convenience Method\n");

    let client = OpenAiClient::from_env().build()?;

    // The with_audio_voice() method automatically sets modalities and audio config
    let request = ChatRequest::new("gpt-4o-audio-preview")
        .user_message("Tell me a short joke.")
        .with_provider_options(
            OpenAiOptions::default()
                .with_audio_voice(ChatCompletionAudioVoice::Echo)
        );

    match client.generate(&request).await {
        Ok(response) => {
            println!("✅ Response: {}\n", response.content_text().unwrap_or(""));
        }
        Err(e) => {
            println!("❌ Error: {}\n", e);
        }
    }

    Ok(())
}

