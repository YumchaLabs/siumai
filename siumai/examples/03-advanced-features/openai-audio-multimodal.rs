//! OpenAI Audio Multimodal Example
//!
//! This example demonstrates how to use OpenAI's audio multimodal capabilities:
//! - Audio input (base64-encoded audio data)
//! - Audio output (text + audio modalities)
//! - Bidirectional audio (audio input + audio output)
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
use siumai::provider_ext::openai::{
    ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
    ChatCompletionModalities, OpenAiChatRequestExt, OpenAiOptions,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OpenAI Audio Multimodal Example\n");

    let api_key = std::env::var("OPENAI_API_KEY")?;

    let client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-audio-preview")
        .build()
        .await?;

    example_audio_input(&client).await?;
    example_audio_output(&client).await?;
    example_audio_bidirectional(&client).await?;

    Ok(())
}

async fn example_audio_input(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Audio Input\n");

    let audio_data = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";

    let message = ChatMessage::user("What does this audio say?")
        .with_content_parts(vec![ContentPart::audio_base64(audio_data, "audio/wav")])
        .build();

    let req = ChatRequest::new(vec![message]);
    let resp = client.chat_request(req).await?;
    println!("Text: {}\n", resp.content_text().unwrap_or_default());

    Ok(())
}

async fn example_audio_output(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Audio Output\n");

    let audio_config = ChatCompletionAudio {
        voice: ChatCompletionAudioVoice::Alloy,
        format: ChatCompletionAudioFormat::Wav,
    };

    let req = ChatRequest::new(vec![user!(
        "Please say 'Hello, world!' in a friendly tone."
    )])
    .with_openai_options(
        OpenAiOptions::new()
            .with_modalities(vec![
                ChatCompletionModalities::Text,
                ChatCompletionModalities::Audio,
            ])
            .with_audio(audio_config),
    );

    let resp = client.chat_request(req).await?;
    println!("Text: {}", resp.content_text().unwrap_or_default());
    println!("Has audio: {}\n", resp.audio.is_some());

    Ok(())
}

async fn example_audio_bidirectional(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Bidirectional Audio\n");

    let audio_data = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";

    let audio_config = ChatCompletionAudio {
        voice: ChatCompletionAudioVoice::Shimmer,
        format: ChatCompletionAudioFormat::Mp3,
    };

    let message = ChatMessage::user("Listen to this audio and respond with a summary.")
        .with_content_parts(vec![ContentPart::audio_base64(audio_data, "audio/wav")])
        .build();

    let req = ChatRequest::new(vec![message]).with_openai_options(
        OpenAiOptions::new()
            .with_modalities(vec![
                ChatCompletionModalities::Text,
                ChatCompletionModalities::Audio,
            ])
            .with_audio(audio_config),
    );

    let resp = client.chat_request(req).await?;
    println!("Text: {}", resp.content_text().unwrap_or_default());
    println!("Has audio: {}\n", resp.audio.is_some());

    Ok(())
}
