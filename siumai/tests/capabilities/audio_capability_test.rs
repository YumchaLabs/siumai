//! Audio Family Integration Tests
//!
//! These tests verify speech/transcription functionality across supported providers.
//! They are ignored by default to prevent accidental API usage during normal testing.
//!
//! ## Running Tests
//!
//! ```bash
//! # Test specific provider audio capabilities
//! export OPENAI_API_KEY="your-key"
//! cargo test test_openai_audio -- --ignored
//!
//! # Test all available providers
//! cargo test test_all_provider_audio -- --ignored
//! ```

#![allow(deprecated)]

use siumai::experimental::client::LlmClient;
use siumai::prelude::unified::{SpeechCapability, TranscriptionCapability, TtsRequest};
use siumai::prelude::*;
use std::env;

async fn test_text_to_speech<T: SpeechCapability>(client: &T, provider_name: &str) {
    println!("  Testing Text-to-Speech for {}...", provider_name);

    let (voice, format, model) = match provider_name {
        "OpenAI" => ("alloy", "mp3", "tts-1"),
        _ => ("alloy", "mp3", "default"),
    };

    let request = TtsRequest {
        text: "Hello, this is a test of text-to-speech functionality.".to_string(),
        voice: Some(voice.to_string()),
        format: Some(format.to_string()),
        speed: Some(1.0),
        model: Some(model.to_string()),
        provider_options_map: Default::default(),
        extra_params: std::collections::HashMap::new(),
        http_config: None,
    };

    match client.tts(request).await {
        Ok(response) => {
            println!("    Text-to-Speech successful");
            println!("    Audio data size: {} bytes", response.audio_data.len());

            if response.audio_data.len() > 1000 {
                println!("    Audio data appears to be valid (size > 1KB)");
            } else {
                println!("    Audio data seems small, may not be valid");
            }

            if !response.format.is_empty() {
                println!("    Audio format: {}", response.format);
            }
        }
        Err(err) => {
            println!("    Text-to-Speech failed: {}", err);
            println!("    Note: TTS may not be available for this provider/model");
        }
    }
}

async fn test_speech_to_text<T: TranscriptionCapability>(_client: &T, provider_name: &str) {
    println!("  Testing Speech-to-Text for {}...", provider_name);
    println!("    STT test skipped - requires actual audio file");
    println!("    To test STT manually:");
    println!("       1. Record or obtain an audio file");
    println!("       2. Load it as bytes");
    println!("       3. Create SttRequest with the audio data");
    println!("       4. Call client.stt(request)");
}

fn test_audio_family_surface<T: LlmClient>(client: &T, provider_name: &str) {
    println!("  Testing audio-family surface for {}...", provider_name);
    println!(
        "    - speech capability: {}",
        client.as_speech_capability().is_some()
    );
    println!(
        "    - speech extras: {}",
        client.as_speech_extras().is_some()
    );
    println!(
        "    - transcription capability: {}",
        client.as_transcription_capability().is_some()
    );
    println!(
        "    - transcription extras: {}",
        client.as_transcription_extras().is_some()
    );
}

async fn test_openai_audio() {
    if env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping OpenAI audio tests: OPENAI_API_KEY not set");
        return;
    }

    println!("Testing OpenAI speech/transcription families...");
    let api_key = env::var("OPENAI_API_KEY").unwrap();

    let client = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("Failed to build OpenAI client");

    test_audio_family_surface(&client, "OpenAI");
    test_text_to_speech(&client, "OpenAI").await;
    test_speech_to_text(&client, "OpenAI").await;

    println!("OpenAI audio-family testing completed\n");
}

async fn test_audio_capability_availability() {
    println!("Testing audio-family availability across providers...");

    let providers = vec![("OpenAI", env::var("OPENAI_API_KEY").is_ok(), true)];

    println!("  Speech/Transcription family status:");
    for (provider, has_key, has_traits) in providers {
        let status = match (has_key, has_traits) {
            (true, true) => "Available",
            (true, false) => "API available but narrow family traits missing",
            (false, _) => "No API key",
        };
        println!("    {} - {}", provider, status);
    }

    println!("  Note: compatibility-only AudioCapability still exists");
    println!("     but recommended call sites should prefer narrow speech/transcription families");
    println!("Audio-family availability check completed\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_openai_audio_capability() {
        test_openai_audio().await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_all_provider_audio() {
        println!("Running audio-family tests for all available providers...\n");

        test_openai_audio().await;
        test_audio_capability_availability().await;

        println!("All provider audio-family testing completed!");
    }

    #[tokio::test]
    async fn test_audio_availability() {
        test_audio_capability_availability().await;
    }
}

#[cfg(test)]
mod manual_test_utils {
    use super::*;

    #[allow(dead_code)]
    pub async fn test_tts_with_text(
        text: &str,
        provider: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match provider {
            "openai" => {
                if let Ok(api_key) = env::var("OPENAI_API_KEY") {
                    let client = Siumai::builder()
                        .openai()
                        .api_key(api_key)
                        .model("gpt-4o-mini")
                        .build()
                        .await?;

                    let request = TtsRequest {
                        text: text.to_string(),
                        voice: Some("alloy".to_string()),
                        format: Some("mp3".to_string()),
                        speed: Some(1.0),
                        model: Some("tts-1".to_string()),
                        provider_options_map: Default::default(),
                        extra_params: std::collections::HashMap::new(),
                        http_config: None,
                    };

                    let response = client.tts(request).await?;
                    std::fs::write("manual_test_output.mp3", response.audio_data)?;
                    println!("Audio saved to manual_test_output.mp3");
                }
            }
            _ => {
                println!("Unknown provider: {}", provider);
            }
        }

        Ok(())
    }
}
