//! Audio Capability Integration Tests
//!
//! These tests verify audio functionality (TTS and STT) across supported providers.
//! They are ignored by default to prevent accidental API usage during normal testing.
//!
//! ## Running Tests
//!
//! ```bash
//! # Test specific provider audio capabilities
//! export OPENAI_API_KEY="your-key"
//! cargo test test_openai_audio -- --ignored
//!
//! export GROQ_API_KEY="your-key"
//! cargo test test_groq_audio -- --ignored
//!
//! # Test all available providers
//! cargo test test_all_provider_audio -- --ignored
//! ```

use siumai::prelude::*;
use siumai::traits::AudioCapability;
use siumai::types::TtsRequest;
use std::env;

/// Test Text-to-Speech functionality
async fn test_text_to_speech<T: AudioCapability>(client: &T, provider_name: &str) {
    println!("  🔊 Testing Text-to-Speech for {}...", provider_name);

    let (voice, format, model) = match provider_name {
        "OpenAI" => ("alloy", "mp3", "tts-1"),
        "Groq" => ("Fritz-PlayAI", "wav", "playai-tts"),
        _ => ("alloy", "mp3", "default"),
    };

    let request = TtsRequest {
        text: "Hello, this is a test of text-to-speech functionality.".to_string(),
        voice: Some(voice.to_string()),
        format: Some(format.to_string()),
        speed: Some(1.0),
        model: Some(model.to_string()),
        provider_options: Default::default(),
        extra_params: std::collections::HashMap::new(),
        http_config: None,
    };

    match client.text_to_speech(request).await {
        Ok(response) => {
            println!("    ✅ Text-to-Speech successful");
            println!(
                "    📊 Audio data size: {} bytes",
                response.audio_data.len()
            );

            // Verify we got actual audio data
            if response.audio_data.len() > 1000 {
                println!("    🎵 Audio data appears to be valid (size > 1KB)");
            } else {
                println!("    ⚠️ Audio data seems small, may not be valid");
            }

            // Check format if provided
            if !response.format.is_empty() {
                println!("    🎵 Audio format: {}", response.format);
            }

            // Note: In a real test, you might want to save the audio file
            // std::fs::write("test_output.mp3", response.audio_data)?;
        }
        Err(e) => {
            println!("    ⚠️ Text-to-Speech failed: {}", e);
            println!("    💡 Note: TTS may not be available for this provider/model");
        }
    }
}

/// Test Speech-to-Text functionality
async fn test_speech_to_text<T: AudioCapability>(_client: &T, provider_name: &str) {
    println!("  🎤 Testing Speech-to-Text for {}...", provider_name);

    // Note: For a real test, you would need actual audio data
    // This is a mock test since we don't have audio files in the test suite
    println!("    ⚠️ STT test skipped - requires actual audio file");
    println!("    💡 To test STT manually:");
    println!("       1. Record or obtain an audio file");
    println!("       2. Load it as bytes");
    println!("       3. Create SttRequest with the audio data");
    println!("       4. Call client.speech_to_text(request)");

    // Example of how STT would be tested with real audio data:
    /*
    let audio_data = std::fs::read("test_audio.mp3")?;
    let request = SttRequest {
        audio_data,
        format: Some("mp3".to_string()),
        language: Some("en".to_string()),
        model: Some(match provider_name {
            "OpenAI" => "whisper-1".to_string(),
            "Groq" => "whisper-large-v3".to_string(),
            _ => "default".to_string(),
        }),
        extra_params: std::collections::HashMap::new(),
    };

    match client.speech_to_text(request).await {
        Ok(response) => {
            println!("    ✅ Speech-to-Text successful");
            println!("    📝 Transcription: {}", response.text);
        }
        Err(e) => {
            println!("    ⚠️ Speech-to-Text failed: {}", e);
        }
    }
    */
}

/// Test audio features discovery
async fn test_audio_features<T: AudioCapability>(client: &T, provider_name: &str) {
    println!("  🔍 Testing audio features for {}...", provider_name);

    let features = client.supported_features();
    println!("    📋 Supported audio features:");

    for feature in features {
        println!("      - {:?}", feature);
    }

    if features.is_empty() {
        println!("    ⚠️ No audio features reported");
    } else {
        println!("    ✅ {} audio features available", features.len());
    }
}

/// Test OpenAI audio capabilities
async fn test_openai_audio() {
    if env::var("OPENAI_API_KEY").is_err() {
        println!("⏭️ Skipping OpenAI audio tests: OPENAI_API_KEY not set");
        return;
    }

    println!("🔊 Testing OpenAI audio capabilities...");
    let api_key = env::var("OPENAI_API_KEY").unwrap();

    let client = LlmBuilder::new()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("Failed to build OpenAI client");

    test_audio_features(&client, "OpenAI").await;
    test_text_to_speech(&client, "OpenAI").await;
    test_speech_to_text(&client, "OpenAI").await;

    println!("✅ OpenAI audio testing completed\n");
}

/// Test Groq audio capabilities
async fn test_groq_audio() {
    if env::var("GROQ_API_KEY").is_err() {
        println!("⏭️ Skipping Groq audio tests: GROQ_API_KEY not set");
        return;
    }

    println!("⚡ Testing Groq audio capabilities...");
    let api_key = env::var("GROQ_API_KEY").unwrap();

    let client = LlmBuilder::new()
        .groq()
        .api_key(api_key)
        // Chat model is irrelevant for TTS; the request's `model` controls `/audio/speech`.
        .model(models::groq::LLAMA_3_1_8B_INSTANT)
        .build()
        .await
        .expect("Failed to build Groq client");

    test_audio_features(&client, "Groq").await;
    test_text_to_speech(&client, "Groq").await;
    test_speech_to_text(&client, "Groq").await;

    println!("✅ Groq audio testing completed\n");
}

/// Test audio capability availability across providers
async fn test_audio_capability_availability() {
    println!("📊 Testing audio capability availability across providers...");

    // Check which providers claim to support audio
    let providers_with_audio = vec![
        ("OpenAI", env::var("OPENAI_API_KEY").is_ok(), true),
        ("Groq", env::var("GROQ_API_KEY").is_ok(), true),
    ];

    println!("  📋 Audio capability status:");
    for (provider, has_key, has_trait) in providers_with_audio {
        let status = match (has_key, has_trait) {
            (true, true) => "✅ Available",
            (true, false) => "⚠️ API available but AudioCapability trait not implemented",
            (false, _) => "❌ No API key",
        };
        println!("    {} - {}", provider, status);
    }

    println!("  💡 Note: Other providers (Anthropic, Gemini, etc.) may support audio");
    println!("     through their native APIs but not through Siumai's AudioCapability trait yet");

    println!("✅ Audio capability availability check completed\n");
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
    async fn test_groq_audio_capability() {
        test_groq_audio().await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_all_provider_audio() {
        println!("🚀 Running audio capability tests for all available providers...\n");

        test_openai_audio().await;
        test_groq_audio().await;
        test_audio_capability_availability().await;

        println!("🎉 All provider audio testing completed!");
    }

    #[tokio::test]
    async fn test_audio_availability() {
        test_audio_capability_availability().await;
    }
}

/// Additional test utilities for manual testing
#[cfg(test)]
mod manual_test_utils {
    use super::*;

    /// Helper function to test TTS with custom text
    /// This can be used for manual testing with different text inputs
    #[allow(dead_code)]
    pub async fn test_tts_with_text(
        text: &str,
        provider: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match provider {
            "openai" => {
                if let Ok(api_key) = env::var("OPENAI_API_KEY") {
                    let client = LlmBuilder::new()
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
                        provider_options: Default::default(),
                        extra_params: std::collections::HashMap::new(),
                        http_config: None,
                    };

                    let response = client.text_to_speech(request).await?;
                    std::fs::write("manual_test_output.mp3", response.audio_data)?;
                    println!("Audio saved to manual_test_output.mp3");
                }
            }
            "groq" => {
                if let Ok(api_key) = env::var("GROQ_API_KEY") {
                    let client = LlmBuilder::new()
                        .groq()
                        .api_key(api_key)
                        .model(models::groq::LLAMA_3_1_8B_INSTANT)
                        .build()
                        .await?;

                    let request = TtsRequest {
                        text: text.to_string(),
                        voice: Some("Fritz-PlayAI".to_string()),
                        format: Some("wav".to_string()),
                        speed: Some(1.0),
                        model: Some("playai-tts".to_string()),
                        provider_options: Default::default(),
                        extra_params: std::collections::HashMap::new(),
                        http_config: None,
                    };

                    let response = client.text_to_speech(request).await?;
                    std::fs::write("manual_test_output.wav", response.audio_data)?;
                    println!("Audio saved to manual_test_output.wav");
                }
            }
            _ => {
                println!("Unknown provider: {}", provider);
            }
        }

        Ok(())
    }
}
