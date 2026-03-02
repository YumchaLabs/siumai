#![allow(deprecated)]
//! Vision Capability Integration Tests
//!
//! These tests verify vision and multimodal functionality across supported providers.
//! They are ignored by default to prevent accidental API usage during normal testing.
//!
//! ## Running Tests
//!
//! ```bash
//! # Test specific provider vision capabilities
//! export OPENAI_API_KEY="your-key"
//! cargo test test_openai_vision -- --ignored
//!
//! export ANTHROPIC_API_KEY="your-key"
//! cargo test test_anthropic_vision -- --ignored
//!
//! # Test all available providers
//! cargo test test_all_provider_vision -- --ignored
//! ```

use siumai::prelude::*;
use std::env;

/// Test image analysis with URL
async fn test_image_analysis_url<T: ChatCapability>(client: &T, provider_name: &str) {
    println!("  🖼️ Testing image analysis (URL) for {}...", provider_name);

    // Use a simple, publicly accessible test image
    let image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg";

    let message = ChatMessage::user("What do you see in this image? Describe it briefly.")
        .with_image(image_url.to_string(), Some("high".to_string()))
        .build();

    let messages = vec![message];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            if !content.is_empty() {
                println!("    ✅ Image analysis successful");
                println!("    📝 Description: {}", content.trim());

                // Check if the response seems to contain actual image analysis
                let content_lower = content.to_lowercase();
                if content_lower.contains("image")
                    || content_lower.contains("see")
                    || content_lower.contains("picture")
                    || content_lower.contains("photo")
                {
                    println!("    🎯 Response appears to contain image analysis");
                } else {
                    println!("    ⚠️ Response may not contain actual image analysis");
                }
            } else {
                println!("    ⚠️ Empty response received");
            }

            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!("    ⚠️ Image analysis failed: {}", e);
            println!("    💡 Note: Vision capability may not be available for this model/provider");
        }
    }
}

/// Test multimodal conversation (text + image)
async fn test_multimodal_conversation<T: ChatCapability>(client: &T, provider_name: &str) {
    println!(
        "  💬 Testing multimodal conversation for {}...",
        provider_name
    );

    // Simple geometric image for testing
    let image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Square_-_black_simple.svg/240px-Square_-_black_simple.svg.png";

    let messages = vec![
        system!(
            "You are a helpful assistant that can analyze images and answer questions about them."
        ),
        ChatMessage::user(
            "I'm going to show you a shape. Please tell me what shape it is and what color it is.",
        )
        .with_image(image_url.to_string(), Some("high".to_string()))
        .build(),
    ];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            if !content.is_empty() {
                println!("    ✅ Multimodal conversation successful");
                println!("    📝 Response: {}", content.trim());

                // Check for expected content
                let content_lower = content.to_lowercase();
                if content_lower.contains("square") || content_lower.contains("rectangle") {
                    println!("    🎯 Correctly identified shape");
                }
                if content_lower.contains("black") {
                    println!("    🎯 Correctly identified color");
                }
            }

            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!("    ⚠️ Multimodal conversation failed: {}", e);
        }
    }
}

/// Test multiple images in one message
async fn test_multiple_images<T: ChatCapability>(client: &T, provider_name: &str) {
    println!("  🖼️🖼️ Testing multiple images for {}...", provider_name);

    // Two simple test images
    let image1_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Square_-_black_simple.svg/240px-Square_-_black_simple.svg.png";
    let image2_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Circle_-_black_simple.svg/240px-Circle_-_black_simple.svg.png";

    let message = ChatMessage::user("I'm showing you two shapes. Please describe each one.")
        .with_image(image1_url.to_string(), Some("high".to_string()))
        .with_image(image2_url.to_string(), Some("high".to_string()))
        .build();

    let messages = vec![message];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            if !content.is_empty() {
                println!("    ✅ Multiple images analysis successful");
                println!("    📝 Response: {}", content.trim());

                // Check if both shapes are mentioned
                let content_lower = content.to_lowercase();
                let mentions_square =
                    content_lower.contains("square") || content_lower.contains("rectangle");
                let mentions_circle =
                    content_lower.contains("circle") || content_lower.contains("round");

                if mentions_square && mentions_circle {
                    println!("    🎯 Both shapes correctly identified");
                } else if mentions_square || mentions_circle {
                    println!("    🎯 At least one shape identified");
                } else {
                    println!("    ⚠️ Shapes may not have been correctly identified");
                }
            }

            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!("    ⚠️ Multiple images analysis failed: {}", e);
            println!("    💡 Note: Some providers may not support multiple images in one message");
        }
    }
}

/// Test vision with reasoning
async fn test_vision_reasoning<T: ChatCapability>(client: &T, provider_name: &str) {
    println!(
        "  🧠 Testing vision with reasoning for {}...",
        provider_name
    );

    // Image with some complexity for reasoning
    let image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg";

    let message = ChatMessage::user("Look at this image and tell me: 1) What type of environment is this? 2) What time of day might it be? 3) What activities could someone do here? Please explain your reasoning.")
        .with_image(image_url.to_string(), Some("high".to_string()))
        .build();

    let messages = vec![message];

    match client.chat(messages).await {
        Ok(response) => {
            let content = response.content_text().unwrap_or_default();
            if !content.is_empty() {
                println!("    ✅ Vision reasoning successful");
                println!("    📝 Analysis: {}", content.trim());

                // Check for reasoning indicators
                let content_lower = content.to_lowercase();
                let has_reasoning = content_lower.contains("because")
                    || content_lower.contains("since")
                    || content_lower.contains("due to")
                    || content_lower.contains("appears")
                    || content_lower.contains("suggests");

                if has_reasoning {
                    println!("    🧠 Response contains reasoning elements");
                }
            }

            if let Some(usage) = response.usage {
                println!(
                    "    📊 Usage: {} prompt + {} completion = {} total tokens",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!("    ⚠️ Vision reasoning failed: {}", e);
        }
    }
}

/// Generic provider vision testing
async fn test_provider_vision(provider_name: &str, api_key_env: &str, model: &str) {
    if env::var(api_key_env).is_err() {
        println!(
            "⏭️ Skipping {} vision tests: {} not set",
            provider_name, api_key_env
        );
        return;
    }

    println!("👁️ Testing {} vision capabilities...", provider_name);

    match provider_name {
        "OpenAI" => {
            let api_key = env::var(api_key_env).unwrap();
            let mut builder = Siumai::builder().openai().api_key(api_key).model(model);

            if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            match builder.build().await {
                Ok(client) => {
                    test_image_analysis_url(&client, provider_name).await;
                    test_multimodal_conversation(&client, provider_name).await;
                    test_multiple_images(&client, provider_name).await;
                    test_vision_reasoning(&client, provider_name).await;
                }
                Err(e) => {
                    println!("❌ Failed to build OpenAI client: {}", e);
                    return;
                }
            }
        }
        "Anthropic" => {
            let api_key = env::var(api_key_env).unwrap();
            let mut builder = Siumai::builder().anthropic().api_key(api_key).model(model);

            if let Ok(base_url) = env::var("ANTHROPIC_BASE_URL") {
                builder = builder.base_url(base_url);
            }

            match builder.build().await {
                Ok(client) => {
                    test_image_analysis_url(&client, provider_name).await;
                    test_multimodal_conversation(&client, provider_name).await;
                    test_multiple_images(&client, provider_name).await;
                    test_vision_reasoning(&client, provider_name).await;
                }
                Err(e) => {
                    println!("❌ Failed to build Anthropic client: {}", e);
                    return;
                }
            }
        }
        "Gemini" => {
            let api_key = env::var(api_key_env).unwrap();
            match Siumai::builder()
                .gemini()
                .api_key(api_key)
                .model(model)
                .build()
                .await
            {
                Ok(client) => {
                    test_image_analysis_url(&client, provider_name).await;
                    test_multimodal_conversation(&client, provider_name).await;
                    test_multiple_images(&client, provider_name).await;
                    test_vision_reasoning(&client, provider_name).await;
                }
                Err(e) => {
                    println!("❌ Failed to build Gemini client: {}", e);
                    return;
                }
            }
        }
        "xAI" => {
            let api_key = env::var(api_key_env).unwrap();
            match Siumai::builder()
                .xai()
                .api_key(api_key)
                .model(model)
                .build()
                .await
            {
                Ok(client) => {
                    test_image_analysis_url(&client, provider_name).await;
                    test_multimodal_conversation(&client, provider_name).await;
                    test_multiple_images(&client, provider_name).await;
                    test_vision_reasoning(&client, provider_name).await;
                }
                Err(e) => {
                    println!("❌ Failed to build xAI client: {}", e);
                    return;
                }
            }
        }
        _ => {
            println!("❌ Unknown provider: {}", provider_name);
            return;
        }
    }

    println!("✅ {} vision testing completed\n", provider_name);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_openai_vision() {
        test_provider_vision("OpenAI", "OPENAI_API_KEY", "gpt-4o").await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_anthropic_vision() {
        test_provider_vision(
            "Anthropic",
            "ANTHROPIC_API_KEY",
            "claude-3-5-sonnet-20241022",
        )
        .await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_gemini_vision() {
        test_provider_vision("Gemini", "GEMINI_API_KEY", "gemini-2.5-pro").await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_xai_vision() {
        test_provider_vision("xAI", "XAI_API_KEY", "grok-2-vision-1212").await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_all_provider_vision() {
        println!("🚀 Running vision capability tests for all available providers...\n");

        let providers = vec![
            ("OpenAI", "OPENAI_API_KEY", "gpt-4o"),
            (
                "Anthropic",
                "ANTHROPIC_API_KEY",
                "claude-3-5-sonnet-20241022",
            ),
            ("Gemini", "GEMINI_API_KEY", "gemini-2.5-pro"),
            ("xAI", "XAI_API_KEY", "grok-2-vision-1212"),
        ];

        for (provider_name, api_key_env, model) in providers {
            test_provider_vision(provider_name, api_key_env, model).await;
        }

        println!("🎉 All provider vision testing completed!");
    }
}
