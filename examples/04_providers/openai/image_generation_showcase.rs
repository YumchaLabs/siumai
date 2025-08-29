//! Image Generation Showcase
//!
//! This example demonstrates image generation capabilities across different providers,
//! including OpenAI DALL-E and SiliconFlow models.

use siumai::prelude::*;
use siumai::providers::openai_compatible::siliconflow;
use siumai::traits::ImageGenerationCapability;
use siumai::types::ImageGenerationRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for better debugging
    tracing_subscriber::fmt::init();

    println!("🎨 Image Generation Showcase");
    println!("============================\n");

    // Example 1: OpenAI DALL-E Image Generation
    if let Ok(openai_key) = std::env::var("OPENAI_API_KEY") {
        println!("🖼️  OpenAI DALL-E Image Generation");
        println!("-----------------------------------");

        let client = LlmBuilder::new()
            .openai()
            .api_key(&openai_key)
            .build()
            .await?;

        // Basic image generation
        let request = ImageGenerationRequest {
            prompt: "A futuristic city with flying cars at sunset, digital art style".to_string(),
            size: Some("1024x1024".to_string()),
            count: 1,
            model: Some("dall-e-3".to_string()),
            quality: Some("hd".to_string()),
            style: Some("vivid".to_string()),
            ..Default::default()
        };

        match client.generate_images(request).await {
            Ok(response) => {
                println!("✅ Generated {} image(s)", response.images.len());
                for (i, image) in response.images.iter().enumerate() {
                    if let Some(url) = &image.url {
                        println!("  {}. Image URL: {}", i + 1, url);
                    }
                    if let Some(revised_prompt) = &image.revised_prompt {
                        println!("     Revised prompt: {}", revised_prompt);
                    }
                }
            }
            Err(e) => println!("❌ Error: {}", e),
        }

        // Test capabilities
        if let Some(image_cap) = client.as_image_generation_capability() {
            println!("\n📋 OpenAI Capabilities:");
            println!("  • Supported sizes: {:?}", image_cap.get_supported_sizes());
            println!(
                "  • Supported formats: {:?}",
                image_cap.get_supported_formats()
            );
            println!("  • Image editing: {}", image_cap.supports_image_editing());
            println!(
                "  • Image variations: {}",
                image_cap.supports_image_variations()
            );
        }

        println!();
    } else {
        println!("⚠️  Skipping OpenAI example (OPENAI_API_KEY not set)\n");
    }

    // Example 2: SiliconFlow Image Generation
    if let Ok(siliconflow_key) = std::env::var("SILICONFLOW_API_KEY") {
        println!("🌟 SiliconFlow Image Generation");
        println!("-------------------------------");

        let client = LlmBuilder::new()
            .siliconflow()
            .api_key(&siliconflow_key)
            .build()
            .await?;

        // SiliconFlow image generation with negative prompt
        let request = ImageGenerationRequest {
            prompt: "A beautiful landscape with mountains and a lake, photorealistic".to_string(),
            negative_prompt: Some("blurry, low quality, distorted".to_string()),
            size: Some("1024x1024".to_string()),
            count: 1,
            model: Some(siliconflow::KOLORS.to_string()),
            steps: Some(20),
            guidance_scale: Some(7.5),
            seed: Some(42),
            ..Default::default()
        };

        match client.generate_images(request).await {
            Ok(response) => {
                println!("✅ Generated {} image(s)", response.images.len());
                for (i, image) in response.images.iter().enumerate() {
                    if let Some(url) = &image.url {
                        println!("  {}. Image URL: {}", i + 1, url);
                    }
                }

                // Print metadata
                if !response.metadata.is_empty() {
                    println!("  📊 Metadata:");
                    for (key, value) in &response.metadata {
                        println!("     {}: {}", key, value);
                    }
                }
            }
            Err(e) => println!("❌ Error: {}", e),
        }

        // Test different models
        println!("\n🎯 Testing Different SiliconFlow Models:");

        let models = vec![
            (siliconflow::KOLORS, "Kolors (Chinese model)"),
            (
                siliconflow::FLUX_1_SCHNELL,
                "FLUX.1 Schnell (Fast generation)",
            ),
            (
                siliconflow::STABLE_DIFFUSION_3_5_LARGE,
                "Stable Diffusion 3.5 Large",
            ),
        ];

        for (model, description) in models {
            println!("  • {}: {}", model, description);
        }

        // Test capabilities
        if let Some(image_cap) = client.as_image_generation_capability() {
            println!("\n📋 SiliconFlow Capabilities:");
            println!("  • Supported sizes: {:?}", image_cap.get_supported_sizes());
            println!(
                "  • Supported formats: {:?}",
                image_cap.get_supported_formats()
            );
            println!("  • Image editing: {}", image_cap.supports_image_editing());
            println!(
                "  • Image variations: {}",
                image_cap.supports_image_variations()
            );
        }

        println!();
    } else {
        println!("⚠️  Skipping SiliconFlow example (SILICONFLOW_API_KEY not set)\n");
    }

    // Example 3: Convenience Methods
    println!("🚀 Convenience Methods");
    println!("----------------------");

    if let Ok(openai_key) = std::env::var("OPENAI_API_KEY") {
        let client = LlmBuilder::new()
            .openai()
            .api_key(&openai_key)
            .build()
            .await?;

        if let Some(image_cap) = client.as_image_generation_capability() {
            // Simple image generation
            match image_cap
                .generate_image(
                    "A cute robot reading a book".to_string(),
                    Some("512x512".to_string()),
                    Some(1),
                )
                .await
            {
                Ok(urls) => {
                    println!("✅ Simple generation successful:");
                    for url in urls {
                        println!("  • {}", url);
                    }
                }
                Err(e) => println!("❌ Error: {}", e),
            }
        }
    }

    // Example 4: Error Handling and Validation
    println!("\n🛡️  Error Handling Examples");
    println!("---------------------------");

    if let Ok(openai_key) = std::env::var("OPENAI_API_KEY") {
        let client = LlmBuilder::new()
            .openai()
            .api_key(&openai_key)
            .build()
            .await?;

        // Test invalid size
        let invalid_request = ImageGenerationRequest {
            prompt: "Test image".to_string(),
            size: Some("999x999".to_string()), // Invalid size
            count: 1,
            model: Some("dall-e-3".to_string()),
            ..Default::default()
        };

        match client.generate_images(invalid_request).await {
            Ok(_) => println!("Unexpected success with invalid size"),
            Err(e) => println!("✅ Correctly caught error: {}", e),
        }

        // Test too many images for DALL-E 3
        let too_many_request = ImageGenerationRequest {
            prompt: "Test image".to_string(),
            size: Some("1024x1024".to_string()),
            count: 5, // DALL-E 3 only supports 1 image
            model: Some("dall-e-3".to_string()),
            ..Default::default()
        };

        match client.generate_images(too_many_request).await {
            Ok(_) => println!("Unexpected success with too many images"),
            Err(e) => println!("✅ Correctly caught error: {}", e),
        }
    }

    println!("\n🎉 Image Generation Showcase Complete!");
    println!("\n💡 Tips:");
    println!("  • Use descriptive prompts for better results");
    println!("  • Experiment with different models and parameters");
    println!("  • Consider using negative prompts with SiliconFlow");
    println!("  • Check provider capabilities before making requests");

    Ok(())
}
