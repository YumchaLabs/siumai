//! SiliconFlow Model Listing Example
//!
//! This example demonstrates how to list available models from SiliconFlow
//! and get detailed information about specific models.
//!
//! Before running, set your API key:
//! ```bash
//! export SILICONFLOW_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example siliconflow_model_listing
//! ```

use siumai::prelude::*;
use siumai::traits::ModelListingCapability;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 SiliconFlow Model Listing Example\n");

    // Get API key from environment
    let api_key = match std::env::var("SILICONFLOW_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("❌ SILICONFLOW_API_KEY environment variable not set");
            println!("   Please set it with: export SILICONFLOW_API_KEY=\"your-key\"");
            return Ok(());
        }
    };

    // Create SiliconFlow client
    let client = LlmBuilder::new()
        .siliconflow()
        .api_key(&api_key)
        .build()
        .await?;

    println!("📋 Listing all available models from SiliconFlow...\n");

    // List all models
    match client.list_models().await {
        Ok(models) => {
            println!("✅ Found {} models:\n", models.len());

            // Group models by capability
            let mut chat_models = Vec::new();
            let mut embedding_models = Vec::new();
            let mut rerank_models = Vec::new();
            let mut image_models = Vec::new();
            let mut thinking_models = Vec::new();

            for model in &models {
                if model.capabilities.contains(&"chat".to_string()) {
                    chat_models.push(model);
                }
                if model.capabilities.contains(&"embedding".to_string()) {
                    embedding_models.push(model);
                }
                if model.capabilities.contains(&"rerank".to_string()) {
                    rerank_models.push(model);
                }
                if model.capabilities.contains(&"image_generation".to_string()) {
                    image_models.push(model);
                }
                if model.capabilities.contains(&"thinking".to_string()) {
                    thinking_models.push(model);
                }
            }

            // Display chat models
            if !chat_models.is_empty() {
                println!("💬 Chat Models ({}):", chat_models.len());
                for model in &chat_models {
                    println!("   • {} - {}", model.id, model.description.as_deref().unwrap_or("No description"));
                }
                println!();
            }

            // Display thinking models
            if !thinking_models.is_empty() {
                println!("🧠 Thinking Models ({}):", thinking_models.len());
                for model in &thinking_models {
                    println!("   • {} - {}", model.id, model.description.as_deref().unwrap_or("No description"));
                }
                println!();
            }

            // Display embedding models
            if !embedding_models.is_empty() {
                println!("🔤 Embedding Models ({}):", embedding_models.len());
                for model in &embedding_models {
                    println!("   • {} - {}", model.id, model.description.as_deref().unwrap_or("No description"));
                }
                println!();
            }

            // Display rerank models
            if !rerank_models.is_empty() {
                println!("📊 Rerank Models ({}):", rerank_models.len());
                for model in &rerank_models {
                    println!("   • {} - {}", model.id, model.description.as_deref().unwrap_or("No description"));
                }
                println!();
            }

            // Display image generation models
            if !image_models.is_empty() {
                println!("🎨 Image Generation Models ({}):", image_models.len());
                for model in &image_models {
                    println!("   • {} - {}", model.id, model.description.as_deref().unwrap_or("No description"));
                }
                println!();
            }

            // Test getting specific model info
            if let Some(first_model) = models.first() {
                println!("🔍 Getting detailed info for model: {}", first_model.id);
                match client.get_model(first_model.id.clone()).await {
                    Ok(model_info) => {
                        println!("✅ Model Details:");
                        println!("   ID: {}", model_info.id);
                        println!("   Name: {}", model_info.name.as_deref().unwrap_or("N/A"));
                        println!("   Owner: {}", model_info.owned_by);
                        println!("   Capabilities: {:?}", model_info.capabilities);
                        if let Some(created) = model_info.created {
                            println!("   Created: {}", created);
                        }
                        if let Some(context_window) = model_info.context_window {
                            println!("   Context Window: {} tokens", context_window);
                        }
                        if let Some(max_output) = model_info.max_output_tokens {
                            println!("   Max Output: {} tokens", max_output);
                        }
                    }
                    Err(e) => {
                        println!("❌ Failed to get model details: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to list models: {}", e);
            println!("   This might be due to:");
            println!("   • Invalid API key");
            println!("   • Network connectivity issues");
            println!("   • SiliconFlow API being temporarily unavailable");
        }
    }

    println!("\n✅ Model listing example completed!");
    Ok(())
}

/*
🎯 Key Model Listing Concepts:

Model Categories:
- Chat Models: For conversational AI (deepseek-chat, qwen-turbo, etc.)
- Thinking Models: For reasoning tasks (deepseek-reasoner, etc.)
- Embedding Models: For text embeddings (bge-large-en, text-embedding-ada-002, etc.)
- Rerank Models: For search result reranking (bge-reranker-v2-m3, etc.)
- Image Models: For image generation (flux-1-schnell, stable-diffusion-3-5-large, etc.)

Model Information:
- ID: Unique model identifier
- Name: Human-readable model name
- Description: Model description and capabilities
- Owner: Model provider/organization
- Capabilities: List of supported features
- Context Window: Maximum input tokens (if available)
- Max Output: Maximum output tokens (if available)
- Costs: Input/output pricing (if available)

Usage Tips:
1. Always check model capabilities before using
2. Some models may have usage restrictions
3. Pricing information may not be available via API
4. Model availability can change over time
5. Use appropriate models for specific tasks

Next Steps:
- ../thinking_models.rs: Use thinking-capable models
- ../embedding_models.rs: Use embedding models
- ../image_generation.rs: Use image generation models
*/
