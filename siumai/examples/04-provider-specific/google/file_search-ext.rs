//! Gemini File Search Stores (extension API + downcast)
//!
//! This example demonstrates:
//! - Building via the unified `Siumai` interface
//! - Using `downcast_client_cloned::<GeminiClient>()` to access provider-specific APIs
//! - Managing File Search Stores via `siumai::provider_ext::gemini::ext::file_search_stores`
//! - Querying via the provider-defined tool `siumai::hosted_tools::google::file_search`
//!
//! Run:
//! ```bash
//! export GOOGLE_API_KEY="your-key"
//! cargo run --example file_search-ext --features google
//! ```

use siumai::prelude::unified::*;
use siumai::user;

fn read_api_key() -> Result<String, Box<dyn std::error::Error>> {
    if let Ok(k) = std::env::var("GOOGLE_API_KEY") {
        return Ok(k);
    }
    if let Ok(k) = std::env::var("GEMINI_API_KEY") {
        return Ok(k);
    }
    Err("Missing GOOGLE_API_KEY or GEMINI_API_KEY".into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = read_api_key()?;

    let client = Siumai::builder()
        .gemini()
        .api_key(&api_key)
        .model("gemini-2.5-flash")
        .build()
        .await?;

    // Escape hatch: provider-specific Gemini client (cloned)
    let gemini = client
        .downcast_client_cloned::<siumai::provider_ext::gemini::GeminiClient>()
        .expect("this Siumai instance is backed by GeminiClient");

    let stores = siumai::provider_ext::gemini::ext::file_search_stores::stores(&gemini);

    println!("📦 Creating File Search Store...");
    let store = stores
        .create_store(Some("siumai-example-store".into()))
        .await?;
    println!("Created store: {}", store.name);

    // Upload small content with whitespace chunking
    let content = b"Robert Graves was an English poet, novelist, critic, and classicist.".to_vec();
    let cfg = siumai::provider_ext::gemini::ext::file_search_stores::FileSearchUploadConfig {
        chunking_config: Some(
            siumai::provider_ext::gemini::ext::file_search_stores::ChunkingConfig {
                white_space_config: Some(
                    siumai::provider_ext::gemini::ext::file_search_stores::WhiteSpaceChunkingConfig {
                        max_tokens_per_chunk: 200,
                        max_overlap_tokens: 20,
                    },
                ),
            },
        ),
    };

    println!("⬆️  Uploading file to store...");
    let op = stores
        .upload_to_file_search_store(
            store.name.clone(),
            content,
            "sample.txt".into(),
            Some("text/plain".into()),
            Some("sample".into()),
            Some(cfg),
        )
        .await?;

    println!("⏳ Waiting for operation {}...", op.name);
    let _done = stores.wait_operation(op.name.clone(), 120).await?;

    println!("💬 Querying with File Search tool...");
    let req = ChatRequest::builder()
        .message(user!("Who is Robert Graves?"))
        .model("gemini-2.5-flash")
        .tools(vec![
            siumai::hosted_tools::google::file_search()
                .with_file_search_store_names(vec![store.name.clone()])
                .build(),
        ])
        .build();
    let resp = client.chat_request(req).await?;
    println!("AI:\n{}\n", resp.content_text().unwrap_or_default());

    // Cleanup
    let cleanup = std::env::var("CLEANUP").unwrap_or_else(|_| "1".to_string()) != "0";
    if cleanup {
        println!("🧹 Deleting store {}", store.name);
        stores.delete_store(store.name).await?;
    } else {
        println!("CLEANUP=0, keeping store {}", store.name);
    }

    Ok(())
}
