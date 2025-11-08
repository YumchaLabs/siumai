//! Google Gemini File Search Stores + Query Example
//!
//! Demonstrates provider-specific File Search store management and querying
//! via the File Search tool in a chat request.
//!
//! - Creates a File Search Store
//! - Uploads a small text via `uploadToFileSearchStore` with chunking config
//! - Waits for the long-running operation
//! - Queries with File Search by injecting the store via `GeminiOptions`
//! - Optionally deletes the store (set CLEANUP=0 to skip)
//!
//! Run:
//! ```bash
//! # Set your key (either works)
//! export GOOGLE_API_KEY="your-key"
//! # or
//! export GEMINI_API_KEY="your-key"
//!
//! cargo run --example file_search --features google
//! ```

use siumai::prelude::*;
use siumai::providers::gemini::{
    ChunkingConfig, FileSearchUploadConfig, GeminiClient, WhiteSpaceChunkingConfig,
};

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

    // Provider-specific client (Gemini-only features)
    // Note: File Search is only supported by gemini-2.5-pro and gemini-2.5-flash
    let client: GeminiClient = Provider::gemini()
        .api_key(&api_key)
        .model("gemini-2.5-flash")
        .build()
        .await?;

    println!("📦 Creating File Search Store...");
    let store = client
        .file_search_stores()
        .create_store(Some("siumai-example-store".into()))
        .await?;
    println!("Created store: {}", store.name);

    // Upload small content with whitespace chunking
    let content = b"Robert Graves was an English poet, novelist, critic, and classicist.".to_vec();
    let cfg = FileSearchUploadConfig {
        chunking_config: Some(ChunkingConfig {
            white_space_config: Some(WhiteSpaceChunkingConfig {
                max_tokens_per_chunk: 200,
                max_overlap_tokens: 20,
            }),
        }),
    };

    println!("⬆️  Uploading file to store (with chunking config)...");
    let op = client
        .file_search_stores()
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
    let _done = client
        .file_search_stores()
        .wait_operation(op.name.clone(), 120)
        .await?;

    println!("💬 Querying with File Search tool...");
    let req = ChatRequest::builder()
        .message(ChatMessage::user("Who is Robert Graves?").build())
        .model("gemini-2.5-flash")
        .gemini_options(GeminiOptions::new().with_file_search_store_names(vec![store.name.clone()]))
        .build();

    let resp = client.chat_request(req).await?;
    println!(
        "AI (grounded):\n{}\n",
        resp.content_text().unwrap_or_default()
    );

    // Cleanup: delete the store
    println!("🧹 Deleting store {}", store.name);
    client.file_search_stores().delete_store(store.name).await?;

    Ok(())
}
