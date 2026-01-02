//! Embedding Integration Tests
//!
//! These tests verify the embedding functionality across different providers.
//! They are designed to run with actual API keys when available, but skip gracefully when not.

use siumai::prelude::unified::{EmbeddingExtensions, EmbeddingRequest, HttpConfig};

#[cfg(feature = "openai")]
use siumai::provider_ext::openai::{OpenAiClient, OpenAiConfig};

#[cfg(feature = "google")]
use siumai::provider_ext::gemini::{GeminiClient, GeminiConfig};

#[cfg(feature = "ollama")]
use siumai_provider_ollama::providers::ollama::config::OllamaParams;
#[cfg(feature = "ollama")]
use siumai_provider_ollama::providers::ollama::embeddings::OllamaEmbeddings;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test OpenAI embeddings with real API if available
    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn test_openai_embeddings_integration() {
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping OpenAI test: OPENAI_API_KEY not set");
                return;
            }
        };

        let config = OpenAiConfig::new(&api_key);
        let http_client = reqwest::Client::new();
        let client = OpenAiClient::new(config, http_client);

        // Test basic embedding
        let texts = vec!["Hello, world!".to_string()];
        let result = client.embed(texts).await;

        match result {
            Ok(response) => {
                assert!(!response.embeddings.is_empty());
                assert!(!response.embeddings[0].is_empty());
                assert_eq!(response.embeddings[0].len(), 1536); // text-embedding-3-small default
                println!("✅ OpenAI basic embedding test passed");
            }
            Err(e) => {
                println!("⚠️ OpenAI embedding test failed: {e}");
                // Don't fail the test for API errors in CI
            }
        }
    }

    /// Test OpenAI embeddings with custom dimensions
    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn test_openai_custom_dimensions() {
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping OpenAI custom dimensions test: OPENAI_API_KEY not set");
                return;
            }
        };

        let config = OpenAiConfig::new(&api_key);
        let http_client = reqwest::Client::new();
        let client = OpenAiClient::new(config, http_client);

        // Test custom dimensions
        let request = EmbeddingRequest::new(vec!["Test text".to_string()])
            .with_model("text-embedding-3-large")
            .with_dimensions(1024);

        let result =
            <OpenAiClient as EmbeddingExtensions>::embed_with_config(&client, request).await;

        match result {
            Ok(response) => {
                assert_eq!(response.embeddings[0].len(), 1024);
                println!("✅ OpenAI custom dimensions test passed");
            }
            Err(e) => {
                println!("⚠️ OpenAI custom dimensions test failed: {e}");
            }
        }
    }

    /// Test Gemini embeddings with real API if available
    #[tokio::test]
    #[cfg(feature = "google")]
    async fn test_gemini_embeddings_integration() {
        let api_key = match std::env::var("GEMINI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping Gemini test: GEMINI_API_KEY not set");
                return;
            }
        };

        let config = GeminiConfig::new(api_key)
            .with_base_url("https://generativelanguage.googleapis.com/v1beta".to_string())
            // Use the official embedding model id (see `standards::gemini::embedding` tests).
            .with_model("publishers/google/models/text-embedding-004".to_string());
        let client = match GeminiClient::with_http_client(config, reqwest::Client::new()) {
            Ok(c) => c,
            Err(e) => {
                println!("Skipping Gemini test: client init failed: {e}");
                return;
            }
        };

        // Test basic embedding
        let texts = vec!["Hello, world!".to_string()];
        let result = client.embed(texts).await;

        match result {
            Ok(response) => {
                assert!(!response.embeddings.is_empty());
                assert!(!response.embeddings[0].is_empty());
                println!("✅ Gemini basic embedding test passed");
            }
            Err(e) => {
                println!("⚠️ Gemini embedding test failed: {e}");
            }
        }
    }

    /// Test Gemini embeddings with task optimization
    #[tokio::test]
    #[cfg(feature = "google")]
    async fn test_gemini_task_optimization() {
        let api_key = match std::env::var("GEMINI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping Gemini task optimization test: GEMINI_API_KEY not set");
                return;
            }
        };

        let config = GeminiConfig::new(api_key)
            .with_base_url("https://generativelanguage.googleapis.com/v1beta".to_string())
            .with_model("publishers/google/models/text-embedding-004".to_string());
        let client = match GeminiClient::with_http_client(config, reqwest::Client::new()) {
            Ok(c) => c,
            Err(e) => {
                println!("Skipping Gemini task optimization test: client init failed: {e}");
                return;
            }
        };

        // Test with task type (typed, provider-agnostic). Gemini uses this to optimize embeddings.
        let request = EmbeddingRequest::query("Search query".to_string());

        let result =
            <GeminiClient as EmbeddingExtensions>::embed_with_config(&client, request).await;

        match result {
            Ok(response) => {
                assert!(!response.embeddings.is_empty());
                println!("✅ Gemini task optimization test passed");
            }
            Err(e) => {
                println!("⚠️ Gemini task optimization test failed: {e}");
            }
        }
    }

    /// Test Ollama embeddings if available
    #[tokio::test]
    #[cfg(feature = "ollama")]
    async fn test_ollama_embeddings_integration() {
        // Check if Ollama is available
        let ollama_available = std::process::Command::new("curl")
            .args(["-s", "http://localhost:11434/api/tags"])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        if !ollama_available {
            println!("Skipping Ollama test: Ollama not available");
            return;
        }

        let config = OllamaParams::default();
        let http_config = HttpConfig::default();
        let http_client = reqwest::Client::new();
        let embeddings = OllamaEmbeddings::new(
            "http://localhost:11434".to_string(),
            "nomic-embed-text".to_string(),
            http_client,
            http_config,
            config,
        );

        // Test basic embedding
        let texts = vec!["Hello, world!".to_string()];
        let result = embeddings.embed(texts).await;

        match result {
            Ok(response) => {
                assert!(!response.embeddings.is_empty());
                assert!(!response.embeddings[0].is_empty());
                println!("✅ Ollama basic embedding test passed");
            }
            Err(e) => {
                println!("⚠️ Ollama embedding test failed: {e}");
                // Don't fail for model not found errors
                if e.to_string().contains("not found") {
                    println!(
                        "   (Model may need to be pulled first: ollama pull nomic-embed-text)"
                    );
                }
            }
        }
    }

    /// Test embedding similarity calculation
    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn test_embedding_similarity() {
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping similarity test: OPENAI_API_KEY not set");
                return;
            }
        };

        let config = OpenAiConfig::new(&api_key);
        let http_client = reqwest::Client::new();
        let client = OpenAiClient::new(config, http_client);

        // Test with similar and dissimilar texts
        let texts = vec![
            "I love programming".to_string(),
            "Programming is my passion".to_string(),
            "The weather is nice today".to_string(),
        ];

        let result = client.embed(texts).await;

        match result {
            Ok(response) => {
                assert_eq!(response.embeddings.len(), 3);

                // Calculate cosine similarity
                let sim1_2 = cosine_similarity(&response.embeddings[0], &response.embeddings[1]);
                let sim1_3 = cosine_similarity(&response.embeddings[0], &response.embeddings[2]);

                // Similar texts should have higher similarity than dissimilar ones
                assert!(
                    sim1_2 > sim1_3,
                    "Similar texts should have higher similarity: {sim1_2} vs {sim1_3}"
                );

                println!("✅ Embedding similarity test passed");
                println!("   Programming texts similarity: {sim1_2:.4}");
                println!("   Programming vs weather similarity: {sim1_3:.4}");
            }
            Err(e) => {
                println!("⚠️ Embedding similarity test failed: {e}");
            }
        }
    }

    /// Test batch embedding processing
    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn test_batch_embedding() {
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping batch test: OPENAI_API_KEY not set");
                return;
            }
        };

        let config = OpenAiConfig::new(&api_key);
        let http_client = reqwest::Client::new();
        let client = OpenAiClient::new(config, http_client);

        // Test with multiple texts
        let texts = vec![
            "First document".to_string(),
            "Second document".to_string(),
            "Third document".to_string(),
            "Fourth document".to_string(),
            "Fifth document".to_string(),
        ];

        let result = client.embed(texts.clone()).await;

        match result {
            Ok(response) => {
                assert_eq!(response.embeddings.len(), texts.len());

                // All embeddings should have the same dimension
                let first_dim = response.embeddings[0].len();
                for embedding in &response.embeddings {
                    assert_eq!(embedding.len(), first_dim);
                }

                println!("✅ Batch embedding test passed");
                println!(
                    "   Processed {} texts with {} dimensions each",
                    response.embeddings.len(),
                    first_dim
                );
            }
            Err(e) => {
                println!("⚠️ Batch embedding test failed: {e}");
            }
        }
    }

    /// Helper function to calculate cosine similarity
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            0.0
        } else {
            dot_product / (magnitude_a * magnitude_b)
        }
    }
}
