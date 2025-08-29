//! SiliconFlow Provider Example
//!
//! This example demonstrates how to use SiliconFlow's comprehensive AI capabilities
//! including chat, embeddings, reranking, and image generation.

use siumai::prelude::*;
use siumai::providers::openai_compatible::siliconflow;
use siumai::traits::{ChatCapability, EmbeddingCapability, RerankCapability};
use siumai::types::{EmbeddingRequest, RerankRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for better debugging
    tracing_subscriber::fmt::init();

    // Get API key from environment
    let api_key = std::env::var("SILICONFLOW_API_KEY")
        .expect("SILICONFLOW_API_KEY environment variable not set");

    println!("🚀 SiliconFlow Provider Showcase");
    println!("=================================\n");

    // Create SiliconFlow client
    let client = LlmBuilder::new()
        .siliconflow()
        .api_key(&api_key)
        .build()
        .await?;

    // Example 1: Chat with DeepSeek models
    chat_example(&client).await?;

    // Example 2: Text embeddings
    embedding_example(&client).await?;

    // Example 3: Document reranking (SiliconFlow's unique feature)
    rerank_example(&client).await?;

    // Example 4: RAG workflow combining all capabilities
    rag_workflow_example(&client).await?;

    Ok(())
}

/// Demonstrate chat capabilities with various SiliconFlow models
async fn chat_example(client: &impl ChatCapability) -> Result<(), Box<dyn std::error::Error>> {
    println!("💬 Chat Example with SiliconFlow Models");
    println!("----------------------------------------");

    // Test with DeepSeek Chat
    println!("Using DeepSeek Chat model:");
    let messages = vec![
        system!("You are a helpful AI assistant specialized in explaining complex topics simply."),
        user!("Explain quantum computing in simple terms.")
    ];

    let response = client.chat(messages).await?;
    println!("Response: {}\n", response.content);

    // Test with Qwen model
    println!("Using Qwen 2.5 model:");
    let messages = vec![
        user!("Write a haiku about artificial intelligence.")
    ];

    // Note: You would need to create a new client with different model for this
    // This is just to show the model variety available
    println!("Available chat models:");
    for model in siliconflow::all_chat_models() {
        println!("  - {}", model);
    }
    println!();

    Ok(())
}

/// Demonstrate embedding capabilities
async fn embedding_example(client: &impl EmbeddingCapability) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Embedding Example");
    println!("--------------------");

    let texts = vec![
        "Artificial intelligence is transforming technology.".to_string(),
        "Machine learning algorithms can recognize patterns.".to_string(),
        "Deep learning uses neural networks for complex tasks.".to_string(),
    ];

    println!("Generating embeddings for {} texts...", texts.len());

    let request = EmbeddingRequest::new(
        siliconflow::BGE_M3.to_string(),
        texts.clone(),
    );

    let response = client.embed(request).await?;

    println!("Generated {} embeddings", response.data.len());
    println!("Embedding dimensions: {}", response.data[0].embedding.len());
    println!("Token usage: {}", response.usage.total_tokens);

    // Show similarity between first two embeddings
    if response.data.len() >= 2 {
        let similarity = cosine_similarity(&response.data[0].embedding, &response.data[1].embedding);
        println!("Similarity between first two texts: {:.4}", similarity);
    }

    println!();
    Ok(())
}

/// Demonstrate reranking capabilities (SiliconFlow's specialty)
async fn rerank_example(client: &impl RerankCapability) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Document Reranking Example");
    println!("------------------------------");

    let query = "machine learning algorithms for natural language processing";
    let documents = vec![
        "BERT is a transformer-based model for natural language understanding.".to_string(),
        "Convolutional neural networks are primarily used for image processing.".to_string(),
        "GPT models use transformer architecture for text generation.".to_string(),
        "Random forests are ensemble methods for classification tasks.".to_string(),
        "LSTM networks are effective for sequential data processing.".to_string(),
        "Support vector machines work well for text classification.".to_string(),
    ];

    println!("Query: {}", query);
    println!("Reranking {} documents...\n", documents.len());

    let request = RerankRequest::new(
        siliconflow::BGE_RERANKER_V2_M3.to_string(),
        query.to_string(),
        documents,
    )
    .with_top_n(4)
    .with_return_documents(true);

    let response = client.rerank(request).await?;

    println!("🏆 Top Relevant Documents:");
    for (i, result) in response.results.iter().enumerate() {
        println!(
            "  {}. [Score: {:.4}] {}",
            i + 1,
            result.relevance_score,
            result.document.as_ref().map(|d| &d.text).unwrap_or("N/A")
        );
    }

    println!("\n📊 Token Usage:");
    println!("  Input tokens: {}", response.tokens.input_tokens);
    println!("  Output tokens: {}", response.tokens.output_tokens);
    println!();

    Ok(())
}

/// Demonstrate a complete RAG workflow using SiliconFlow
async fn rag_workflow_example(client: &(impl ChatCapability + EmbeddingCapability + RerankCapability)) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Complete RAG Workflow Example");
    println!("---------------------------------");

    let user_question = "How do transformers work in natural language processing?";
    
    // Simulate a knowledge base
    let knowledge_base = vec![
        "Transformers use self-attention mechanisms to process sequences of data in parallel.".to_string(),
        "The attention mechanism allows models to focus on relevant parts of the input sequence.".to_string(),
        "BERT uses bidirectional training to understand context from both directions.".to_string(),
        "GPT models are autoregressive and generate text one token at a time.".to_string(),
        "Positional encoding helps transformers understand the order of tokens in a sequence.".to_string(),
        "Multi-head attention allows the model to attend to different representation subspaces.".to_string(),
        "Layer normalization and residual connections help with training deep transformer networks.".to_string(),
        "Transformers have largely replaced RNNs and CNNs for many NLP tasks.".to_string(),
    ];

    println!("User Question: {}", user_question);
    println!("Knowledge Base: {} documents", knowledge_base.len());

    // Step 1: Generate embeddings for the question (for semantic search)
    println!("\n📝 Step 1: Generate query embedding...");
    let query_embedding_request = EmbeddingRequest::new(
        siliconflow::BGE_M3.to_string(),
        vec![user_question.to_string()],
    );
    let _query_embedding = client.embed(query_embedding_request).await?;
    println!("✅ Query embedding generated");

    // Step 2: Rerank documents based on relevance
    println!("\n🎯 Step 2: Rerank documents by relevance...");
    let rerank_request = RerankRequest::new(
        siliconflow::BGE_RERANKER_V2_M3_PRO.to_string(),
        user_question.to_string(),
        knowledge_base,
    )
    .with_top_n(3)
    .with_return_documents(true);

    let rerank_response = client.rerank(rerank_request).await?;
    
    println!("✅ Top {} relevant documents selected:", rerank_response.results.len());
    for (i, result) in rerank_response.results.iter().enumerate() {
        println!("  {}. [Score: {:.4}] {}", 
            i + 1, 
            result.relevance_score,
            result.document.as_ref().map(|d| &d.text).unwrap_or("N/A")
        );
    }

    // Step 3: Use top documents as context for chat
    println!("\n💬 Step 3: Generate answer using top documents as context...");
    let context = rerank_response.results
        .iter()
        .take(3)
        .filter_map(|r| r.document.as_ref().map(|d| &d.text))
        .collect::<Vec<_>>()
        .join("\n\n");

    let chat_messages = vec![
        system!("You are a helpful AI assistant. Answer the user's question based on the provided context. If the context doesn't contain enough information, say so."),
        user!("Context:\n{}\n\nQuestion: {}", context, user_question)
    ];

    let chat_response = client.chat(chat_messages).await?;
    
    println!("🤖 AI Answer:");
    println!("{}", chat_response.content);
    println!();

    Ok(())
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 1e-6);

        let vec3 = vec![1.0, 0.0, 0.0];
        let vec4 = vec![0.0, 1.0, 0.0];
        let similarity2 = cosine_similarity(&vec3, &vec4);
        assert!((similarity2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_siliconflow_models() {
        let chat_models = siliconflow::all_chat_models();
        assert!(!chat_models.is_empty());
        assert!(chat_models.contains(&siliconflow::DEEPSEEK_CHAT.to_string()));

        let embedding_models = siliconflow::all_embedding_models();
        assert!(!embedding_models.is_empty());
        assert!(embedding_models.contains(&siliconflow::BGE_M3.to_string()));

        let rerank_models = siliconflow::all_rerank_models();
        assert!(!rerank_models.is_empty());
        assert!(rerank_models.contains(&siliconflow::BGE_RERANKER_V2_M3.to_string()));
    }
}
