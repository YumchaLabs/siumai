//! Gemini Embedding with Task Types
//!
//! This example demonstrates how to use Gemini's embedding capabilities with
//! task type optimization and provider-specific configurations.
//!
//! Gemini supports 8 different task types that optimize embeddings for specific use cases:
//! - RETRIEVAL_QUERY: For search queries
//! - RETRIEVAL_DOCUMENT: For documents to be searched
//! - SEMANTIC_SIMILARITY: For similarity comparison
//! - CLASSIFICATION: For text classification
//! - CLUSTERING: For grouping similar texts
//! - QUESTION_ANSWERING: For Q&A systems
//! - FACT_VERIFICATION: For fact checking
//! - TASK_TYPE_UNSPECIFIED: Default behavior
//!
//! Run with: cargo run --example embedding_with_task_types

use siumai::prelude::*;
use siumai::traits::EmbeddingExtensions;
use siumai::types::{EmbeddingRequest, EmbeddingTaskType};

// Import Gemini-specific extensions
use siumai::providers::gemini::{GeminiEmbeddingOptions, GeminiEmbeddingRequestExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Gemini Embedding with Task Types Demo\n");

    // Get API key from environment
    let api_key =
        std::env::var("GEMINI_API_KEY").expect("Please set GEMINI_API_KEY environment variable");

    // Create Gemini client
    let client = Siumai::builder()
        .gemini()
        .api_key(&api_key)
        .model("gemini-embedding-001")
        .build()
        .await?;

    // Example texts for different use cases
    let query_text = "What is machine learning?";
    let document_text = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.";
    let similarity_texts = vec![
        "The weather is sunny today".to_string(),
        "It's a beautiful sunny day".to_string(),
        "I love programming in Rust".to_string(),
    ];

    println!("📊 Demonstrating different task type optimizations:\n");

    // 1. Basic embedding (no task type)
    demonstrate_basic_embedding(&client, query_text).await?;

    // 2. Task type optimizations
    demonstrate_task_type_optimizations(&client, query_text, document_text).await?;

    // 3. Provider-specific configurations
    demonstrate_provider_configurations(&client, &similarity_texts).await?;

    // 4. Batch processing with task types
    demonstrate_batch_processing(&client, &similarity_texts).await?;

    // 5. Different configuration methods
    demonstrate_configuration_methods(&client, query_text).await?;

    println!("✅ All Gemini embedding examples completed!");
    Ok(())
}

/// Demonstrate basic embedding without task type optimization
async fn demonstrate_basic_embedding(
    client: &Siumai,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("1️⃣ Basic Embedding (no task type):");

    let response = client.embed(vec![text.to_string()]).await?;
    println!(
        "   ✅ Embedding dimension: {}",
        response.embeddings[0].len()
    );
    println!("   📝 Model: {}\n", response.model);

    Ok(())
}

/// Demonstrate different task type optimizations
async fn demonstrate_task_type_optimizations(
    client: &Siumai,
    query_text: &str,
    document_text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("2️⃣ Task Type Optimizations:");

    // Retrieval Query - optimized for search queries
    let query_request = EmbeddingRequest::query(query_text);
    let query_response = client.embed_with_config(query_request).await?;
    println!(
        "   🔍 Retrieval Query: {} dimensions",
        query_response.embeddings[0].len()
    );

    // Retrieval Document - optimized for document indexing
    let doc_request = EmbeddingRequest::document(document_text);
    let doc_response = client.embed_with_config(doc_request).await?;
    println!(
        "   📄 Retrieval Document: {} dimensions",
        doc_response.embeddings[0].len()
    );

    // Semantic Similarity - optimized for similarity comparison
    let sim_request = EmbeddingRequest::similarity("Text for similarity comparison");
    let sim_response = client.embed_with_config(sim_request).await?;
    println!(
        "   🔗 Semantic Similarity: {} dimensions",
        sim_response.embeddings[0].len()
    );

    // Classification - optimized for text classification
    let class_request = EmbeddingRequest::classification("This is a positive review");
    let class_response = client.embed_with_config(class_request).await?;
    println!(
        "   🏷️ Classification: {} dimensions",
        class_response.embeddings[0].len()
    );

    // Question Answering - optimized for Q&A systems
    let qa_request = EmbeddingRequest::new(vec!["What is the capital of France?".to_string()])
        .with_task_type(EmbeddingTaskType::QuestionAnswering);
    let qa_response = client.embed_with_config(qa_request).await?;
    println!(
        "   ❓ Question Answering: {} dimensions",
        qa_response.embeddings[0].len()
    );

    println!();
    Ok(())
}

/// Demonstrate Gemini-specific provider configurations
async fn demonstrate_provider_configurations(
    client: &Siumai,
    texts: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("3️⃣ Provider-Specific Configurations:");

    // Method 1: Using GeminiEmbeddingOptions struct
    println!("   📋 Method 1: Using GeminiEmbeddingOptions");
    let config = GeminiEmbeddingOptions::new()
        .with_task_type(EmbeddingTaskType::SemanticSimilarity)
        .with_title("Weather Comparison Context")
        .with_output_dimensionality(512);

    let request = EmbeddingRequest::new(vec![texts[0].clone()]).with_gemini_config(config);

    let response = client.embed_with_config(request).await?;
    println!(
        "      ✅ Custom config: {} dimensions",
        response.embeddings[0].len()
    );

    // Method 2: Using extension trait methods
    println!("   🔧 Method 2: Using extension trait methods");
    let request = EmbeddingRequest::new(vec![texts[1].clone()])
        .with_gemini_task_type(EmbeddingTaskType::Classification)
        .with_gemini_title("Sentiment Analysis Context")
        .with_gemini_dimensions(768);

    let response = client.embed_with_config(request).await?;
    println!(
        "      ✅ Extension methods: {} dimensions",
        response.embeddings[0].len()
    );

    println!();
    Ok(())
}

/// Demonstrate batch processing with task types
async fn demonstrate_batch_processing(
    client: &Siumai,
    texts: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("4️⃣ Batch Processing with Task Types:");

    // Process multiple texts with the same task type
    let batch_request = EmbeddingRequest::new(texts.to_vec())
        .with_gemini_task_type(EmbeddingTaskType::Clustering)
        .with_gemini_title("Text Clustering Analysis");

    let batch_response = client.embed_with_config(batch_request).await?;

    println!(
        "   📦 Processed {} texts with clustering optimization",
        batch_response.embeddings.len()
    );
    for (i, embedding) in batch_response.embeddings.iter().enumerate() {
        println!("      Text {}: {} dimensions", i + 1, embedding.len());
    }

    println!();
    Ok(())
}

/// Demonstrate different configuration methods
async fn demonstrate_configuration_methods(
    client: &Siumai,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("5️⃣ Different Configuration Methods:");

    // Method 1: Fluent configuration
    println!("   🌊 Fluent Configuration:");
    let request = EmbeddingRequest::new(vec![text.to_string()])
        .with_gemini_task_type(EmbeddingTaskType::FactVerification)
        .with_gemini_title("Fact Checking Context")
        .with_dimensions(1024); // Using common field

    let response = client.embed_with_config(request).await?;
    println!(
        "      ✅ Fluent: {} dimensions",
        response.embeddings[0].len()
    );

    // Method 2: Builder pattern with config object
    println!("   🏗️ Builder Pattern:");
    let config = GeminiEmbeddingOptions::new()
        .with_task_type(EmbeddingTaskType::Clustering)
        .with_title("Document Clustering")
        .with_output_dimensionality(256);

    let request = EmbeddingRequest::new(vec![text.to_string()]).with_gemini_config(config);

    let response = client.embed_with_config(request).await?;
    println!(
        "      ✅ Builder: {} dimensions",
        response.embeddings[0].len()
    );

    // Method 3: Mixed approach (common + provider-specific)
    println!("   🔀 Mixed Approach:");
    let request = EmbeddingRequest::new(vec![text.to_string()])
        .with_dimensions(512) // Common field
        .with_gemini_task_type(EmbeddingTaskType::SemanticSimilarity) // Provider-specific
        .with_gemini_title("Mixed Configuration Example"); // Provider-specific

    let response = client.embed_with_config(request).await?;
    println!(
        "      ✅ Mixed: {} dimensions",
        response.embeddings[0].len()
    );

    println!();
    Ok(())
}

/// Compare embeddings from different task types (optional demonstration)
#[allow(dead_code)]
async fn compare_task_type_embeddings(
    client: &Siumai,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Comparing Task Type Optimizations:");

    let task_types = vec![
        ("Query", EmbeddingTaskType::RetrievalQuery),
        ("Document", EmbeddingTaskType::RetrievalDocument),
        ("Similarity", EmbeddingTaskType::SemanticSimilarity),
        ("Classification", EmbeddingTaskType::Classification),
    ];

    for (name, task_type) in task_types {
        let request =
            EmbeddingRequest::new(vec![text.to_string()]).with_gemini_task_type(task_type);

        let response = client.embed_with_config(request).await?;
        let embedding = &response.embeddings[0];

        // Calculate embedding magnitude as a simple comparison metric
        let magnitude = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("   {}: magnitude = {:.4}", name, magnitude);
    }

    Ok(())
}

/// Demonstrate error handling with invalid configurations
#[allow(dead_code)]
async fn demonstrate_error_handling(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚠️ Error Handling Examples:");

    // Example: Empty input
    let empty_request = EmbeddingRequest::new(vec![]);
    match client.embed_with_config(empty_request).await {
        Ok(_) => println!("   Unexpected success with empty input"),
        Err(e) => println!("   ✅ Correctly handled empty input: {}", e),
    }

    // Example: Very long text (if model has limits)
    let long_text = "word ".repeat(10000);
    let long_request = EmbeddingRequest::new(vec![long_text])
        .with_gemini_task_type(EmbeddingTaskType::RetrievalDocument);

    match client.embed_with_config(long_request).await {
        Ok(response) => println!(
            "   ✅ Handled long text: {} dimensions",
            response.embeddings[0].len()
        ),
        Err(e) => println!("   ⚠️ Long text error: {}", e),
    }

    Ok(())
}
