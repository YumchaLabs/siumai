//! SiliconFlow Rerank Integration Tests
//!
//! These tests verify the rerank functionality with SiliconFlow provider.

use siumai::prelude::*;
use siumai::models::openai_compatible::siliconflow;

/// Test basic rerank functionality
#[tokio::test]
#[ignore = "Requires SILICONFLOW_API_KEY environment variable"]
async fn test_basic_rerank() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("SILICONFLOW_API_KEY")
        .expect("SILICONFLOW_API_KEY environment variable not set");

    let client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key(&api_key)
        .build()
        .await?;

    let query = "machine learning algorithms";
    let documents = vec![
        "Linear regression is a statistical method.".to_string(),
        "Deep learning uses neural networks.".to_string(),
        "Random forests combine decision trees.".to_string(),
        "Cooking recipes for dinner.".to_string(),
    ];

    let request = RerankRequest::new(
        siliconflow::BGE_RERANKER_V2_M3.to_string(),
        query.to_string(),
        documents.clone(),
    )
    .with_top_n(3)
    .with_return_documents(true);

    let response = client.rerank(request).await?;

    // Verify response structure
    assert!(!response.id.is_empty());
    assert_eq!(response.results.len(), 3);
    assert!(response.tokens.input_tokens > 0);

    // Verify results are sorted by relevance (descending)
    for i in 1..response.results.len() {
        assert!(
            response.results[i - 1].relevance_score >= response.results[i].relevance_score,
            "Results should be sorted by relevance score"
        );
    }

    // Verify document indices are valid
    for result in &response.results {
        assert!(
            result.index < documents.len() as u32,
            "Document index should be valid"
        );
    }

    // Verify documents are returned when requested
    for result in &response.results {
        assert!(
            result.document.is_some(),
            "Documents should be returned when requested"
        );
    }

    println!("✅ Basic rerank test passed");
    Ok(())
}

/// Test rerank with custom instruction (Qwen models)
#[tokio::test]
#[ignore = "Requires SILICONFLOW_API_KEY environment variable"]
async fn test_rerank_with_instruction() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("SILICONFLOW_API_KEY")
        .expect("SILICONFLOW_API_KEY environment variable not set");

    let client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key(&api_key)
        .build()
        .await?;

    let query = "programming languages";
    let documents = vec![
        "Python is a high-level programming language.".to_string(),
        "JavaScript runs in web browsers.".to_string(),
        "Rust is a systems programming language.".to_string(),
        "Cooking with various ingredients.".to_string(),
    ];

    let request = RerankRequest::new(
        siliconflow::QWEN3_RERANKER_4B.to_string(),
        query.to_string(),
        documents,
    )
    .with_instruction("Focus on programming languages used for web development.".to_string())
    .with_top_n(2)
    .with_return_documents(true);

    let response = client.rerank(request).await?;

    // Verify response
    assert!(!response.id.is_empty());
    assert_eq!(response.results.len(), 2);

    // JavaScript should likely rank higher due to the instruction
    let top_result = &response.results[0];
    assert!(top_result.relevance_score > 0.0);

    println!("✅ Rerank with instruction test passed");
    Ok(())
}

/// Test rerank with advanced parameters
#[tokio::test]
#[ignore = "Requires SILICONFLOW_API_KEY environment variable"]
async fn test_rerank_advanced_parameters() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("SILICONFLOW_API_KEY")
        .expect("SILICONFLOW_API_KEY environment variable not set");

    let client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key(&api_key)
        .build()
        .await?;

    let query = "artificial intelligence";
    let documents = vec![
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.".to_string(),
        "Deep learning uses neural networks with multiple layers to process data.".to_string(),
        "Natural language processing enables computers to understand human language.".to_string(),
        "Computer vision allows machines to interpret and understand visual information.".to_string(),
        "Robotics combines AI with mechanical engineering to create autonomous systems.".to_string(),
    ];

    let request = RerankRequest::new(
        siliconflow::BGE_RERANKER_V2_M3_PRO.to_string(),
        query.to_string(),
        documents,
    )
    .with_top_n(4)
    .with_return_documents(false) // Don't return documents
    .with_max_chunks_per_doc(3)
    .with_overlap_tokens(5);

    let response = client.rerank(request).await?;

    // Verify response
    assert!(!response.id.is_empty());
    assert_eq!(response.results.len(), 4);

    // Verify documents are not returned when not requested
    for result in &response.results {
        assert!(
            result.document.is_none(),
            "Documents should not be returned when not requested"
        );
    }

    // Verify all results have valid scores
    for result in &response.results {
        assert!(
            result.relevance_score >= 0.0 && result.relevance_score <= 1.0,
            "Relevance score should be between 0 and 1"
        );
    }

    println!("✅ Advanced parameters test passed");
    Ok(())
}

/// Test rerank capability methods
#[tokio::test]
#[ignore = "Requires SILICONFLOW_API_KEY environment variable"]
async fn test_rerank_capabilities() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("SILICONFLOW_API_KEY")
        .expect("SILICONFLOW_API_KEY environment variable not set");

    let client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key(&api_key)
        .build()
        .await?;

    // Test max_documents
    let max_docs = client.max_documents();
    assert!(max_docs.is_some());
    assert!(max_docs.unwrap() > 0);

    // Test supported_models
    let models = client.supported_models();
    assert!(!models.is_empty());
    assert!(models.contains(&siliconflow::BGE_RERANKER_V2_M3.to_string()));

    println!("✅ Rerank capabilities test passed");
    Ok(())
}

/// Test error handling
#[tokio::test]
#[ignore = "Requires SILICONFLOW_API_KEY environment variable"]
async fn test_rerank_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key("invalid-key")
        .build()
        .await?;

    let request = RerankRequest::new(
        siliconflow::BGE_RERANKER_V2_M3.to_string(),
        "test query".to_string(),
        vec!["test document".to_string()],
    );

    let result = client.rerank(request).await;
    assert!(result.is_err());

    match result {
        Err(LlmError::ApiError(_)) => {
            println!("✅ Correctly handled API error");
        }
        Err(e) => {
            println!("⚠️  Unexpected error type: {:?}", e);
        }
        Ok(_) => {
            panic!("Expected error but got success");
        }
    }

    Ok(())
}

/// Test empty documents handling
#[tokio::test]
#[ignore = "Requires SILICONFLOW_API_KEY environment variable"]
async fn test_empty_documents() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("SILICONFLOW_API_KEY")
        .expect("SILICONFLOW_API_KEY environment variable not set");

    let client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key(&api_key)
        .build()
        .await?;

    let request = RerankRequest::new(
        siliconflow::BGE_RERANKER_V2_M3.to_string(),
        "test query".to_string(),
        vec![], // Empty documents
    );

    let result = client.rerank(request).await;
    
    // This should either succeed with empty results or return an appropriate error
    match result {
        Ok(response) => {
            assert!(response.results.is_empty());
            println!("✅ Empty documents handled successfully");
        }
        Err(LlmError::ApiError(_)) => {
            println!("✅ Empty documents correctly rejected by API");
        }
        Err(e) => {
            println!("⚠️  Unexpected error: {:?}", e);
        }
    }

    Ok(())
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_rerank_request_builder() {
        let request = RerankRequest::new(
            "test-model".to_string(),
            "test query".to_string(),
            vec!["doc1".to_string(), "doc2".to_string()],
        )
        .with_instruction("test instruction".to_string())
        .with_top_n(5)
        .with_return_documents(true)
        .with_max_chunks_per_doc(3)
        .with_overlap_tokens(10);

        assert_eq!(request.model, "test-model");
        assert_eq!(request.query, "test query");
        assert_eq!(request.documents_len(), 2);
        assert_eq!(request.instruction, Some("test instruction".to_string()));
        assert_eq!(request.top_n, Some(5));
        assert_eq!(request.return_documents, Some(true));
        assert_eq!(request.max_chunks_per_doc, Some(3));
        assert_eq!(request.overlap_tokens, Some(10));
    }

    #[test]
    fn test_siliconflow_model_constants() {
        // Test that model constants are correct
        assert_eq!(siliconflow::BGE_RERANKER_V2_M3, "BAAI/bge-reranker-v2-m3");
        assert_eq!(siliconflow::BGE_RERANKER_V2_M3_PRO, "Pro/BAAI/bge-reranker-v2-m3");
        assert_eq!(siliconflow::QWEN3_RERANKER_8B, "Qwen/Qwen3-Reranker-8B");
        assert_eq!(siliconflow::QWEN3_RERANKER_4B, "Qwen/Qwen3-Reranker-4B");
        assert_eq!(siliconflow::QWEN3_RERANKER_0_6B, "Qwen/Qwen3-Reranker-0.6B");
        assert_eq!(siliconflow::BCE_RERANKER_BASE_V1, "netease-youdao/bce-reranker-base_v1");
    }
}
