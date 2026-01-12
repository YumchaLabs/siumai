use super::*;

#[cfg(test)]
mod images_tests {
    use super::*;
    use crate::execution::transformers::request::RequestTransformer;
    use crate::execution::transformers::response::ResponseTransformer;

    fn cfg() -> GeminiConfig {
        use secrecy::SecretString;
        GeminiConfig {
            api_key: SecretString::from("x".to_string()),
            base_url: "https://example.com".into(),
            model: "gemini-1.5-flash".into(),
            common_params: crate::types::CommonParams::default(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: crate::types::HttpConfig::default(),
            token_provider: None,
            http_transport: None,
            provider_metadata_key: None,
        }
    }

    #[test]
    fn test_transform_image_builds_generate_content_body() {
        let tx = GeminiRequestTransformer { config: cfg() };
        let req = crate::types::ImageGenerationRequest {
            prompt: "a cat".into(),
            count: 1,
            ..Default::default()
        };
        let body = tx.transform_image(&req).unwrap();
        // Basic presence checks
        assert_eq!(body["model"], "gemini-1.5-flash");
        assert!(body.get("contents").is_some());
        // Ensure modalities include IMAGE via generationConfig if present after transform
        // Note: we don't require presence if not set; behavior depends on config merging
    }

    #[test]
    fn test_transform_image_response_extracts_images() {
        let tx = GeminiResponseTransformer { config: cfg() };
        let json = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            { "inlineData": { "mime_type": "image/png", "data": "iVBORw0..." } },
                            { "fileData": { "file_uri": "https://storage.example/image.png", "mime_type": "image/png" } }
                        ]
                    }
                }
            ]
        });
        let out = tx.transform_image_response(&json).unwrap();
        assert_eq!(out.images.len(), 2);
        assert!(out.images.iter().any(|i| i.b64_json.is_some()));
        assert!(out.images.iter().any(|i| i.url.is_some()));
    }
}

#[cfg(test)]
mod embeddings_tests {
    use super::*;
    use crate::execution::transformers::request::RequestTransformer;
    use crate::execution::transformers::response::ResponseTransformer;

    fn cfg() -> GeminiConfig {
        use secrecy::SecretString;
        GeminiConfig {
            api_key: SecretString::from("x".to_string()),
            base_url: "https://example.com".into(),
            model: "gemini-embedding-001".into(),
            common_params: crate::types::CommonParams::default(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: crate::types::HttpConfig::default(),
            token_provider: None,
            http_transport: None,
            provider_metadata_key: None,
        }
    }

    #[test]
    fn test_transform_embedding_response_single() {
        let tx = GeminiResponseTransformer { config: cfg() };
        let json = serde_json::json!({
            "embedding": { "values": [0.1, 0.2, 0.3] }
        });
        let out = tx.transform_embedding_response(&json).unwrap();
        assert_eq!(out.embeddings.len(), 1);
        assert_eq!(out.embeddings[0].len(), 3);
        assert_eq!(out.model, "gemini-embedding-001");
    }

    #[test]
    fn test_transform_embedding_response_batch() {
        let tx = GeminiResponseTransformer { config: cfg() };
        let json = serde_json::json!({
            "embeddings": [
                { "values": [0.1, 0.2] },
                { "values": [0.3, 0.4] }
            ]
        });
        let out = tx.transform_embedding_response(&json).unwrap();
        assert_eq!(out.embeddings.len(), 2);
        assert_eq!(out.embeddings[0], vec![0.1_f32, 0.2_f32]);
        assert_eq!(out.embeddings[1], vec![0.3_f32, 0.4_f32]);
    }

    #[test]
    fn test_transform_embedding_request_single_flattened() {
        let tx = GeminiRequestTransformer { config: cfg() };
        let req = crate::types::EmbeddingRequest::new(vec!["Hello".to_string()])
            .with_dimensions(768)
            .with_task_type(crate::types::EmbeddingTaskType::RetrievalQuery)
            .with_title("My Title");

        let body = tx.transform_embedding(&req).expect("serialize request");

        // Ensure there is no nested embeddingConfig and fields are flattened
        assert!(body.get("embeddingConfig").is_none());
        assert_eq!(body["model"], "models/gemini-embedding-001");
        assert_eq!(body["taskType"], "RETRIEVAL_QUERY");
        assert_eq!(body["title"], "My Title");
        assert_eq!(body["outputDimensionality"], 768);
        assert_eq!(body["content"]["parts"][0]["text"], "Hello");
        // role is optional for single
        assert!(body["content"].get("role").is_none());
    }

    #[test]
    fn test_transform_embedding_request_batch_flattened_with_role() {
        let tx = GeminiRequestTransformer { config: cfg() };
        let req = crate::types::EmbeddingRequest::new(vec!["A".to_string(), "B".to_string()])
            .with_dimensions(64)
            .with_task_type(crate::types::EmbeddingTaskType::SemanticSimilarity);

        let body = tx.transform_embedding(&req).expect("serialize request");

        let requests = body["requests"].as_array().expect("requests array");
        assert_eq!(requests.len(), 2);
        for (i, value) in ["A", "B"].iter().enumerate() {
            let item = &requests[i];
            assert_eq!(item["model"], "models/gemini-embedding-001");
            assert_eq!(item["taskType"], "SEMANTIC_SIMILARITY");
            assert_eq!(item["outputDimensionality"], 64);
            assert_eq!(item["content"]["role"], "user");
            assert_eq!(item["content"]["parts"][0]["text"], *value);
            assert!(item.get("embeddingConfig").is_none());
        }
    }
}
