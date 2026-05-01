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
            generate_id: None,
            provider_name: None,
            provider_metadata_key: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
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

        let google = out
            .metadata
            .get("google")
            .and_then(|value| value.as_object())
            .expect("google image metadata");
        assert_eq!(
            google
                .get("images")
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn test_transform_imagen_response_adds_ai_sdk_google_image_metadata() {
        let tx = GeminiResponseTransformer { config: cfg() };
        let json = serde_json::json!({
            "predictions": [
                {
                    "bytesBase64Encoded": "b64-image",
                    "prompt": "a revised prompt"
                }
            ]
        });

        let out = tx.transform_image_response(&json).unwrap();

        assert_eq!(out.images.len(), 1);
        assert_eq!(out.images[0].b64_json.as_deref(), Some("b64-image"));
        let google = out
            .metadata
            .get("google")
            .and_then(|value| value.as_object())
            .expect("google image metadata");
        assert_eq!(
            google
                .get("images")
                .and_then(|value| value.as_array())
                .and_then(|images| images.first())
                .and_then(|image| image.get("revisedPrompt")),
            Some(&serde_json::json!("a revised prompt"))
        );
    }

    #[test]
    fn test_transform_image_response_uses_vertex_provider_metadata_key() {
        let mut cfg = cfg();
        cfg.provider_metadata_key = Some("vertex".to_string());
        let tx = GeminiResponseTransformer { config: cfg };
        let json = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            { "inlineData": { "mimeType": "image/png", "data": "iVBORw0..." } }
                        ]
                    }
                }
            ]
        });

        let out = tx.transform_image_response(&json).unwrap();

        assert!(out.metadata.contains_key("vertex"));
        assert!(!out.metadata.contains_key("google"));
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
            generate_id: None,
            provider_name: None,
            provider_metadata_key: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
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
        let response = out.response.expect("response metadata");
        assert_eq!(response.model_id.as_deref(), Some("gemini-embedding-001"));
        assert!(response.headers.is_empty());
        assert_eq!(
            response.body.as_ref().expect("raw body")["embedding"]["values"][1],
            serde_json::json!(0.2)
        );
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
        let response = out.response.expect("response metadata");
        assert_eq!(response.model_id.as_deref(), Some("gemini-embedding-001"));
        assert_eq!(
            response.body.as_ref().expect("raw body")["embeddings"][1]["values"][0],
            serde_json::json!(0.3)
        );
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

    #[test]
    fn test_transform_embedding_request_google_provider_content_merges_multimodal_parts() {
        let tx = GeminiRequestTransformer { config: cfg() };
        let req = crate::types::EmbeddingRequest::new(vec!["A".to_string(), "B".to_string()])
            .with_provider_option(
                "google",
                serde_json::json!({
                    "taskType": "SEMANTIC_SIMILARITY",
                    "outputDimensionality": 64,
                    "content": [
                        [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "Zm9v"
                                }
                            }
                        ],
                        null
                    ]
                }),
            );

        let body = tx.transform_embedding(&req).expect("serialize request");

        let requests = body["requests"].as_array().expect("requests array");
        assert_eq!(
            requests[0]["taskType"],
            serde_json::json!("SEMANTIC_SIMILARITY")
        );
        assert_eq!(requests[0]["outputDimensionality"], serde_json::json!(64));
        assert_eq!(
            requests[0]["content"]["parts"][0]["text"],
            serde_json::json!("A")
        );
        assert_eq!(
            requests[0]["content"]["parts"][1]["inlineData"]["mimeType"],
            serde_json::json!("image/png")
        );
        assert_eq!(
            requests[1]["content"]["parts"][0]["text"],
            serde_json::json!("B")
        );
        assert_eq!(
            requests[1]["content"]["parts"]
                .as_array()
                .map(|parts| parts.len()),
            Some(1)
        );
    }
}

#[cfg(test)]
mod chat_tests {
    use super::*;
    use crate::execution::transformers::request::RequestTransformer;

    fn cfg() -> GeminiConfig {
        use secrecy::SecretString;
        GeminiConfig {
            api_key: SecretString::from("x".to_string()),
            base_url: "https://example.com".into(),
            model: "gemini-2.5-flash".into(),
            common_params: crate::types::CommonParams::default(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: crate::types::HttpConfig::default(),
            token_provider: None,
            http_transport: None,
            generate_id: None,
            provider_name: None,
            provider_metadata_key: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    #[test]
    fn test_transform_chat_request_google_provider_service_tier_is_top_level() {
        let tx = GeminiRequestTransformer { config: cfg() };
        let req =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
                .with_model_params(crate::types::CommonParams::with_model_capacity(
                    "gemini-2.5-flash".to_string(),
                    0,
                ))
                .with_provider_option(
                    "google",
                    serde_json::json!({
                        "serviceTier": "flex"
                    }),
                );

        let body = tx.transform_chat(&req).expect("serialize request");
        assert_eq!(body["serviceTier"], serde_json::json!("flex"));
    }
}
