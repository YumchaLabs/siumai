//! Integration tests for OpenAI-Compatible Provider using Provider-Model architecture

#[cfg(feature = "openai")]
mod openai_compatible_provider_tests {
    use siumai::error::LlmError;
    use siumai::provider_model::Provider;
    use siumai::providers::openai_compatible::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
    use siumai::registry::get_provider_adapter;

    #[test]
    fn test_create_provider_with_deepseek() {
        // Get DeepSeek adapter from registry
        let adapter = get_provider_adapter("deepseek").expect("DeepSeek adapter should exist");

        // Create configuration
        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-api-key",
            "https://api.deepseek.com/v1",
            adapter,
        );

        // Create provider
        let provider = OpenAiCompatibleProvider::new(config);

        // Verify provider ID
        assert_eq!(provider.id(), "deepseek");
    }

    #[test]
    fn test_create_provider_with_siliconflow() {
        // Get SiliconFlow adapter from registry
        let adapter =
            get_provider_adapter("siliconflow").expect("SiliconFlow adapter should exist");

        // Create configuration
        let config = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-api-key",
            "https://api.siliconflow.cn/v1",
            adapter,
        );

        // Create provider
        let provider = OpenAiCompatibleProvider::new(config);

        // Verify provider ID
        assert_eq!(provider.id(), "siliconflow");
    }

    #[test]
    fn test_create_chat_model() {
        let adapter = get_provider_adapter("deepseek").expect("DeepSeek adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-api-key",
            "https://api.deepseek.com/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // Create chat model
        let chat_model = provider.chat("deepseek-chat");
        assert!(chat_model.is_ok());
    }

    #[test]
    fn test_create_embedding_model() {
        let adapter = get_provider_adapter("deepseek").expect("DeepSeek adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-api-key",
            "https://api.deepseek.com/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // Create embedding model
        let embedding_model = provider.embedding("deepseek-embedding");
        assert!(embedding_model.is_ok());
    }

    #[test]
    fn test_create_image_model() {
        let adapter = get_provider_adapter("together").expect("Together adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "together",
            "test-api-key",
            "https://api.together.xyz/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // Create image model
        let image_model = provider.image("stabilityai/stable-diffusion-xl-base-1.0");
        assert!(image_model.is_ok());
    }

    #[test]
    fn test_create_rerank_model_with_siliconflow() {
        let adapter =
            get_provider_adapter("siliconflow").expect("SiliconFlow adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-api-key",
            "https://api.siliconflow.cn/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // SiliconFlow supports rerank
        let rerank_model = provider.rerank("BAAI/bge-reranker-v2-m3");
        assert!(rerank_model.is_ok());
    }

    #[test]
    fn test_create_rerank_model_without_support() {
        let adapter = get_provider_adapter("deepseek").expect("DeepSeek adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-api-key",
            "https://api.deepseek.com/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // DeepSeek doesn't support rerank
        let rerank_model = provider.rerank("some-model");
        assert!(rerank_model.is_err());

        if let Err(LlmError::UnsupportedOperation(msg)) = rerank_model {
            assert!(msg.contains("does not support rerank"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[test]
    fn test_supports_capability() {
        let adapter = get_provider_adapter("deepseek").expect("DeepSeek adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-api-key",
            "https://api.deepseek.com/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // DeepSeek supports tools, vision, reasoning (from registry)
        assert!(provider.supports_capability("tools"));
        assert!(provider.supports_capability("vision"));
        assert!(provider.supports_capability("reasoning"));

        // DeepSeek doesn't support rerank
        assert!(!provider.supports_capability("rerank"));
    }

    #[test]
    fn test_create_chat_executor() {
        use siumai::provider_model::ChatModel;

        let adapter = get_provider_adapter("deepseek").expect("DeepSeek adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-api-key",
            "https://api.deepseek.com/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // Create chat model
        let chat_model = provider
            .chat("deepseek-chat")
            .expect("Should create chat model");

        // Create executor
        let http_client = reqwest::Client::new();
        let executor = chat_model.create_executor(
            http_client,
            vec![], // interceptors
            vec![], // middlewares
            None,   // retry_options
        );

        // Verify executor provider ID
        assert_eq!(executor.provider_id, "deepseek");
    }

    #[test]
    fn test_create_embedding_executor() {
        use siumai::provider_model::EmbeddingModel;

        let adapter = get_provider_adapter("deepseek").expect("DeepSeek adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-api-key",
            "https://api.deepseek.com/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // Create embedding model
        let embedding_model = provider
            .embedding("deepseek-embedding")
            .expect("Should create embedding model");

        // Create executor
        let http_client = reqwest::Client::new();
        let executor = embedding_model.create_executor(
            http_client,
            vec![], // interceptors
            None,   // retry_options
        );

        // Verify executor provider ID
        assert_eq!(executor.provider_id, "deepseek");
    }

    #[test]
    fn test_create_rerank_executor() {
        use siumai::provider_model::RerankModel;

        let adapter =
            get_provider_adapter("siliconflow").expect("SiliconFlow adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-api-key",
            "https://api.siliconflow.cn/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // Create rerank model
        let rerank_model = provider
            .rerank("BAAI/bge-reranker-v2-m3")
            .expect("Should create rerank model");

        // Create executor
        let http_client = reqwest::Client::new();
        let executor = rerank_model.create_executor(
            http_client,
            vec![], // interceptors
            None,   // retry_options
        );

        // Verify executor provider ID
        assert_eq!(executor.provider_id, "siliconflow");

        // Verify URL is correct
        assert_eq!(executor.url, "https://api.siliconflow.cn/v1/rerank");
    }

    #[test]
    fn test_multiple_providers() {
        // Test creating multiple providers
        let providers = vec![
            ("deepseek", "https://api.deepseek.com/v1"),
            ("siliconflow", "https://api.siliconflow.cn/v1"),
            ("openrouter", "https://openrouter.ai/api/v1"),
            ("together", "https://api.together.xyz/v1"),
        ];

        for (provider_id, base_url) in providers {
            let adapter = get_provider_adapter(provider_id);
            if adapter.is_err() {
                // Skip if adapter not found (some providers might not be registered)
                continue;
            }

            let config = OpenAiCompatibleConfig::new(
                provider_id,
                "test-api-key",
                base_url,
                adapter.unwrap(),
            );
            let provider = OpenAiCompatibleProvider::new(config);

            assert_eq!(provider.id(), provider_id);

            // All should support chat
            assert!(provider.chat("test-model").is_ok());
        }
    }

    #[test]
    fn test_provider_config_retrieval() {
        let adapter = get_provider_adapter("deepseek").expect("DeepSeek adapter should exist");
        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-api-key",
            "https://api.deepseek.com/v1",
            adapter,
        );
        let provider = OpenAiCompatibleProvider::new(config);

        // Get provider config from registry
        let provider_config = provider.provider_config();
        assert!(provider_config.is_some());

        if let Some(config) = provider_config {
            assert_eq!(config.id, "deepseek");
            // DeepSeek capabilities include tools, vision, reasoning
            assert!(config.capabilities.contains(&"tools".to_string()));
            assert!(config.capabilities.contains(&"vision".to_string()));
            assert!(config.capabilities.contains(&"reasoning".to_string()));
            assert!(config.supports_reasoning);
        }
    }
}
