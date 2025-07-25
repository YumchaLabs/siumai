//! Tests for Groq Provider
//!
//! Unit tests for the Groq provider implementation.

#[cfg(test)]
mod groq_tests {
    use super::super::*;
    use crate::client::LlmClient;
    use crate::types::*;

    #[test]
    fn test_groq_config_creation() {
        let config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(0.7)
            .with_max_tokens(1000);

        assert_eq!(config.api_key, "test-api-key");
        assert_eq!(config.common_params.model, "llama-3.3-70b-versatile");
        assert_eq!(config.common_params.temperature, Some(0.7));
        assert_eq!(config.common_params.max_tokens, Some(1000));
        assert_eq!(config.base_url, GroqConfig::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_groq_config_validation() {
        // Valid configuration
        let valid_config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(0.7);
        assert!(valid_config.validate().is_ok());

        // Invalid temperature
        let invalid_temp_config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(3.0);
        assert!(invalid_temp_config.validate().is_err());

        // Empty API key
        let empty_key_config = GroqConfig::new("").with_model("llama-3.3-70b-versatile");
        assert!(empty_key_config.validate().is_err());
    }

    #[test]
    fn test_groq_supported_models() {
        let models = GroqConfig::supported_models();
        assert!(models.contains(&"llama-3.3-70b-versatile"));
        assert!(models.contains(&"whisper-large-v3"));

        assert!(GroqConfig::is_model_supported("llama-3.3-70b-versatile"));
        assert!(!GroqConfig::is_model_supported("non-existent-model"));
    }

    #[test]
    fn test_groq_builder() {
        let builder = GroqBuilder::new()
            .api_key("test-key")
            .model("llama-3.3-70b-versatile")
            .temperature(0.7)
            .max_tokens(1000);

        let config = builder.config();
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.common_params.model, "llama-3.3-70b-versatile");
        assert_eq!(config.common_params.temperature, Some(0.7));
        assert_eq!(config.common_params.max_tokens, Some(1000));
    }

    #[test]
    fn test_groq_builder_tools() {
        let tool = Tool {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: "test_function".to_string(),
                description: "A test function".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        };

        let builder = GroqBuilder::new().api_key("test-key").tool(tool.clone());

        let config = builder.config();
        assert_eq!(config.built_in_tools.len(), 1);
        assert_eq!(config.built_in_tools[0].function.name, "test_function");
    }

    #[tokio::test]
    async fn test_groq_client_creation() {
        let config = GroqConfig::new("test-api-key").with_model("llama-3.3-70b-versatile");

        let http_client = reqwest::Client::new();
        let client = GroqClient::new(config, http_client);

        assert_eq!(client.provider_name(), "groq");
        assert!(
            client
                .supported_models()
                .contains(&"llama-3.3-70b-versatile".to_string())
        );

        let capabilities = client.capabilities();
        assert!(capabilities.supports("chat"));
        assert!(capabilities.supports("streaming"));
        assert!(capabilities.supports("tools"));
    }

    #[test]
    fn test_groq_utils_build_headers() {
        use super::super::utils::build_headers;
        use std::collections::HashMap;

        let custom_headers = HashMap::new();
        let headers = build_headers("test-api-key", &custom_headers).unwrap();

        assert_eq!(
            headers.get(reqwest::header::AUTHORIZATION).unwrap(),
            "Bearer test-api-key"
        );
        assert_eq!(
            headers.get(reqwest::header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        assert!(headers.get(reqwest::header::USER_AGENT).is_some());
    }

    #[test]
    fn test_groq_utils_convert_messages() {
        use super::super::utils::convert_messages;

        let messages = vec![
            ChatMessage::system("You are a helpful assistant").build(),
            ChatMessage::user("Hello, world!").build(),
        ];

        let groq_messages = convert_messages(&messages).unwrap();
        assert_eq!(groq_messages.len(), 2);
        assert_eq!(groq_messages[0]["role"], "system");
        assert_eq!(groq_messages[1]["role"], "user");
    }

    #[test]
    fn test_groq_utils_parse_finish_reason() {
        use super::super::utils::parse_finish_reason;

        assert_eq!(parse_finish_reason(Some("stop")), FinishReason::Stop);
        assert_eq!(parse_finish_reason(Some("length")), FinishReason::Length);
        assert_eq!(
            parse_finish_reason(Some("tool_calls")),
            FinishReason::ToolCalls
        );
        assert_eq!(
            parse_finish_reason(Some("unknown")),
            FinishReason::Other("unknown".to_string())
        );
        assert_eq!(
            parse_finish_reason(None),
            FinishReason::Other("unknown".to_string())
        );
    }

    #[test]
    fn test_groq_utils_validate_params() {
        use super::super::utils::validate_groq_params;

        // Valid parameters
        let valid_params = serde_json::json!({
            "temperature": 0.7,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "service_tier": "auto"
        });
        assert!(validate_groq_params(&valid_params).is_ok());

        // Invalid temperature
        let invalid_temp = serde_json::json!({
            "temperature": 3.0
        });
        assert!(validate_groq_params(&invalid_temp).is_err());

        // Invalid service_tier
        let invalid_tier = serde_json::json!({
            "service_tier": "invalid"
        });
        assert!(validate_groq_params(&invalid_tier).is_err());
    }

    #[test]
    fn test_groq_params() {
        let params = GroqParams::new()
            .with_frequency_penalty(0.5)
            .with_presence_penalty(-0.2)
            .with_parallel_tool_calls(true)
            .with_service_tier("auto")
            .with_reasoning_effort("default")
            .with_reasoning_format("hidden");

        assert_eq!(params.frequency_penalty, Some(0.5));
        assert_eq!(params.presence_penalty, Some(-0.2));
        assert_eq!(params.parallel_tool_calls, Some(true));
        assert_eq!(params.service_tier, Some("auto".to_string()));
        assert_eq!(params.reasoning_effort, Some("default".to_string()));
        assert_eq!(params.reasoning_format, Some("hidden".to_string()));
    }

    #[test]
    fn test_groq_audio_capability() {
        use super::super::audio::GroqAudio;
        use crate::types::HttpConfig;

        let audio = GroqAudio::new(
            "test-api-key".to_string(),
            "https://api.groq.com/openai/v1".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
        );

        assert!(audio.supports_transcription());
        assert!(audio.supports_translation());
        assert!(audio.supports_speech_synthesis());

        let models = audio.supported_audio_models();
        assert!(models.contains(&"whisper-large-v3".to_string()));
        assert!(models.contains(&"whisper-large-v3-turbo".to_string()));
    }

    #[test]
    fn test_groq_files_capability() {
        use super::super::files::GroqFiles;
        use crate::types::HttpConfig;

        let _files = GroqFiles::new(
            "test-api-key".to_string(),
            "https://api.groq.com/openai/v1".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
        );

        // Note: Methods are private, so we just test creation
        // In a real implementation, these would be public or tested through public interfaces
    }

    #[test]
    fn test_groq_models_capability() {
        use super::super::models::GroqModels;
        use crate::types::HttpConfig;

        let _models = GroqModels::new(
            "test-api-key".to_string(),
            "https://api.groq.com/openai/v1".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
        );

        // Note: Methods are private, so we just test creation
        // In a real implementation, these would be public or tested through public interfaces
    }

    #[test]
    fn test_provider_type_display() {
        assert_eq!(ProviderType::Groq.to_string(), "groq");
    }

    #[test]
    fn test_parameter_mapper_factory() {
        use crate::params::ParameterMapperFactory;

        let _mapper = ParameterMapperFactory::create_mapper(&ProviderType::Groq);
        // Should use OpenAI-compatible mapper
        assert!(ParameterMapperFactory::has_mapper(&ProviderType::Groq));

        let available = ParameterMapperFactory::available_mappers();
        assert!(available.contains(&ProviderType::Groq));
    }
}
