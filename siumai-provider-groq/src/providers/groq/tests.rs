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
        use secrecy::ExposeSecret;
        let config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(0.7)
            .with_max_tokens(1000);

        assert_eq!(config.api_key.expose_secret(), "test-api-key");
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

        // High temperature (now allowed with relaxed validation)
        let high_temp_config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(3.0);
        assert!(high_temp_config.validate().is_ok()); // Now allowed

        // Negative temperature (still invalid)
        let invalid_temp_config = GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile")
            .with_temperature(-1.0);
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
        use secrecy::ExposeSecret;
        let builder = GroqBuilder::new(crate::builder::BuilderBase::default())
            .api_key("test-key")
            .model("llama-3.3-70b-versatile")
            .temperature(0.7)
            .max_tokens(1000);

        // Access config field directly
        assert_eq!(builder.config.api_key.expose_secret(), "test-key");
        assert_eq!(
            builder.config.common_params.model,
            "llama-3.3-70b-versatile"
        );
        assert_eq!(builder.config.common_params.temperature, Some(0.7));
        assert_eq!(builder.config.common_params.max_tokens, Some(1000));
    }

    #[tokio::test]
    async fn test_groq_client_creation() {
        let config = GroqConfig::new("test-api-key")
            .with_model(crate::providers::groq::models::popular::FLAGSHIP);

        let http_client = reqwest::Client::new();
        let client = GroqClient::new(config, http_client);

        assert_eq!(client.provider_id(), std::borrow::Cow::Borrowed("groq"));
        assert!(
            client
                .supported_models()
                .contains(&crate::providers::groq::models::popular::FLAGSHIP.to_string())
        );

        let capabilities = client.capabilities();
        assert!(capabilities.supports("chat"));
        assert!(capabilities.supports("streaming"));
        assert!(capabilities.supports("tools"));
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

        // High temperature (now allowed with relaxed validation)
        let high_temp = serde_json::json!({
            "temperature": 3.0
        });
        assert!(validate_groq_params(&high_temp).is_ok());

        // Negative temperature (still invalid)
        let invalid_temp = serde_json::json!({
            "temperature": -1.0
        });
        assert!(validate_groq_params(&invalid_temp).is_err());

        // Invalid service_tier
        let invalid_tier = serde_json::json!({
            "service_tier": "invalid"
        });
        assert!(validate_groq_params(&invalid_tier).is_err());
    }

    #[test]
    fn test_groq_chat_transformer_maps_developer_role_to_system() {
        use crate::core::ProviderContext;
        use crate::core::ProviderSpec;
        use crate::providers::groq::spec::GroqSpec;

        let spec = GroqSpec;
        let ctx = ProviderContext::new(
            "groq",
            GroqConfig::DEFAULT_BASE_URL.to_string(),
            None,
            Default::default(),
        );

        let messages = vec![ChatMessage::developer("dev-msg").build()];
        let req = ChatRequest::builder()
            .messages(messages)
            .common_params(CommonParams {
                model: "llama-3.3-70b-versatile".to_string(),
                ..Default::default()
            })
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        assert_eq!(body["messages"][0]["role"], "system");
    }

    #[test]
    fn test_groq_chat_transformer_omits_stream_options() {
        use crate::core::ProviderContext;
        use crate::core::ProviderSpec;
        use crate::providers::groq::spec::GroqSpec;

        let spec = GroqSpec;
        let ctx = ProviderContext::new(
            "groq",
            GroqConfig::DEFAULT_BASE_URL.to_string(),
            None,
            Default::default(),
        );

        let messages = vec![ChatMessage::user("hi").build()];
        let req = ChatRequest::builder()
            .messages(messages)
            .common_params(CommonParams {
                model: "llama-3.3-70b-versatile".to_string(),
                ..Default::default()
            })
            .stream(true)
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        assert!(body.get("stream_options").is_none());
        assert_eq!(
            body.get("stream")
                .and_then(|v: &serde_json::Value| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn test_groq_chat_transformer_uses_max_tokens() {
        use crate::core::ProviderContext;
        use crate::core::ProviderSpec;
        use crate::providers::groq::spec::GroqSpec;

        let spec = GroqSpec;
        let ctx = ProviderContext::new(
            "groq",
            GroqConfig::DEFAULT_BASE_URL.to_string(),
            None,
            Default::default(),
        );

        let messages = vec![ChatMessage::user("hi").build()];
        let req = ChatRequest::builder()
            .messages(messages)
            .common_params(CommonParams {
                model: "llama-3.3-70b-versatile".to_string(),
                max_completion_tokens: Some(123),
                ..Default::default()
            })
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).unwrap();
        assert!(body.get("max_completion_tokens").is_none());
        assert_eq!(
            body.get("max_tokens")
                .and_then(|v: &serde_json::Value| v.as_u64()),
            Some(123)
        );
    }

    #[test]
    fn test_groq_audio_transformer_defaults() {
        use crate::execution::transformers::audio::{AudioHttpBody, AudioTransformer};
        let tx = super::super::transformers::GroqAudioTransformer;

        // TTS defaults
        let tts = crate::types::TtsRequest {
            text: "hello".to_string(),
            model: None,
            voice: None,
            format: None,
            speed: None,
            provider_options_map: Default::default(),
            extra_params: std::collections::HashMap::new(),
            http_config: None,
        };
        match tx.build_tts_body(&tts).unwrap() {
            AudioHttpBody::Json(j) => {
                assert_eq!(j["model"], "playai-tts");
                assert_eq!(j["voice"], "Fritz-PlayAI");
                assert_eq!(j["response_format"], "wav");
                assert_eq!(j["speed"], 1.0);
            }
            _ => panic!("expected JSON body for TTS"),
        }

        // STT defaults
        let stt = crate::types::SttRequest {
            audio_data: Some(vec![1, 2, 3]),
            file_path: None,
            format: None,
            media_type: None,
            model: None,
            language: None,
            timestamp_granularities: None,
            provider_options_map: Default::default(),
            extra_params: std::collections::HashMap::new(),
            http_config: None,
        };
        match tx.build_stt_body(&stt).unwrap() {
            AudioHttpBody::Multipart(_) => {}
            _ => panic!("expected multipart body for STT"),
        }

        assert_eq!(tx.tts_endpoint(), "/audio/speech");
        assert_eq!(tx.stt_endpoint(), "/audio/transcriptions");
    }

    #[test]
    fn test_groq_client_audio_features() {
        use crate::traits::AudioCapability;

        let config = super::super::config::GroqConfig::new("test-api-key")
            .with_model("llama-3.3-70b-versatile");
        let http_client = reqwest::Client::new();
        let client = super::super::client::GroqClient::new(config, http_client);

        let feats =
            <super::super::client::GroqClient as AudioCapability>::supported_features(&client);
        assert!(
            feats
                .iter()
                .any(|f| matches!(f, crate::types::AudioFeature::TextToSpeech))
        );
        assert!(
            feats
                .iter()
                .any(|f| matches!(f, crate::types::AudioFeature::SpeechToText))
        );
    }

    // Files capability removed for Groq; test omitted

    #[test]
    fn test_groq_models_capability() {
        use super::super::api::GroqModels;
        use crate::types::HttpConfig;
        use secrecy::SecretString;

        let _models = GroqModels::new(
            SecretString::from("test-api-key".to_string()),
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
}
