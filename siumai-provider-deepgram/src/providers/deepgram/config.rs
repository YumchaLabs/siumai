use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::types::HttpConfig;
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

/// Provider-owned config-first surface for Deepgram.
#[derive(Clone)]
pub struct DeepgramConfig {
    pub api_key: SecretString,
    pub base_url: String,
    pub speech_model: String,
    pub transcription_model: String,
    pub http_config: HttpConfig,
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
}

impl std::fmt::Debug for DeepgramConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepgramConfig")
            .field("base_url", &self.base_url)
            .field("speech_model", &self.speech_model)
            .field("transcription_model", &self.transcription_model)
            .field("http_config", &self.http_config)
            .field("has_api_key", &!self.api_key.expose_secret().is_empty())
            .field("has_http_transport", &self.http_transport.is_some())
            .field("http_interceptors_len", &self.http_interceptors.len())
            .finish()
    }
}

impl DeepgramConfig {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.deepgram.com";
    pub const API_KEY_ENV: &'static str = "DEEPGRAM_API_KEY";

    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            speech_model: super::models::speech::AURA_2_HELENA_EN.to_string(),
            transcription_model: super::models::transcription::NOVA_3.to_string(),
            http_config: crate::defaults::http::config_default(),
            http_transport: None,
            http_interceptors: Vec::new(),
        }
    }

    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var(Self::API_KEY_ENV).map_err(|_| {
            LlmError::MissingApiKey(format!("{} is not configured", Self::API_KEY_ENV))
        })?;
        Ok(Self::new(api_key))
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = SecretString::from(api_key.into());
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    pub fn with_speech_model(mut self, model: impl Into<String>) -> Self {
        self.speech_model = model.into();
        self
    }

    pub fn with_transcription_model(mut self, model: impl Into<String>) -> Self {
        self.transcription_model = model.into();
        self
    }

    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = http_config;
        self
    }

    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.http_config.headers.extend(headers);
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.http_config.headers.insert(name.into(), value.into());
        self
    }

    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    pub fn validate(&self) -> Result<(), LlmError> {
        if self.api_key.expose_secret().trim().is_empty() {
            return Err(LlmError::MissingApiKey(
                "Deepgram API key not provided".to_string(),
            ));
        }
        if self.base_url.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Deepgram base_url cannot be empty".to_string(),
            ));
        }
        if self.speech_model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Deepgram speech_model cannot be empty".to_string(),
            ));
        }
        if self.transcription_model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Deepgram transcription_model cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}
