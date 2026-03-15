//! `Bedrock` configuration.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::types::{CommonParams, HttpConfig};
use secrecy::{ExposeSecret, SecretString};
use std::sync::Arc;

/// Provider-owned config-first surface for Amazon Bedrock.
#[derive(Clone)]
pub struct BedrockConfig {
    /// Optional bearer token auth. AWS SigV4 headers can also be injected via `http_config.headers`.
    pub api_key: Option<SecretString>,
    /// Runtime endpoint used by Converse / ConverseStream.
    pub runtime_base_url: String,
    /// Agent runtime endpoint used by reranking.
    pub agent_runtime_base_url: String,
    /// Default AWS region used for derived endpoints and rerank ARN shaping.
    pub region: String,
    /// Shared default chat parameters.
    pub common_params: CommonParams,
    /// Optional default rerank model id.
    pub default_rerank_model: Option<String>,
    /// HTTP configuration.
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport.
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    /// Optional HTTP interceptors applied to all requests.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
}

impl Default for BedrockConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BedrockConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("BedrockConfig");
        ds.field("runtime_base_url", &self.runtime_base_url)
            .field("agent_runtime_base_url", &self.agent_runtime_base_url)
            .field("region", &self.region)
            .field("common_params", &self.common_params)
            .field("default_rerank_model", &self.default_rerank_model)
            .field("http_config", &self.http_config);

        if self
            .api_key
            .as_ref()
            .is_some_and(|api_key| !api_key.expose_secret().trim().is_empty())
        {
            ds.field("has_api_key", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }
        if !self.http_interceptors.is_empty() {
            ds.field("http_interceptors_len", &self.http_interceptors.len());
        }

        ds.finish()
    }
}

impl BedrockConfig {
    /// Default AWS region for Bedrock examples and tests.
    pub const DEFAULT_REGION: &'static str = "us-east-1";

    /// Create a new config with the default `us-east-1` endpoint pair.
    pub fn new() -> Self {
        let region = Self::DEFAULT_REGION.to_string();
        Self {
            api_key: None,
            runtime_base_url: Self::runtime_base_url_for_region(&region),
            agent_runtime_base_url: Self::agent_runtime_base_url_for_region(&region),
            region,
            common_params: CommonParams::default(),
            default_rerank_model: None,
            http_config: HttpConfig::default(),
            http_transport: None,
            http_interceptors: Vec::new(),
        }
    }

    /// Create config using `AWS_REGION` / `AWS_DEFAULT_REGION` and optional `BEDROCK_API_KEY`.
    pub fn from_env() -> Self {
        let region = std::env::var("AWS_REGION")
            .ok()
            .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| Self::DEFAULT_REGION.to_string());

        let mut config = Self::new().with_region(region);
        if let Ok(api_key) = std::env::var("BEDROCK_API_KEY") {
            config = config.with_api_key(api_key);
        }
        config
    }

    /// Build the Bedrock Runtime endpoint for a region.
    pub fn runtime_base_url_for_region(region: &str) -> String {
        format!("https://bedrock-runtime.{}.amazonaws.com", region.trim())
    }

    /// Build the Bedrock Agent Runtime endpoint for a region.
    pub fn agent_runtime_base_url_for_region(region: &str) -> String {
        format!(
            "https://bedrock-agent-runtime.{}.amazonaws.com",
            region.trim()
        )
    }

    fn extract_region_from_base_url(base_url: &str) -> Option<String> {
        ["bedrock-runtime.", "bedrock-agent-runtime."]
            .into_iter()
            .find_map(|marker| {
                let (_, tail) = base_url.split_once(marker)?;
                let (region, _) = tail.split_once(".amazonaws.com")?;
                let trimmed = region.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
    }

    fn paired_base_urls(base_url: &str) -> (String, String, Option<String>) {
        let normalized = base_url.trim().trim_end_matches('/').to_string();
        if normalized.contains("bedrock-runtime.") {
            let agent = normalized.replacen("bedrock-runtime.", "bedrock-agent-runtime.", 1);
            let region = Self::extract_region_from_base_url(&normalized);
            return (normalized, agent, region);
        }
        if normalized.contains("bedrock-agent-runtime.") {
            let runtime = normalized.replacen("bedrock-agent-runtime.", "bedrock-runtime.", 1);
            let region = Self::extract_region_from_base_url(&normalized);
            return (runtime, normalized, region);
        }
        (normalized.clone(), normalized, None)
    }

    /// Set the optional bearer token.
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(SecretString::from(api_key.into()));
        self
    }

    /// Set both runtime and agent runtime URLs from a single base URL.
    ///
    /// If the URL matches a standard Bedrock runtime host, the counterpart host is derived.
    /// Otherwise the same base URL is reused for both capabilities.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        let base_url = base_url.into();
        let (runtime_base_url, agent_runtime_base_url, region) = Self::paired_base_urls(&base_url);
        self.runtime_base_url = runtime_base_url;
        self.agent_runtime_base_url = agent_runtime_base_url;
        if let Some(region) = region {
            self.region = region;
        }
        self
    }

    /// Set the Bedrock Runtime endpoint explicitly.
    pub fn with_runtime_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.runtime_base_url = base_url.into().trim().trim_end_matches('/').to_string();
        if let Some(region) = Self::extract_region_from_base_url(&self.runtime_base_url) {
            self.region = region;
        }
        self
    }

    /// Set the Bedrock Agent Runtime endpoint explicitly.
    pub fn with_agent_runtime_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.agent_runtime_base_url = base_url.into().trim().trim_end_matches('/').to_string();
        if let Some(region) = Self::extract_region_from_base_url(&self.agent_runtime_base_url) {
            self.region = region;
        }
        self
    }

    /// Set the default AWS region and regenerate both standard Bedrock endpoints.
    pub fn with_region<S: Into<String>>(mut self, region: S) -> Self {
        let region = region.into();
        let region = if region.trim().is_empty() {
            Self::DEFAULT_REGION.to_string()
        } else {
            region.trim().to_string()
        };
        self.runtime_base_url = Self::runtime_base_url_for_region(&region);
        self.agent_runtime_base_url = Self::agent_runtime_base_url_for_region(&region);
        self.region = region;
        self
    }

    /// Set the default chat model id.
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set the default rerank model id.
    pub fn with_rerank_model<S: Into<String>>(mut self, model: S) -> Self {
        self.default_rerank_model = Some(model.into());
        self
    }

    pub fn with_http_config(mut self, http: HttpConfig) -> Self {
        self.http_config = http;
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
        if self.runtime_base_url.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Bedrock runtime_base_url cannot be empty".to_string(),
            ));
        }
        if self.agent_runtime_base_url.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Bedrock agent_runtime_base_url cannot be empty".to_string(),
            ));
        }
        if self.region.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Bedrock region cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_base_url_derives_runtime_and_agent_runtime_pair() {
        let config =
            BedrockConfig::new().with_base_url("https://bedrock-runtime.us-west-2.amazonaws.com");

        assert_eq!(config.region, "us-west-2");
        assert_eq!(
            config.runtime_base_url,
            "https://bedrock-runtime.us-west-2.amazonaws.com"
        );
        assert_eq!(
            config.agent_runtime_base_url,
            "https://bedrock-agent-runtime.us-west-2.amazonaws.com"
        );
    }

    #[test]
    fn with_agent_runtime_base_url_updates_region_when_possible() {
        let config = BedrockConfig::new().with_agent_runtime_base_url(
            "https://bedrock-agent-runtime.eu-central-1.amazonaws.com",
        );

        assert_eq!(config.region, "eu-central-1");
    }
}
