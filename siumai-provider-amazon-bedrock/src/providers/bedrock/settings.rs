use crate::LlmError;
use crate::builder::BuilderBase;
use crate::execution::http::transport::HttpTransport;
use std::collections::HashMap;
use std::sync::Arc;

use super::{BedrockBuilder, BedrockConfig};

/// Package-level provider settings aligned with the supported subset of
/// `repo-ref/ai/packages/amazon-bedrock/src/bedrock-provider.ts`.
///
/// Unlike `BedrockConfig`, this carrier intentionally does not require a model id.
/// Model selection happens later through `into_builder_for_model(...)`.
///
/// Note: the upstream AWS-credential-provider fields (`accessKeyId`, `secretAccessKey`,
/// `sessionToken`, `credentialProvider`) and test-only `generateId` hook are intentionally
/// deferred here. Siumai's current Bedrock runtime accepts bearer auth or pre-signed/custom
/// HTTP transport/header injection, but it does not yet own a first-class SigV4 credential
/// provider abstraction comparable to the upstream package.
#[derive(Clone, Default)]
pub struct AmazonBedrockProviderSettings {
    pub region: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl AmazonBedrockProviderSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_region<S: Into<String>>(mut self, region: S) -> Self {
        self.region = Some(region.into());
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, name: K, value: V) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    pub fn with_fetch(mut self, fetch: Arc<dyn HttpTransport>) -> Self {
        self.fetch = Some(fetch);
        self
    }

    /// Convert package-level provider settings into the provider-owned builder surface.
    pub fn into_builder(self) -> BedrockBuilder {
        let mut builder = BedrockBuilder::new(BuilderBase::default());

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(region) = self.region {
            builder = builder.region(region);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if !self.headers.is_empty() {
            builder = builder.headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }

        builder
    }

    /// Convert package-level provider settings into a builder with a selected model id.
    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> BedrockBuilder {
        self.into_builder().model(model)
    }

    /// Convert package-level provider settings into the config-first carrier for a specific model.
    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<BedrockConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransportGetRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use async_trait::async_trait;
    use reqwest::header::HeaderMap;

    #[derive(Clone, Default)]
    struct NoopTransport;

    #[async_trait]
    impl HttpTransport for NoopTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Ok(HttpTransportResponse {
                status: 200,
                headers: HeaderMap::new(),
                body: b"{}".to_vec(),
            })
        }

        async fn execute_get(
            &self,
            _request: HttpTransportGetRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Ok(HttpTransportResponse {
                status: 200,
                headers: HeaderMap::new(),
                body: b"{}".to_vec(),
            })
        }
    }

    #[test]
    fn amazon_bedrock_provider_settings_into_config_preserve_supported_inputs() {
        let config = AmazonBedrockProviderSettings::new()
            .with_api_key("test-key")
            .with_region("us-west-2")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("amazon.nova-lite-v1:0")
            .expect("settings into config");

        assert_eq!(config.region, "us-west-2");
        assert_eq!(
            config.runtime_base_url,
            "https://bedrock-runtime.us-west-2.amazonaws.com"
        );
        assert_eq!(
            config.agent_runtime_base_url,
            "https://bedrock-agent-runtime.us-west-2.amazonaws.com"
        );
        assert_eq!(config.common_params.model, "amazon.nova-lite-v1:0");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn amazon_bedrock_provider_settings_base_url_override_still_derives_paired_hosts() {
        let config = AmazonBedrockProviderSettings::new()
            .with_base_url("https://bedrock-runtime.eu-central-1.amazonaws.com")
            .into_config_for_model("amazon.nova-lite-v1:0")
            .expect("settings into config");

        assert_eq!(config.region, "eu-central-1");
        assert_eq!(
            config.agent_runtime_base_url,
            "https://bedrock-agent-runtime.eu-central-1.amazonaws.com"
        );
    }
}
