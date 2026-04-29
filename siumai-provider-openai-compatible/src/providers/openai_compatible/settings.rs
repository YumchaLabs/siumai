use crate::LlmError;
use crate::auth::TokenProvider;
use crate::auth::vertex::google_vertex_maas_base_url;
use crate::builder::BuilderBase;
use crate::execution::http::transport::HttpTransport;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::alibaba_video::{ALIBABA_VIDEO_DEFAULT_BASE_URL, AlibabaVideoModel};
use super::{
    AlibabaConfig, DeepInfraConfig, DeepSeekConfig, FireworksConfig, GoogleVertexMaasConfig,
    GroqConfig, MistralConfig, MoonshotAIConfig, OpenAiCompatibleBuilder, OpenAiCompatibleConfig,
    PerplexityConfig, RequestBodyTransformer, ResponseMetadataExtractor, TogetherAIConfig,
    XaiConfig,
};

const GOOGLE_VERTEX_MAAS_DEFAULT_LOCATION: &str = "global";

fn non_empty(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

fn env_non_empty(name: &str) -> Option<String> {
    non_empty(std::env::var(name).ok())
}

macro_rules! simple_compat_provider_settings {
    (
        $(#[$meta:meta])*
        pub struct $settings:ident => $config:ty, $provider_id:literal;
    ) => {
        $(#[$meta])*
        #[derive(Clone, Default)]
        pub struct $settings {
            pub api_key: Option<String>,
            pub base_url: Option<String>,
            pub headers: HashMap<String, String>,
            pub fetch: Option<Arc<dyn HttpTransport>>,
        }

        impl $settings {
            pub fn new() -> Self {
                Self::default()
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

            pub fn with_header<K: Into<String>, V: Into<String>>(
                mut self,
                name: K,
                value: V,
            ) -> Self {
                self.headers.insert(name.into(), value.into());
                self
            }

            pub fn with_fetch(mut self, fetch: Arc<dyn HttpTransport>) -> Self {
                self.fetch = Some(fetch);
                self
            }

            pub fn into_builder(self) -> OpenAiCompatibleBuilder {
                let mut builder =
                    OpenAiCompatibleBuilder::new(BuilderBase::default(), $provider_id);

                if let Some(api_key) = self.api_key {
                    builder = builder.api_key(api_key);
                }
                if let Some(base_url) = self.base_url {
                    builder = builder.base_url(base_url);
                }
                if !self.headers.is_empty() {
                    builder = builder.custom_headers(self.headers);
                }
                if let Some(fetch) = self.fetch {
                    builder = builder.fetch(fetch);
                }

                builder
            }

            pub fn into_builder_for_model<S: Into<String>>(
                self,
                model: S,
            ) -> OpenAiCompatibleBuilder {
                self.into_builder().model(model)
            }

            pub fn into_config_for_model<S: Into<String>>(
                self,
                model: S,
            ) -> Result<$config, LlmError> {
                self.into_builder_for_model(model).into_config()
            }
        }
    };
}

/// Package-level generic OpenAI-compatible provider settings aligned with
/// `repo-ref/ai/packages/openai-compatible/src/openai-compatible-provider.ts`.
///
/// This carrier intentionally uses a generic provider config even when `name` matches one of
/// Siumai's built-in provider IDs. That mirrors AI SDK's `createOpenAICompatible(...)` semantics:
/// `name` is a package-level provider label, not a request-transform preset selector.
#[derive(Clone)]
pub struct OpenAICompatibleProviderSettings {
    /// Provider name used as the provider id for the generic compat surface.
    pub name: String,
    /// Base URL for the API calls.
    pub base_url: String,
    /// Optional API key. When present, requests use `Authorization: Bearer ...`.
    pub api_key: Option<String>,
    /// Optional custom headers appended after API-key auth headers.
    pub headers: HashMap<String, String>,
    /// Optional URL query parameters appended to provider request routes.
    pub query_params: BTreeMap<String, String>,
    /// Optional custom HTTP transport, mirroring AI SDK `fetch`.
    pub fetch: Option<Arc<dyn HttpTransport>>,
    /// Include usage information in streaming responses.
    pub include_usage: Option<bool>,
    /// Whether chat models support structured outputs.
    pub supports_structured_outputs: Option<bool>,
    /// Optional request body transformer.
    pub transform_request_body: Option<Arc<dyn RequestBodyTransformer>>,
    /// Optional response metadata extractor.
    pub metadata_extractor: Option<Arc<dyn ResponseMetadataExtractor>>,
}

impl OpenAICompatibleProviderSettings {
    pub fn new<N: Into<String>, U: Into<String>>(name: N, base_url: U) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            api_key: None,
            headers: HashMap::new(),
            query_params: BTreeMap::new(),
            fetch: None,
            include_usage: None,
            supports_structured_outputs: None,
            transform_request_body: None,
            metadata_extractor: None,
        }
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
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

    pub fn with_query_params<K, V, I>(mut self, query_params: I) -> Self
    where
        K: Into<String>,
        V: Into<String>,
        I: IntoIterator<Item = (K, V)>,
    {
        self.query_params = query_params
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }

    pub fn with_query_param<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.query_params.insert(key.into(), value.into());
        self
    }

    pub fn with_fetch(mut self, fetch: Arc<dyn HttpTransport>) -> Self {
        self.fetch = Some(fetch);
        self
    }

    pub fn with_include_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = Some(include_usage);
        self
    }

    pub fn with_supports_structured_outputs(mut self, supports: bool) -> Self {
        self.supports_structured_outputs = Some(supports);
        self
    }

    pub fn with_transform_request_body(
        mut self,
        transformer: Arc<dyn RequestBodyTransformer>,
    ) -> Self {
        self.transform_request_body = Some(transformer);
        self
    }

    pub fn with_request_body_transformer(
        self,
        transformer: Arc<dyn RequestBodyTransformer>,
    ) -> Self {
        self.with_transform_request_body(transformer)
    }

    pub fn with_metadata_extractor(
        mut self,
        extractor: Arc<dyn ResponseMetadataExtractor>,
    ) -> Self {
        self.metadata_extractor = Some(extractor);
        self
    }

    pub fn into_builder(self) -> OpenAiCompatibleBuilder {
        let provider_id = if super::config::get_provider_config(&self.name).is_some() {
            format!("openai-compatible:{}", self.name)
        } else {
            self.name.clone()
        };
        let provider_config =
            super::config::generic_provider_config(&provider_id, &self.name, &self.base_url);
        let mut builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), &provider_id)
            .with_provider_config(provider_config)
            .base_url(self.base_url)
            .with_auth_required(false);

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if !self.headers.is_empty() {
            builder = builder.custom_headers(self.headers);
        }
        if !self.query_params.is_empty() {
            builder = builder.with_query_params(self.query_params);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }
        if let Some(include_usage) = self.include_usage {
            builder = builder.with_include_usage(include_usage);
        }
        if let Some(supports) = self.supports_structured_outputs {
            builder = builder.with_supports_structured_outputs(supports);
        }
        if let Some(transformer) = self.transform_request_body {
            builder = builder.with_request_body_transformer(transformer);
        }
        if let Some(extractor) = self.metadata_extractor {
            builder = builder.with_metadata_extractor(extractor);
        }

        builder
    }

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiCompatibleBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<OpenAiCompatibleConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}

simple_compat_provider_settings! {
    /// Package-level DeepSeek provider settings aligned with
    /// `repo-ref/ai/packages/deepseek/src/deepseek-provider.ts`.
    ///
    /// This carrier is model-agnostic. Model selection happens later through
    /// `into_builder_for_model(...)` or `into_config_for_model(...)`.
    pub struct DeepSeekProviderSettings => DeepSeekConfig, "deepseek";
}

simple_compat_provider_settings! {
    /// Package-level Groq provider settings aligned with
    /// `repo-ref/ai/packages/groq/src/groq-provider.ts`.
    ///
    /// This carrier is model-agnostic. Model selection happens later through
    /// `into_builder_for_model(...)` or `into_config_for_model(...)`.
    pub struct GroqProviderSettings => GroqConfig, "groq";
}

simple_compat_provider_settings! {
    /// Package-level xAI provider settings aligned with
    /// `repo-ref/ai/packages/xai/src/xai-provider.ts`.
    ///
    /// This carrier covers the shared OpenAI-compatible chat/language-model surface. Native xAI
    /// image, video, file, and responses APIs remain owned by `siumai-provider-xai`.
    pub struct XaiProviderSettings => XaiConfig, "xai";
}

simple_compat_provider_settings! {
    /// Package-level TogetherAI provider settings aligned with
    /// `repo-ref/ai/packages/togetherai/src/togetherai-provider.ts`.
    ///
    /// This carrier covers the shared OpenAI-compatible text/audio/image surface. Native
    /// TogetherAI reranking remains owned by `siumai-provider-togetherai`.
    pub struct TogetherAIProviderSettings => TogetherAIConfig, "togetherai";
}

/// Package-level DeepInfra provider settings aligned with
/// `repo-ref/ai/packages/deepinfra/src/deepinfra-provider.ts`.
///
/// This carrier is model-agnostic. Model selection happens later through
/// `into_builder_for_model(...)` or `into_config_for_model(...)`.
#[derive(Clone, Default)]
pub struct DeepInfraProviderSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl DeepInfraProviderSettings {
    pub fn new() -> Self {
        Self::default()
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

    pub fn into_builder(self) -> OpenAiCompatibleBuilder {
        let mut builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepinfra");

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if !self.headers.is_empty() {
            builder = builder.custom_headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }

        builder
    }

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiCompatibleBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<DeepInfraConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}
/// Package-level Google Vertex MaaS provider settings aligned with
/// `repo-ref/ai/packages/google-vertex/src/maas/google-vertex-maas-provider-node.ts`.
///
/// This carrier is model-agnostic. Model selection happens later through
/// `into_builder_for_model(...)` or `into_config_for_model(...)`.
#[derive(Clone, Default)]
pub struct GoogleVertexMaasProviderSettings {
    /// Google Cloud project id. Defaults to `GOOGLE_VERTEX_PROJECT` when omitted.
    pub project: Option<String>,
    /// Google Cloud location / region. Defaults to `GOOGLE_VERTEX_LOCATION`, then `global`.
    pub location: Option<String>,
    /// Optional base URL override. If omitted, project/location derive the OpenAPI MaaS base URL.
    pub base_url: Option<String>,
    /// Default headers applied to requests built from this settings object.
    pub headers: HashMap<String, String>,
    /// Optional custom HTTP transport, mirroring AI SDK `fetch`.
    pub fetch: Option<Arc<dyn HttpTransport>>,
    /// Rust-side auth analogue for the Node-only Google auth wrapper.
    pub token_provider: Option<Arc<dyn TokenProvider>>,
}

impl GoogleVertexMaasProviderSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    pub fn with_location<S: Into<String>>(mut self, location: S) -> Self {
        self.location = Some(location.into());
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

    pub fn with_token_provider(mut self, token_provider: Arc<dyn TokenProvider>) -> Self {
        self.token_provider = Some(token_provider);
        self
    }

    fn resolved_base_url(&self) -> Result<String, LlmError> {
        if let Some(base_url) = non_empty(self.base_url.clone()) {
            return Ok(base_url);
        }

        let project = non_empty(self.project.clone())
            .or_else(|| env_non_empty("GOOGLE_VERTEX_PROJECT"))
            .ok_or_else(|| {
                LlmError::ConfigurationError(
                    "Google Vertex MaaS requires `project`, `base_url`, or GOOGLE_VERTEX_PROJECT"
                        .to_string(),
                )
            })?;
        let location = non_empty(self.location.clone())
            .or_else(|| env_non_empty("GOOGLE_VERTEX_LOCATION"))
            .unwrap_or_else(|| GOOGLE_VERTEX_MAAS_DEFAULT_LOCATION.to_string());

        Ok(google_vertex_maas_base_url(&project, &location))
    }

    pub fn into_builder(self) -> OpenAiCompatibleBuilder {
        let base_url = self.resolved_base_url().ok();
        let mut builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "vertex-maas");

        if let Some(base_url) = base_url {
            builder = builder.base_url(base_url);
        }
        if !self.headers.is_empty() {
            builder = builder.custom_headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }
        if let Some(token_provider) = self.token_provider {
            builder = builder.with_token_provider(token_provider);
        }

        builder
    }

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiCompatibleBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<GoogleVertexMaasConfig, LlmError> {
        let base_url = self.resolved_base_url()?;
        let mut builder = self.into_builder_for_model(model);
        builder = builder.base_url(base_url);
        builder.into_config()
    }
}
/// Package-level Alibaba provider settings aligned with
/// `repo-ref/ai/packages/alibaba/src/alibaba-provider.ts`.
#[derive(Clone, Default)]
pub struct AlibabaProviderSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub video_base_url: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
    pub include_usage: Option<bool>,
}

impl AlibabaProviderSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_video_base_url<S: Into<String>>(mut self, video_base_url: S) -> Self {
        self.video_base_url = Some(video_base_url.into());
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

    pub fn with_include_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = Some(include_usage);
        self
    }

    pub fn into_builder(self) -> OpenAiCompatibleBuilder {
        let mut builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "alibaba")
            .with_include_usage(self.include_usage.unwrap_or(true));

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if !self.headers.is_empty() {
            builder = builder.custom_headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }

        builder
    }

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiCompatibleBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<AlibabaConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }

    pub fn into_video_model<S: Into<String>>(self, model: S) -> AlibabaVideoModel {
        let api_key = self
            .api_key
            .or_else(|| env_non_empty("ALIBABA_API_KEY"))
            .or_else(|| env_non_empty("DASHSCOPE_API_KEY"))
            .or_else(|| env_non_empty("QWEN_API_KEY"));
        let mut video_model = AlibabaVideoModel::new(model)
            .with_base_url(
                self.video_base_url
                    .unwrap_or_else(|| ALIBABA_VIDEO_DEFAULT_BASE_URL.to_string()),
            )
            .with_headers(self.headers);

        if let Some(api_key) = api_key {
            video_model = video_model.with_api_key(api_key);
        }
        if let Some(fetch) = self.fetch {
            video_model = video_model.with_fetch(fetch);
        }

        video_model
    }
}

/// Package-level MoonshotAI provider settings aligned with
/// `repo-ref/ai/packages/moonshotai/src/moonshotai-provider.ts`.
///
/// This carrier is model-agnostic. Model selection happens later through
/// `into_builder_for_model(...)` or `into_config_for_model(...)`.
#[derive(Clone, Default)]
pub struct MoonshotAIProviderSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl MoonshotAIProviderSettings {
    pub fn new() -> Self {
        Self::default()
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

    pub fn into_builder(self) -> OpenAiCompatibleBuilder {
        let mut builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "moonshotai");

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if !self.headers.is_empty() {
            builder = builder.custom_headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }

        builder
    }

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiCompatibleBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<MoonshotAIConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}
/// Package-level Fireworks provider settings aligned with
/// `repo-ref/ai/packages/fireworks/src/fireworks-provider.ts`.
///
/// This carrier is model-agnostic. Model selection happens later through
/// `into_builder_for_model(...)` or `into_config_for_model(...)`.
#[derive(Clone, Default)]
pub struct FireworksProviderSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl FireworksProviderSettings {
    pub fn new() -> Self {
        Self::default()
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

    pub fn into_builder(self) -> OpenAiCompatibleBuilder {
        let mut builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "fireworks");

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if !self.headers.is_empty() {
            builder = builder.custom_headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }

        builder
    }

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiCompatibleBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<FireworksConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}
/// Package-level Mistral provider settings aligned with
/// `repo-ref/ai/packages/mistral/src/mistral-provider.ts`.
///
/// This carrier is model-agnostic. Model selection happens later through
/// `into_builder_for_model(...)` or `into_config_for_model(...)`.
///
/// Note: upstream `generateId` is intentionally deferred until the shared compat runtime owns an
/// honest provider-level stable-id hook.
#[derive(Clone, Default)]
pub struct MistralProviderSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl MistralProviderSettings {
    pub fn new() -> Self {
        Self::default()
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

    pub fn into_builder(self) -> OpenAiCompatibleBuilder {
        let mut builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "mistral");

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if !self.headers.is_empty() {
            builder = builder.custom_headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }

        builder
    }

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiCompatibleBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<MistralConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}

/// Package-level Perplexity provider settings aligned with
/// `repo-ref/ai/packages/perplexity/src/perplexity-provider.ts`.
///
/// This carrier is model-agnostic. Model selection happens later through
/// `into_builder_for_model(...)` or `into_config_for_model(...)`.
#[derive(Clone, Default)]
pub struct PerplexityProviderSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl PerplexityProviderSettings {
    pub fn new() -> Self {
        Self::default()
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

    pub fn into_builder(self) -> OpenAiCompatibleBuilder {
        let mut builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "perplexity");

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if !self.headers.is_empty() {
            builder = builder.custom_headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }

        builder
    }

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiCompatibleBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<PerplexityConfig, LlmError> {
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
    use siumai_core::traits::ModelMetadata;

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
    fn openai_compatible_provider_settings_into_config_preserve_supported_inputs() {
        let config = OpenAICompatibleProviderSettings::new("acme", "https://example.com/v1")
            .with_api_key("test-key")
            .with_header("x-test", "1")
            .with_query_param("api-version", "2025-04-01")
            .with_include_usage(true)
            .with_supports_structured_outputs(false)
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("acme-chat")
            .expect("settings into config");

        assert_eq!(config.provider_id, "acme");
        assert_eq!(config.base_url, "https://example.com/v1");
        assert_eq!(config.common_params.model, "acme-chat");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            config.query_params.get("api-version").map(String::as_str),
            Some("2025-04-01")
        );
        assert_eq!(config.include_usage, Some(true));
        assert_eq!(config.supports_structured_outputs, Some(false));
        assert!(!config.auth_required);
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn openai_compatible_provider_settings_allow_unauthenticated_gateway() {
        let config = OpenAICompatibleProviderSettings::new("local", "http://localhost:11434/v1")
            .into_config_for_model("llama3.2")
            .expect("settings into config");

        assert_eq!(config.provider_id, "local");
        assert!(config.api_key.is_empty());
        assert!(!config.auth_required);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn openai_compatible_provider_settings_do_not_reuse_builtin_preset_on_name_collision() {
        let config = OpenAICompatibleProviderSettings::new("groq", "https://example.com/v1")
            .into_config_for_model("custom-groq-model")
            .expect("settings into config");

        assert_eq!(config.provider_id, "openai-compatible:groq");
        assert_eq!(config.base_url, "https://example.com/v1");
        assert_eq!(config.adapter.base_url(), "https://example.com/v1");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn deepseek_provider_settings_into_config_preserve_supported_inputs() {
        let config = DeepSeekProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/deepseek")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("deepseek-chat")
            .expect("settings into config");

        assert_eq!(config.provider_id, "deepseek");
        assert_eq!(config.base_url, "https://example.com/deepseek");
        assert_eq!(config.common_params.model, "deepseek-chat");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn groq_provider_settings_into_config_preserve_supported_inputs() {
        let config = GroqProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/groq/openai/v1")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("openai/gpt-oss-20b")
            .expect("settings into config");

        assert_eq!(config.provider_id, "groq");
        assert_eq!(config.base_url, "https://example.com/groq/openai/v1");
        assert_eq!(config.common_params.model, "openai/gpt-oss-20b");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn xai_provider_settings_into_config_preserve_supported_inputs() {
        let config = XaiProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/xai/v1")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("grok-4")
            .expect("settings into config");

        assert_eq!(config.provider_id, "xai");
        assert_eq!(config.base_url, "https://example.com/xai/v1");
        assert_eq!(config.common_params.model, "grok-4");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn togetherai_provider_settings_into_config_preserve_supported_inputs() {
        let config = TogetherAIProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/together/v1")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("meta-llama/Llama-3.3-70B-Instruct-Turbo")
            .expect("settings into config");

        assert_eq!(config.provider_id, "togetherai");
        assert_eq!(config.base_url, "https://example.com/together/v1");
        assert_eq!(
            config.common_params.model,
            "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        );
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn mistral_provider_settings_into_config_preserve_supported_inputs() {
        let config = MistralProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/mistral")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("mistral-large-latest")
            .expect("settings into config");

        assert_eq!(config.provider_id, "mistral");
        assert_eq!(config.base_url, "https://example.com/mistral");
        assert_eq!(config.common_params.model, "mistral-large-latest");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn perplexity_provider_settings_into_config_preserve_supported_inputs() {
        let config = PerplexityProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/perplexity")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("sonar")
            .expect("settings into config");

        assert_eq!(config.provider_id, "perplexity");
        assert_eq!(config.base_url, "https://example.com/perplexity");
        assert_eq!(config.common_params.model, "sonar");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn fireworks_provider_settings_into_config_preserve_supported_inputs() {
        let config = FireworksProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/fireworks")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("accounts/fireworks/models/llama-v3p1-8b-instruct")
            .expect("settings into config");

        assert_eq!(config.provider_id, "fireworks");
        assert_eq!(config.base_url, "https://example.com/fireworks");
        assert_eq!(
            config.common_params.model,
            "accounts/fireworks/models/llama-v3p1-8b-instruct"
        );
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn moonshotai_provider_settings_into_config_preserve_supported_inputs() {
        let config = MoonshotAIProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/moonshot")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("kimi-k2.5")
            .expect("settings into config");

        assert_eq!(config.provider_id, "moonshotai");
        assert_eq!(config.base_url, "https://example.com/moonshot");
        assert_eq!(config.common_params.model, "kimi-k2.5");
        assert_eq!(config.include_usage, Some(true));
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn alibaba_provider_settings_into_config_preserve_supported_inputs() {
        let config = AlibabaProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/alibaba/compatible-mode/v1")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("qwen-plus")
            .expect("settings into config");

        assert_eq!(config.provider_id, "alibaba");
        assert_eq!(
            config.base_url,
            "https://example.com/alibaba/compatible-mode/v1"
        );
        assert_eq!(config.common_params.model, "qwen-plus");
        assert_eq!(config.include_usage, Some(true));
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());

        let no_usage = AlibabaProviderSettings::new()
            .with_api_key("test-key")
            .with_include_usage(false)
            .into_config_for_model("qwen-plus")
            .expect("settings into config");
        assert_eq!(no_usage.include_usage, Some(false));

        let video_model = AlibabaProviderSettings::new()
            .with_api_key("test-key")
            .with_video_base_url("https://example.com/dashscope")
            .with_header("x-video", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_video_model("wan2.6-t2v");
        assert_eq!(video_model.provider_id(), "alibaba.video");
        assert_eq!(video_model.model_id(), "wan2.6-t2v");
        assert_eq!(video_model.base_url(), "https://example.com/dashscope");
    }

    #[test]
    fn deepinfra_provider_settings_into_config_preserve_supported_inputs() {
        let config = DeepInfraProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/deepinfra")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("meta-llama/Llama-3.3-70B-Instruct")
            .expect("settings into config");

        assert_eq!(config.provider_id, "deepinfra");
        assert_eq!(config.base_url, "https://example.com/deepinfra/openai");
        assert_eq!(
            config.common_params.model,
            "meta-llama/Llama-3.3-70B-Instruct"
        );
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn google_vertex_maas_provider_settings_into_config_preserve_supported_inputs() {
        let config = GoogleVertexMaasProviderSettings::new()
            .with_project("test-project")
            .with_location("us-central1")
            .with_header("Authorization", "Bearer test-token")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("deepseek-ai/deepseek-v3.2-maas")
            .expect("settings into config");

        assert_eq!(config.provider_id, "vertex-maas");
        assert_eq!(
            config.base_url,
            "https://aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/endpoints/openapi"
        );
        assert_eq!(config.common_params.model, "deepseek-ai/deepseek-v3.2-maas");
        assert_eq!(
            config
                .http_config
                .headers
                .get("Authorization")
                .map(String::as_str),
            Some("Bearer test-token")
        );
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn google_vertex_maas_provider_settings_support_token_provider_auth() {
        let config = GoogleVertexMaasProviderSettings::new()
            .with_project("test-project")
            .with_token_provider(Arc::new(crate::auth::StaticTokenProvider::new(
                "test-token",
            )))
            .into_config_for_model("deepseek-ai/deepseek-v3.2-maas")
            .expect("settings into config");

        assert_eq!(config.provider_id, "vertex-maas");
        assert_eq!(
            config.base_url,
            "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/endpoints/openapi"
        );
        assert!(config.api_key.is_empty());
        assert!(config.token_provider.is_some());
    }
}
