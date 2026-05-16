//! High-level skill upload helpers aligned with AI SDK `uploadSkill`.

use async_trait::async_trait;
use siumai_core::client::LlmClient;
use siumai_core::error::LlmError;
use siumai_core::traits::SkillsCapability;
use siumai_core::types::{
    HttpConfig, ProviderOptionsMap, SkillFileContent, SkillProviderMetadata, SkillUploadFile,
    SkillUploadRequest, SkillUploadResult,
};
use std::borrow::Cow;

/// Provider-id keyed metadata map used by `UploadSkillResult`.
pub type UploadSkillProviderMetadata = SkillProviderMetadata;

/// File content accepted by `skills::upload`.
pub type UploadSkillFileContent = SkillFileContent;

/// One skill file passed to `skills::upload`.
pub type UploadSkillFile = SkillUploadFile;

/// Resolved payload passed to `UploadSkillApi`.
pub type UploadSkillPayload = SkillUploadRequest;

/// Result returned by `skills::upload`.
pub type UploadSkillResult = SkillUploadResult;

/// Options for `skills::upload`.
#[derive(Debug, Clone, Default)]
pub struct UploadSkillOptions {
    /// Optional human-readable skill title.
    pub display_title: Option<String>,
    /// Optional provider-specific options (`providerOptions`).
    pub provider_options: ProviderOptionsMap,
    /// Optional per-request HTTP overrides.
    pub http_config: Option<HttpConfig>,
}

impl UploadSkillOptions {
    /// Create empty upload options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the display title.
    pub fn with_display_title(mut self, display_title: impl Into<String>) -> Self {
        self.display_title = Some(display_title.into());
        self
    }

    /// Replace the provider options map.
    pub fn with_provider_options(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Insert one provider option entry.
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        value: serde_json::Value,
    ) -> Self {
        self.provider_options.insert(provider_id, value);
        self
    }

    /// Set the per-request HTTP config.
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }

    /// Add one per-request header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut http_config = self.http_config.take().unwrap_or_else(HttpConfig::empty);
        http_config.headers.insert(key.into(), value.into());
        self.http_config = Some(http_config);
        self
    }
}

/// Advanced upload hook implemented by provider clients/resources.
#[async_trait]
pub trait UploadSkillApi: Send + Sync {
    /// Canonical provider id.
    fn provider_id(&self) -> Cow<'static, str>;

    /// Upload a prepared skill payload.
    async fn upload_prepared_skill(
        &self,
        payload: UploadSkillPayload,
    ) -> Result<UploadSkillResult, LlmError>;
}

/// Provider-id hook for the generic `SkillsCapability` upload adapter.
pub trait SkillUploadProvider: Send + Sync {
    /// Canonical provider id used for `providerReference`.
    fn upload_skill_provider_id(&self) -> Cow<'static, str>;
}

/// Upload a skill through a high-level API surface.
pub async fn upload<A: UploadSkillApi + ?Sized>(
    api: &A,
    files: Vec<UploadSkillFile>,
    options: UploadSkillOptions,
) -> Result<UploadSkillResult, LlmError> {
    api.upload_prepared_skill(SkillUploadRequest {
        files,
        display_title: options.display_title,
        provider_options: options.provider_options,
        http_config: options.http_config,
    })
    .await
}

#[async_trait]
impl<T> UploadSkillApi for T
where
    T: SkillsCapability + SkillUploadProvider + Send + Sync,
{
    fn provider_id(&self) -> Cow<'static, str> {
        self.upload_skill_provider_id()
    }

    async fn upload_prepared_skill(
        &self,
        payload: UploadSkillPayload,
    ) -> Result<UploadSkillResult, LlmError> {
        self.upload_skill(payload).await
    }
}

impl SkillUploadProvider for crate::compat::Siumai {
    fn upload_skill_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

impl SkillUploadProvider for crate::registry::LanguageModelHandle {
    fn upload_skill_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

#[cfg(feature = "anthropic")]
impl SkillUploadProvider for siumai_provider_anthropic::providers::anthropic::AnthropicClient {
    fn upload_skill_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

#[cfg(feature = "anthropic")]
impl SkillUploadProvider for siumai_provider_anthropic::providers::anthropic::AnthropicSkills {
    fn upload_skill_provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("anthropic")
    }
}

#[cfg(feature = "openai")]
impl SkillUploadProvider for siumai_provider_openai::providers::openai::OpenAiClient {
    fn upload_skill_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

#[cfg(feature = "openai")]
impl SkillUploadProvider for siumai_provider_openai::providers::openai::OpenAiSkills {
    fn upload_skill_provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("openai")
    }
}

#[cfg(test)]
mod tests {
    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn upload_helper_keeps_provider_policy_delegated_to_api() {
        let source = include_str!("skills.rs");
        let upload_flow = source_section(
            source,
            "pub async fn upload<A: UploadSkillApi + ?Sized>",
            "#[async_trait]\nimpl<T> UploadSkillApi for T",
        );

        for provider_literal in ["\"anthropic\"", "\"openai\"", "\"gemini\"", "\"google\""] {
            assert!(
                !upload_flow.contains(provider_literal),
                "facade skill upload helper must keep provider-specific policy delegated to provider APIs"
            );
        }
    }
}
