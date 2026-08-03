use std::fmt;
use std::sync::Arc;

use siumai_core::{
    ApiModeId, ApiStability, GenericSupportClaim, ModelFamily, PlatformId, ProfileId, ProtocolId,
    ProviderId, ProviderProfile, ProviderScope, SupportScope,
};
use siumai_protocol_openai::chat_completions::{API_MODE_ID, ChatCompletionsDialect, PROTOCOL_ID};
use siumai_transport::{EndpointConfig, EndpointError, EndpointPolicy};

use super::provider::OpenAiCompatibleConfigError;

/// One immutable endpoint, protocol dialect, and evidence-backed support profile.
#[derive(Clone)]
pub struct OpenAiCompatibleProfile {
    profile: Arc<ProviderProfile>,
    support_scope: SupportScope,
    scope: Arc<ProviderScope>,
    endpoint: EndpointConfig,
    dialect: ChatCompletionsDialect,
}

impl OpenAiCompatibleProfile {
    pub fn verified(
        profile: ProviderProfile,
        endpoint: EndpointConfig,
        dialect: ChatCompletionsDialect,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        let claims = profile
            .verified_claims()
            .ok_or(OpenAiCompatibleConfigError::ExpectedVerifiedProfile)?;
        if claims.len() != 1 {
            return Err(OpenAiCompatibleConfigError::ExpectedSingleLanguageClaim);
        }
        if !matches!(endpoint.policy(), EndpointPolicy::Official(_)) {
            return Err(OpenAiCompatibleConfigError::VerifiedEndpointMustBeOfficial);
        }
        let support_scope = claims[0].scope().clone();
        Self::from_parts(profile, support_scope, endpoint, dialect)
    }

    pub fn public_custom(
        provider: ProviderId,
        base_url: impl AsRef<str>,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        Self::generic(provider, base_url, false)
    }

    pub fn local_explicit(
        provider: ProviderId,
        base_url: impl AsRef<str>,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        Self::generic(provider, base_url, true)
    }

    fn generic(
        provider: ProviderId,
        base_url: impl AsRef<str>,
        local: bool,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        let support_scope = SupportScope::new(
            provider.clone(),
            PlatformId::new(if local { "local" } else { "custom-endpoint" })?,
            ModelFamily::Language,
            ProtocolId::new(PROTOCOL_ID)?,
            ApiModeId::new(API_MODE_ID)?,
        );
        let profile = ProviderProfile::generic(
            ProfileId::new(provider.as_str())?,
            GenericSupportClaim::new(support_scope.clone(), ApiStability::Experimental),
        );
        let endpoint = if local {
            EndpointConfig::local_explicit(base_url)
        } else {
            EndpointConfig::public_custom(base_url)
        }?;
        Self::from_parts(
            profile,
            support_scope,
            endpoint,
            ChatCompletionsDialect::generic(),
        )
    }

    fn from_parts(
        profile: ProviderProfile,
        support_scope: SupportScope,
        endpoint: EndpointConfig,
        dialect: ChatCompletionsDialect,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        if support_scope.family() != ModelFamily::Language
            || support_scope.protocol().as_str() != PROTOCOL_ID
            || support_scope.api_mode().as_str() != API_MODE_ID
        {
            return Err(OpenAiCompatibleConfigError::IncompatibleSupportScope);
        }
        let scope = Arc::new(
            ProviderScope::new(support_scope.provider().clone())
                .with_platform(support_scope.platform().clone())
                .with_protocol(support_scope.protocol().clone())
                .with_api_mode(support_scope.api_mode().clone()),
        );
        Ok(Self {
            profile: Arc::new(profile),
            support_scope,
            scope,
            endpoint,
            dialect,
        })
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub fn scope(&self) -> &Arc<ProviderScope> {
        &self.scope
    }

    pub(crate) fn support_scope(&self) -> &SupportScope {
        &self.support_scope
    }

    pub(crate) fn profile_arc(&self) -> Arc<ProviderProfile> {
        self.profile.clone()
    }

    pub(crate) fn endpoint(&self) -> &EndpointConfig {
        &self.endpoint
    }

    pub(crate) fn dialect(&self) -> &ChatCompletionsDialect {
        &self.dialect
    }
}

impl fmt::Debug for OpenAiCompatibleProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleProfile")
            .field("profile_id", self.profile.id())
            .field("scope", &self.scope)
            .field("endpoint", &self.endpoint)
            .field("dialect", &self.dialect)
            .finish()
    }
}

impl From<EndpointError> for OpenAiCompatibleConfigError {
    fn from(source: EndpointError) -> Self {
        Self::Endpoint(source)
    }
}
