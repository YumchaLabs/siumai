use std::fmt;
use std::sync::Arc;

use siumai_core::{
    ApiModeId, ApiStability, GenericSupportClaim, ModelFamily, PlatformId, ProfileId, ProtocolId,
    ProviderId, ProviderProfile, ProviderScope, SupportScope,
};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID as CHAT_API_MODE_ID, ChatCompletionsDialect, PROTOCOL_ID as CHAT_PROTOCOL_ID,
};
use siumai_protocol_openai::responses_next::{
    API_MODE_ID as RESPONSES_API_MODE_ID, OPENAI_RESPONSES_PROTOCOL,
};
use siumai_transport::{EndpointConfig, EndpointError, EndpointPolicy};

use super::codec_policy::{
    ChatCodecPolicy, IdentityChatCodecPolicy, IdentityResponsesCodecPolicy, ResponsesCodecPolicy,
};
use super::mode::OpenAiCompatibleApiMode;
use super::provider::OpenAiCompatibleConfigError;

#[derive(Clone)]
pub(crate) struct ChatModeProfile {
    support_scope: SupportScope,
    scope: Arc<ProviderScope>,
    dialect: ChatCompletionsDialect,
    codec_policy: Arc<dyn ChatCodecPolicy>,
}

#[derive(Clone)]
pub(crate) struct ResponsesModeProfile {
    support_scope: SupportScope,
    scope: Arc<ProviderScope>,
    codec_policy: Arc<dyn ResponsesCodecPolicy>,
}

/// Fully validated execution data captured by a language-model handle.
///
/// Keeping the scope and codec together prevents the handle from re-deriving mode-specific
/// invariants from the provider profile during every call.
#[derive(Clone)]
pub(crate) enum LanguageModeProfile {
    ChatCompletions {
        scope: Arc<ProviderScope>,
        dialect: ChatCompletionsDialect,
        codec_policy: Arc<dyn ChatCodecPolicy>,
    },
    Responses {
        scope: Arc<ProviderScope>,
        codec_policy: Arc<dyn ResponsesCodecPolicy>,
    },
}

impl LanguageModeProfile {
    pub(crate) const fn api_mode(&self) -> OpenAiCompatibleApiMode {
        match self {
            Self::ChatCompletions { .. } => OpenAiCompatibleApiMode::ChatCompletions,
            Self::Responses { .. } => OpenAiCompatibleApiMode::Responses,
        }
    }

    pub(crate) fn scope(&self) -> &Arc<ProviderScope> {
        match self {
            Self::ChatCompletions { scope, .. } | Self::Responses { scope, .. } => scope,
        }
    }
}

/// One immutable endpoint and evidence-backed set of OpenAI-family language modes.
#[derive(Clone)]
pub struct OpenAiCompatibleProfile {
    profile: Arc<ProviderProfile>,
    endpoint: EndpointConfig,
    chat: Option<ChatModeProfile>,
    responses: Option<ResponsesModeProfile>,
    recommended_mode: OpenAiCompatibleApiMode,
    recommended_scope: Arc<ProviderScope>,
}

impl OpenAiCompatibleProfile {
    pub fn verified_chat(
        profile: ProviderProfile,
        endpoint: EndpointConfig,
        dialect: ChatCompletionsDialect,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        Self::verified_modes(profile, endpoint, Some(dialect), false)
    }

    pub fn verified_responses(
        profile: ProviderProfile,
        endpoint: EndpointConfig,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        Self::verified_modes(profile, endpoint, None, true)
    }

    pub fn verified_chat_and_responses(
        profile: ProviderProfile,
        endpoint: EndpointConfig,
        chat_dialect: ChatCompletionsDialect,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        Self::verified_modes(profile, endpoint, Some(chat_dialect), true)
    }

    fn verified_modes(
        profile: ProviderProfile,
        endpoint: EndpointConfig,
        chat_dialect: Option<ChatCompletionsDialect>,
        responses: bool,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        let claims = profile
            .verified_claims()
            .ok_or(OpenAiCompatibleConfigError::ExpectedVerifiedProfile)?;
        if !matches!(endpoint.policy(), EndpointPolicy::Official(_)) {
            return Err(OpenAiCompatibleConfigError::VerifiedEndpointMustBeOfficial);
        }

        let mut chat_scope = None;
        let mut responses_scope = None;
        for claim in claims {
            match mode_from_support_scope(claim.scope())? {
                OpenAiCompatibleApiMode::ChatCompletions => {
                    if chat_scope.replace(claim.scope().clone()).is_some() {
                        return Err(OpenAiCompatibleConfigError::DuplicateLanguageModeClaim);
                    }
                }
                OpenAiCompatibleApiMode::Responses => {
                    if responses_scope.replace(claim.scope().clone()).is_some() {
                        return Err(OpenAiCompatibleConfigError::DuplicateLanguageModeClaim);
                    }
                }
            }
        }
        if chat_scope.is_some() != chat_dialect.is_some() || responses_scope.is_some() != responses
        {
            return Err(OpenAiCompatibleConfigError::LanguageModeClaimMismatch);
        }

        Self::from_parts(
            profile,
            endpoint,
            chat_scope.zip(chat_dialect),
            responses_scope,
        )
    }

    pub fn public_custom(
        provider: ProviderId,
        base_url: impl AsRef<str>,
        mode: OpenAiCompatibleApiMode,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        Self::generic(provider, base_url, false, mode)
    }

    pub fn local_explicit(
        provider: ProviderId,
        base_url: impl AsRef<str>,
        mode: OpenAiCompatibleApiMode,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        Self::generic(provider, base_url, true, mode)
    }

    #[doc(hidden)]
    pub fn custom_chat(
        provider: ProviderId,
        endpoint: EndpointConfig,
        chat_dialect: ChatCompletionsDialect,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        let platform = PlatformId::new(match endpoint.policy() {
            EndpointPolicy::LocalExplicit(_) => "local",
            EndpointPolicy::PublicCustom => "custom-endpoint",
            EndpointPolicy::Official(_) => "official-custom-endpoint",
            _ => "custom-endpoint",
        })?;
        let chat_scope = SupportScope::new(
            provider.clone(),
            platform,
            ModelFamily::Language,
            ProtocolId::new(CHAT_PROTOCOL_ID)?,
            ApiModeId::new(CHAT_API_MODE_ID)?,
        );
        let profile = ProviderProfile::generic(
            ProfileId::new(provider.as_str())?,
            GenericSupportClaim::new(chat_scope.clone(), ApiStability::Experimental),
        );
        Self::from_parts(profile, endpoint, Some((chat_scope, chat_dialect)), None)
    }

    #[doc(hidden)]
    pub fn custom_chat_and_responses(
        provider: ProviderId,
        endpoint: EndpointConfig,
        chat_dialect: ChatCompletionsDialect,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        let platform = PlatformId::new(match endpoint.policy() {
            EndpointPolicy::LocalExplicit(_) => "local",
            EndpointPolicy::PublicCustom => "custom-endpoint",
            EndpointPolicy::Official(_) => "official-custom-endpoint",
            _ => "custom-endpoint",
        })?;
        let chat_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Language,
            ProtocolId::new(CHAT_PROTOCOL_ID)?,
            ApiModeId::new(CHAT_API_MODE_ID)?,
        );
        let responses_scope = SupportScope::new(
            provider.clone(),
            platform,
            ModelFamily::Language,
            ProtocolId::new(OPENAI_RESPONSES_PROTOCOL)?,
            ApiModeId::new(RESPONSES_API_MODE_ID)?,
        );
        let profile = ProviderProfile::generic_many(
            ProfileId::new(provider.as_str())?,
            vec![
                GenericSupportClaim::new(chat_scope.clone(), ApiStability::Experimental),
                GenericSupportClaim::new(responses_scope.clone(), ApiStability::Experimental),
            ],
        )?;
        Self::from_parts(
            profile,
            endpoint,
            Some((chat_scope, chat_dialect)),
            Some(responses_scope),
        )
    }

    fn generic(
        provider: ProviderId,
        base_url: impl AsRef<str>,
        local: bool,
        mode: OpenAiCompatibleApiMode,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        let (protocol, api_mode) = mode_ids(mode)?;
        let support_scope = SupportScope::new(
            provider.clone(),
            PlatformId::new(if local { "local" } else { "custom-endpoint" })?,
            ModelFamily::Language,
            protocol,
            api_mode,
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
        match mode {
            OpenAiCompatibleApiMode::ChatCompletions => Self::from_parts(
                profile,
                endpoint,
                Some((support_scope, ChatCompletionsDialect::generic())),
                None,
            ),
            OpenAiCompatibleApiMode::Responses => {
                Self::from_parts(profile, endpoint, None, Some(support_scope))
            }
        }
    }

    fn from_parts(
        profile: ProviderProfile,
        endpoint: EndpointConfig,
        chat: Option<(SupportScope, ChatCompletionsDialect)>,
        responses: Option<SupportScope>,
    ) -> Result<Self, OpenAiCompatibleConfigError> {
        if chat.is_none() && responses.is_none() {
            return Err(OpenAiCompatibleConfigError::MissingLanguageModeClaim);
        }
        if let (Some((chat_scope, _)), Some(responses_scope)) = (&chat, &responses)
            && (chat_scope.provider() != responses_scope.provider()
                || chat_scope.platform() != responses_scope.platform())
        {
            return Err(OpenAiCompatibleConfigError::SharedEndpointScopeMismatch);
        }
        let chat = chat.map(|(support_scope, dialect)| ChatModeProfile {
            scope: provider_scope(&support_scope),
            support_scope,
            dialect,
            codec_policy: Arc::new(IdentityChatCodecPolicy),
        });
        let responses = responses.map(|support_scope| ResponsesModeProfile {
            scope: provider_scope(&support_scope),
            support_scope,
            codec_policy: Arc::new(IdentityResponsesCodecPolicy),
        });
        let (recommended_mode, recommended_scope) = match (&chat, &responses) {
            (_, Some(responses)) => (OpenAiCompatibleApiMode::Responses, responses.scope.clone()),
            (Some(chat), None) => (OpenAiCompatibleApiMode::ChatCompletions, chat.scope.clone()),
            (None, None) => return Err(OpenAiCompatibleConfigError::MissingLanguageModeClaim),
        };
        Ok(Self {
            profile: Arc::new(profile),
            endpoint,
            chat,
            responses,
            recommended_mode,
            recommended_scope,
        })
    }

    #[doc(hidden)]
    pub fn with_chat_codec_policy(mut self, codec_policy: Arc<dyn ChatCodecPolicy>) -> Self {
        if let Some(chat) = &mut self.chat {
            chat.codec_policy = codec_policy;
        }
        self
    }

    #[doc(hidden)]
    pub fn with_responses_codec_policy(
        mut self,
        codec_policy: Arc<dyn ResponsesCodecPolicy>,
    ) -> Self {
        if let Some(responses) = &mut self.responses {
            responses.codec_policy = codec_policy;
        }
        self
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub const fn recommended_mode(&self) -> OpenAiCompatibleApiMode {
        self.recommended_mode
    }

    pub(crate) fn recommended_scope(&self) -> &Arc<ProviderScope> {
        &self.recommended_scope
    }

    pub fn supports_mode(&self, mode: OpenAiCompatibleApiMode) -> bool {
        self.scope(mode).is_some()
    }

    pub fn scope(&self, mode: OpenAiCompatibleApiMode) -> Option<&ProviderScope> {
        match mode {
            OpenAiCompatibleApiMode::Responses => {
                self.responses.as_ref().map(|mode| mode.scope.as_ref())
            }
            OpenAiCompatibleApiMode::ChatCompletions => {
                self.chat.as_ref().map(|mode| mode.scope.as_ref())
            }
        }
    }

    pub(crate) fn scope_arc(&self, mode: OpenAiCompatibleApiMode) -> Option<Arc<ProviderScope>> {
        match mode {
            OpenAiCompatibleApiMode::Responses => {
                self.responses.as_ref().map(|mode| mode.scope.clone())
            }
            OpenAiCompatibleApiMode::ChatCompletions => {
                self.chat.as_ref().map(|mode| mode.scope.clone())
            }
        }
    }

    pub(crate) fn support_scope(&self, mode: OpenAiCompatibleApiMode) -> Option<&SupportScope> {
        match mode {
            OpenAiCompatibleApiMode::Responses => {
                self.responses.as_ref().map(|mode| &mode.support_scope)
            }
            OpenAiCompatibleApiMode::ChatCompletions => {
                self.chat.as_ref().map(|mode| &mode.support_scope)
            }
        }
    }

    pub(crate) fn profile_arc(&self) -> Arc<ProviderProfile> {
        self.profile.clone()
    }

    pub(crate) fn endpoint(&self) -> &EndpointConfig {
        &self.endpoint
    }

    #[cfg(test)]
    pub(crate) fn chat_mode(&self) -> Option<&ChatModeProfile> {
        self.chat.as_ref()
    }

    pub(crate) fn language_mode(
        &self,
        mode: OpenAiCompatibleApiMode,
    ) -> Option<LanguageModeProfile> {
        match mode {
            OpenAiCompatibleApiMode::ChatCompletions => {
                self.chat
                    .as_ref()
                    .map(|profile| LanguageModeProfile::ChatCompletions {
                        scope: profile.scope.clone(),
                        dialect: profile.dialect.clone(),
                        codec_policy: profile.codec_policy.clone(),
                    })
            }
            OpenAiCompatibleApiMode::Responses => {
                self.responses
                    .as_ref()
                    .map(|profile| LanguageModeProfile::Responses {
                        scope: profile.scope.clone(),
                        codec_policy: profile.codec_policy.clone(),
                    })
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn with_test_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = endpoint;
        self
    }
}

#[cfg(test)]
impl ChatModeProfile {
    pub(crate) fn dialect(&self) -> &ChatCompletionsDialect {
        &self.dialect
    }

    pub(crate) fn codec_policy(&self) -> &Arc<dyn ChatCodecPolicy> {
        &self.codec_policy
    }
}

impl fmt::Debug for OpenAiCompatibleProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleProfile")
            .field("profile_id", self.profile.id())
            .field("endpoint", &self.endpoint)
            .field(
                "chat",
                &self
                    .chat
                    .as_ref()
                    .map(|mode| (mode.scope.as_ref(), &mode.dialect, mode.codec_policy.name())),
            )
            .field(
                "responses",
                &self
                    .responses
                    .as_ref()
                    .map(|mode| (mode.scope.as_ref(), mode.codec_policy.name())),
            )
            .finish()
    }
}

fn mode_from_support_scope(
    scope: &SupportScope,
) -> Result<OpenAiCompatibleApiMode, OpenAiCompatibleConfigError> {
    if scope.family() != ModelFamily::Language {
        return Err(OpenAiCompatibleConfigError::IncompatibleSupportScope);
    }
    match (scope.protocol().as_str(), scope.api_mode().as_str()) {
        (CHAT_PROTOCOL_ID, CHAT_API_MODE_ID) => Ok(OpenAiCompatibleApiMode::ChatCompletions),
        (OPENAI_RESPONSES_PROTOCOL, RESPONSES_API_MODE_ID) => {
            Ok(OpenAiCompatibleApiMode::Responses)
        }
        _ => Err(OpenAiCompatibleConfigError::IncompatibleSupportScope),
    }
}

fn mode_ids(
    mode: OpenAiCompatibleApiMode,
) -> Result<(ProtocolId, ApiModeId), OpenAiCompatibleConfigError> {
    match mode {
        OpenAiCompatibleApiMode::Responses => Ok((
            ProtocolId::new(OPENAI_RESPONSES_PROTOCOL)?,
            ApiModeId::new(RESPONSES_API_MODE_ID)?,
        )),
        OpenAiCompatibleApiMode::ChatCompletions => Ok((
            ProtocolId::new(CHAT_PROTOCOL_ID)?,
            ApiModeId::new(CHAT_API_MODE_ID)?,
        )),
    }
}

fn provider_scope(scope: &SupportScope) -> Arc<ProviderScope> {
    Arc::new(
        ProviderScope::new(scope.provider().clone())
            .with_platform(scope.platform().clone())
            .with_protocol(scope.protocol().clone())
            .with_api_mode(scope.api_mode().clone()),
    )
}

impl From<EndpointError> for OpenAiCompatibleConfigError {
    fn from(source: EndpointError) -> Self {
        Self::Endpoint(source)
    }
}
