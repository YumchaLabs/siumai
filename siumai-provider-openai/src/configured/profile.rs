use std::sync::Arc;

use chrono::NaiveDate;
use siumai_core::{
    ApiModeId, ApiStability, GenericSupportClaim, ModelCatalog, ModelFamily, ModelId,
    ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileId,
    ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ProviderScope, SupportScope,
    VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use siumai_protocol_openai::chat_completions::PROTOCOL_ID as CHAT_COMPLETIONS_PROTOCOL;
use siumai_protocol_openai::responses_next::OPENAI_RESPONSES_PROTOCOL;

use super::catalog::{GPT_5_6, GPT_5_6_LUNA, GPT_5_6_SOL, GPT_5_6_TERRA};
use super::mode::OpenAiApiMode;
use super::provider::OpenAiConfigError;

pub(crate) const PROVIDER_ID: &str = "openai";
pub(crate) const PLATFORM_ID: &str = "openai-api";
const CUSTOM_PLATFORM_ID: &str = "custom-openai-api";

const MODEL_GUIDANCE_SOURCE: &str = "https://developers.openai.com/api/docs/guides/latest-model";
const RESPONSES_CONTRACT: &str = "openai-responses-2026-08-04";
const CHAT_COMPLETIONS_CONTRACT: &str = "openai-chat-completions-2026-08-04";

/// Evidence-backed OpenAI language profile for both explicit API modes.
#[derive(Debug, Clone)]
pub struct OpenAiProfile {
    profile: Arc<ProviderProfile>,
    responses_scope: SupportScope,
    chat_completions_scope: SupportScope,
    responses_provider_scope: Arc<ProviderScope>,
    chat_completions_provider_scope: Arc<ProviderScope>,
}

impl OpenAiProfile {
    /// Build the provider-owned profile verified on August 4, 2026.
    pub fn current() -> Result<Self, OpenAiConfigError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new(PLATFORM_ID)?;
        let responses_protocol = ProtocolId::new(OPENAI_RESPONSES_PROTOCOL)?;
        let chat_completions_protocol = ProtocolId::new(CHAT_COMPLETIONS_PROTOCOL)?;
        let responses_mode = OpenAiApiMode::Responses.id()?;
        let chat_mode = OpenAiApiMode::ChatCompletions.id()?;
        let responses_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Language,
            responses_protocol.clone(),
            responses_mode.clone(),
        );
        let chat_completions_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Language,
            chat_completions_protocol.clone(),
            chat_mode.clone(),
        );
        let verified_at = NaiveDate::from_ymd_opt(2026, 8, 4)
            .map(VerificationDate::new)
            .ok_or(OpenAiConfigError::InvalidVerificationDate)?;
        let source = OfficialSource::new(MODEL_GUIDANCE_SOURCE)?;
        let responses_evidence = VerificationEvidence::new(
            source.clone(),
            verified_at,
            ProtocolContractId::new(RESPONSES_CONTRACT)?,
        );
        let chat_evidence = VerificationEvidence::new(
            source,
            verified_at,
            ProtocolContractId::new(CHAT_COMPLETIONS_CONTRACT)?,
        );
        let claims = vec![
            VerifiedSupportClaim::new(
                responses_scope.clone(),
                VerifiedFidelity::Native,
                ApiStability::Stable,
                responses_evidence.clone(),
            ),
            VerifiedSupportClaim::new(
                chat_completions_scope.clone(),
                VerifiedFidelity::Native,
                ApiStability::Stable,
                chat_evidence.clone(),
            ),
        ];
        let mut models = Vec::with_capacity(8);
        extend_current_models(&mut models, &responses_scope, &responses_evidence)?;
        extend_current_models(&mut models, &chat_completions_scope, &chat_evidence)?;
        let profile = ProviderProfile::verified(
            ProfileId::new(PROVIDER_ID)?,
            claims,
            ModelCatalog::new(models)?,
        )?;
        let responses_provider_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(responses_protocol)
                .with_api_mode(responses_mode),
        );
        let chat_completions_provider_scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(chat_completions_protocol)
                .with_api_mode(chat_mode),
        );
        Ok(Self {
            profile: Arc::new(profile),
            responses_scope,
            chat_completions_scope,
            responses_provider_scope,
            chat_completions_provider_scope,
        })
    }

    /// Build an unverified profile for a caller-supplied OpenAI-compatible endpoint.
    pub fn custom() -> Result<Self, OpenAiConfigError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new(CUSTOM_PLATFORM_ID)?;
        let responses_protocol = ProtocolId::new(OPENAI_RESPONSES_PROTOCOL)?;
        let chat_completions_protocol = ProtocolId::new(CHAT_COMPLETIONS_PROTOCOL)?;
        let responses_mode = OpenAiApiMode::Responses.id()?;
        let chat_mode = OpenAiApiMode::ChatCompletions.id()?;
        let responses_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Language,
            responses_protocol.clone(),
            responses_mode.clone(),
        );
        let chat_completions_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Language,
            chat_completions_protocol.clone(),
            chat_mode.clone(),
        );
        let profile = ProviderProfile::generic_many(
            ProfileId::new("openai-custom")?,
            vec![
                GenericSupportClaim::new(responses_scope.clone(), ApiStability::Experimental),
                GenericSupportClaim::new(
                    chat_completions_scope.clone(),
                    ApiStability::Experimental,
                ),
            ],
        )?;
        let responses_provider_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(responses_protocol)
                .with_api_mode(responses_mode),
        );
        let chat_completions_provider_scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(chat_completions_protocol)
                .with_api_mode(chat_mode),
        );
        Ok(Self {
            profile: Arc::new(profile),
            responses_scope,
            chat_completions_scope,
            responses_provider_scope,
            chat_completions_provider_scope,
        })
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub(crate) fn profile_arc(&self) -> Arc<ProviderProfile> {
        self.profile.clone()
    }

    pub(crate) fn support_scope(&self, mode: OpenAiApiMode) -> &SupportScope {
        match mode {
            OpenAiApiMode::Responses => &self.responses_scope,
            OpenAiApiMode::ChatCompletions => &self.chat_completions_scope,
        }
    }

    pub(crate) fn provider_scope(&self, mode: OpenAiApiMode) -> &Arc<ProviderScope> {
        match mode {
            OpenAiApiMode::Responses => &self.responses_provider_scope,
            OpenAiApiMode::ChatCompletions => &self.chat_completions_provider_scope,
        }
    }
}

fn extend_current_models(
    models: &mut Vec<ModelProfile>,
    scope: &SupportScope,
    evidence: &VerificationEvidence,
) -> Result<(), OpenAiConfigError> {
    for (model, lifecycle) in [
        (GPT_5_6_SOL, ModelLifecycle::Active),
        (GPT_5_6_TERRA, ModelLifecycle::Active),
        (GPT_5_6_LUNA, ModelLifecycle::Active),
        (GPT_5_6, ModelLifecycle::RollingAlias),
    ] {
        models.push(ModelProfile::new(
            ModelId::new(model)?,
            scope.clone(),
            [ModelOperation::Generate, ModelOperation::Stream],
            lifecycle,
            evidence.clone(),
        )?);
    }
    Ok(())
}

pub(crate) fn mode_from_scope(scope: &ProviderScope) -> Option<OpenAiApiMode> {
    match scope.api_mode().map(ApiModeId::as_str) {
        Some("responses") => Some(OpenAiApiMode::Responses),
        Some("chat-completions") => Some(OpenAiApiMode::ChatCompletions),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_has_two_native_mode_claims_and_open_catalog_rows() {
        let profile = OpenAiProfile::current().unwrap();
        let claims = profile.provider_profile().verified_claims().unwrap();

        assert_eq!(claims.len(), 2);
        assert!(claims.iter().all(|claim| {
            claim.fidelity() == VerifiedFidelity::Native
                && claim.stability() == ApiStability::Stable
        }));
        assert_eq!(
            profile.provider_profile().catalog().unwrap().iter().count(),
            8
        );
    }

    #[test]
    fn custom_profile_has_generic_mode_claims_without_a_catalog() {
        let profile = OpenAiProfile::custom().unwrap();

        assert!(profile.provider_profile().verified_claims().is_none());
        assert_eq!(
            profile.provider_profile().generic_claims().unwrap().len(),
            2
        );
        assert!(profile.provider_profile().catalog().is_none());
    }
}
