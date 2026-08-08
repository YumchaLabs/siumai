use std::sync::Arc;

use chrono::NaiveDate;
use siumai_core::{
    ApiModeId, ApiStability, GenericSupportClaim, ModelCatalog, ModelFamily, ModelId,
    ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileError,
    ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ProviderScope,
    ReplayDomain, SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity,
    VerifiedSupportClaim,
};
use thiserror::Error;

use crate::models::{current_image_models, current_interactions_models};

pub const PROVIDER_ID: &str = "google";
pub const PLATFORM_ID: &str = "gemini-api";
pub const PROTOCOL_ID: &str = "gemini-interactions";
pub const API_MODE_ID: &str = "interactions";
pub const INTERACTIONS_SOURCE: &str = "https://ai.google.dev/api/interactions-api";
pub const IMAGE_SOURCE: &str = "https://ai.google.dev/gemini-api/docs/image-generation";
pub const VERIFIED_ON: &str = "2026-08-08";

/// Evidence-backed product profile for the configured Gemini API endpoint.
#[derive(Debug, Clone)]
pub struct GeminiProfile {
    profile: Arc<ProviderProfile>,
    interactions_scope: Arc<ProviderScope>,
    image_scope: Arc<ProviderScope>,
}

impl GeminiProfile {
    pub(crate) fn current(replay_domain: ReplayDomain) -> Result<Self, GeminiProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new(PLATFORM_ID)?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let language_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Language,
            protocol.clone(),
            api_mode.clone(),
        );
        let image_support_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Image,
            protocol.clone(),
            api_mode.clone(),
        );
        let verified_at = VerificationDate::new(
            NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
                .map_err(|_| GeminiProfileError::InvalidVerificationDate)?,
        );
        let language_evidence = VerificationEvidence::new(
            OfficialSource::new(INTERACTIONS_SOURCE)?,
            verified_at,
            ProtocolContractId::new("gemini-interactions-v1-language-2026-08")?,
        );
        let image_evidence = VerificationEvidence::new(
            OfficialSource::new(IMAGE_SOURCE)?,
            verified_at,
            ProtocolContractId::new("gemini-interactions-v1-image-2026-08")?,
        );
        let claims = vec![
            VerifiedSupportClaim::new(
                language_scope.clone(),
                VerifiedFidelity::Native,
                ApiStability::Stable,
                language_evidence.clone(),
            ),
            VerifiedSupportClaim::new(
                image_support_scope.clone(),
                VerifiedFidelity::Native,
                ApiStability::Stable,
                image_evidence.clone(),
            ),
        ];
        let mut models = Vec::new();
        models.extend(current_interactions_models().into_iter().map(|model| {
            ModelProfile::new(
                ModelId::new(model).expect("Google model IDs are static"),
                language_scope.clone(),
                [ModelOperation::Generate, ModelOperation::Stream],
                ModelLifecycle::Active,
                language_evidence.clone(),
            )
            .expect("Gemini language model profiles declare two operations")
        }));
        models.extend(current_image_models().into_iter().map(|model| {
            ModelProfile::new(
                ModelId::new(model).expect("Google model IDs are static"),
                image_support_scope.clone(),
                [ModelOperation::GenerateImage],
                ModelLifecycle::Active,
                image_evidence.clone(),
            )
            .expect("Gemini image model profiles declare one operation")
        }));
        let profile = ProviderProfile::verified(
            ProfileId::new(PROVIDER_ID)?,
            claims,
            ModelCatalog::new(models)?,
        )?;
        let execution_scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(protocol)
                .with_api_mode(api_mode)
                .with_replay_domain(replay_domain),
        );
        Ok(Self {
            profile: Arc::new(profile),
            interactions_scope: execution_scope.clone(),
            image_scope: execution_scope,
        })
    }

    pub(crate) fn custom(replay_domain: ReplayDomain) -> Result<Self, GeminiProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new("custom-gemini-api")?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let claims = [ModelFamily::Language, ModelFamily::Image]
            .into_iter()
            .map(|family| {
                GenericSupportClaim::new(
                    SupportScope::new(
                        provider.clone(),
                        platform.clone(),
                        family,
                        protocol.clone(),
                        api_mode.clone(),
                    ),
                    ApiStability::Stable,
                )
            })
            .collect();
        let profile =
            ProviderProfile::generic_many(ProfileId::new("google-custom-gemini")?, claims)?;
        let execution_scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(protocol)
                .with_api_mode(api_mode)
                .with_replay_domain(replay_domain),
        );
        Ok(Self {
            profile: Arc::new(profile),
            interactions_scope: execution_scope.clone(),
            image_scope: execution_scope,
        })
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub(crate) fn interactions_scope(&self) -> Arc<ProviderScope> {
        self.interactions_scope.clone()
    }

    pub(crate) fn image_scope(&self) -> Arc<ProviderScope> {
        self.image_scope.clone()
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GeminiProfileError {
    #[error("invalid Gemini profile identifier: {0}")]
    Identifier(#[from] siumai_core::InvalidId),
    #[error("invalid Gemini support profile: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid Gemini model catalog: {0}")]
    Catalog(#[from] siumai_core::CatalogError),
    #[error("Gemini verification date is invalid")]
    InvalidVerificationDate,
}
