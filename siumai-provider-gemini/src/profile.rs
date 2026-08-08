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

use crate::models::current_image_models;

pub const PROVIDER_ID: &str = "google";
pub const PLATFORM_ID: &str = "gemini-api";
pub const PROTOCOL_ID: &str = "gemini-interactions";
pub const API_MODE_ID: &str = "interactions";
pub const OFFICIAL_SOURCE: &str = "https://ai.google.dev/gemini-api/docs/image-generation";
pub const VERIFIED_ON: &str = "2026-08-08";

/// Evidence-backed product profile for the configured Gemini API endpoint.
#[derive(Debug, Clone)]
pub struct GeminiProfile {
    profile: Arc<ProviderProfile>,
    scope: Arc<ProviderScope>,
}

impl GeminiProfile {
    pub(crate) fn current(replay_domain: ReplayDomain) -> Result<Self, GeminiProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new(PLATFORM_ID)?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let support_scope = SupportScope::new(
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
        let evidence = VerificationEvidence::new(
            OfficialSource::new(OFFICIAL_SOURCE)?,
            verified_at,
            ProtocolContractId::new("gemini-interactions-v1-image-2026-08")?,
        );
        let catalog = ModelCatalog::new(current_image_models().into_iter().map(|model| {
            ModelProfile::new(
                ModelId::new(model).expect("Google model IDs are static"),
                support_scope.clone(),
                [ModelOperation::GenerateImage],
                ModelLifecycle::Active,
                evidence.clone(),
            )
            .expect("Google image model profiles declare one operation")
        }))?;
        let profile = ProviderProfile::verified(
            ProfileId::new(PROVIDER_ID)?,
            vec![VerifiedSupportClaim::new(
                support_scope,
                VerifiedFidelity::Native,
                ApiStability::Stable,
                evidence,
            )],
            catalog,
        )?;
        let scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(protocol)
                .with_api_mode(api_mode)
                .with_replay_domain(replay_domain),
        );
        Ok(Self {
            profile: Arc::new(profile),
            scope,
        })
    }

    pub(crate) fn custom(replay_domain: ReplayDomain) -> Result<Self, GeminiProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new("custom-gemini-api")?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let support_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Image,
            protocol.clone(),
            api_mode.clone(),
        );
        let profile = ProviderProfile::generic(
            ProfileId::new("google-custom-gemini")?,
            GenericSupportClaim::new(support_scope, ApiStability::Stable),
        );
        let scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(protocol)
                .with_api_mode(api_mode)
                .with_replay_domain(replay_domain),
        );
        Ok(Self {
            profile: Arc::new(profile),
            scope,
        })
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub(crate) fn image_scope(&self) -> Arc<ProviderScope> {
        self.scope.clone()
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GeminiProfileError {
    #[error("invalid Google image profile identifier: {0}")]
    Identifier(#[from] siumai_core::InvalidId),
    #[error("invalid Google image support profile: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid Google image model catalog: {0}")]
    Catalog(#[from] siumai_core::CatalogError),
    #[error("Google image verification date is invalid")]
    InvalidVerificationDate,
}
