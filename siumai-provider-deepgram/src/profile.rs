use std::sync::Arc;

use chrono::NaiveDate;
use siumai_core::{
    ApiModeId, ApiStability, GenericSupportClaim, ModelCatalog, ModelFamily, ModelId,
    ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileError,
    ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ProviderScope,
    SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use thiserror::Error;

pub const PROVIDER_ID: &str = "deepgram";
pub const PLATFORM_ID: &str = "public-api";
pub const PROTOCOL_ID: &str = "deepgram-prerecorded";
pub const API_MODE_ID: &str = "prerecorded";
pub const SPEECH_PROTOCOL_ID: &str = "deepgram-aura-tts";
pub const SPEECH_API_MODE_ID: &str = "tts";
pub const VERIFIED_ON: &str = "2026-08-09";
pub const API_SOURCE: &str =
    "https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded";
pub const MODEL_SOURCE: &str = "https://developers.deepgram.com/docs/models-languages-overview";
pub const SPEECH_SOURCE: &str = "https://developers.deepgram.com/reference/text-to-speech/speak";
pub const SPEECH_MODEL_SOURCE: &str = "https://developers.deepgram.com/docs/tts-models";

/// Evidence-backed Deepgram prerecorded transcription profile.
#[derive(Debug, Clone)]
pub struct DeepgramProfile {
    profile: Arc<ProviderProfile>,
    scope: Arc<ProviderScope>,
    speech_scope: Arc<ProviderScope>,
}

impl DeepgramProfile {
    pub fn current() -> Result<Self, DeepgramProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new(PLATFORM_ID)?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let support_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Transcription,
            protocol.clone(),
            api_mode.clone(),
        );
        let api_evidence = evidence(API_SOURCE, "deepgram-prerecorded-2026-08")?;
        let model_evidence = evidence(MODEL_SOURCE, "deepgram-model-catalog-2026-08")?;
        let speech_protocol = ProtocolId::new(SPEECH_PROTOCOL_ID)?;
        let speech_api_mode = ApiModeId::new(SPEECH_API_MODE_ID)?;
        let speech_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Speech,
            speech_protocol.clone(),
            speech_api_mode.clone(),
        );
        let speech_evidence = evidence(SPEECH_SOURCE, "deepgram-aura-tts-2026-08")?;
        let speech_model_evidence =
            evidence(SPEECH_MODEL_SOURCE, "deepgram-aura-model-catalog-2026-08")?;
        let catalog = ModelCatalog::new(
            crate::models::CURRENT_TRANSCRIPTION_MODELS
                .iter()
                .map(|model| {
                    ModelProfile::new(
                        ModelId::new(*model).expect("Deepgram model IDs are static"),
                        support_scope.clone(),
                        [ModelOperation::Transcribe],
                        ModelLifecycle::Active,
                        model_evidence.clone(),
                    )
                    .expect("Deepgram model profiles declare one operation")
                })
                .chain(crate::models::CURRENT_SPEECH_MODELS.iter().map(|model| {
                    ModelProfile::new(
                        ModelId::new(*model).expect("Deepgram speech model IDs are static"),
                        speech_scope.clone(),
                        [ModelOperation::SynthesizeSpeech],
                        ModelLifecycle::Active,
                        speech_model_evidence.clone(),
                    )
                    .expect("Deepgram speech model profiles declare one operation")
                })),
        )?;
        let profile = ProviderProfile::verified(
            ProfileId::new(PROVIDER_ID)?,
            vec![
                VerifiedSupportClaim::new(
                    support_scope,
                    VerifiedFidelity::Native,
                    ApiStability::Stable,
                    api_evidence,
                ),
                VerifiedSupportClaim::new(
                    speech_scope,
                    VerifiedFidelity::Native,
                    ApiStability::Stable,
                    speech_evidence,
                ),
            ],
            catalog,
        )?;
        let scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(protocol)
                .with_api_mode(api_mode),
        );
        let speech_scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(speech_protocol)
                .with_api_mode(speech_api_mode),
        );
        Ok(Self {
            profile: Arc::new(profile),
            scope,
            speech_scope,
        })
    }

    pub fn custom() -> Result<Self, DeepgramProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new("custom-deepgram")?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let support_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Transcription,
            protocol.clone(),
            api_mode.clone(),
        );
        let speech_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Speech,
            ProtocolId::new(SPEECH_PROTOCOL_ID)?,
            ApiModeId::new(SPEECH_API_MODE_ID)?,
        );
        let profile = ProviderProfile::generic_many(
            ProfileId::new("deepgram-custom")?,
            vec![
                GenericSupportClaim::new(support_scope, ApiStability::Experimental),
                GenericSupportClaim::new(speech_scope, ApiStability::Experimental),
            ],
        )?;
        let scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(protocol)
                .with_api_mode(api_mode),
        );
        let speech_scope = Arc::new(
            ProviderScope::new(scope.provider_id().clone())
                .with_platform(scope.platform().cloned().expect("custom platform"))
                .with_protocol(ProtocolId::new(SPEECH_PROTOCOL_ID)?)
                .with_api_mode(ApiModeId::new(SPEECH_API_MODE_ID)?),
        );
        Ok(Self {
            profile: Arc::new(profile),
            scope,
            speech_scope,
        })
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub(crate) fn scope(&self) -> Arc<ProviderScope> {
        self.scope.clone()
    }

    pub(crate) fn speech_scope(&self) -> Arc<ProviderScope> {
        self.speech_scope.clone()
    }
}

fn evidence(source: &str, contract: &str) -> Result<VerificationEvidence, DeepgramProfileError> {
    let verified_at = VerificationDate::new(
        NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
            .map_err(|_| DeepgramProfileError::InvalidVerificationDate)?,
    );
    Ok(VerificationEvidence::new(
        OfficialSource::new(source)?,
        verified_at,
        ProtocolContractId::new(contract)?,
    ))
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DeepgramProfileError {
    #[error("invalid Deepgram profile identifier: {0}")]
    Identifier(#[from] siumai_core::InvalidId),
    #[error("invalid Deepgram support profile: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid Deepgram model catalog: {0}")]
    Catalog(#[from] siumai_core::CatalogError),
    #[error("Deepgram verification date is invalid")]
    InvalidVerificationDate,
}
