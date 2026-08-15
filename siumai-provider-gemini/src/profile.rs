use std::sync::Arc;

use chrono::NaiveDate;
use siumai_core::{
    ApiModeId, ApiStability, GenericSupportClaim, ModelCatalog, ModelFamily, ModelId,
    ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileError,
    ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ProviderScope,
    ReplayDomain, SupportScope, UpstreamLifecycle, UpstreamMaturity, UpstreamSupportStatus,
    VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use thiserror::Error;

use crate::embedding::{GEMINI_EMBEDDING_001, GEMINI_EMBEDDING_2, GEMINI_EMBEDDING_API_MODE_ID};
use crate::models::{current_image_models, current_interactions_models};
use crate::multimodal_embedding::GEMINI_MULTIMODAL_EMBEDDING_API_MODE_ID;
use crate::speech::{
    GEMINI_2_5_FLASH_PREVIEW_TTS, GEMINI_2_5_PRO_PREVIEW_TTS, GEMINI_3_1_FLASH_TTS_PREVIEW,
    GEMINI_SPEECH_API_MODE_ID,
};
use crate::veo::{VEO_API_MODE_ID, VEO_PROTOCOL_ID};

pub const PROVIDER_ID: &str = "google";
pub const PLATFORM_ID: &str = "gemini-api";
pub const PROTOCOL_ID: &str = "gemini-interactions";
pub const API_MODE_ID: &str = "interactions";
pub const EMBEDDING_PROTOCOL_ID: &str = "gemini-embed-content";
pub const GENERATE_CONTENT_PROTOCOL_ID: &str = "gemini-generate-content";
pub const GENERATE_CONTENT_API_MODE_ID: &str = "generate-content";
pub const INTERACTIONS_SOURCE: &str = "https://ai.google.dev/api/interactions-api";
pub const IMAGE_SOURCE: &str = "https://ai.google.dev/gemini-api/docs/image-generation";
pub const EMBEDDING_SOURCE: &str = "https://ai.google.dev/gemini-api/docs/embeddings";
pub const SPEECH_SOURCE: &str = "https://ai.google.dev/gemini-api/docs/speech-generation";
pub const GENERATE_CONTENT_SOURCE: &str =
    "https://ai.google.dev/gemini-api/docs/generate-content/text-generation";
pub const FILES_SOURCE: &str = "https://ai.google.dev/gemini-api/docs/files";
pub const VEO_SOURCE: &str = "https://ai.google.dev/gemini-api/docs/veo";
pub const VERIFIED_ON: &str = "2026-08-08";

/// Evidence-backed product profile for the configured Gemini API endpoint.
#[derive(Debug, Clone)]
pub struct GeminiProfile {
    profile: Arc<ProviderProfile>,
    interactions_scope: Arc<ProviderScope>,
    image_scope: Arc<ProviderScope>,
    embedding_scope: Arc<ProviderScope>,
    multimodal_embedding_scope: Arc<ProviderScope>,
    speech_scope: Arc<ProviderScope>,
    veo_scope: Arc<ProviderScope>,
    generate_content_scope: Arc<ProviderScope>,
}

impl GeminiProfile {
    pub(crate) fn current(replay_domain: ReplayDomain) -> Result<Self, GeminiProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new(PLATFORM_ID)?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let embedding_protocol = ProtocolId::new(EMBEDDING_PROTOCOL_ID)?;
        let embedding_api_mode = ApiModeId::new(GEMINI_EMBEDDING_API_MODE_ID)?;
        let multimodal_embedding_api_mode =
            ApiModeId::new(GEMINI_MULTIMODAL_EMBEDDING_API_MODE_ID)?;
        let speech_api_mode = ApiModeId::new(GEMINI_SPEECH_API_MODE_ID)?;
        let generate_content_protocol = ProtocolId::new(GENERATE_CONTENT_PROTOCOL_ID)?;
        let generate_content_api_mode = ApiModeId::new(GENERATE_CONTENT_API_MODE_ID)?;
        let veo_protocol = ProtocolId::new(VEO_PROTOCOL_ID)?;
        let veo_api_mode = ApiModeId::new(VEO_API_MODE_ID)?;
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
        let embedding_support_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Embedding,
            embedding_protocol.clone(),
            embedding_api_mode.clone(),
        );
        let speech_support_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Speech,
            protocol.clone(),
            speech_api_mode.clone(),
        );
        let generate_content_support_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Language,
            generate_content_protocol.clone(),
            generate_content_api_mode.clone(),
        );
        let verified_at = VerificationDate::new(
            NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
                .map_err(|_| GeminiProfileError::InvalidVerificationDate)?,
        );
        let language_evidence = VerificationEvidence::new(
            OfficialSource::new(INTERACTIONS_SOURCE)?,
            verified_at,
            ProtocolContractId::new("gemini-interactions-v1-language-2026-08")?,
        )
        .with_upstream(UpstreamLifecycle::new(
            Some(UpstreamMaturity::Stable),
            Some(UpstreamSupportStatus::Active),
            Some("v1".to_string()),
        ));
        let image_evidence = VerificationEvidence::new(
            OfficialSource::new(IMAGE_SOURCE)?,
            verified_at,
            ProtocolContractId::new("gemini-interactions-v1-image-2026-08")?,
        )
        .with_upstream(UpstreamLifecycle::new(
            Some(UpstreamMaturity::Stable),
            Some(UpstreamSupportStatus::Active),
            Some("v1".to_string()),
        ));
        let embedding_evidence = VerificationEvidence::new(
            OfficialSource::new(EMBEDDING_SOURCE)?,
            verified_at,
            ProtocolContractId::new("gemini-embed-content-v1-text-2026-08")?,
        )
        .with_upstream(UpstreamLifecycle::new(
            Some(UpstreamMaturity::Stable),
            Some(UpstreamSupportStatus::Active),
            Some("v1".to_string()),
        ));
        let speech_evidence = VerificationEvidence::new(
            OfficialSource::new(SPEECH_SOURCE)?,
            verified_at,
            ProtocolContractId::new("gemini-interactions-v1beta-speech-2026-08")?,
        )
        .with_upstream(UpstreamLifecycle::new(
            Some(UpstreamMaturity::Preview),
            Some(UpstreamSupportStatus::Active),
            Some("preview".to_string()),
        ));
        let generate_content_evidence = VerificationEvidence::new(
            OfficialSource::new(GENERATE_CONTENT_SOURCE)?,
            verified_at,
            ProtocolContractId::new("gemini-generate-content-v1-legacy-2026-08")?,
        )
        .with_upstream(UpstreamLifecycle::new(
            Some(UpstreamMaturity::Stable),
            Some(UpstreamSupportStatus::Legacy),
            Some("Generate Content API (Legacy)".to_string()),
        ));
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
            VerifiedSupportClaim::new(
                embedding_support_scope.clone(),
                VerifiedFidelity::Native,
                ApiStability::Stable,
                embedding_evidence.clone(),
            ),
            VerifiedSupportClaim::new(
                speech_support_scope.clone(),
                VerifiedFidelity::Native,
                ApiStability::Experimental,
                speech_evidence.clone(),
            ),
            VerifiedSupportClaim::new(
                generate_content_support_scope.clone(),
                VerifiedFidelity::Native,
                ApiStability::Stable,
                generate_content_evidence.clone(),
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
        models.extend([GEMINI_EMBEDDING_2, GEMINI_EMBEDDING_001].map(|model| {
            ModelProfile::new(
                ModelId::new(model).expect("Google model IDs are static"),
                embedding_support_scope.clone(),
                [ModelOperation::Embed],
                ModelLifecycle::Active,
                embedding_evidence.clone(),
            )
            .expect("Gemini embedding model profiles declare one operation")
        }));
        models.extend(
            [
                GEMINI_3_1_FLASH_TTS_PREVIEW,
                GEMINI_2_5_FLASH_PREVIEW_TTS,
                GEMINI_2_5_PRO_PREVIEW_TTS,
            ]
            .map(|model| {
                ModelProfile::new(
                    ModelId::new(model).expect("Google model IDs are static"),
                    speech_support_scope.clone(),
                    [ModelOperation::SynthesizeSpeech],
                    ModelLifecycle::Active,
                    speech_evidence.clone(),
                )
                .expect("Gemini speech model profiles declare one operation")
            }),
        );
        models.extend(current_interactions_models().into_iter().map(|model| {
            ModelProfile::new(
                ModelId::new(model).expect("Google model IDs are static"),
                generate_content_support_scope.clone(),
                [ModelOperation::Generate, ModelOperation::Stream],
                ModelLifecycle::Active,
                generate_content_evidence.clone(),
            )
            .expect("Gemini Generate Content model profiles declare two operations")
        }));
        let profile = ProviderProfile::verified(
            ProfileId::new(PROVIDER_ID)?,
            claims,
            ModelCatalog::new(models)?,
        )?;
        let execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(protocol.clone())
                .with_api_mode(api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let embedding_execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(embedding_protocol.clone())
                .with_api_mode(embedding_api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let multimodal_embedding_execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(embedding_protocol)
                .with_api_mode(multimodal_embedding_api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let veo_execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(veo_protocol)
                .with_api_mode(veo_api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let speech_execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(protocol)
                .with_api_mode(speech_api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let generate_content_execution_scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(generate_content_protocol)
                .with_api_mode(generate_content_api_mode)
                .with_replay_domain(replay_domain),
        );
        Ok(Self {
            profile: Arc::new(profile),
            interactions_scope: execution_scope.clone(),
            image_scope: execution_scope,
            embedding_scope: embedding_execution_scope,
            multimodal_embedding_scope: multimodal_embedding_execution_scope,
            speech_scope: speech_execution_scope,
            veo_scope: veo_execution_scope,
            generate_content_scope: generate_content_execution_scope,
        })
    }

    pub(crate) fn custom(replay_domain: ReplayDomain) -> Result<Self, GeminiProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new("custom-gemini-api")?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let embedding_protocol = ProtocolId::new(EMBEDDING_PROTOCOL_ID)?;
        let embedding_api_mode = ApiModeId::new(GEMINI_EMBEDDING_API_MODE_ID)?;
        let multimodal_embedding_api_mode =
            ApiModeId::new(GEMINI_MULTIMODAL_EMBEDDING_API_MODE_ID)?;
        let speech_api_mode = ApiModeId::new(GEMINI_SPEECH_API_MODE_ID)?;
        let generate_content_protocol = ProtocolId::new(GENERATE_CONTENT_PROTOCOL_ID)?;
        let generate_content_api_mode = ApiModeId::new(GENERATE_CONTENT_API_MODE_ID)?;
        let veo_protocol = ProtocolId::new(VEO_PROTOCOL_ID)?;
        let veo_api_mode = ApiModeId::new(VEO_API_MODE_ID)?;
        let claims = vec![
            GenericSupportClaim::new(
                SupportScope::new(
                    provider.clone(),
                    platform.clone(),
                    ModelFamily::Language,
                    protocol.clone(),
                    api_mode.clone(),
                ),
                ApiStability::Stable,
            ),
            GenericSupportClaim::new(
                SupportScope::new(
                    provider.clone(),
                    platform.clone(),
                    ModelFamily::Image,
                    protocol.clone(),
                    api_mode.clone(),
                ),
                ApiStability::Stable,
            ),
            GenericSupportClaim::new(
                SupportScope::new(
                    provider.clone(),
                    platform.clone(),
                    ModelFamily::Embedding,
                    embedding_protocol.clone(),
                    embedding_api_mode.clone(),
                ),
                ApiStability::Stable,
            ),
            GenericSupportClaim::new(
                SupportScope::new(
                    provider.clone(),
                    platform.clone(),
                    ModelFamily::Speech,
                    protocol.clone(),
                    speech_api_mode.clone(),
                ),
                ApiStability::Experimental,
            ),
            GenericSupportClaim::new(
                SupportScope::new(
                    provider.clone(),
                    platform.clone(),
                    ModelFamily::Language,
                    generate_content_protocol.clone(),
                    generate_content_api_mode.clone(),
                ),
                ApiStability::Stable,
            ),
        ];
        let profile =
            ProviderProfile::generic_many(ProfileId::new("google-custom-gemini")?, claims)?;
        let execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(protocol.clone())
                .with_api_mode(api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let embedding_execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(embedding_protocol.clone())
                .with_api_mode(embedding_api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let multimodal_embedding_execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(embedding_protocol)
                .with_api_mode(multimodal_embedding_api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let veo_execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(veo_protocol)
                .with_api_mode(veo_api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let speech_execution_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(protocol)
                .with_api_mode(speech_api_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let generate_content_execution_scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(generate_content_protocol)
                .with_api_mode(generate_content_api_mode)
                .with_replay_domain(replay_domain),
        );
        Ok(Self {
            profile: Arc::new(profile),
            interactions_scope: execution_scope.clone(),
            image_scope: execution_scope,
            embedding_scope: embedding_execution_scope,
            multimodal_embedding_scope: multimodal_embedding_execution_scope,
            speech_scope: speech_execution_scope,
            veo_scope: veo_execution_scope,
            generate_content_scope: generate_content_execution_scope,
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

    pub(crate) fn embedding_scope(&self) -> Arc<ProviderScope> {
        self.embedding_scope.clone()
    }

    pub(crate) fn multimodal_embedding_scope(&self) -> Arc<ProviderScope> {
        self.multimodal_embedding_scope.clone()
    }

    pub(crate) fn speech_scope(&self) -> Arc<ProviderScope> {
        self.speech_scope.clone()
    }

    pub(crate) fn veo_scope(&self) -> Arc<ProviderScope> {
        self.veo_scope.clone()
    }

    pub(crate) fn generate_content_scope(&self) -> Arc<ProviderScope> {
        self.generate_content_scope.clone()
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
