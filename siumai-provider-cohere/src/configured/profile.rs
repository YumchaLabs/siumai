use std::sync::Arc;

use chrono::NaiveDate;
use siumai_core::{
    ApiModeId, ApiStability, GenericSupportClaim, ModelCatalog, ModelFamily, ModelId,
    ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileError,
    ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ProviderScope,
    SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use thiserror::Error;

pub const PROVIDER_ID: &str = "cohere";
pub const PLATFORM_ID: &str = "public-api";
pub const PROTOCOL_ID: &str = "cohere-native";
pub const API_MODE_ID: &str = "v2";
pub const VERIFIED_ON: &str = "2026-08-06";
pub const EMBED_SOURCE: &str = "https://docs.cohere.com/v2/reference/embed";
pub const RERANK_SOURCE: &str = "https://docs.cohere.com/v2/reference/rerank";
pub const MODEL_SOURCE: &str = "https://docs.cohere.com/docs/models";

/// Evidence-backed Cohere v2 embedding and rerank profile.
#[derive(Debug, Clone)]
pub struct CohereProfile {
    profile: Arc<ProviderProfile>,
    scope: Arc<ProviderScope>,
}

impl CohereProfile {
    pub fn current() -> Result<Self, CohereProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new(PLATFORM_ID)?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let embedding_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Embedding,
            protocol.clone(),
            api_mode.clone(),
        );
        let rerank_scope = SupportScope::new(
            provider.clone(),
            platform.clone(),
            ModelFamily::Rerank,
            protocol.clone(),
            api_mode.clone(),
        );
        let embedding_evidence = evidence(EMBED_SOURCE, "cohere-embed-v2-2026-08")?;
        let rerank_evidence = evidence(RERANK_SOURCE, "cohere-rerank-v2-2026-08")?;
        let model_evidence = evidence(MODEL_SOURCE, "cohere-model-catalog-2026-08")?;
        let catalog = ModelCatalog::new(
            crate::models::CURRENT_EMBEDDING_MODELS
                .iter()
                .map(|model| {
                    model_profile(
                        model,
                        embedding_scope.clone(),
                        ModelOperation::Embed,
                        model_evidence.clone(),
                    )
                })
                .chain(crate::models::CURRENT_RERANK_MODELS.iter().map(|model| {
                    model_profile(
                        model,
                        rerank_scope.clone(),
                        ModelOperation::Rerank,
                        model_evidence.clone(),
                    )
                })),
        )?;
        let profile = ProviderProfile::verified(
            ProfileId::new(PROVIDER_ID)?,
            vec![
                VerifiedSupportClaim::new(
                    embedding_scope,
                    VerifiedFidelity::Native,
                    ApiStability::Stable,
                    embedding_evidence,
                ),
                VerifiedSupportClaim::new(
                    rerank_scope,
                    VerifiedFidelity::Native,
                    ApiStability::Stable,
                    rerank_evidence,
                ),
            ],
            catalog,
        )?;
        let scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(protocol)
                .with_api_mode(api_mode),
        );
        Ok(Self {
            profile: Arc::new(profile),
            scope,
        })
    }

    pub fn custom() -> Result<Self, CohereProfileError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new("custom-cohere-v2")?;
        let protocol = ProtocolId::new(PROTOCOL_ID)?;
        let api_mode = ApiModeId::new(API_MODE_ID)?;
        let claims = [ModelFamily::Embedding, ModelFamily::Rerank]
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
                    ApiStability::Experimental,
                )
            })
            .collect();
        let profile = ProviderProfile::generic_many(ProfileId::new("cohere-custom")?, claims)?;
        let scope = Arc::new(
            ProviderScope::new(provider)
                .with_platform(platform)
                .with_protocol(protocol)
                .with_api_mode(api_mode),
        );
        Ok(Self {
            profile: Arc::new(profile),
            scope,
        })
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub(crate) fn scope(&self) -> Arc<ProviderScope> {
        self.scope.clone()
    }
}

fn evidence(source: &str, contract: &str) -> Result<VerificationEvidence, CohereProfileError> {
    let verified_at = VerificationDate::new(
        NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
            .map_err(|_| CohereProfileError::InvalidVerificationDate)?,
    );
    Ok(VerificationEvidence::new(
        OfficialSource::new(source)?,
        verified_at,
        ProtocolContractId::new(contract)?,
    ))
}

fn model_profile(
    model: &str,
    scope: SupportScope,
    operation: ModelOperation,
    evidence: VerificationEvidence,
) -> ModelProfile {
    ModelProfile::new(
        ModelId::new(model).expect("Cohere model IDs are static"),
        scope,
        [operation],
        ModelLifecycle::Active,
        evidence,
    )
    .expect("Cohere model profiles declare one operation")
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CohereProfileError {
    #[error("invalid Cohere profile identifier: {0}")]
    Identifier(#[from] siumai_core::InvalidId),
    #[error("invalid Cohere support profile: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid Cohere model catalog: {0}")]
    Catalog(#[from] siumai_core::CatalogError),
    #[error("Cohere verification date is invalid")]
    InvalidVerificationDate,
}
