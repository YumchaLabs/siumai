use std::sync::Arc;

use chrono::NaiveDate;
use siumai_anthropic_compatible::{AnthropicCompatibleConfigError, AnthropicCompatibleProfile};
use siumai_core::{
    ApiModeId, ApiStability, CatalogError, InvalidId, ModelCatalog, ModelFamily, ModelId,
    ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileError,
    ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ReplayDomain,
    SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, MessagesAnnotationResolver, MessagesEncodingRules, PROTOCOL_ID,
};
use siumai_transport::EndpointConfig;
use thiserror::Error;

use crate::models::{
    CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5, CLAUDE_HAIKU_4_5_20251001, CLAUDE_MYTHOS_5,
    CLAUDE_MYTHOS_PREVIEW, CLAUDE_OPUS_4_1_20250805, CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7,
    CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5,
};
use crate::request_policy::AnthropicRequestPolicy;

pub(crate) const PROVIDER_ID: &str = "anthropic";
pub(crate) const PLATFORM_ID: &str = "anthropic-api";
pub(crate) const PROFILE_ID: &str = "anthropic";
pub(crate) const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1/";
pub(crate) const OFFICIAL_ORIGIN: &str = "https://api.anthropic.com";
pub(crate) const API_VERSION: &str = "2023-06-01";

const MESSAGES_SOURCE: &str = "https://platform.claude.com/docs/en/api/messages";
const MODELS_SOURCE: &str = "https://platform.claude.com/docs/en/about-claude/models/overview";
const DEPRECATIONS_SOURCE: &str =
    "https://platform.claude.com/docs/en/about-claude/model-deprecations";
const VERIFIED_ON: &str = "2026-08-06";

pub(crate) fn profile(
    endpoint: EndpointConfig,
    provider_verified_endpoint: bool,
    replay_domain: ReplayDomain,
    resolver: Arc<dyn MessagesAnnotationResolver>,
    beta_features: &[String],
) -> Result<AnthropicCompatibleProfile, AnthropicProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let platform = PlatformId::new(PLATFORM_ID)?;
    let mut profile = if provider_verified_endpoint {
        verified_profile(provider, platform, endpoint)?.with_replay_domain(replay_domain)?
    } else {
        AnthropicCompatibleProfile::custom(
            ProfileId::new(PROFILE_ID)?,
            provider,
            platform,
            endpoint,
            replay_domain,
            API_VERSION,
        )?
    }
    .with_annotation_resolver(resolver)
    .with_encoding_rules(MessagesEncodingRules::native())
    .with_request_policy(Arc::new(AnthropicRequestPolicy));

    for feature in beta_features {
        profile = profile.with_beta_feature(feature.clone())?;
    }
    Ok(profile)
}

fn verified_profile(
    provider: ProviderId,
    platform: PlatformId,
    endpoint: EndpointConfig,
) -> Result<AnthropicCompatibleProfile, AnthropicProfileError> {
    let scope = SupportScope::new(
        provider,
        platform,
        ModelFamily::Language,
        ProtocolId::new(PROTOCOL_ID)?,
        ApiModeId::new(API_MODE_ID)?,
    );
    let verified_at = VerificationDate::new(
        NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
            .map_err(AnthropicProfileError::VerificationDate)?,
    );
    let messages_evidence = VerificationEvidence::new(
        OfficialSource::new(MESSAGES_SOURCE)?,
        verified_at,
        ProtocolContractId::new("anthropic-messages-2026-08")?,
    );
    let model_evidence = VerificationEvidence::new(
        OfficialSource::new(MODELS_SOURCE)?,
        verified_at,
        ProtocolContractId::new("anthropic-model-catalog-2026-08")?,
    );
    let retirement_evidence = VerificationEvidence::new(
        OfficialSource::new(DEPRECATIONS_SOURCE)?,
        verified_at,
        ProtocolContractId::new("anthropic-model-retirements-2026-08")?,
    );
    let catalog = ModelCatalog::new([
        model_profile(
            CLAUDE_OPUS_5,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_SONNET_5,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_FABLE_5,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_MYTHOS_5,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_MYTHOS_PREVIEW,
            &scope,
            ModelLifecycle::Retired {
                replacement: Some(ModelId::new(CLAUDE_MYTHOS_5)?),
            },
            &retirement_evidence,
        )?,
        model_profile(
            CLAUDE_HAIKU_4_5,
            &scope,
            ModelLifecycle::RollingAlias,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_HAIKU_4_5_20251001,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_OPUS_4_8,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_OPUS_4_7,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_OPUS_4_6,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_SONNET_4_6,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_OPUS_4_1_20250805,
            &scope,
            ModelLifecycle::Retired {
                replacement: Some(ModelId::new(CLAUDE_OPUS_4_8)?),
            },
            &retirement_evidence,
        )?,
    ])?;
    let provider_profile = ProviderProfile::verified(
        ProfileId::new(PROFILE_ID)?,
        vec![VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Native,
            ApiStability::Stable,
            messages_evidence,
        )],
        catalog,
    )?;
    Ok(AnthropicCompatibleProfile::verified(
        provider_profile,
        endpoint,
        API_VERSION,
    )?)
}

fn model_profile(
    model: &str,
    scope: &SupportScope,
    lifecycle: ModelLifecycle,
    evidence: &VerificationEvidence,
) -> Result<ModelProfile, AnthropicProfileError> {
    Ok(ModelProfile::new(
        ModelId::new(model)?,
        scope.clone(),
        [ModelOperation::Generate, ModelOperation::Stream],
        lifecycle,
        evidence.clone(),
    )?)
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AnthropicProfileError {
    #[error("invalid Anthropic identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Anthropic support evidence: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid Anthropic model catalog: {0}")]
    Catalog(#[from] CatalogError),
    #[error("invalid Anthropic compatible profile: {0}")]
    Compatible(#[from] AnthropicCompatibleConfigError),
    #[error("invalid Anthropic verification date: {0}")]
    VerificationDate(#[source] chrono::ParseError),
}
