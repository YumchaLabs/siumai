use std::sync::Arc;

use chrono::NaiveDate;
use siumai_anthropic_compatible::{AnthropicCompatibleConfigError, AnthropicCompatibleProfile};
use siumai_core::{
    ApiModeId, ApiStability, CatalogError, InvalidId, ModelCatalog, ModelFamily, ModelId,
    ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileError,
    ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ReplayDomain,
    SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use siumai_protocol_anthropic::messages::{API_MODE_ID, MessagesAnnotationResolver, PROTOCOL_ID};
use siumai_transport::EndpointConfig;
use thiserror::Error;

use super::models::{
    CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5_20251001, CLAUDE_OPUS_4_1_20250805, CLAUDE_OPUS_4_5_20251101,
    CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7, CLAUDE_OPUS_4_8, CLAUDE_OPUS_4_20250514, CLAUDE_OPUS_5,
    CLAUDE_SONNET_4_5_20250929, CLAUDE_SONNET_4_6, CLAUDE_SONNET_4_20250514, CLAUDE_SONNET_5,
};
use super::projection::GoogleVertexAnthropicProjection;
use super::request_policy::GoogleVertexAnthropicRequestPolicy;

pub(crate) const PROVIDER_ID: &str = "google";
pub(crate) const PLATFORM_ID: &str = "vertex-ai";
pub(crate) const PROFILE_ID: &str = "google-vertex-anthropic";
pub(crate) const API_VERSION: &str = "vertex-2023-10-16";

const MESSAGES_SOURCE: &str =
    "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude";
const MODEL_SOURCE: &str =
    "https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai";
const VERIFIED_ON: &str = "2026-08-06";

pub(crate) fn profile(
    endpoint: EndpointConfig,
    verified_endpoint: bool,
    replay_domain: ReplayDomain,
    resolver: Arc<dyn MessagesAnnotationResolver>,
) -> Result<AnthropicCompatibleProfile, GoogleVertexAnthropicProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let platform = PlatformId::new(PLATFORM_ID)?;
    let profile = if verified_endpoint {
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
    };
    Ok(profile
        .with_annotation_resolver(resolver)
        .with_request_policy(Arc::new(GoogleVertexAnthropicRequestPolicy))
        .with_request_projection(Arc::new(GoogleVertexAnthropicProjection)))
}

fn verified_profile(
    provider: ProviderId,
    platform: PlatformId,
    endpoint: EndpointConfig,
) -> Result<AnthropicCompatibleProfile, GoogleVertexAnthropicProfileError> {
    let scope = SupportScope::new(
        provider,
        platform,
        ModelFamily::Language,
        ProtocolId::new(PROTOCOL_ID)?,
        ApiModeId::new(API_MODE_ID)?,
    );
    let verified_at = VerificationDate::new(
        NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
            .map_err(GoogleVertexAnthropicProfileError::VerificationDate)?,
    );
    let messages_evidence = VerificationEvidence::new(
        OfficialSource::new(MESSAGES_SOURCE)?,
        verified_at,
        ProtocolContractId::new("google-vertex-anthropic-messages-2026-08")?,
    );
    let model_evidence = VerificationEvidence::new(
        OfficialSource::new(MODEL_SOURCE)?,
        verified_at,
        ProtocolContractId::new("google-vertex-claude-models-2026-08")?,
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
            CLAUDE_OPUS_4_5_20251101,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_SONNET_4_5_20250929,
            &scope,
            ModelLifecycle::Active,
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_OPUS_4_1_20250805,
            &scope,
            ModelLifecycle::Deprecated { replacement: None },
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_OPUS_4_20250514,
            &scope,
            ModelLifecycle::Deprecated { replacement: None },
            &model_evidence,
        )?,
        model_profile(
            CLAUDE_SONNET_4_20250514,
            &scope,
            ModelLifecycle::Deprecated { replacement: None },
            &model_evidence,
        )?,
    ])?;
    let provider_profile = ProviderProfile::verified(
        ProfileId::new(PROFILE_ID)?,
        vec![VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Compatible,
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
) -> Result<ModelProfile, GoogleVertexAnthropicProfileError> {
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
pub enum GoogleVertexAnthropicProfileError {
    #[error("invalid Google Vertex Anthropic identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Google Vertex Anthropic support evidence: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid Google Vertex Anthropic model catalog: {0}")]
    Catalog(#[from] CatalogError),
    #[error("invalid Anthropic-compatible profile: {0}")]
    Compatible(#[from] AnthropicCompatibleConfigError),
    #[error("invalid Google Vertex Anthropic verification date: {0}")]
    VerificationDate(#[source] chrono::ParseError),
}
