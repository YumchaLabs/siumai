//! DeepSeek's verified OpenAI-compatible Chat Completions profile.

use chrono::NaiveDate;
use siumai_core::{
    ApiModeId, ApiStability, ModelCatalog, ModelFamily, ModelId, ModelLifecycle, ModelOperation,
    ModelProfile, OfficialSource, PlatformId, ProfileId, ProtocolContractId, ProtocolId,
    ProviderId, ProviderProfile, SupportScope, VerificationDate, VerificationEvidence,
    VerifiedFidelity, VerifiedSupportClaim,
};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID, ChatCompletionsDialect, PROTOCOL_ID, WireFieldName,
};
use siumai_transport::{EndpointConfig, OfficialOrigin};

use crate::{OpenAiCompatibleConfigError, OpenAiCompatibleProfile};

pub const FLASH: &str = "deepseek-v4-flash";
pub const PRO: &str = "deepseek-v4-pro";
pub const CHAT: &str = FLASH;
pub const REASONER: &str = PRO;
pub const VERIFIED_ON: &str = "2026-08-05";
pub const OFFICIAL_SOURCE: &str = "https://api-docs.deepseek.com/quick_start/pricing";

pub fn profile() -> Result<OpenAiCompatibleProfile, OpenAiCompatibleConfigError> {
    let scope = SupportScope::new(
        ProviderId::new("deepseek")?,
        PlatformId::new("public-api")?,
        ModelFamily::Language,
        ProtocolId::new(PROTOCOL_ID)?,
        ApiModeId::new(API_MODE_ID)?,
    );
    let evidence = VerificationEvidence::new(
        OfficialSource::new(OFFICIAL_SOURCE).expect("DeepSeek source URL is static and validated"),
        VerificationDate::new(
            NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
                .expect("DeepSeek verification date is static and validated"),
        ),
        ProtocolContractId::new("deepseek-v4-openai-chat-2026-08")?,
    );
    let catalog = ModelCatalog::new([
        ModelProfile::new(
            ModelId::new(CHAT)?,
            scope.clone(),
            [ModelOperation::Generate, ModelOperation::Stream],
            ModelLifecycle::Active,
            evidence.clone(),
        )
        .expect("DeepSeek chat declaration has operations"),
        ModelProfile::new(
            ModelId::new(REASONER)?,
            scope.clone(),
            [ModelOperation::Generate, ModelOperation::Stream],
            ModelLifecycle::Active,
            evidence.clone(),
        )
        .expect("DeepSeek reasoner declaration has operations"),
    ])
    .expect("DeepSeek static catalog has no duplicate or invalid replacement rows");
    let provider_profile = ProviderProfile::verified(
        ProfileId::new("deepseek")?,
        vec![VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Compatible,
            ApiStability::Stable,
            evidence,
        )],
        catalog,
    )
    .expect("DeepSeek static profile and catalog scopes match");
    let endpoint = EndpointConfig::official(
        "https://api.deepseek.com/v1",
        OfficialOrigin::new("https://api.deepseek.com")
            .expect("DeepSeek official origin is static and validated"),
    )?;
    let dialect = ChatCompletionsDialect::generic().with_reasoning_output_field(
        WireFieldName::new("reasoning_content")
            .expect("DeepSeek reasoning field is static and validated"),
    );
    OpenAiCompatibleProfile::verified(provider_profile, endpoint, dialect)
}

#[cfg(test)]
mod tests {
    use siumai_core::{ModelFamily, ModelId, ModelOperation, ModelPolicy, ModelPolicyContext};

    use super::*;
    use crate::configured::policy::OpenAiCompatibleModelPolicy;

    #[test]
    fn declaration_is_verified_exact_and_unknown_models_stay_unknown() {
        let profile = profile().unwrap();
        let claims = profile
            .provider_profile()
            .verified_claims()
            .expect("named profile is verified");
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].evidence().source().as_str(), OFFICIAL_SOURCE);
        assert_eq!(
            profile.provider_profile().catalog().unwrap().iter().count(),
            2
        );

        let policy = OpenAiCompatibleModelPolicy::new(
            profile.profile_arc(),
            profile.support_scope().clone(),
        );
        let known = policy.evaluate(&ModelPolicyContext {
            scope: profile.scope().clone(),
            model: ModelId::new(CHAT).unwrap(),
            family: ModelFamily::Language,
            operation: ModelOperation::Generate,
        });
        assert_eq!(known.state(), &siumai_core::SupportState::Supported);
        let unknown = policy.evaluate(&ModelPolicyContext {
            scope: profile.scope().clone(),
            model: ModelId::new("deepseek-future").unwrap(),
            family: ModelFamily::Language,
            operation: ModelOperation::Generate,
        });
        assert_eq!(unknown.state(), &siumai_core::SupportState::Unknown);
    }
}
