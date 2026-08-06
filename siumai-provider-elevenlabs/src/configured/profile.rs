use std::fmt;
use std::sync::Arc;

use chrono::NaiveDate;
use siumai_core::{
    ApiModeId, ApiStability, GenericSupportClaim, ModelCatalog, ModelFamily, ModelId,
    ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileId,
    ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ProviderScope, SpeechLimits,
    SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use siumai_transport::{EndpointConfig, OfficialOrigin};

use super::models;
use super::provider::ElevenLabsConfigError;

pub const PROVIDER_ID: &str = "elevenlabs";
pub const PROTOCOL_ID: &str = "elevenlabs-native";
pub const API_MODE_ID: &str = "text-to-speech";
pub const VERIFIED_ON: &str = "2026-08-04";
pub const OFFICIAL_SOURCE: &str = "https://elevenlabs.io/docs/api-reference/text-to-speech/convert";
pub const MODEL_SOURCE: &str = "https://elevenlabs.io/docs/overview/models";

/// Immutable endpoint and evidence profile for one ElevenLabs runtime.
#[derive(Clone)]
pub struct ElevenLabsProfile {
    profile: Arc<ProviderProfile>,
    support_scope: SupportScope,
    scope: Arc<ProviderScope>,
    endpoint: EndpointConfig,
}

impl ElevenLabsProfile {
    pub fn official() -> Result<Self, ElevenLabsConfigError> {
        let support_scope = SupportScope::new(
            ProviderId::new(PROVIDER_ID)?,
            PlatformId::new("public-api")?,
            ModelFamily::Speech,
            ProtocolId::new(PROTOCOL_ID)?,
            ApiModeId::new(API_MODE_ID)?,
        );
        let endpoint_evidence = evidence(OFFICIAL_SOURCE);
        let model_evidence = evidence(MODEL_SOURCE);
        let model = |id: &str, lifecycle: ModelLifecycle| {
            ModelProfile::new(
                ModelId::new(id).expect("verified ElevenLabs model IDs are static"),
                support_scope.clone(),
                [ModelOperation::SynthesizeSpeech],
                lifecycle,
                model_evidence.clone(),
            )
            .expect("verified ElevenLabs model profiles declare one operation")
        };
        let catalog = ModelCatalog::new([
            model(models::ELEVEN_V3, ModelLifecycle::Active),
            model(models::ELEVEN_MULTILINGUAL_V2, ModelLifecycle::Active),
            model(models::ELEVEN_FLASH_V2_5, ModelLifecycle::Active),
            model(models::ELEVEN_FLASH_V2, ModelLifecycle::Active),
            model(
                models::ELEVEN_TURBO_V2_5,
                ModelLifecycle::Deprecated {
                    replacement: Some(
                        ModelId::new(models::ELEVEN_FLASH_V2_5)
                            .expect("replacement model ID is static"),
                    ),
                },
            ),
            model(
                models::ELEVEN_TURBO_V2,
                ModelLifecycle::Deprecated {
                    replacement: Some(
                        ModelId::new(models::ELEVEN_FLASH_V2)
                            .expect("replacement model ID is static"),
                    ),
                },
            ),
            model(models::ELEVEN_MULTILINGUAL_V1, ModelLifecycle::Active),
        ])
        .expect("verified ElevenLabs model catalog has valid replacement rows");
        let profile = ProviderProfile::verified(
            ProfileId::new(PROVIDER_ID)?,
            vec![VerifiedSupportClaim::new(
                support_scope.clone(),
                VerifiedFidelity::Native,
                ApiStability::Stable,
                endpoint_evidence,
            )],
            catalog,
        )
        .expect("verified ElevenLabs profile and catalog scopes match");
        let endpoint = EndpointConfig::official(
            "https://api.elevenlabs.io",
            OfficialOrigin::new("https://api.elevenlabs.io")
                .expect("ElevenLabs official origin is static and valid"),
        )?;
        Self::from_parts(profile, support_scope, endpoint)
    }

    pub fn public_custom(base_url: impl AsRef<str>) -> Result<Self, ElevenLabsConfigError> {
        Self::generic(base_url, false)
    }

    pub fn local_explicit(base_url: impl AsRef<str>) -> Result<Self, ElevenLabsConfigError> {
        Self::generic(base_url, true)
    }

    fn generic(base_url: impl AsRef<str>, local: bool) -> Result<Self, ElevenLabsConfigError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let support_scope = SupportScope::new(
            provider,
            PlatformId::new(if local { "local" } else { "custom-endpoint" })?,
            ModelFamily::Speech,
            ProtocolId::new(PROTOCOL_ID)?,
            ApiModeId::new(API_MODE_ID)?,
        );
        let profile = ProviderProfile::generic(
            ProfileId::new(if local {
                "elevenlabs-local"
            } else {
                "elevenlabs-custom"
            })?,
            GenericSupportClaim::new(support_scope.clone(), ApiStability::Experimental),
        );
        let endpoint = if local {
            EndpointConfig::local_explicit(base_url)
        } else {
            EndpointConfig::public_custom(base_url)
        }?;
        Self::from_parts(profile, support_scope, endpoint)
    }

    fn from_parts(
        profile: ProviderProfile,
        support_scope: SupportScope,
        endpoint: EndpointConfig,
    ) -> Result<Self, ElevenLabsConfigError> {
        if support_scope.family() != ModelFamily::Speech
            || support_scope.protocol().as_str() != PROTOCOL_ID
            || support_scope.api_mode().as_str() != API_MODE_ID
        {
            return Err(ElevenLabsConfigError::IncompatibleSupportScope);
        }
        let scope = Arc::new(
            ProviderScope::new(support_scope.provider().clone())
                .with_platform(support_scope.platform().clone())
                .with_protocol(support_scope.protocol().clone())
                .with_api_mode(support_scope.api_mode().clone()),
        );
        Ok(Self {
            profile: Arc::new(profile),
            support_scope,
            scope,
            endpoint,
        })
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub fn scope(&self) -> &ProviderScope {
        self.scope.as_ref()
    }

    pub(crate) fn scope_arc(&self) -> Arc<ProviderScope> {
        self.scope.clone()
    }

    pub fn limits_for(&self, model: &ModelId) -> SpeechLimits {
        SpeechLimits {
            max_text_bytes: None,
            max_text_chars: models::max_text_chars(model.as_str()),
        }
    }

    pub(crate) fn support_scope(&self) -> &SupportScope {
        &self.support_scope
    }

    pub(crate) fn profile_arc(&self) -> Arc<ProviderProfile> {
        self.profile.clone()
    }

    pub(crate) fn endpoint(&self) -> &EndpointConfig {
        &self.endpoint
    }
}

impl fmt::Debug for ElevenLabsProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ElevenLabsProfile")
            .field("profile_id", self.profile.id())
            .field("scope", &self.scope)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

fn evidence(source: &str) -> VerificationEvidence {
    VerificationEvidence::new(
        OfficialSource::new(source).expect("ElevenLabs evidence URL is static and valid"),
        VerificationDate::new(
            NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
                .expect("ElevenLabs verification date is static and valid"),
        ),
        ProtocolContractId::new("elevenlabs-text-to-speech-2026-08")
            .expect("ElevenLabs protocol contract ID is static and valid"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_profile_is_native_verified_and_future_models_stay_open() {
        let profile = ElevenLabsProfile::official().unwrap();
        let claims = profile
            .provider_profile()
            .verified_claims()
            .expect("official profile is verified");
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].fidelity(), VerifiedFidelity::Native);
        assert_eq!(claims[0].evidence().source().as_str(), OFFICIAL_SOURCE);
        assert_eq!(
            profile
                .limits_for(&ModelId::new(models::ELEVEN_V3).unwrap())
                .max_text_chars,
            Some(5_000)
        );
        assert_eq!(
            profile
                .limits_for(&ModelId::new("eleven-future").unwrap())
                .max_text_chars,
            None
        );
    }
}
