use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use siumai_core::{
    ApiModeId, ApiStability, GenericSupportClaim, ModelCatalog, ModelFamily, ModelId,
    ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileId,
    ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ProviderScope, ReplayDomain,
    ReplayDomainId, SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity,
    VerifiedSupportClaim,
};
use siumai_protocol_openai::chat_completions::PROTOCOL_ID as CHAT_COMPLETIONS_PROTOCOL;
use siumai_protocol_openai::embedding::{
    API_MODE_ID as EMBEDDING_MODE, PROTOCOL_ID as EMBEDDING_PROTOCOL,
};
use siumai_protocol_openai::image::{API_MODE_ID as IMAGE_MODE, PROTOCOL_ID as IMAGE_PROTOCOL};
use siumai_protocol_openai::responses::OPENAI_RESPONSES_PROTOCOL;
use siumai_protocol_openai::speech::{API_MODE_ID as SPEECH_MODE, PROTOCOL_ID as SPEECH_PROTOCOL};
use siumai_protocol_openai::transcription::{
    API_MODE_ID as TRANSCRIPTION_MODE, PROTOCOL_ID as TRANSCRIPTION_PROTOCOL,
};

use super::catalog::{GPT_5_6, GPT_5_6_LUNA, GPT_5_6_SOL, GPT_5_6_TERRA};
use super::embedding::{TEXT_EMBEDDING_3_LARGE, TEXT_EMBEDDING_3_SMALL, TEXT_EMBEDDING_ADA_002};
use super::image::{
    CHATGPT_IMAGE_LATEST, DALL_E_2, DALL_E_3, GPT_IMAGE_1, GPT_IMAGE_1_5, GPT_IMAGE_1_MINI,
    GPT_IMAGE_2,
};
use super::mode::OpenAiApiMode;
use super::provider::OpenAiConfigError;
use super::speech::{
    GPT_4O_MINI_TTS, GPT_4O_MINI_TTS_2025_03_20, GPT_4O_MINI_TTS_2025_12_15, TTS_1, TTS_1_1106,
    TTS_1_HD, TTS_1_HD_1106,
};
use super::transcription::{
    GPT_4O_MINI_TRANSCRIBE, GPT_4O_MINI_TRANSCRIBE_2025_03_20, GPT_4O_MINI_TRANSCRIBE_2025_12_15,
    GPT_4O_TRANSCRIBE, GPT_4O_TRANSCRIBE_DIARIZE, GPT_TRANSCRIBE, WHISPER_1,
};

pub(crate) const PROVIDER_ID: &str = "openai";
pub(crate) const PLATFORM_ID: &str = "openai-api";
const CUSTOM_PLATFORM_ID: &str = "custom-openai-api";

const MODEL_GUIDANCE_SOURCE: &str = "https://developers.openai.com/api/docs/guides/latest-model";
const RESPONSES_SOURCE: &str =
    "https://developers.openai.com/api/reference/resources/responses/methods/create";
const CHAT_COMPLETIONS_SOURCE: &str = "https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create";
const EMBEDDING_SOURCE: &str = "https://developers.openai.com/api/docs/guides/embeddings";
const IMAGE_SOURCE: &str = "https://developers.openai.com/api/docs/guides/image-generation";
const SPEECH_SOURCE: &str = "https://developers.openai.com/api/docs/guides/text-to-speech";
const TRANSCRIPTION_SOURCE: &str = "https://developers.openai.com/api/docs/guides/speech-to-text";

const MODEL_GUIDANCE_CONTRACT: &str = "openai-model-guidance-2026-08-04";
const RESPONSES_CONTRACT: &str = "openai-responses-2026-08-11";
const CHAT_COMPLETIONS_CONTRACT: &str = "openai-chat-completions-2026-08-11";
const EMBEDDING_CONTRACT: &str = "openai-embeddings-2026-08-08";
const IMAGE_CONTRACT: &str = "openai-image-generations-2026-08-15";
const SPEECH_CONTRACT: &str = "openai-audio-speech-2026-08-15";
const TRANSCRIPTION_CONTRACT: &str = "openai-audio-transcriptions-2026-08-15";

/// Evidence-backed OpenAI profile for the stable portable families and explicit language modes.
#[derive(Debug, Clone)]
pub struct OpenAiProfile {
    profile: Arc<ProviderProfile>,
    responses_provider_scope: Arc<ProviderScope>,
    chat_completions_provider_scope: Arc<ProviderScope>,
    family_provider_scopes: BTreeMap<ModelFamily, Arc<ProviderScope>>,
}

impl OpenAiProfile {
    /// Build the current provider-owned profile from dated official evidence.
    pub fn current() -> Result<Self, OpenAiConfigError> {
        let provider = ProviderId::new(PROVIDER_ID)?;
        let platform = PlatformId::new(PLATFORM_ID)?;
        let replay_domain = ReplayDomain::official(ReplayDomainId::new("official")?);

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
        let responses_provider_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(responses_protocol)
                .with_api_mode(responses_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let chat_completions_provider_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(chat_completions_protocol)
                .with_api_mode(chat_mode)
                .with_replay_domain(replay_domain.clone()),
        );

        let (embedding_scope, embedding_provider_scope) = family_scope_pair(
            &provider,
            &platform,
            ModelFamily::Embedding,
            EMBEDDING_PROTOCOL,
            EMBEDDING_MODE,
            &replay_domain,
        )?;
        let (image_scope, image_provider_scope) = family_scope_pair(
            &provider,
            &platform,
            ModelFamily::Image,
            IMAGE_PROTOCOL,
            IMAGE_MODE,
            &replay_domain,
        )?;
        let (speech_scope, speech_provider_scope) = family_scope_pair(
            &provider,
            &platform,
            ModelFamily::Speech,
            SPEECH_PROTOCOL,
            SPEECH_MODE,
            &replay_domain,
        )?;
        let (transcription_scope, transcription_provider_scope) = family_scope_pair(
            &provider,
            &platform,
            ModelFamily::Transcription,
            TRANSCRIPTION_PROTOCOL,
            TRANSCRIPTION_MODE,
            &replay_domain,
        )?;

        let model_evidence = evidence(MODEL_GUIDANCE_SOURCE, MODEL_GUIDANCE_CONTRACT, 2026, 8, 4)?;
        let responses_evidence = evidence(RESPONSES_SOURCE, RESPONSES_CONTRACT, 2026, 8, 11)?;
        let chat_evidence = evidence(
            CHAT_COMPLETIONS_SOURCE,
            CHAT_COMPLETIONS_CONTRACT,
            2026,
            8,
            11,
        )?;
        let embedding_evidence = evidence(EMBEDDING_SOURCE, EMBEDDING_CONTRACT, 2026, 8, 8)?;
        let image_evidence = evidence(IMAGE_SOURCE, IMAGE_CONTRACT, 2026, 8, 15)?;
        let speech_evidence = evidence(SPEECH_SOURCE, SPEECH_CONTRACT, 2026, 8, 15)?;
        let transcription_evidence =
            evidence(TRANSCRIPTION_SOURCE, TRANSCRIPTION_CONTRACT, 2026, 8, 15)?;

        let claims = vec![
            verified_claim(responses_scope.clone(), responses_evidence.clone()),
            verified_claim(chat_completions_scope.clone(), chat_evidence.clone()),
            verified_claim(embedding_scope.clone(), embedding_evidence.clone()),
            verified_claim(image_scope.clone(), image_evidence.clone()),
            verified_claim(speech_scope.clone(), speech_evidence.clone()),
            verified_claim(transcription_scope.clone(), transcription_evidence.clone()),
        ];
        let mut models = Vec::with_capacity(32);
        extend_models(
            &mut models,
            &responses_scope,
            &model_evidence,
            [ModelOperation::Generate, ModelOperation::Stream],
            language_models(),
        )?;
        extend_models(
            &mut models,
            &chat_completions_scope,
            &model_evidence,
            [ModelOperation::Generate, ModelOperation::Stream],
            language_models(),
        )?;
        extend_models(
            &mut models,
            &embedding_scope,
            &embedding_evidence,
            [ModelOperation::Embed],
            embedding_models(),
        )?;
        extend_models(
            &mut models,
            &image_scope,
            &image_evidence,
            [ModelOperation::GenerateImage],
            image_models()?,
        )?;
        extend_models(
            &mut models,
            &speech_scope,
            &speech_evidence,
            [ModelOperation::SynthesizeSpeech],
            speech_models(),
        )?;
        extend_models(
            &mut models,
            &transcription_scope,
            &transcription_evidence,
            [ModelOperation::Transcribe],
            transcription_models(),
        )?;
        let profile = ProviderProfile::verified(
            ProfileId::new(PROVIDER_ID)?,
            claims,
            ModelCatalog::new(models)?,
        )?;

        Ok(Self {
            profile: Arc::new(profile),
            responses_provider_scope,
            chat_completions_provider_scope,
            family_provider_scopes: BTreeMap::from([
                (ModelFamily::Embedding, embedding_provider_scope),
                (ModelFamily::Image, image_provider_scope),
                (ModelFamily::Speech, speech_provider_scope),
                (ModelFamily::Transcription, transcription_provider_scope),
            ]),
        })
    }

    /// Build an unverified profile for a caller-supplied OpenAI-compatible endpoint.
    pub fn custom(replay_domain: ReplayDomain) -> Result<Self, OpenAiConfigError> {
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
        let responses_provider_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(responses_protocol)
                .with_api_mode(responses_mode)
                .with_replay_domain(replay_domain.clone()),
        );
        let chat_completions_provider_scope = Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(chat_completions_protocol)
                .with_api_mode(chat_mode)
                .with_replay_domain(replay_domain.clone()),
        );

        let (embedding_scope, embedding_provider_scope) = family_scope_pair(
            &provider,
            &platform,
            ModelFamily::Embedding,
            EMBEDDING_PROTOCOL,
            EMBEDDING_MODE,
            &replay_domain,
        )?;
        let (image_scope, image_provider_scope) = family_scope_pair(
            &provider,
            &platform,
            ModelFamily::Image,
            IMAGE_PROTOCOL,
            IMAGE_MODE,
            &replay_domain,
        )?;
        let (speech_scope, speech_provider_scope) = family_scope_pair(
            &provider,
            &platform,
            ModelFamily::Speech,
            SPEECH_PROTOCOL,
            SPEECH_MODE,
            &replay_domain,
        )?;
        let (transcription_scope, transcription_provider_scope) = family_scope_pair(
            &provider,
            &platform,
            ModelFamily::Transcription,
            TRANSCRIPTION_PROTOCOL,
            TRANSCRIPTION_MODE,
            &replay_domain,
        )?;
        let mut claims = vec![
            GenericSupportClaim::new(responses_scope.clone(), ApiStability::Experimental),
            GenericSupportClaim::new(chat_completions_scope.clone(), ApiStability::Experimental),
        ];
        claims.extend(
            [
                embedding_scope,
                image_scope,
                speech_scope,
                transcription_scope,
            ]
            .into_iter()
            .map(|scope| GenericSupportClaim::new(scope, ApiStability::Experimental)),
        );
        let profile = ProviderProfile::generic_many(ProfileId::new("openai-custom")?, claims)?;

        Ok(Self {
            profile: Arc::new(profile),
            responses_provider_scope,
            chat_completions_provider_scope,
            family_provider_scopes: BTreeMap::from([
                (ModelFamily::Embedding, embedding_provider_scope),
                (ModelFamily::Image, image_provider_scope),
                (ModelFamily::Speech, speech_provider_scope),
                (ModelFamily::Transcription, transcription_provider_scope),
            ]),
        })
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.profile
    }

    pub(crate) fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.responses_provider_scope = Arc::new(
            self.responses_provider_scope
                .as_ref()
                .clone()
                .with_replay_domain(replay_domain.clone()),
        );
        self.chat_completions_provider_scope = Arc::new(
            self.chat_completions_provider_scope
                .as_ref()
                .clone()
                .with_replay_domain(replay_domain.clone()),
        );
        for scope in self.family_provider_scopes.values_mut() {
            *scope = Arc::new(
                scope
                    .as_ref()
                    .clone()
                    .with_replay_domain(replay_domain.clone()),
            );
        }
        self
    }

    pub(crate) fn provider_scope(&self, mode: OpenAiApiMode) -> &Arc<ProviderScope> {
        match mode {
            OpenAiApiMode::Responses => &self.responses_provider_scope,
            OpenAiApiMode::ChatCompletions => &self.chat_completions_provider_scope,
        }
    }

    pub(crate) fn family_provider_scope(&self, family: ModelFamily) -> Option<&Arc<ProviderScope>> {
        self.family_provider_scopes.get(&family)
    }
}

fn family_scope_pair(
    provider: &ProviderId,
    platform: &PlatformId,
    family: ModelFamily,
    protocol: &str,
    api_mode: &str,
    replay_domain: &ReplayDomain,
) -> Result<(SupportScope, Arc<ProviderScope>), OpenAiConfigError> {
    let protocol = ProtocolId::new(protocol)?;
    let api_mode = ApiModeId::new(api_mode)?;
    Ok((
        SupportScope::new(
            provider.clone(),
            platform.clone(),
            family,
            protocol.clone(),
            api_mode.clone(),
        ),
        Arc::new(
            ProviderScope::new(provider.clone())
                .with_platform(platform.clone())
                .with_protocol(protocol)
                .with_api_mode(api_mode)
                .with_replay_domain(replay_domain.clone()),
        ),
    ))
}

fn evidence(
    source: &str,
    contract: &str,
    year: i32,
    month: u32,
    day: u32,
) -> Result<VerificationEvidence, OpenAiConfigError> {
    let verified_at = NaiveDate::from_ymd_opt(year, month, day)
        .map(VerificationDate::new)
        .ok_or(OpenAiConfigError::InvalidVerificationDate)?;
    Ok(VerificationEvidence::new(
        OfficialSource::new(source)?,
        verified_at,
        ProtocolContractId::new(contract)?,
    ))
}

fn verified_claim(scope: SupportScope, evidence: VerificationEvidence) -> VerifiedSupportClaim {
    VerifiedSupportClaim::new(
        scope,
        VerifiedFidelity::Native,
        ApiStability::Stable,
        evidence,
    )
}

fn extend_models<const N: usize>(
    models: &mut Vec<ModelProfile>,
    scope: &SupportScope,
    evidence: &VerificationEvidence,
    operations: [ModelOperation; N],
    hints: impl IntoIterator<Item = (&'static str, ModelLifecycle)>,
) -> Result<(), OpenAiConfigError> {
    for (model, lifecycle) in hints {
        models.push(ModelProfile::new(
            ModelId::new(model)?,
            scope.clone(),
            operations,
            lifecycle,
            evidence.clone(),
        )?);
    }
    Ok(())
}

fn language_models() -> impl IntoIterator<Item = (&'static str, ModelLifecycle)> {
    [
        (GPT_5_6_SOL, ModelLifecycle::Active),
        (GPT_5_6_TERRA, ModelLifecycle::Active),
        (GPT_5_6_LUNA, ModelLifecycle::Active),
        (GPT_5_6, ModelLifecycle::RollingAlias),
    ]
}

fn embedding_models() -> impl IntoIterator<Item = (&'static str, ModelLifecycle)> {
    [
        (TEXT_EMBEDDING_3_SMALL, ModelLifecycle::Active),
        (TEXT_EMBEDDING_3_LARGE, ModelLifecycle::Active),
        (TEXT_EMBEDDING_ADA_002, ModelLifecycle::Active),
    ]
}

fn image_models() -> Result<Vec<(&'static str, ModelLifecycle)>, OpenAiConfigError> {
    let gpt_image_2 = ModelId::new(GPT_IMAGE_2)?;
    Ok(vec![
        (GPT_IMAGE_1, ModelLifecycle::Active),
        (
            GPT_IMAGE_1_MINI,
            ModelLifecycle::Deprecated {
                replacement: Some(gpt_image_2.clone()),
            },
        ),
        (
            GPT_IMAGE_1_5,
            ModelLifecycle::Deprecated {
                replacement: Some(gpt_image_2.clone()),
            },
        ),
        (GPT_IMAGE_2, ModelLifecycle::Active),
        (
            CHATGPT_IMAGE_LATEST,
            ModelLifecycle::Deprecated {
                replacement: Some(gpt_image_2.clone()),
            },
        ),
        (
            DALL_E_2,
            ModelLifecycle::Retired {
                replacement: Some(gpt_image_2.clone()),
            },
        ),
        (
            DALL_E_3,
            ModelLifecycle::Retired {
                replacement: Some(gpt_image_2),
            },
        ),
    ])
}

fn speech_models() -> impl IntoIterator<Item = (&'static str, ModelLifecycle)> {
    [
        (GPT_4O_MINI_TTS, ModelLifecycle::RollingAlias),
        (GPT_4O_MINI_TTS_2025_03_20, ModelLifecycle::Active),
        (GPT_4O_MINI_TTS_2025_12_15, ModelLifecycle::Active),
        (TTS_1, ModelLifecycle::Active),
        (TTS_1_1106, ModelLifecycle::Active),
        (TTS_1_HD, ModelLifecycle::Active),
        (TTS_1_HD_1106, ModelLifecycle::Active),
    ]
}

fn transcription_models() -> impl IntoIterator<Item = (&'static str, ModelLifecycle)> {
    [
        (WHISPER_1, ModelLifecycle::Active),
        (GPT_4O_MINI_TRANSCRIBE, ModelLifecycle::Active),
        (GPT_4O_MINI_TRANSCRIBE_2025_03_20, ModelLifecycle::Active),
        (GPT_4O_MINI_TRANSCRIBE_2025_12_15, ModelLifecycle::Active),
        (GPT_4O_TRANSCRIBE, ModelLifecycle::Active),
        (GPT_4O_TRANSCRIBE_DIARIZE, ModelLifecycle::Active),
        (GPT_TRANSCRIBE, ModelLifecycle::Active),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_has_six_portable_claims_and_open_catalog_rows() {
        let profile = OpenAiProfile::current().unwrap();
        let claims = profile.provider_profile().verified_claims().unwrap();

        assert_eq!(claims.len(), 6);
        assert!(claims.iter().all(|claim| {
            claim.fidelity() == VerifiedFidelity::Native
                && claim.stability() == ApiStability::Stable
        }));
        assert_eq!(
            profile.provider_profile().catalog().unwrap().iter().count(),
            32
        );
    }

    #[test]
    fn custom_profile_has_generic_family_claims_without_a_catalog() {
        let profile = OpenAiProfile::custom(ReplayDomain::custom(
            ReplayDomainId::new("test-relay").unwrap(),
        ))
        .unwrap();

        assert!(profile.provider_profile().verified_claims().is_none());
        assert_eq!(
            profile.provider_profile().generic_claims().unwrap().len(),
            6
        );
        assert!(profile.provider_profile().catalog().is_none());
    }

    #[test]
    fn image_lifecycle_advisories_point_to_gpt_image_2() {
        let profile = OpenAiProfile::current().unwrap();
        let scope = profile
            .provider_profile()
            .verified_claims()
            .unwrap()
            .iter()
            .find(|claim| claim.scope().family() == ModelFamily::Image)
            .unwrap()
            .scope();
        let catalog = profile.provider_profile().catalog().unwrap();

        for model in [DALL_E_2, DALL_E_3] {
            assert_eq!(
                catalog
                    .get(scope, &ModelId::new(model).unwrap())
                    .unwrap()
                    .lifecycle(),
                &ModelLifecycle::Retired {
                    replacement: Some(ModelId::new(GPT_IMAGE_2).unwrap()),
                }
            );
        }

        for model in [GPT_IMAGE_1_MINI, GPT_IMAGE_1_5, CHATGPT_IMAGE_LATEST] {
            assert_eq!(
                catalog
                    .get(scope, &ModelId::new(model).unwrap())
                    .unwrap()
                    .lifecycle(),
                &ModelLifecycle::Deprecated {
                    replacement: Some(ModelId::new(GPT_IMAGE_2).unwrap()),
                }
            );
        }
    }
}
