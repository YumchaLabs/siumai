//! ElevenLabs provider factory.

use super::*;
use crate::provider::ids;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;
use siumai_provider_elevenlabs::providers::elevenlabs::{
    ElevenLabsClient, ElevenLabsConfig, ElevenLabsSpeechModel, ElevenLabsTranscriptionModel,
};
use std::borrow::Cow;

#[cfg(feature = "elevenlabs")]
fn elevenlabs_capabilities() -> ProviderCapabilities {
    crate::native_provider_metadata::native_providers_metadata()
        .into_iter()
        .find(|m| m.id == ids::ELEVENLABS)
        .map(|m| m.capabilities)
        .unwrap_or_else(|| ProviderCapabilities::new().with_audio())
}

#[cfg(feature = "elevenlabs")]
fn resolve_api_key(ctx: &BuildContext) -> Result<String, LlmError> {
    crate::provider_utils::builder_helpers::get_api_key_with_envs(
        ctx.api_key.clone(),
        ids::ELEVENLABS,
        Some(ElevenLabsConfig::API_KEY_ENV),
        &[],
    )
    .map_err(|_| {
        LlmError::ConfigurationError(
            "Missing ELEVENLABS_API_KEY or explicit api_key in BuildContext".to_string(),
        )
    })
}

#[cfg(feature = "elevenlabs")]
fn resolve_base_url(ctx: &BuildContext) -> String {
    crate::provider_utils::builder_helpers::resolve_base_url(
        ctx.base_url.clone(),
        ElevenLabsConfig::DEFAULT_BASE_URL,
    )
}

#[cfg(feature = "elevenlabs")]
enum ElevenLabsAudioFamily<'a> {
    Speech(&'a str),
    Transcription(&'a str),
}

#[cfg(feature = "elevenlabs")]
fn build_client_for_family(
    family: ElevenLabsAudioFamily<'_>,
    ctx: &BuildContext,
) -> Result<ElevenLabsClient, LlmError> {
    let http_config = ctx.http_config.clone().unwrap_or_default();
    let http_client = if let Some(client) = &ctx.http_client {
        client.clone()
    } else {
        build_http_client_from_config(&http_config)?
    };

    let mut config = ElevenLabsConfig::new(resolve_api_key(ctx)?)
        .with_base_url(resolve_base_url(ctx))
        .with_http_config(http_config)
        .with_http_interceptors(ctx.http_interceptors.clone());

    match family {
        ElevenLabsAudioFamily::Speech(model_id) => {
            config = config.with_speech_model(model_id);
        }
        ElevenLabsAudioFamily::Transcription(model_id) => {
            config = config.with_transcription_model(model_id);
        }
    }

    if let Some(http_transport) = ctx.http_transport.clone() {
        config = config.with_http_transport(http_transport);
    }

    let mut client = ElevenLabsClient::with_http_client(config, http_client)?;
    if let Some(retry_options) = ctx.retry_options.clone() {
        client = client.with_retry_options(retry_options);
    }
    Ok(client)
}

#[cfg(feature = "elevenlabs")]
fn build_speech_client_arc(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<Arc<ElevenLabsClient>, LlmError> {
    let client = build_client_for_family(ElevenLabsAudioFamily::Speech(model_id), ctx)?;
    Ok(Arc::new(client))
}

#[cfg(feature = "elevenlabs")]
fn build_transcription_client_arc(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<Arc<ElevenLabsClient>, LlmError> {
    let client = build_client_for_family(ElevenLabsAudioFamily::Transcription(model_id), ctx)?;
    Ok(Arc::new(client))
}

/// ElevenLabs provider factory.
#[cfg(feature = "elevenlabs")]
pub struct ElevenLabsProviderFactory;

#[cfg(feature = "elevenlabs")]
#[async_trait::async_trait]
impl ProviderFactory for ElevenLabsProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        elevenlabs_capabilities()
    }

    async fn compat_language_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "ElevenLabs does not expose a language family path".to_string(),
        ))
    }

    async fn compat_completion_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "ElevenLabs does not expose a completion family path".to_string(),
        ))
    }

    async fn compat_embedding_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "ElevenLabs does not expose an embedding family path".to_string(),
        ))
    }

    async fn compat_image_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "ElevenLabs does not expose an image family path".to_string(),
        ))
    }

    async fn compat_speech_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_speech_client_arc(model_id, ctx)?;
        Ok(client)
    }

    async fn speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let client = build_client_for_family(ElevenLabsAudioFamily::Speech(model_id), ctx)?;
        let model: ElevenLabsSpeechModel = client.speech_model(model_id.to_string());
        Ok(Arc::new(model))
    }

    async fn compat_transcription_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_transcription_client_arc(model_id, ctx)?;
        Ok(client)
    }

    async fn transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        let client = build_client_for_family(ElevenLabsAudioFamily::Transcription(model_id), ctx)?;
        let model: ElevenLabsTranscriptionModel = client.transcription_model(model_id.to_string());
        Ok(Arc::new(model))
    }

    async fn compat_reranking_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "ElevenLabs does not expose a reranking family path".to_string(),
        ))
    }

    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::ELEVENLABS)
    }
}
