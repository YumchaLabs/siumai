use std::sync::Arc;
use std::time::Duration;

use lru::LruCache;
use tokio::sync::Mutex as TokioMutex;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::traits::AudioCapability;
use crate::types::{
    AudioFeature, AudioStream, AudioTranslationRequest, LanguageInfo, SttRequest, SttResponse,
    TtsRequest, TtsResponse, VoiceInfo,
};
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;

use super::super::ProviderFactory;
use super::super::build_context::build_registry_context;
use super::super::cache::{SpeechCacheEntry, TranscriptionCacheEntry};

fn request_model_missing(slot: Option<&str>) -> bool {
    match slot {
        Some(value) => value.trim().is_empty(),
        None => true,
    }
}

fn apply_speech_handle_default_model(mut request: TtsRequest, model_id: &str) -> TtsRequest {
    if request_model_missing(request.model.as_deref()) && !model_id.trim().is_empty() {
        request.model = Some(model_id.to_string());
    }
    request
}

fn apply_transcription_handle_default_model(mut request: SttRequest, model_id: &str) -> SttRequest {
    if request_model_missing(request.model.as_deref()) && !model_id.trim().is_empty() {
        request.model = Some(model_id.to_string());
    }
    request
}

fn apply_translation_handle_default_model(
    mut request: AudioTranslationRequest,
    model_id: &str,
) -> AudioTranslationRequest {
    if request_model_missing(request.model.as_deref()) && !model_id.trim().is_empty() {
        request.model = Some(model_id.to_string());
    }
    request
}

/// Speech model handle (TTS) - delegates to factory for client creation
#[derive(Clone)]
pub struct SpeechModelHandle {
    pub(in crate::registry::entry) factory: Arc<dyn ProviderFactory>,
    pub(in crate::registry::entry) provider_id: String,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    pub(in crate::registry::entry) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    pub(in crate::registry::entry) http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    pub(in crate::registry::entry) http_transport:
        Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level retry options copied into the handle
    pub(in crate::registry::entry) retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    pub(in crate::registry::entry) http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    pub(in crate::registry::entry) api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    pub(in crate::registry::entry) base_url: Option<String>,
    /// Shared LRU cache for speech-family models
    pub(in crate::registry::entry) cache: Arc<TokioMutex<LruCache<String, SpeechCacheEntry>>>,
    /// TTL for cached speech-family models
    pub(in crate::registry::entry) client_ttl: Option<Duration>,
}

/// Implementation of AudioCapability for SpeechModelHandle
///
/// This allows the handle to be used directly as a TTS client, aligning with
/// Vercel AI SDK's design where registry.speechModel() returns a callable model.
#[async_trait::async_trait]
impl AudioCapability for SpeechModelHandle {
    fn supported_features(&self) -> &[AudioFeature] {
        &[AudioFeature::TextToSpeech]
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        let model = self.get_or_create_speech_model(&self.model_id).await?;
        model
            .synthesize(apply_speech_handle_default_model(request, &self.model_id))
            .await
    }

    async fn text_to_speech_stream(&self, request: TtsRequest) -> Result<AudioStream, LlmError> {
        let client = self.build_speech_client(&self.model_id).await?;
        let extras = client.as_speech_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support streaming text-to-speech.",
                self.provider_id
            ))
        })?;

        extras
            .tts_stream(apply_speech_handle_default_model(request, &self.model_id))
            .await
    }

    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        let client = self.build_speech_client(&self.model_id).await?;
        let extras = client.as_speech_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support voice listing.",
                self.provider_id
            ))
        })?;

        extras.get_voices().await
    }
}

impl crate::traits::ModelMetadata for SpeechModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl SpeechModelHandle {
    async fn build_speech_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        self.factory
            .compat_speech_client_with_ctx(model_id, &ctx)
            .await
    }

    async fn get_or_create_speech_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let cache_key = format!("{}:{}", self.provider_id, model_id);

        let mut cache = self.cache.lock().await;
        if let Some(entry) = cache.get(&cache_key) {
            if !entry.is_expired(self.client_ttl) {
                return Ok(entry.model.clone());
            }
            cache.pop(&cache_key);
        }

        drop(cache);
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let model = self
            .factory
            .speech_model_family_with_ctx(model_id, &ctx)
            .await?;

        let mut cache = self.cache.lock().await;
        cache.put(cache_key, SpeechCacheEntry::new(model.clone()));

        Ok(model)
    }

    /// Text to speech (deprecated - use trait method directly)
    #[deprecated(
        since = "0.10.3",
        note = "Use the AudioCapability trait method directly"
    )]
    pub async fn text_to_speech(
        &self,
        req: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        AudioCapability::text_to_speech(self, req).await
    }
}

/// Transcription model handle (STT) - delegates to factory for client creation
#[derive(Clone)]
pub struct TranscriptionModelHandle {
    pub(in crate::registry::entry) factory: Arc<dyn ProviderFactory>,
    pub(in crate::registry::entry) provider_id: String,
    pub model_id: String,
    /// Registry-level HTTP interceptors to attempt injecting into clients
    pub(in crate::registry::entry) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Registry-level pre-built HTTP client copied into the handle
    pub(in crate::registry::entry) http_client: Option<reqwest::Client>,
    /// Registry-level custom HTTP transport copied into the handle
    pub(in crate::registry::entry) http_transport:
        Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Registry-level retry options copied into the handle
    pub(in crate::registry::entry) retry_options: Option<RetryOptions>,
    /// Registry-level HTTP configuration copied into the handle
    pub(in crate::registry::entry) http_config: Option<crate::types::HttpConfig>,
    /// Registry-level API key copied into the handle
    pub(in crate::registry::entry) api_key: Option<String>,
    /// Registry-level base URL copied into the handle
    pub(in crate::registry::entry) base_url: Option<String>,
    /// Shared LRU cache for transcription-family models
    pub(in crate::registry::entry) cache:
        Arc<TokioMutex<LruCache<String, TranscriptionCacheEntry>>>,
    /// TTL for cached transcription-family models
    pub(in crate::registry::entry) client_ttl: Option<Duration>,
}

/// Implementation of AudioCapability for TranscriptionModelHandle
///
/// This allows the handle to be used directly as an STT client, aligning with
/// Vercel AI SDK's design where registry.transcriptionModel() returns a callable model.
#[async_trait::async_trait]
impl AudioCapability for TranscriptionModelHandle {
    fn supported_features(&self) -> &[AudioFeature] {
        &[AudioFeature::SpeechToText]
    }

    async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        let model = self
            .get_or_create_transcription_model(&self.model_id)
            .await?;
        model
            .transcribe(apply_transcription_handle_default_model(
                request,
                &self.model_id,
            ))
            .await
    }

    async fn speech_to_text_stream(&self, request: SttRequest) -> Result<AudioStream, LlmError> {
        let client = self.build_transcription_client(&self.model_id).await?;
        let extras = client.as_transcription_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support streaming speech-to-text.",
                self.provider_id
            ))
        })?;

        extras
            .stt_stream(apply_transcription_handle_default_model(
                request,
                &self.model_id,
            ))
            .await
    }

    async fn translate_audio(
        &self,
        request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        let client = self.build_transcription_client(&self.model_id).await?;
        let extras = client.as_transcription_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support audio translation.",
                self.provider_id
            ))
        })?;

        extras
            .audio_translate(apply_translation_handle_default_model(
                request,
                &self.model_id,
            ))
            .await
    }

    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        let client = self.build_transcription_client(&self.model_id).await?;
        let extras = client.as_transcription_extras().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Provider {} does not support language listing.",
                self.provider_id
            ))
        })?;

        extras.get_supported_languages().await
    }
}

impl crate::traits::ModelMetadata for TranscriptionModelHandle {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl TranscriptionModelHandle {
    async fn build_transcription_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        self.factory
            .compat_transcription_client_with_ctx(model_id, &ctx)
            .await
    }

    async fn get_or_create_transcription_model(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        let cache_key = format!("{}:{}", self.provider_id, model_id);

        let mut cache = self.cache.lock().await;
        if let Some(entry) = cache.get(&cache_key) {
            if !entry.is_expired(self.client_ttl) {
                return Ok(entry.model.clone());
            }
            cache.pop(&cache_key);
        }

        drop(cache);
        let ctx = build_registry_context(
            &self.provider_id,
            &self.http_interceptors,
            &self.retry_options,
            &self.http_client,
            &self.http_transport,
            &self.http_config,
            &self.api_key,
            &self.base_url,
            None,
            None,
        );
        let model = self
            .factory
            .transcription_model_family_with_ctx(model_id, &ctx)
            .await?;

        let mut cache = self.cache.lock().await;
        cache.put(cache_key, TranscriptionCacheEntry::new(model.clone()));

        Ok(model)
    }
}
