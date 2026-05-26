//! ElevenLabs voice catalog resources.
//!
//! ElevenLabs exposes voice listing and voice detail APIs outside Siumai's unified
//! speech/transcription families. This module keeps that surface provider-owned.

use crate::error::LlmError;
use crate::provider_utils::url::join_url;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use reqwest::Url;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::config::ElevenLabsConfig;
use super::resource_http::{
    execute_delete_json, execute_get_json, execute_multipart_json, execute_post_json,
};

/// Provider-owned client for ElevenLabs voice catalog resources.
#[derive(Clone)]
pub struct ElevenLabsVoices {
    config: ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for ElevenLabsVoices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElevenLabsVoices")
            .field("base_url", &self.config.base_url)
            .field(
                "has_api_key",
                &!self.config.api_key.expose_secret().is_empty(),
            )
            .field("has_transport", &self.config.http_transport.is_some())
            .field("interceptors", &self.config.http_interceptors.len())
            .field("has_retry", &self.retry_options.is_some())
            .finish()
    }
}

impl ElevenLabsVoices {
    pub(crate) fn new(
        config: ElevenLabsConfig,
        http_client: reqwest::Client,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            config,
            http_client,
            retry_options,
        }
    }

    /// Retrieve default voice settings using `GET /v1/voices/settings/default`.
    pub async fn default_settings(&self) -> Result<ElevenLabsVoiceSettingsResponse, LlmError> {
        self.default_settings_with_http_config(None).await
    }

    /// Retrieve default voice settings with per-request HTTP configuration.
    pub async fn default_settings_with_http_config(
        &self,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsVoiceSettingsResponse, LlmError> {
        let url = join_url(&self.base_url(), "v1/voices/settings/default");
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "get default voice settings",
        )
        .await
    }

    /// Retrieve settings for a voice using `GET /v1/voices/{voice_id}/settings`.
    pub async fn settings(
        &self,
        voice_id: impl AsRef<str>,
    ) -> Result<ElevenLabsVoiceSettingsResponse, LlmError> {
        self.settings_with_http_config(voice_id, None).await
    }

    /// Retrieve settings for a voice with per-request HTTP configuration.
    pub async fn settings_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsVoiceSettingsResponse, LlmError> {
        let url = self.voice_action_url(voice_id, "settings")?;
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "get voice settings",
        )
        .await
    }

    /// Update settings for a voice using `POST /v1/voices/{voice_id}/settings/edit`.
    pub async fn update_settings(
        &self,
        voice_id: impl AsRef<str>,
        request: ElevenLabsUpdateVoiceSettingsRequest,
    ) -> Result<ElevenLabsVoiceSettingsUpdateResponse, LlmError> {
        request.validate()?;
        let url = self.voice_action_url(voice_id, "settings/edit")?;
        let body = request.body()?;
        execute_post_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            body,
            request.http_config.as_ref(),
            "update voice settings",
        )
        .await
    }

    /// Create an instant voice clone using `POST /v1/voices/add`.
    pub async fn create_ivc_voice(
        &self,
        request: ElevenLabsCreateIvcVoiceRequest,
    ) -> Result<ElevenLabsCreateIvcVoiceResponse, LlmError> {
        request.validate()?;
        let url = join_url(&self.base_url(), "v1/voices/add");
        let request_clone = request.clone();
        execute_multipart_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            move || request_clone.build_form(),
            request.http_config.as_ref(),
            "create IVC voice",
        )
        .await
    }

    /// Create a professional voice clone using `POST /v1/voices/pvc`.
    pub async fn create_pvc_voice(
        &self,
        request: ElevenLabsCreatePvcVoiceRequest,
    ) -> Result<ElevenLabsPvcVoiceResponse, LlmError> {
        request.validate()?;
        let url = join_url(&self.base_url(), "v1/voices/pvc");
        let body = request.body()?;
        execute_post_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            body,
            request.http_config.as_ref(),
            "create PVC voice",
        )
        .await
    }

    /// Update professional voice clone metadata using `POST /v1/voices/pvc/{voice_id}`.
    pub async fn update_pvc_voice(
        &self,
        voice_id: impl AsRef<str>,
        request: ElevenLabsUpdatePvcVoiceRequest,
    ) -> Result<ElevenLabsPvcVoiceResponse, LlmError> {
        request.validate()?;
        let url = self.pvc_voice_url(voice_id)?;
        let body = request.body()?;
        execute_post_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            body,
            request.http_config.as_ref(),
            "update PVC voice",
        )
        .await
    }

    /// Start professional voice clone training using `POST /v1/voices/pvc/{voice_id}/train`.
    pub async fn train_pvc_voice(
        &self,
        voice_id: impl AsRef<str>,
        request: ElevenLabsTrainPvcVoiceRequest,
    ) -> Result<ElevenLabsVoiceStatusResponse, LlmError> {
        request.validate()?;
        let url = self.pvc_voice_action_url(voice_id, "train")?;
        let body = request.body()?;
        execute_post_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            body,
            request.http_config.as_ref(),
            "train PVC voice",
        )
        .await
    }

    /// Add samples to a professional voice clone using `POST /v1/voices/pvc/{voice_id}/samples`.
    pub async fn add_pvc_voice_samples(
        &self,
        voice_id: impl AsRef<str>,
        request: ElevenLabsAddPvcVoiceSamplesRequest,
    ) -> Result<Vec<ElevenLabsPvcVoiceSample>, LlmError> {
        request.validate()?;
        let url = self.pvc_voice_samples_url(voice_id)?;
        let request_clone = request.clone();
        execute_multipart_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            move || request_clone.build_form(),
            request.http_config.as_ref(),
            "add PVC voice samples",
        )
        .await
    }

    /// Update a professional voice clone sample using
    /// `POST /v1/voices/pvc/{voice_id}/samples/{sample_id}`.
    pub async fn update_pvc_voice_sample(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        request: ElevenLabsUpdatePvcVoiceSampleRequest,
    ) -> Result<ElevenLabsPvcVoiceResponse, LlmError> {
        request.validate()?;
        let url = self.pvc_voice_sample_url(voice_id, sample_id)?;
        let body = request.body()?;
        execute_post_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            body,
            request.http_config.as_ref(),
            "update PVC voice sample",
        )
        .await
    }

    /// Delete a professional voice clone sample using
    /// `DELETE /v1/voices/pvc/{voice_id}/samples/{sample_id}`.
    pub async fn delete_pvc_voice_sample(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
    ) -> Result<ElevenLabsVoiceStatusResponse, LlmError> {
        self.delete_pvc_voice_sample_with_http_config(voice_id, sample_id, None)
            .await
    }

    /// Delete a professional voice clone sample with per-request HTTP configuration.
    pub async fn delete_pvc_voice_sample_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsVoiceStatusResponse, LlmError> {
        let url = self.pvc_voice_sample_url(voice_id, sample_id)?;
        execute_delete_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "delete PVC voice sample",
        )
        .await
    }

    /// Retrieve a professional voice clone sample audio preview using
    /// `GET /v1/voices/pvc/{voice_id}/samples/{sample_id}/audio`.
    pub async fn get_pvc_voice_sample_audio(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        query: Option<ElevenLabsPvcVoiceSampleAudioQuery>,
    ) -> Result<ElevenLabsPvcVoiceSampleAudioResponse, LlmError> {
        let url = self.pvc_voice_sample_audio_url(voice_id, sample_id, query.as_ref())?;
        let http_config = query.as_ref().and_then(|query| query.http_config.as_ref());
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "get PVC voice sample audio",
        )
        .await
    }

    /// Retrieve a professional voice clone sample waveform using
    /// `GET /v1/voices/pvc/{voice_id}/samples/{sample_id}/waveform`.
    pub async fn get_pvc_voice_sample_waveform(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
    ) -> Result<ElevenLabsPvcVoiceSampleWaveformResponse, LlmError> {
        self.get_pvc_voice_sample_waveform_with_http_config(voice_id, sample_id, None)
            .await
    }

    /// Retrieve a professional voice clone sample waveform with per-request HTTP configuration.
    pub async fn get_pvc_voice_sample_waveform_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsPvcVoiceSampleWaveformResponse, LlmError> {
        let url = self.pvc_voice_sample_action_url(voice_id, sample_id, "waveform")?;
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "get PVC voice sample waveform",
        )
        .await
    }

    /// Retrieve speaker separation state using
    /// `GET /v1/voices/pvc/{voice_id}/samples/{sample_id}/speakers`.
    pub async fn get_pvc_voice_sample_speakers(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
    ) -> Result<ElevenLabsPvcSpeakerSeparationResponse, LlmError> {
        self.get_pvc_voice_sample_speakers_with_http_config(voice_id, sample_id, None)
            .await
    }

    /// Retrieve speaker separation state with per-request HTTP configuration.
    pub async fn get_pvc_voice_sample_speakers_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsPvcSpeakerSeparationResponse, LlmError> {
        let url = self.pvc_voice_sample_action_url(voice_id, sample_id, "speakers")?;
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "get PVC voice sample speakers",
        )
        .await
    }

    /// Start speaker separation using
    /// `POST /v1/voices/pvc/{voice_id}/samples/{sample_id}/separate-speakers`.
    pub async fn start_pvc_voice_sample_speaker_separation(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
    ) -> Result<ElevenLabsVoiceStatusResponse, LlmError> {
        self.start_pvc_voice_sample_speaker_separation_with_http_config(voice_id, sample_id, None)
            .await
    }

    /// Start speaker separation with per-request HTTP configuration.
    pub async fn start_pvc_voice_sample_speaker_separation_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsVoiceStatusResponse, LlmError> {
        let url = self.pvc_voice_sample_action_url(voice_id, sample_id, "separate-speakers")?;
        execute_post_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            Value::Object(Default::default()),
            http_config,
            "start PVC voice sample speaker separation",
        )
        .await
    }

    /// Retrieve separated speaker audio using
    /// `GET /v1/voices/pvc/{voice_id}/samples/{sample_id}/speakers/{speaker_id}/audio`.
    pub async fn get_pvc_voice_sample_speaker_audio(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        speaker_id: impl AsRef<str>,
    ) -> Result<ElevenLabsPvcSpeakerAudioResponse, LlmError> {
        self.get_pvc_voice_sample_speaker_audio_with_http_config(
            voice_id, sample_id, speaker_id, None,
        )
        .await
    }

    /// Retrieve separated speaker audio with per-request HTTP configuration.
    pub async fn get_pvc_voice_sample_speaker_audio_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        speaker_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsPvcSpeakerAudioResponse, LlmError> {
        let url = self.pvc_voice_sample_speaker_audio_url(voice_id, sample_id, speaker_id)?;
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "get PVC voice sample speaker audio",
        )
        .await
    }

    /// Delete a voice using `DELETE /v1/voices/{voice_id}`.
    pub async fn delete_voice(
        &self,
        voice_id: impl AsRef<str>,
    ) -> Result<ElevenLabsVoiceStatusResponse, LlmError> {
        self.delete_voice_with_http_config(voice_id, None).await
    }

    /// Delete a voice with per-request HTTP configuration.
    pub async fn delete_voice_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsVoiceStatusResponse, LlmError> {
        let url = self.voice_url(voice_id)?;
        execute_delete_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "delete voice",
        )
        .await
    }

    /// Delete a voice sample using `DELETE /v1/voices/{voice_id}/samples/{sample_id}`.
    pub async fn delete_sample(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
    ) -> Result<ElevenLabsVoiceStatusResponse, LlmError> {
        self.delete_sample_with_http_config(voice_id, sample_id, None)
            .await
    }

    /// Delete a voice sample with per-request HTTP configuration.
    pub async fn delete_sample_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsVoiceStatusResponse, LlmError> {
        let url = self.voice_sample_url(voice_id, sample_id)?;
        execute_delete_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "delete voice sample",
        )
        .await
    }

    /// List voices using `GET /v2/voices`.
    pub async fn list(
        &self,
        query: Option<ElevenLabsVoiceListQuery>,
    ) -> Result<ElevenLabsVoiceListResponse, LlmError> {
        let url = self.list_url(query.as_ref())?;
        let per_request_http_config = query.as_ref().and_then(|query| query.http_config.as_ref());
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            per_request_http_config,
            "list voices",
        )
        .await
    }

    /// Retrieve a single voice using `GET /v1/voices/{voice_id}`.
    pub async fn get(&self, voice_id: impl AsRef<str>) -> Result<ElevenLabsVoice, LlmError> {
        self.get_with_http_config(voice_id, None).await
    }

    /// Retrieve a single voice with per-request HTTP configuration.
    pub async fn get_with_http_config(
        &self,
        voice_id: impl AsRef<str>,
        http_config: Option<&HttpConfig>,
    ) -> Result<ElevenLabsVoice, LlmError> {
        let voice_id = voice_id.as_ref().trim();
        if voice_id.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice_id cannot be empty".to_string(),
            ));
        }

        let encoded = urlencoding::encode(voice_id);
        let url = join_url(&self.base_url(), &format!("v1/voices/{encoded}"));
        execute_get_json(
            &self.config,
            self.http_client.clone(),
            self.retry_options.clone(),
            &url,
            http_config,
            "get voice",
        )
        .await
    }

    fn base_url(&self) -> String {
        self.config.base_url.trim_end_matches('/').to_string()
    }

    fn list_url(&self, query: Option<&ElevenLabsVoiceListQuery>) -> Result<String, LlmError> {
        let mut url = Url::parse(&join_url(&self.base_url(), "v2/voices"))
            .map_err(|e| LlmError::InvalidInput(format!("Invalid ElevenLabs voices URL: {e}")))?;

        if let Some(query) = query {
            query.validate()?;
            let pairs = query.to_query_pairs();
            if !pairs.is_empty() {
                url.query_pairs_mut().extend_pairs(pairs.iter());
            }
        }

        Ok(url.to_string())
    }

    fn voice_url(&self, voice_id: impl AsRef<str>) -> Result<String, LlmError> {
        let voice_id = voice_id.as_ref().trim();
        if voice_id.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice_id cannot be empty".to_string(),
            ));
        }

        let encoded = urlencoding::encode(voice_id);
        Ok(join_url(&self.base_url(), &format!("v1/voices/{encoded}")))
    }

    fn voice_action_url(
        &self,
        voice_id: impl AsRef<str>,
        action: &str,
    ) -> Result<String, LlmError> {
        let base = self.voice_url(voice_id)?;
        Ok(format!("{base}/{action}"))
    }

    fn voice_sample_url(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
    ) -> Result<String, LlmError> {
        let base = self.voice_action_url(voice_id, "samples")?;
        let sample_id = sample_id.as_ref().trim();
        if sample_id.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs sample_id cannot be empty".to_string(),
            ));
        }

        let encoded = urlencoding::encode(sample_id);
        Ok(format!("{base}/{encoded}"))
    }

    fn pvc_voice_url(&self, voice_id: impl AsRef<str>) -> Result<String, LlmError> {
        let voice_id = voice_id.as_ref().trim();
        if voice_id.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice_id cannot be empty".to_string(),
            ));
        }

        let encoded = urlencoding::encode(voice_id);
        Ok(join_url(
            &self.base_url(),
            &format!("v1/voices/pvc/{encoded}"),
        ))
    }

    fn pvc_voice_action_url(
        &self,
        voice_id: impl AsRef<str>,
        action: &str,
    ) -> Result<String, LlmError> {
        let base = self.pvc_voice_url(voice_id)?;
        Ok(format!("{base}/{action}"))
    }

    fn pvc_voice_samples_url(&self, voice_id: impl AsRef<str>) -> Result<String, LlmError> {
        self.pvc_voice_action_url(voice_id, "samples")
    }

    fn pvc_voice_sample_url(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
    ) -> Result<String, LlmError> {
        let base = self.pvc_voice_samples_url(voice_id)?;
        let sample_id = sample_id.as_ref().trim();
        if sample_id.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs sample_id cannot be empty".to_string(),
            ));
        }

        let encoded = urlencoding::encode(sample_id);
        Ok(format!("{base}/{encoded}"))
    }

    fn pvc_voice_sample_action_url(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        action: &str,
    ) -> Result<String, LlmError> {
        let base = self.pvc_voice_sample_url(voice_id, sample_id)?;
        Ok(format!("{base}/{action}"))
    }

    fn pvc_voice_sample_audio_url(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        query: Option<&ElevenLabsPvcVoiceSampleAudioQuery>,
    ) -> Result<String, LlmError> {
        let base = self.pvc_voice_sample_action_url(voice_id, sample_id, "audio")?;
        let mut url = Url::parse(&base).map_err(|e| {
            LlmError::InvalidInput(format!("Invalid ElevenLabs PVC audio URL: {e}"))
        })?;

        if let Some(query) = query {
            if let Some(remove_background_noise) = query.remove_background_noise {
                url.query_pairs_mut().append_pair(
                    "remove_background_noise",
                    &remove_background_noise.to_string(),
                );
            }
        }

        Ok(url.to_string())
    }

    fn pvc_voice_sample_speaker_audio_url(
        &self,
        voice_id: impl AsRef<str>,
        sample_id: impl AsRef<str>,
        speaker_id: impl AsRef<str>,
    ) -> Result<String, LlmError> {
        let base = self.pvc_voice_sample_action_url(voice_id, sample_id, "speakers")?;
        let speaker_id = speaker_id.as_ref().trim();
        if speaker_id.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs speaker_id cannot be empty".to_string(),
            ));
        }

        let encoded = urlencoding::encode(speaker_id);
        Ok(format!("{base}/{encoded}/audio"))
    }
}

/// Query parameters for `GET /v2/voices`.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsVoiceListQuery {
    pub next_page_token: Option<String>,
    pub page_size: Option<u32>,
    pub search: Option<String>,
    pub sort: Option<String>,
    pub sort_direction: Option<String>,
    pub voice_type: Option<String>,
    pub category: Option<String>,
    pub fine_tuning_state: Option<String>,
    pub collection_id: Option<String>,
    pub include_total_count: Option<bool>,
    pub voice_ids: Vec<String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsVoiceListQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_next_page_token(mut self, value: impl Into<String>) -> Self {
        self.next_page_token = Some(value.into());
        self
    }

    pub fn with_page_size(mut self, value: u32) -> Self {
        self.page_size = Some(value);
        self
    }

    pub fn with_search(mut self, value: impl Into<String>) -> Self {
        self.search = Some(value.into());
        self
    }

    pub fn with_sort(mut self, value: impl Into<String>) -> Self {
        self.sort = Some(value.into());
        self
    }

    pub fn with_sort_direction(mut self, value: impl Into<String>) -> Self {
        self.sort_direction = Some(value.into());
        self
    }

    pub fn with_voice_type(mut self, value: impl Into<String>) -> Self {
        self.voice_type = Some(value.into());
        self
    }

    pub fn with_category(mut self, value: impl Into<String>) -> Self {
        self.category = Some(value.into());
        self
    }

    pub fn with_fine_tuning_state(mut self, value: impl Into<String>) -> Self {
        self.fine_tuning_state = Some(value.into());
        self
    }

    pub fn with_collection_id(mut self, value: impl Into<String>) -> Self {
        self.collection_id = Some(value.into());
        self
    }

    pub fn with_include_total_count(mut self, value: bool) -> Self {
        self.include_total_count = Some(value);
        self
    }

    pub fn with_voice_id(mut self, value: impl Into<String>) -> Self {
        self.voice_ids.push(value.into());
        self
    }

    pub fn with_voice_ids<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.voice_ids.extend(values.into_iter().map(Into::into));
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.page_size == Some(0) {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice list page_size must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }

    fn to_query_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        push_optional(
            &mut pairs,
            "next_page_token",
            self.next_page_token.as_deref(),
        );
        if let Some(page_size) = self.page_size {
            pairs.push(("page_size".to_string(), page_size.to_string()));
        }
        push_optional(&mut pairs, "search", self.search.as_deref());
        push_optional(&mut pairs, "sort", self.sort.as_deref());
        push_optional(&mut pairs, "sort_direction", self.sort_direction.as_deref());
        push_optional(&mut pairs, "voice_type", self.voice_type.as_deref());
        push_optional(&mut pairs, "category", self.category.as_deref());
        push_optional(
            &mut pairs,
            "fine_tuning_state",
            self.fine_tuning_state.as_deref(),
        );
        push_optional(&mut pairs, "collection_id", self.collection_id.as_deref());
        if let Some(include_total_count) = self.include_total_count {
            pairs.push((
                "include_total_count".to_string(),
                include_total_count.to_string(),
            ));
        }
        for voice_id in &self.voice_ids {
            push_optional(&mut pairs, "voice_ids", Some(voice_id));
        }
        pairs
    }
}

fn push_optional(pairs: &mut Vec<(String, String)>, key: &str, value: Option<&str>) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        pairs.push((key.to_string(), value.to_string()));
    }
}

fn optional_trimmed(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Response body for `GET /v2/voices`.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoiceListResponse {
    #[serde(default)]
    pub voices: Vec<ElevenLabsVoice>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(default)]
    pub total_count: Option<u64>,
    #[serde(default)]
    pub next_page_token: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Voice object returned by ElevenLabs voice APIs.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoice {
    pub voice_id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub samples: Option<Vec<Value>>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub fine_tuning: Option<Value>,
    #[serde(default)]
    pub labels: HashMap<String, Value>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub preview_url: Option<String>,
    #[serde(default)]
    pub available_for_tiers: Option<Vec<String>>,
    #[serde(default)]
    pub settings: Option<ElevenLabsVoiceSettingsResponse>,
    #[serde(default)]
    pub sharing: Option<Value>,
    #[serde(default)]
    pub high_quality_base_model_ids: Option<Vec<String>>,
    #[serde(default)]
    pub verified_languages: Option<Vec<ElevenLabsVerifiedLanguage>>,
    #[serde(default)]
    pub collection_ids: Option<Vec<String>>,
    #[serde(default)]
    pub safety_control: Option<String>,
    #[serde(default)]
    pub voice_verification: Option<Value>,
    #[serde(default)]
    pub permission_on_resource: Option<String>,
    #[serde(default)]
    pub is_owner: Option<bool>,
    #[serde(default)]
    pub is_legacy: Option<bool>,
    #[serde(default)]
    pub is_mixed: Option<bool>,
    #[serde(default)]
    pub favorited_at_unix: Option<i64>,
    #[serde(default)]
    pub created_at_unix: Option<i64>,
    #[serde(default)]
    pub is_bookmarked: Option<bool>,
    #[serde(default)]
    pub recording_quality: Option<String>,
    #[serde(default)]
    pub labelling_status: Option<String>,
    #[serde(default)]
    pub recording_quality_reason: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Voice settings returned by ElevenLabs voice APIs.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoiceSettingsResponse {
    #[serde(default)]
    pub stability: Option<f64>,
    #[serde(default)]
    pub similarity_boost: Option<f64>,
    #[serde(default)]
    pub style: Option<f64>,
    #[serde(default)]
    pub use_speaker_boost: Option<bool>,
    #[serde(default)]
    pub speed: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Request body for updating ElevenLabs voice settings.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsUpdateVoiceSettingsRequest {
    pub stability: Option<f64>,
    pub similarity_boost: Option<f64>,
    pub style: Option<f64>,
    pub use_speaker_boost: Option<bool>,
    pub speed: Option<f64>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsUpdateVoiceSettingsRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_stability(mut self, value: f64) -> Self {
        self.stability = Some(value);
        self
    }

    pub const fn with_similarity_boost(mut self, value: f64) -> Self {
        self.similarity_boost = Some(value);
        self
    }

    pub const fn with_style(mut self, value: f64) -> Self {
        self.style = Some(value);
        self
    }

    pub const fn with_use_speaker_boost(mut self, value: bool) -> Self {
        self.use_speaker_boost = Some(value);
        self
    }

    pub const fn with_speed(mut self, value: f64) -> Self {
        self.speed = Some(value);
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.stability.is_none()
            && self.similarity_boost.is_none()
            && self.style.is_none()
            && self.use_speaker_boost.is_none()
            && self.speed.is_none()
        {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice settings update request cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn body(&self) -> Result<Value, LlmError> {
        #[derive(Serialize)]
        struct Body {
            #[serde(skip_serializing_if = "Option::is_none")]
            stability: Option<f64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            similarity_boost: Option<f64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            style: Option<f64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            use_speaker_boost: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            speed: Option<f64>,
        }

        serde_json::to_value(Body {
            stability: self.stability,
            similarity_boost: self.similarity_boost,
            style: self.style,
            use_speaker_boost: self.use_speaker_boost,
            speed: self.speed,
        })
        .map_err(|e| {
            LlmError::InvalidInput(format!(
                "Invalid ElevenLabs voice settings update request: {e}"
            ))
        })
    }
}

/// Response body for `POST /v1/voices/{voice_id}/settings/edit`.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoiceSettingsUpdateResponse {
    pub status: String,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Caller-provided audio file used by ElevenLabs voice mutation endpoints.
#[derive(Debug, Clone)]
pub struct ElevenLabsVoiceSampleFile {
    pub bytes: Vec<u8>,
    pub filename: Option<String>,
    pub mime_type: Option<String>,
}

impl ElevenLabsVoiceSampleFile {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            filename: None,
            mime_type: None,
        }
    }

    pub fn with_filename(mut self, value: impl Into<String>) -> Self {
        self.filename = Some(value.into());
        self
    }

    pub fn with_mime_type(mut self, value: impl Into<String>) -> Self {
        self.mime_type = Some(value.into());
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.bytes.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice sample file cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn build_part(&self) -> Result<reqwest::multipart::Part, LlmError> {
        let mut part = reqwest::multipart::Part::bytes(self.bytes.clone());
        if let Some(filename) = optional_trimmed(self.filename.as_deref()) {
            part = part.file_name(filename.to_string());
        }
        if let Some(mime_type) = optional_trimmed(self.mime_type.as_deref()) {
            part = part.mime_str(mime_type).map_err(|e| {
                LlmError::InvalidInput(format!(
                    "Invalid ElevenLabs voice sample file MIME type '{mime_type}': {e}"
                ))
            })?;
        }
        Ok(part)
    }
}

/// Request body for creating an instant voice clone.
#[derive(Debug, Clone)]
pub struct ElevenLabsCreateIvcVoiceRequest {
    pub name: String,
    pub files: Vec<ElevenLabsVoiceSampleFile>,
    pub remove_background_noise: Option<bool>,
    pub description: Option<String>,
    pub labels: HashMap<String, String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsCreateIvcVoiceRequest {
    pub fn new<I>(name: impl Into<String>, files: I) -> Self
    where
        I: IntoIterator<Item = ElevenLabsVoiceSampleFile>,
    {
        Self {
            name: name.into(),
            files: files.into_iter().collect(),
            remove_background_noise: None,
            description: None,
            labels: HashMap::new(),
            http_config: None,
        }
    }

    pub fn with_file(mut self, value: ElevenLabsVoiceSampleFile) -> Self {
        self.files.push(value);
        self
    }

    pub const fn with_remove_background_noise(mut self, value: bool) -> Self {
        self.remove_background_noise = Some(value);
        self
    }

    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        self.description = Some(value.into());
        self
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.name.trim().is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice name cannot be empty".to_string(),
            ));
        }
        if self.files.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs IVC voice creation requires at least one sample file".to_string(),
            ));
        }
        for file in &self.files {
            file.validate()?;
        }
        if self.labels.keys().any(|key| key.trim().is_empty()) {
            return Err(LlmError::InvalidInput(
                "ElevenLabs voice label keys cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn build_form(&self) -> Result<reqwest::multipart::Form, LlmError> {
        let mut form = reqwest::multipart::Form::new().text("name", self.name.trim().to_string());

        for file in &self.files {
            form = form.part("files", file.build_part()?);
        }
        if let Some(remove_background_noise) = self.remove_background_noise {
            form = form.text(
                "remove_background_noise",
                remove_background_noise.to_string(),
            );
        }
        if let Some(description) = optional_trimmed(self.description.as_deref()) {
            form = form.text("description", description.to_string());
        }
        if !self.labels.is_empty() {
            let labels = serde_json::to_string(&self.labels).map_err(|e| {
                LlmError::InvalidInput(format!("Invalid ElevenLabs voice labels: {e}"))
            })?;
            form = form.text("labels", labels);
        }

        Ok(form)
    }
}

/// Response body for `POST /v1/voices/add`.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsCreateIvcVoiceResponse {
    pub voice_id: String,
    pub requires_verification: bool,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Request body for creating a professional voice clone.
#[derive(Debug, Clone)]
pub struct ElevenLabsCreatePvcVoiceRequest {
    pub name: String,
    pub language: String,
    pub description: Option<String>,
    pub labels: HashMap<String, String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsCreatePvcVoiceRequest {
    pub fn new(name: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            language: language.into(),
            description: None,
            labels: HashMap::new(),
            http_config: None,
        }
    }

    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        self.description = Some(value.into());
        self
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.name.trim().is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs PVC voice name cannot be empty".to_string(),
            ));
        }
        if self.language.trim().is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs PVC voice language cannot be empty".to_string(),
            ));
        }
        validate_voice_labels(&self.labels)
    }

    fn body(&self) -> Result<Value, LlmError> {
        #[derive(Serialize)]
        struct Body<'a> {
            name: &'a str,
            language: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            description: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            labels: Option<&'a HashMap<String, String>>,
        }

        serde_json::to_value(Body {
            name: self.name.trim(),
            language: self.language.trim(),
            description: optional_trimmed(self.description.as_deref()),
            labels: (!self.labels.is_empty()).then_some(&self.labels),
        })
        .map_err(|e| {
            LlmError::InvalidInput(format!("Invalid ElevenLabs PVC voice create request: {e}"))
        })
    }
}

/// Request body for updating professional voice clone metadata.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsUpdatePvcVoiceRequest {
    pub name: Option<String>,
    pub language: Option<String>,
    pub description: Option<String>,
    pub labels: HashMap<String, String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsUpdatePvcVoiceRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(mut self, value: impl Into<String>) -> Self {
        self.name = Some(value.into());
        self
    }

    pub fn with_language(mut self, value: impl Into<String>) -> Self {
        self.language = Some(value.into());
        self
    }

    pub fn with_description(mut self, value: impl Into<String>) -> Self {
        self.description = Some(value.into());
        self
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self
            .name
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(LlmError::InvalidInput(
                "ElevenLabs PVC voice name cannot be empty".to_string(),
            ));
        }
        if self
            .language
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(LlmError::InvalidInput(
                "ElevenLabs PVC voice language cannot be empty".to_string(),
            ));
        }
        validate_voice_labels(&self.labels)
    }

    fn body(&self) -> Result<Value, LlmError> {
        #[derive(Serialize)]
        struct Body<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            language: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            description: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            labels: Option<&'a HashMap<String, String>>,
        }

        serde_json::to_value(Body {
            name: optional_trimmed(self.name.as_deref()),
            language: optional_trimmed(self.language.as_deref()),
            description: optional_trimmed(self.description.as_deref()),
            labels: (!self.labels.is_empty()).then_some(&self.labels),
        })
        .map_err(|e| {
            LlmError::InvalidInput(format!("Invalid ElevenLabs PVC voice update request: {e}"))
        })
    }
}

/// Request body for starting professional voice clone training.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsTrainPvcVoiceRequest {
    pub model_id: Option<String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsTrainPvcVoiceRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model_id(mut self, value: impl Into<String>) -> Self {
        self.model_id = Some(value.into());
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self
            .model_id
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(LlmError::InvalidInput(
                "ElevenLabs PVC training model_id cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn body(&self) -> Result<Value, LlmError> {
        #[derive(Serialize)]
        struct Body<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            model_id: Option<&'a str>,
        }

        serde_json::to_value(Body {
            model_id: optional_trimmed(self.model_id.as_deref()),
        })
        .map_err(|e| {
            LlmError::InvalidInput(format!("Invalid ElevenLabs PVC voice train request: {e}"))
        })
    }
}

/// Response body containing a professional voice clone ID.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPvcVoiceResponse {
    pub voice_id: String,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Request body for adding samples to a professional voice clone.
#[derive(Debug, Clone)]
pub struct ElevenLabsAddPvcVoiceSamplesRequest {
    pub files: Vec<ElevenLabsVoiceSampleFile>,
    pub remove_background_noise: Option<bool>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsAddPvcVoiceSamplesRequest {
    pub fn new<I>(files: I) -> Self
    where
        I: IntoIterator<Item = ElevenLabsVoiceSampleFile>,
    {
        Self {
            files: files.into_iter().collect(),
            remove_background_noise: None,
            http_config: None,
        }
    }

    pub fn with_file(mut self, value: ElevenLabsVoiceSampleFile) -> Self {
        self.files.push(value);
        self
    }

    pub const fn with_remove_background_noise(mut self, value: bool) -> Self {
        self.remove_background_noise = Some(value);
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.files.is_empty() {
            return Err(LlmError::InvalidInput(
                "ElevenLabs PVC sample upload requires at least one sample file".to_string(),
            ));
        }
        for file in &self.files {
            file.validate()?;
        }
        Ok(())
    }

    fn build_form(&self) -> Result<reqwest::multipart::Form, LlmError> {
        let mut form = reqwest::multipart::Form::new();
        for file in &self.files {
            form = form.part("files", file.build_part()?);
        }
        if let Some(remove_background_noise) = self.remove_background_noise {
            form = form.text(
                "remove_background_noise",
                remove_background_noise.to_string(),
            );
        }
        Ok(form)
    }
}

/// Request body for updating a professional voice clone sample.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsUpdatePvcVoiceSampleRequest {
    pub remove_background_noise: Option<bool>,
    pub selected_speaker_ids: Option<Vec<String>>,
    pub trim_start_time: Option<u64>,
    pub trim_end_time: Option<u64>,
    pub file_name: Option<String>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsUpdatePvcVoiceSampleRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_remove_background_noise(mut self, value: bool) -> Self {
        self.remove_background_noise = Some(value);
        self
    }

    pub fn with_selected_speaker_id(mut self, value: impl Into<String>) -> Self {
        self.selected_speaker_ids
            .get_or_insert_with(Vec::new)
            .push(value.into());
        self
    }

    pub fn with_selected_speaker_ids<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.selected_speaker_ids = Some(values.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_trim_start_time(mut self, value: u64) -> Self {
        self.trim_start_time = Some(value);
        self
    }

    pub const fn with_trim_end_time(mut self, value: u64) -> Self {
        self.trim_end_time = Some(value);
        self
    }

    pub fn with_file_name(mut self, value: impl Into<String>) -> Self {
        self.file_name = Some(value.into());
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }

    fn validate(&self) -> Result<(), LlmError> {
        if let Some(speaker_ids) = &self.selected_speaker_ids {
            if speaker_ids.iter().any(|value| value.trim().is_empty()) {
                return Err(LlmError::InvalidInput(
                    "ElevenLabs PVC selected speaker IDs cannot be empty".to_string(),
                ));
            }
        }
        if self
            .file_name
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(LlmError::InvalidInput(
                "ElevenLabs PVC sample file_name cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn body(&self) -> Result<Value, LlmError> {
        #[derive(Serialize)]
        struct Body<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            remove_background_noise: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            selected_speaker_ids: Option<&'a Vec<String>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            trim_start_time: Option<u64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            trim_end_time: Option<u64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            file_name: Option<&'a str>,
        }

        serde_json::to_value(Body {
            remove_background_noise: self.remove_background_noise,
            selected_speaker_ids: self.selected_speaker_ids.as_ref(),
            trim_start_time: self.trim_start_time,
            trim_end_time: self.trim_end_time,
            file_name: optional_trimmed(self.file_name.as_deref()),
        })
        .map_err(|e| {
            LlmError::InvalidInput(format!(
                "Invalid ElevenLabs PVC voice sample update request: {e}"
            ))
        })
    }
}

/// Query parameters for retrieving a professional voice clone sample audio preview.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsPvcVoiceSampleAudioQuery {
    pub remove_background_noise: Option<bool>,
    pub http_config: Option<HttpConfig>,
}

impl ElevenLabsPvcVoiceSampleAudioQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_remove_background_noise(mut self, value: bool) -> Self {
        self.remove_background_noise = Some(value);
        self
    }

    pub fn with_http_config(mut self, value: HttpConfig) -> Self {
        self.http_config = Some(value);
        self
    }
}

/// Sample metadata returned by ElevenLabs PVC sample endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPvcVoiceSample {
    #[serde(default)]
    pub sample_id: Option<String>,
    #[serde(default)]
    pub file_name: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub hash: Option<String>,
    #[serde(default)]
    pub duration_secs: Option<f64>,
    #[serde(default)]
    pub remove_background_noise: Option<bool>,
    #[serde(default)]
    pub has_isolated_audio: Option<bool>,
    #[serde(default)]
    pub has_isolated_audio_preview: Option<bool>,
    #[serde(default)]
    pub speaker_separation: Option<ElevenLabsPvcSpeakerSeparationResponse>,
    #[serde(default)]
    pub trim_start: Option<i64>,
    #[serde(default)]
    pub trim_end: Option<i64>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Speaker separation state returned by ElevenLabs PVC sample endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPvcSpeakerSeparationResponse {
    pub voice_id: String,
    pub sample_id: String,
    pub status: String,
    #[serde(default)]
    pub speakers: Option<HashMap<String, ElevenLabsPvcSpeakerResponse>>,
    #[serde(default)]
    pub selected_speaker_ids: Option<Vec<String>>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Speaker metadata returned by ElevenLabs PVC speaker separation endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPvcSpeakerResponse {
    pub speaker_id: String,
    pub duration_secs: f64,
    #[serde(default)]
    pub utterances: Option<Vec<ElevenLabsPvcUtterance>>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Utterance time range returned by ElevenLabs PVC speaker separation endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPvcUtterance {
    pub start: f64,
    pub end: f64,
}

/// Audio preview response returned by ElevenLabs PVC sample audio endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPvcVoiceSampleAudioResponse {
    pub audio_base_64: String,
    pub voice_id: String,
    pub sample_id: String,
    pub media_type: String,
    #[serde(default)]
    pub duration_secs: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Visual waveform response returned by ElevenLabs PVC sample waveform endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPvcVoiceSampleWaveformResponse {
    pub sample_id: String,
    pub visual_waveform: Vec<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Separated speaker audio response returned by ElevenLabs PVC sample endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsPvcSpeakerAudioResponse {
    pub audio_base_64: String,
    pub media_type: String,
    pub duration_secs: f64,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Status response body returned by ElevenLabs voice mutation endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVoiceStatusResponse {
    pub status: String,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

fn validate_voice_labels(labels: &HashMap<String, String>) -> Result<(), LlmError> {
    if labels.keys().any(|key| key.trim().is_empty()) {
        return Err(LlmError::InvalidInput(
            "ElevenLabs voice label keys cannot be empty".to_string(),
        ));
    }
    Ok(())
}

/// Verified language metadata returned by ElevenLabs voice APIs.
#[derive(Debug, Clone, Deserialize)]
pub struct ElevenLabsVerifiedLanguage {
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub accent: Option<String>,
    #[serde(default)]
    pub locale: Option<String>,
    #[serde(default)]
    pub preview_url: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportDeleteRequest, HttpTransportGetRequest,
        HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::Mutex;

    use super::super::resource_http::XI_API_KEY;

    #[derive(Clone)]
    struct JsonGetTransport {
        response: Value,
        last_json: Arc<Mutex<Option<HttpTransportRequest>>>,
        last_get: Arc<Mutex<Option<HttpTransportGetRequest>>>,
        last_delete: Arc<Mutex<Option<HttpTransportDeleteRequest>>>,
        last_multipart: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
    }

    impl JsonGetTransport {
        fn new(response: Value) -> Self {
            Self {
                response,
                last_json: Arc::new(Mutex::new(None)),
                last_get: Arc::new(Mutex::new(None)),
                last_delete: Arc::new(Mutex::new(None)),
                last_multipart: Arc::new(Mutex::new(None)),
            }
        }

        fn take_json(&self) -> HttpTransportRequest {
            self.last_json
                .lock()
                .expect("json transport lock")
                .take()
                .expect("captured json request")
        }

        fn take_get(&self) -> HttpTransportGetRequest {
            self.last_get
                .lock()
                .expect("get transport lock")
                .take()
                .expect("captured get request")
        }

        fn take_delete(&self) -> HttpTransportDeleteRequest {
            self.last_delete
                .lock()
                .expect("delete transport lock")
                .take()
                .expect("captured delete request")
        }

        fn take_multipart(&self) -> HttpTransportMultipartRequest {
            self.last_multipart
                .lock()
                .expect("multipart transport lock")
                .take()
                .expect("captured multipart request")
        }
    }

    #[async_trait]
    impl HttpTransport for JsonGetTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last_json.lock().expect("json transport lock") = Some(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&self.response).expect("serialize response"),
            })
        }

        async fn execute_get(
            &self,
            request: HttpTransportGetRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last_get.lock().expect("get transport lock") = Some(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&self.response).expect("serialize response"),
            })
        }

        async fn execute_delete(
            &self,
            request: HttpTransportDeleteRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last_delete.lock().expect("delete transport lock") = Some(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&self.response).expect("serialize response"),
            })
        }

        async fn execute_multipart(
            &self,
            request: HttpTransportMultipartRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self
                .last_multipart
                .lock()
                .expect("multipart transport lock") = Some(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&self.response).expect("serialize response"),
            })
        }
    }

    fn query_values(url: &str, key: &str) -> Vec<String> {
        let parsed = Url::parse(url).expect("valid url");
        parsed
            .query_pairs()
            .filter_map(|(name, value)| (name == key).then(|| value.into_owned()))
            .collect()
    }

    fn header_value<'a>(headers: &'a reqwest::header::HeaderMap, key: &str) -> Option<&'a str> {
        headers.get(key).and_then(|value| value.to_str().ok())
    }

    #[tokio::test]
    async fn voices_list_maps_query_headers_and_response_metadata() {
        let transport = JsonGetTransport::new(json!({
            "voices": [
                {
                    "voice_id": "voice-1",
                    "name": "Rachel",
                    "category": "premade",
                    "labels": { "accent": "american" },
                    "description": "Narration voice",
                    "preview_url": "https://cdn.elevenlabs.test/preview.mp3",
                    "settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.1,
                        "use_speaker_boost": true,
                        "speed": 1.0,
                        "future_setting": "kept"
                    },
                    "sharing": { "status": "public" },
                    "verified_languages": [
                        {
                            "language": "en",
                            "accent": "american",
                            "locale": "en-US",
                            "preview_url": "https://cdn.elevenlabs.test/en.mp3",
                            "future_language": "kept"
                        }
                    ],
                    "created_at_unix": 1_700_000_000,
                    "unknown_voice": "kept"
                }
            ],
            "has_more": true,
            "total_count": 1,
            "next_page_token": "next-token",
            "unknown_list": "kept"
        }));
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());
        request_http
            .headers
            .insert("x-shared".to_string(), "request-wins".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_header("x-global-header", "global")
            .with_header("x-shared", "global")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(
            config,
            reqwest::Client::new(),
            Some(RetryOptions::policy_default().with_max_attempts(1)),
        );

        let response = voices
            .list(Some(
                ElevenLabsVoiceListQuery::new()
                    .with_next_page_token("previous-token")
                    .with_page_size(25)
                    .with_search("rachel")
                    .with_sort("created_at_unix")
                    .with_sort_direction("desc")
                    .with_voice_type("default")
                    .with_category("premade")
                    .with_fine_tuning_state("fine_tuned")
                    .with_collection_id("collection-1")
                    .with_include_total_count(true)
                    .with_voice_ids(["voice-1", "voice-2"])
                    .with_http_config(request_http),
            ))
            .await
            .expect("voices response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url.split('?').next(),
            Some("https://api.elevenlabs.test/v2/voices")
        );
        assert_eq!(
            query_values(&captured.url, "next_page_token"),
            vec!["previous-token"]
        );
        assert_eq!(query_values(&captured.url, "page_size"), vec!["25"]);
        assert_eq!(query_values(&captured.url, "search"), vec!["rachel"]);
        assert_eq!(query_values(&captured.url, "sort"), vec!["created_at_unix"]);
        assert_eq!(query_values(&captured.url, "sort_direction"), vec!["desc"]);
        assert_eq!(query_values(&captured.url, "voice_type"), vec!["default"]);
        assert_eq!(query_values(&captured.url, "category"), vec!["premade"]);
        assert_eq!(
            query_values(&captured.url, "fine_tuning_state"),
            vec!["fine_tuned"]
        );
        assert_eq!(
            query_values(&captured.url, "collection_id"),
            vec!["collection-1"]
        );
        assert_eq!(
            query_values(&captured.url, "include_total_count"),
            vec!["true"]
        );
        assert_eq!(
            query_values(&captured.url, "voice_ids"),
            vec!["voice-1", "voice-2"]
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "x-global-header"),
            Some("global")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        assert_eq!(
            header_value(&captured.headers, "x-shared"),
            Some("request-wins")
        );

        assert_eq!(response.has_more, true);
        assert_eq!(response.total_count, Some(1));
        assert_eq!(response.next_page_token.as_deref(), Some("next-token"));
        assert_eq!(response.extra.get("unknown_list"), Some(&json!("kept")));
        let voice = response.voices.first().expect("voice");
        assert_eq!(voice.voice_id, "voice-1");
        assert_eq!(voice.name.as_deref(), Some("Rachel"));
        assert_eq!(voice.category.as_deref(), Some("premade"));
        assert_eq!(voice.labels.get("accent"), Some(&json!("american")));
        assert_eq!(voice.description.as_deref(), Some("Narration voice"));
        assert_eq!(
            voice.preview_url.as_deref(),
            Some("https://cdn.elevenlabs.test/preview.mp3")
        );
        assert_eq!(
            voice.sharing.as_ref().and_then(|value| value.get("status")),
            Some(&json!("public"))
        );
        let language = voice
            .verified_languages
            .as_ref()
            .and_then(|languages| languages.first())
            .expect("verified language");
        assert_eq!(language.language.as_deref(), Some("en"));
        assert_eq!(language.accent.as_deref(), Some("american"));
        assert_eq!(language.locale.as_deref(), Some("en-US"));
        assert_eq!(
            language.preview_url.as_deref(),
            Some("https://cdn.elevenlabs.test/en.mp3")
        );
        assert_eq!(language.extra.get("future_language"), Some(&json!("kept")));
        assert_eq!(voice.created_at_unix, Some(1_700_000_000));
        assert_eq!(voice.extra.get("unknown_voice"), Some(&json!("kept")));
        let settings = voice.settings.as_ref().expect("settings");
        assert_eq!(settings.stability, Some(0.5));
        assert_eq!(settings.similarity_boost, Some(0.75));
        assert_eq!(settings.style, Some(0.1));
        assert_eq!(settings.use_speaker_boost, Some(true));
        assert_eq!(settings.speed, Some(1.0));
        assert_eq!(settings.extra.get("future_setting"), Some(&json!("kept")));
    }

    #[tokio::test]
    async fn voices_get_encodes_voice_id_path() {
        let transport = JsonGetTransport::new(json!({
            "voice_id": "voice/id with space",
            "name": "Custom"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .get("voice/id with space")
            .await
            .expect("voice response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/voice%2Fid%20with%20space"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(response.voice_id, "voice/id with space");
        assert_eq!(response.name.as_deref(), Some("Custom"));
    }

    #[tokio::test]
    async fn voices_settings_get_default_and_voice_settings() {
        let transport = JsonGetTransport::new(json!({
            "stability": 0.33,
            "similarity_boost": 0.77,
            "style": 0.12,
            "use_speaker_boost": true,
            "speed": 1.05,
            "future_setting": "kept"
        }));
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let default_settings = voices
            .default_settings_with_http_config(Some(&request_http))
            .await
            .expect("default settings response");
        let captured = transport.take_get();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/settings/default"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        assert_eq!(default_settings.stability, Some(0.33));
        assert_eq!(default_settings.similarity_boost, Some(0.77));
        assert_eq!(default_settings.style, Some(0.12));
        assert_eq!(default_settings.use_speaker_boost, Some(true));
        assert_eq!(default_settings.speed, Some(1.05));
        assert_eq!(
            default_settings.extra.get("future_setting"),
            Some(&json!("kept"))
        );

        let voice_settings = voices
            .settings("voice/id with space")
            .await
            .expect("voice settings response");
        let captured = transport.take_get();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/voice%2Fid%20with%20space/settings"
        );
        assert_eq!(voice_settings.stability, Some(0.33));
    }

    #[tokio::test]
    async fn voices_settings_update_posts_json_and_maps_status() {
        let transport = JsonGetTransport::new(json!({
            "status": "ok",
            "future_status": "kept"
        }));
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .update_settings(
                "voice/id with space",
                ElevenLabsUpdateVoiceSettingsRequest::new()
                    .with_stability(0.45)
                    .with_similarity_boost(0.82)
                    .with_style(0.2)
                    .with_use_speaker_boost(false)
                    .with_speed(0.95)
                    .with_http_config(request_http),
            )
            .await
            .expect("update settings response");

        let captured = transport.take_json();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/voice%2Fid%20with%20space/settings/edit"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        assert_eq!(
            captured.body,
            json!({
                "stability": 0.45,
                "similarity_boost": 0.82,
                "style": 0.2,
                "use_speaker_boost": false,
                "speed": 0.95
            })
        );
        assert_eq!(response.status, "ok");
        assert_eq!(response.extra.get("future_status"), Some(&json!("kept")));
    }

    #[tokio::test]
    async fn voices_settings_update_rejects_empty_request() {
        let transport = JsonGetTransport::new(json!({ "status": "ok" }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_http_transport(Arc::new(transport));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let err = voices
            .update_settings("voice-1", ElevenLabsUpdateVoiceSettingsRequest::new())
            .await
            .expect_err("empty update should fail");

        assert!(
            err.to_string()
                .contains("ElevenLabs voice settings update request cannot be empty"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn voices_pvc_metadata_posts_json_and_maps_responses() {
        let transport = JsonGetTransport::new(json!({
            "voice_id": "pvc-voice-1",
            "future_pvc": "kept"
        }));
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .create_pvc_voice(
                ElevenLabsCreatePvcVoiceRequest::new("Professional Clone", "en")
                    .with_description("Narration PVC voice")
                    .with_label("accent", "american")
                    .with_http_config(request_http),
            )
            .await
            .expect("create PVC voice response");

        let captured = transport.take_json();
        assert_eq!(captured.url, "https://api.elevenlabs.test/v1/voices/pvc");
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        assert_eq!(
            captured.body,
            json!({
                "name": "Professional Clone",
                "language": "en",
                "description": "Narration PVC voice",
                "labels": { "accent": "american" }
            })
        );
        assert_eq!(response.voice_id, "pvc-voice-1");
        assert_eq!(response.extra.get("future_pvc"), Some(&json!("kept")));

        let response = voices
            .update_pvc_voice(
                "voice/id with space",
                ElevenLabsUpdatePvcVoiceRequest::new()
                    .with_name("Updated PVC")
                    .with_language("en-US")
                    .with_description("Updated description")
                    .with_label("age", "middle_aged"),
            )
            .await
            .expect("update PVC voice response");

        let captured = transport.take_json();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space"
        );
        assert_eq!(
            captured.body,
            json!({
                "name": "Updated PVC",
                "language": "en-US",
                "description": "Updated description",
                "labels": { "age": "middle_aged" }
            })
        );
        assert_eq!(response.voice_id, "pvc-voice-1");
    }

    #[tokio::test]
    async fn voices_pvc_metadata_train_posts_optional_model_id() {
        let transport = JsonGetTransport::new(json!({
            "status": "ok",
            "future_status": "kept"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .train_pvc_voice(
                "voice/id with space",
                ElevenLabsTrainPvcVoiceRequest::new().with_model_id("eleven_multilingual_v2"),
            )
            .await
            .expect("train PVC voice response");

        let captured = transport.take_json();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space/train"
        );
        assert_eq!(
            captured.body,
            json!({
                "model_id": "eleven_multilingual_v2"
            })
        );
        assert_eq!(response.status, "ok");
        assert_eq!(response.extra.get("future_status"), Some(&json!("kept")));
    }

    #[tokio::test]
    async fn voices_pvc_metadata_create_rejects_missing_required_metadata() {
        let transport = JsonGetTransport::new(json!({ "voice_id": "pvc-voice-1" }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_http_transport(Arc::new(transport));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let err = voices
            .create_pvc_voice(ElevenLabsCreatePvcVoiceRequest::new(" ", "en"))
            .await
            .expect_err("empty name should fail");

        assert!(
            err.to_string()
                .contains("ElevenLabs PVC voice name cannot be empty"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn voices_pvc_samples_add_posts_multipart_and_maps_samples() {
        let transport = JsonGetTransport::new(json!([
            {
                "sample_id": "sample-1",
                "file_name": "sample-one.wav",
                "mime_type": "audio/wav",
                "size_bytes": 1234,
                "hash": "hash-1",
                "duration_secs": 12.5,
                "remove_background_noise": true,
                "has_isolated_audio": true,
                "has_isolated_audio_preview": false,
                "trim_start": 10,
                "trim_end": 12000,
                "speaker_separation": {
                    "voice_id": "pvc-voice-1",
                    "sample_id": "sample-1",
                    "status": "completed",
                    "selected_speaker_ids": ["speaker-1"],
                    "speakers": {
                        "speaker-1": {
                            "speaker_id": "speaker-1",
                            "duration_secs": 11.2,
                            "utterances": [
                                { "start": 0.0, "end": 1.25 }
                            ],
                            "future_speaker": "kept"
                        }
                    },
                    "future_separation": "kept"
                },
                "future_sample": "kept"
            }
        ]));
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .add_pvc_voice_samples(
                "voice/id with space",
                ElevenLabsAddPvcVoiceSamplesRequest::new([ElevenLabsVoiceSampleFile::new(
                    b"audio-one".to_vec(),
                )
                .with_filename("sample-one.wav")
                .with_mime_type("audio/wav")])
                .with_remove_background_noise(true)
                .with_http_config(request_http),
            )
            .await
            .expect("add PVC samples response");

        let captured = transport.take_multipart();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space/samples"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        let body = String::from_utf8_lossy(&captured.body);
        assert!(body.contains("name=\"files\"; filename=\"sample-one.wav\""));
        assert!(body.contains("Content-Type: audio/wav"));
        assert!(body.contains("audio-one"));
        assert!(body.contains("name=\"remove_background_noise\""));
        assert!(body.contains("true"));

        let sample = response.first().expect("sample");
        assert_eq!(sample.sample_id.as_deref(), Some("sample-1"));
        assert_eq!(sample.file_name.as_deref(), Some("sample-one.wav"));
        assert_eq!(sample.mime_type.as_deref(), Some("audio/wav"));
        assert_eq!(sample.size_bytes, Some(1234));
        assert_eq!(sample.hash.as_deref(), Some("hash-1"));
        assert_eq!(sample.duration_secs, Some(12.5));
        assert_eq!(sample.remove_background_noise, Some(true));
        assert_eq!(sample.has_isolated_audio, Some(true));
        assert_eq!(sample.has_isolated_audio_preview, Some(false));
        assert_eq!(sample.trim_start, Some(10));
        assert_eq!(sample.trim_end, Some(12000));
        assert_eq!(sample.extra.get("future_sample"), Some(&json!("kept")));

        let separation = sample
            .speaker_separation
            .as_ref()
            .expect("speaker separation");
        assert_eq!(separation.voice_id, "pvc-voice-1");
        assert_eq!(separation.sample_id, "sample-1");
        assert_eq!(separation.status, "completed");
        assert_eq!(
            separation
                .selected_speaker_ids
                .as_ref()
                .expect("selected speakers"),
            &vec!["speaker-1".to_string()]
        );
        assert_eq!(
            separation.extra.get("future_separation"),
            Some(&json!("kept"))
        );
        let speaker = separation
            .speakers
            .as_ref()
            .and_then(|speakers| speakers.get("speaker-1"))
            .expect("speaker");
        assert_eq!(speaker.speaker_id, "speaker-1");
        assert_eq!(speaker.duration_secs, 11.2);
        assert_eq!(
            speaker
                .utterances
                .as_ref()
                .and_then(|utterances| utterances.first())
                .map(|utterance| (utterance.start, utterance.end)),
            Some((0.0, 1.25))
        );
        assert_eq!(speaker.extra.get("future_speaker"), Some(&json!("kept")));
    }

    #[tokio::test]
    async fn voices_pvc_samples_update_and_delete_use_json_and_delete() {
        let transport = JsonGetTransport::new(json!({
            "voice_id": "pvc-voice-1",
            "future_pvc": "kept"
        }));
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .update_pvc_voice_sample(
                "voice/id with space",
                "sample/id with space",
                ElevenLabsUpdatePvcVoiceSampleRequest::new()
                    .with_remove_background_noise(true)
                    .with_selected_speaker_ids(["speaker-1", "speaker-2"])
                    .with_trim_start_time(250)
                    .with_trim_end_time(12_500)
                    .with_file_name("clean-sample.wav")
                    .with_http_config(request_http),
            )
            .await
            .expect("update PVC sample response");

        let captured = transport.take_json();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space/samples/sample%2Fid%20with%20space"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        assert_eq!(
            captured.body,
            json!({
                "remove_background_noise": true,
                "selected_speaker_ids": ["speaker-1", "speaker-2"],
                "trim_start_time": 250,
                "trim_end_time": 12500,
                "file_name": "clean-sample.wav"
            })
        );
        assert_eq!(response.voice_id, "pvc-voice-1");
        assert_eq!(response.extra.get("future_pvc"), Some(&json!("kept")));

        let transport = JsonGetTransport::new(json!({
            "status": "ok",
            "future_status": "kept"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .delete_pvc_voice_sample("voice/id with space", "sample/id with space")
            .await
            .expect("delete PVC sample response");

        let captured = transport.take_delete();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space/samples/sample%2Fid%20with%20space"
        );
        assert_eq!(response.status, "ok");
        assert_eq!(response.extra.get("future_status"), Some(&json!("kept")));
    }

    #[tokio::test]
    async fn voices_pvc_samples_get_audio_waveform_and_speakers() {
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());

        let transport = JsonGetTransport::new(json!({
            "audio_base_64": "YXVkaW8=",
            "voice_id": "pvc-voice-1",
            "sample_id": "sample-1",
            "media_type": "audio/mpeg",
            "duration_secs": 3.5,
            "future_audio": "kept"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .get_pvc_voice_sample_audio(
                "voice/id with space",
                "sample/id with space",
                Some(
                    ElevenLabsPvcVoiceSampleAudioQuery::new()
                        .with_remove_background_noise(true)
                        .with_http_config(request_http),
                ),
            )
            .await
            .expect("PVC sample audio response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url.split('?').next(),
            Some(
                "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space/samples/sample%2Fid%20with%20space/audio"
            )
        );
        assert_eq!(
            query_values(&captured.url, "remove_background_noise"),
            vec!["true"]
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        assert_eq!(response.audio_base_64, "YXVkaW8=");
        assert_eq!(response.voice_id, "pvc-voice-1");
        assert_eq!(response.sample_id, "sample-1");
        assert_eq!(response.media_type, "audio/mpeg");
        assert_eq!(response.duration_secs, Some(3.5));
        assert_eq!(response.extra.get("future_audio"), Some(&json!("kept")));

        let transport = JsonGetTransport::new(json!({
            "sample_id": "sample-1",
            "visual_waveform": [0.0, 0.5, 1.0],
            "future_waveform": "kept"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .get_pvc_voice_sample_waveform("voice/id with space", "sample/id with space")
            .await
            .expect("PVC waveform response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space/samples/sample%2Fid%20with%20space/waveform"
        );
        assert_eq!(response.sample_id, "sample-1");
        assert_eq!(response.visual_waveform, vec![0.0, 0.5, 1.0]);
        assert_eq!(response.extra.get("future_waveform"), Some(&json!("kept")));

        let transport = JsonGetTransport::new(json!({
            "voice_id": "pvc-voice-1",
            "sample_id": "sample-1",
            "status": "pending",
            "speakers": null,
            "selected_speaker_ids": null,
            "future_separation": "kept"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .get_pvc_voice_sample_speakers("voice/id with space", "sample/id with space")
            .await
            .expect("PVC speakers response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space/samples/sample%2Fid%20with%20space/speakers"
        );
        assert_eq!(response.voice_id, "pvc-voice-1");
        assert_eq!(response.sample_id, "sample-1");
        assert_eq!(response.status, "pending");
        assert!(response.speakers.is_none());
        assert_eq!(
            response.extra.get("future_separation"),
            Some(&json!("kept"))
        );

        let transport = JsonGetTransport::new(json!({
            "status": "ok",
            "future_status": "kept"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .start_pvc_voice_sample_speaker_separation(
                "voice/id with space",
                "sample/id with space",
            )
            .await
            .expect("start PVC speaker separation response");

        let captured = transport.take_json();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space/samples/sample%2Fid%20with%20space/separate-speakers"
        );
        assert_eq!(captured.body, json!({}));
        assert_eq!(response.status, "ok");

        let transport = JsonGetTransport::new(json!({
            "audio_base_64": "c3BlYWtlci1hdWRpbw==",
            "media_type": "audio/wav",
            "duration_secs": 2.25,
            "future_speaker_audio": "kept"
        }));
        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .get_pvc_voice_sample_speaker_audio(
                "voice/id with space",
                "sample/id with space",
                "speaker/id with space",
            )
            .await
            .expect("PVC separated speaker audio response");

        let captured = transport.take_get();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/pvc/voice%2Fid%20with%20space/samples/sample%2Fid%20with%20space/speakers/speaker%2Fid%20with%20space/audio"
        );
        assert_eq!(response.audio_base_64, "c3BlYWtlci1hdWRpbw==");
        assert_eq!(response.media_type, "audio/wav");
        assert_eq!(response.duration_secs, 2.25);
        assert_eq!(
            response.extra.get("future_speaker_audio"),
            Some(&json!("kept"))
        );
    }

    #[tokio::test]
    async fn voices_ivc_create_posts_multipart_and_maps_response() {
        let transport = JsonGetTransport::new(json!({
            "voice_id": "voice-clone-1",
            "requires_verification": true,
            "future_create": "kept"
        }));
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());
        request_http
            .headers
            .insert("x-shared".to_string(), "request-wins".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_header("x-global-header", "global")
            .with_header("x-shared", "global")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .create_ivc_voice(
                ElevenLabsCreateIvcVoiceRequest::new(
                    "Narrator Clone",
                    [ElevenLabsVoiceSampleFile::new(b"audio-one".to_vec())
                        .with_filename("sample-one.wav")
                        .with_mime_type("audio/wav")],
                )
                .with_file(
                    ElevenLabsVoiceSampleFile::new(b"audio-two".to_vec())
                        .with_filename("sample-two.mp3")
                        .with_mime_type("audio/mpeg"),
                )
                .with_remove_background_noise(true)
                .with_description("Narration voice")
                .with_label("accent", "american")
                .with_http_config(request_http),
            )
            .await
            .expect("create IVC voice response");

        let captured = transport.take_multipart();
        assert_eq!(captured.url, "https://api.elevenlabs.test/v1/voices/add");
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert!(
            header_value(&captured.headers, "content-type")
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );
        assert!(header_value(&captured.headers, "content-length").is_some());
        assert_eq!(
            header_value(&captured.headers, "x-global-header"),
            Some("global")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        assert_eq!(
            header_value(&captured.headers, "x-shared"),
            Some("request-wins")
        );

        let body = String::from_utf8_lossy(&captured.body);
        assert!(body.contains("name=\"name\""));
        assert!(body.contains("Narrator Clone"));
        assert!(body.contains("name=\"files\"; filename=\"sample-one.wav\""));
        assert!(body.contains("Content-Type: audio/wav"));
        assert!(body.contains("audio-one"));
        assert!(body.contains("name=\"files\"; filename=\"sample-two.mp3\""));
        assert!(body.contains("Content-Type: audio/mpeg"));
        assert!(body.contains("audio-two"));
        assert!(body.contains("name=\"remove_background_noise\""));
        assert!(body.contains("true"));
        assert!(body.contains("name=\"description\""));
        assert!(body.contains("Narration voice"));
        assert!(body.contains("name=\"labels\""));
        assert!(body.contains("\"accent\":\"american\""));

        assert_eq!(response.voice_id, "voice-clone-1");
        assert_eq!(response.requires_verification, true);
        assert_eq!(response.extra.get("future_create"), Some(&json!("kept")));
    }

    #[tokio::test]
    async fn voices_delete_voice_and_sample_delete_use_delete_json() {
        let transport = JsonGetTransport::new(json!({
            "status": "ok",
            "future_status": "kept"
        }));
        let mut request_http = HttpConfig::empty();
        request_http
            .headers
            .insert("x-request-header".to_string(), "request".to_string());

        let config = ElevenLabsConfig::new("test-key")
            .with_base_url("https://api.elevenlabs.test/")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        let response = voices
            .delete_voice_with_http_config("voice/id with space", Some(&request_http))
            .await
            .expect("delete voice response");
        let captured = transport.take_delete();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/voice%2Fid%20with%20space"
        );
        assert_eq!(
            header_value(&captured.headers, XI_API_KEY),
            Some("test-key")
        );
        assert_eq!(
            header_value(&captured.headers, "x-request-header"),
            Some("request")
        );
        assert_eq!(response.status, "ok");
        assert_eq!(response.extra.get("future_status"), Some(&json!("kept")));

        let response = voices
            .delete_sample("voice/id with space", "sample/id with space")
            .await
            .expect("delete sample response");
        let captured = transport.take_delete();
        assert_eq!(
            captured.url,
            "https://api.elevenlabs.test/v1/voices/voice%2Fid%20with%20space/samples/sample%2Fid%20with%20space"
        );
        assert_eq!(response.status, "ok");
    }

    #[tokio::test]
    async fn voices_resources_use_explicit_api_key_override() {
        assert_eq!(ElevenLabsConfig::API_KEY_ENV, "ELEVENLABS_API_KEY");

        let transport = JsonGetTransport::new(json!({
            "voices": [],
            "has_more": false,
            "total_count": 0
        }));
        let config = ElevenLabsConfig::new("initial-key")
            .with_api_key("explicit-key")
            .with_base_url("https://api.elevenlabs.test")
            .with_http_transport(Arc::new(transport.clone()));
        let voices = ElevenLabsVoices::new(config, reqwest::Client::new(), None);

        voices
            .list(None)
            .await
            .expect("explicit key voices response");

        assert_eq!(
            header_value(&transport.take_get().headers, XI_API_KEY),
            Some("explicit-key")
        );
    }
}
