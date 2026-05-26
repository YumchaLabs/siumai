//! ElevenLabs typed provider options.

use crate::error::LlmError;
use crate::types::CustomProviderOptions;
use serde::{Deserialize, Serialize};

/// AI SDK-style text normalization policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApplyTextNormalization {
    Auto,
    On,
    Off,
}

impl ApplyTextNormalization {
    pub const fn as_wire_value(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

/// AI SDK-style typed voice settings for ElevenLabs speech models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ElevenLabsVoiceSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_boost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
}

impl ElevenLabsVoiceSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_stability(mut self, stability: f64) -> Self {
        self.stability = Some(stability);
        self
    }

    pub const fn with_similarity_boost(mut self, similarity_boost: f64) -> Self {
        self.similarity_boost = Some(similarity_boost);
        self
    }

    pub const fn with_style(mut self, style: f64) -> Self {
        self.style = Some(style);
        self
    }

    pub const fn with_use_speaker_boost(mut self, use_speaker_boost: bool) -> Self {
        self.use_speaker_boost = Some(use_speaker_boost);
        self
    }
}

/// Pronunciation dictionary locator for ElevenLabs speech requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElevenLabsPronunciationDictionaryLocator {
    pub pronunciation_dictionary_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version_id: Option<String>,
}

impl ElevenLabsPronunciationDictionaryLocator {
    pub fn new(pronunciation_dictionary_id: impl Into<String>) -> Self {
        Self {
            pronunciation_dictionary_id: pronunciation_dictionary_id.into(),
            version_id: None,
        }
    }

    pub fn with_version_id(mut self, version_id: impl Into<String>) -> Self {
        self.version_id = Some(version_id.into());
        self
    }
}

/// AI SDK-style typed options for ElevenLabs speech models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ElevenLabsSpeechModelOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_settings: Option<ElevenLabsVoiceSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pronunciation_dictionary_locators: Option<Vec<ElevenLabsPronunciationDictionaryLocator>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_request_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_request_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apply_text_normalization: Option<ApplyTextNormalization>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apply_language_text_normalization: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_logging: Option<bool>,
}

impl ElevenLabsSpeechModelOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_language_code(mut self, language_code: impl Into<String>) -> Self {
        self.language_code = Some(language_code.into());
        self
    }

    pub fn with_voice_settings(mut self, voice_settings: ElevenLabsVoiceSettings) -> Self {
        self.voice_settings = Some(voice_settings);
        self
    }

    pub fn with_pronunciation_dictionary_locators(
        mut self,
        locators: Vec<ElevenLabsPronunciationDictionaryLocator>,
    ) -> Self {
        self.pronunciation_dictionary_locators = Some(locators);
        self
    }

    pub const fn with_seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_previous_text(mut self, previous_text: impl Into<String>) -> Self {
        self.previous_text = Some(previous_text.into());
        self
    }

    pub fn with_next_text(mut self, next_text: impl Into<String>) -> Self {
        self.next_text = Some(next_text.into());
        self
    }

    pub fn with_previous_request_ids(mut self, ids: Vec<String>) -> Self {
        self.previous_request_ids = Some(ids);
        self
    }

    pub fn with_next_request_ids(mut self, ids: Vec<String>) -> Self {
        self.next_request_ids = Some(ids);
        self
    }

    pub fn with_apply_text_normalization(mut self, policy: ApplyTextNormalization) -> Self {
        self.apply_text_normalization = Some(policy);
        self
    }

    pub const fn with_apply_language_text_normalization(mut self, enabled: bool) -> Self {
        self.apply_language_text_normalization = Some(enabled);
        self
    }

    pub const fn with_enable_logging(mut self, enabled: bool) -> Self {
        self.enable_logging = Some(enabled);
        self
    }

    pub fn into_provider_options_map_entry(self) -> Result<(String, serde_json::Value), LlmError> {
        self.to_provider_options_map_entry()
    }
}

impl CustomProviderOptions for ElevenLabsSpeechModelOptions {
    fn provider_id(&self) -> &str {
        "elevenlabs"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        serde_json::to_value(self).map_err(|e| {
            LlmError::JsonError(format!(
                "Failed to serialize ElevenLabs speech options: {e}"
            ))
        })
    }
}

/// AI SDK-style timestamp granularity for ElevenLabs transcription.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ElevenLabsTranscriptionTimestampsGranularity {
    None,
    Word,
    Character,
}

impl ElevenLabsTranscriptionTimestampsGranularity {
    pub const fn as_wire_value(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Word => "word",
            Self::Character => "character",
        }
    }
}

/// AI SDK-style input file format for ElevenLabs transcription.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElevenLabsTranscriptionFileFormat {
    #[serde(rename = "pcm_s16le_16")]
    PcmS16le16,
    #[serde(rename = "other")]
    Other,
}

impl ElevenLabsTranscriptionFileFormat {
    pub const fn as_wire_value(&self) -> &'static str {
        match self {
            Self::PcmS16le16 => "pcm_s16le_16",
            Self::Other => "other",
        }
    }
}

/// AI SDK-style typed options for ElevenLabs transcription models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ElevenLabsTranscriptionModelOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag_audio_events: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_speakers: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamps_granularity: Option<ElevenLabsTranscriptionTimestampsGranularity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_format: Option<ElevenLabsTranscriptionFileFormat>,
}

impl ElevenLabsTranscriptionModelOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_language_code(mut self, language_code: impl Into<String>) -> Self {
        self.language_code = Some(language_code.into());
        self
    }

    pub const fn with_tag_audio_events(mut self, enabled: bool) -> Self {
        self.tag_audio_events = Some(enabled);
        self
    }

    pub const fn with_num_speakers(mut self, num_speakers: u8) -> Self {
        self.num_speakers = Some(num_speakers);
        self
    }

    pub fn with_timestamps_granularity(
        mut self,
        granularity: ElevenLabsTranscriptionTimestampsGranularity,
    ) -> Self {
        self.timestamps_granularity = Some(granularity);
        self
    }

    pub const fn with_diarize(mut self, diarize: bool) -> Self {
        self.diarize = Some(diarize);
        self
    }

    pub fn with_file_format(mut self, file_format: ElevenLabsTranscriptionFileFormat) -> Self {
        self.file_format = Some(file_format);
        self
    }

    pub fn into_provider_options_map_entry(self) -> Result<(String, serde_json::Value), LlmError> {
        self.to_provider_options_map_entry()
    }
}

impl CustomProviderOptions for ElevenLabsTranscriptionModelOptions {
    fn provider_id(&self) -> &str {
        "elevenlabs"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        serde_json::to_value(self).map_err(|e| {
            LlmError::JsonError(format!(
                "Failed to serialize ElevenLabs transcription options: {e}"
            ))
        })
    }
}

pub type ElevenLabsSpeechOptions = ElevenLabsSpeechModelOptions;
pub type ElevenLabsSttOptions = ElevenLabsTranscriptionModelOptions;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CustomProviderOptions;

    #[test]
    fn elevenlabs_speech_options_serialize_ai_sdk_style_keys() {
        let value = ElevenLabsSpeechModelOptions::new()
            .with_language_code("en")
            .with_voice_settings(
                ElevenLabsVoiceSettings::new()
                    .with_stability(0.5)
                    .with_similarity_boost(0.75)
                    .with_style(0.2)
                    .with_use_speaker_boost(true),
            )
            .with_pronunciation_dictionary_locators(vec![
                ElevenLabsPronunciationDictionaryLocator::new("dict-1").with_version_id("v2"),
            ])
            .with_seed(42)
            .with_apply_text_normalization(ApplyTextNormalization::Auto)
            .with_enable_logging(false)
            .to_json()
            .expect("json");

        assert_eq!(
            value,
            serde_json::json!({
                "languageCode": "en",
                "voiceSettings": {
                    "stability": 0.5,
                    "similarityBoost": 0.75,
                    "style": 0.2,
                    "useSpeakerBoost": true
                },
                "pronunciationDictionaryLocators": [
                    {
                        "pronunciationDictionaryId": "dict-1",
                        "versionId": "v2"
                    }
                ],
                "seed": 42,
                "applyTextNormalization": "auto",
                "enableLogging": false
            })
        );
    }

    #[test]
    fn elevenlabs_transcription_options_serialize_ai_sdk_style_keys() {
        let value = ElevenLabsTranscriptionModelOptions::new()
            .with_language_code("en")
            .with_tag_audio_events(true)
            .with_num_speakers(2)
            .with_timestamps_granularity(ElevenLabsTranscriptionTimestampsGranularity::Character)
            .with_diarize(false)
            .with_file_format(ElevenLabsTranscriptionFileFormat::PcmS16le16)
            .to_json()
            .expect("json");

        assert_eq!(
            value,
            serde_json::json!({
                "languageCode": "en",
                "tagAudioEvents": true,
                "numSpeakers": 2,
                "timestampsGranularity": "character",
                "diarize": false,
                "fileFormat": "pcm_s16le_16"
            })
        );
    }
}
