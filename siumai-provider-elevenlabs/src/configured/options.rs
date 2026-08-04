use serde::{Deserialize, Serialize};
use siumai_core::{ProviderOptionError, TypedProviderOptions};

const MAX_PRONUNCIATION_DICTIONARIES: usize = 3;

/// ElevenLabs text-normalization policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApplyTextNormalization {
    Auto,
    On,
    Off,
}

/// Per-request voice controls that do not overlap canonical speech fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElevenLabsVoiceSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    stability: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    similarity_boost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    use_speaker_boost: Option<bool>,
}

impl ElevenLabsVoiceSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_stability(mut self, value: f64) -> Self {
        self.stability = Some(value);
        self
    }

    pub fn with_similarity_boost(mut self, value: f64) -> Self {
        self.similarity_boost = Some(value);
        self
    }

    pub fn with_style(mut self, value: f64) -> Self {
        self.style = Some(value);
        self
    }

    pub fn with_speaker_boost(mut self, enabled: bool) -> Self {
        self.use_speaker_boost = Some(enabled);
        self
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        validate_unit_interval("voice_settings.stability", self.stability)?;
        validate_unit_interval("voice_settings.similarity_boost", self.similarity_boost)?;
        validate_unit_interval("voice_settings.style", self.style)
    }
}

/// A versioned pronunciation dictionary applied to one generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElevenLabsPronunciationDictionaryLocator {
    pronunciation_dictionary_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    version_id: Option<String>,
}

impl ElevenLabsPronunciationDictionaryLocator {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            pronunciation_dictionary_id: id.into(),
            version_id: None,
        }
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version_id = Some(version.into());
        self
    }

    fn validate(&self, index: usize) -> Result<(), ProviderOptionError> {
        validate_non_empty(
            &format!("pronunciation_dictionary_locators[{index}].pronunciation_dictionary_id"),
            &self.pronunciation_dictionary_id,
        )?;
        if let Some(version) = &self.version_id {
            validate_non_empty(
                &format!("pronunciation_dictionary_locators[{index}].version_id"),
                version,
            )?;
        }
        Ok(())
    }
}

/// Typed ElevenLabs options carried through [`siumai_core::CallOptions`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElevenLabsSpeechOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_settings: Option<ElevenLabsVoiceSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pronunciation_dictionary_locators: Option<Vec<ElevenLabsPronunciationDictionaryLocator>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_request_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_request_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    apply_text_normalization: Option<ApplyTextNormalization>,
    #[serde(skip_serializing_if = "Option::is_none")]
    apply_language_text_normalization: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_logging: Option<bool>,
}

impl ElevenLabsSpeechOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_voice_settings(mut self, settings: ElevenLabsVoiceSettings) -> Self {
        self.voice_settings = Some(settings);
        self
    }

    pub fn with_pronunciation_dictionaries(
        mut self,
        locators: Vec<ElevenLabsPronunciationDictionaryLocator>,
    ) -> Self {
        self.pronunciation_dictionary_locators = Some(locators);
        self
    }

    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_previous_text(mut self, text: impl Into<String>) -> Self {
        self.previous_text = Some(text.into());
        self
    }

    pub fn with_next_text(mut self, text: impl Into<String>) -> Self {
        self.next_text = Some(text.into());
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

    pub fn with_text_normalization(mut self, policy: ApplyTextNormalization) -> Self {
        self.apply_text_normalization = Some(policy);
        self
    }

    pub fn with_language_text_normalization(mut self, enabled: bool) -> Self {
        self.apply_language_text_normalization = Some(enabled);
        self
    }

    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.enable_logging = Some(enabled);
        self
    }

    pub(crate) fn merge_from(&mut self, later: Self) {
        macro_rules! replace_some {
            ($($field:ident),+ $(,)?) => {
                $(if later.$field.is_some() {
                    self.$field = later.$field;
                })+
            };
        }
        replace_some!(
            voice_settings,
            pronunciation_dictionary_locators,
            seed,
            previous_text,
            next_text,
            previous_request_ids,
            next_request_ids,
            apply_text_normalization,
            apply_language_text_normalization,
            enable_logging,
        );
    }

    pub(crate) fn voice_settings(&self) -> Option<&ElevenLabsVoiceSettings> {
        self.voice_settings.as_ref()
    }

    pub(crate) fn enable_logging(&self) -> Option<bool> {
        self.enable_logging
    }

    pub(crate) fn body_fields(
        &self,
    ) -> Result<serde_json::Map<String, serde_json::Value>, serde_json::Error> {
        let mut fields = serde_json::to_value(self)?
            .as_object()
            .cloned()
            .unwrap_or_default();
        fields.remove("enable_logging");
        fields.remove("voice_settings");
        Ok(fields)
    }
}

impl TypedProviderOptions for ElevenLabsSpeechOptions {
    const NAMESPACE: &'static str = "elevenlabs";

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(settings) = &self.voice_settings {
            settings.validate()?;
        }
        if let Some(locators) = &self.pronunciation_dictionary_locators {
            if locators.len() > MAX_PRONUNCIATION_DICTIONARIES {
                return Err(rejected(
                    "pronunciation_dictionary_locators",
                    "at most three pronunciation dictionaries are allowed",
                ));
            }
            for (index, locator) in locators.iter().enumerate() {
                locator.validate(index)?;
            }
        }
        for (path, value) in [
            ("previous_text", self.previous_text.as_deref()),
            ("next_text", self.next_text.as_deref()),
        ] {
            if let Some(value) = value {
                validate_non_empty(path, value)?;
            }
        }
        for (path, values) in [
            ("previous_request_ids", self.previous_request_ids.as_deref()),
            ("next_request_ids", self.next_request_ids.as_deref()),
        ] {
            if let Some(values) = values
                && (values.is_empty() || values.iter().any(|value| value.trim().is_empty()))
            {
                return Err(rejected(
                    path,
                    "request ID lists must contain non-empty IDs",
                ));
            }
        }
        Ok(())
    }
}

fn validate_unit_interval(path: &str, value: Option<f64>) -> Result<(), ProviderOptionError> {
    if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(rejected(
            path,
            "value must be finite and between zero and one",
        ));
    }
    Ok(())
}

fn validate_non_empty(path: &str, value: &str) -> Result<(), ProviderOptionError> {
    if value.trim().is_empty() {
        return Err(rejected(path, "value must not be empty"));
    }
    Ok(())
}

fn rejected(path: impl Into<String>, reason: impl Into<String>) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: path.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::ProviderOptions;

    #[test]
    fn typed_options_reject_invalid_voice_settings_and_dictionary_cardinality() {
        let invalid_voice = ElevenLabsSpeechOptions::new()
            .with_voice_settings(ElevenLabsVoiceSettings::new().with_stability(1.5));
        assert!(ProviderOptions::typed(&invalid_voice).is_err());

        let locators = (0..4)
            .map(|index| ElevenLabsPronunciationDictionaryLocator::new(format!("dict-{index}")))
            .collect();
        let invalid_dictionaries =
            ElevenLabsSpeechOptions::new().with_pronunciation_dictionaries(locators);
        assert!(ProviderOptions::typed(&invalid_dictionaries).is_err());
    }
}
