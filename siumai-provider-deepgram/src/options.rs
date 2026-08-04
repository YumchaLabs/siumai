use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use siumai_core::{ProviderOptionError, TypedProviderOptions};

/// Deepgram's supported summarization values for prerecorded audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeepgramSummarizeOption {
    Disabled,
    Version2,
}

/// Current diarization model choices accepted by prerecorded transcription.
///
/// Deepgram deprecated the legacy `diarize=true` switch in favor of the
/// explicit `diarize_model` parameter. Keeping this typed prevents the
/// configured provider from silently depending on a retiring default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum DeepgramDiarizeModel {
    Latest,
    V1,
}

impl Serialize for DeepgramSummarizeOption {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Disabled => serializer.serialize_bool(false),
            Self::Version2 => serializer.serialize_str("v2"),
        }
    }
}

impl<'de> Deserialize<'de> for DeepgramSummarizeOption {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::Bool(false) => Ok(Self::Disabled),
            serde_json::Value::String(value) if value == "v2" => Ok(Self::Version2),
            _ => Err(serde::de::Error::custom(
                "summarize must be false or the string `v2`",
            )),
        }
    }
}

/// One or more Deepgram redaction categories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeepgramRedaction {
    Single(String),
    Multiple(Vec<String>),
}

/// Typed Deepgram query options owned by the provider crate.
///
/// Language is intentionally part of [`siumai_core::TranscriptionRequest`], so
/// it cannot be specified a second time through provider options.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct DeepgramTranscriptionOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detect_language: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) smart_format: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) punctuate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) paragraphs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) summarize: Option<DeepgramSummarizeOption>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) topics: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) intents: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sentiment: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detect_entities: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) redact: Option<DeepgramRedaction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) replace: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) keyterm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) diarize_model: Option<DeepgramDiarizeModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) utterances: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) utt_split: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) filler_words: Option<bool>,
}

impl fmt::Debug for DeepgramTranscriptionOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let redaction_count = self.redact.as_ref().map(|redaction| match redaction {
            DeepgramRedaction::Single(_) => 1,
            DeepgramRedaction::Multiple(values) => values.len(),
        });
        formatter
            .debug_struct("DeepgramTranscriptionOptions")
            .field("detect_language", &self.detect_language)
            .field("smart_format", &self.smart_format)
            .field("punctuate", &self.punctuate)
            .field("paragraphs", &self.paragraphs)
            .field("summarize", &self.summarize)
            .field("topics", &self.topics)
            .field("intents", &self.intents)
            .field("sentiment", &self.sentiment)
            .field("detect_entities", &self.detect_entities)
            .field("redaction_count", &redaction_count)
            .field("replace_present", &self.replace.is_some())
            .field("search_present", &self.search.is_some())
            .field("keyterm_present", &self.keyterm.is_some())
            .field("diarize_model", &self.diarize_model)
            .field("utterances", &self.utterances)
            .field("utt_split", &self.utt_split)
            .field("filler_words", &self.filler_words)
            .finish()
    }
}

impl DeepgramTranscriptionOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_detect_language(mut self, value: bool) -> Self {
        self.detect_language = Some(value);
        self
    }

    pub const fn with_smart_format(mut self, value: bool) -> Self {
        self.smart_format = Some(value);
        self
    }

    pub const fn with_punctuate(mut self, value: bool) -> Self {
        self.punctuate = Some(value);
        self
    }

    pub const fn with_paragraphs(mut self, value: bool) -> Self {
        self.paragraphs = Some(value);
        self
    }

    pub const fn with_summarize(mut self, value: DeepgramSummarizeOption) -> Self {
        self.summarize = Some(value);
        self
    }

    pub const fn with_topics(mut self, value: bool) -> Self {
        self.topics = Some(value);
        self
    }

    pub const fn with_intents(mut self, value: bool) -> Self {
        self.intents = Some(value);
        self
    }

    pub const fn with_sentiment(mut self, value: bool) -> Self {
        self.sentiment = Some(value);
        self
    }

    pub const fn with_detect_entities(mut self, value: bool) -> Self {
        self.detect_entities = Some(value);
        self
    }

    pub fn with_redact(mut self, value: impl Into<String>) -> Self {
        self.redact = Some(DeepgramRedaction::Single(value.into()));
        self
    }

    pub fn with_redactions<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.redact = Some(DeepgramRedaction::Multiple(
            values.into_iter().map(Into::into).collect(),
        ));
        self
    }

    pub fn with_replace(mut self, value: impl Into<String>) -> Self {
        self.replace = Some(value.into());
        self
    }

    pub fn with_search(mut self, value: impl Into<String>) -> Self {
        self.search = Some(value.into());
        self
    }

    pub fn with_keyterm(mut self, value: impl Into<String>) -> Self {
        self.keyterm = Some(value.into());
        self
    }

    pub const fn with_diarize_model(mut self, value: DeepgramDiarizeModel) -> Self {
        self.diarize_model = Some(value);
        self
    }

    pub const fn with_utterances(mut self, value: bool) -> Self {
        self.utterances = Some(value);
        self
    }

    pub const fn with_utt_split(mut self, value: f64) -> Self {
        self.utt_split = Some(value);
        self
    }

    pub const fn with_filler_words(mut self, value: bool) -> Self {
        self.filler_words = Some(value);
        self
    }

    pub fn detect_language(&self) -> Option<bool> {
        self.detect_language
    }

    pub fn diarize_model(&self) -> Option<DeepgramDiarizeModel> {
        self.diarize_model
    }

    fn validate_values(&self) -> Result<(), ProviderOptionError> {
        for (path, value) in [
            ("replace", self.replace.as_deref()),
            ("search", self.search.as_deref()),
            ("keyterm", self.keyterm.as_deref()),
        ] {
            if value.is_some_and(invalid_text) {
                return Err(rejected(path, "must not be empty or contain controls"));
            }
        }
        if let Some(redaction) = &self.redact {
            let valid = match redaction {
                DeepgramRedaction::Single(value) => !invalid_text(value),
                DeepgramRedaction::Multiple(values) => {
                    !values.is_empty() && values.iter().all(|value| !invalid_text(value))
                }
            };
            if !valid {
                return Err(rejected(
                    "redact",
                    "must contain one or more non-empty categories",
                ));
            }
        }
        if self
            .utt_split
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(rejected("uttSplit", "must be a finite non-negative number"));
        }
        Ok(())
    }
}

impl TypedProviderOptions for DeepgramTranscriptionOptions {
    const NAMESPACE: &'static str = "deepgram";

    fn validate(&self) -> Result<(), ProviderOptionError> {
        self.validate_values()
    }
}

fn invalid_text(value: &str) -> bool {
    value.trim().is_empty() || value.chars().any(char::is_control)
}

fn rejected(path: &str, reason: &str) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: path.to_string(),
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use siumai_core::ProviderOptions;

    use super::*;

    #[test]
    fn typed_options_use_deepgram_names_and_values() {
        let options = DeepgramTranscriptionOptions::new()
            .with_detect_language(true)
            .with_smart_format(true)
            .with_summarize(DeepgramSummarizeOption::Version2)
            .with_diarize_model(DeepgramDiarizeModel::Latest)
            .with_redactions(["pci", "numbers"])
            .with_utt_split(0.8);
        let options = ProviderOptions::typed(&options).unwrap();

        assert_eq!(options.namespace().as_str(), "deepgram");
        assert_eq!(options.value()["detectLanguage"], true);
        assert_eq!(options.value()["smartFormat"], true);
        assert_eq!(options.value()["summarize"], "v2");
        assert_eq!(options.value()["diarizeModel"], "latest");
        assert_eq!(
            options.value()["redact"],
            serde_json::json!(["pci", "numbers"])
        );
        assert_eq!(options.value()["uttSplit"], 0.8);
    }

    #[test]
    fn invalid_values_are_rejected_before_erasure() {
        let options = DeepgramTranscriptionOptions::new().with_utt_split(f64::NAN);
        assert!(ProviderOptions::typed(&options).is_err());
        let options = DeepgramTranscriptionOptions::new().with_redactions(Vec::<String>::new());
        assert!(ProviderOptions::typed(&options).is_err());
    }

    #[test]
    fn debug_redacts_textual_query_values() {
        let options = DeepgramTranscriptionOptions::new()
            .with_replace("canary-replace")
            .with_search("canary-search")
            .with_keyterm("canary-keyterm");
        let debug = format!("{options:?}");

        assert!(!debug.contains("canary-replace"));
        assert!(!debug.contains("canary-search"));
        assert!(!debug.contains("canary-keyterm"));
    }
}
