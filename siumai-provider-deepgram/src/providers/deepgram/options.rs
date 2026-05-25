//! Deepgram typed provider options.

use crate::error::LlmError;
use crate::types::CustomProviderOptions;
use serde::{Deserialize, Serialize};

/// AI SDK-style typed options for Deepgram speech models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DeepgramSpeechModelOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bit_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mip_opt_out: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

impl DeepgramSpeechModelOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = Some(encoding.into());
        self
    }

    pub fn with_container(mut self, container: impl Into<String>) -> Self {
        self.container = Some(container.into());
        self
    }

    pub const fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    pub const fn with_bit_rate(mut self, bit_rate: u32) -> Self {
        self.bit_rate = Some(bit_rate);
        self
    }

    pub fn with_callback(mut self, callback: impl Into<String>) -> Self {
        self.callback = Some(callback.into());
        self
    }

    pub fn with_callback_method(mut self, callback_method: impl Into<String>) -> Self {
        self.callback_method = Some(callback_method.into());
        self
    }

    pub const fn with_mip_opt_out(mut self, mip_opt_out: bool) -> Self {
        self.mip_opt_out = Some(mip_opt_out);
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn into_provider_options_map_entry(self) -> Result<(String, serde_json::Value), LlmError> {
        self.to_provider_options_map_entry()
    }
}

impl CustomProviderOptions for DeepgramSpeechModelOptions {
    fn provider_id(&self) -> &str {
        "deepgram"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut obj = serde_json::Map::new();
        if let Some(value) = self.encoding.as_deref() {
            obj.insert("encoding".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.container.as_deref() {
            obj.insert("container".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.sample_rate {
            obj.insert("sampleRate".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.bit_rate {
            obj.insert("bitRate".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.callback.as_deref() {
            obj.insert("callback".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.callback_method.as_deref() {
            obj.insert("callbackMethod".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.mip_opt_out {
            obj.insert("mipOptOut".to_string(), serde_json::json!(value));
        }
        if let Some(value) = self.tag.as_deref() {
            obj.insert("tag".to_string(), serde_json::json!(value));
        }
        Ok(serde_json::Value::Object(obj))
    }
}

/// AI SDK-style typed options for Deepgram transcription models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DeepgramTranscriptionModelOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detect_language: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smart_format: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub punctuate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paragraphs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summarize: Option<DeepgramSummarizeOption>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topics: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intents: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentiment: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detect_entities: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redact: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replace: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyterm: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utterances: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utt_split: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filler_words: Option<bool>,
}

/// Deepgram summarization option.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeepgramSummarizeOption {
    Bool(bool),
    Version(String),
}

impl DeepgramTranscriptionModelOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    pub const fn with_detect_language(mut self, detect_language: bool) -> Self {
        self.detect_language = Some(detect_language);
        self
    }

    pub const fn with_smart_format(mut self, smart_format: bool) -> Self {
        self.smart_format = Some(smart_format);
        self
    }

    pub const fn with_punctuate(mut self, punctuate: bool) -> Self {
        self.punctuate = Some(punctuate);
        self
    }

    pub const fn with_paragraphs(mut self, paragraphs: bool) -> Self {
        self.paragraphs = Some(paragraphs);
        self
    }

    pub fn with_summarize(mut self, summarize: DeepgramSummarizeOption) -> Self {
        self.summarize = Some(summarize);
        self
    }

    pub const fn with_topics(mut self, topics: bool) -> Self {
        self.topics = Some(topics);
        self
    }

    pub const fn with_intents(mut self, intents: bool) -> Self {
        self.intents = Some(intents);
        self
    }

    pub const fn with_sentiment(mut self, sentiment: bool) -> Self {
        self.sentiment = Some(sentiment);
        self
    }

    pub const fn with_detect_entities(mut self, detect_entities: bool) -> Self {
        self.detect_entities = Some(detect_entities);
        self
    }

    pub fn with_redact(mut self, redact: Vec<String>) -> Self {
        self.redact = Some(redact);
        self
    }

    pub fn with_replace(mut self, replace: Vec<String>) -> Self {
        self.replace = Some(replace);
        self
    }

    pub fn with_search(mut self, search: Vec<String>) -> Self {
        self.search = Some(search);
        self
    }

    pub fn with_keyterm(mut self, keyterm: Vec<String>) -> Self {
        self.keyterm = Some(keyterm);
        self
    }

    pub const fn with_diarize(mut self, diarize: bool) -> Self {
        self.diarize = Some(diarize);
        self
    }

    pub const fn with_utterances(mut self, utterances: bool) -> Self {
        self.utterances = Some(utterances);
        self
    }

    pub const fn with_utt_split(mut self, utt_split: f64) -> Self {
        self.utt_split = Some(utt_split);
        self
    }

    pub const fn with_filler_words(mut self, filler_words: bool) -> Self {
        self.filler_words = Some(filler_words);
        self
    }

    pub fn into_provider_options_map_entry(self) -> Result<(String, serde_json::Value), LlmError> {
        self.to_provider_options_map_entry()
    }
}

impl CustomProviderOptions for DeepgramTranscriptionModelOptions {
    fn provider_id(&self) -> &str {
        "deepgram"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut obj = serde_json::Map::new();
        insert_string(&mut obj, "language", self.language.as_deref());
        insert_bool(&mut obj, "detectLanguage", self.detect_language);
        insert_bool(&mut obj, "smartFormat", self.smart_format);
        insert_bool(&mut obj, "punctuate", self.punctuate);
        insert_bool(&mut obj, "paragraphs", self.paragraphs);
        if let Some(value) = self.summarize.as_ref() {
            obj.insert("summarize".to_string(), serde_json::to_value(value)?);
        }
        insert_bool(&mut obj, "topics", self.topics);
        insert_bool(&mut obj, "intents", self.intents);
        insert_bool(&mut obj, "sentiment", self.sentiment);
        insert_bool(&mut obj, "detectEntities", self.detect_entities);
        insert_string_array(&mut obj, "redact", self.redact.as_ref());
        insert_string_array(&mut obj, "replace", self.replace.as_ref());
        insert_string_array(&mut obj, "search", self.search.as_ref());
        insert_string_array(&mut obj, "keyterm", self.keyterm.as_ref());
        insert_bool(&mut obj, "diarize", self.diarize);
        insert_bool(&mut obj, "utterances", self.utterances);
        if let Some(value) = self.utt_split {
            obj.insert("uttSplit".to_string(), serde_json::json!(value));
        }
        insert_bool(&mut obj, "fillerWords", self.filler_words);
        Ok(serde_json::Value::Object(obj))
    }
}

fn insert_string(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<&str>,
) {
    if let Some(value) = value {
        obj.insert(key.to_string(), serde_json::json!(value));
    }
}

fn insert_bool(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<bool>,
) {
    if let Some(value) = value {
        obj.insert(key.to_string(), serde_json::json!(value));
    }
}

fn insert_string_array(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<&Vec<String>>,
) {
    if let Some(value) = value {
        obj.insert(key.to_string(), serde_json::json!(value));
    }
}

pub type DeepgramSpeechOptions = DeepgramSpeechModelOptions;
pub type DeepgramSttOptions = DeepgramTranscriptionModelOptions;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CustomProviderOptions;

    #[test]
    fn deepgram_speech_options_serialize_ai_sdk_style_keys() {
        let value = DeepgramSpeechModelOptions::new()
            .with_encoding("linear16")
            .with_container("wav")
            .with_sample_rate(24_000)
            .with_bit_rate(128_000)
            .with_callback("https://example.com/callback")
            .with_callback_method("POST")
            .with_mip_opt_out(true)
            .with_tag("tag-a")
            .to_json()
            .expect("speech options serialize");

        assert_eq!(value["encoding"], serde_json::json!("linear16"));
        assert_eq!(value["container"], serde_json::json!("wav"));
        assert_eq!(value["sampleRate"], serde_json::json!(24_000));
        assert_eq!(value["bitRate"], serde_json::json!(128_000));
        assert_eq!(
            value["callback"],
            serde_json::json!("https://example.com/callback")
        );
        assert_eq!(value["callbackMethod"], serde_json::json!("POST"));
        assert_eq!(value["mipOptOut"], serde_json::json!(true));
        assert_eq!(value["tag"], serde_json::json!("tag-a"));
    }

    #[test]
    fn deepgram_transcription_options_serialize_ai_sdk_style_keys() {
        let value = DeepgramTranscriptionModelOptions::new()
            .with_language("en")
            .with_detect_language(true)
            .with_smart_format(true)
            .with_punctuate(false)
            .with_paragraphs(true)
            .with_summarize(DeepgramSummarizeOption::Version("v2".to_string()))
            .with_topics(true)
            .with_intents(true)
            .with_sentiment(true)
            .with_detect_entities(true)
            .with_redact(vec!["pii".to_string()])
            .with_replace(vec!["foo:bar".to_string()])
            .with_search(vec!["hello".to_string()])
            .with_keyterm(vec!["Deepgram".to_string()])
            .with_diarize(false)
            .with_utterances(true)
            .with_utt_split(0.8)
            .with_filler_words(true)
            .to_json()
            .expect("transcription options serialize");

        assert_eq!(value["language"], serde_json::json!("en"));
        assert_eq!(value["detectLanguage"], serde_json::json!(true));
        assert_eq!(value["smartFormat"], serde_json::json!(true));
        assert_eq!(value["punctuate"], serde_json::json!(false));
        assert_eq!(value["paragraphs"], serde_json::json!(true));
        assert_eq!(value["summarize"], serde_json::json!("v2"));
        assert_eq!(value["topics"], serde_json::json!(true));
        assert_eq!(value["intents"], serde_json::json!(true));
        assert_eq!(value["sentiment"], serde_json::json!(true));
        assert_eq!(value["detectEntities"], serde_json::json!(true));
        assert_eq!(value["redact"], serde_json::json!(["pii"]));
        assert_eq!(value["replace"], serde_json::json!(["foo:bar"]));
        assert_eq!(value["search"], serde_json::json!(["hello"]));
        assert_eq!(value["keyterm"], serde_json::json!(["Deepgram"]));
        assert_eq!(value["diarize"], serde_json::json!(false));
        assert_eq!(value["utterances"], serde_json::json!(true));
        assert_eq!(value["uttSplit"], serde_json::json!(0.8));
        assert_eq!(value["fillerWords"], serde_json::json!(true));
    }
}
