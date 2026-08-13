//! OpenAI final-result audio transcription codecs.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use siumai_core::{
    Error, ModelId, ResponseMetadata, TranscriptSegment, TranscriptionRequest,
    TranscriptionResponse, Usage,
};

/// OpenAI Transcriptions API mode identifier.
pub const API_MODE_ID: &str = "audio-transcriptions";
/// OpenAI Audio protocol identifier used by final-result transcription.
pub const PROTOCOL_ID: &str = "openai.audio";
/// Relative OpenAI transcription endpoint.
pub const TARGET: &str = "audio/transcriptions";

/// JSON response representations supported by the final-result endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TranscriptionResponseFormat {
    #[default]
    Json,
    VerboseJson,
    DiarizedJson,
}

impl TranscriptionResponseFormat {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::VerboseJson => "verbose_json",
            Self::DiarizedJson => "diarized_json",
        }
    }
}

/// Timestamp units requested from a verbose transcription response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionTimestampGranularity {
    Segment,
    Word,
}

impl TranscriptionTimestampGranularity {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Segment => "segment",
            Self::Word => "word",
        }
    }
}

/// Checked provider-owned fields for one final-result transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionConfig {
    pub response_format: TranscriptionResponseFormat,
    pub temperature: Option<f32>,
    pub timestamp_granularities: Vec<TranscriptionTimestampGranularity>,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            response_format: TranscriptionResponseFormat::Json,
            temperature: None,
            timestamp_granularities: Vec::new(),
        }
    }
}

/// One textual multipart field produced by the protocol codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionFormField {
    name: &'static str,
    value: String,
}

impl TranscriptionFormField {
    pub const fn name(&self) -> &'static str {
        self.name
    }

    pub fn value(&self) -> &str {
        &self.value
    }
}

/// Encode all non-file multipart fields for one transcription request.
pub fn encode_transcription_fields(
    request: &TranscriptionRequest,
    model: &ModelId,
    config: &TranscriptionConfig,
) -> Vec<TranscriptionFormField> {
    let mut fields = vec![
        field("model", model.as_str()),
        field("response_format", config.response_format.as_wire()),
    ];
    if let Some(language) = request.language() {
        fields.push(field("language", language));
    }
    if let Some(prompt) = request.prompt() {
        fields.push(field("prompt", prompt));
    }
    if let Some(temperature) = config.temperature {
        fields.push(field("temperature", temperature.to_string()));
    }
    fields.extend(config.timestamp_granularities.iter().map(|granularity| {
        field(
            "timestamp_granularities[]",
            granularity.as_wire().to_string(),
        )
    }));
    fields
}

/// Decode JSON, verbose JSON, or diarized JSON into the portable contract.
pub fn decode_transcription_response(
    body: &[u8],
    requested_model: &ModelId,
) -> Result<TranscriptionResponse, Error> {
    let response = serde_json::from_slice::<TranscriptionResponseWire>(body).map_err(|source| {
        Error::protocol_violation("OpenAI transcription response is not valid JSON")
            .with_source(source)
    })?;
    let segments = if !response.segments.is_empty() {
        response
            .segments
            .iter()
            .map(|segment| TranscriptSegment {
                start_seconds: segment.start,
                end_seconds: segment.end,
                text: segment.text.clone(),
                confidence: None,
            })
            .collect()
    } else {
        response
            .words
            .iter()
            .map(|word| TranscriptSegment {
                start_seconds: word.start,
                end_seconds: word.end,
                text: word.word.clone(),
                confidence: None,
            })
            .collect()
    };
    let mut provider = BTreeMap::new();
    let speakers = response
        .segments
        .iter()
        .enumerate()
        .filter_map(|(index, segment)| segment.speaker.as_ref().map(|speaker| (index, speaker)))
        .map(|(index, speaker)| {
            if speaker.len() > 256 || speaker.chars().any(char::is_control) {
                return Err(Error::protocol_violation(
                    "OpenAI transcription response contains an invalid speaker label",
                ));
            }
            Ok(serde_json::json!({
                "segment_index": index,
                "speaker": speaker,
            }))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if !speakers.is_empty() {
        provider.insert("speakers".to_string(), Value::Array(speakers));
    }
    let usage = response.usage.unwrap_or_default();
    if let Some(seconds) = usage.seconds {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(Error::protocol_violation(
                "OpenAI transcription response contains invalid usage duration",
            ));
        }
        provider.insert("usage_seconds".to_string(), Value::from(seconds));
    }
    let decoded = TranscriptionResponse {
        text: response.text,
        language: response.language,
        confidence: None,
        duration_seconds: response.duration,
        segments,
        metadata: ResponseMetadata {
            response_id: None,
            request_id: None,
            model: Some(requested_model.clone()),
        },
        usage: Usage::default()
            .with_input_tokens(usage.input_tokens)
            .with_output_tokens(usage.output_tokens)
            .with_total_tokens(usage.total_tokens),
        warnings: Vec::new(),
        provider,
    };
    decoded.validate()?;
    Ok(decoded)
}

fn field(name: &'static str, value: impl Into<String>) -> TranscriptionFormField {
    TranscriptionFormField {
        name,
        value: value.into(),
    }
}

#[derive(Debug, Deserialize)]
struct TranscriptionResponseWire {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    segments: Vec<TranscriptionSegmentWire>,
    #[serde(default)]
    words: Vec<TranscriptionWordWire>,
    #[serde(default)]
    usage: Option<TranscriptionUsageWire>,
}

#[derive(Debug, Deserialize)]
struct TranscriptionSegmentWire {
    start: f64,
    end: f64,
    text: String,
    #[serde(default)]
    speaker: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TranscriptionWordWire {
    word: String,
    start: f64,
    end: f64,
}

#[derive(Debug, Default, Deserialize)]
struct TranscriptionUsageWire {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    seconds: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn form_fields_keep_portable_language_and_prompt() {
        let request = TranscriptionRequest::new(Bytes::from_static(b"audio"), "audio/wav")
            .unwrap()
            .with_language("en")
            .unwrap()
            .with_prompt("Siumai")
            .unwrap();
        let fields = encode_transcription_fields(
            &request,
            &ModelId::new("gpt-4o-mini-transcribe").unwrap(),
            &TranscriptionConfig {
                response_format: TranscriptionResponseFormat::VerboseJson,
                temperature: Some(0.2),
                timestamp_granularities: vec![TranscriptionTimestampGranularity::Segment],
            },
        );
        assert!(
            fields
                .iter()
                .any(|field| { field.name() == "language" && field.value() == "en" })
        );
        assert!(
            fields
                .iter()
                .any(|field| { field.name() == "prompt" && field.value() == "Siumai" })
        );
    }

    #[test]
    fn decoder_prefers_segments_and_preserves_usage() {
        let decoded = decode_transcription_response(
            br#"{"text":"hello","language":"en","duration":1.2,"segments":[{"start":0.0,"end":1.2,"text":"hello","speaker":"A"}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}"#,
            &ModelId::new("gpt-4o-transcribe-diarize").unwrap(),
        )
        .unwrap();
        assert_eq!(decoded.text, "hello");
        assert_eq!(decoded.segments.len(), 1);
        assert_eq!(decoded.provider["speakers"][0]["speaker"], "A");
    }
}
