use crate::error::LlmError;
use crate::types::AudioStreamEvent;

pub(crate) fn openai_sse_error_event(label: &str, payload: &serde_json::Value) -> Option<LlmError> {
    let err = payload.get("error")?;
    Some(LlmError::ApiError {
        code: 200,
        message: format!("OpenAI {label} SSE error event"),
        details: Some(err.clone()),
    })
}

pub(crate) fn openai_sse_json_config(label: &str) -> crate::streaming::SseJsonStreamConfig {
    openai_sse_json_config_with_done_markers(label, &["[DONE]"])
}

pub(crate) fn openai_sse_json_config_with_done_markers(
    label: &str,
    done_markers: &[&str],
) -> crate::streaming::SseJsonStreamConfig {
    let mut cfg = crate::streaming::SseJsonStreamConfig::new(format!("openai {label}"));
    cfg.done_markers = done_markers.iter().map(|s| (*s).to_string()).collect();
    cfg
}

pub(crate) fn openai_sse_event_type(payload: &serde_json::Value) -> Option<&str> {
    payload.get("type").and_then(|v| v.as_str())
}

pub(crate) fn openai_sse_should_ignore_event_type(kind: &str, allowed_prefixes: &[&str]) -> bool {
    if kind.is_empty() {
        return true;
    }
    !allowed_prefixes.iter().any(|p| kind.starts_with(p))
}

pub(crate) fn openai_speech_audio_delta(
    payload: &serde_json::Value,
    format: &str,
) -> Result<AudioStreamEvent, LlmError> {
    use base64::Engine;

    let Some(b64) = payload.get("audio").and_then(|v| v.as_str()) else {
        return Err(LlmError::ParseError(
            "Missing 'audio' field in speech.audio.delta event".to_string(),
        ));
    };
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| LlmError::ParseError(format!("Invalid base64 audio chunk: {e}")))?;

    Ok(AudioStreamEvent::AudioDelta {
        data: decoded,
        format: format.to_string(),
    })
}

pub(crate) fn openai_speech_audio_done(
    payload: &serde_json::Value,
) -> Result<AudioStreamEvent, LlmError> {
    let mut metadata: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();
    if let Some(usage) = payload.get("usage") {
        metadata.insert("usage".to_string(), usage.clone());
    }
    metadata.insert("provider".to_string(), serde_json::json!("openai"));
    metadata.insert("event".to_string(), payload.clone());

    Ok(AudioStreamEvent::Done {
        duration: None,
        metadata,
    })
}

pub(crate) fn openai_transcript_text_delta(
    payload: &serde_json::Value,
) -> Result<super::transcription_streaming::OpenAiTranscriptionStreamEvent, LlmError> {
    let Some(delta) = payload.get("delta").and_then(|v| v.as_str()) else {
        return Err(LlmError::ParseError(
            "Missing 'delta' field in transcript.text.delta event".to_string(),
        ));
    };
    let logprobs = payload.get("logprobs").cloned();
    Ok(
        super::transcription_streaming::OpenAiTranscriptionStreamEvent::TextDelta {
            delta: delta.to_string(),
            logprobs,
        },
    )
}

pub(crate) fn openai_transcript_text_segment(
    payload: &serde_json::Value,
) -> Result<super::transcription_streaming::OpenAiTranscriptionStreamEvent, LlmError> {
    let Some(id) = payload.get("id").and_then(|v| v.as_str()) else {
        return Err(LlmError::ParseError(
            "Missing 'id' field in transcript.text.segment event".to_string(),
        ));
    };
    let Some(start) = payload.get("start").and_then(|v| v.as_f64()) else {
        return Err(LlmError::ParseError(
            "Missing 'start' field in transcript.text.segment event".to_string(),
        ));
    };
    let Some(end) = payload.get("end").and_then(|v| v.as_f64()) else {
        return Err(LlmError::ParseError(
            "Missing 'end' field in transcript.text.segment event".to_string(),
        ));
    };
    let Some(text) = payload.get("text").and_then(|v| v.as_str()) else {
        return Err(LlmError::ParseError(
            "Missing 'text' field in transcript.text.segment event".to_string(),
        ));
    };
    let speaker = payload
        .get("speaker")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok(
        super::transcription_streaming::OpenAiTranscriptionStreamEvent::Segment {
            id: id.to_string(),
            start: start as f32,
            end: end as f32,
            text: text.to_string(),
            speaker,
        },
    )
}

pub(crate) fn openai_transcript_text_done(
    payload: &serde_json::Value,
) -> Result<super::transcription_streaming::OpenAiTranscriptionStreamEvent, LlmError> {
    let text = payload
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let usage = payload.get("usage").cloned();
    let logprobs = payload.get("logprobs").cloned();
    Ok(
        super::transcription_streaming::OpenAiTranscriptionStreamEvent::Done {
            text,
            usage,
            logprobs,
        },
    )
}

pub(crate) fn openai_tts_force_stream_format_sse(body: &mut serde_json::Value) {
    body["stream_format"] = serde_json::Value::String("sse".to_string());
}

pub(crate) fn openai_stt_force_stream_true(
    form: reqwest::multipart::Form,
) -> reqwest::multipart::Form {
    form.text("stream", "true")
}

pub(crate) fn ensure_openai_sse_content_type(
    label: &str,
    headers: &reqwest::header::HeaderMap,
) -> Result<(), LlmError> {
    let is_sse = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.to_ascii_lowercase().contains("text/event-stream"))
        .unwrap_or(false);
    if is_sse {
        return Ok(());
    }

    let ct = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("<missing>");
    Err(LlmError::ParseError(format!(
        "Expected 'text/event-stream' for OpenAI {label} SSE streaming, got '{ct}'."
    )))
}
