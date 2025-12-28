//! OpenAI Speech streaming extensions (SSE audio).
//!
//! This module intentionally sits outside the Vercel-aligned unified surface.
//! Use it when you need OpenAI-specific TTS streaming (`stream_format: "sse"`).

use crate::error::LlmError;
use crate::types::{AudioStream, TtsRequest};

/// Stream OpenAI TTS audio using SSE (`stream_format: "sse"`).
///
/// Notes:
/// - Requires a model that supports SSE streaming (OpenAI docs: not supported for `tts-1` / `tts-1-hd`).
/// - The unified `SpeechCapability` remains non-streaming (Vercel-aligned).
pub async fn tts_sse_stream(
    client: &crate::providers::openai::OpenAiClient,
    request: TtsRequest,
) -> Result<AudioStream, LlmError> {
    client.tts_sse_stream(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::{OpenAiClient, OpenAiConfig};
    use crate::types::AudioStreamEvent;
    use futures_util::StreamExt;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn openai_tts_sse_stream_sends_stream_format_and_decodes_audio() {
        let server = MockServer::start().await;

        let sse = concat!(
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"YWJj\"}\n\n",
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"ZA==\"}\n\n",
            "data: {\"type\":\"speech.audio.done\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .and(header("accept", "text/event-stream"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(sse, "text/event-stream"),
            )
            .mount(&server)
            .await;

        let cfg = OpenAiConfig::new("KEY").with_base_url(format!("{}/v1", server.uri()));
        let client = OpenAiClient::new(cfg, reqwest::Client::new());

        let req = TtsRequest::new("hello".to_string())
            .with_model("gpt-4o-mini-tts".to_string())
            .with_format("mp3".to_string());

        let mut stream = tts_sse_stream(&client, req).await.unwrap();

        let mut collected: Vec<AudioStreamEvent> = Vec::new();
        while let Some(item) = stream.next().await {
            let ev = item.unwrap();
            collected.push(ev);
            if matches!(collected.last(), Some(AudioStreamEvent::Done { .. })) {
                break;
            }
        }

        assert!(matches!(
            collected.first(),
            Some(AudioStreamEvent::AudioDelta { .. })
        ));
        assert!(matches!(
            collected.get(1),
            Some(AudioStreamEvent::AudioDelta { .. })
        ));
        assert!(matches!(collected.last(), Some(AudioStreamEvent::Done { .. })));

        match collected.first().unwrap() {
            AudioStreamEvent::AudioDelta { data, format } => {
                assert_eq!(format, "mp3");
                assert_eq!(data, &b"abc".to_vec());
            }
            _ => panic!("expected audio delta"),
        }

        match collected.get(1).unwrap() {
            AudioStreamEvent::AudioDelta { data, .. } => {
                assert_eq!(data, &b"d".to_vec());
            }
            _ => panic!("expected audio delta"),
        }

        match collected.last().unwrap() {
            AudioStreamEvent::Done { metadata, .. } => {
                assert_eq!(metadata.get("provider"), Some(&serde_json::json!("openai")));
                assert_eq!(metadata["usage"]["total_tokens"], serde_json::json!(3));
            }
            _ => panic!("expected done"),
        }

        // Verify request JSON includes `stream_format: "sse"`.
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let json: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(json["stream_format"], "sse");
        assert_eq!(json["model"], "gpt-4o-mini-tts");
    }

    #[tokio::test]
    async fn openai_tts_sse_stream_rejects_tts_1_defaults() {
        let server = MockServer::start().await;
        let cfg = OpenAiConfig::new("KEY").with_base_url(format!("{}/v1", server.uri()));
        let client = OpenAiClient::new(cfg, reqwest::Client::new());

        let req = TtsRequest::new("hello".to_string()); // defaults to tts-1
        match tts_sse_stream(&client, req).await {
            Ok(_) => panic!("expected error"),
            Err(err) => assert!(matches!(err, LlmError::UnsupportedOperation(_))),
        }
    }
}
