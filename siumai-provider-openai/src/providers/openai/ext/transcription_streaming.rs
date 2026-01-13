//! OpenAI transcription streaming extensions (SSE transcript events).
//!
//! This module intentionally sits outside the Vercel-aligned unified surface.
//! Use it when you need OpenAI-specific STT streaming (`stream=true`).

use crate::error::LlmError;
use crate::types::SttRequest;

pub use crate::providers::openai::client::transcription_streaming::{
    OpenAiTranscriptionStream, OpenAiTranscriptionStreamEvent,
};

/// Stream OpenAI STT transcript using SSE (`stream=true`).
///
/// Notes:
/// - This is a provider extension (not part of the unified `TranscriptionCapability`).
/// - Uses `POST /audio/transcriptions` with `multipart/form-data`.
pub async fn stt_sse_stream(
    client: &crate::providers::openai::OpenAiClient,
    request: SttRequest,
) -> Result<OpenAiTranscriptionStream, LlmError> {
    client.stt_sse_stream(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::{OpenAiClient, OpenAiConfig};
    use futures_util::StreamExt;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn openai_stt_sse_stream_parses_delta_and_done() {
        let server = MockServer::start().await;

        let sse = concat!(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hel\"}\n\n",
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"lo\"}\n\n",
            "data: {\"type\":\"transcript.text.done\",\"text\":\"hello\",\"usage\":{\"type\":\"tokens\",\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
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

        let req = SttRequest::from_audio(b"abc".to_vec());
        let mut req = req;
        req.model = Some("gpt-4o-mini-transcribe".to_string());

        let mut stream = stt_sse_stream(&client, req).await.unwrap();

        let mut out = String::new();
        let mut saw_done = false;
        while let Some(item) = stream.next().await {
            match item.unwrap() {
                OpenAiTranscriptionStreamEvent::TextDelta { delta, .. } => out.push_str(&delta),
                OpenAiTranscriptionStreamEvent::Done { text, usage, .. } => {
                    assert_eq!(text.as_deref(), Some("hello"));
                    assert_eq!(usage.unwrap()["total_tokens"], serde_json::json!(3));
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }

        assert_eq!(out, "hello");
        assert!(saw_done);

        // Verify request contains `stream=true` multipart field.
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .body
                .windows(b"name=\"stream\"".len())
                .any(|w| w == b"name=\"stream\"")
        );
        assert!(
            requests[0]
                .body
                .windows(b"true".len())
                .any(|w| w == b"true")
        );
    }

    #[tokio::test]
    async fn openai_stt_sse_stream_uses_done_marker_as_eof() {
        let server = MockServer::start().await;

        let sse = concat!(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hel\"}\n\n",
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"lo\"}\n\n",
            "data: [DONE]\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
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

        let mut req = SttRequest::from_audio(b"abc".to_vec());
        req.model = Some("gpt-4o-mini-transcribe".to_string());

        let mut stream = stt_sse_stream(&client, req).await.unwrap();

        let mut deltas = String::new();
        let mut done_text: Option<String> = None;
        while let Some(item) = stream.next().await {
            match item.unwrap() {
                OpenAiTranscriptionStreamEvent::TextDelta { delta, .. } => deltas.push_str(&delta),
                OpenAiTranscriptionStreamEvent::Done { text, .. } => {
                    done_text = text;
                    break;
                }
                _ => {}
            }
        }

        assert_eq!(deltas, "hello");
        assert_eq!(done_text.as_deref(), Some("hello"));
    }
}
