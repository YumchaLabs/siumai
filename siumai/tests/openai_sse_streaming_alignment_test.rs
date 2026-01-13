#![cfg(feature = "openai")]

use futures_util::StreamExt;
use siumai::prelude::unified::{AudioStreamEvent, SttRequest, TtsRequest};
use siumai::providers::openai::OpenAiClient;
use siumai::providers::openai::OpenAiConfig;
use siumai::providers::openai::ext::speech_streaming::tts_sse_stream;
use siumai::providers::openai::ext::transcription_streaming::{
    OpenAiTranscriptionStreamEvent, stt_sse_stream,
};
use std::path::{Path, PathBuf};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("sse")
}

fn read_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture text")
}

#[tokio::test]
async fn openai_speech_sse_stream_matches_fixture() {
    let server = MockServer::start().await;
    let sse = read_text(fixtures_dir().join("speech.1.chunks.txt"));

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

    let mut events: Vec<AudioStreamEvent> = Vec::new();
    while let Some(item) = stream.next().await {
        let ev = item.unwrap();
        events.push(ev);
        if matches!(events.last(), Some(AudioStreamEvent::Done { .. })) {
            break;
        }
    }

    assert!(matches!(
        events.first(),
        Some(AudioStreamEvent::AudioDelta { .. })
    ));
    assert!(matches!(
        events.get(1),
        Some(AudioStreamEvent::AudioDelta { .. })
    ));
    assert!(matches!(events.last(), Some(AudioStreamEvent::Done { .. })));
}

#[tokio::test]
async fn openai_transcription_sse_stream_done_marker_yields_done_text() {
    let server = MockServer::start().await;
    let sse = read_text(fixtures_dir().join("transcription.1.chunks.txt"));

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
    let mut delta_text = String::new();
    let mut done_text: Option<String> = None;
    while let Some(item) = stream.next().await {
        match item.unwrap() {
            OpenAiTranscriptionStreamEvent::TextDelta { delta, .. } => delta_text.push_str(&delta),
            OpenAiTranscriptionStreamEvent::Done { text, .. } => {
                done_text = text;
                break;
            }
            _ => {}
        }
    }

    assert_eq!(delta_text, "hello");
    assert_eq!(done_text.as_deref(), Some("hello"));
}
