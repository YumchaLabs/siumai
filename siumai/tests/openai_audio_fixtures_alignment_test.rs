#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use siumai::extensions::AudioCapability;
use siumai::prelude::unified::{ProviderOptionsMap, Siumai, SttRequest, TtsRequest};
use std::path::{Path, PathBuf};
use wiremock::matchers::{
    body_json, body_string_contains, header, header_exists, header_regex, method, path,
};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("audio")
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let text = std::fs::read_to_string(path).expect("read fixture json");
    serde_json::from_str(&text).expect("parse fixture json")
}

#[tokio::test]
async fn openai_tts_sends_json_and_returns_audio_bytes() {
    let server = MockServer::start().await;

    let expected = serde_json::json!({
        "model": "gpt-4o-mini-tts",
        "input": "hello",
        "voice": "alloy",
        "response_format": "wav",
        "speed": 1.0,
        "instructions": "speak clearly"
    });

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(body_json(expected))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(b"RIFF....WAVE".to_vec())
                .insert_header("content-type", "audio/wav"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .openai()
        .api_key("test-api-key")
        .base_url(format!("{}/v1", server.uri()))
        .model("gpt-4o")
        .build()
        .await
        .expect("build ok");

    let mut provider_options_map = ProviderOptionsMap::default();
    provider_options_map.insert(
        "openai",
        serde_json::json!({
            "instructions": "speak clearly"
        }),
    );

    let resp = client
        .text_to_speech(
            TtsRequest::new("hello".to_string())
                .with_model("gpt-4o-mini-tts".to_string())
                .with_voice("alloy".to_string())
                .with_format("wav".to_string())
                .with_speed(1.0)
                .with_provider_options_map(provider_options_map),
        )
        .await
        .expect("tts ok");

    assert_eq!(resp.format, "wav");
    assert_eq!(resp.audio_data, b"RIFF....WAVE".to_vec());
}

#[tokio::test]
async fn openai_stt_sends_multipart_and_maps_metadata() {
    let server = MockServer::start().await;
    let stt_body: serde_json::Value = read_json(fixtures_dir().join("stt_response.json"));

    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(header_exists("content-type"))
        .and(header_regex("content-type", "multipart/form-data"))
        .and(body_string_contains("name=\"model\""))
        .and(body_string_contains("whisper-1"))
        .and(body_string_contains("name=\"response_format\""))
        .and(body_string_contains("json"))
        .and(body_string_contains("name=\"language\""))
        .and(body_string_contains("en"))
        .and(body_string_contains("timestamp_granularities[]"))
        .and(body_string_contains("word"))
        .and(body_string_contains("name=\"prompt\""))
        .and(body_string_contains("test prompt"))
        .and(body_string_contains("name=\"temperature\""))
        .and(body_string_contains("0.2"))
        .and(body_string_contains("include[]"))
        .and(body_string_contains("logprobs"))
        .and(body_string_contains("known_speaker_names[]"))
        .and(body_string_contains("Alice"))
        .and(body_string_contains("known_speaker_references[]"))
        .and(body_string_contains("ref_1"))
        .and(body_string_contains("audio.mp3"))
        .and(body_string_contains("hello"))
        .respond_with(ResponseTemplate::new(200).set_body_json(stt_body))
        .expect(1)
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .openai()
        .api_key("test-api-key")
        .base_url(format!("{}/v1", server.uri()))
        .model("gpt-4o")
        .build()
        .await
        .expect("build ok");

    let mut req = SttRequest::from_audio(b"hello".to_vec());
    req.language = Some("en".to_string());
    req.timestamp_granularities = Some(vec!["word".to_string()]);

    req.extra_params.insert(
        "prompt".to_string(),
        serde_json::Value::String("test prompt".to_string()),
    );
    req.extra_params
        .insert("temperature".to_string(), serde_json::json!(0.2));
    req.extra_params
        .insert("include".to_string(), serde_json::json!(["logprobs"]));
    req.extra_params.insert(
        "known_speaker_names".to_string(),
        serde_json::json!(["Alice"]),
    );
    req.extra_params.insert(
        "known_speaker_references".to_string(),
        serde_json::json!(["ref_1"]),
    );

    let resp = client.speech_to_text(req).await.expect("stt ok");
    assert_eq!(resp.text, "hello");
    assert_eq!(resp.language.as_deref(), Some("en"));
    assert!(resp.duration.unwrap_or_default() > 1.0);

    let words = resp.words.expect("words");
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].word, "hello");

    assert!(resp.metadata.contains_key("usage"));
    assert!(resp.metadata.contains_key("segments"));
    assert!(resp.metadata.contains_key("logprobs"));
}

#[tokio::test]
async fn openai_audio_translation_sends_multipart_and_parses_json() {
    let server = MockServer::start().await;
    let translation_body: serde_json::Value =
        read_json(fixtures_dir().join("translation_response.json"));

    Mock::given(method("POST"))
        .and(path("/v1/audio/translations"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(header_exists("content-type"))
        .and(header_regex("content-type", "multipart/form-data"))
        .and(body_string_contains("name=\"model\""))
        .and(body_string_contains("whisper-1"))
        .and(body_string_contains("name=\"response_format\""))
        .and(body_string_contains("json"))
        .and(body_string_contains("name=\"prompt\""))
        .and(body_string_contains("translate this"))
        .and(body_string_contains("name=\"temperature\""))
        .and(body_string_contains("0"))
        .and(body_string_contains("audio.mp3"))
        .and(body_string_contains("hello"))
        .respond_with(ResponseTemplate::new(200).set_body_json(translation_body))
        .expect(1)
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .openai()
        .api_key("test-api-key")
        .base_url(format!("{}/v1", server.uri()))
        .model("gpt-4o")
        .build()
        .await
        .expect("build ok");

    let mut req = siumai_core::types::AudioTranslationRequest::from_audio(b"hello".to_vec());
    req.extra_params.insert(
        "prompt".to_string(),
        serde_json::Value::String("translate this".to_string()),
    );
    req.extra_params
        .insert("temperature".to_string(), serde_json::json!(0));
    req.extra_params.insert(
        "response_format".to_string(),
        serde_json::Value::String("json".to_string()),
    );

    let resp = client.translate_audio(req).await.expect("translate ok");
    assert_eq!(resp.text, "hello (translated)");
    assert!(resp.metadata.contains_key("usage"));
    assert!(resp.metadata.contains_key("segments"));
}
