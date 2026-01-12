use std::sync::{Arc, atomic::AtomicUsize};

use siumai::experimental::core::ProviderContext;
use siumai::experimental::execution::executors::audio::{AudioExecutor, HttpAudioExecutor};
use siumai::experimental::execution::transformers::audio::{AudioHttpBody, AudioTransformer};
use siumai::prelude::unified::{LlmError, SttRequest, TtsRequest};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct TestAudioTransformer;
impl AudioTransformer for TestAudioTransformer {
    fn provider_id(&self) -> &str {
        "test"
    }
    fn build_tts_body(&self, req: &TtsRequest) -> Result<AudioHttpBody, LlmError> {
        let form = reqwest::multipart::Form::new()
            .text("text", req.text.clone())
            .text("voice", req.voice.clone().unwrap_or_else(|| "alloy".into()));
        Ok(AudioHttpBody::Multipart(form))
    }
    fn build_stt_body(&self, _req: &SttRequest) -> Result<AudioHttpBody, LlmError> {
        Ok(AudioHttpBody::Json(serde_json::json!({})))
    }
    fn tts_endpoint(&self) -> &str {
        "/tts"
    }
    fn stt_endpoint(&self) -> &str {
        "/stt"
    }
    fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError> {
        Ok(json["text"].as_str().unwrap_or("").to_string())
    }
}

#[tokio::test]
async fn audio_tts_bytes_retries_on_401() {
    let server = MockServer::start().await;

    // First 401 when Authorization=Bearer bad, then 200 when Authorization=Bearer ok
    Mock::given(method("POST"))
        .and(path("/tts"))
        .and(header("authorization", "Bearer bad"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/tts"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("WAV", "audio/wav"))
        .expect(1)
        .mount(&server)
        .await;

    // Shared flipping header spec (first bad, then ok)
    let spec = Arc::new(crate::support::FlippingAuthSpec {
        counter: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = ProviderContext::new("test", server.uri(), None, Default::default());

    let exec = HttpAudioExecutor {
        provider_id: "test".into(),
        http_client: reqwest::Client::new(),
        transformer: Arc::new(TestAudioTransformer),
        provider_spec: spec,
        provider_context: ctx,
        policy: siumai::experimental::execution::ExecutionPolicy::new(),
    };

    let result = AudioExecutor::tts(
        &exec,
        TtsRequest::new("hello".into()).with_voice("alloy".into()),
    )
    .await
    .unwrap();
    assert_eq!(result.audio_data.as_slice(), b"WAV");
}
