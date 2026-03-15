use super::XaiClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::audio::{AudioExecutor, AudioExecutorBuilder};
use crate::execution::transformers::audio::{
    AudioHttpBody, AudioTransformer as RequestAudioTransformer,
};
use crate::traits::{AudioCapability, ProviderCapabilities};
use crate::types::{AudioFeature, SttRequest, SttResponse, TtsRequest, TtsResponse};
use async_trait::async_trait;
use reqwest::header::HeaderMap;
use std::sync::Arc;

const PROVIDER_ID: &str = "xai";
const DEFAULT_VOICE_ID: &str = "eve";
const DEFAULT_CODEC: &str = "mp3";
const DEFAULT_SAMPLE_RATE: u64 = 24_000;
const DEFAULT_BIT_RATE: u64 = 128_000;

#[derive(Clone, Copy, Default)]
struct XaiSpeechSpec;

impl ProviderSpec for XaiSpeechSpec {
    fn id(&self) -> &'static str {
        PROVIDER_ID
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_speech()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        siumai_protocol_openai::standards::openai::headers::build_openai_compatible_json_headers(
            ctx,
        )
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        siumai_protocol_openai::standards::openai::errors::classify_openai_compatible_http_error(
            PROVIDER_ID,
            status,
            body_text,
        )
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn choose_audio_transformer(&self, _ctx: &ProviderContext) -> crate::core::AudioTransformer {
        crate::core::AudioTransformer {
            transformer: Arc::new(XaiSpeechTransformer),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct XaiSpeechTransformer;

impl RequestAudioTransformer for XaiSpeechTransformer {
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    fn build_tts_body(&self, req: &TtsRequest) -> Result<AudioHttpBody, LlmError> {
        let voice_id = resolve_voice_id(req)?;
        let output_format = build_output_format(req)?;
        let mut body = serde_json::Map::new();
        if let Some(model) = req.model.as_ref() {
            body.insert("model".to_string(), serde_json::json!(model));
        }
        body.insert("text".to_string(), serde_json::json!(req.text));
        body.insert("voice_id".to_string(), serde_json::json!(voice_id));
        body.insert("output_format".to_string(), output_format);

        Ok(AudioHttpBody::Json(serde_json::Value::Object(body)))
    }

    fn build_stt_body(&self, _req: &SttRequest) -> Result<AudioHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "xAI provider-owned audio path currently supports text-to-speech only".to_string(),
        ))
    }

    fn tts_endpoint(&self) -> &str {
        "/tts"
    }

    fn stt_endpoint(&self) -> &str {
        "/stt-unsupported"
    }

    fn parse_stt_response(&self, _json: &serde_json::Value) -> Result<String, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "xAI provider-owned audio path currently supports text-to-speech only".to_string(),
        ))
    }
}

fn provider_option_object(req: &TtsRequest) -> Option<&serde_json::Map<String, serde_json::Value>> {
    req.provider_options_map.get_object(PROVIDER_ID)
}

fn provider_value<'a>(req: &'a TtsRequest, key: &str) -> Option<&'a serde_json::Value> {
    if let Some(obj) = provider_option_object(req)
        && let Some(value) = obj.get(key)
    {
        return Some(value);
    }
    req.extra_params.get(key)
}

fn output_format_object<'a>(
    req: &'a TtsRequest,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    provider_value(req, "output_format").and_then(|value| value.as_object())
}

fn output_value<'a>(req: &'a TtsRequest, key: &str) -> Option<&'a serde_json::Value> {
    output_format_object(req).and_then(|obj| obj.get(key))
}

fn resolve_voice_id(req: &TtsRequest) -> Result<String, LlmError> {
    if let Some(voice) = &req.voice {
        return Ok(voice.clone());
    }
    if let Some(value) = provider_value(req, "voice_id")
        && let Some(voice) = value.as_str()
    {
        return Ok(voice.to_string());
    }
    Ok(DEFAULT_VOICE_ID.to_string())
}

fn resolve_codec(req: &TtsRequest) -> Result<String, LlmError> {
    if let Some(format) = &req.format {
        return Ok(format.clone());
    }
    if let Some(value) = output_value(req, "codec").or_else(|| provider_value(req, "codec"))
        && let Some(codec) = value.as_str()
    {
        return Ok(codec.to_string());
    }
    Ok(DEFAULT_CODEC.to_string())
}

fn parse_u64_field(req: &TtsRequest, key: &str) -> Result<Option<u64>, LlmError> {
    let value = output_value(req, key).or_else(|| provider_value(req, key));
    let Some(value) = value else {
        return Ok(None);
    };

    value.as_u64().map(Some).ok_or_else(|| {
        LlmError::InvalidParameter(format!(
            "xAI TTS field '{key}' must be an unsigned integer when provided"
        ))
    })
}

fn build_output_format(req: &TtsRequest) -> Result<serde_json::Value, LlmError> {
    let codec = resolve_codec(req)?;
    let sample_rate = parse_u64_field(req, "sample_rate")?.unwrap_or(DEFAULT_SAMPLE_RATE);
    let bit_rate = parse_u64_field(req, "bit_rate")?;

    let mut output_format = serde_json::Map::new();
    output_format.insert("codec".to_string(), serde_json::json!(codec.clone()));
    output_format.insert("sample_rate".to_string(), serde_json::json!(sample_rate));

    if codec == "mp3" {
        output_format.insert(
            "bit_rate".to_string(),
            serde_json::json!(bit_rate.unwrap_or(DEFAULT_BIT_RATE)),
        );
    } else if let Some(bit_rate) = bit_rate {
        output_format.insert("bit_rate".to_string(), serde_json::json!(bit_rate));
    }

    Ok(serde_json::Value::Object(output_format))
}

fn resolve_response_format(req: &TtsRequest) -> Result<String, LlmError> {
    resolve_codec(req)
}

fn build_audio_executor(
    client: &XaiClient,
) -> Arc<crate::execution::executors::audio::HttpAudioExecutor> {
    let spec = Arc::new(XaiSpeechSpec);
    let ctx = client.provider_context();

    let mut builder = AudioExecutorBuilder::new(PROVIDER_ID, client.http_client())
        .with_spec(spec)
        .with_context(ctx)
        .with_interceptors(client.http_interceptors());

    if let Some(transport) = client.http_transport() {
        builder = builder.with_transport(transport);
    }

    if let Some(retry) = client.retry_options() {
        builder = builder.with_retry_options(retry);
    }

    builder.build()
}

#[async_trait]
impl AudioCapability for XaiClient {
    fn supported_features(&self) -> &[AudioFeature] {
        const FEATURES: &[AudioFeature] = &[AudioFeature::TextToSpeech];
        FEATURES
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        let request = request.with_model_if_missing(self.inner().model().to_string());
        let format = resolve_response_format(&request)?;
        let exec = build_audio_executor(self);
        let result = AudioExecutor::tts(&*exec, request).await?;

        Ok(TtsResponse {
            audio_data: result.audio_data,
            format,
            duration: result.duration,
            sample_rate: result.sample_rate,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn speech_to_text(&self, _request: SttRequest) -> Result<SttResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "xAI provider-owned audio path currently supports text-to-speech only".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::LlmClient;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::provider_options::XaiTtsOptions;
    use crate::providers::xai::XaiConfig;
    use crate::providers::xai::ext::XaiTtsRequestExt;
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct BinaryCaptureTransport {
        body: Arc<Vec<u8>>,
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl Default for BinaryCaptureTransport {
        fn default() -> Self {
            Self {
                body: Arc::new(vec![1, 2, 3, 4]),
                last: Arc::new(Mutex::new(None)),
            }
        }
    }

    impl BinaryCaptureTransport {
        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().expect("lock transport request").take()
        }
    }

    #[async_trait]
    impl HttpTransport for BinaryCaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().expect("lock transport request") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("audio/mpeg"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.body.as_ref().clone(),
            })
        }

        async fn execute_stream(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Ok(HttpTransportStreamResponse {
                status: 400,
                headers,
                body: HttpTransportStreamBody::from_bytes(br#"{}"#.to_vec()),
            })
        }
    }

    #[test]
    fn xai_speech_spec_declares_speech_only() {
        let caps = XaiSpeechSpec.capabilities();
        assert!(caps.supports("speech"));
        assert!(caps.supports("audio"));
        assert!(!caps.supports("transcription"));
    }

    #[test]
    fn xai_speech_transformer_rejects_stt_body() {
        match XaiSpeechTransformer.build_stt_body(&SttRequest::from_audio(b"abc".to_vec())) {
            Err(LlmError::UnsupportedOperation(message)) => {
                assert!(message.contains("text-to-speech only"));
            }
            Ok(_) => panic!("expected UnsupportedOperation for xai stt body building"),
            Err(other) => panic!("expected UnsupportedOperation, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn xai_client_exposes_provider_owned_speech_capability() {
        let cfg = XaiConfig::new("test-key")
            .with_model("grok-4")
            .with_base_url("https://example.com/v1");
        let client = XaiClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("build xai client");

        let caps = client.capabilities();
        assert!(caps.supports("speech"));
        assert!(caps.supports("audio"));
        assert!(!caps.supports("transcription"));
        assert!(client.as_audio_capability().is_some());
        assert!(client.as_speech_capability().is_some());
        assert!(client.as_speech_extras().is_none());
        assert!(client.as_transcription_capability().is_none());
        assert!(client.as_transcription_extras().is_none());
        assert_eq!(
            client
                .as_audio_capability()
                .expect("audio capability")
                .supported_features(),
            &[AudioFeature::TextToSpeech]
        );
    }

    #[tokio::test]
    async fn xai_client_speech_to_text_is_unsupported() {
        let cfg = XaiConfig::new("test-key")
            .with_model("grok-4")
            .with_base_url("https://example.com/v1");
        let client = XaiClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("build xai client");

        let err = client
            .speech_to_text(SttRequest::from_audio(b"abc".to_vec()))
            .await
            .expect_err("xai stt should remain unsupported");

        match err {
            LlmError::UnsupportedOperation(message) => {
                assert!(message.contains("text-to-speech only"));
            }
            other => panic!("expected UnsupportedOperation, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn xai_client_tts_uses_provider_owned_tts_endpoint_and_shape() {
        let transport = BinaryCaptureTransport::default();
        let cfg = XaiConfig::new("test-key")
            .with_model("grok-4")
            .with_base_url("https://example.com/v1")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("build xai client");

        let request = TtsRequest::new("hello from xai".to_string())
            .with_voice("aria".to_string())
            .with_format("mp3".to_string())
            .with_xai_tts_options(
                XaiTtsOptions::new()
                    .with_sample_rate(44_100)
                    .with_bit_rate(192_000),
            );

        let response = client
            .text_to_speech(request)
            .await
            .expect("tts should succeed over custom transport");

        assert_eq!(response.audio_data, vec![1, 2, 3, 4]);
        assert_eq!(response.format, "mp3");

        let captured = transport.take().expect("captured request");
        assert_eq!(captured.url, "https://example.com/v1/tts");
        assert_eq!(captured.body["model"], serde_json::json!("grok-4"));
        assert_eq!(captured.body["text"], serde_json::json!("hello from xai"));
        assert_eq!(captured.body["voice_id"], serde_json::json!("aria"));
        assert_eq!(
            captured.body["output_format"]["codec"],
            serde_json::json!("mp3")
        );
        assert_eq!(
            captured.body["output_format"]["sample_rate"],
            serde_json::json!(44100)
        );
        assert_eq!(
            captured.body["output_format"]["bit_rate"],
            serde_json::json!(192000)
        );
    }
}
