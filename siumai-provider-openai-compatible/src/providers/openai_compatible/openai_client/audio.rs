use super::OpenAiCompatibleClient;
use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::executors::audio::{AudioExecutor, AudioExecutorBuilder, HttpAudioExecutor};
use crate::traits::AudioCapability;
use crate::types::{AudioFeature, SttRequest, SttResponse, TtsRequest, TtsResponse, WordTimestamp};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

impl OpenAiCompatibleClient {
    fn resolve_speech_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::super::config::get_default_speech_model(
            &self.config.provider_id,
        ))
    }

    fn resolve_transcription_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::super::config::get_default_transcription_model(
            &self.config.provider_id,
        ))
    }

    async fn build_audio_executor(&self) -> Result<Arc<HttpAudioExecutor>, LlmError> {
        let ctx = self.build_context().await?;
        let spec = Arc::new(self.compat_spec());

        let mut builder =
            AudioExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
                .with_spec(spec)
                .with_context(ctx)
                .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        Ok(builder.build())
    }
}

#[async_trait]
impl AudioCapability for OpenAiCompatibleClient {
    fn supported_features(&self) -> &[AudioFeature] {
        use AudioFeature::{SpeechToText, TextToSpeech};

        const SPEECH_ONLY: &[AudioFeature] = &[TextToSpeech];
        const TRANSCRIPTION_ONLY: &[AudioFeature] = &[SpeechToText];
        const SPEECH_AND_TRANSCRIPTION: &[AudioFeature] = &[TextToSpeech, SpeechToText];
        const NONE: &[AudioFeature] = &[];

        let caps = self.capabilities();
        match (caps.supports("speech"), caps.supports("transcription")) {
            (true, true) => SPEECH_AND_TRANSCRIPTION,
            (true, false) => SPEECH_ONLY,
            (false, true) => TRANSCRIPTION_ONLY,
            (false, false) => NONE,
        }
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        let request = if let Some(model) = self.resolve_speech_model_default() {
            request.with_model_if_missing(model)
        } else {
            request
        };
        let exec = self.build_audio_executor().await?;
        let result = AudioExecutor::tts(&*exec, request.clone()).await?;

        Ok(TtsResponse {
            audio_data: result.audio_data,
            format: request.format.unwrap_or_else(|| "mp3".to_string()),
            duration: result.duration,
            sample_rate: result.sample_rate,
            metadata: HashMap::new(),
            warnings: None,
            provider_metadata: None,
            request: result.request,
            response: result.response,
        })
    }

    async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        let request = if let Some(model) = self.resolve_transcription_model_default() {
            request.with_model_if_missing(model)
        } else {
            request
        };
        let exec = self.build_audio_executor().await?;
        let result = AudioExecutor::stt(&*exec, request).await?;
        let request = result.request;
        let response = result.response;
        let raw = result.raw;

        let language = raw
            .get("language")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string());
        let duration = raw
            .get("duration")
            .and_then(|value| value.as_f64())
            .map(|value| value as f32);
        let words = raw
            .get("words")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        let object = item.as_object()?;
                        let word = object.get("word")?.as_str()?.to_string();
                        let start = object.get("start")?.as_f64()? as f32;
                        let end = object.get("end")?.as_f64()? as f32;
                        Some(WordTimestamp {
                            word,
                            start,
                            end,
                            confidence: None,
                        })
                    })
                    .collect::<Vec<_>>()
            });

        let mut metadata = HashMap::new();
        if let Some(usage) = raw.get("usage") {
            metadata.insert("usage".to_string(), usage.clone());
        }
        if let Some(segments) = raw.get("segments") {
            metadata.insert("segments".to_string(), segments.clone());
        }
        if let Some(logprobs) = raw.get("logprobs") {
            metadata.insert("logprobs".to_string(), logprobs.clone());
        }

        Ok(SttResponse {
            text: result.text,
            language,
            confidence: None,
            words,
            duration,
            metadata,
            warnings: None,
            provider_metadata: None,
            request,
            response,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use crate::providers::openai_compatible::OpenAiCompatibleConfig;
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl CaptureTransport {
        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 401,
                headers,
                body: br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                    .to_vec(),
            })
        }
    }

    #[derive(Clone)]
    struct MultipartResponseTransport {
        response_body: Arc<Vec<u8>>,
        last: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
    }

    impl MultipartResponseTransport {
        fn new(response: serde_json::Value) -> Self {
            Self {
                response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
                last: Arc::new(Mutex::new(None)),
            }
        }

        fn take(&self) -> Option<HttpTransportMultipartRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for MultipartResponseTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 501,
                headers,
                body: br#"{"error":{"message":"json unsupported in test","type":"test_error","code":"unsupported"}}"#
                    .to_vec(),
            })
        }

        async fn execute_multipart(
            &self,
            request: HttpTransportMultipartRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.response_body.as_ref().clone(),
            })
        }
    }

    #[derive(Clone)]
    struct BytesResponseTransport {
        response_body: Arc<Vec<u8>>,
        response_content_type: &'static str,
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl BytesResponseTransport {
        fn new(response_body: Vec<u8>, response_content_type: &'static str) -> Self {
            Self {
                response_body: Arc::new(response_body),
                response_content_type,
                last: Arc::new(Mutex::new(None)),
            }
        }

        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for BytesResponseTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_static(self.response_content_type),
            );

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.response_body.as_ref().clone(),
            })
        }
    }

    fn make_audio_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "compat-audio".to_string(),
            name: "Compat Audio".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tts".to_string(), "stt".to_string()],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_fireworks_transcription_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "fireworks".to_string(),
            name: "Fireworks AI".to_string(),
            base_url: "https://api.fireworks.ai/inference/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["transcription".to_string()],
            default_model: Some("whisper-v3".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_together_audio_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "together".to_string(),
            name: "Together AI".to_string(),
            base_url: "https://api.together.xyz/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["speech".to_string(), "transcription".to_string()],
            default_model: Some("cartesia/sonic-2".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_siliconflow_audio_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "siliconflow".to_string(),
            name: "SiliconFlow".to_string(),
            base_url: "https://api.siliconflow.cn/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["speech".to_string(), "transcription".to_string()],
            default_model: Some("FunAudioLLM/SenseVoiceSmall".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    #[tokio::test]
    async fn openai_compatible_client_exposes_audio_capability_views() {
        let cfg = OpenAiCompatibleConfig::new(
            "compat-audio",
            "test-key",
            "https://api.test.com/v1",
            make_audio_adapter(),
        )
        .with_model("gpt-audio-mini");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let caps = client.capabilities();
        assert!(caps.supports("audio"));
        assert!(caps.supports("speech"));
        assert!(caps.supports("transcription"));
        assert!(client.as_audio_capability().is_some());
        assert!(client.as_speech_capability().is_some());
        assert!(client.as_transcription_capability().is_some());

        let features = client.as_audio_capability().unwrap().supported_features();
        assert_eq!(features.len(), 2);
        assert!(features.contains(&AudioFeature::TextToSpeech));
        assert!(features.contains(&AudioFeature::SpeechToText));
    }

    #[tokio::test]
    async fn openai_compatible_client_exposes_fireworks_transcription_only_audio_views() {
        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            "https://api.fireworks.ai/inference/v1",
            make_fireworks_transcription_adapter(),
        )
        .with_model("whisper-v3");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let caps = client.capabilities();
        assert!(caps.supports("transcription"));
        assert!(caps.supports("audio"));
        assert!(!caps.supports("speech"));
        assert!(client.as_audio_capability().is_some());
        assert!(client.as_transcription_capability().is_some());
        assert!(client.as_speech_capability().is_none());
        assert_eq!(
            client.as_audio_capability().unwrap().supported_features(),
            &[AudioFeature::SpeechToText]
        );
    }

    #[tokio::test]
    async fn openai_compatible_client_fireworks_stt_uses_provider_audio_base_with_custom_transport()
    {
        let transport = MultipartResponseTransport::new(serde_json::json!({
            "text": "hello from fireworks",
            "language": "en"
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            "https://api.fireworks.ai/inference/v1",
            make_fireworks_transcription_adapter(),
        )
        .with_model("whisper-v3")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
        request.model = Some("whisper-v3".to_string());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = AudioCapability::speech_to_text(&client, request)
            .await
            .expect("fireworks stt should succeed through custom multipart transport");

        assert_eq!(response.text, "hello from fireworks");
        assert_eq!(response.language.as_deref(), Some("en"));

        let captured = transport.take().expect("captured multipart request");
        assert_eq!(
            captured.url,
            "https://audio.fireworks.ai/v1/audio/transcriptions"
        );
        assert!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = String::from_utf8_lossy(&captured.body);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("whisper-v3"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
    }

    #[tokio::test]
    async fn openai_compatible_client_fireworks_stt_uses_explicit_base_url_override() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/v1/audio/transcriptions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"text":"hello from fireworks","language":"en"}"#)
            .create_async()
            .await;

        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            &format!("{}/v1", server.url()),
            make_fireworks_transcription_adapter(),
        )
        .with_model("whisper-v3");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
        request.model = Some("whisper-v3".to_string());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = AudioCapability::speech_to_text(&client, request)
            .await
            .expect("fireworks stt should succeed against overridden base url");

        assert_eq!(response.text, "hello from fireworks");
        assert_eq!(response.language.as_deref(), Some("en"));
    }

    #[tokio::test]
    async fn openai_compatible_client_together_tts_uses_default_audio_base_with_custom_transport() {
        let transport = BytesResponseTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_audio_adapter(),
        )
        .with_model("cartesia/sonic-2")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let request = TtsRequest::new("hello from together".to_string())
            .with_voice("alloy".to_string())
            .with_format("mp3".to_string());

        let response = AudioCapability::text_to_speech(&client, request)
            .await
            .expect("together tts should succeed through custom transport");

        assert_eq!(response.audio_data, vec![1, 2, 3, 4]);
        assert_eq!(response.format, "mp3");

        let captured = transport.take().expect("captured json request");
        assert_eq!(captured.url, "https://api.together.xyz/v1/audio/speech");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("cartesia/sonic-2")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!("hello from together")
        );
        assert_eq!(captured.body["voice"], serde_json::json!("alloy"));
        assert_eq!(captured.body["response_format"], serde_json::json!("mp3"));
    }

    #[tokio::test]
    async fn openai_compatible_client_together_tts_missing_model_uses_speech_family_default() {
        let transport = BytesResponseTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_audio_adapter(),
        )
        .with_model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let request = TtsRequest::new("hello from together".to_string())
            .with_voice("alloy".to_string())
            .with_format("mp3".to_string());

        let _ = AudioCapability::text_to_speech(&client, request)
            .await
            .expect("together tts should succeed through custom transport");

        let captured = transport.take().expect("captured json request");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("cartesia/sonic-2")
        );
    }

    #[tokio::test]
    async fn openai_compatible_client_together_stt_uses_default_audio_base_with_custom_transport() {
        let transport = MultipartResponseTransport::new(serde_json::json!({
            "text": "hello from together",
            "language": "en"
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_audio_adapter(),
        )
        .with_model("openai/whisper-large-v3")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
        request = request.with_media_type("audio/mpeg".to_string());

        let response = AudioCapability::speech_to_text(&client, request)
            .await
            .expect("together stt should succeed through custom multipart transport");

        assert_eq!(response.text, "hello from together");
        assert_eq!(response.language.as_deref(), Some("en"));

        let captured = transport.take().expect("captured multipart request");
        assert_eq!(
            captured.url,
            "https://api.together.xyz/v1/audio/transcriptions"
        );
        assert!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = String::from_utf8_lossy(&captured.body);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("openai/whisper-large-v3"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
    }

    #[tokio::test]
    async fn openai_compatible_client_fireworks_stt_missing_model_uses_transcription_family_default()
     {
        let transport = MultipartResponseTransport::new(serde_json::json!({
            "text": "hello from fireworks",
            "language": "en"
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            "https://api.fireworks.ai/inference/v1",
            make_fireworks_transcription_adapter(),
        )
        .with_model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");

        let _ = AudioCapability::speech_to_text(&client, request)
            .await
            .expect("fireworks stt should succeed through custom multipart transport");

        let captured = transport.take().expect("captured multipart request");
        let body_text = String::from_utf8_lossy(&captured.body);

        assert_eq!(
            captured.url,
            "https://audio.fireworks.ai/v1/audio/transcriptions"
        );
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("whisper-v3"));
    }

    #[tokio::test]
    async fn openai_compatible_client_siliconflow_stt_uses_default_audio_base_with_custom_transport()
     {
        let transport = MultipartResponseTransport::new(serde_json::json!({
            "text": "hello from siliconflow",
            "language": "zh"
        }));

        let cfg = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            make_siliconflow_audio_adapter(),
        )
        .with_model("FunAudioLLM/SenseVoiceSmall")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
        request.model = Some("FunAudioLLM/SenseVoiceSmall".to_string());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = AudioCapability::speech_to_text(&client, request)
            .await
            .expect("siliconflow stt should succeed through custom multipart transport");

        assert_eq!(response.text, "hello from siliconflow");
        assert_eq!(response.language.as_deref(), Some("zh"));

        let captured = transport.take().expect("captured multipart request");
        assert_eq!(
            captured.url,
            "https://api.siliconflow.cn/v1/audio/transcriptions"
        );
        assert!(
            captured
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = String::from_utf8_lossy(&captured.body);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("FunAudioLLM/SenseVoiceSmall"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
    }

    #[tokio::test]
    async fn openai_compatible_client_siliconflow_tts_uses_default_audio_base_with_custom_transport()
     {
        let transport = BytesResponseTransport::new(vec![9, 8, 7, 6], "audio/wav");

        let cfg = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            make_siliconflow_audio_adapter(),
        )
        .with_model("FunAudioLLM/CosyVoice2-0.5B")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let request = TtsRequest::new("hello from siliconflow".to_string())
            .with_model("FunAudioLLM/CosyVoice2-0.5B".to_string())
            .with_voice("alloy".to_string())
            .with_format("wav".to_string());

        let response = AudioCapability::text_to_speech(&client, request)
            .await
            .expect("siliconflow tts should succeed through custom transport");

        assert_eq!(response.audio_data, vec![9, 8, 7, 6]);
        assert_eq!(response.format, "wav");

        let captured = transport.take().expect("captured json request");
        assert_eq!(captured.url, "https://api.siliconflow.cn/v1/audio/speech");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("FunAudioLLM/CosyVoice2-0.5B")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!("hello from siliconflow")
        );
        assert_eq!(captured.body["voice"], serde_json::json!("alloy"));
        assert_eq!(captured.body["response_format"], serde_json::json!("wav"));
    }

    #[tokio::test]
    async fn text_to_speech_runtime_appends_query_params_to_transport_url() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "together".to_string(),
                name: "Together".to_string(),
                base_url: "https://api.together.xyz/v1".to_string(),
                field_mappings: ProviderFieldMappings::default(),
                capabilities: vec![
                    "chat".to_string(),
                    "audio".to_string(),
                    "speech".to_string(),
                ],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            })),
        )
        .with_model("cartesia/sonic-2")
        .with_query_param("api-version", "2025-04-01")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = TtsRequest::new("hello from together".to_string())
            .with_model("cartesia/sonic-2".to_string())
            .with_voice("alloy".to_string())
            .with_format("mp3".to_string());

        let _ = AudioCapability::text_to_speech(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.url,
            "https://api.together.xyz/v1/audio/speech?api-version=2025-04-01"
        );
    }

    #[tokio::test]
    async fn build_audio_executor_wires_openai_compatible_audio_spec() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "compat-audio",
            "test-key",
            "https://api.test.com/v1",
            make_audio_adapter(),
        )
        .with_model("gpt-audio-mini")
        .with_http_transport(Arc::new(transport));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap()
            .with_http_interceptors(vec![Arc::new(NoopInterceptor)]);

        let exec = client.build_audio_executor().await.unwrap();

        assert_eq!(exec.policy.interceptors.len(), 1);
        assert!(exec.policy.transport.is_some());
        assert!(exec.provider_spec.capabilities().supports("audio"));
        assert!(exec.provider_spec.capabilities().supports("speech"));
        assert!(exec.provider_spec.capabilities().supports("transcription"));
        assert_eq!(exec.provider_context.provider_id, "compat-audio");
        assert_eq!(exec.provider_context.base_url, "https://api.test.com/v1");
        assert_eq!(exec.transformer.provider_id(), "compat-audio");
        assert_eq!(exec.transformer.tts_endpoint(), "/audio/speech");
        assert_eq!(exec.transformer.stt_endpoint(), "/audio/transcriptions");
    }

    #[test]
    fn audio_logic_stays_out_of_monolithic_client_module() {
        let source = include_str!("../openai_client.rs");
        for forbidden in [
            "fn resolve_speech_model_default(",
            "fn resolve_transcription_model_default(",
            "fn build_audio_executor(",
            "impl AudioCapability for OpenAiCompatibleClient",
            "AudioExecutorBuilder::new(",
        ] {
            assert!(
                !source.contains(forbidden),
                "OpenAI-compatible audio logic should live in openai_client/audio.rs"
            );
        }
    }
}
