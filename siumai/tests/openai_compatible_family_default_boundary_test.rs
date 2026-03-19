#![cfg(feature = "openai")]
#![allow(deprecated)]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
};
use siumai::extensions::AudioCapability;
use siumai::prelude::unified::*;
use siumai_core::types::EmbeddingFormat;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct CaptureTransport {
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl CaptureTransport {
    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock request").take()
    }
}

#[async_trait]
impl HttpTransport for CaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 401,
            headers,
            body:
                br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                    .to_vec(),
        })
    }
}

#[derive(Clone)]
struct BinaryCaptureTransport {
    response_body: Arc<Vec<u8>>,
    response_content_type: &'static str,
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl BinaryCaptureTransport {
    fn new(response_body: Vec<u8>, response_content_type: &'static str) -> Self {
        Self {
            response_body: Arc::new(response_body),
            response_content_type,
            last: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock binary request").take()
    }
}

#[async_trait]
impl HttpTransport for BinaryCaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock binary request") = Some(request);

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

#[derive(Clone)]
struct MultipartCaptureTransport {
    response_body: Arc<Vec<u8>>,
    last: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
}

impl MultipartCaptureTransport {
    fn new(response: serde_json::Value) -> Self {
        Self {
            response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
            last: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<HttpTransportMultipartRequest> {
        self.last.lock().expect("lock multipart request").take()
    }
}

#[async_trait]
impl HttpTransport for MultipartCaptureTransport {
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
        *self.last.lock().expect("lock multipart request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: self.response_body.as_ref().clone(),
        })
    }
}

fn normalize_multipart_body(request: &HttpTransportMultipartRequest) -> String {
    String::from_utf8_lossy(&request.body).replace("\r\n", "\n")
}

fn builtin_primary_default_model(provider_id: &str) -> String {
    siumai::provider_ext::openai_compatible::get_provider_config(provider_id)
        .and_then(|config| config.default_model)
        .unwrap_or_else(|| panic!("missing builtin default model for {provider_id}"))
}

async fn make_config_client(
    provider_id: &str,
    model: &str,
    transport: Arc<dyn HttpTransport>,
) -> siumai::provider_ext::openai_compatible::OpenAiCompatibleClient {
    let provider = siumai::provider_ext::openai_compatible::get_provider_config(provider_id)
        .expect("builtin provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );

    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        provider_id,
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model(model)
    .with_http_transport(transport);

    siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
        .await
        .expect("build config client")
}

#[tokio::test]
async fn compat_together_embedding_missing_request_model_uses_family_default_across_public_paths() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .together()
        .api_key("test-key")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .together()
        .api_key("test-key")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "together",
        &builtin_primary_default_model("together"),
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = EmbeddingRequest::single("hello together embedding")
        .with_dimensions(384)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-family-default");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("captured siumai request");
    let provider_req = provider_transport
        .take()
        .expect("captured provider request");
    let config_req = config_transport.take().expect("captured config request");

    for req in [&siumai_req, &provider_req, &config_req] {
        assert_eq!(req.url, "https://api.together.xyz/v1/embeddings");
        assert_eq!(
            req.body["model"],
            serde_json::json!("togethercomputer/m2-bert-80M-8k-retrieval")
        );
        assert_eq!(
            req.body["input"],
            serde_json::json!(["hello together embedding"])
        );
        assert_eq!(req.body["dimensions"], serde_json::json!(384));
        assert_eq!(req.body["encoding_format"], serde_json::json!("float"));
        assert_eq!(req.body["user"], serde_json::json!("compat-family-default"));
    }
}

#[tokio::test]
async fn compat_together_embedding_missing_request_model_preserves_explicit_config_override_across_public_paths()
 {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .together()
        .api_key("test-key")
        .model("custom-embedding-override")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .together()
        .api_key("test-key")
        .model("custom-embedding-override")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "together",
        "custom-embedding-override",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = EmbeddingRequest::single("hello together embedding");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("captured siumai request");
    let provider_req = provider_transport
        .take()
        .expect("captured provider request");
    let config_req = config_transport.take().expect("captured config request");

    for req in [&siumai_req, &provider_req, &config_req] {
        assert_eq!(req.url, "https://api.together.xyz/v1/embeddings");
        assert_eq!(
            req.body["model"],
            serde_json::json!("custom-embedding-override")
        );
    }
}

#[tokio::test]
async fn compat_together_tts_missing_request_model_uses_family_default_across_public_paths() {
    let siumai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let provider_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let config_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

    let siumai_client = Siumai::builder()
        .openai()
        .together()
        .api_key("test-key")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .together()
        .api_key("test-key")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "together",
        &builtin_primary_default_model("together"),
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = TtsRequest::new("hello from together".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());

    let _ = siumai_client
        .text_to_speech(request.clone())
        .await
        .expect("siumai tts ok");
    let _ = provider_client
        .text_to_speech(request.clone())
        .await
        .expect("provider tts ok");
    let _ = config_client
        .text_to_speech(request)
        .await
        .expect("config tts ok");

    let siumai_req = siumai_transport.take().expect("captured siumai request");
    let provider_req = provider_transport
        .take()
        .expect("captured provider request");
    let config_req = config_transport.take().expect("captured config request");

    for req in [&siumai_req, &provider_req, &config_req] {
        assert_eq!(req.url, "https://api.together.xyz/v1/audio/speech");
        assert_eq!(req.body["model"], serde_json::json!("cartesia/sonic-2"));
        assert_eq!(req.body["input"], serde_json::json!("hello from together"));
        assert_eq!(req.body["voice"], serde_json::json!("alloy"));
        assert_eq!(req.body["response_format"], serde_json::json!("mp3"));
    }
}

#[tokio::test]
async fn compat_fireworks_stt_missing_request_model_uses_family_default_across_public_paths() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from fireworks",
        "language": "en"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from fireworks",
        "language": "en"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from fireworks",
        "language": "en"
    }));

    let siumai_client = Siumai::builder()
        .openai()
        .fireworks()
        .api_key("test-key")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "fireworks",
        &builtin_primary_default_model("fireworks"),
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = SttRequest::from_audio(b"abc".to_vec()).with_media_type("audio/mpeg".to_string());

    let _ = siumai_client
        .speech_to_text(request.clone())
        .await
        .expect("siumai stt ok");
    let _ = provider_client
        .speech_to_text(request.clone())
        .await
        .expect("provider stt ok");
    let _ = config_client
        .speech_to_text(request)
        .await
        .expect("config stt ok");

    let siumai_req = siumai_transport.take().expect("captured siumai request");
    let provider_req = provider_transport
        .take()
        .expect("captured provider request");
    let config_req = config_transport.take().expect("captured config request");

    for req in [&siumai_req, &provider_req, &config_req] {
        assert_eq!(
            req.url,
            "https://audio.fireworks.ai/v1/audio/transcriptions"
        );

        let body_text = normalize_multipart_body(req);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("whisper-v3"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
    }
}

#[tokio::test]
async fn compat_together_image_missing_request_model_uses_family_default_across_public_paths() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .together()
        .api_key("test-key")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .together()
        .api_key("test-key")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "together",
        &builtin_primary_default_model("together"),
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = ImageGenerationRequest {
        prompt: "a tiny purple robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        count: 1,
        model: None,
        quality: None,
        style: None,
        seed: None,
        steps: None,
        guidance_scale: None,
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    let _ = siumai_client.generate_images(request.clone()).await;
    let _ = provider_client.generate_images(request.clone()).await;
    let _ = config_client.generate_images(request).await;

    let siumai_req = siumai_transport.take().expect("captured siumai request");
    let provider_req = provider_transport
        .take()
        .expect("captured provider request");
    let config_req = config_transport.take().expect("captured config request");

    for req in [&siumai_req, &provider_req, &config_req] {
        assert_eq!(req.url, "https://api.together.xyz/v1/images/generations");
        assert_eq!(
            req.body["model"],
            serde_json::json!("black-forest-labs/FLUX.1-schnell")
        );
        assert_eq!(req.body["prompt"], serde_json::json!("a tiny purple robot"));
        assert_eq!(req.body["size"], serde_json::json!("1024x1024"));
        assert_eq!(req.body["n"], serde_json::json!(1));
        assert_eq!(req.body["response_format"], serde_json::json!("url"));
    }
}

#[tokio::test]
async fn compat_jina_rerank_missing_request_model_uses_family_default_across_public_paths() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .jina()
        .api_key("test-key")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .jina()
        .api_key("test-key")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "jina",
        &builtin_primary_default_model("jina"),
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = RerankRequest::new(
        String::new(),
        "query".to_string(),
        vec!["doc-1".to_string(), "doc-2".to_string()],
    )
    .with_top_n(1);

    let _ = siumai_client.rerank(request.clone()).await;
    let _ = provider_client.rerank(request.clone()).await;
    let _ = config_client.rerank(request).await;

    let siumai_req = siumai_transport.take().expect("captured siumai request");
    let provider_req = provider_transport
        .take()
        .expect("captured provider request");
    let config_req = config_transport.take().expect("captured config request");

    for req in [&siumai_req, &provider_req, &config_req] {
        assert_eq!(req.url, "https://api.jina.ai/v1/rerank");
        assert_eq!(req.body["model"], serde_json::json!("jina-reranker-m0"));
        assert_eq!(req.body["query"], serde_json::json!("query"));
        assert_eq!(req.body["documents"], serde_json::json!(["doc-1", "doc-2"]));
        assert_eq!(req.body["top_n"], serde_json::json!(1));
    }
}
