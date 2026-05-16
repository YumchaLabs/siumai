#![cfg(feature = "deepinfra")]
#![allow(deprecated)]

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::compat::Provider;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    HttpTransportStreamBody, HttpTransportStreamResponse,
};
use siumai::prelude::compat::Siumai;
use siumai::prelude::unified::{
    ChatCapability, ChatMessage, ChatRequest, ChatStreamEvent, ChatStreamPart, LlmError,
};
use siumai::provider_ext::deepinfra::{DeepInfraClient, DeepInfraConfig};
use siumai::registry::ProviderBuildOverrides;
use siumai_core::builder::BuilderBase;
use siumai_provider_openai_compatible::providers::openai_compatible::{
    ConfigurableAdapter, OpenAiCompatibleBuilder, ResponseMetadataExtractor, get_provider_config,
};
use siumai_registry::registry::builder::RegistryBuilder;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct StreamCaptureTransport {
    response_body: Arc<Vec<u8>>,
    last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl StreamCaptureTransport {
    fn new(response_body: Vec<u8>) -> Self {
        Self {
            response_body: Arc::new(response_body),
            last_stream: Arc::new(Mutex::new(None)),
        }
    }

    fn take_stream(&self) -> Option<HttpTransportRequest> {
        self.last_stream.lock().expect("lock stream request").take()
    }
}

#[async_trait]
impl HttpTransport for StreamCaptureTransport {
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

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        *self.last_stream.lock().expect("lock stream request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

        Ok(HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_bytes(self.response_body.as_ref().clone()),
        })
    }

    async fn execute_multipart(
        &self,
        _request: HttpTransportMultipartRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 501,
            headers,
            body: br#"{"error":{"message":"multipart unsupported in test","type":"test_error","code":"unsupported"}}"#
                .to_vec(),
        })
    }
}

fn header_value(req: &HttpTransportRequest, key: &str) -> Option<String> {
    req.headers
        .get(key)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string)
}

fn assert_requests_equivalent(left: &HttpTransportRequest, right: &HttpTransportRequest) {
    assert_eq!(left.url, right.url);
    assert_eq!(
        header_value(left, "authorization"),
        header_value(right, "authorization")
    );
    assert_eq!(header_value(left, "accept"), header_value(right, "accept"));
    assert_eq!(left.body, right.body);
}

fn make_chat_request(model: &str) -> ChatRequest {
    let mut request = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    request.common_params.model = model.to_string();
    request
}

fn make_registry(
    base_url: &str,
    transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    let mut providers = std::collections::HashMap::new();
    providers.insert(
        "deepinfra".to_string(),
        Arc::new(siumai::registry::factories::DeepInfraProviderFactory)
            as Arc<dyn siumai::registry::ProviderFactory>,
    );

    RegistryBuilder::new(providers)
        .with_api_key("test-key")
        .with_base_url(base_url)
        .fetch(transport.clone())
        .with_provider_build_overrides(
            "deepinfra",
            ProviderBuildOverrides::default()
                .with_api_key("test-key")
                .with_base_url(base_url)
                .fetch(transport),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry")
}

async fn make_config_client(
    base_url: &str,
    model: &str,
    transport: Arc<dyn HttpTransport>,
    extractor: Option<Arc<dyn ResponseMetadataExtractor>>,
) -> DeepInfraClient {
    let mut provider = get_provider_config("deepinfra").expect("deepinfra provider config");
    provider.base_url = base_url.to_string();
    let adapter = Arc::new(ConfigurableAdapter::new(provider));

    let mut config = DeepInfraConfig::new("deepinfra", "test-key", base_url, adapter)
        .with_model(model)
        .with_http_transport(transport);
    if let Some(extractor) = extractor {
        config = config.with_metadata_extractor(extractor);
    }

    DeepInfraClient::from_config(config)
        .await
        .expect("build config client")
}

async fn collect_ok_events<C>(client: &C, request: ChatRequest) -> Vec<ChatStreamEvent>
where
    C: ChatCapability + Sync,
{
    client
        .chat_stream_request(request)
        .await
        .expect("stream ok")
        .map(|event| event.expect("stream event"))
        .collect::<Vec<_>>()
        .await
}

fn assert_finish_and_stream_end_metadata(events: &[ChatStreamEvent], expected: &str) {
    let finish_provider_metadata = events
        .iter()
        .find_map(|event| match event {
            ChatStreamEvent::Part {
                part:
                    ChatStreamPart::Finish {
                        provider_metadata, ..
                    },
            } => provider_metadata.clone(),
            _ => None,
        })
        .expect("finish provider metadata");
    assert_eq!(
        finish_provider_metadata
            .get("deepinfra")
            .and_then(|value| value.get("value")),
        Some(&serde_json::json!(expected))
    );

    let stream_end_provider_metadata = events
        .iter()
        .find_map(|event| match event {
            ChatStreamEvent::StreamEnd { response } => response.provider_metadata.clone(),
            _ => None,
        })
        .expect("stream-end provider metadata");
    assert_eq!(
        stream_end_provider_metadata
            .get("deepinfra")
            .and_then(|value| value.get("value")),
        Some(&serde_json::json!(expected))
    );
}

#[tokio::test]
async fn deepinfra_chat_stream_include_raw_chunks_stays_runtime_only_across_public_paths() {
    let base_url = "https://example.com/deepinfra/v1";
    let model = "meta-llama/Llama-3.3-70B-Instruct";
    let done_body = b"data: [DONE]\n\n".to_vec();

    let builder_transport = StreamCaptureTransport::new(done_body.clone());
    let siumai_transport = StreamCaptureTransport::new(done_body.clone());
    let provider_transport = StreamCaptureTransport::new(done_body.clone());
    let config_transport = StreamCaptureTransport::new(done_body.clone());
    let registry_transport = StreamCaptureTransport::new(done_body);

    let builder_client = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepinfra")
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build provider-owned builder client");

    let siumai_client = Siumai::builder()
        .deepinfra()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepinfra()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client(base_url, model, Arc::new(config_transport.clone()), None).await;

    let registry = make_registry(base_url, Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("deepinfra:{model}"))
        .expect("build registry model");

    let request = make_chat_request(model).with_include_raw_chunks(true);

    let _ = collect_ok_events(&builder_client, request.clone()).await;
    let _ = collect_ok_events(&siumai_client, request.clone()).await;
    let _ = collect_ok_events(&provider_client, request.clone()).await;
    let _ = collect_ok_events(&config_client, request.clone()).await;
    let _ = collect_ok_events(&registry_model, request).await;

    let builder_req = builder_transport.take_stream().expect("builder request");
    let siumai_req = siumai_transport.take_stream().expect("siumai request");
    let provider_req = provider_transport.take_stream().expect("provider request");
    let config_req = config_transport.take_stream().expect("config request");
    let registry_req = registry_transport.take_stream().expect("registry request");

    assert_requests_equivalent(&builder_req, &siumai_req);
    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);

    assert_eq!(
        siumai_req.url,
        "https://example.com/deepinfra/v1/openai/chat/completions"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert!(siumai_req.body.get("includeRawChunks").is_none());
    assert!(
        siumai_req
            .body
            .get("stream_options")
            .and_then(|value| value.get("includeRawChunks"))
            .is_none()
    );
}

#[tokio::test]
async fn deepinfra_siumai_chat_stream_include_raw_chunks_emits_raw_before_response_metadata() {
    let base_url = "https://example.com/deepinfra/v1";
    let model = "meta-llama/Llama-3.3-70B-Instruct";
    let transport = StreamCaptureTransport::new(
        br#"data: {"id":"chat_1","model":"meta-llama/Llama-3.3-70B-Instruct","choices":[{"index":0,"delta":{"content":"Hello"}}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

"#
        .to_vec(),
    );

    let client = Siumai::builder()
        .deepinfra()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(transport))
        .build()
        .await
        .expect("build siumai client");

    let events = collect_ok_events(
        &client,
        make_chat_request(model).with_include_raw_chunks(true),
    )
    .await;

    let raw_pos = events
        .iter()
        .position(|event| {
            matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::Raw { raw_value }
                } if raw_value["id"] == serde_json::json!("chat_1")
            )
        })
        .expect("raw part");
    let metadata_pos = events
        .iter()
        .position(|event| {
            matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ResponseMetadata(metadata)
                } if metadata.id.as_deref() == Some("chat_1")
            )
        })
        .expect("response metadata part");

    assert!(
        raw_pos < metadata_pos,
        "public stream should emit raw before response metadata when includeRawChunks is enabled"
    );
}

#[tokio::test]
async fn deepinfra_provider_owned_builder_and_config_merge_stream_metadata_extractor() {
    let base_url = "https://example.com/deepinfra/v1";
    let model = "meta-llama/Llama-3.3-70B-Instruct";
    let sse_body = br#"data: {"id":"chat_2","model":"meta-llama/Llama-3.3-70B-Instruct","choices":[{"index":0,"delta":{"content":"Hello"}}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"test_field":"test_value"}

data: [DONE]

"#
    .to_vec();
    let extractor: Arc<dyn ResponseMetadataExtractor> = Arc::new(|raw: &serde_json::Value| {
        raw.get("test_field").map(|value| {
            std::collections::HashMap::from([(
                "deepinfra".to_string(),
                serde_json::json!({ "value": value }),
            )])
        })
    });

    let builder_transport = StreamCaptureTransport::new(sse_body.clone());
    let config_transport = StreamCaptureTransport::new(sse_body);

    let builder_client = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepinfra")
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_metadata_extractor(extractor.clone())
        .with_http_transport(Arc::new(builder_transport))
        .build()
        .await
        .expect("build builder client");

    let config_client =
        make_config_client(base_url, model, Arc::new(config_transport), Some(extractor)).await;

    let request = make_chat_request(model);

    let builder_events = collect_ok_events(&builder_client, request.clone()).await;
    let config_events = collect_ok_events(&config_client, request).await;

    assert_finish_and_stream_end_metadata(&builder_events, "test_value");
    assert_finish_and_stream_end_metadata(&config_events, "test_value");
}
