#![cfg(feature = "openai")]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::Provider;
use siumai::experimental::client::LlmClient;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
    HttpTransportStreamResponse,
};
use siumai::prelude::unified::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
enum MockResponse {
    Json(serde_json::Value),
    Bytes(Vec<u8>, &'static str),
}

#[derive(Clone, Default)]
struct SequenceTransport {
    requests: Arc<Mutex<Vec<HttpTransportRequest>>>,
    responses: Arc<Mutex<HashMap<String, VecDeque<MockResponse>>>>,
}

impl SequenceTransport {
    fn push_json(&self, url: impl Into<String>, value: serde_json::Value) {
        self.responses
            .lock()
            .expect("lock responses")
            .entry(url.into())
            .or_default()
            .push_back(MockResponse::Json(value));
    }

    fn push_bytes(&self, url: impl Into<String>, bytes: Vec<u8>, content_type: &'static str) {
        self.responses
            .lock()
            .expect("lock responses")
            .entry(url.into())
            .or_default()
            .push_back(MockResponse::Bytes(bytes, content_type));
    }

    fn take_requests(&self) -> Vec<HttpTransportRequest> {
        std::mem::take(&mut *self.requests.lock().expect("lock requests"))
    }
}

#[async_trait]
impl HttpTransport for SequenceTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        self.requests
            .lock()
            .expect("lock requests")
            .push(request.clone());

        let response = self
            .responses
            .lock()
            .expect("lock responses")
            .get_mut(&request.url)
            .and_then(VecDeque::pop_front)
            .ok_or_else(|| {
                LlmError::InternalError(format!("missing mocked response for {}", request.url))
            })?;

        let mut headers = HeaderMap::new();
        match response {
            MockResponse::Json(value) => {
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                Ok(HttpTransportResponse {
                    status: 200,
                    headers,
                    body: serde_json::to_vec(&value).expect("serialize json"),
                })
            }
            MockResponse::Bytes(bytes, content_type) => {
                headers.insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
                Ok(HttpTransportResponse {
                    status: 200,
                    headers,
                    body: bytes,
                })
            }
        }
    }

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        self.requests
            .lock()
            .expect("lock requests")
            .push(request.clone());

        let response = self
            .responses
            .lock()
            .expect("lock responses")
            .get_mut(&request.url)
            .and_then(VecDeque::pop_front)
            .ok_or_else(|| {
                LlmError::InternalError(format!("missing mocked response for {}", request.url))
            })?;

        let mut headers = HeaderMap::new();
        match response {
            MockResponse::Json(value) => {
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                Ok(HttpTransportStreamResponse {
                    status: 200,
                    headers,
                    body: HttpTransportStreamBody::from_bytes(
                        serde_json::to_vec(&value).expect("serialize json"),
                    ),
                })
            }
            MockResponse::Bytes(bytes, content_type) => {
                headers.insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
                Ok(HttpTransportStreamResponse {
                    status: 200,
                    headers,
                    body: HttpTransportStreamBody::from_bytes(bytes),
                })
            }
        }
    }
}

fn header_value(request: &HttpTransportRequest, key: &str) -> Option<String> {
    request
        .headers
        .get(key)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string)
}

#[tokio::test]
async fn fireworks_public_builder_exposes_unified_capabilities() {
    let client = Provider::fireworks()
        .api_key("test-key")
        .model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        .build()
        .await
        .expect("build fireworks client");

    let caps = client.capabilities();
    assert!(caps.supports("chat"));
    assert!(caps.supports("completion"));
    assert!(caps.supports("embedding"));
    assert!(caps.supports("image_generation"));
    assert!(caps.supports("transcription"));
    assert!(caps.supports("audio"));
    assert!(!caps.supports("speech"));
    assert!(client.as_completion_capability().is_some());
    assert!(client.as_embedding_capability().is_some());
    assert!(client.as_image_generation_capability().is_some());
    assert!(client.as_image_extras().is_some());
    assert!(client.as_transcription_capability().is_some());
    assert!(client.as_speech_capability().is_none());
}

#[tokio::test]
#[allow(deprecated)]
async fn fireworks_public_builder_chat_typed_options_normalize_to_wire_shape() {
    use siumai::provider_ext::fireworks::options::*;

    let base_url = "https://example.com/fireworks/inference/v1";
    let chat_url = format!("{base_url}/chat/completions");
    let model = "accounts/fireworks/models/llama-v3p1-8b-instruct";
    let response_json = serde_json::json!({
        "id": "chatcmpl-fireworks-typed-options",
        "object": "chat.completion",
        "created": 1_718_345_013u64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "ok"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5
        }
    });

    let siumai_transport = SequenceTransport::default();
    siumai_transport.push_json(&chat_url, response_json.clone());
    let provider_transport = SequenceTransport::default();
    provider_transport.push_json(&chat_url, response_json);

    let siumai_client = Siumai::builder()
        .fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai fireworks client");

    let provider_client = Provider::fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider fireworks client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_provider_option(
            "fireworks",
            serde_json::json!({ "reasoningEffort": "minimal" }),
        )
        .with_fireworks_options(
            FireworksChatOptions::new()
                .with_thinking(
                    FireworksThinkingConfig::new()
                        .with_type(FireworksThinkingType::Enabled)
                        .with_budget_tokens(2048),
                )
                .with_reasoning_history(FireworksReasoningHistory::Interleaved),
        );

    let siumai_response = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai chat ok");
    let provider_response = provider_client
        .chat_request(request)
        .await
        .expect("provider chat ok");

    assert_eq!(siumai_response.text(), Some("ok".to_string()));
    assert_eq!(provider_response.text(), Some("ok".to_string()));

    let requests = siumai_transport.take_requests();
    let provider_requests = provider_transport.take_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(provider_requests.len(), 1);

    let siumai_request = &requests[0];
    let provider_request = &provider_requests[0];

    assert_eq!(siumai_request.url, chat_url);
    assert_eq!(siumai_request.url, provider_request.url);
    assert_eq!(siumai_request.body, provider_request.body);
    assert_eq!(
        header_value(siumai_request, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(siumai_request.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_request.body["reasoning_effort"],
        serde_json::json!("low")
    );
    assert_eq!(
        siumai_request.body["reasoning_history"],
        serde_json::json!("interleaved")
    );
    assert_eq!(
        siumai_request.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 2048
        })
    );
    assert!(siumai_request.body.get("reasoningHistory").is_none());
    assert!(
        siumai_request.body["thinking"]
            .get("budgetTokens")
            .is_none()
    );
}

#[tokio::test]
async fn fireworks_public_builder_routes_image_generation_to_provider_owned_default_model() {
    let base_url = "https://example.com/fireworks/inference/v1";
    let image_url =
        format!("{base_url}/workflows/accounts/fireworks/models/flux-1-dev-fp8/text_to_image");
    let transport = SequenceTransport::default();
    transport.push_bytes(&image_url, b"fireworks-image".to_vec(), "image/png");

    let client = Provider::fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build fireworks client");

    let _response = client
        .as_image_generation_capability()
        .expect("image capability")
        .generate_images(ImageGenerationRequest {
            prompt: "a tiny blue robot".to_string(),
            count: 1,
            ..Default::default()
        })
        .await
        .expect("image generation ok");

    let requests = transport.take_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, image_url);
    assert_eq!(
        header_value(&requests[0], "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        requests[0].body["prompt"],
        serde_json::json!("a tiny blue robot")
    );
    assert_eq!(requests[0].body["samples"], serde_json::json!(1));
}

#[tokio::test]
async fn fireworks_public_builder_routes_image_edit_to_async_kontext_path() {
    let base_url = "https://example.com/fireworks/inference/v1";
    let submit_url = format!("{base_url}/workflows/accounts/fireworks/models/flux-kontext-pro");
    let poll_url = format!("{submit_url}/get_result");
    let transport = SequenceTransport::default();
    transport.push_json(&submit_url, serde_json::json!({ "request_id": "req-123" }));
    transport.push_json(
        &poll_url,
        serde_json::json!({
            "status": "Ready",
            "result": {
                "sample": "data:image/png;base64,aGVsbG8="
            }
        }),
    );

    let client = Provider::fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build fireworks client");

    let mut provider_options = ProviderOptionsMap::default();
    provider_options.insert(
        "fireworks",
        serde_json::json!({
            "output_format": "jpeg"
        }),
    );

    let response = client
        .as_image_extras()
        .expect("image extras")
        .edit_image(siumai::extensions::types::ImageEditRequest {
            images: vec![
                siumai::extensions::types::ImageEditInput::file_with_media_type(
                    vec![137, 80, 78, 71],
                    "image/png",
                ),
            ],
            mask: None,
            prompt: "edit this image".to_string(),
            model: Some("accounts/fireworks/models/flux-kontext-pro".to_string()),
            count: Some(1),
            size: None,
            aspect_ratio: Some("16:9".to_string()),
            seed: None,
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: provider_options,
            http_config: None,
        })
        .await
        .expect("image edit ok");

    let requests = transport.take_requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].url, submit_url);
    assert_eq!(requests[1].url, poll_url);
    assert_eq!(
        requests[0].body["prompt"],
        serde_json::json!("edit this image")
    );
    assert_eq!(requests[0].body["aspect_ratio"], serde_json::json!("16:9"));
    assert_eq!(requests[0].body["samples"], serde_json::json!(1));
    assert_eq!(requests[0].body["output_format"], serde_json::json!("jpeg"));
    assert_eq!(
        requests[0].body["input_image"],
        serde_json::json!("data:image/png;base64,iVBORw==")
    );
    assert_eq!(requests[1].body["id"], serde_json::json!("req-123"));
    assert_eq!(response.images[0].b64_json.as_deref(), Some("aGVsbG8="));
}

#[tokio::test]
#[allow(deprecated)]
async fn fireworks_public_builder_routes_completion_to_openai_compat_completions() {
    let base_url = "https://example.com/fireworks/inference/v1";
    let completion_url = format!("{base_url}/completions");
    let model = "accounts/fireworks/models/llama-v3p1-8b-instruct";

    let response_json = serde_json::json!({
        "id": "cmpl-fireworks-test",
        "object": "text_completion",
        "created": 1_718_345_013u64,
        "model": model,
        "choices": [
            {
                "text": "done",
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9
        }
    });

    let siumai_transport = SequenceTransport::default();
    siumai_transport.push_json(&completion_url, response_json.clone());
    let provider_transport = SequenceTransport::default();
    provider_transport.push_json(&completion_url, response_json);

    let siumai_client = Siumai::builder()
        .fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai fireworks client");

    let provider_client = Provider::fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider fireworks client");

    let request = CompletionRequest::from_prompt(vec![
        ChatMessage::system("Be terse.").build(),
        ChatMessage::user("Hello").build(),
        ChatMessage::assistant("Hi").build(),
        ChatMessage::user("Continue").build(),
    ])
    .with_model(model)
    .with_provider_option("fireworks", serde_json::json!({ "suffix": "!" }));

    let siumai_response = siumai_client
        .as_completion_capability()
        .expect("siumai completion capability")
        .complete(request.clone())
        .await
        .expect("siumai completion ok");
    let provider_response = provider_client
        .as_completion_capability()
        .expect("provider completion capability")
        .complete(request)
        .await
        .expect("provider completion ok");

    assert_eq!(siumai_response.text(), "done");
    assert_eq!(provider_response.text(), "done");

    let requests = siumai_transport.take_requests();
    let provider_requests = provider_transport.take_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(provider_requests.len(), 1);

    let siumai_request = &requests[0];
    let provider_request = &provider_requests[0];

    assert_eq!(siumai_request.url, completion_url);
    assert_eq!(siumai_request.url, provider_request.url);
    assert_eq!(siumai_request.body, provider_request.body);
    assert_eq!(
        header_value(siumai_request, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(siumai_request.body["model"], serde_json::json!(model));
    assert_eq!(siumai_request.body["suffix"], serde_json::json!("!"));
    assert_eq!(
        siumai_request.body["prompt"],
        serde_json::json!(
            "Be terse.\n\nuser:\nHello\n\nassistant:\nHi\n\nuser:\nContinue\n\nassistant:\n"
        )
    );
    assert_eq!(siumai_request.body["stop"], serde_json::json!(["\nuser:"]));
}

#[tokio::test]
#[allow(deprecated)]
async fn fireworks_public_builder_completion_stream_keeps_raw_chunks_runtime_only() {
    use futures_util::StreamExt;

    let base_url = "https://example.com/fireworks/inference/v1";
    let completion_url = format!("{base_url}/completions");
    let model = "accounts/fireworks/models/llama-v3p1-8b-instruct";
    let stream_body = concat!(
        "data: {\"id\":\"cmpl-fireworks-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"accounts/fireworks/models/llama-v3p1-8b-instruct\",\"choices\":[{\"text\":\"hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"cmpl-fireworks-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"accounts/fireworks/models/llama-v3p1-8b-instruct\",\"choices\":[{\"text\":\" world\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
        "data: [DONE]\n\n"
    )
    .as_bytes()
    .to_vec();

    let siumai_transport = SequenceTransport::default();
    siumai_transport.push_bytes(&completion_url, stream_body.clone(), "text/event-stream");
    let provider_transport = SequenceTransport::default();
    provider_transport.push_bytes(&completion_url, stream_body, "text/event-stream");

    let siumai_client = Siumai::builder()
        .fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai fireworks client");

    let provider_client = Provider::fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider fireworks client");

    let request = CompletionRequest::new("hi")
        .with_model(model)
        .with_include_raw_chunks(true);

    let mut siumai_stream = siumai_client
        .as_completion_capability()
        .expect("siumai completion capability")
        .complete_stream(request.clone())
        .await
        .expect("siumai completion stream ok");
    let mut provider_stream = provider_client
        .as_completion_capability()
        .expect("provider completion capability")
        .complete_stream(request)
        .await
        .expect("provider completion stream ok");

    while siumai_stream.next().await.is_some() {}
    while provider_stream.next().await.is_some() {}

    let requests = siumai_transport.take_requests();
    let provider_requests = provider_transport.take_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(provider_requests.len(), 1);

    let siumai_request = &requests[0];
    let provider_request = &provider_requests[0];

    assert_eq!(siumai_request.url, completion_url);
    assert_eq!(siumai_request.url, provider_request.url);
    assert_eq!(siumai_request.body, provider_request.body);
    assert_eq!(
        header_value(siumai_request, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        header_value(siumai_request, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_request.body["stream"], serde_json::json!(true));
    assert!(siumai_request.body.get("stream_options").is_none());
    assert!(siumai_request.body.get("includeRawChunks").is_none());
}
