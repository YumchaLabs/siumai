#![cfg(feature = "google-vertex")]
#![allow(deprecated)]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use siumai::prelude::unified::registry::{RegistryOptions, create_provider_registry};
use siumai::prelude::unified::*;
use siumai::provider_ext::google_vertex::{
    GoogleVertexClient, GoogleVertexConfig, VertexChatResponseExt, VertexContentPartExt,
};
use siumai::registry::ProviderBuildOverrides;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct JsonCaptureTransport {
    response_body: Arc<Vec<u8>>,
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl JsonCaptureTransport {
    fn new(response: serde_json::Value) -> Self {
        Self {
            response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
            last: Arc::new(Mutex::new(None)),
        }
    }
}

#[async_trait]
impl HttpTransport for JsonCaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: self.response_body.as_ref().clone(),
        })
    }
}

fn fixture_response(case: &str) -> serde_json::Value {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vertex")
        .join("chat")
        .join(case)
        .join("response.json");
    let text = std::fs::read_to_string(&root).expect("read fixture response");
    serde_json::from_str(&text).expect("parse fixture response")
}

fn make_registry(transport: Arc<dyn HttpTransport>) -> siumai::registry::ProviderRegistryHandle {
    let mut providers = std::collections::HashMap::new();
    providers.insert(
        "vertex".to_string(),
        Arc::new(siumai::registry::factories::GoogleVertexProviderFactory)
            as Arc<dyn siumai::prelude::unified::registry::ProviderFactory>,
    );

    let mut provider_build_overrides = std::collections::HashMap::new();
    provider_build_overrides.insert(
        "vertex".to_string(),
        ProviderBuildOverrides::default()
            .with_api_key("test-key")
            .fetch(transport),
    );

    create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: Vec::new(),
            http_client: None,
            http_transport: None,
            http_config: None,
            api_key: None,
            base_url: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            provider_build_overrides,
            retry_options: None,
            max_cache_entries: None,
            client_ttl: None,
            auto_middleware: true,
        }),
    )
}

#[tokio::test]
async fn google_vertex_metadata_ext_reads_vertex_namespace_across_public_paths() {
    let response_json = fixture_response("google-thought-signature-text-and-reasoning.1");
    let siumai_transport = JsonCaptureTransport::new(response_json.clone());
    let provider_transport = JsonCaptureTransport::new(response_json.clone());
    let config_transport = JsonCaptureTransport::new(response_json);
    let registry_transport = JsonCaptureTransport::new(fixture_response(
        "google-thought-signature-text-and-reasoning.1",
    ));

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .model("gemini-pro")
        .fetch(Arc::new(siumai_transport))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = siumai::Provider::vertex()
        .api_key("test-key")
        .model("gemini-pro")
        .fetch(Arc::new(provider_transport))
        .build()
        .expect("build provider client");

    let config_client = GoogleVertexClient::from_config(
        GoogleVertexConfig::express("test-key", "gemini-pro")
            .with_http_transport(Arc::new(config_transport)),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport));
    let registry_model = registry
        .language_model("vertex:gemini-pro")
        .expect("build registry language model");

    for response in [
        siumai_client
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("siumai chat"),
        provider_client
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("provider chat"),
        config_client
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("config chat"),
        registry_model
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("registry chat"),
    ] {
        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected provider metadata");
        assert!(provider_metadata.contains_key("vertex"));
        assert!(!provider_metadata.contains_key("google"));

        let typed = response.vertex_metadata().expect("typed vertex metadata");
        assert_eq!(
            typed
                .usage_metadata
                .as_ref()
                .and_then(|usage| usage.total_token_count),
            Some(30)
        );
        let first_rating = typed
            .safety_ratings
            .as_ref()
            .and_then(|ratings| ratings.first())
            .expect("expected first safety rating");
        assert_eq!(
            serde_json::to_value(first_rating)
                .expect("serialize safety rating")
                .get("category")
                .cloned(),
            Some(serde_json::json!("HARM_CATEGORY_DEROGATORY"))
        );

        let MessageContent::MultiModal(parts) = &response.content else {
            panic!("expected multimodal content");
        };
        assert_eq!(parts.len(), 3);

        assert_eq!(
            parts[0]
                .vertex_metadata()
                .and_then(|meta| meta.thought_signature),
            Some("sig1".to_string())
        );
        assert_eq!(
            parts[1]
                .vertex_metadata()
                .and_then(|meta| meta.thought_signature),
            Some("sig2".to_string())
        );
        assert_eq!(
            parts[2]
                .vertex_metadata()
                .and_then(|meta| meta.thought_signature),
            Some("sig3".to_string())
        );
    }
}

#[tokio::test]
async fn google_vertex_metadata_ext_reads_grounding_sources_and_url_context() {
    let response_json = serde_json::json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Grounded answer from Vertex."
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "groundingMetadata": {
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.com/article",
                                "title": "Example Article"
                            }
                        },
                        {
                            "retrievedContext": {
                                "uri": "gs://vertex-bucket/manual.pdf",
                                "title": "Manual PDF",
                                "text": "manual body"
                            }
                        },
                        {
                            "retrievedContext": {
                                "fileSearchStore": "stores/docs/report.txt",
                                "title": "Stored Report",
                                "text": "store body"
                            }
                        }
                    ],
                    "webSearchQueries": [
                        "vertex grounding"
                    ],
                    "retrievalMetadata": {
                        "googleSearchDynamicRetrievalScore": 0.87
                    }
                },
                "urlContextMetadata": {
                    "urlMetadata": [
                        {
                            "retrievedUrl": "https://example.com/context",
                            "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 18,
            "totalTokenCount": 30
        },
        "modelVersion": "gemini-pro"
    });
    let siumai_transport = JsonCaptureTransport::new(response_json.clone());
    let provider_transport = JsonCaptureTransport::new(response_json.clone());
    let config_transport = JsonCaptureTransport::new(response_json);
    let registry_transport = JsonCaptureTransport::new(serde_json::json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Grounded answer from Vertex."
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "groundingMetadata": {
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.com/article",
                                "title": "Example Article"
                            }
                        },
                        {
                            "retrievedContext": {
                                "uri": "gs://vertex-bucket/manual.pdf",
                                "title": "Manual PDF",
                                "text": "manual body"
                            }
                        },
                        {
                            "retrievedContext": {
                                "fileSearchStore": "stores/docs/report.txt",
                                "title": "Stored Report",
                                "text": "store body"
                            }
                        }
                    ],
                    "webSearchQueries": [
                        "vertex grounding"
                    ],
                    "retrievalMetadata": {
                        "googleSearchDynamicRetrievalScore": 0.87
                    }
                },
                "urlContextMetadata": {
                    "urlMetadata": [
                        {
                            "retrievedUrl": "https://example.com/context",
                            "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 18,
            "totalTokenCount": 30
        },
        "modelVersion": "gemini-pro"
    }));

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .model("gemini-pro")
        .fetch(Arc::new(siumai_transport))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = siumai::Provider::vertex()
        .api_key("test-key")
        .model("gemini-pro")
        .fetch(Arc::new(provider_transport))
        .build()
        .expect("build provider client");

    let config_client = GoogleVertexClient::from_config(
        GoogleVertexConfig::express("test-key", "gemini-pro")
            .with_http_transport(Arc::new(config_transport)),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport));
    let registry_model = registry
        .language_model("vertex:gemini-pro")
        .expect("build registry language model");

    for response in [
        siumai_client
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("siumai chat"),
        provider_client
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("provider chat"),
        config_client
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("config chat"),
        registry_model
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("registry chat"),
    ] {
        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected provider metadata");
        assert!(provider_metadata.contains_key("vertex"));
        assert!(!provider_metadata.contains_key("google"));

        let typed = response.vertex_metadata().expect("typed vertex metadata");
        assert_eq!(
            typed
                .grounding_metadata
                .as_ref()
                .and_then(|meta| meta.web_search_queries.as_ref())
                .and_then(|queries| queries.first())
                .map(String::as_str),
            Some("vertex grounding")
        );
        assert_eq!(
            typed
                .url_context_metadata
                .as_ref()
                .and_then(|meta| meta.url_metadata.as_ref())
                .and_then(|items| items.first())
                .and_then(|item| item.retrieved_url.as_deref()),
            Some("https://example.com/context")
        );

        let sources = typed.sources.as_ref().expect("expected typed sources");
        assert_eq!(sources.len(), 3);

        let url_source = sources
            .iter()
            .find(|source| source.source_type == "url")
            .expect("expected url source");
        assert_eq!(
            url_source.url.as_deref(),
            Some("https://example.com/article")
        );
        assert_eq!(url_source.title.as_deref(), Some("Example Article"));

        let pdf_source = sources
            .iter()
            .find(|source| source.filename.as_deref() == Some("manual.pdf"))
            .expect("expected pdf document source");
        assert_eq!(pdf_source.source_type, "document");
        assert_eq!(pdf_source.media_type.as_deref(), Some("application/pdf"));
        assert_eq!(pdf_source.title.as_deref(), Some("Manual PDF"));

        let store_source = sources
            .iter()
            .find(|source| source.filename.as_deref() == Some("report.txt"))
            .expect("expected file-search-store source");
        assert_eq!(store_source.source_type, "document");
        assert_eq!(
            store_source.media_type.as_deref(),
            Some("application/octet-stream")
        );
        assert_eq!(store_source.title.as_deref(), Some("Stored Report"));
    }
}
