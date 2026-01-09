#![cfg(feature = "google-vertex")]

use async_trait::async_trait;
use reqwest::header::HeaderMap;
use serde_json::json;
use siumai::experimental::core::ProviderContext;
use siumai::experimental::execution::executors::embedding::{
    EmbeddingExecutor, EmbeddingExecutorBuilder,
};
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use siumai::prelude::unified::{EmbeddingRequest, LlmError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct RecordingTransport {
    calls: Arc<Mutex<Vec<HttpTransportRequest>>>,
}

impl RecordingTransport {
    fn calls(&self) -> Vec<HttpTransportRequest> {
        self.calls.lock().expect("lock").clone()
    }
}

#[async_trait]
impl HttpTransport for RecordingTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        self.calls.lock().expect("lock").push(request.clone());

        let body = json!({
            "predictions": [
                { "embeddings": { "values": [0.1, 0.2, 0.3], "statistics": { "token_count": 1 } } },
                { "embeddings": { "values": [0.4, 0.5, 0.6], "statistics": { "token_count": 1 } } }
            ]
        });

        Ok(HttpTransportResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: serde_json::to_vec(&body).expect("serialize"),
        })
    }
}

#[tokio::test]
async fn vertex_embedding_uses_custom_transport_and_passes_request_content() {
    let transport = RecordingTransport::default();
    let transport_arc: Arc<dyn HttpTransport> = Arc::new(transport.clone());

    let ctx = ProviderContext::new(
        "vertex",
        "https://custom-endpoint.com".to_string(),
        None,
        HashMap::new(),
    );
    let spec = Arc::new(
        siumai::experimental::providers::google_vertex::standards::vertex_embedding::VertexEmbeddingStandard::new()
            .create_spec("vertex"),
    );

    let req = EmbeddingRequest::new(vec!["test text one".into(), "test text two".into()])
        .with_model("textembedding-gecko@001")
        .with_provider_option("google", json!({ "outputDimensionality": 768 }));

    let exec = EmbeddingExecutorBuilder::new("vertex", reqwest::Client::new())
        .with_spec(spec)
        .with_context(ctx)
        .with_transport(transport_arc)
        .build_for_request(&req);

    let out = EmbeddingExecutor::execute(&*exec, req)
        .await
        .expect("execute ok");
    assert_eq!(out.embeddings.len(), 2);

    let calls = transport.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].url,
        "https://custom-endpoint.com/models/textembedding-gecko@001:predict"
    );
    assert_eq!(
        calls[0].body,
        json!({
            "instances": [
                { "content": "test text one" },
                { "content": "test text two" }
            ],
            "parameters": { "outputDimensionality": 768 }
        })
    );
}
