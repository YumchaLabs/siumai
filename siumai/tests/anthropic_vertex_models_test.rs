use serde_json::json;
use siumai::providers::anthropic_vertex::client::{VertexAnthropicClient, VertexAnthropicConfig};
use siumai::traits::ModelListingCapability;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn anthropic_vertex_list_models_parses_ids() {
    let server = MockServer::start().await;

    let body = json!({
        "models": [
            {
                "name": format!("projects/p/locations/l/publishers/anthropic/models/{}", "claude-3-7-sonnet-latest"),
                "displayName": "Claude 3.7 Sonnet"
            },
            {
                "name": format!("projects/p/locations/l/publishers/anthropic/models/{}", "claude-3-5-haiku-latest")
            }
        ]
    });

    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let cfg = VertexAnthropicConfig {
        base_url: server.uri(),
        model: "claude-3-7-sonnet-latest".to_string(),
        http_config: siumai::types::HttpConfig {
            headers: std::iter::once(("Authorization".to_string(), "Bearer ok".to_string()))
                .collect(),
            ..Default::default()
        },
    };
    let cli = VertexAnthropicClient::new(cfg, reqwest::Client::new());
    let models = ModelListingCapability::list_models(&cli).await.unwrap();
    let ids: Vec<String> = models.into_iter().map(|m| m.id).collect();
    assert!(ids.contains(&"claude-3-7-sonnet-latest".to_string()));
    assert!(ids.contains(&"claude-3-5-haiku-latest".to_string()));
}
