#![cfg(feature = "anthropic")]

use serde::de::DeserializeOwned;
use siumai::extensions::ModelListingCapability;
use siumai::prelude::unified::Siumai;
use std::path::{Path, PathBuf};
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Match, Mock, MockServer, Request, ResponseTemplate};

#[derive(Debug, Clone, Copy)]
struct QueryParamAbsent(&'static str);

impl Match for QueryParamAbsent {
    fn matches(&self, request: &Request) -> bool {
        let key = self.0;
        !request.url.query_pairs().any(|(k, _)| k == key)
    }
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("anthropic")
        .join("models")
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let text = std::fs::read_to_string(path).expect("read fixture json");
    serde_json::from_str(&text).expect("parse fixture json")
}

#[tokio::test]
async fn anthropic_models_list_paginates_and_maps_fields() {
    let server = MockServer::start().await;

    let page_1: serde_json::Value = read_json(fixtures_dir().join("list.1.json"));
    let page_2: serde_json::Value = read_json(fixtures_dir().join("list.2.json"));

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(QueryParamAbsent("after_id"))
        .and(query_param("limit", "100"))
        .and(header("x-api-key", "test-api-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(page_1))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(query_param("after_id", "claude-3-7-sonnet-latest"))
        .and(query_param("limit", "100"))
        .and(header("x-api-key", "test-api-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(page_2))
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .anthropic()
        .api_key("test-api-key")
        .base_url(server.uri())
        .model("claude-3-7-sonnet-latest")
        .build()
        .await
        .expect("build ok");

    let models = ModelListingCapability::list_models(&client)
        .await
        .expect("list models ok");

    assert_eq!(models.len(), 2);

    let mut ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
    ids.sort();
    assert_eq!(
        ids,
        vec![
            "claude-3-5-haiku-latest".to_string(),
            "claude-3-7-sonnet-latest".to_string(),
        ]
    );

    let sonnet = models
        .iter()
        .find(|m| m.id == "claude-3-7-sonnet-latest")
        .expect("sonnet model present");
    assert_eq!(sonnet.name.as_deref(), Some("Claude 3.7 Sonnet"));
    assert_eq!(sonnet.owned_by, "anthropic");
    assert_eq!(sonnet.created, Some(1_735_689_600));
    assert!(sonnet.capabilities.contains(&"tools".to_string()));
    assert!(sonnet.capabilities.contains(&"streaming".to_string()));
    assert!(sonnet.capabilities.contains(&"vision".to_string()));
    assert!(sonnet.capabilities.contains(&"thinking".to_string()));

    let haiku = models
        .iter()
        .find(|m| m.id == "claude-3-5-haiku-latest")
        .expect("haiku model present");
    assert_eq!(haiku.name.as_deref(), Some("Claude 3.5 Haiku"));
    assert_eq!(haiku.owned_by, "anthropic");
    assert!(haiku.created.is_some());
    assert!(haiku.capabilities.contains(&"tools".to_string()));
    assert!(haiku.capabilities.contains(&"streaming".to_string()));
    assert!(haiku.capabilities.contains(&"vision".to_string()));
    assert!(!haiku.capabilities.contains(&"thinking".to_string()));
}

#[tokio::test]
async fn anthropic_models_get_model_uses_models_endpoint() {
    let server = MockServer::start().await;

    let body: serde_json::Value = read_json(fixtures_dir().join("get.json"));

    Mock::given(method("GET"))
        .and(path("/v1/models/claude-3-7-sonnet-latest"))
        .and(header("x-api-key", "test-api-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .anthropic()
        .api_key("test-api-key")
        .base_url(server.uri())
        .model("claude-3-7-sonnet-latest")
        .build()
        .await
        .expect("build ok");

    let model = ModelListingCapability::get_model(&client, "claude-3-7-sonnet-latest".to_string())
        .await
        .expect("get model ok");

    assert_eq!(model.id, "claude-3-7-sonnet-latest");
    assert_eq!(model.name.as_deref(), Some("Claude 3.7 Sonnet"));
    assert_eq!(model.owned_by, "anthropic");
    assert_eq!(model.created, Some(1_735_689_600));
    assert!(model.capabilities.contains(&"vision".to_string()));
    assert!(model.capabilities.contains(&"thinking".to_string()));
}

#[tokio::test]
async fn anthropic_models_base_url_with_v1_does_not_duplicate() {
    let server = MockServer::start().await;

    let page_1: serde_json::Value = read_json(fixtures_dir().join("list.single.json"));

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(QueryParamAbsent("after_id"))
        .and(query_param("limit", "100"))
        .and(header("x-api-key", "test-api-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(page_1))
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .anthropic()
        .api_key("test-api-key")
        .base_url(format!("{}/v1", server.uri()))
        .model("claude-3-7-sonnet-latest")
        .build()
        .await
        .expect("build ok");

    let models = ModelListingCapability::list_models(&client)
        .await
        .expect("list models ok");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "claude-3-7-sonnet-latest");
}
