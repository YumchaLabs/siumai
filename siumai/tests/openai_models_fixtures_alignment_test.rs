#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use siumai::extensions::ModelListingCapability;
use siumai::prelude::unified::Siumai;
use std::path::{Path, PathBuf};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("models")
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let text = std::fs::read_to_string(path).expect("read fixture json");
    serde_json::from_str(&text).expect("parse fixture json")
}

#[tokio::test]
async fn openai_models_list_and_get_use_models_endpoint() {
    let server = MockServer::start().await;

    let list_body: serde_json::Value = read_json(fixtures_dir().join("list.json"));
    let get_body: serde_json::Value = read_json(fixtures_dir().join("get.json"));

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(list_body))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/models/gpt-4o"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(get_body))
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .openai()
        .api_key("test-api-key")
        .base_url(format!("{}/v1", server.uri()))
        .model("gpt-4o")
        .build()
        .await
        .expect("build ok");

    let models = ModelListingCapability::list_models(&client)
        .await
        .expect("list models ok");
    assert_eq!(models.len(), 2);
    assert!(models.iter().any(|m| m.id == "gpt-4o"));

    let model = ModelListingCapability::get_model(&client, "gpt-4o".to_string())
        .await
        .expect("get model ok");
    assert_eq!(model.id, "gpt-4o");
    assert_eq!(model.owned_by, "openai");
    assert_eq!(model.created, Some(1710000000));
    assert!(model.capabilities.contains(&"chat".to_string()));
}
