#![cfg(feature = "openai")]

use siumai::extensions::ModerationCapability;
use siumai::extensions::types::ModerationRequest;
use siumai::prelude::unified::Siumai;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_moderation_defaults_to_omni_model() {
    let server = MockServer::start().await;

    let expected = serde_json::json!({
        "model": "omni-moderation-latest",
        "input": "hello"
    });

    Mock::given(method("POST"))
        .and(path("/v1/moderations"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(body_json(expected))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "modr_123",
            "model": "omni-moderation-latest",
            "results": [{
                "flagged": false,
                "categories": { "hate": false },
                "category_scores": { "hate": 0.0 }
            }]
        })))
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

    let resp = ModerationCapability::moderate(
        &client,
        ModerationRequest {
            input: "hello".to_string(),
            inputs: None,
            model: None,
        },
    )
    .await
    .expect("moderate ok");

    assert_eq!(resp.model, "omni-moderation-latest");
    assert_eq!(resp.results.len(), 1);
    assert!(!resp.results[0].flagged);
}

#[tokio::test]
async fn openai_moderation_accepts_array_input() {
    let server = MockServer::start().await;

    let expected = serde_json::json!({
        "model": "omni-moderation-latest",
        "input": ["a", "b"]
    });

    Mock::given(method("POST"))
        .and(path("/v1/moderations"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(body_json(expected))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "modr_456",
            "model": "omni-moderation-latest",
            "results": [
                { "flagged": false, "categories": { "hate": false }, "category_scores": { "hate": 0.0 } },
                { "flagged": true,  "categories": { "hate": true },  "category_scores": { "hate": 0.9 } }
            ]
        })))
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

    let resp = ModerationCapability::moderate(
        &client,
        ModerationRequest {
            input: "".to_string(),
            inputs: Some(vec!["a".to_string(), "b".to_string()]),
            model: None,
        },
    )
    .await
    .expect("moderate ok");

    assert_eq!(resp.results.len(), 2);
    assert!(resp.results[1].flagged);
}
