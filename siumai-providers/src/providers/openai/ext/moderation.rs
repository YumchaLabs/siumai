//! OpenAI Moderations extension helpers (multi-modal input).
//!
//! This module intentionally sits outside the Vercel-aligned unified surface.
//! The unified `ModerationRequest` is text-only by design; OpenAI supports
//! multi-modal moderation inputs (text + image URLs) that don't map cleanly
//! across providers.

use crate::error::LlmError;
use crate::types::ModerationResponse;

/// OpenAI moderation input union (multi-modal).
///
/// OpenAI accepts `input` as either:
/// - a single string
/// - an array of strings
/// - an array of multi-modal objects (`type: "text"` / `type: "image_url"`)
///
/// This extension focuses on the multi-modal object form.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiModerationInput {
    /// Text input (`type: "text"`).
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL or data URL (`type: "image_url"`).
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OpenAiModerationImageUrl },
}

impl OpenAiModerationInput {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: OpenAiModerationImageUrl { url: url.into() },
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAiModerationImageUrl {
    pub url: String,
}

/// OpenAI multi-modal Moderations request (provider-specific).
#[derive(Debug, Clone)]
pub struct OpenAiModerationRequest {
    pub input: Vec<OpenAiModerationInput>,
    pub model: Option<String>,
}

impl OpenAiModerationRequest {
    pub fn new(input: Vec<OpenAiModerationInput>) -> Self {
        Self { input, model: None }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Execute an OpenAI multi-modal moderation request.
pub async fn moderate_multimodal(
    client: &crate::providers::openai::OpenAiClient,
    request: OpenAiModerationRequest,
) -> Result<ModerationResponse, LlmError> {
    client.moderate_multimodal(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::{OpenAiClient, OpenAiConfig};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn openai_multimodal_moderation_sends_object_array_input() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/moderations"))
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

        let cfg = OpenAiConfig::new("KEY").with_base_url(format!("{}/v1", server.uri()));
        let client = OpenAiClient::new(cfg, reqwest::Client::new());

        let req = OpenAiModerationRequest::new(vec![
            OpenAiModerationInput::text("hello"),
            OpenAiModerationInput::image_url("https://example.com/image.jpg"),
        ]);

        let resp = moderate_multimodal(&client, req).await.unwrap();
        assert_eq!(resp.model, "omni-moderation-latest");
        assert_eq!(resp.results.len(), 1);

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let json: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(json["model"], "omni-moderation-latest");
        let input = json["input"].as_array().expect("input array");
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["type"], "text");
        assert_eq!(input[0]["text"], "hello");
        assert_eq!(input[1]["type"], "image_url");
        assert_eq!(input[1]["image_url"]["url"], "https://example.com/image.jpg");
    }
}
