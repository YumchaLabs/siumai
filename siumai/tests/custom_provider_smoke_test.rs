//! Smoke test to ensure custom provider API remains usable after refactor.
//! The test defines a minimal custom provider entirely in-process (no network),
//! then exercises chat() and chat_stream() paths.

use async_trait::async_trait;
use futures::{StreamExt, stream};
use siumai::ProviderCapabilities;
use siumai::custom_provider::*;
use siumai::error::LlmError;
use siumai::streaming::{ChatStream, ChatStreamEvent};
use siumai::traits::ChatCapability;
use siumai::types::*;

#[derive(Clone)]
struct MiniProvider {
    name: String,
    models: Vec<String>,
}

impl MiniProvider {
    fn new() -> Self {
        Self {
            name: "mini".to_string(),
            models: vec!["mini-1".to_string()],
        }
    }
}

#[async_trait]
impl CustomProvider for MiniProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn supported_models(&self) -> Vec<String> {
        self.models.clone()
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat().with_streaming()
    }

    async fn chat(&self, request: CustomChatRequest) -> Result<CustomChatResponse, LlmError> {
        let prompt = request
            .messages
            .first()
            .and_then(|m| m.content_text())
            .unwrap_or("");
        Ok(CustomChatResponse::new(format!("echo:{prompt}")))
    }

    async fn chat_stream(&self, request: CustomChatRequest) -> Result<ChatStream, LlmError> {
        let prompt = request
            .messages
            .first()
            .and_then(|m| m.content_text())
            .unwrap_or("");
        let seq = vec![
            Ok(ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: None,
                    model: None,
                    created: None,
                    provider: "mini".to_string(),
                    request_id: None,
                },
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: "echo:".to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: prompt.to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: None,
                    content: MessageContent::Text(String::new()),
                    model: None,
                    usage: None,
                    finish_reason: Some(FinishReason::Stop),
                    tool_calls: None,
                    thinking: None,
                    metadata: std::collections::HashMap::new(),
                },
            }),
        ];
        Ok(Box::pin(stream::iter(seq)))
    }
}

#[test]
fn custom_provider_config_builds() {
    let cfg = CustomProviderConfig::new("mini", "http://local", "k").with_model("mini-1");
    assert_eq!(cfg.model.as_deref(), Some("mini-1"));
}

#[tokio::test]
async fn custom_provider_chat_and_stream_smoke() -> Result<(), LlmError> {
    let cfg = CustomProviderConfig::new("mini", "http://local", "k").with_model("mini-1");
    let provider = Box::new(MiniProvider::new());
    let client = CustomProviderClient::new(provider, cfg)?;

    // chat()
    let resp = client.chat(vec![ChatMessage::user("ping").build()]).await?;
    assert_eq!(resp.content_text(), Some("echo:ping"));

    // chat_stream()
    let mut stream = client
        .chat_stream(vec![ChatMessage::user("pong").build()], None)
        .await?;
    let mut acc = String::new();
    while let Some(ev) = stream.next().await {
        if let ChatStreamEvent::ContentDelta { delta, .. } = ev? {
            acc.push_str(&delta);
        }
    }
    assert_eq!(acc, "echo:pong");
    Ok(())
}
