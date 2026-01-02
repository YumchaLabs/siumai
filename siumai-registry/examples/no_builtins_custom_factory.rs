use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use futures::{StreamExt, stream};

use siumai_registry::LlmClient;
use siumai_registry::error::LlmError;
use siumai_registry::registry::entry::{ProviderFactory, create_provider_registry};
use siumai_registry::streaming::ChatStream;
use siumai_registry::traits::{ChatCapability, ProviderCapabilities};
use siumai_registry::types::{
    ChatMessage, ChatResponse, ChatStreamEvent, FinishReason, MessageContent, ResponseMetadata,
    Tool,
};

#[derive(Clone)]
struct MockClient {
    provider: &'static str,
    model_id: String,
}

#[async_trait::async_trait]
impl ChatCapability for MockClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let last = messages.last().and_then(|m| m.content_text()).unwrap_or("");

        let mut response = ChatResponse::new(MessageContent::Text(format!(
            "echo({}): {}",
            self.model_id, last
        )));
        response.model = Some(self.model_id.clone());
        response.finish_reason = Some(FinishReason::Stop);
        Ok(response)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let response = self.chat_with_tools(messages, tools).await?;
        let response_text = response.content_text().unwrap_or("").to_string();

        let metadata = ResponseMetadata {
            id: None,
            model: response.model.clone(),
            created: None,
            provider: self.provider.to_string(),
            request_id: None,
        };

        Ok(Box::pin(stream::iter([
            Ok(ChatStreamEvent::StreamStart { metadata }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: response_text,
                index: Some(0),
            }),
            Ok(ChatStreamEvent::StreamEnd { response }),
        ])))
    }
}

impl LlmClient for MockClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(self.provider)
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.model_id.clone()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat().with_streaming()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

struct MockFactory;

#[async_trait::async_trait]
impl ProviderFactory for MockFactory {
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(MockClient {
            provider: "mock",
            model_id: model_id.to_string(),
        }))
    }

    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("mock")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat().with_streaming()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut providers: HashMap<String, Arc<dyn ProviderFactory>> = HashMap::new();
    providers.insert("mock".to_string(), Arc::new(MockFactory));

    let registry = create_provider_registry(providers, None);
    let model = registry.language_model("mock:demo")?;

    let response = model
        .chat(vec![ChatMessage::user("Hello from registry!").build()])
        .await?;
    println!("chat: {}", response.content_text().unwrap_or(""));

    let mut stream = model
        .chat_stream(vec![ChatMessage::user("Stream this!").build()], None)
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ContentDelta { delta, .. } => println!("stream delta: {delta}"),
            ChatStreamEvent::StreamEnd { .. } => println!("stream end"),
            _ => {}
        }
    }

    Ok(())
}
