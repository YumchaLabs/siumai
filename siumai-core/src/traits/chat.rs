//! Chat capability traits and extensions

use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::types::{ChatMessage, ChatRequest, ChatResponse, Tool};
use async_trait::async_trait;

#[async_trait]
pub trait ChatCapability: Send + Sync {
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools(messages, None).await
    }

    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError>;

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError>;

    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        let stream = self.chat_stream(messages, tools).await?;
        let (cancellable, cancel) = crate::utils::cancel::make_cancellable_stream(stream);
        Ok(ChatStreamHandle {
            stream: cancellable,
            cancel,
        })
    }

    /// Full chat request (preferred unified path). Default falls back to chat_with_tools.
    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools(request.messages, request.tools).await
    }

    /// Full streaming chat request (preferred unified path). Default falls back to chat_stream.
    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream(request.messages, request.tools).await
    }

    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        let stream = self.chat_stream_request(request).await?;
        let (cancellable, cancel) = crate::utils::cancel::make_cancellable_stream(stream);
        Ok(ChatStreamHandle {
            stream: cancellable,
            cancel,
        })
    }
}

#[async_trait]
pub trait ChatExtensions: ChatCapability {
    async fn chat_with_retry(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        options: crate::retry_api::RetryOptions,
    ) -> Result<ChatResponse, LlmError> {
        let msgs = messages;
        let tls = tools;
        crate::retry_api::retry_with(
            || {
                let m = msgs.clone();
                let t = tls.clone();
                async move { self.chat_with_tools(m, t).await }
            },
            options,
        )
        .await
    }

    async fn memory_contents(&self) -> Result<Option<Vec<ChatMessage>>, LlmError> {
        Ok(None)
    }

    async fn summarize_history(&self, messages: Vec<ChatMessage>) -> Result<String, LlmError> {
        let prompt = format!(
            "Summarize in 2-3 sentences:\n{}",
            messages
                .iter()
                .map(|m| format!("{:?}: {}", m.role, m.content_text().unwrap_or("")))
                .collect::<Vec<_>>()
                .join("\n")
        );
        let request_messages = vec![ChatMessage::user(prompt).build()];
        let response = self.chat(request_messages).await?;
        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in summary response".to_string()))
            .map(std::string::ToString::to_string)
    }

    async fn ask(&self, prompt: String) -> Result<String, LlmError> {
        let message = ChatMessage::user(prompt).build();
        let response = self.chat(vec![message]).await?;
        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))
            .map(std::string::ToString::to_string)
    }

    async fn ask_with_retry(
        &self,
        prompt: String,
        options: crate::retry_api::RetryOptions,
    ) -> Result<String, LlmError> {
        let message = ChatMessage::user(prompt).build();
        let response = self.chat_with_retry(vec![message], None, options).await?;
        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))
            .map(std::string::ToString::to_string)
    }

    async fn ask_with_system(
        &self,
        system_prompt: String,
        user_prompt: String,
    ) -> Result<String, LlmError> {
        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(user_prompt).build(),
        ];
        let response = self.chat(messages).await?;
        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))
            .map(std::string::ToString::to_string)
    }

    async fn continue_conversation(
        &self,
        mut conversation: Vec<ChatMessage>,
        new_message: String,
    ) -> Result<(String, Vec<ChatMessage>), LlmError> {
        conversation.push(ChatMessage::user(new_message).build());
        let response = self.chat(conversation.clone()).await?;
        let response_text = response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))?
            .to_string();
        conversation.push(ChatMessage::assistant(response_text.clone()).build());
        Ok((response_text, conversation))
    }

    async fn translate(&self, text: String, target_language: String) -> Result<String, LlmError> {
        let prompt = format!("Translate the following text to {target_language}: {text}");
        self.ask(prompt).await
    }

    async fn explain(&self, concept: String, audience: Option<String>) -> Result<String, LlmError> {
        let audience_str = audience
            .map(|a| format!(" to {a}"))
            .unwrap_or_else(|| " in simple terms".to_string());
        let prompt = format!("Explain {concept}{audience_str}");
        self.ask(prompt).await
    }

    async fn generate(&self, content_type: String, prompt: String) -> Result<String, LlmError> {
        let system_prompt = format!(
            "You are a creative writer. Generate a {content_type} based on the user's prompt."
        );
        self.ask_with_system(system_prompt, prompt).await
    }
}

impl<T: ChatCapability> ChatExtensions for T {}
