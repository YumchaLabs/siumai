use super::OpenAiCompatibleClient;
use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, Tool};
use async_trait::async_trait;
use std::sync::Arc;

impl OpenAiCompatibleClient {
    fn ensure_chat_surface(&self, stream: bool) -> Result<(), LlmError> {
        let caps = self.config.adapter.capabilities();
        if !caps.chat {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support chat",
                self.config.provider_id
            )));
        }
        if stream && !caps.streaming {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support chat streaming",
                self.config.provider_id
            )));
        }

        Ok(())
    }

    /// Build a chat executor with an explicit provider spec.
    pub async fn build_chat_executor_with_spec(
        &self,
        request: &ChatRequest,
        spec: Arc<dyn ProviderSpec>,
    ) -> Result<Arc<crate::execution::executors::chat::HttpChatExecutor>, LlmError> {
        let ctx = self.build_context().await?;
        let bundle = spec.choose_chat_transformers(request, &ctx);

        let mut builder =
            ChatExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
                .with_spec(spec)
                .with_context(ctx)
                .with_transformer_bundle(bundle)
                .with_runtime_transformer_selection()
                .with_stream_disable_compression(self.config.http_config.stream_disable_compression)
                .with_interceptors(self.http_interceptors.clone())
                .with_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        Ok(builder.build())
    }

    async fn build_chat_executor(
        &self,
        request: &ChatRequest,
    ) -> Result<Arc<crate::execution::executors::chat::HttpChatExecutor>, LlmError> {
        let spec = Arc::new(self.compat_spec());
        self.build_chat_executor_with_spec(request, spec).await
    }

    fn prepare_chat_request(
        &self,
        request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        self.ensure_chat_surface(stream)?;
        let request = crate::utils::chat_request::normalize_chat_request(
            request,
            crate::utils::chat_request::ChatRequestDefaults::new(&self.config.common_params)
                .with_http_config(&self.config.http_config),
            stream,
        );
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "OpenAI-compatible request requires a model".to_string(),
            ));
        }
        Ok(request)
    }

    /// Execute a non-stream chat via an explicit ProviderSpec.
    pub async fn chat_request_with_spec(
        &self,
        request: ChatRequest,
        spec: Arc<dyn ProviderSpec>,
    ) -> Result<ChatResponse, LlmError> {
        let request = self.prepare_chat_request(request, false)?;
        let exec = self.build_chat_executor_with_spec(&request, spec).await?;
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute a stream chat via an explicit ProviderSpec.
    pub async fn chat_stream_request_with_spec(
        &self,
        request: ChatRequest,
        spec: Arc<dyn ProviderSpec>,
    ) -> Result<ChatStream, LlmError> {
        let request = self.prepare_chat_request(request, true)?;
        let exec = self.build_chat_executor_with_spec(&request, spec).await?;
        ChatExecutor::execute_stream(&*exec, request).await
    }

    /// Execute a non-stream chat via ProviderSpec.
    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let exec = self.build_chat_executor(&request).await?;
        ChatExecutor::execute(&*exec, request).await
    }

    /// Execute a stream chat via ProviderSpec.
    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        let exec = self.build_chat_executor(&request).await?;
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

#[async_trait]
impl ChatCapability for OpenAiCompatibleClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.ensure_chat_surface(false)?;
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.config.common_params.clone())
            .http_config(self.config.http_config.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let req = builder.build();

        self.chat_request_via_spec(req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.ensure_chat_surface(true)?;
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.config.common_params.clone())
            .http_config(self.config.http_config.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        self.chat_stream_request_via_spec(request).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let request = self.prepare_chat_request(request, false)?;
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let request = self.prepare_chat_request(request, true)?;
        self.chat_stream_request_via_spec(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai_compatible::OpenAiCompatibleConfig;
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use crate::types::CommonParams;

    fn make_text_streaming_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "compat-chat".to_string(),
            name: "Compat Chat".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: Some("compat-default-model".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    #[tokio::test]
    async fn prepare_chat_request_for_stream_sets_stream_and_fills_defaults() {
        let cfg = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_text_streaming_adapter(),
        )
        .with_model("compat-default-model")
        .with_http_config(crate::defaults::http::config_default());
        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "compat-default-model");
        assert!(prepared.http_config.is_some());
    }

    #[tokio::test]
    async fn prepare_chat_request_for_non_stream_clears_stream_and_preserves_explicit_model() {
        let cfg = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_text_streaming_adapter(),
        )
        .with_model("compat-default-model")
        .with_http_config(crate::defaults::http::config_default());
        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .model("compat-explicit-model")
            .messages(vec![ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let prepared = client
            .prepare_chat_request(request, false)
            .expect("prepare non-stream request");

        assert!(!prepared.stream);
        assert_eq!(prepared.common_params.model, "compat-explicit-model");
        assert!(prepared.http_config.is_some());
    }

    #[tokio::test]
    async fn prepare_chat_request_merges_missing_common_params_and_http_config_defaults() {
        let mut cfg = OpenAiCompatibleConfig::new(
            "compat-chat",
            "test-key",
            "https://api.test.com/v1",
            make_text_streaming_adapter(),
        )
        .with_model("compat-default-model")
        .with_http_config(crate::defaults::http::config_default());
        cfg.common_params.temperature = Some(0.7);
        cfg.common_params.max_tokens = Some(256);
        cfg.common_params.top_p = Some(0.9);
        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(CommonParams {
                temperature: Some(0.2),
                ..Default::default()
            })
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "compat-default-model");
        assert_eq!(prepared.common_params.temperature, Some(0.2));
        assert_eq!(prepared.common_params.max_tokens, Some(256));
        assert_eq!(prepared.common_params.top_p, Some(0.9));
        assert!(prepared.http_config.is_some());
    }

    #[tokio::test]
    async fn build_chat_executor_exposes_runtime_provider_before_send_via_provider_spec() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg =
            OpenAiCompatibleConfig::new("deepseek", "test-key", "https://api.test.com/v1", adapter)
                .with_model("test-model");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap();

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_common_params(CommonParams {
                model: "test-model".to_string(),
                ..Default::default()
            })
            .with_provider_option("deepseek", serde_json::json!({ "my_custom": 1 }));

        let exec = client.build_chat_executor(&req).await.unwrap();
        assert!(exec.policy.before_send.is_none());
        assert!(
            exec.provider_spec
                .chat_before_send(&req, &exec.provider_context)
                .is_some()
        );
    }

    #[test]
    fn chat_execution_logic_stays_out_of_monolithic_client_module() {
        let source = include_str!("../openai_client.rs");
        for forbidden in [
            "fn ensure_chat_surface(",
            "fn prepare_chat_request(",
            "fn build_chat_executor(",
            "impl ChatCapability for OpenAiCompatibleClient",
        ] {
            assert!(
                !source.contains(forbidden),
                "OpenAI-compatible chat execution logic should live in openai_client/chat.rs"
            );
        }
    }
}
