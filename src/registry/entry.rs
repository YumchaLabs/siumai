//! Experimental Provider Registry entry (Iteration A)
//!
//! This is a minimal skeleton intended to provide a
//! typed place for the future registry-as-entrypoint. It focuses on parsing
//! `provider:model` identifiers and holding model-level middleware options.

use std::collections::HashMap;
use std::sync::Arc;

use crate::builder::LlmBuilder;
use crate::client::LlmClient;
use crate::error::LlmError;
use crate::middleware::language_model::LanguageModelMiddleware;
#[cfg(feature = "xai")]
use crate::prelude::quick_xai_with_model;
use crate::stream::ChatStream;
#[cfg(test)]
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, Tool};

/// Options for creating a provider registry handle.
pub struct RegistryOptions {
    pub separator: char,
    pub language_model_middleware: Vec<Arc<dyn LanguageModelMiddleware>>,
}

/// Minimal provider registry handle.
///
/// Iteration A: only provides parsing and stores model-level middlewares for
/// future use. Client construction is intentionally not implemented here to
/// avoid breaking existing builders.
pub struct ProviderRegistryHandle {
    separator: char,
    pub(crate) middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl ProviderRegistryHandle {
    /// Split a registry model id like "provider:model" into (provider, model).
    pub fn split_id(&self, id: &str) -> Result<(String, String), LlmError> {
        if let Some((p, m)) = id.split_once(self.separator) {
            if p.is_empty() || m.is_empty() {
                return Err(LlmError::InvalidParameter(format!(
                    "Invalid model id for registry: {} (must be 'provider{}model')",
                    id, self.separator
                )));
            }
            Ok((p.to_string(), m.to_string()))
        } else {
            Err(LlmError::InvalidParameter(format!(
                "Invalid model id for registry: {} (must be 'provider{}model')",
                id, self.separator
            )))
        }
    }

    /// Experimental API: resolve language model in future iterations.
    /// Returns a minimal language model handle with chat APIs.
    pub fn language_model(&self, id: &str) -> Result<LanguageModelHandle, LlmError> {
        let (provider, model) = self.split_id(id)?;
        Ok(LanguageModelHandle {
            provider_id: provider,
            model_id: model,
            middlewares: self.middlewares.clone(),
        })
    }

    /// Experimental API: resolve embedding model in future iterations.
    pub fn embedding_model(&self, id: &str) -> Result<EmbeddingModelHandle, LlmError> {
        let (provider, model) = self.split_id(id)?;
        Ok(EmbeddingModelHandle {
            provider_id: provider,
            model_id: model,
        })
    }

    /// Experimental API: resolve image model.
    pub fn image_model(&self, id: &str) -> Result<ImageModelHandle, LlmError> {
        let (provider, model) = self.split_id(id)?;
        Ok(ImageModelHandle {
            provider_id: provider,
            model_id: model,
        })
    }

    /// Experimental API: resolve speech model (TTS/STT).
    pub fn speech_model(&self, id: &str) -> Result<SpeechModelHandle, LlmError> {
        let (provider, model) = self.split_id(id)?;
        Ok(SpeechModelHandle {
            provider_id: provider,
            model_id: model,
        })
    }

    /// Experimental API: resolve transcription model (STT alias).
    pub fn transcription_model(&self, id: &str) -> Result<TranscriptionModelHandle, LlmError> {
        let (provider, model) = self.split_id(id)?;
        Ok(TranscriptionModelHandle {
            provider_id: provider,
            model_id: model,
        })
    }
}

/// Create a provider registry handle (experimental, Iteration A)
pub fn create_provider_registry(
    _providers: HashMap<String, ()>,
    opts: Option<RegistryOptions>,
) -> ProviderRegistryHandle {
    let (separator, middlewares) = if let Some(o) = opts {
        (o.separator, o.language_model_middleware)
    } else {
        (':', Vec::new())
    };
    ProviderRegistryHandle {
        separator,
        middlewares,
    }
}

/// Minimal language model handle.
///
/// Chat methods build a provider client on-demand using quick_* helpers and
/// attach registry-level middlewares before executing the request. This avoids
/// introducing stateful clients at registry level and keeps behavior explicit.
#[derive(Clone)]
pub struct LanguageModelHandle {
    pub provider_id: String,
    pub model_id: String,
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>, // applied before provider mapping
}

impl LanguageModelHandle {
    /// Build a client for this language model handle.
    /// Creates a new client instance on each call - client creation is cheap (no network I/O).
    async fn build_client(&self) -> Result<Arc<dyn LlmClient>, LlmError> {
        let built: Arc<dyn LlmClient> = match self.provider_id.as_str() {
            #[cfg(test)]
            "testprov" => {
                TEST_BUILD_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Arc::new(TestProvClient)
            }
            #[cfg(feature = "openai")]
            "openai" => {
                let client = crate::quick_openai_with_model(&self.model_id).await?;
                let client = client.with_model_middlewares(self.middlewares.clone());
                Arc::new(client)
            }
            #[cfg(feature = "google")]
            "gemini" => {
                let client = crate::quick_gemini_with_model(&self.model_id).await?;
                let client = client.with_model_middlewares(self.middlewares.clone());
                Arc::new(client)
            }
            #[cfg(feature = "anthropic")]
            "anthropic" => {
                let client = crate::quick_anthropic_with_model(&self.model_id).await?;
                let client = client.with_model_middlewares(self.middlewares.clone());
                Arc::new(client)
            }
            #[cfg(feature = "groq")]
            "groq" => {
                let client = crate::quick_groq_with_model(&self.model_id).await?;
                let client = client.with_model_middlewares(self.middlewares.clone());
                Arc::new(client)
            }
            #[cfg(feature = "xai")]
            "xai" => {
                let client = quick_xai_with_model(&self.model_id).await?;
                let client = client.with_model_middlewares(self.middlewares.clone());
                Arc::new(client)
            }
            // OpenAI-compatible (selected common providers)
            #[cfg(feature = "openai")]
            "openrouter" => {
                let client = LlmBuilder::new()
                    .openrouter()
                    .model(&self.model_id)
                    .build()
                    .await?;
                let client = client.with_model_middlewares(self.middlewares.clone());
                Arc::new(client)
            }
            #[cfg(feature = "openai")]
            "deepseek" => {
                let client = LlmBuilder::new()
                    .deepseek()
                    .model(&self.model_id)
                    .build()
                    .await?;
                let client = client.with_model_middlewares(self.middlewares.clone());
                Arc::new(client)
            }
            other => {
                #[cfg(feature = "openai")]
                {
                    if crate::providers::openai_compatible::config::get_provider_config(other)
                        .is_some()
                    {
                        let mut b =
                            crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                                LlmBuilder::new(),
                                other,
                            );
                        let env_key = format!("{}_API_KEY", other.to_uppercase());
                        let api_key = std::env::var(&env_key).map_err(|_| {
                            LlmError::ConfigurationError(format!(
                                "Missing {} for OpenAI-compatible provider {}",
                                env_key, other
                            ))
                        })?;
                        b = b.api_key(api_key).model(&self.model_id);
                        let client = b.build().await?;
                        let client = client.with_model_middlewares(self.middlewares.clone());
                        Arc::new(client)
                    } else {
                        return Err(LlmError::ConfigurationError(format!(
                            "Unsupported provider for registry handle: {}",
                            other
                        )));
                    }
                }
                #[cfg(not(feature = "openai"))]
                {
                    return Err(LlmError::ConfigurationError(format!(
                        "Unsupported provider for registry handle: {}",
                        other
                    )));
                }
            }
        };

        Ok(built)
    }

    /// Non-streaming chat (messages + optional tools)
    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let client = self.build_client().await?;
        client.chat_with_tools(messages, tools).await
    }

    /// Streaming chat (messages + optional tools)
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let client = self.build_client().await?;
        client.chat_stream(messages, tools).await
    }
}

/// Minimal embedding model handle.
#[derive(Clone)]
pub struct EmbeddingModelHandle {
    pub provider_id: String,
    pub model_id: String,
}

impl EmbeddingModelHandle {
    /// Build a client for this embedding model handle.
    async fn build_client(&self) -> Result<Arc<dyn LlmClient>, LlmError> {
        let built: Arc<dyn LlmClient> = match self.provider_id.as_str() {
            #[cfg(test)]
            "testprov_embed" => {
                TEST_BUILD_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Arc::new(TestProvEmbedClient)
            }
            #[cfg(feature = "openai")]
            "openai" => {
                let client = crate::quick_openai_with_model(&self.model_id).await?;
                Arc::new(client)
            }
            #[cfg(feature = "google")]
            "gemini" => {
                let client = crate::quick_gemini_with_model(&self.model_id).await?;
                Arc::new(client)
            }
            #[cfg(feature = "anthropic")]
            "anthropic" => {
                let client = crate::quick_anthropic_with_model(&self.model_id).await?;
                Arc::new(client)
            }
            #[cfg(feature = "groq")]
            "groq" => {
                let client = crate::quick_groq_with_model(&self.model_id).await?;
                Arc::new(client)
            }
            #[cfg(feature = "xai")]
            "xai" => {
                let client = crate::prelude::quick_xai_with_model(&self.model_id).await?;
                Arc::new(client)
            }
            #[cfg(feature = "openai")]
            other => {
                if crate::providers::openai_compatible::config::get_provider_config(other).is_some()
                {
                    let mut b = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                        crate::builder::LlmBuilder::new(),
                        other,
                    );
                    let env_key = format!("{}_API_KEY", other.to_uppercase());
                    let api_key = std::env::var(&env_key).map_err(|_| {
                        LlmError::ConfigurationError(format!(
                            "Missing {} for OpenAI-compatible provider {}",
                            env_key, other
                        ))
                    })?;
                    b = b.api_key(api_key).model(&self.model_id);
                    let client = b.build().await?;
                    Arc::new(client)
                } else {
                    return Err(LlmError::ConfigurationError(format!(
                        "Unsupported provider for registry embedding handle: {}",
                        other
                    )));
                }
            }
            #[cfg(not(feature = "openai"))]
            other => {
                return Err(LlmError::ConfigurationError(format!(
                    "Unsupported provider for registry embedding handle: {}",
                    other
                )));
            }
        };

        Ok(built)
    }

    /// Simple embedding call
    pub async fn embed(
        &self,
        input: Vec<String>,
    ) -> Result<crate::types::EmbeddingResponse, LlmError> {
        let client = self.build_client().await?;
        let Some(cap) = client.as_embedding_capability() else {
            return Err(LlmError::UnsupportedOperation(
                "Embedding not supported by this client".into(),
            ));
        };
        cap.embed(input).await
    }
}

/// Minimal image model handle.
#[derive(Clone)]
pub struct ImageModelHandle {
    pub provider_id: String,
    pub model_id: String,
}

impl ImageModelHandle {
    /// Build a client for this image model handle.
    async fn build_client(&self) -> Result<Arc<dyn LlmClient>, LlmError> {
        let built: Arc<dyn LlmClient> = match self.provider_id.as_str() {
            #[cfg(feature = "openai")]
            "openai" => {
                let client = crate::quick_openai_with_model(&self.model_id).await?;
                Arc::new(client)
            }
            #[cfg(feature = "google")]
            "gemini" => {
                let client = crate::quick_gemini_with_model(&self.model_id).await?;
                Arc::new(client)
            }
            other => {
                #[cfg(feature = "openai")]
                {
                    if crate::providers::openai_compatible::config::get_provider_config(other)
                        .is_some()
                    {
                        let mut b =
                            crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                                crate::builder::LlmBuilder::new(),
                                other,
                            );
                        let env_key = format!("{}_API_KEY", other.to_uppercase());
                        let api_key = std::env::var(&env_key).map_err(|_| {
                            LlmError::ConfigurationError(format!(
                                "Missing {} for OpenAI-compatible provider {}",
                                env_key, other
                            ))
                        })?;
                        b = b.api_key(api_key).model(&self.model_id);
                        let client = b.build().await?;
                        Arc::new(client)
                    } else {
                        return Err(LlmError::ConfigurationError(format!(
                            "Unsupported provider for registry image handle: {}",
                            other
                        )));
                    }
                }
                #[cfg(not(feature = "openai"))]
                {
                    return Err(LlmError::ConfigurationError(format!(
                        "Unsupported provider for registry image handle: {}",
                        other
                    )));
                }
            }
        };
        Ok(built)
    }

    pub async fn generate_images(
        &self,
        request: crate::types::ImageGenerationRequest,
    ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
        let client = self.build_client().await?;
        let Some(cap) = client.as_image_generation_capability() else {
            return Err(LlmError::UnsupportedOperation(
                "Image generation not supported by this client".into(),
            ));
        };
        cap.generate_images(request).await
    }
}

/// Minimal speech model handle (TTS/STT).
#[derive(Clone)]
pub struct SpeechModelHandle {
    pub provider_id: String,
    pub model_id: String,
}

impl SpeechModelHandle {
    /// Build a client for this speech model handle.
    async fn build_client(&self) -> Result<Arc<dyn LlmClient>, LlmError> {
        let built: Arc<dyn LlmClient> = match self.provider_id.as_str() {
            #[cfg(feature = "openai")]
            "openai" => Arc::new(crate::quick_openai_with_model(&self.model_id).await?),
            #[cfg(feature = "google")]
            "gemini" => Arc::new(crate::quick_gemini_with_model(&self.model_id).await?),
            other => {
                #[cfg(feature = "openai")]
                {
                    if crate::providers::openai_compatible::config::get_provider_config(other)
                        .is_some()
                    {
                        let mut b =
                            crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                                crate::builder::LlmBuilder::new(),
                                other,
                            );
                        let env_key = format!("{}_API_KEY", other.to_uppercase());
                        let api_key = std::env::var(&env_key).map_err(|_| {
                            LlmError::ConfigurationError(format!(
                                "Missing {} for {}",
                                env_key, other
                            ))
                        })?;
                        b = b.api_key(api_key).model(&self.model_id);
                        Arc::new(b.build().await?)
                    } else {
                        return Err(LlmError::ConfigurationError(format!(
                            "Unsupported provider for registry speech handle: {}",
                            other
                        )));
                    }
                }
                #[cfg(not(feature = "openai"))]
                {
                    return Err(LlmError::ConfigurationError(format!(
                        "Unsupported provider for registry speech handle: {}",
                        other
                    )));
                }
            }
        };
        Ok(built)
    }

    pub async fn text_to_speech(
        &self,
        req: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        let client = self.build_client().await?;
        let Some(cap) = client.as_audio_capability() else {
            return Err(LlmError::UnsupportedOperation(
                "TTS not supported by this client".into(),
            ));
        };
        cap.text_to_speech(req).await
    }

    pub async fn speech_to_text(
        &self,
        req: crate::types::SttRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        let client = self.build_client().await?;
        let Some(cap) = client.as_audio_capability() else {
            return Err(LlmError::UnsupportedOperation(
                "STT not supported by this client".into(),
            ));
        };
        cap.speech_to_text(req).await
    }
}

/// Minimal transcription model handle (alias of speech for STT).
#[derive(Clone)]
pub struct TranscriptionModelHandle {
    pub provider_id: String,
    pub model_id: String,
}

impl TranscriptionModelHandle {
    pub async fn speech_to_text(
        &self,
        req: crate::types::SttRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        // Reuse SpeechModelHandle semantics
        let tmp = SpeechModelHandle {
            provider_id: self.provider_id.clone(),
            model_id: self.model_id.clone(),
        };
        tmp.speech_to_text(req).await
    }
}
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
#[cfg(test)]
pub static TEST_BUILD_COUNT: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub struct TestProvClient;
#[cfg(test)]
#[async_trait::async_trait]
impl ChatCapability for TestProvClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Ok(crate::types::ChatResponse::new(
            crate::types::MessageContent::Text("ok".to_string()),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::stream::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation("mock stream".into()))
    }
}
#[cfg(test)]
impl LlmClient for TestProvClient {
    fn provider_name(&self) -> &'static str {
        "testprov"
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_chat()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvClient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct MockClient(std::sync::Arc<std::sync::Mutex<usize>>);

    #[async_trait::async_trait]
    impl ChatCapability for MockClient {
        async fn chat_with_tools(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::types::ChatResponse, LlmError> {
            *self.0.lock().unwrap() += 1;
            Ok(crate::types::ChatResponse::new(
                crate::types::MessageContent::Text("ok".to_string()),
            ))
        }

        async fn chat_stream(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::stream::ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation("mock stream".into()))
        }
    }

    impl LlmClient for MockClient {
        fn provider_name(&self) -> &'static str {
            "mock"
        }
        fn supported_models(&self) -> Vec<String> {
            vec!["mock-model".into()]
        }
        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            crate::traits::ProviderCapabilities::new().with_chat()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn clone_box(&self) -> Box<dyn LlmClient> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn language_model_handle_builds_client() {
        let reg = create_provider_registry(HashMap::new(), None);
        let handle = reg.language_model("testprov:model").unwrap();

        // Each call builds a new client (no caching)
        TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
        let resp = handle.chat(vec![], None).await.unwrap();
        assert_eq!(resp.content_text().unwrap_or_default(), "ok");
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            1
        );

        // Second call also builds a new client
        let resp = handle.chat(vec![], None).await.unwrap();
        assert_eq!(resp.content_text().unwrap_or_default(), "ok");
        assert_eq!(
            TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            2
        );
    }
}

#[cfg(test)]
pub struct TestProvEmbedClient;
#[cfg(test)]
#[async_trait::async_trait]
impl crate::traits::EmbeddingCapability for TestProvEmbedClient {
    async fn embed(&self, input: Vec<String>) -> Result<crate::types::EmbeddingResponse, LlmError> {
        Ok(crate::types::EmbeddingResponse::new(
            vec![vec![input.len() as f32]],
            "test-embed-model".to_string(),
        ))
    }
    fn embedding_dimension(&self) -> usize {
        1
    }
}
#[cfg(test)]
impl LlmClient for TestProvEmbedClient {
    fn provider_name(&self) -> &'static str {
        "testprov_embed"
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["model".into()]
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(TestProvEmbedClient)
    }
    fn as_embedding_capability(&self) -> Option<&dyn crate::traits::EmbeddingCapability> {
        Some(self)
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;
    #[tokio::test]
    async fn embedding_model_handle_builds_client() {
        let reg = create_provider_registry(HashMap::new(), None);
        let handle = reg.embedding_model("testprov_embed:model").unwrap();

        // Client is built on each call (no caching)
        let out = handle.embed(vec!["a".into(), "b".into()]).await.unwrap();
        assert_eq!(out.embeddings[0][0], 2.0);
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl crate::traits::ChatCapability for TestProvEmbedClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat not supported in TestProvEmbedClient".into(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::stream::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat stream not supported in TestProvEmbedClient".into(),
        ))
    }
}
