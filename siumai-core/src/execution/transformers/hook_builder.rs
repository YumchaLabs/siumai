#![allow(clippy::collapsible_if)]
//! Hook Builder for composable request transformers
//!
//! This module provides a builder pattern for creating ProviderRequestHooks
//! with reusable components, making it easier to create custom transformers.

use crate::error::LlmError;
use crate::execution::transformers::request::ProviderRequestHooks;
use crate::types::{ChatRequest, EmbeddingRequest, ImageGenerationRequest};
use serde_json::Value;
use std::sync::Arc;

/// Type alias for chat body builder function
pub type ChatBodyBuilder = Arc<dyn Fn(&ChatRequest) -> Result<Value, LlmError> + Send + Sync>;

/// Type alias for chat post-processor function
pub type ChatPostProcessor =
    Arc<dyn Fn(&ChatRequest, &mut Value) -> Result<(), LlmError> + Send + Sync>;

/// Type alias for embedding body builder function
pub type EmbeddingBodyBuilder =
    Arc<dyn Fn(&EmbeddingRequest) -> Result<Value, LlmError> + Send + Sync>;

/// Type alias for embedding post-processor function
pub type EmbeddingPostProcessor =
    Arc<dyn Fn(&EmbeddingRequest, &mut Value) -> Result<(), LlmError> + Send + Sync>;

/// Type alias for image body builder function
pub type ImageBodyBuilder =
    Arc<dyn Fn(&ImageGenerationRequest) -> Result<Value, LlmError> + Send + Sync>;

/// Type alias for image post-processor function
pub type ImagePostProcessor =
    Arc<dyn Fn(&ImageGenerationRequest, &mut Value) -> Result<(), LlmError> + Send + Sync>;

/// Builder for creating composable ProviderRequestHooks
///
/// # Example
/// ```rust,no_run
/// use siumai::experimental::execution::transformers::hook_builder::HookBuilder;
/// use serde_json::json;
///
/// let hooks = HookBuilder::new()
///     .with_openai_base()
///     .with_chat_validator(|req, body| {
///         // Custom validation logic
///         if req.messages.is_empty() {
///             return Err(siumai::error::LlmError::InvalidInput(
///                 "Messages cannot be empty".to_string()
///             ));
///         }
///         Ok(())
///     })
///     .with_chat_post_processor(|req, body| {
///         // Custom post-processing
///         if let Some(obj) = body.as_object_mut() {
///             obj.insert("custom_field".to_string(), json!("custom_value"));
///         }
///         Ok(())
///     })
///     .build();
/// ```
pub struct HookBuilder {
    chat_body_builder: Option<ChatBodyBuilder>,
    chat_post_processors: Vec<ChatPostProcessor>,
    embedding_body_builder: Option<EmbeddingBodyBuilder>,
    embedding_post_processors: Vec<EmbeddingPostProcessor>,
    image_body_builder: Option<ImageBodyBuilder>,
    image_post_processors: Vec<ImagePostProcessor>,
}

impl HookBuilder {
    /// Create a new empty hook builder
    pub fn new() -> Self {
        Self {
            chat_body_builder: None,
            chat_post_processors: Vec::new(),
            embedding_body_builder: None,
            embedding_post_processors: Vec::new(),
            image_body_builder: None,
            image_post_processors: Vec::new(),
        }
    }

    /// Start with OpenAI-compatible base chat body builder
    ///
    /// This provides a standard OpenAI-compatible chat request structure
    pub fn with_openai_base(mut self) -> Self {
        self.chat_body_builder = Some(Arc::new(openai_base_chat_body));
        self
    }

    /// Start with Anthropic-compatible base chat body builder
    ///
    /// This provides a standard Anthropic-compatible chat request structure
    pub fn with_anthropic_base(mut self) -> Self {
        self.chat_body_builder = Some(Arc::new(anthropic_base_chat_body));
        self
    }

    /// Set a custom chat body builder
    pub fn with_chat_body_builder<F>(mut self, builder: F) -> Self
    where
        F: Fn(&ChatRequest) -> Result<Value, LlmError> + Send + Sync + 'static,
    {
        self.chat_body_builder = Some(Arc::new(builder));
        self
    }

    /// Add a chat validator (runs before post-processors)
    ///
    /// Validators can return errors to reject invalid requests
    pub fn with_chat_validator<F>(mut self, validator: F) -> Self
    where
        F: Fn(&ChatRequest, &mut Value) -> Result<(), LlmError> + Send + Sync + 'static,
    {
        self.chat_post_processors.insert(0, Arc::new(validator));
        self
    }

    /// Add a chat post-processor
    ///
    /// Post-processors can modify the request body after it's been built
    pub fn with_chat_post_processor<F>(mut self, processor: F) -> Self
    where
        F: Fn(&ChatRequest, &mut Value) -> Result<(), LlmError> + Send + Sync + 'static,
    {
        self.chat_post_processors.push(Arc::new(processor));
        self
    }

    /// Set a custom embedding body builder
    pub fn with_embedding_body_builder<F>(mut self, builder: F) -> Self
    where
        F: Fn(&EmbeddingRequest) -> Result<Value, LlmError> + Send + Sync + 'static,
    {
        self.embedding_body_builder = Some(Arc::new(builder));
        self
    }

    /// Add an embedding post-processor
    pub fn with_embedding_post_processor<F>(mut self, processor: F) -> Self
    where
        F: Fn(&EmbeddingRequest, &mut Value) -> Result<(), LlmError> + Send + Sync + 'static,
    {
        self.embedding_post_processors.push(Arc::new(processor));
        self
    }

    /// Set a custom image body builder
    pub fn with_image_body_builder<F>(mut self, builder: F) -> Self
    where
        F: Fn(&ImageGenerationRequest) -> Result<Value, LlmError> + Send + Sync + 'static,
    {
        self.image_body_builder = Some(Arc::new(builder));
        self
    }

    /// Add an image post-processor
    pub fn with_image_post_processor<F>(mut self, processor: F) -> Self
    where
        F: Fn(&ImageGenerationRequest, &mut Value) -> Result<(), LlmError> + Send + Sync + 'static,
    {
        self.image_post_processors.push(Arc::new(processor));
        self
    }

    /// Build the final ComposableHooks instance
    pub fn build(self) -> ComposableHooks {
        ComposableHooks {
            chat_body_builder: self.chat_body_builder,
            chat_post_processors: self.chat_post_processors,
            embedding_body_builder: self.embedding_body_builder,
            embedding_post_processors: self.embedding_post_processors,
            image_body_builder: self.image_body_builder,
            image_post_processors: self.image_post_processors,
        }
    }
}

impl Default for HookBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Composable hooks implementation
pub struct ComposableHooks {
    chat_body_builder: Option<ChatBodyBuilder>,
    chat_post_processors: Vec<ChatPostProcessor>,
    embedding_body_builder: Option<EmbeddingBodyBuilder>,
    embedding_post_processors: Vec<EmbeddingPostProcessor>,
    image_body_builder: Option<ImageBodyBuilder>,
    image_post_processors: Vec<ImagePostProcessor>,
}

impl ProviderRequestHooks for ComposableHooks {
    fn build_base_chat_body(&self, req: &ChatRequest) -> Result<Value, LlmError> {
        match &self.chat_body_builder {
            Some(builder) => builder(req),
            None => Err(LlmError::UnsupportedOperation(
                "No chat body builder configured".to_string(),
            )),
        }
    }

    fn post_process_chat(&self, req: &ChatRequest, body: &mut Value) -> Result<(), LlmError> {
        for processor in &self.chat_post_processors {
            processor(req, body)?;
        }
        Ok(())
    }

    fn build_base_embedding_body(&self, req: &EmbeddingRequest) -> Result<Value, LlmError> {
        match &self.embedding_body_builder {
            Some(builder) => builder(req),
            None => Err(LlmError::UnsupportedOperation(
                "No embedding body builder configured".to_string(),
            )),
        }
    }

    fn post_process_embedding(
        &self,
        req: &EmbeddingRequest,
        body: &mut Value,
    ) -> Result<(), LlmError> {
        for processor in &self.embedding_post_processors {
            processor(req, body)?;
        }
        Ok(())
    }

    fn build_base_image_body(&self, req: &ImageGenerationRequest) -> Result<Value, LlmError> {
        match &self.image_body_builder {
            Some(builder) => builder(req),
            None => Err(LlmError::UnsupportedOperation(
                "No image body builder configured".to_string(),
            )),
        }
    }

    fn post_process_image(
        &self,
        req: &ImageGenerationRequest,
        body: &mut Value,
    ) -> Result<(), LlmError> {
        for processor in &self.image_post_processors {
            processor(req, body)?;
        }
        Ok(())
    }
}

// ============================================================================
// Built-in base body builders
// ============================================================================

/// OpenAI-compatible base chat body builder
fn openai_base_chat_body(req: &ChatRequest) -> Result<Value, LlmError> {
    use serde_json::json;

    let mut body = json!({
        "model": req.common_params.model,
        "messages": req.messages,
    });

    // Add common parameters if present
    if let Some(temp) = req.common_params.temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(max_tokens) = req.common_params.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    if let Some(top_p) = req.common_params.top_p {
        body["top_p"] = json!(top_p);
    }
    if req.stream {
        body["stream"] = json!(true);
    }

    Ok(body)
}

/// Anthropic-compatible base chat body builder
fn anthropic_base_chat_body(req: &ChatRequest) -> Result<Value, LlmError> {
    use serde_json::json;

    // Anthropic requires system messages to be separate
    let (system_messages, user_messages): (Vec<_>, Vec<_>) = req
        .messages
        .iter()
        .partition(|m| matches!(m.role, crate::types::MessageRole::System));

    let mut body = json!({
        "model": req.common_params.model,
        "messages": user_messages,
        "max_tokens": req.common_params.max_tokens.unwrap_or(1024),
    });

    // Add system message if present
    if !system_messages.is_empty()
        && let Some(first_system) = system_messages.first()
        && let crate::types::MessageContent::Text(text) = &first_system.content
    {
        body["system"] = json!(text);
    }

    // Add common parameters
    if let Some(temp) = req.common_params.temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(top_p) = req.common_params.top_p {
        body["top_p"] = json!(top_p);
    }

    Ok(body)
}
