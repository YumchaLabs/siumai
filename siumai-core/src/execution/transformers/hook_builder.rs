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
/// use siumai_core::execution::transformers::hook_builder::HookBuilder;
/// use serde_json::json;
///
/// let hooks = HookBuilder::new()
///     .with_chat_body_builder(|req| {
///         Ok(json!({
///             "model": req.common_params.model.clone(),
///             "input": req.messages.clone(),
///             "stream": req.stream,
///         }))
///     })
///     .with_chat_validator(|req, body| {
///         // Custom validation logic
///         if req.messages.is_empty() {
///             return Err(siumai_core::error::LlmError::InvalidInput(
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
