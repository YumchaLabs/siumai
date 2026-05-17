//! OpenAI Compatible Client
//!
//! This module provides a client implementation for OpenAI-compatible providers.

use super::openai_config::OpenAiCompatibleConfig;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use std::sync::Arc;

mod audio;
mod chat;
mod compatibility;
mod completion;
mod embedding;
mod image;
mod models;
mod rerank;
mod runtime;
mod types;

pub(crate) use runtime::{DEPRECATED_OPENAI_COMPATIBLE_KEY_WARNING, model_slot_is_missing};
pub use types::{
    OpenAiCompatibleChatResponse, OpenAiCompatibleChoice, OpenAiCompatibleFunction,
    OpenAiCompatibleMessage, OpenAiCompatibleToolCall, OpenAiCompatibleUsage,
};

/// OpenAI compatible client
///
/// This is a separate client implementation that uses the adapter system
/// to handle provider-specific differences without modifying the core OpenAI client.
#[derive(Clone)]
pub struct OpenAiCompatibleClient {
    config: OpenAiCompatibleConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

#[cfg(test)]
mod tests;
