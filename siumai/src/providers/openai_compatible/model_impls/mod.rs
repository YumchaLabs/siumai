//! OpenAI-Compatible Model Implementations
//!
//! This module contains model implementations for OpenAI-compatible providers.
//! Each model type (chat, embedding, image, rerank) has its own implementation
//! that uses the OpenAI Standard Layer with provider-specific adapters.

pub mod chat;
pub mod embedding;
pub mod image;
pub mod rerank;

pub use chat::OpenAiCompatibleChatModel;
pub use embedding::OpenAiCompatibleEmbeddingModel;
pub use image::OpenAiCompatibleImageModel;
pub use rerank::OpenAiCompatibleRerankModel;
