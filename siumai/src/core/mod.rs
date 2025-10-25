//! Core Abstractions
//!
//! This module contains the core abstractions and traits that define the siumai architecture.
//!
//! ## Architecture Overview
//!
//! ```text
//! User Code
//!     ↓
//! Provider Client (OpenAiClient, AnthropicClient, etc.)
//!     ↓
//! ProviderSpec (defines HTTP routing, headers, transformers)
//!     ↓
//! Executors (HTTP execution with middleware/interceptor/retry)
//!     ↓
//! Transformers (request/response/stream conversion)
//!     ↓
//! HTTP Client
//! ```
//!
//! ## Key Components
//!
//! ### Provider Specification
//! - **`ProviderSpec`** - Unified specification for provider behavior
//! - **`ProviderContext`** - Runtime context for provider execution
//! - **`CapabilityKind`** - Enumeration of supported capabilities
//!
//! ### Capability Traits (Re-exported from `traits` module)
//! - **`ChatCapability`** - Chat completion capability
//! - **`EmbeddingCapability`** - Embedding generation capability
//! - **`VisionCapability`** - Vision/image analysis capability
//! - **`AudioCapability`** - Audio processing capability
//! - **`ImageGenerationCapability`** - Image generation capability
//! - **`FileManagementCapability`** - File management capability
//! - **`ModerationCapability`** - Content moderation capability
//! - **`RerankCapability`** - Document reranking capability
//!
//! ### Client Abstractions
//! - **`LlmClient`** - Unified client trait for all providers
//! - **`ProviderCore`** - Core builder functionality shared across providers
//!
//! ## Design Principles
//!
//! 1. **Trait-based capabilities** - Providers implement capability traits
//! 2. **Unified specification** - `ProviderSpec` defines provider behavior
//! 3. **Composable transformers** - Request/response/stream transformers
//! 4. **Provider-agnostic execution** - Executors work with any provider
//!
//! ## Usage
//!
//! Most users don't interact with this module directly. Instead, use the high-level APIs:
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//!
//! // High-level API (recommended)
//! let client = LlmBuilder::new().openai().build().await?;
//! let response = client.chat(vec![user!("Hello!")]).await?;
//! ```
//!
//! For advanced use cases, you can use the core abstractions:
//!
//! ```rust,ignore
//! use siumai::core::{ChatCapability, ProviderSpec};
//!
//! async fn generic_chat(client: &dyn ChatCapability) -> Result<(), LlmError> {
//!     let response = client.chat(vec![user!("Hello!")]).await?;
//!     Ok(())
//! }
//! ```

pub mod builder_core;
pub mod builder_macros;
pub mod client;
pub mod provider_spec;

// Re-export main types
pub use provider_spec::{
    AudioTransformer, CapabilityKind, ChatTransformers, EmbeddingTransformers, FilesTransformer,
    ImageTransformers, ProviderContext, ProviderSpec, RerankTransformers,
    default_custom_options_hook, matches_provider_id,
};

// Re-export capability traits directly from traits module
pub use crate::traits::{
    AudioCapability, ChatCapability, ChatExtensions, EmbeddingCapability, EmbeddingExtensions,
    FileManagementCapability, ImageGenerationCapability, ModelListingCapability,
    ModerationCapability, ProviderCapabilities, RerankCapability, TimeoutCapability,
    VisionCapability,
};

pub use client::LlmClient;

pub use builder_core::ProviderCore;
