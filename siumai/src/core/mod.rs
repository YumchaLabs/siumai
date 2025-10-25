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
//! - **ProviderSpec**: Unified specification for provider behavior
//! - **ProviderContext**: Runtime context for provider execution
//! - **Capabilities**: Trait-based capability system (Chat, Embedding, Vision, etc.)
//! - **LlmClient**: Unified client interface

pub mod builder_core;
pub mod capabilities;
pub mod client;
pub mod provider_spec;

// Re-export main types
pub use provider_spec::{
    AudioTransformer, CapabilityKind, ChatTransformers, EmbeddingTransformers, FilesTransformer,
    ImageTransformers, ProviderContext, ProviderSpec, RerankTransformers,
    default_custom_options_hook, matches_provider_id,
};

pub use capabilities::{
    AudioCapability, ChatCapability, EmbeddingCapability, FileManagementCapability,
    ImageGenerationCapability, ModelListingCapability, ModerationCapability, ProviderCapabilities,
    RerankCapability, VisionCapability,
};

pub use client::LlmClient;

pub use builder_core::ProviderCore;
