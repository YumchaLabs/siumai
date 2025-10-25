//! Provider Core Abstractions (Deprecated)
//!
//! This module is deprecated. Use `siumai::core` instead.
//!
//! All types have been moved to the `core` module for better organization.

// Re-export builder_core module for backward compatibility
pub mod builder_core {
    pub use crate::core::builder_core::*;
}

// Re-export from core for backward compatibility
pub use crate::core::{
    AudioTransformer, CapabilityKind, ChatTransformers, EmbeddingTransformers, FilesTransformer,
    ImageTransformers, ProviderContext, ProviderCore, ProviderSpec, RerankTransformers,
};

// Re-export helper functions
pub use crate::core::provider_spec::{default_custom_options_hook, matches_provider_id};
