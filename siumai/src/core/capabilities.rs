//! Capability Traits
//!
//! This module re-exports all capability traits from the traits module.

// Re-export all capability traits from the traits module
pub use crate::traits::{
    AudioCapability, ChatCapability, ChatExtensions, EmbeddingCapability, EmbeddingExtensions,
    FileManagementCapability, ImageGenerationCapability, ModelListingCapability,
    ModerationCapability, ProviderCapabilities, RerankCapability, TimeoutCapability,
    VisionCapability,
};
