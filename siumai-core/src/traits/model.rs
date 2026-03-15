//! Shared model metadata traits.
//!
//! These traits provide a family-agnostic metadata contract that can be reused by
//! language, embedding, image, reranking, speech, and transcription model traits.

/// Version marker for stable model-family contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSpecVersion {
    /// Initial family-model contract introduced during the V4 refactor spike.
    V1,
}

/// Shared metadata contract for model-family objects.
pub trait ModelMetadata: Send + Sync {
    /// Canonical provider id (for example, `openai`).
    fn provider_id(&self) -> &str;

    /// Provider-specific model id (for example, `gpt-4o-mini`).
    fn model_id(&self) -> &str;

    /// Specification version for the model-family contract.
    fn specification_version(&self) -> ModelSpecVersion {
        ModelSpecVersion::V1
    }
}
