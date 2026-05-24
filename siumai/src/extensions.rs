//! Extension capabilities (non-unified surface).
//!
//! These are capability/adapter-level traits and payloads rather than the stable family-model
//! entrypoints. Prefer `siumai::prelude::unified` for stable family execution. Video's stable
//! surface is `siumai::video::*` / `VideoModel`; the low-level `VideoGenerationCapability` remains
//! here for provider adapters and compatibility code. Music remains extension-only.

pub use siumai_core::traits::{
    AudioCapability, EmbeddingCapability, FileManagementCapability, ImageExtras,
    ModelListingCapability, ModerationCapability, MusicGenerationCapability, RerankCapability,
    SkillsCapability, SpeechExtras, TimeoutCapability, TranscriptionExtras,
    VideoGenerationCapability,
};

/// Types used by non-unified extension capabilities.
pub mod types {
    pub use siumai_core::types::{
        FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
        ImageEditInput, ImageEditRequest, ImageVariationRequest, ModerationRequest,
        ModerationResponse, SkillFileContent, SkillProviderMetadata, SkillUploadFile,
        SkillUploadRequest, SkillUploadResult, VideoGenerationInput, VideoGenerationRequest,
        VideoGenerationResponse, VideoTaskStatus, VideoTaskStatusResponse,
    };
}
