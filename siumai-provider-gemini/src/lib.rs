//! Product-level Google Gemini provider for Siumai.
#![deny(unsafe_code)]

mod embedding;
mod files;
mod generate_content;
mod http;
mod image;
mod language;
mod models;
mod options;
mod profile;
mod provider;
mod speech;
mod veo;

pub use embedding::{
    GEMINI_EMBEDDING_001, GEMINI_EMBEDDING_2, GEMINI_EMBEDDING_API_MODE_ID, GeminiEmbeddingModel,
    GeminiEmbeddingOptions, GeminiEmbeddingTaskType,
};
pub use files::{
    GeminiDownloadUri, GeminiFile, GeminiFileListQuery, GeminiFileName, GeminiFileNameError,
    GeminiFilePage, GeminiFileProcessingError, GeminiFileSource, GeminiFileState, GeminiFiles,
};
pub use generate_content::{
    GEMINI_GENERATE_CONTENT_API_MODE_ID, GeminiGenerateContentModel, GeminiGenerateContentOptions,
    GeminiGenerateContentServiceTier, GeminiGenerateContentThinking,
    GeminiGenerateContentThinkingLevel,
};
pub use image::GeminiImageModel;
pub use language::GeminiLanguageModel;
pub use models::{
    GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_LITE_IMAGE, GEMINI_3_5_FLASH, GEMINI_3_5_FLASH_LITE,
    GEMINI_3_6_FLASH, GEMINI_3_PRO_IMAGE, current_image_models, current_interactions_models,
};
pub use options::{
    GeminiImageAspectRatio, GeminiImageOptions, GeminiImageSize, GeminiInteractionStorage,
    GeminiInteractionsOptions, GeminiThinkingLevel, GeminiThinkingSummaries,
};
pub use profile::{GeminiProfile, GeminiProfileError};
pub use provider::{GeminiConfigError, GeminiCredential, GeminiProvider, GeminiProviderBuilder};
pub use siumai_protocol_gemini::generate_content::DecodedGenerateContent as GeminiDecodedGenerateContent;
pub use siumai_protocol_gemini::interactions::DecodedInteraction as GeminiDecodedInteraction;
pub use speech::{
    GEMINI_2_5_FLASH_PREVIEW_TTS, GEMINI_2_5_PRO_PREVIEW_TTS, GEMINI_3_1_FLASH_TTS_PREVIEW,
    GEMINI_SPEECH_API_MODE_ID, GeminiSpeechModel, GeminiSpeechOptions,
};
pub use veo::{
    GeminiGeneratedVideo, GeminiGeneratedVideoUri, GeminiVeo, GeminiVeoAspectRatio,
    GeminiVeoConfig, GeminiVeoFailure, GeminiVeoImage, GeminiVeoOperation,
    GeminiVeoOperationMetadata, GeminiVeoOperationRef, GeminiVeoOperationState,
    GeminiVeoPersonGeneration, GeminiVeoRequest, GeminiVeoResolution, GeminiVeoResult,
    GeminiVeoSource, VEO_3_1_FAST_GENERATE_PREVIEW, VEO_3_1_GENERATE_PREVIEW,
    VEO_3_1_LITE_GENERATE_PREVIEW,
};
