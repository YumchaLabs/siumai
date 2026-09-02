//! Curated Google Gemini provider facade.

pub use siumai_provider_gemini::{
    GEMINI_EMBEDDING_API_MODE_ID, GEMINI_GENERATE_CONTENT_API_MODE_ID,
    GEMINI_MULTIMODAL_EMBEDDING_API_MODE_ID, GEMINI_SPEECH_API_MODE_ID, GeminiConfigError,
    GeminiCredential, GeminiDecodedGenerateContent, GeminiDecodedInteraction, GeminiDownloadUri,
    GeminiEmbeddingContentPart, GeminiEmbeddingModalityUsage, GeminiEmbeddingModel,
    GeminiEmbeddingOptions, GeminiEmbeddingTaskType, GeminiFile, GeminiFileListQuery,
    GeminiFileName, GeminiFileNameError, GeminiFilePage, GeminiFileProcessingError,
    GeminiFileSource, GeminiFileState, GeminiFiles, GeminiGenerateContentModel,
    GeminiGenerateContentOptions, GeminiGenerateContentServiceTier, GeminiGenerateContentThinking,
    GeminiGenerateContentThinkingLevel, GeminiGeneratedVideo, GeminiGeneratedVideoUri,
    GeminiImageModel, GeminiLanguageModel, GeminiMultimodalEmbeddingModel,
    GeminiMultimodalEmbeddingRequest, GeminiMultimodalEmbeddingResponse, GeminiProfile,
    GeminiProfileError, GeminiProvider, GeminiProviderBuilder, GeminiSpeechModel,
    GeminiSpeechOptions, GeminiVeo, GeminiVeoAspectRatio, GeminiVeoConfig, GeminiVeoFailure,
    GeminiVeoImage, GeminiVeoOperation, GeminiVeoOperationMetadata, GeminiVeoOperationRef,
    GeminiVeoOperationState, GeminiVeoPersonGeneration, GeminiVeoRequest, GeminiVeoResolution,
    GeminiVeoResult, GeminiVeoSource,
};

pub mod models {
    pub use siumai_provider_gemini::{
        GEMINI_2_5_FLASH_PREVIEW_TTS, GEMINI_2_5_PRO_PREVIEW_TTS, GEMINI_3_1_FLASH_IMAGE,
        GEMINI_3_1_FLASH_LITE_IMAGE, GEMINI_3_1_FLASH_TTS_PREVIEW, GEMINI_3_5_FLASH,
        GEMINI_3_5_FLASH_LITE, GEMINI_3_6_FLASH, GEMINI_3_PRO_IMAGE, GEMINI_EMBEDDING_001,
        GEMINI_EMBEDDING_2, VEO_3_1_FAST_GENERATE_PREVIEW, VEO_3_1_GENERATE_PREVIEW,
        VEO_3_1_LITE_GENERATE_PREVIEW, current_image_models, current_interactions_models,
    };
}

pub mod options {
    pub use siumai_provider_gemini::{
        GeminiEmbeddingOptions, GeminiEmbeddingTaskType, GeminiGenerateContentOptions,
        GeminiGenerateContentServiceTier, GeminiGenerateContentThinking,
        GeminiGenerateContentThinkingLevel, GeminiImageAspectRatio, GeminiImageOptions,
        GeminiImageSize, GeminiInteractionStorage, GeminiInteractionsOptions, GeminiSpeechOptions,
        GeminiThinkingLevel, GeminiThinkingSummaries,
    };
}
