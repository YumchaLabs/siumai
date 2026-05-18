#[allow(deprecated)]
pub use siumai_provider_gemini::providers::gemini::GoogleGenerativeAIProviderSettings;
pub use siumai_provider_gemini::providers::gemini::types::GeminiConfig;
pub use siumai_provider_gemini::providers::gemini::{
    GeminiBuilder, GeminiClient, GoogleInteractionsLanguageModel, GoogleInteractionsModelInput,
    GoogleProviderSettings, SharedIdGenerator, VERSION,
};

/// Curated model-id groups aligned with the audited `@ai-sdk/google` package surface.
pub mod models;

pub use models::{agents, chat, embedding, image, interactions, model_sets, video};

/// Provider tool factories that return `Tool` directly (Vercel-aligned).
pub mod tools {
    pub use crate::tools::google::*;
}

/// Provider-executed tool builders (typed args).
pub mod hosted_tools {
    pub use crate::hosted_tools::google::*;
}

/// Compatibility alias for older imports.
pub mod provider_tools {
    pub use crate::tools::google::*;
}

/// Typed provider options (`provider_options_map["google"]`).
pub mod options {
    pub use siumai_provider_gemini::provider_options::gemini::{
        GeminiHarmBlockThreshold, GeminiHarmCategory, GeminiImageOptions, GeminiOptions,
        GeminiResponseModality, GeminiSafetySetting, GeminiThinkingConfig, GeminiThinkingLevel,
        GoogleEmbeddingContentPart, GoogleEmbeddingInlineData, GoogleEmbeddingModelOptions,
        GoogleFilesUploadOptions, GoogleImageModelOptions, GoogleInteractionsAgentConfig,
        GoogleInteractionsAgentName, GoogleInteractionsImageConfig, GoogleInteractionsModelId,
        GoogleInteractionsResponseFormatEntry, GoogleLanguageModelInteractionsOptions,
        GoogleLanguageModelOptions, GoogleVideoModelId, GoogleVideoModelOptions,
    };
    #[allow(deprecated)]
    pub use siumai_provider_gemini::provider_options::gemini::{
        GoogleGenerativeAIEmbeddingProviderOptions, GoogleGenerativeAIImageProviderOptions,
        GoogleGenerativeAIProviderOptions, GoogleGenerativeAIVideoModelId,
        GoogleGenerativeAIVideoProviderOptions,
    };
    pub use siumai_provider_gemini::providers::gemini::ext::{
        GeminiChatRequestExt, GeminiImageRequestExt, GoogleChatRequestExt,
        GoogleEmbeddingRequestExt, GoogleImageRequestExt, GoogleVideoRequestExt,
    };
    pub use siumai_provider_gemini::providers::gemini::types::{
        GeminiEmbeddingOptions, GeminiEmbeddingRequestExt,
    };
}

// Provider-owned typed options (kept out of `siumai-core`).
pub use options::{
    GeminiChatRequestExt, GeminiEmbeddingOptions, GeminiEmbeddingRequestExt,
    GeminiHarmBlockThreshold, GeminiHarmCategory, GeminiImageOptions, GeminiImageRequestExt,
    GeminiOptions, GeminiResponseModality, GeminiSafetySetting, GeminiThinkingConfig,
    GeminiThinkingLevel, GoogleChatRequestExt, GoogleEmbeddingContentPart,
    GoogleEmbeddingInlineData, GoogleEmbeddingModelOptions, GoogleEmbeddingRequestExt,
    GoogleFilesUploadOptions, GoogleImageModelOptions, GoogleImageRequestExt,
    GoogleInteractionsAgentConfig, GoogleInteractionsAgentName, GoogleInteractionsImageConfig,
    GoogleInteractionsModelId, GoogleInteractionsResponseFormatEntry,
    GoogleLanguageModelInteractionsOptions, GoogleLanguageModelOptions, GoogleVideoModelId,
    GoogleVideoModelOptions, GoogleVideoRequestExt,
};
#[allow(deprecated)]
pub use options::{
    GoogleGenerativeAIEmbeddingProviderOptions, GoogleGenerativeAIImageProviderOptions,
    GoogleGenerativeAIProviderOptions, GoogleGenerativeAIVideoModelId,
    GoogleGenerativeAIVideoProviderOptions,
};

/// Typed response metadata helpers (`ChatResponse.provider_metadata["google"]`).
pub mod metadata {
    #[allow(deprecated)]
    pub use siumai_provider_gemini::provider_metadata::gemini::{
        GeminiChatResponseExt, GeminiContentPartExt, GeminiMetadata, GeminiSource,
        GoogleGenerativeAIProviderMetadata, GoogleInteractionsProviderMetadata,
        GoogleProviderMetadata,
    };
}
#[allow(deprecated)]
pub use metadata::{
    GeminiChatResponseExt, GeminiContentPartExt, GeminiMetadata, GeminiSource,
    GoogleGenerativeAIProviderMetadata, GoogleInteractionsProviderMetadata, GoogleProviderMetadata,
};
pub use siumai_provider_gemini::providers::gemini::{GoogleErrorBody, GoogleErrorData};

/// Non-unified Gemini extension APIs (escape hatches).
pub mod ext {
    pub use siumai_provider_gemini::providers::gemini::ext::{
        code_execution, file_search_stores, tools,
    };
}

/// Provider-specific resources not covered by the unified families.
pub mod resources {
    pub use siumai_provider_gemini::providers::gemini::{
        GeminiCachedContents, GeminiFileSearchStores, GeminiFiles, GeminiModels, GeminiTokens,
        GeminiVideo, GoogleErrorBody, GoogleErrorData, GoogleInteractionsLanguageModel,
        GoogleInteractionsModelInput,
    };
}

// Legacy Gemini parameter structs (provider-owned).
pub use siumai_provider_gemini::params::gemini::{
    GeminiParams, GeminiParamsBuilder, GenerationConfig, SafetyCategory, SafetySetting,
    SafetyThreshold,
};
