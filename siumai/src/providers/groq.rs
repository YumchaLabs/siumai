//! Curated Groq provider facade.

pub use siumai_provider_groq::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
    GroqConfigError, GroqCredential, GroqLanguageApi, GroqLanguageModel, GroqProvider,
    GroqProviderBuilder, GroqTranscriptionModel,
};

pub mod models {
    pub use siumai_provider_groq::models::{VERIFIED_ON, language, preview, transcription};
}

pub mod options {
    pub use siumai_provider_groq::{
        GroqLanguageOptions, GroqReasoningEffort, GroqReasoningFormat, GroqResponsesOptions,
        GroqResponsesServiceTier, GroqServiceTier, GroqTimestampGranularity,
        GroqTranscriptionOptions, GroqTranscriptionResponseFormat,
    };
}

pub mod metadata {
    pub use siumai_provider_groq::{
        GroqLanguageMetadata, GroqLanguageResponseExt, GroqTranscriptionMetadata,
        GroqTranscriptionResponseExt,
    };
}

pub mod tools {
    pub use siumai_provider_groq::tools::{
        browser_search, responses_browser_search, responses_code_execution,
    };
}
