//! Curated Groq provider facade.

pub use siumai_provider_groq::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource, GroqAudio,
    GroqAudioResponse, GroqConfigError, GroqCredential, GroqLanguageApi, GroqLanguageModel,
    GroqProvider, GroqProviderBuilder, GroqSpeechModel, GroqTranscriptionModel,
    GroqUrlAudioRequest,
};

pub mod models {
    pub use siumai_provider_groq::models::{
        CURRENT_SPEECH_MODELS, DEFAULT_SPEECH, VERIFIED_ON, language, preview, speech,
        transcription,
    };
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
        GroqLanguageMetadata, GroqLanguageResponseExt, GroqMcpOutput, GroqMcpOutputKind,
        GroqTranscriptionMetadata, GroqTranscriptionResponseExt,
    };
}

pub mod tools {
    pub use siumai_provider_groq::tools::{
        browser_search, responses_browser_search, responses_code_execution, responses_remote_mcp,
    };
    pub use siumai_provider_groq::{GroqMcpApproval, GroqRemoteMcpTool};
}
