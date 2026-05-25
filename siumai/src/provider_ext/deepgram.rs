pub use siumai_provider_deepgram::providers::deepgram::{
    DeepgramClient, DeepgramConfig, DeepgramSpeechModel, DeepgramTranscriptionModel, VERSION,
};
use siumai_registry::provider::SiumaiBuilder;

/// Curated Deepgram model constants aligned with the AI SDK speech/transcription package surface.
pub mod models {
    pub use siumai_provider_deepgram::providers::deepgram::models::{
        self as model_sets, ALL_SPEECH, ALL_TRANSCRIPTION, DEFAULT_SPEECH, DEFAULT_TRANSCRIPTION,
        speech, transcription,
    };
}

/// Create the unified Deepgram provider builder.
///
/// This mirrors the AI SDK package-level `deepgram` export while routing through Siumai's
/// registry-owned family handles.
pub fn deepgram() -> SiumaiBuilder {
    SiumaiBuilder::new().deepgram()
}

/// Create the unified Deepgram provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createDeepgram()`.
pub fn create_deepgram() -> SiumaiBuilder {
    deepgram()
}

/// Typed provider options (`provider_options_map["deepgram"]`).
pub mod options {
    pub use siumai_provider_deepgram::providers::deepgram::{
        DeepgramSpeechModelOptions, DeepgramSpeechOptions, DeepgramSttOptions,
        DeepgramSummarizeOption, DeepgramTranscriptionModelOptions,
    };
}

/// Request extension traits for attaching Deepgram provider options.
pub mod ext {
    pub use siumai_provider_deepgram::providers::deepgram::{
        DeepgramSttRequestExt, DeepgramTtsRequestExt,
    };
}

pub use ext::{DeepgramSttRequestExt, DeepgramTtsRequestExt};
pub use models::{
    ALL_SPEECH, ALL_TRANSCRIPTION, DEFAULT_SPEECH, DEFAULT_TRANSCRIPTION, model_sets, speech,
    transcription,
};
pub use options::{
    DeepgramSpeechModelOptions, DeepgramSpeechOptions, DeepgramSttOptions, DeepgramSummarizeOption,
    DeepgramTranscriptionModelOptions,
};
