//! Curated MiniMax provider facade.

pub use siumai_provider_minimax::{
    MinimaxAnnotationResolver, MinimaxChatCompletionsOptions, MinimaxConfigError,
    MinimaxContentCache, MinimaxCredential, MinimaxCredentialError, MinimaxLanguageApi,
    MinimaxLanguageModel, MinimaxLanguageProfileError, MinimaxMessageCache, MinimaxMessagesOptions,
    MinimaxProvider, MinimaxProviderBuilder, MinimaxReasoningEffort, MinimaxResponsesOptions,
    MinimaxResponsesReasoning, MinimaxServiceTier, MinimaxThinking, MinimaxToolCache,
};

pub mod models {
    pub use siumai_provider_minimax::models::{
        ALL_LANGUAGE, MINIMAX_M2, MINIMAX_M2_1, MINIMAX_M2_1_HIGHSPEED, MINIMAX_M2_5,
        MINIMAX_M2_5_HIGHSPEED, MINIMAX_M2_7, MINIMAX_M2_7_HIGHSPEED, MINIMAX_M3,
        MODEL_CATALOG_SOURCE, MODEL_CATALOG_VERIFIED_ON, image, music, speech, video,
    };
}

pub mod options {
    pub use siumai_provider_minimax::{
        MinimaxChatCompletionsOptions, MinimaxMessagesOptions, MinimaxReasoningEffort,
        MinimaxResponsesOptions, MinimaxResponsesReasoning, MinimaxServiceTier, MinimaxThinking,
    };
}

pub mod resources {
    pub use siumai_provider_minimax::resources::*;
}
