//! Curated Anthropic provider facade.

pub use siumai_provider_anthropic::{
    AnthropicConfigError, AnthropicCredential, AnthropicCredentialError, AnthropicLanguageModel,
    AnthropicProfileError, AnthropicProvider, AnthropicProviderBuilder,
};

pub mod models {
    pub use siumai_provider_anthropic::{
        CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5, CLAUDE_HAIKU_4_5_20251001, CLAUDE_MYTHOS_5,
        CLAUDE_MYTHOS_PREVIEW, CLAUDE_OPUS_4_1_20250805, CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7,
        CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5, current_models,
    };
}

pub mod options {
    pub use siumai_provider_anthropic::{AnthropicMessagesOptions, AnthropicThinking};
}

pub mod annotations {
    pub use siumai_provider_anthropic::{
        AnthropicAnnotationResolver, AnthropicCacheTtl, AnthropicContentOptions,
        AnthropicMessageCache, AnthropicToolOptions, AnthropicToolSpecError,
    };
}

pub mod messages {
    pub use siumai_provider_anthropic::{
        AdvisorToolOptions, AnthropicTool, AnthropicToolReference, ComputerToolOptions,
        FallbackOutputConfig, InferenceSpeed, McpToolConfig, McpToolsetOptions, MessagesMetadata,
        MidConversationToolChange, OutputEffort, ResponseInclusion, ServerFallback,
        ServerFallbacks, TextEditorToolOptions, ThinkingDisplay, ToolCaller, UserLocation,
        WebFetchToolOptions, WebSearchToolOptions,
    };
}

pub mod resources {
    pub use siumai_provider_anthropic::resources::{
        AnthropicBatchDeleteResult, AnthropicBatchItem, AnthropicBatchList,
        AnthropicBatchListQuery, AnthropicBatchRequest, AnthropicFile, AnthropicFileDeleteResult,
        AnthropicFileList, AnthropicFileListQuery, AnthropicFileUpload, AnthropicFiles,
        AnthropicMessageBatch, AnthropicMessageBatches, AnthropicSkill, AnthropicSkillFile,
        AnthropicSkillUpload, AnthropicSkillUploadResult, AnthropicSkillVersion,
        AnthropicSkillVersionList, AnthropicSkills, AnthropicTokenCount, AnthropicTokens,
    };
}
