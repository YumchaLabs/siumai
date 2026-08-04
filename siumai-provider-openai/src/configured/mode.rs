use siumai_core::ApiModeId;

use super::provider::OpenAiConfigError;

/// Explicit native language API selected by a model handle or Registry route.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpenAiApiMode {
    /// Recommended for new language, reasoning, tool, and multi-turn workloads.
    #[default]
    Responses,
    /// Compatibility mode for integrations that require Chat Completions.
    ChatCompletions,
}

impl OpenAiApiMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Responses => "responses",
            Self::ChatCompletions => "chat-completions",
        }
    }

    pub(crate) fn id(self) -> Result<ApiModeId, OpenAiConfigError> {
        ApiModeId::new(self.as_str()).map_err(OpenAiConfigError::InvalidIdentity)
    }
}
