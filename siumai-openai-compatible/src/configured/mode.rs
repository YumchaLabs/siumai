/// OpenAI-family language API selected by one model handle or Registry route.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpenAiCompatibleApiMode {
    /// Responses-style item protocol.
    #[default]
    Responses,
    /// Chat Completions compatibility protocol.
    ChatCompletions,
}

impl OpenAiCompatibleApiMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Responses => "responses",
            Self::ChatCompletions => "chat-completions",
        }
    }
}
