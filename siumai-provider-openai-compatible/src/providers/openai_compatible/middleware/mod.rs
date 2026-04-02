//! OpenAI-compatible provider middleware.

mod structured_outputs;
mod tool_warnings;

pub(crate) use structured_outputs::OpenAiCompatibleStructuredOutputsWarningMiddleware;
pub use tool_warnings::OpenAiCompatibleToolWarningsMiddleware;
