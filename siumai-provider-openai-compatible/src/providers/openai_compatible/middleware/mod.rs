//! OpenAI-compatible provider middleware.

mod alibaba_cache_control;
mod deprecated_provider_options;
mod structured_outputs;
mod tool_warnings;

pub(crate) use alibaba_cache_control::OpenAiCompatibleAlibabaCacheControlWarningMiddleware;
pub(crate) use deprecated_provider_options::OpenAiCompatibleDeprecatedProviderOptionsWarningMiddleware;
pub(crate) use structured_outputs::OpenAiCompatibleStructuredOutputsWarningMiddleware;
pub use tool_warnings::OpenAiCompatibleToolWarningsMiddleware;
