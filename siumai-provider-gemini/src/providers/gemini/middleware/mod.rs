//! Provider-specific model middlewares for Gemini.

mod tool_warnings;

pub use tool_warnings::GeminiToolWarningsMiddleware;
