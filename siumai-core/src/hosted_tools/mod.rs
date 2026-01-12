//! Provider-defined tools (core crate).
//!
//! Factory helpers for creating "hosted tools" that execute on the provider side
//! (web search, file search, code execution, etc.). These tools are expressed via
//! `Tool::ProviderDefined`, and are treated as part of the provider's capabilities
//! rather than application-level features.
//!
//! Currently supported:
//! - `hosted_tools::openai`     OpenAI Responses API hosted tools
//! - `hosted_tools::anthropic`  Anthropic hosted tools
//! - `hosted_tools::google`     Google/Gemini hosted tools
//!
//! Additional Vercel-aligned tools are added incrementally (e.g., OpenAI `mcp`,
//! Anthropic `memory_20250818`, Anthropic `code_execution_20250825`).

pub mod anthropic;
pub mod google;
pub mod openai;
