//! Provider-defined tools (core crate).
//!
//! Factory helpers for creating "hosted tools" that execute on the provider
//! side (web search, file search, code execution, etc.). These tools are
//! expressed via `Tool::ProviderDefined`, and are treated as part of the
//! LLM provider's capabilities rather than application-level features.
//!
//! Currently supported:
//! - `hosted_tools::openai`   – OpenAI Responses API web_search / file_search / computer_use
//! - `hosted_tools::anthropic` – Anthropic web_search_20250305 / web_fetch_20250910 / tool_search_* / code_execution_20250522
//! - `hosted_tools::google`   – Google/Gemini google_search / file_search / code_execution / url_context

pub mod anthropic;
pub mod google;
pub mod openai;
