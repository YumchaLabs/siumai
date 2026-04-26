//! Tool calling and function definition types

// Deprecated ToolCall and FunctionCall removed. Use ContentPart::ToolCall and tool_result helpers.

mod choice;
mod function;
mod provider_defined;
mod tool;

#[cfg(any())]
mod openai_builtin;

#[cfg(test)]
mod tests;

pub use choice::{LanguageModelV4ToolChoice, ToolChoice, ToolType, prepare_tool_choice};
pub use function::{LanguageModelV4FunctionTool, ToolFunction};
pub use provider_defined::{LanguageModelV4ProviderTool, ProviderDefinedTool};
pub use tool::{LanguageModelV4Tool, Tool};
