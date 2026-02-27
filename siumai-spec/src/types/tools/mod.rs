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

pub use choice::{ToolChoice, ToolType};
pub use function::ToolFunction;
pub use provider_defined::ProviderDefinedTool;
pub use tool::Tool;
