//! Tool runtime (schema + execution binding).
//!
//! This module keeps the public `siumai_core::tooling::*` surface intact while splitting the
//! implementation into named runtime, context, factory, and collection modules.

mod context;
mod factories;
mod runtime;
mod set;

pub use context::{
    ToolExecuteFn, ToolExecuteFunction, ToolExecuteStreamFn, ToolExecuteValueStream,
    ToolExecuteWithOptionsFn, ToolExecutionOptions, ToolExecutionResult, ToolExecutionStream,
    ToolInputAvailableContext, ToolInputAvailableFn, ToolInputDeltaContext, ToolInputDeltaFn,
    ToolInputStartFn, ToolModelOutputContext, ToolModelOutputFn, ToolNeedsApproval,
    ToolNeedsApprovalContext, ToolNeedsApprovalFn, ToolRuntimeContext, ToolRuntimeMetadata,
    model_messages_from_chat_messages,
};
pub use factories::{
    ProviderDefinedToolFactory, ProviderDefinedToolFactoryWithOutputSchema,
    ProviderExecutedToolFactory, create_provider_defined_tool_factory,
    create_provider_defined_tool_factory_with_output_schema, create_provider_executed_tool_factory,
};
pub use runtime::{ExecutableTool, dynamic_tool, execute_tool, is_executable_tool, tool};
pub use set::{ExecutableTools, ToolSet};

#[cfg(test)]
mod tests;
