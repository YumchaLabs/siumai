//! Tool runtime (schema + execution binding).
//!
//! This module re-exports a curated `siumai_core::tooling` surface for the facade crate.

pub use siumai_core::tooling::{
    ExecutableTool, ExecutableTools, ProviderDefinedToolFactory,
    ProviderDefinedToolFactoryWithOutputSchema, ProviderExecutedToolFactory, ToolExecuteFn,
    ToolExecuteFunction, ToolExecuteStreamFn, ToolExecuteValueStream, ToolExecuteWithOptionsFn,
    ToolExecutionOptions, ToolExecutionResult, ToolExecutionStream, ToolInputAvailableContext,
    ToolInputAvailableFn, ToolInputDeltaContext, ToolInputDeltaFn, ToolInputStartFn,
    ToolModelOutputContext, ToolModelOutputFn, ToolNeedsApproval, ToolNeedsApprovalContext,
    ToolNeedsApprovalFn, ToolRuntimeContext, ToolRuntimeMetadata, ToolSet,
    create_provider_defined_tool_factory, create_provider_defined_tool_factory_with_output_schema,
    create_provider_executed_tool_factory, dynamic_tool, execute_tool, is_executable_tool,
    model_messages_from_chat_messages, tool,
};
