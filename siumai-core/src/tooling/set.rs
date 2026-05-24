use std::collections::HashMap;

use serde_json::Value;

use crate::error::LlmError;
use crate::types::{Tool, ToolResultOutput};

use super::context::{
    ToolExecutionOptions, ToolExecutionStream, ToolModelOutputContext, ToolRuntimeMetadata,
};
use super::runtime::ExecutableTool;

/// Alias aligned with AI SDK `ToolSet`.
pub type ToolSet = ExecutableTools;

/// A collection of executable tools with name-based lookup.
#[derive(Clone, Default)]
pub struct ExecutableTools {
    tools: Vec<ExecutableTool>,
    index_by_name: HashMap<String, usize>,
}

impl std::fmt::Debug for ExecutableTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutableTools")
            .field("len", &self.tools.len())
            .finish()
    }
}

impl ExecutableTools {
    /// Create an empty tool collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from an iterator of tools. Later duplicates override earlier ones by name.
    pub fn from_tools(tools: impl IntoIterator<Item = ExecutableTool>) -> Self {
        let mut out = Self::new();
        for tool in tools {
            out.insert(tool);
        }
        out
    }

    /// Insert (or replace) a tool by name.
    pub fn insert(&mut self, tool: ExecutableTool) {
        let name = tool.name().to_string();
        if let Some(&idx) = self.index_by_name.get(&name) {
            self.tools[idx] = tool;
            return;
        }
        let idx = self.tools.len();
        self.tools.push(tool);
        self.index_by_name.insert(name, idx);
    }

    /// Return tool schemas for model calls.
    pub fn schemas(&self) -> Vec<Tool> {
        self.tools.iter().map(|t| t.tool().clone()).collect()
    }

    /// Find a tool by name.
    pub fn get(&self, name: &str) -> Option<&ExecutableTool> {
        let idx = self.index_by_name.get(name).copied()?;
        self.tools.get(idx)
    }

    /// Return runtime-only AI SDK-style metadata by tool name.
    pub fn runtime_metadata(&self, name: &str) -> Option<ToolRuntimeMetadata> {
        self.get(name).map(|tool| tool.runtime_metadata().clone())
    }

    /// Execute a tool by name.
    pub async fn execute(&self, name: &str, args: Value) -> Result<Value, LlmError> {
        self.execute_with_options(name, args, ToolExecutionOptions::default())
            .await
    }

    /// Execute a tool by name with AI SDK-style execution options.
    pub async fn execute_with_options(
        &self,
        name: &str,
        args: Value,
        options: ToolExecutionOptions,
    ) -> Result<Value, LlmError> {
        let tool = self
            .get(name)
            .ok_or_else(|| LlmError::NotFound(format!("Tool not found: '{name}'")))?;
        tool.execute_json_with_options(args, options).await
    }

    /// Execute a tool by name as a normalized preliminary/final stream.
    pub async fn execute_stream(
        &self,
        name: &str,
        args: Value,
        options: ToolExecutionOptions,
    ) -> Result<ToolExecutionStream, LlmError> {
        let tool = self
            .get(name)
            .ok_or_else(|| LlmError::NotFound(format!("Tool not found: '{name}'")))?;
        tool.execute_stream(args, options).await
    }

    /// Convert a runtime tool result into a stable model-facing output by tool name.
    pub fn to_model_output(
        &self,
        name: &str,
        context: ToolModelOutputContext,
    ) -> Result<Option<ToolResultOutput>, LlmError> {
        match self.get(name) {
            Some(tool) => tool.to_model_output(context),
            None => Ok(None),
        }
    }
}
