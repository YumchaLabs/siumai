//! Tool runtime (schema + execution binding).
//!
//! This module provides a single cohesive tool system without introducing additional crates:
//! - `Tool` remains the portable, spec-level schema value (in `siumai-spec`)
//! - `ExecutableTool` binds a `Tool` to an async execution function
//! - `ExecutableTools` provides name-based lookup and execution
//!
//! Higher-level orchestration (multi-step loops, approvals, stop conditions) stays in `siumai-extras`.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use futures::future::BoxFuture;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::LlmError;
use crate::types::Tool;

/// Async execution function signature for tools.
pub type ToolExecuteFn =
    Arc<dyn Fn(Value) -> BoxFuture<'static, Result<Value, LlmError>> + Send + Sync>;

/// A tool definition with an optional bound executor.
#[derive(Clone)]
pub struct ExecutableTool {
    tool: Tool,
    execute: Option<ToolExecuteFn>,
}

impl std::fmt::Debug for ExecutableTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutableTool")
            .field("name", &self.name())
            .field("has_execute", &self.execute.is_some())
            .finish()
    }
}

impl ExecutableTool {
    /// Create a tool wrapper without an executor.
    pub const fn new(tool: Tool) -> Self {
        Self {
            tool,
            execute: None,
        }
    }

    /// Bind an executor to an existing tool schema.
    pub fn with_execute(mut self, execute: ToolExecuteFn) -> Self {
        self.execute = Some(execute);
        self
    }

    /// Create a JSON-based function tool with an executor.
    pub fn function<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        execute: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, LlmError>> + Send + 'static,
    {
        let tool = Tool::function(name, description, parameters);
        let exec: ToolExecuteFn = Arc::new(move |args: Value| Box::pin(execute(args)));
        Self {
            tool,
            execute: Some(exec),
        }
    }

    /// Create a typed function tool.
    ///
    /// `TArgs` is deserialized from JSON tool call arguments.
    /// `TOut` is serialized into JSON tool result output.
    pub fn typed_function<TArgs, TOut, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        execute: F,
    ) -> Self
    where
        TArgs: DeserializeOwned + Send + 'static,
        TOut: Serialize + Send + 'static,
        F: Fn(TArgs) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<TOut, LlmError>> + Send + 'static,
    {
        let tool = Tool::function(name, description, parameters);
        let exec: ToolExecuteFn = Arc::new(move |args: Value| {
            let parsed: Result<TArgs, LlmError> = serde_json::from_value(args).map_err(|e| {
                LlmError::InvalidParameter(format!("Failed to parse tool arguments: {e}"))
            });
            match parsed {
                Ok(parsed) => {
                    let fut = execute(parsed);
                    Box::pin(async move {
                        let out = fut.await?;
                        serde_json::to_value(out).map_err(|e| {
                            LlmError::InternalError(format!(
                                "Failed to serialize tool output as JSON: {e}"
                            ))
                        })
                    })
                }
                Err(e) => Box::pin(async move { Err(e) }),
            }
        });

        Self {
            tool,
            execute: Some(exec),
        }
    }

    /// Return the portable tool schema (for sending to the model).
    pub const fn tool(&self) -> &Tool {
        &self.tool
    }

    /// Tool name used in tool calls.
    pub fn name(&self) -> &str {
        match &self.tool {
            Tool::Function { function } => function.name.as_str(),
            Tool::ProviderDefined(t) => t.name.as_str(),
        }
    }

    /// Execute the tool with JSON arguments.
    pub async fn execute_json(&self, args: Value) -> Result<Value, LlmError> {
        let exec = self.execute.as_ref().ok_or_else(|| {
            LlmError::UnsupportedOperation(format!(
                "Tool '{}' does not have an executor bound.",
                self.name()
            ))
        })?;
        exec(args).await
    }
}

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

    /// Execute a tool by name.
    pub async fn execute(&self, name: &str, args: Value) -> Result<Value, LlmError> {
        let tool = self
            .get(name)
            .ok_or_else(|| LlmError::NotFound(format!("Tool not found: '{name}'")))?;
        tool.execute_json(args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[tokio::test]
    async fn typed_tool_parses_args_and_serializes_output() {
        #[derive(Deserialize)]
        struct Args {
            x: i64,
            y: i64,
        }

        #[derive(Serialize)]
        struct Out {
            sum: i64,
        }

        let tool = ExecutableTool::typed_function::<Args, Out, _, _>(
            "add",
            "Add two integers",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "x": { "type": "integer" },
                    "y": { "type": "integer" }
                },
                "required": ["x", "y"]
            }),
            |args| async move {
                Ok(Out {
                    sum: args.x + args.y,
                })
            },
        );

        let out = tool
            .execute_json(serde_json::json!({ "x": 1, "y": 2 }))
            .await
            .unwrap();

        assert_eq!(out, serde_json::json!({ "sum": 3 }));
    }

    #[tokio::test]
    async fn tool_set_executes_by_name() {
        let mut tools = ExecutableTools::new();
        tools.insert(ExecutableTool::function(
            "echo",
            "Echo input",
            serde_json::json!({"type":"object"}),
            |v| async move { Ok(v) },
        ));

        let out = tools
            .execute("echo", serde_json::json!({"a":1}))
            .await
            .unwrap();
        assert_eq!(out, serde_json::json!({"a":1}));
    }
}
