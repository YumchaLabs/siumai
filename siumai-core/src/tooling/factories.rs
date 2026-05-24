use serde_json::Value;

use crate::types::{ProviderDefinedTool, Tool};

use super::runtime::ExecutableTool;

/// Rust facade for AI SDK provider-utils `createProviderDefinedToolFactory`.
///
/// The produced tool is `type: "provider"` with `isProviderExecuted: false`; callers can bind a
/// local executor by wrapping the returned `Tool` in `ExecutableTool`.
#[derive(Debug, Clone)]
pub struct ProviderDefinedToolFactory {
    id: String,
    name: String,
    input_schema: Value,
}

impl ProviderDefinedToolFactory {
    /// Create a provider-defined tool factory.
    pub fn new(id: impl Into<String>, name: impl Into<String>, input_schema: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input_schema,
        }
    }

    /// Tool id in `<provider>.<tool>` format.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Siumai tool-map name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the provider-defined input schema.
    pub fn input_schema(&self) -> &Value {
        &self.input_schema
    }

    /// Create a passive provider-defined tool.
    pub fn create_tool(&self, args: Value) -> Tool {
        Tool::ProviderDefined(
            ProviderDefinedTool::provider_defined(self.id.clone(), self.name.clone())
                .with_input_schema(self.input_schema.clone())
                .with_args(args),
        )
    }

    /// Create a passive provider-defined tool with an output schema.
    pub fn create_tool_with_output_schema(&self, args: Value, output_schema: Value) -> Tool {
        self.create_tool(args).with_output_schema(output_schema)
    }

    /// Create an executable wrapper around a provider-defined tool.
    pub fn create_executable_tool(&self, args: Value) -> ExecutableTool {
        ExecutableTool::new(self.create_tool(args))
    }

    /// Create an executable wrapper around a provider-defined tool with an output schema.
    pub fn create_executable_tool_with_output_schema(
        &self,
        args: Value,
        output_schema: Value,
    ) -> ExecutableTool {
        ExecutableTool::new(self.create_tool_with_output_schema(args, output_schema))
    }
}

/// Create an AI SDK-style provider-defined tool factory.
pub fn create_provider_defined_tool_factory(
    id: impl Into<String>,
    name: impl Into<String>,
    input_schema: Value,
) -> ProviderDefinedToolFactory {
    ProviderDefinedToolFactory::new(id, name, input_schema)
}

/// Rust facade for AI SDK provider-utils
/// `createProviderDefinedToolFactoryWithOutputSchema`.
///
/// The produced tool is `type: "provider"` with `isProviderExecuted: false` and a fixed output
/// schema captured by the factory.
#[derive(Debug, Clone)]
pub struct ProviderDefinedToolFactoryWithOutputSchema {
    factory: ProviderDefinedToolFactory,
    output_schema: Value,
}

impl ProviderDefinedToolFactoryWithOutputSchema {
    /// Create a provider-defined tool factory with a fixed output schema.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        input_schema: Value,
        output_schema: Value,
    ) -> Self {
        Self {
            factory: ProviderDefinedToolFactory::new(id, name, input_schema),
            output_schema,
        }
    }

    /// Tool id in `<provider>.<tool>` format.
    pub fn id(&self) -> &str {
        self.factory.id()
    }

    /// Siumai tool-map name.
    pub fn name(&self) -> &str {
        self.factory.name()
    }

    /// Borrow the provider-defined input schema.
    pub fn input_schema(&self) -> &Value {
        self.factory.input_schema()
    }

    /// Borrow the provider-defined output schema.
    pub fn output_schema(&self) -> &Value {
        &self.output_schema
    }

    /// Create a passive provider-defined tool with the fixed output schema.
    pub fn create_tool(&self, args: Value) -> Tool {
        self.factory
            .create_tool_with_output_schema(args, self.output_schema.clone())
    }

    /// Create an executable wrapper around a provider-defined tool with the fixed output schema.
    pub fn create_executable_tool(&self, args: Value) -> ExecutableTool {
        ExecutableTool::new(self.create_tool(args))
    }
}

/// Create an AI SDK-style provider-defined tool factory with a fixed output schema.
pub fn create_provider_defined_tool_factory_with_output_schema(
    id: impl Into<String>,
    name: impl Into<String>,
    input_schema: Value,
    output_schema: Value,
) -> ProviderDefinedToolFactoryWithOutputSchema {
    ProviderDefinedToolFactoryWithOutputSchema::new(id, name, input_schema, output_schema)
}

/// Rust facade for AI SDK provider-utils `createProviderExecutedToolFactory`.
///
/// The produced tool is `type: "provider"` with `isProviderExecuted: true`.
#[derive(Debug, Clone)]
pub struct ProviderExecutedToolFactory {
    id: String,
    name: String,
    input_schema: Value,
    output_schema: Value,
    supports_deferred_results: Option<bool>,
}

impl ProviderExecutedToolFactory {
    /// Create a provider-executed tool factory.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        input_schema: Value,
        output_schema: Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input_schema,
            output_schema,
            supports_deferred_results: None,
        }
    }

    /// Tool id in `<provider>.<tool>` format.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Siumai tool-map name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the provider-defined input schema.
    pub fn input_schema(&self) -> &Value {
        &self.input_schema
    }

    /// Borrow the provider-defined output schema.
    pub fn output_schema(&self) -> &Value {
        &self.output_schema
    }

    /// Mark whether generated tools support deferred results.
    pub fn with_supports_deferred_results(mut self, supports_deferred_results: bool) -> Self {
        self.supports_deferred_results = Some(supports_deferred_results);
        self
    }

    /// Create a passive provider-executed tool.
    pub fn create_tool(&self, args: Value) -> Tool {
        let mut tool = ProviderDefinedTool::provider_executed(self.id.clone(), self.name.clone())
            .with_input_schema(self.input_schema.clone())
            .with_output_schema(self.output_schema.clone())
            .with_args(args);

        if let Some(supports_deferred_results) = self.supports_deferred_results {
            tool = tool.with_supports_deferred_results(supports_deferred_results);
        }

        Tool::ProviderDefined(tool)
    }

    /// Create an executable wrapper around a provider-executed tool.
    pub fn create_executable_tool(&self, args: Value) -> ExecutableTool {
        ExecutableTool::new(self.create_tool(args))
    }
}

/// Create an AI SDK-style provider-executed tool factory.
pub fn create_provider_executed_tool_factory(
    id: impl Into<String>,
    name: impl Into<String>,
    input_schema: Value,
    output_schema: Value,
) -> ProviderExecutedToolFactory {
    ProviderExecutedToolFactory::new(id, name, input_schema, output_schema)
}
