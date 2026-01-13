//! Dynamic step preparation for orchestrator.
//!
//! Allows modifying orchestrator behavior before each step is executed.

use std::sync::Arc;

use super::types::StepResult;
use siumai::prelude::unified::{ChatMessage, ProviderOptionsMap, Tool};

/// Context provided to the prepare step callback.
pub struct PrepareStepContext<'a> {
    /// The current step number (0-indexed).
    pub step_number: usize,
    /// All steps executed so far.
    pub steps: &'a [StepResult],
    /// The current message history that will be sent to the model.
    pub messages: &'a [ChatMessage],
}

/// Result returned from the prepare step callback.
///
/// Any field set to `Some` will override the default value for this step.
/// Fields set to `None` will use the default value from the orchestrator options.
#[derive(Default)]
pub struct PrepareStepResult {
    /// Override the tool choice for this step.
    pub tool_choice: Option<ToolChoice>,
    /// Limit which tools are available for this step.
    pub active_tools: Option<Vec<String>>,
    /// Override the system message for this step.
    pub system: Option<String>,
    /// Override the message history for this step.
    pub messages: Option<Vec<ChatMessage>>,
    /// Provider options overrides for this step (Vercel-aligned `providerOptions`).
    ///
    /// Merged into the outgoing `ChatRequest.provider_options_map` using
    /// `ProviderOptionsMap::merge_overrides` (deep-merge for JSON objects).
    pub provider_options_map: Option<ProviderOptionsMap>,
}

// Default is derived

impl PrepareStepResult {
    /// Create a new empty PrepareStepResult.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the tool choice for this step.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set the active tools for this step.
    pub fn with_active_tools(mut self, tools: Vec<String>) -> Self {
        self.active_tools = Some(tools);
        self
    }

    /// Set the system message for this step.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the message history for this step.
    pub fn with_messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// Merge provider options into this step.
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        match self.provider_options_map.as_mut() {
            Some(existing) => existing.merge_overrides(map),
            None => self.provider_options_map = Some(map),
        }
        self
    }

    /// Set provider options under a provider id (convenience wrapper).
    pub fn with_provider_option(
        self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        let mut map = ProviderOptionsMap::new();
        map.insert(provider_id, options);
        self.with_provider_options_map(map)
    }
}

/// Tool choice strategy for a step.
#[derive(Debug, Clone)]
pub enum ToolChoice {
    /// Let the model decide whether to call tools.
    Auto,
    /// Require the model to call at least one tool.
    Required,
    /// Prevent the model from calling any tools.
    None,
    /// Force the model to call a specific tool.
    Specific {
        /// Name of the tool that must be called.
        tool_name: String,
    },
}

/// Callback function for preparing each step.
///
/// This callback is invoked before each step is executed, allowing you to:
/// - Force or prevent tool calls
/// - Limit which tools are available
/// - Modify the message history (e.g., to compress context)
/// - Change the system message
///
/// # Example
///
/// ```rust,ignore
/// use siumai_extras::orchestrator::{PrepareStepFn, PrepareStepResult, ToolChoice};
/// use std::sync::Arc;
///
/// let prepare_step: PrepareStepFn = Arc::new(|ctx| {
///     if ctx.step_number == 0 {
///         // Force the first step to call a specific tool
///         PrepareStepResult::new()
///             .with_tool_choice(ToolChoice::Specific {
///                 tool_name: "search".to_string(),
///             })
///             .with_active_tools(vec!["search".to_string()])
///     } else if ctx.messages.len() > 20 {
///         // Compress message history if it gets too long
///         let compressed = ctx.messages[ctx.messages.len() - 10..].to_vec();
///         PrepareStepResult::new().with_messages(compressed)
///     } else {
///         PrepareStepResult::default()
///     }
/// });
/// ```
pub type PrepareStepFn = Arc<dyn Fn(PrepareStepContext) -> PrepareStepResult + Send + Sync>;

/// Find the most recent Anthropic container id from prior steps (Vercel-aligned).
///
/// This inspects `StepResult.provider_metadata.anthropic.container.id` searching backwards.
pub fn find_anthropic_container_id_from_last_step(steps: &[StepResult]) -> Option<String> {
    for step in steps.iter().rev() {
        let Some(provider_meta) = step.provider_metadata.as_ref() else {
            continue;
        };

        let Some(provider) = provider_meta
            .get("anthropic")
            .or_else(|| provider_meta.get("Anthropic"))
        else {
            continue;
        };

        let id = provider
            .get("container")
            .and_then(|v| v.as_object())
            .and_then(|obj| obj.get("id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if id.as_deref().is_some_and(|s| !s.is_empty()) {
            return id;
        }
    }

    None
}

/// Create providerOptions overrides that forward Anthropic container id between steps.
///
/// Equivalent to Vercel AI SDK's `forwardAnthropicContainerIdFromLastStep` helper.
pub fn forward_anthropic_container_id_from_last_step(
    steps: &[StepResult],
) -> Option<ProviderOptionsMap> {
    let container_id = find_anthropic_container_id_from_last_step(steps)?;

    let mut map = ProviderOptionsMap::new();
    map.insert(
        "anthropic",
        serde_json::json!({
            "container": { "id": container_id }
        }),
    );

    Some(map)
}

/// Helper function to filter tools based on active_tools list.
pub(crate) fn filter_active_tools(tools: &[Tool], active_tools: &Option<Vec<String>>) -> Vec<Tool> {
    if let Some(active) = active_tools {
        tools
            .iter()
            .filter(|t| {
                let tool_name = match t {
                    Tool::Function { function } => &function.name,
                    Tool::ProviderDefined(provider_tool) => &provider_tool.name,
                };
                active.contains(tool_name)
            })
            .cloned()
            .collect()
    } else {
        tools.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_step_result_builder() {
        let result = PrepareStepResult::new()
            .with_tool_choice(ToolChoice::Required)
            .with_active_tools(vec!["tool1".to_string(), "tool2".to_string()])
            .with_system("Custom system message");

        assert!(matches!(result.tool_choice, Some(ToolChoice::Required)));
        assert_eq!(
            result.active_tools,
            Some(vec!["tool1".to_string(), "tool2".to_string()])
        );
        assert_eq!(result.system, Some("Custom system message".to_string()));
        assert!(result.messages.is_none());
    }

    #[test]
    fn forward_anthropic_container_id_from_last_step_reads_provider_metadata() {
        use std::collections::HashMap;

        let step = StepResult {
            messages: vec![],
            finish_reason: None,
            usage: None,
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                HashMap::from([(
                    "container".to_string(),
                    serde_json::json!({ "id": "container_123", "expiresAt": "2025-01-01T00:00:00Z" }),
                )]),
            )])),
        };

        let map = forward_anthropic_container_id_from_last_step(&[step]).expect("expected map");
        assert_eq!(
            map.get("anthropic"),
            Some(&serde_json::json!({ "container": { "id": "container_123" } }))
        );
    }

    #[test]
    fn test_filter_active_tools() {
        use siumai::prelude::unified::Tool;

        let tools = vec![
            Tool::function(
                "tool1".to_string(),
                "Tool 1".to_string(),
                serde_json::json!({}),
            ),
            Tool::function(
                "tool2".to_string(),
                "Tool 2".to_string(),
                serde_json::json!({}),
            ),
            Tool::function(
                "tool3".to_string(),
                "Tool 3".to_string(),
                serde_json::json!({}),
            ),
        ];

        // Test with active_tools filter
        let active = Some(vec!["tool1".to_string(), "tool3".to_string()]);
        let filtered = filter_active_tools(&tools, &active);
        assert_eq!(filtered.len(), 2);
        match &filtered[0] {
            Tool::Function { function } => assert_eq!(function.name, "tool1"),
            _ => panic!("Expected Function variant"),
        }
        match &filtered[1] {
            Tool::Function { function } => assert_eq!(function.name, "tool3"),
            _ => panic!("Expected Function variant"),
        }

        // Test without filter
        let filtered = filter_active_tools(&tools, &None);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_prepare_step_fn() {
        let prepare_fn: PrepareStepFn = Arc::new(|ctx| {
            if ctx.step_number == 0 {
                PrepareStepResult::new().with_tool_choice(ToolChoice::Required)
            } else {
                PrepareStepResult::default()
            }
        });

        let ctx = PrepareStepContext {
            step_number: 0,
            steps: &[],
            messages: &[],
        };

        let result = prepare_fn(ctx);
        assert!(matches!(result.tool_choice, Some(ToolChoice::Required)));

        let ctx = PrepareStepContext {
            step_number: 1,
            steps: &[],
            messages: &[],
        };

        let result = prepare_fn(ctx);
        assert!(result.tool_choice.is_none());
    }
}
