//! Stop condition system for orchestrator.
//!
//! Provides flexible control over when to stop multi-step tool calling loops.

use super::types::StepResult;

/// A condition that determines when to stop orchestration.
///
/// Stop conditions are evaluated after each step. If any condition returns `true`,
/// the orchestration loop stops.
pub trait StopCondition: Send + Sync {
    /// Check if the orchestration should stop based on the current steps.
    fn should_stop(&self, steps: &[StepResult]) -> bool;
}

/// Stop when the number of steps reaches a specified count.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::orchestrator::{step_count_is, generate};
///
/// let stop_condition = step_count_is(5);
/// ```
pub struct StepCountIs {
    count: usize,
}

impl StopCondition for StepCountIs {
    fn should_stop(&self, steps: &[StepResult]) -> bool {
        steps.len() >= self.count
    }
}

/// Create a stop condition that stops after a specific number of steps.
///
/// # Arguments
///
/// * `count` - The maximum number of steps to execute
///
/// # Example
///
/// ```rust,ignore
/// let condition = step_count_is(10); // Stop after 10 steps
/// ```
pub fn step_count_is(count: usize) -> Box<dyn StopCondition> {
    Box::new(StepCountIs { count })
}

/// Stop when a specific tool is called.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::orchestrator::{has_tool_call, generate};
///
/// let stop_condition = has_tool_call("finalAnswer");
/// ```
pub struct HasToolCall {
    tool_name: String,
}

impl StopCondition for HasToolCall {
    fn should_stop(&self, steps: &[StepResult]) -> bool {
        if let Some(last_step) = steps.last() {
            last_step.tool_calls.iter().any(|call| {
                if let crate::types::ContentPart::ToolCall { tool_name, .. } = call {
                    tool_name == &self.tool_name
                } else {
                    false
                }
            })
        } else {
            false
        }
    }
}

/// Create a stop condition that stops when a specific tool is called.
///
/// # Arguments
///
/// * `tool_name` - The name of the tool to watch for
///
/// # Example
///
/// ```rust,ignore
/// let condition = has_tool_call("finalAnswer"); // Stop when finalAnswer is called
/// ```
pub fn has_tool_call(tool_name: impl Into<String>) -> Box<dyn StopCondition> {
    Box::new(HasToolCall {
        tool_name: tool_name.into(),
    })
}

/// Stop when the model generates text (no tool calls).
///
/// This is useful for agents that should continue until they provide a final answer.
pub struct HasTextResponse;

impl StopCondition for HasTextResponse {
    fn should_stop(&self, steps: &[StepResult]) -> bool {
        if let Some(last_step) = steps.last() {
            // Stop if there are no tool calls in the last step
            last_step.tool_calls.is_empty()
        } else {
            false
        }
    }
}

/// Create a stop condition that stops when the model generates text without tool calls.
///
/// # Example
///
/// ```rust,ignore
/// let condition = has_text_response(); // Stop when model provides text response
/// ```
pub fn has_text_response() -> Box<dyn StopCondition> {
    Box::new(HasTextResponse)
}

/// Combine multiple stop conditions with OR logic.
///
/// Stops when ANY of the conditions are met.
pub struct AnyOf {
    conditions: Vec<Box<dyn StopCondition>>,
}

impl StopCondition for AnyOf {
    fn should_stop(&self, steps: &[StepResult]) -> bool {
        self.conditions.iter().any(|c| c.should_stop(steps))
    }
}

/// Create a stop condition that stops when any of the given conditions are met.
///
/// # Arguments
///
/// * `conditions` - A vector of stop conditions
///
/// # Example
///
/// ```rust,ignore
/// let condition = any_of(vec![
///     step_count_is(20),
///     has_tool_call("finalAnswer"),
/// ]);
/// ```
pub fn any_of(conditions: Vec<Box<dyn StopCondition>>) -> Box<dyn StopCondition> {
    Box::new(AnyOf { conditions })
}

/// Combine multiple stop conditions with AND logic.
///
/// Stops when ALL of the conditions are met.
pub struct AllOf {
    conditions: Vec<Box<dyn StopCondition>>,
}

impl StopCondition for AllOf {
    fn should_stop(&self, steps: &[StepResult]) -> bool {
        !self.conditions.is_empty() && self.conditions.iter().all(|c| c.should_stop(steps))
    }
}

/// Create a stop condition that stops when all of the given conditions are met.
///
/// # Arguments
///
/// * `conditions` - A vector of stop conditions
///
/// # Example
///
/// ```rust,ignore
/// let condition = all_of(vec![
///     step_count_is(5),
///     has_text_response(),
/// ]);
/// ```
pub fn all_of(conditions: Vec<Box<dyn StopCondition>>) -> Box<dyn StopCondition> {
    Box::new(AllOf { conditions })
}

/// Custom stop condition using a closure.
///
/// # Example
///
/// ```rust,ignore
/// let condition = custom_condition(|steps| {
///     // Stop if any step has more than 3 tool calls
///     steps.iter().any(|s| s.tool_calls.len() > 3)
/// });
/// ```
pub struct CustomCondition<F>
where
    F: Fn(&[StepResult]) -> bool + Send + Sync,
{
    predicate: F,
}

impl<F> StopCondition for CustomCondition<F>
where
    F: Fn(&[StepResult]) -> bool + Send + Sync,
{
    fn should_stop(&self, steps: &[StepResult]) -> bool {
        (self.predicate)(steps)
    }
}

/// Create a custom stop condition using a closure.
///
/// # Arguments
///
/// * `predicate` - A function that takes the current steps and returns true to stop
///
/// # Example
///
/// ```rust,ignore
/// let condition = custom_condition(|steps| {
///     steps.iter().any(|s| s.tool_calls.len() > 3)
/// });
/// ```
pub fn custom_condition<F>(predicate: F) -> Box<dyn StopCondition>
where
    F: Fn(&[StepResult]) -> bool + Send + Sync + 'static,
{
    Box::new(CustomCondition { predicate })
}

/// Stop when the last step has tool results.
///
/// This is useful for stopping after tools have been executed and returned results.
pub struct HasToolResult;

impl StopCondition for HasToolResult {
    fn should_stop(&self, steps: &[StepResult]) -> bool {
        if let Some(last_step) = steps.last() {
            !last_step.tool_results.is_empty()
        } else {
            false
        }
    }
}

/// Create a stop condition that stops when the last step has tool results.
///
/// Similar to Vercel AI SDK's `has_tool_result()` stop condition.
///
/// # Example
///
/// ```rust,ignore
/// let condition = has_tool_result(); // Stop after tools return results
/// ```
pub fn has_tool_result() -> Box<dyn StopCondition> {
    Box::new(HasToolResult)
}

/// Stop when the last step has no tool calls.
///
/// This is useful for stopping when the model decides not to use any tools.
pub struct HasNoToolCalls;

impl StopCondition for HasNoToolCalls {
    fn should_stop(&self, steps: &[StepResult]) -> bool {
        if let Some(last_step) = steps.last() {
            last_step.tool_calls.is_empty()
        } else {
            false
        }
    }
}

/// Create a stop condition that stops when the last step has no tool calls.
///
/// Similar to Vercel AI SDK's `has_no_tool_calls()` stop condition.
///
/// # Example
///
/// ```rust,ignore
/// let condition = has_no_tool_calls(); // Stop when model doesn't call tools
/// ```
pub fn has_no_tool_calls() -> Box<dyn StopCondition> {
    Box::new(HasNoToolCalls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, FunctionCall, ToolCall};

    fn create_step_with_tools(tool_names: Vec<&str>) -> StepResult {
        use crate::types::ContentPart;

        StepResult {
            messages: vec![],
            finish_reason: None,
            usage: None,
            tool_calls: tool_names
                .into_iter()
                .map(|name| {
                    ContentPart::tool_call(
                        format!("call_{}", name),
                        name,
                        serde_json::json!({}),
                        None,
                    )
                })
                .collect(),
        }
    }

    fn create_step_without_tools() -> StepResult {
        StepResult {
            messages: vec![ChatMessage::assistant("Final answer").build()],
            finish_reason: None,
            usage: None,
            tool_calls: vec![],
        }
    }

    #[test]
    fn test_step_count_is() {
        let condition = step_count_is(3);
        let steps = vec![
            create_step_with_tools(vec!["tool1"]),
            create_step_with_tools(vec!["tool2"]),
        ];

        assert!(!condition.should_stop(&steps)); // 2 steps < 3

        let steps = vec![
            create_step_with_tools(vec!["tool1"]),
            create_step_with_tools(vec!["tool2"]),
            create_step_with_tools(vec!["tool3"]),
        ];

        assert!(condition.should_stop(&steps)); // 3 steps >= 3
    }

    #[test]
    fn test_has_tool_call() {
        let condition = has_tool_call("finalAnswer");

        let steps = vec![create_step_with_tools(vec!["search", "calculate"])];
        assert!(!condition.should_stop(&steps));

        let steps = vec![create_step_with_tools(vec!["finalAnswer"])];
        assert!(condition.should_stop(&steps));
    }

    #[test]
    fn test_has_text_response() {
        let condition = has_text_response();

        let steps = vec![create_step_with_tools(vec!["tool1"])];
        assert!(!condition.should_stop(&steps));

        let steps = vec![create_step_without_tools()];
        assert!(condition.should_stop(&steps));
    }

    #[test]
    fn test_any_of() {
        let condition = any_of(vec![step_count_is(5), has_tool_call("finalAnswer")]);

        // Should stop when finalAnswer is called, even if step count < 5
        let steps = vec![create_step_with_tools(vec!["finalAnswer"])];
        assert!(condition.should_stop(&steps));

        // Should stop when step count reaches 5
        let steps = vec![
            create_step_with_tools(vec!["tool1"]),
            create_step_with_tools(vec!["tool2"]),
            create_step_with_tools(vec!["tool3"]),
            create_step_with_tools(vec!["tool4"]),
            create_step_with_tools(vec!["tool5"]),
        ];
        assert!(condition.should_stop(&steps));
    }

    #[test]
    fn test_custom_condition() {
        let condition = custom_condition(|steps| steps.iter().any(|s| s.tool_calls.len() > 2));

        let steps = vec![create_step_with_tools(vec!["tool1", "tool2"])];
        assert!(!condition.should_stop(&steps));

        let steps = vec![create_step_with_tools(vec!["tool1", "tool2", "tool3"])];
        assert!(condition.should_stop(&steps));
    }

    #[test]
    fn test_has_tool_result() {
        let condition = has_tool_result();

        // No steps yet
        assert!(!condition.should_stop(&vec![]));

        // Step with no tool results
        let step_no_results = StepResult {
            messages: vec![],
            finish_reason: None,
            usage: None,
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
        };
        assert!(!condition.should_stop(&vec![step_no_results.clone()]));

        // Step with tool results
        let step_with_results = StepResult {
            messages: vec![],
            finish_reason: None,
            usage: None,
            tool_calls: vec![],
            tool_results: vec![crate::types::ContentPart::Text {
                text: "result".to_string(),
            }],
            warnings: None,
        };
        assert!(condition.should_stop(&vec![step_with_results]));
    }

    #[test]
    fn test_has_no_tool_calls() {
        let condition = has_no_tool_calls();

        // No steps yet
        assert!(!condition.should_stop(&vec![]));

        // Step with tool calls
        let step_with_calls = create_step_with_tools(vec!["tool1"]);
        assert!(!condition.should_stop(&vec![step_with_calls]));

        // Step with no tool calls
        let step_no_calls = StepResult {
            messages: vec![],
            finish_reason: None,
            usage: None,
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
        };
        assert!(condition.should_stop(&vec![step_no_calls]));
    }
}
