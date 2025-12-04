//! Comprehensive tests for the orchestrator module.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::*;
use siumai::error::LlmError;
use siumai::types::{
    ChatMessage, ChatResponse, ContentPart, FinishReason, MessageContent, Tool, Usage,
};

// ============================================================================
// Mock Implementations
// ============================================================================

/// Mock ChatCapability for testing
#[derive(Clone)]
struct MockChatModel {
    /// Responses to return for each call
    responses: Arc<Mutex<Vec<ChatResponse>>>,
    /// Track calls made
    calls: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
}

impl MockChatModel {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[allow(dead_code)]
    fn get_calls(&self) -> Vec<Vec<ChatMessage>> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl siumai::traits::ChatCapability for MockChatModel {
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse, LlmError> {
        self.calls.lock().unwrap().push(messages.clone());
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Err(LlmError::InternalError("No more mock responses".into()));
        }
        Ok(responses.remove(0))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<siumai::streaming::ChatStream, LlmError> {
        // Not used in these tests
        Err(LlmError::InternalError(
            "Stream not implemented in mock".into(),
        ))
    }

    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.chat(messages).await
    }
}

/// Mock ToolResolver for testing
struct MockToolResolver {
    /// Map of tool name to result
    results: Arc<Mutex<std::collections::HashMap<String, Value>>>,
    /// Track calls made
    calls: Arc<Mutex<Vec<(String, Value)>>>,
}

impl MockToolResolver {
    fn new() -> Self {
        Self {
            results: Arc::new(Mutex::new(std::collections::HashMap::new())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn with_result(self, tool_name: impl Into<String>, result: Value) -> Self {
        self.results
            .lock()
            .unwrap()
            .insert(tool_name.into(), result);
        self
    }

    fn get_calls(&self) -> Vec<(String, Value)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl ToolResolver for MockToolResolver {
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
        self.calls
            .lock()
            .unwrap()
            .push((name.to_string(), arguments.clone()));

        let results = self.results.lock().unwrap();
        results
            .get(name)
            .cloned()
            .ok_or_else(|| LlmError::InternalError(format!("Unknown tool: {}", name)))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_tool_call(name: &str, args: Value) -> ContentPart {
    ContentPart::tool_call(format!("call_{}", name), name.to_string(), args, None)
}

fn create_response_with_tools(tool_calls: Vec<ContentPart>) -> ChatResponse {
    let parts: Vec<ContentPart> = tool_calls;

    let mut response = ChatResponse::new(MessageContent::MultiModal(parts));
    response.finish_reason = Some(FinishReason::ToolCalls);
    response.usage = Some(Usage::new(100, 50));
    response
}

fn create_text_response(text: &str) -> ChatResponse {
    let mut response = ChatResponse::new(MessageContent::Text(text.to_string()));
    response.finish_reason = Some(FinishReason::Stop);
    response.usage = Some(Usage::new(100, 50));
    response
}

fn create_tool(name: &str) -> Tool {
    Tool::function(
        name.to_string(),
        format!("{} tool", name),
        json!({
            "type": "object",
            "properties": {},
        }),
    )
}

// ============================================================================
// Tests for StepResult
// ============================================================================

#[test]
fn test_step_result_merge_usage() {
    let steps = vec![
        StepResult {
            messages: vec![],
            finish_reason: None,
            usage: Some(
                Usage::builder()
                    .prompt_tokens(100)
                    .completion_tokens(50)
                    .total_tokens(150)
                    .with_cached_tokens(10)
                    .with_reasoning_tokens(5)
                    .build(),
            ),
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
        },
        StepResult {
            messages: vec![],
            finish_reason: None,
            usage: Some(
                Usage::builder()
                    .prompt_tokens(200)
                    .completion_tokens(100)
                    .total_tokens(300)
                    .with_cached_tokens(20)
                    .with_reasoning_tokens(10)
                    .build(),
            ),
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
        },
    ];

    let merged = StepResult::merge_usage(&steps).unwrap();
    assert_eq!(merged.prompt_tokens, 300); // 100 + 200
    assert_eq!(merged.completion_tokens, 150); // 50 + 100
    assert_eq!(merged.total_tokens, 450); // 150 + 300 (simple addition, Vercel AI style)
    assert_eq!(
        merged
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens),
        Some(30)
    ); // 10 + 20
    assert_eq!(
        merged
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        Some(15)
    ); // 5 + 10
}

#[test]
fn test_step_result_merge_usage_empty() {
    let steps: Vec<StepResult> = vec![];
    let merged = StepResult::merge_usage(&steps);
    assert!(merged.is_none());
}

// ============================================================================
// Tests for generate() function
// ============================================================================

#[tokio::test]
async fn test_generate_simple_text_response() {
    let model = MockChatModel::new(vec![create_text_response("Hello, world!")]);

    let messages = vec![ChatMessage::user("Hi").build()];
    let stop_conditions: Vec<&dyn StopCondition> = vec![];

    let (response, steps) = generate(
        &model,
        messages,
        None,
        None,
        &stop_conditions,
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(response.content_text().unwrap(), "Hello, world!");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].tool_calls.len(), 0);
}

#[tokio::test]
async fn test_generate_with_tool_calls() {
    let tool_call = create_tool_call("get_weather", json!({"city": "Tokyo"}));
    let responses = vec![
        create_response_with_tools(vec![tool_call]),
        create_text_response("The weather in Tokyo is sunny, 25°C."),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result(
        "get_weather",
        json!({"temperature": 25, "condition": "sunny"}),
    );

    let messages = vec![ChatMessage::user("What's the weather in Tokyo?").build()];
    let tools = vec![create_tool("get_weather")];
    let stop_conditions: Vec<&dyn StopCondition> = vec![];

    let (response, steps) = generate(
        &model,
        messages,
        Some(tools),
        Some(&resolver),
        &stop_conditions,
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        response.content_text().unwrap(),
        "The weather in Tokyo is sunny, 25°C."
    );
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].tool_calls.len(), 1);
    assert_eq!(steps[1].tool_calls.len(), 0);

    // Verify tool was called
    let tool_calls = resolver.get_calls();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].0, "get_weather");
}

#[tokio::test]
async fn test_generate_with_stop_condition() {
    let tool_call = create_tool_call("search", json!({"query": "test"}));
    let responses = vec![
        create_response_with_tools(vec![tool_call.clone()]),
        create_response_with_tools(vec![tool_call.clone()]),
        create_response_with_tools(vec![tool_call]),
        create_text_response("Done"),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("search", json!({"results": []}));

    let messages = vec![ChatMessage::user("Search for something").build()];
    let tools = vec![create_tool("search")];
    let stop_condition = step_count_is(2);
    let stop_conditions: Vec<&dyn StopCondition> = vec![&*stop_condition];

    let (_response, steps) = generate(
        &model,
        messages,
        Some(tools),
        Some(&resolver),
        &stop_conditions,
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    // Should stop after 2 steps due to stop condition
    assert_eq!(steps.len(), 2);
}

#[tokio::test]
async fn test_generate_with_tool_approval() {
    let tool_call = create_tool_call("dangerous_operation", json!({"action": "delete"}));
    let responses = vec![
        create_response_with_tools(vec![tool_call]),
        create_text_response("Operation denied"),
    ];

    let model = MockChatModel::new(responses);
    let resolver =
        MockToolResolver::new().with_result("dangerous_operation", json!({"status": "executed"}));

    let messages = vec![ChatMessage::user("Do something dangerous").build()];
    let tools = vec![create_tool("dangerous_operation")];

    let approval_called = Arc::new(Mutex::new(false));
    let approval_called_clone = approval_called.clone();

    let options = OrchestratorOptions {
        on_tool_approval: Some(Arc::new(move |tool_name, _args| {
            *approval_called_clone.lock().unwrap() = true;
            if tool_name == "dangerous_operation" {
                ToolApproval::Deny {
                    reason: "Operation not allowed".to_string(),
                }
            } else {
                ToolApproval::Approve(_args.clone())
            }
        })),
        ..Default::default()
    };

    let (_response, steps) = generate(&model, messages, Some(tools), Some(&resolver), &[], options)
        .await
        .unwrap();

    // Verify approval callback was called
    assert!(*approval_called.lock().unwrap());

    // Tool should not have been executed (denied)
    let tool_calls = resolver.get_calls();
    assert_eq!(tool_calls.len(), 0);

    // But we should have a tool message with error
    assert_eq!(steps.len(), 2);
}

#[tokio::test]
async fn test_generate_with_callbacks() {
    let responses = vec![
        create_response_with_tools(vec![create_tool_call("tool1", json!({}))]),
        create_text_response("Done"),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("tool1", json!({"result": "ok"}));

    let step_finish_called = Arc::new(Mutex::new(0));
    let step_finish_clone = step_finish_called.clone();

    let finish_called = Arc::new(Mutex::new(false));
    let finish_clone = finish_called.clone();

    let options = OrchestratorOptions {
        on_step_finish: Some(Arc::new(move |_step| {
            *step_finish_clone.lock().unwrap() += 1;
        })),
        on_finish: Some(Arc::new(move |_steps| {
            *finish_clone.lock().unwrap() = true;
        })),
        ..Default::default()
    };

    let messages = vec![ChatMessage::user("Test").build()];
    let tools = vec![create_tool("tool1")];

    let (_response, _steps) =
        generate(&model, messages, Some(tools), Some(&resolver), &[], options)
            .await
            .unwrap();

    // Verify callbacks were called
    assert_eq!(*step_finish_called.lock().unwrap(), 2); // 2 steps
    assert!(*finish_called.lock().unwrap());
}

#[tokio::test]
async fn test_generate_max_steps_reached() {
    // Create infinite loop of tool calls
    let tool_call = create_tool_call("loop_tool", json!({}));
    let responses = vec![
        create_response_with_tools(vec![tool_call.clone()]),
        create_response_with_tools(vec![tool_call.clone()]),
        create_response_with_tools(vec![tool_call.clone()]),
        create_response_with_tools(vec![tool_call.clone()]),
        create_response_with_tools(vec![tool_call]),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("loop_tool", json!({"continue": true}));

    let messages = vec![ChatMessage::user("Start loop").build()];
    let tools = vec![create_tool("loop_tool")];

    let options = OrchestratorOptions {
        max_steps: 3,
        ..Default::default()
    };

    let (_response, steps) = generate(&model, messages, Some(tools), Some(&resolver), &[], options)
        .await
        .unwrap();

    // Should stop at max_steps
    assert_eq!(steps.len(), 3);
}

// ============================================================================
// Tests for ToolLoopAgent
// ============================================================================

#[tokio::test]
async fn test_agent_basic_usage() {
    let responses = vec![create_text_response("Agent response")];
    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new();

    let agent = ToolLoopAgent::new(model, vec![], vec![step_count_is(10)])
        .with_system("You are a helpful assistant")
        .with_id("test-agent");

    assert_eq!(agent.id(), Some("test-agent"));
    assert_eq!(agent.tools().len(), 0);

    let messages = vec![ChatMessage::user("Hello").build()];
    let AgentResult {
        response, steps, ..
    } = agent.generate(messages, &resolver).await.unwrap();

    assert_eq!(response.content_text().unwrap(), "Agent response");
    assert_eq!(steps.len(), 1);
}

#[tokio::test]
async fn test_agent_with_tools() {
    let tool_call = create_tool_call("calculator", json!({"operation": "add", "a": 2, "b": 3}));
    let responses = vec![
        create_response_with_tools(vec![tool_call]),
        create_text_response("The result is 5"),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("calculator", json!({"result": 5}));

    let tools = vec![create_tool("calculator")];
    let agent = ToolLoopAgent::new(model, tools, vec![step_count_is(10)]);

    let messages = vec![ChatMessage::user("What is 2 + 3?").build()];
    let AgentResult {
        response, steps, ..
    } = agent.generate(messages, &resolver).await.unwrap();

    assert_eq!(response.content_text().unwrap(), "The result is 5");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].tool_calls.len(), 1);
}

#[tokio::test]
async fn test_agent_with_callbacks() {
    let responses = vec![create_text_response("Done")];
    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new();

    let step_called = Arc::new(Mutex::new(false));
    let step_clone = step_called.clone();

    let finish_called = Arc::new(Mutex::new(false));
    let finish_clone = finish_called.clone();

    let agent = ToolLoopAgent::new(model, vec![], vec![step_count_is(10)])
        .on_step_finish(Arc::new(move |_step| {
            *step_clone.lock().unwrap() = true;
        }))
        .on_finish(Arc::new(move |_steps| {
            *finish_clone.lock().unwrap() = true;
        }));

    let messages = vec![ChatMessage::user("Test").build()];
    let _ = agent.generate(messages, &resolver).await.unwrap();

    assert!(*step_called.lock().unwrap());
    assert!(*finish_called.lock().unwrap());
}

// ============================================================================
// Tests for stop conditions (additional to existing tests)
// ============================================================================

#[test]
fn test_all_of_stop_condition() {
    let condition = all_of(vec![step_count_is(2), has_text_response()]);

    // Should not stop if only one condition is met
    let steps = vec![StepResult {
        messages: vec![ChatMessage::assistant("text").build()],
        finish_reason: None,
        usage: None,
        tool_calls: vec![],
        tool_results: vec![],
        warnings: None,
    }];
    assert!(!condition.should_stop(&steps)); // has_text_response but not step_count

    // Should stop when both conditions are met
    let steps = vec![
        StepResult {
            messages: vec![],
            finish_reason: None,
            usage: None,
            tool_calls: vec![siumai::types::ContentPart::tool_call(
                "call_1",
                "tool1",
                json!({}),
                None,
            )],
            tool_results: vec![],
            warnings: None,
        },
        StepResult {
            messages: vec![ChatMessage::assistant("text").build()],
            finish_reason: None,
            usage: None,
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
        },
    ];
    assert!(condition.should_stop(&steps)); // Both conditions met
}

#[test]
fn test_all_of_empty_conditions() {
    let condition = all_of(vec![]);
    let steps = vec![];
    // Empty all_of should not stop (no conditions to satisfy)
    assert!(!condition.should_stop(&steps));
}

// ============================================================================
// Tests for PrepareStep functionality
// ============================================================================

#[tokio::test]
async fn test_generate_with_prepare_step() {
    use super::prepare_step::{PrepareStepContext, PrepareStepResult, ToolChoice};

    let tool_call = create_tool_call("search", json!({"query": "test"}));
    let responses = vec![
        create_response_with_tools(vec![tool_call]),
        create_text_response("Search completed"),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("search", json!({"results": ["item1"]}));

    let prepare_called = Arc::new(Mutex::new(0));
    let prepare_clone = prepare_called.clone();

    let prepare_fn: super::prepare_step::PrepareStepFn =
        Arc::new(move |ctx: PrepareStepContext| {
            *prepare_clone.lock().unwrap() += 1;
            if ctx.step_number == 0 {
                // Force tool call on first step
                PrepareStepResult::new().with_tool_choice(ToolChoice::Required)
            } else {
                PrepareStepResult::default()
            }
        });

    let options = OrchestratorOptions {
        prepare_step: Some(prepare_fn),
        ..Default::default()
    };

    let messages = vec![ChatMessage::user("Search for something").build()];
    let tools = vec![create_tool("search")];

    let (_response, _steps) =
        generate(&model, messages, Some(tools), Some(&resolver), &[], options)
            .await
            .unwrap();

    // Verify prepare_step was called for each step
    assert_eq!(*prepare_called.lock().unwrap(), 2);
}

// ============================================================================
// Tests for ToolApproval::Modify
// ============================================================================

#[tokio::test]
async fn test_tool_approval_modify() {
    let tool_call = create_tool_call("calculator", json!({"a": 10, "b": 5}));
    let responses = vec![
        create_response_with_tools(vec![tool_call]),
        create_text_response("Modified result"),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("calculator", json!({"result": 100})); // Modified result

    let options = OrchestratorOptions {
        on_tool_approval: Some(Arc::new(|_tool_name, args| {
            // Modify the arguments
            let mut modified = args.clone();
            if let Some(obj) = modified.as_object_mut() {
                obj.insert("a".to_string(), json!(50)); // Change 10 to 50
            }
            ToolApproval::Modify(modified)
        })),
        ..Default::default()
    };

    let messages = vec![ChatMessage::user("Calculate").build()];
    let tools = vec![create_tool("calculator")];

    let (_response, _steps) =
        generate(&model, messages, Some(tools), Some(&resolver), &[], options)
            .await
            .unwrap();

    // Verify tool was called with modified arguments
    let tool_calls = resolver.get_calls();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].1["a"], json!(50)); // Modified value
}

// ============================================================================
// Tests for error handling
// ============================================================================

#[tokio::test]
async fn test_generate_with_tool_error() {
    let tool_call = create_tool_call("failing_tool", json!({}));
    let responses = vec![
        create_response_with_tools(vec![tool_call]),
        create_text_response("Handled error"),
    ];

    let model = MockChatModel::new(responses);
    // Don't register the tool in resolver - it will fail
    let resolver = MockToolResolver::new();

    let messages = vec![ChatMessage::user("Call failing tool").build()];
    let tools = vec![create_tool("failing_tool")];

    let (_response, steps) = generate(
        &model,
        messages,
        Some(tools),
        Some(&resolver),
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    // Should still complete, but with error in tool message
    assert_eq!(steps.len(), 2);
    // First step has tool call
    assert_eq!(steps[0].tool_calls.len(), 1);
}

// ============================================================================
// Tests for edge cases
// ============================================================================

#[tokio::test]
async fn test_generate_no_tools() {
    let responses = vec![create_text_response("Simple response")];
    let model = MockChatModel::new(responses);

    let messages = vec![ChatMessage::user("Hello").build()];

    let (response, steps) = generate(
        &model,
        messages,
        None, // No tools
        None, // No resolver
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(response.content_text().unwrap(), "Simple response");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].tool_calls.len(), 0);
}

#[tokio::test]
async fn test_agent_with_empty_tools() {
    let responses = vec![create_text_response("Response")];
    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new();

    let agent = ToolLoopAgent::new(
        model,
        vec![], // Empty tools
        vec![step_count_is(10)],
    );

    let messages = vec![ChatMessage::user("Test").build()];
    let AgentResult {
        response, steps, ..
    } = agent.generate(messages, &resolver).await.unwrap();

    assert_eq!(response.content_text().unwrap(), "Response");
    assert_eq!(steps.len(), 1);
}

#[test]
fn test_step_result_merge_usage_partial() {
    let steps = vec![
        StepResult {
            messages: vec![],
            finish_reason: None,
            usage: Some(Usage::new(100, 50)),
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
        },
        StepResult {
            messages: vec![],
            finish_reason: None,
            usage: None, // No usage for this step
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
        },
    ];

    let merged = StepResult::merge_usage(&steps).unwrap();
    assert_eq!(merged.prompt_tokens, 100);
    assert_eq!(merged.completion_tokens, 50);
    // Simple addition: only first step has usage, so total_tokens = 150
    assert_eq!(merged.total_tokens, 150);
}
