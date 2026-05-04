//! Comprehensive tests for the orchestrator module.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde_json::{Value, json};
use siumai::types::ToolResultOutput;

use super::*;
use super::{PrepareStepResult, StepResult, StopCondition, step_count_is};
use siumai::experimental::observability::telemetry::TelemetryConfig;
use siumai::prelude::unified::*;

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
    /// Provider id exposed via ModelMetadata.
    provider_id: &'static str,
    /// Model id exposed via ModelMetadata.
    model_id: &'static str,
}

impl MockChatModel {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            calls: Arc::new(Mutex::new(Vec::new())),
            provider_id: "mock-provider",
            model_id: "mock-model",
        }
    }

    fn with_identity(mut self, provider_id: &'static str, model_id: &'static str) -> Self {
        self.provider_id = provider_id;
        self.model_id = model_id;
        self
    }

    #[allow(dead_code)]
    fn get_calls(&self) -> Vec<Vec<ChatMessage>> {
        self.calls.lock().unwrap().clone()
    }
}

impl ModelMetadata for MockChatModel {
    fn provider_id(&self) -> &str {
        self.provider_id
    }

    fn model_id(&self) -> &str {
        self.model_id
    }
}

#[async_trait::async_trait]
impl ChatCapability for MockChatModel {
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
        messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.calls.lock().unwrap().push(messages);

        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Err(LlmError::InternalError("No more mock responses".into()));
        }
        let response = responses.remove(0);
        let s = async_stream::try_stream! {
            #[allow(unreachable_patterns)]
            match &response.content {
                MessageContent::Text(text) if !text.is_empty() => {
                    yield ChatStreamEvent::ContentDelta {
                        delta: text.clone(),
                        index: None,
                    };
                }
                MessageContent::MultiModal(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text, .. } if !text.is_empty() => {
                                yield ChatStreamEvent::ContentDelta {
                                    delta: text.clone(),
                                    index: None,
                                };
                            }
                            ContentPart::ToolCall {
                                tool_call_id,
                                tool_name,
                                arguments,
                                provider_executed,
                                dynamic,
                                ..
                            } => {
                                let input = serde_json::to_string(arguments)
                                    .unwrap_or_else(|_| "{}".to_string());
                                yield ChatStreamEvent::Part {
                                    part: ChatStreamPart::ToolInputStart {
                                        id: tool_call_id.clone(),
                                        tool_name: tool_name.clone(),
                                        provider_metadata: None,
                                        provider_executed: *provider_executed,
                                        dynamic: *dynamic,
                                        title: None,
                                    },
                                };
                                yield ChatStreamEvent::Part {
                                    part: ChatStreamPart::ToolInputDelta {
                                        id: tool_call_id.clone(),
                                        delta: input.clone(),
                                        provider_metadata: None,
                                    },
                                };
                                yield ChatStreamEvent::Part {
                                    part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                                        tool_call_id: tool_call_id.clone(),
                                        tool_name: tool_name.clone(),
                                        input,
                                        provider_executed: *provider_executed,
                                        dynamic: *dynamic,
                                        provider_metadata: None,
                                    }),
                                };
                            }
                            ContentPart::ToolApprovalRequest {
                                approval_id,
                                tool_call_id,
                                provider_metadata,
                                ..
                            } => {
                                yield ChatStreamEvent::Part {
                                    part: ChatStreamPart::ToolApprovalRequest(
                                        ChatStreamToolApprovalRequest {
                                            approval_id: approval_id.clone(),
                                            tool_call_id: tool_call_id.clone(),
                                            provider_metadata: provider_metadata.clone(),
                                        },
                                    ),
                                };
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }

            yield ChatStreamEvent::StreamEnd { response };
        };

        Ok(Box::pin(s))
    }

    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.chat(messages).await
    }
}

#[derive(Clone, Default)]
struct MockStableTextStreamModel;

impl ModelMetadata for MockStableTextStreamModel {
    fn provider_id(&self) -> &str {
        "mock-provider"
    }

    fn model_id(&self) -> &str {
        "mock-stable-text-model"
    }
}

#[async_trait::async_trait]
impl ChatCapability for MockStableTextStreamModel {
    async fn chat(&self, _messages: Vec<ChatMessage>) -> Result<ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation("stream only".into()))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let s = async_stream::try_stream! {
            yield ChatStreamEvent::Part {
                part: ChatStreamPart::TextDelta {
                    id: "txt_1".to_string(),
                    delta: "stream stable".to_string(),
                    provider_metadata: None,
                },
            };
        };

        Ok(Box::pin(s))
    }

    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.chat(Vec::new()).await
    }
}

/// Mock ToolResolver for testing
struct MockToolResolver {
    /// Map of tool name to result
    results: Arc<Mutex<std::collections::HashMap<String, Value>>>,
    /// Runtime metadata keyed by tool name.
    runtime_metadata:
        Arc<Mutex<std::collections::HashMap<String, siumai::tooling::ToolRuntimeMetadata>>>,
    /// Track calls made
    calls: Arc<Mutex<Vec<(String, Value)>>>,
}

impl MockToolResolver {
    fn new() -> Self {
        Self {
            results: Arc::new(Mutex::new(std::collections::HashMap::new())),
            runtime_metadata: Arc::new(Mutex::new(std::collections::HashMap::new())),
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

    fn with_runtime_metadata(
        self,
        tool_name: impl Into<String>,
        metadata: siumai::tooling::ToolRuntimeMetadata,
    ) -> Self {
        self.runtime_metadata
            .lock()
            .unwrap()
            .insert(tool_name.into(), metadata);
        self
    }

    fn get_calls(&self) -> Vec<(String, Value)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl ToolResolver for MockToolResolver {
    fn runtime_tool_metadata(&self, name: &str) -> Option<siumai::tooling::ToolRuntimeMetadata> {
        self.runtime_metadata.lock().unwrap().get(name).cloned()
    }

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

/// Context-aware ToolResolver for validating runtime context propagation.
#[derive(Clone)]
struct ContextAwareToolResolver {
    contexts: Arc<Mutex<Vec<OrchestratorContext>>>,
    calls: Arc<Mutex<Vec<(String, Value)>>>,
}

impl ContextAwareToolResolver {
    fn new() -> Self {
        Self {
            contexts: Arc::new(Mutex::new(Vec::new())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_contexts(&self) -> Vec<OrchestratorContext> {
        self.contexts.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl ToolResolver for ContextAwareToolResolver {
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
        self.calls
            .lock()
            .unwrap()
            .push((name.to_string(), arguments.clone()));

        Ok(json!({
            "tool": name,
            "arguments": arguments,
        }))
    }

    async fn call_tool_stream_with_context(
        &self,
        name: &str,
        arguments: Value,
        context: &OrchestratorContext,
    ) -> Result<futures::stream::BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError>
    {
        self.calls
            .lock()
            .unwrap()
            .push((name.to_string(), arguments.clone()));
        self.contexts.lock().unwrap().push(context.clone());

        let output = json!({
            "tool": name,
            "phase": context.get("phase").cloned(),
            "step_marker": context.get("step_marker").cloned(),
        });

        Ok(Box::pin(futures::stream::once(async move {
            Ok(ToolExecutionResult::final_result(output))
        })))
    }
}

/// Runtime-options-aware resolver for validating shared tool execution inputs.
#[derive(Clone)]
struct RuntimeOptionsAwareToolResolver {
    calls: Arc<Mutex<Vec<(String, Value)>>>,
    options: Arc<Mutex<Vec<siumai::tooling::ToolExecutionOptions>>>,
}

impl RuntimeOptionsAwareToolResolver {
    fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            options: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_calls(&self) -> Vec<(String, Value)> {
        self.calls.lock().unwrap().clone()
    }

    fn get_options(&self) -> Vec<siumai::tooling::ToolExecutionOptions> {
        self.options.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl ToolResolver for RuntimeOptionsAwareToolResolver {
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
        self.calls
            .lock()
            .unwrap()
            .push((name.to_string(), arguments.clone()));
        Ok(json!({
            "tool": name,
            "arguments": arguments,
        }))
    }

    async fn call_tool_stream_with_runtime_options(
        &self,
        name: &str,
        arguments: Value,
        options: siumai::tooling::ToolExecutionOptions,
    ) -> Result<futures::stream::BoxStream<'static, Result<ToolExecutionResult, LlmError>>, LlmError>
    {
        self.calls
            .lock()
            .unwrap()
            .push((name.to_string(), arguments.clone()));
        self.options.lock().unwrap().push(options);

        let output = json!({
            "tool": name,
            "arguments": arguments,
        });

        Ok(Box::pin(futures::stream::once(async move {
            Ok(ToolExecutionResult::final_result(output))
        })))
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

fn create_empty_step_response() -> ChatResponse {
    ChatResponse::new(MessageContent::Text(String::new()))
}

fn create_step_text_response(text: &str) -> ChatResponse {
    ChatResponse::new(MessageContent::Text(text.to_string()))
}

fn create_empty_step_request() -> ChatRequest {
    ChatRequest::default()
}

fn test_call_id() -> String {
    "call-test".to_string()
}

fn test_step_model() -> StepModelInfo {
    StepModelInfo {
        provider: "mock-provider".to_string(),
        model_id: "mock-model".to_string(),
    }
}

fn test_context() -> OrchestratorContext {
    OrchestratorContext::default()
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

fn create_deferred_provider_tool(name: &str) -> Tool {
    Tool::provider_defined(format!("mock.{name}"), name.to_string())
        .with_supports_deferred_results(true)
}

fn create_provider_tool_call(tool_call_id: &str, name: &str, args: Value) -> ContentPart {
    ContentPart::tool_call(tool_call_id.to_string(), name.to_string(), args, Some(true))
}

fn create_provider_tool_result(tool_call_id: &str, name: &str, result: Value) -> ContentPart {
    ContentPart::ToolResult {
        tool_call_id: tool_call_id.to_string(),
        tool_name: name.to_string(),
        output: ToolResultOutput::json(result),
        input: None,
        provider_executed: Some(true),
        dynamic: None,
        preliminary: None,
        title: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    }
}

fn create_assistant_tool_approval_request_message(
    tool_call: ContentPart,
    approval_id: &str,
) -> ChatMessage {
    let tool_call_id = tool_call
        .as_tool_call()
        .expect("tool call part")
        .tool_call_id
        .to_string();
    ChatMessage::assistant_with_content(vec![
        tool_call,
        ContentPart::tool_approval_request(approval_id.to_string(), tool_call_id),
    ])
    .build()
}

fn create_tool_approval_response_message(
    approval_id: &str,
    approved: bool,
    reason: Option<&str>,
    provider_executed: Option<bool>,
) -> ChatMessage {
    let mut response = ContentPart::tool_approval_response_with_reason(
        approval_id.to_string(),
        approved,
        reason.map(ToString::to_string),
    );
    if let ContentPart::ToolApprovalResponse {
        provider_executed: slot,
        ..
    } = &mut response
    {
        *slot = provider_executed;
    }

    ChatMessage {
        role: MessageRole::Tool,
        content: MessageContent::MultiModal(vec![response]),
        provider_options: ProviderOptionsMap::default(),
        metadata: MessageMetadata::default(),
    }
}

// ============================================================================
// Tests for StepResult
// ============================================================================

#[test]
fn test_step_result_merge_usage() {
    let steps = vec![
        StepResult {
            call_id: test_call_id(),
            step_number: 0,
            model: test_step_model(),
            request: create_empty_step_request(),
            response: create_empty_step_response(),
            raw_finish_reason: None,
            function_id: None,
            metadata: None,
            context: test_context(),
            content: vec![],
            messages: vec![],
            finish_reason: None,
            usage: Some(
                Usage::builder()
                    .prompt_tokens(100)
                    .completion_tokens(50)
                    .total_tokens(150)
                    .with_cached_tokens(10)
                    .with_reasoning_tokens(5)
                    .with_raw_usage_value(serde_json::json!({
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "provider": "step-1"
                    }))
                    .build(),
            ),
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
            provider_metadata: None,
        },
        StepResult {
            call_id: test_call_id(),
            step_number: 1,
            model: test_step_model(),
            request: create_empty_step_request(),
            response: create_empty_step_response(),
            raw_finish_reason: None,
            function_id: None,
            metadata: None,
            context: test_context(),
            content: vec![],
            messages: vec![],
            finish_reason: None,
            usage: Some(
                Usage::builder()
                    .prompt_tokens(200)
                    .completion_tokens(100)
                    .total_tokens(300)
                    .with_cached_tokens(20)
                    .with_reasoning_tokens(10)
                    .with_raw_usage_value(serde_json::json!({
                        "input_tokens": 200,
                        "output_tokens": 100,
                        "provider": "step-2"
                    }))
                    .build(),
            ),
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
            provider_metadata: None,
        },
    ];

    let merged = StepResult::merge_usage(&steps).unwrap();
    assert_eq!(merged.prompt_tokens(), Some(300)); // 100 + 200
    assert_eq!(merged.completion_tokens(), Some(150)); // 50 + 100
    assert_eq!(merged.total_tokens(), Some(450)); // 150 + 300 (simple addition, Vercel AI style)
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
    assert_eq!(merged.raw_usage_value(), None);
}

#[test]
fn test_step_result_merge_usage_empty() {
    let steps: Vec<StepResult> = vec![];
    let merged = StepResult::merge_usage(&steps);
    assert!(merged.is_none());
}

#[test]
fn test_step_result_content_projections() {
    let response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("hello"),
        ContentPart::text(" world"),
        ContentPart::reasoning("think"),
        ContentPart::file_url("https://example.com/file.txt", "text/plain"),
        ContentPart::source_url("id-1", "https://example.com", "Example"),
        ContentPart::tool_call("call_search", "search", json!({"q": "rust"}), None),
        ContentPart::tool_call(
            "call_dynamic",
            "dynamic-tool",
            json!({"q": "dynamic"}),
            None,
        ),
    ]));
    let tool_results = vec![
        ContentPart::tool_result_json("call_search", "search", json!({"ok": true})),
        ContentPart::tool_result_json("call_dynamic", "dynamic-tool", json!({"ok": "dynamic"})),
    ];
    let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
        .with_tools(vec![create_tool("search")]);

    let step = StepResult {
        call_id: test_call_id(),
        step_number: 0,
        model: test_step_model(),
        request,
        response: response.clone(),
        raw_finish_reason: None,
        function_id: None,
        metadata: None,
        context: test_context(),
        content: StepResult::compose_content(&response, &tool_results),
        messages: vec![],
        finish_reason: None,
        usage: None,
        tool_calls: vec![
            ContentPart::tool_call("call_search", "search", json!({"q": "rust"}), None),
            ContentPart::tool_call(
                "call_dynamic",
                "dynamic-tool",
                json!({"q": "dynamic"}),
                None,
            ),
        ],
        tool_results,
        warnings: None,
        provider_metadata: None,
    };

    assert_eq!(step.content().len(), 9);
    assert_eq!(step.text().as_deref(), Some("hello world"));
    assert_eq!(step.reasoning_parts().len(), 1);
    assert_eq!(step.reasoning_text().as_deref(), Some("think"));
    assert_eq!(step.files().len(), 1);
    assert_eq!(step.sources().len(), 1);
    assert_eq!(step.tool_results().len(), 2);
    assert_eq!(step.tool_call_views().len(), 2);
    assert_eq!(step.static_tool_calls().len(), 1);
    assert_eq!(step.dynamic_tool_calls().len(), 1);
    assert_eq!(step.static_tool_results().len(), 1);
    assert_eq!(step.dynamic_tool_results().len(), 1);
    let expected_static_input = json!({"q": "rust"});
    let expected_dynamic_input = json!({"q": "dynamic"});
    assert_eq!(
        step.static_tool_results()[0].input,
        Some(&expected_static_input)
    );
    assert_eq!(
        step.dynamic_tool_results()[0].input,
        Some(&expected_dynamic_input)
    );
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
    assert_eq!(steps[0].model, test_step_model());
}

#[tokio::test]
async fn test_generate_step_preserves_raw_finish_reason() {
    let mut response = create_text_response("Hello, raw!");
    response.raw_finish_reason = Some("stop".to_string());

    let model = MockChatModel::new(vec![response]);

    let (_response, steps) = generate(
        &model,
        vec![ChatMessage::user("Hi").build()],
        None,
        None,
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].raw_finish_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn test_generate_prepare_step_can_swap_model() {
    let base_model = MockChatModel::new(vec![create_text_response("base")]);
    let base_model_handle = base_model.clone();
    let override_model = MockChatModel::new(vec![create_text_response("override")])
        .with_identity("override-provider", "override-model");
    let override_model_handle = override_model.clone();

    let override_model: StepLanguageModel = Arc::new(override_model);
    let options = OrchestratorOptions {
        prepare_step: Some(Arc::new(move |ctx| {
            assert_eq!(ctx.model.provider_id(), "mock-provider");
            assert_eq!(ctx.model.model_id(), "mock-model");

            if ctx.step_number == 0 {
                PrepareStepResult::new().with_model(override_model.clone())
            } else {
                PrepareStepResult::default()
            }
        })),
        ..Default::default()
    };

    let (response, steps) = generate(
        &base_model,
        vec![ChatMessage::user("swap").build()],
        None,
        None,
        &[],
        options,
    )
    .await
    .unwrap();

    assert_eq!(response.content_text(), Some("override"));
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].model.provider, "override-provider");
    assert_eq!(steps[0].model.model_id, "override-model");
    assert_eq!(base_model_handle.get_calls().len(), 0);
    assert_eq!(override_model_handle.get_calls().len(), 1);
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
async fn test_generate_continues_for_deferred_provider_tool_results() {
    let responses = vec![
        create_response_with_tools(vec![create_provider_tool_call(
            "call_code_execution",
            "code_execution",
            json!({"code":"print('hi')"}),
        )]),
        ChatResponse::new(MessageContent::MultiModal(vec![
            create_provider_tool_result(
                "call_code_execution",
                "code_execution",
                json!({"stdout":"hi"}),
            ),
            ContentPart::text("Deferred result received"),
        ])),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new();
    let tools = vec![create_deferred_provider_tool("code_execution")];

    let (response, steps) = generate(
        &model,
        vec![ChatMessage::user("Run code").build()],
        Some(tools),
        Some(&resolver),
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(steps.len(), 2);
    assert_eq!(resolver.get_calls().len(), 0);
    assert_eq!(steps[0].tool_calls.len(), 1);
    assert_eq!(steps[0].tool_results.len(), 0);
    assert_eq!(steps[1].tool_calls.len(), 0);
    assert_eq!(steps[1].tool_results.len(), 1);
    assert_eq!(
        steps[1].tool_result_views()[0].input,
        None,
        "deferred provider tool results should not require a same-step call input"
    );
    assert_eq!(
        steps[1].tool_result_views()[0].provider_executed,
        Some(true)
    );
    assert_eq!(response.text().as_deref(), Some("Deferred result received"));
}

#[tokio::test]
async fn test_generate_with_executable_tools_resolver() {
    use siumai::tooling::{ExecutableTool, ExecutableTools};

    let tool_call = create_tool_call("get_weather", json!({"city": "Tokyo"}));
    let responses = vec![
        create_response_with_tools(vec![tool_call]),
        create_text_response("The weather in Tokyo is sunny, 25°C."),
    ];

    let model = MockChatModel::new(responses);

    let tools = ExecutableTools::from_tools([ExecutableTool::function(
        "get_weather",
        "Get weather for a city",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
        |args| async move {
            let city = args
                .get("city")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string();
            Ok(json!({"city": city, "temperature": 25, "condition": "sunny"}))
        },
    )]);

    let messages = vec![ChatMessage::user("What's the weather in Tokyo?").build()];
    let stop_conditions: Vec<&dyn StopCondition> = vec![];

    let (response, steps) = generate(
        &model,
        messages,
        Some(tools.schemas()),
        Some(&tools),
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
    assert_eq!(steps[0].tool_results.len(), 1);
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
async fn test_generate_surfaces_tool_approval_request_from_runtime_metadata() {
    use siumai::tooling::ExecutableTool;

    let tool_call = create_tool_call("dangerous_operation", json!({"action": "delete"}));
    let responses = vec![create_response_with_tools(vec![tool_call])];
    let model = MockChatModel::new(responses);

    let seen_inputs = Arc::new(Mutex::new(Vec::<Value>::new()));
    let seen_inputs_clone = seen_inputs.clone();
    let seen_messages = Arc::new(Mutex::new(Vec::<Vec<ModelMessage>>::new()));
    let seen_messages_clone = seen_messages.clone();
    let seen_contexts = Arc::new(Mutex::new(Vec::<Context>::new()));
    let seen_contexts_clone = seen_contexts.clone();
    let seen_abort_flags = Arc::new(Mutex::new(Vec::<bool>::new()));
    let seen_abort_flags_clone = seen_abort_flags.clone();
    let runtime_metadata = ExecutableTool::function(
        "dangerous_operation",
        "Dangerous operation",
        json!({"type":"object"}),
        |args| async move { Ok(args) },
    )
    .with_needs_approval(true)
    .with_on_input_available_fn(move |context| {
        let seen_inputs = seen_inputs_clone.clone();
        let seen_messages = seen_messages_clone.clone();
        let seen_contexts = seen_contexts_clone.clone();
        let seen_abort_flags = seen_abort_flags_clone.clone();
        async move {
            seen_inputs.lock().unwrap().push(context.input);
            seen_messages.lock().unwrap().push(context.messages);
            seen_contexts.lock().unwrap().push(context.context);
            seen_abort_flags
                .lock()
                .unwrap()
                .push(context.abort_signal.is_some());
            Ok(())
        }
    })
    .runtime_metadata()
    .clone();

    let resolver = MockToolResolver::new()
        .with_result("dangerous_operation", json!({"status": "executed"}))
        .with_runtime_metadata("dangerous_operation", runtime_metadata);

    let mut options = OrchestratorOptions::default();
    options.context.insert("requestId", json!("req_generate"));

    let (response, steps) = generate(
        &model,
        vec![ChatMessage::user("Delete it").build()],
        Some(vec![create_tool("dangerous_operation")]),
        Some(&resolver),
        &[],
        options,
    )
    .await
    .unwrap();

    assert_eq!(steps.len(), 1);
    assert_eq!(resolver.get_calls().len(), 0);
    assert_eq!(
        seen_inputs.lock().unwrap().as_slice(),
        &[json!({"action": "delete"})]
    );
    assert_eq!(seen_messages.lock().unwrap().len(), 1);
    assert!(matches!(
        seen_messages.lock().unwrap()[0].first(),
        Some(ModelMessage::User(_))
    ));
    assert_eq!(
        seen_contexts.lock().unwrap()[0].get("requestId"),
        Some(&json!("req_generate"))
    );
    assert_eq!(seen_abort_flags.lock().unwrap().as_slice(), &[false]);

    let approval_count = match &response.content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| matches!(part, ContentPart::ToolApprovalRequest { .. }))
            .count(),
        _ => 0,
    };
    assert_eq!(approval_count, 1);
}

#[tokio::test]
async fn test_generate_preprocesses_local_tool_approval_response_and_executes_tool() {
    let approval_id = "approval_secure";
    let messages = vec![
        ChatMessage::user("Approve local tool").build(),
        create_assistant_tool_approval_request_message(
            create_tool_call("secure_tool", json!({"path": "tmp"})),
            approval_id,
        ),
        create_tool_approval_response_message(approval_id, true, None, None),
    ];

    let model = MockChatModel::new(vec![create_text_response("Local approval continued")]);
    let resolver = MockToolResolver::new().with_result("secure_tool", json!({"ok": true}));

    let (response, steps) = generate(
        &model,
        messages,
        Some(vec![create_tool("secure_tool")]),
        Some(&resolver),
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(response.text().as_deref(), Some("Local approval continued"));
    assert_eq!(steps.len(), 1);
    assert_eq!(
        resolver.get_calls(),
        vec![("secure_tool".to_string(), json!({"path": "tmp"}))]
    );

    let model_calls = model.get_calls();
    assert_eq!(model_calls.len(), 1);
    let tool_result = model_calls[0]
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .find_map(|message| match &message.content {
            MessageContent::MultiModal(parts) => parts.iter().find(|part| {
                matches!(
                    part,
                    ContentPart::ToolResult { tool_call_id, .. }
                        if tool_call_id == "call_secure_tool"
                )
            }),
            _ => None,
        })
        .expect("forwarded tool result");
    let ContentPart::ToolResult { input, output, .. } = tool_result else {
        panic!("expected tool result");
    };
    assert_eq!(input.as_ref(), Some(&json!({"path": "tmp"})));
    match output {
        ToolResultOutput::Json { value, .. } => assert_eq!(value, &json!({"ok": true})),
        _ => panic!("expected json output"),
    }
}

#[tokio::test]
async fn test_generate_preprocesses_local_tool_approval_response_preserves_runtime_messages() {
    let approval_id = "approval_secure_runtime";
    let messages = vec![
        ChatMessage::user("Approve local tool with runtime context").build(),
        create_assistant_tool_approval_request_message(
            create_tool_call("secure_tool_runtime", json!({"path": "tmp"})),
            approval_id,
        ),
        create_tool_approval_response_message(approval_id, true, None, None),
    ];

    let model = MockChatModel::new(vec![create_text_response("Local approval continued")]);
    let resolver = RuntimeOptionsAwareToolResolver::new();

    let mut options = OrchestratorOptions::default();
    options
        .context
        .insert("requestId", json!("req_approval_generate"));

    let (_response, steps) = generate(
        &model,
        messages,
        Some(vec![create_tool("secure_tool_runtime")]),
        Some(&resolver),
        &[],
        options,
    )
    .await
    .unwrap();

    assert_eq!(steps.len(), 1);
    assert_eq!(
        resolver.get_calls(),
        vec![("secure_tool_runtime".to_string(), json!({"path": "tmp"}))]
    );

    let runtime_options = resolver.get_options();
    assert_eq!(runtime_options.len(), 1);
    assert_eq!(runtime_options[0].tool_call_id, "call_secure_tool_runtime");
    assert_eq!(runtime_options[0].messages.len(), 3);
    assert!(matches!(
        runtime_options[0].messages.first(),
        Some(ModelMessage::User(_))
    ));
    assert!(matches!(
        runtime_options[0].messages.get(1),
        Some(ModelMessage::Assistant(_))
    ));
    assert!(matches!(
        runtime_options[0].messages.get(2),
        Some(ModelMessage::Tool(_))
    ));
    assert_eq!(
        runtime_options[0].context.get("requestId"),
        Some(&json!("req_approval_generate"))
    );
    assert!(runtime_options[0].abort_signal.is_none());
}

#[tokio::test]
async fn test_generate_preprocesses_denied_tool_approval_response_into_execution_denied() {
    let approval_id = "approval_dangerous";
    let messages = vec![
        ChatMessage::user("Deny local tool").build(),
        create_assistant_tool_approval_request_message(
            create_tool_call("dangerous_tool", json!({"mode": "rm"})),
            approval_id,
        ),
        create_tool_approval_response_message(
            approval_id,
            false,
            Some("User denied the request"),
            None,
        ),
    ];

    let model = MockChatModel::new(vec![create_text_response("Denied locally")]);
    let resolver = MockToolResolver::new().with_result("dangerous_tool", json!({"ok": true}));

    let (_response, steps) = generate(
        &model,
        messages,
        Some(vec![create_tool("dangerous_tool")]),
        Some(&resolver),
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(steps.len(), 1);
    assert_eq!(resolver.get_calls().len(), 0);

    let model_calls = model.get_calls();
    let tool_result = model_calls[0]
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .find_map(|message| match &message.content {
            MessageContent::MultiModal(parts) => parts.iter().find(|part| {
                matches!(
                    part,
                    ContentPart::ToolResult { tool_call_id, .. }
                        if tool_call_id == "call_dangerous_tool"
                )
            }),
            _ => None,
        })
        .expect("execution denied tool result");
    let ContentPart::ToolResult { input, output, .. } = tool_result else {
        panic!("expected tool result");
    };
    assert_eq!(input.as_ref(), Some(&json!({"mode": "rm"})));
    match output {
        ToolResultOutput::ExecutionDenied { reason, .. } => {
            assert_eq!(reason.as_deref(), Some("User denied the request"));
        }
        _ => panic!("expected execution denied output"),
    }
}

#[tokio::test]
async fn test_generate_forwards_provider_tool_approval_response_and_denied_result() {
    let approval_id = "mcp-approval-1";
    let messages = vec![
        ChatMessage::user("Deny provider tool").build(),
        create_assistant_tool_approval_request_message(
            create_provider_tool_call("mcp-call-1", "mcp_tool", json!({"query": "test"})),
            approval_id,
        ),
        create_tool_approval_response_message(
            approval_id,
            false,
            Some("User denied the request"),
            Some(true),
        ),
    ];

    let model = MockChatModel::new(vec![create_text_response("Provider denial continued")]);

    let (_response, steps) = generate(
        &model,
        messages,
        Some(vec![create_deferred_provider_tool("mcp_tool")]),
        None,
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(steps.len(), 1);

    let model_calls = model.get_calls();
    assert_eq!(model_calls.len(), 1);

    let approval_response_count = model_calls[0]
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .flat_map(|message| match &message.content {
            MessageContent::MultiModal(parts) => parts.iter().collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .filter(|part| {
            matches!(
                part,
                ContentPart::ToolApprovalResponse {
                    approval_id: current_approval_id,
                    provider_executed: Some(true),
                    ..
                } if current_approval_id == approval_id
            )
        })
        .count();
    assert_eq!(approval_response_count, 2);

    let tool_result = model_calls[0]
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .find_map(|message| match &message.content {
            MessageContent::MultiModal(parts) => parts.iter().find(|part| {
                matches!(
                    part,
                    ContentPart::ToolResult { tool_call_id, .. } if tool_call_id == "mcp-call-1"
                )
            }),
            _ => None,
        })
        .expect("provider execution denied tool result");
    let ContentPart::ToolResult { output, .. } = tool_result else {
        panic!("expected tool result");
    };
    match output {
        ToolResultOutput::ExecutionDenied {
            reason,
            provider_options,
        } => {
            assert_eq!(reason.as_deref(), Some("User denied the request"));
            assert_eq!(
                provider_options.get("openai"),
                Some(&json!({ "approvalId": approval_id }))
            );
        }
        _ => panic!("expected execution denied output"),
    }
}

#[tokio::test]
async fn test_generate_marks_runtime_dynamic_tool_calls_and_results() {
    use siumai::tooling::ExecutableTool;

    let responses = vec![
        create_response_with_tools(vec![create_tool_call(
            "dynamic_lookup",
            json!({"query": "rust"}),
        )]),
        create_text_response("dynamic done"),
    ];
    let model = MockChatModel::new(responses);
    let runtime_metadata = ExecutableTool::function(
        "dynamic_lookup",
        "Dynamic lookup",
        json!({"type":"object"}),
        |args| async move { Ok(args) },
    )
    .with_dynamic(true)
    .runtime_metadata()
    .clone();
    let resolver = MockToolResolver::new()
        .with_result("dynamic_lookup", json!({"ok": true}))
        .with_runtime_metadata("dynamic_lookup", runtime_metadata);

    let (_response, steps) = generate(
        &model,
        vec![ChatMessage::user("Run dynamic lookup").build()],
        Some(vec![create_tool("dynamic_lookup")]),
        Some(&resolver),
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    let expected_input = json!({"query": "rust"});
    assert_eq!(steps[0].static_tool_calls().len(), 0);
    assert_eq!(steps[0].dynamic_tool_calls().len(), 1);
    assert_eq!(steps[0].static_tool_results().len(), 0);
    assert_eq!(steps[0].dynamic_tool_results().len(), 1);
    assert_eq!(
        steps[0].dynamic_tool_results()[0].input,
        Some(&expected_input)
    );
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

    let finish_steps = Arc::new(Mutex::new(0usize));
    let finish_steps_clone = finish_steps.clone();
    let finish_total_prompt = Arc::new(Mutex::new(None));
    let finish_total_prompt_clone = finish_total_prompt.clone();
    let finish_function_id = Arc::new(Mutex::new(None::<String>));
    let finish_function_id_clone = finish_function_id.clone();

    let options = OrchestratorOptions {
        on_step_finish: Some(Arc::new(move |_step| {
            *step_finish_clone.lock().unwrap() += 1;
        })),
        on_finish: Some(Arc::new(move |event| {
            *finish_steps_clone.lock().unwrap() = event.steps.len();
            *finish_total_prompt_clone.lock().unwrap() = event
                .total_usage
                .as_ref()
                .and_then(|usage| usage.prompt_tokens());
            *finish_function_id_clone.lock().unwrap() = event.step.function_id.clone();
        })),
        telemetry: Some(
            TelemetryConfig::builder()
                .enabled(true)
                .function_id("orchestrator-test")
                .metadata("suite", "orchestrator")
                .build(),
        ),
        ..Default::default()
    };

    let messages = vec![ChatMessage::user("Test").build()];
    let tools = vec![create_tool("tool1")];

    let (_response, steps) = generate(&model, messages, Some(tools), Some(&resolver), &[], options)
        .await
        .unwrap();

    // Verify callbacks were called
    assert_eq!(*step_finish_called.lock().unwrap(), 2); // 2 steps
    assert_eq!(*finish_steps.lock().unwrap(), 2);
    assert_eq!(*finish_total_prompt.lock().unwrap(), Some(200));
    assert_eq!(
        *finish_function_id.lock().unwrap(),
        Some("orchestrator-test".to_string())
    );
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].request.messages.len(), 1);
    assert_eq!(steps[0].call_id, steps[1].call_id);
    assert_eq!(steps[0].function_id.as_deref(), Some("orchestrator-test"));
    assert_eq!(
        steps[0]
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("suite")),
        Some(&json!("orchestrator"))
    );
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

    let finish_text = Arc::new(Mutex::new(None::<String>));
    let finish_text_clone = finish_text.clone();

    let agent = ToolLoopAgent::new(model, vec![], vec![step_count_is(10)])
        .on_step_finish(Arc::new(move |_step| {
            *step_clone.lock().unwrap() = true;
        }))
        .on_finish(Arc::new(move |event| {
            *finish_text_clone.lock().unwrap() = event.text().map(ToString::to_string);
        }));

    let messages = vec![ChatMessage::user("Test").build()];
    let _ = agent.generate(messages, &resolver).await.unwrap();

    assert!(*step_called.lock().unwrap());
    assert_eq!(*finish_text.lock().unwrap(), Some("Done".to_string()));
}

#[tokio::test]
async fn test_generate_stream_basic_exposes_steps_and_total_usage() {
    let model = MockChatModel::new(vec![create_text_response("stream basic")]);
    let messages = vec![ChatMessage::user("Hello").build()];

    let StreamOrchestration {
        mut stream,
        steps,
        total_usage,
        ..
    } = generate_stream(
        &model,
        messages,
        None,
        None,
        OrchestratorStreamOptions::default(),
    )
    .await
    .unwrap();

    let mut collected = String::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            ChatStreamEvent::ContentDelta { delta, .. } => collected.push_str(&delta),
            ChatStreamEvent::StreamEnd { .. } => {}
            _ => {}
        }
    }

    let steps = steps.await.unwrap();
    let total_usage = total_usage.await.unwrap();

    assert_eq!(collected, "stream basic");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].text().as_deref(), Some("stream basic"));
    assert_eq!(
        total_usage.as_ref().and_then(|usage| usage.prompt_tokens()),
        Some(100)
    );
    assert_eq!(
        total_usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens()),
        Some(50)
    );
}

#[tokio::test]
async fn test_generate_stream_accepts_stable_text_parts_without_stream_end_response() {
    let model = MockStableTextStreamModel;
    let messages = vec![ChatMessage::user("Hello").build()];

    let StreamOrchestration {
        mut stream, steps, ..
    } = generate_stream(
        &model,
        messages,
        None,
        None,
        OrchestratorStreamOptions::default(),
    )
    .await
    .unwrap();

    let mut collected = String::new();
    while let Some(event) = stream.next().await {
        if let ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta { delta, .. },
        } = event.unwrap()
        {
            collected.push_str(&delta);
        }
    }

    let steps = steps.await.unwrap();

    assert_eq!(collected, "stream stable");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].text().as_deref(), Some("stream stable"));
}

#[tokio::test]
async fn test_generate_stream_step_preserves_raw_finish_reason() {
    let mut response = create_text_response("stream raw");
    response.raw_finish_reason = Some("end_turn".to_string());
    let model = MockChatModel::new(vec![response]);

    let StreamOrchestration {
        mut stream, steps, ..
    } = generate_stream(
        &model,
        vec![ChatMessage::user("Hello").build()],
        None,
        None,
        OrchestratorStreamOptions::default(),
    )
    .await
    .unwrap();

    while let Some(event) = stream.next().await {
        if let ChatStreamEvent::StreamEnd { .. } = event.unwrap() {
            break;
        }
    }

    let steps = steps.await.unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].raw_finish_reason.as_deref(), Some("end_turn"));
}

#[tokio::test]
async fn test_generate_stream_owned_on_finish_reports_total_usage() {
    let responses = vec![
        create_response_with_tools(vec![create_tool_call("tool1", json!({}))]),
        create_text_response("Stream done"),
    ];
    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("tool1", json!({"result": "ok"}));
    let tools = vec![create_tool("tool1")];

    let finish_event = Arc::new(Mutex::new(None::<OrchestratorFinishEvent>));
    let finish_event_clone = finish_event.clone();

    let options = OrchestratorStreamOptions {
        on_finish: Some(Arc::new(move |event| {
            *finish_event_clone.lock().unwrap() = Some(event.clone());
        })),
        ..Default::default()
    };

    let StreamOrchestration {
        mut stream,
        steps,
        total_usage,
        ..
    } = generate_stream_owned(
        model,
        vec![ChatMessage::user("Go").build()],
        Some(tools),
        Some(resolver),
        options,
    )
    .await
    .unwrap();

    while let Some(event) = stream.next().await {
        event.unwrap();
    }

    let steps = steps.await.unwrap();
    let total_usage = total_usage.await.unwrap();
    let finish_event = finish_event.lock().unwrap().clone().unwrap();

    assert_eq!(steps.len(), 2);
    assert_eq!(finish_event.steps.len(), 2);
    assert_eq!(finish_event.text(), Some("Stream done"));
    assert_eq!(
        total_usage.as_ref().and_then(|usage| usage.prompt_tokens()),
        Some(200)
    );
    assert_eq!(
        total_usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens()),
        Some(100)
    );
    assert_eq!(
        total_usage.as_ref().and_then(|usage| usage.total_tokens()),
        Some(300)
    );
    assert_eq!(
        finish_event
            .total_usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(300)
    );
    assert_eq!(
        finish_event
            .total_usage
            .as_ref()
            .and_then(|usage| usage.raw_usage_value()),
        None
    );
}

#[tokio::test]
async fn test_generate_stream_owned_respects_stop_conditions() {
    let responses = vec![
        create_response_with_tools(vec![create_tool_call("tool1", json!({}))]),
        create_text_response("Should not execute second step"),
    ];
    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("tool1", json!({"result": "ok"}));
    let tools = vec![create_tool("tool1")];

    let options = OrchestratorStreamOptions {
        stop_conditions: vec![Arc::<dyn StopCondition>::from(step_count_is(1))],
        ..Default::default()
    };

    let StreamOrchestration {
        mut stream, steps, ..
    } = generate_stream_owned(
        model.clone(),
        vec![ChatMessage::user("Go").build()],
        Some(tools),
        Some(resolver),
        options,
    )
    .await
    .unwrap();

    while let Some(event) = stream.next().await {
        event.unwrap();
    }

    let steps = steps.await.unwrap();
    let model_calls = model.get_calls();

    assert_eq!(steps.len(), 1);
    assert_eq!(model_calls.len(), 1);
    assert_eq!(steps[0].tool_calls.len(), 1);
    assert_eq!(steps[0].tool_results.len(), 1);
}

#[tokio::test]
async fn test_generate_stream_owned_continues_for_deferred_provider_tool_results() {
    let responses = vec![
        create_response_with_tools(vec![create_provider_tool_call(
            "call_code_execution",
            "code_execution",
            json!({"code":"print('hi')"}),
        )]),
        ChatResponse::new(MessageContent::MultiModal(vec![
            create_provider_tool_result(
                "call_code_execution",
                "code_execution",
                json!({"stdout":"hi"}),
            ),
            ContentPart::text("Deferred stream result received"),
        ])),
    ];
    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new();

    let StreamOrchestration {
        mut stream, steps, ..
    } = generate_stream_owned(
        model,
        vec![ChatMessage::user("Run code").build()],
        Some(vec![create_deferred_provider_tool("code_execution")]),
        Some(resolver),
        OrchestratorStreamOptions::default(),
    )
    .await
    .unwrap();

    while let Some(event) = stream.next().await {
        event.unwrap();
    }

    let steps = steps.await.unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].tool_calls.len(), 1);
    assert_eq!(steps[0].tool_results.len(), 0);
    assert_eq!(steps[1].tool_calls.len(), 0);
    assert_eq!(steps[1].tool_results.len(), 1);
    assert_eq!(steps[1].tool_result_views()[0].input, None);
    assert_eq!(
        steps[1].text().as_deref(),
        Some("Deferred stream result received")
    );
}

#[tokio::test]
async fn test_generate_stream_owned_emits_runtime_tool_approval_request_and_input_callbacks() {
    use siumai::tooling::ExecutableTool;

    let responses = vec![create_response_with_tools(vec![create_tool_call(
        "dangerous_stream",
        json!({"mode": "rm"}),
    )])];
    let model = MockChatModel::new(responses);

    let input_start_calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let input_start_calls_clone = input_start_calls.clone();
    let input_start_messages = Arc::new(Mutex::new(Vec::<Vec<ModelMessage>>::new()));
    let input_start_messages_clone = input_start_messages.clone();
    let input_start_contexts = Arc::new(Mutex::new(Vec::<Context>::new()));
    let input_start_contexts_clone = input_start_contexts.clone();
    let input_start_abort_flags = Arc::new(Mutex::new(Vec::<bool>::new()));
    let input_start_abort_flags_clone = input_start_abort_flags.clone();
    let input_deltas = Arc::new(Mutex::new(Vec::<String>::new()));
    let input_deltas_clone = input_deltas.clone();
    let input_delta_abort_flags = Arc::new(Mutex::new(Vec::<bool>::new()));
    let input_delta_abort_flags_clone = input_delta_abort_flags.clone();
    let available_inputs = Arc::new(Mutex::new(Vec::<Value>::new()));
    let available_inputs_clone = available_inputs.clone();
    let available_contexts = Arc::new(Mutex::new(Vec::<Context>::new()));
    let available_contexts_clone = available_contexts.clone();
    let available_abort_flags = Arc::new(Mutex::new(Vec::<bool>::new()));
    let available_abort_flags_clone = available_abort_flags.clone();

    let runtime_metadata = ExecutableTool::function(
        "dangerous_stream",
        "Dangerous stream",
        json!({"type":"object"}),
        |args| async move { Ok(args) },
    )
    .with_needs_approval(true)
    .with_on_input_start_fn(move |context| {
        let input_start_calls = input_start_calls_clone.clone();
        let input_start_messages = input_start_messages_clone.clone();
        let input_start_contexts = input_start_contexts_clone.clone();
        let input_start_abort_flags = input_start_abort_flags_clone.clone();
        async move {
            input_start_calls.lock().unwrap().push(context.tool_call_id);
            input_start_messages.lock().unwrap().push(context.messages);
            input_start_contexts.lock().unwrap().push(context.context);
            input_start_abort_flags
                .lock()
                .unwrap()
                .push(context.abort_signal.is_some());
            Ok(())
        }
    })
    .with_on_input_delta_fn(move |context| {
        let input_deltas = input_deltas_clone.clone();
        let input_delta_abort_flags = input_delta_abort_flags_clone.clone();
        async move {
            input_deltas.lock().unwrap().push(context.input_text_delta);
            input_delta_abort_flags
                .lock()
                .unwrap()
                .push(context.abort_signal.is_some());
            Ok(())
        }
    })
    .with_on_input_available_fn(move |context| {
        let available_inputs = available_inputs_clone.clone();
        let available_contexts = available_contexts_clone.clone();
        let available_abort_flags = available_abort_flags_clone.clone();
        async move {
            available_inputs.lock().unwrap().push(context.input);
            available_contexts.lock().unwrap().push(context.context);
            available_abort_flags
                .lock()
                .unwrap()
                .push(context.abort_signal.is_some());
            Ok(())
        }
    })
    .runtime_metadata()
    .clone();

    let resolver = MockToolResolver::new()
        .with_result("dangerous_stream", json!({"ok": true}))
        .with_runtime_metadata("dangerous_stream", runtime_metadata);

    let mut options = OrchestratorStreamOptions::default();
    options.context.insert("requestId", json!("req_stream"));

    let StreamOrchestration {
        mut stream, steps, ..
    } = generate_stream_owned(
        model,
        vec![ChatMessage::user("Stream dangerous tool").build()],
        Some(vec![create_tool("dangerous_stream")]),
        Some(resolver),
        options,
    )
    .await
    .unwrap();

    let mut saw_approval_request = false;
    while let Some(event) = stream.next().await {
        if let ChatStreamEvent::Part {
            part: ChatStreamPart::ToolApprovalRequest(request),
        } = event.unwrap()
        {
            saw_approval_request = true;
            assert_eq!(request.tool_call_id, "call_dangerous_stream");
        }
    }

    let steps = steps.await.unwrap();
    assert!(saw_approval_request);
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].tool_results.len(), 0);
    assert_eq!(
        input_start_calls.lock().unwrap().as_slice(),
        &["call_dangerous_stream".to_string()]
    );
    assert_eq!(input_start_messages.lock().unwrap().len(), 1);
    assert!(matches!(
        input_start_messages.lock().unwrap()[0].first(),
        Some(ModelMessage::User(_))
    ));
    assert_eq!(
        input_start_contexts.lock().unwrap()[0].get("requestId"),
        Some(&json!("req_stream"))
    );
    assert_eq!(input_start_abort_flags.lock().unwrap().as_slice(), &[true]);
    assert_eq!(
        input_deltas.lock().unwrap().as_slice(),
        &["{\"mode\":\"rm\"}".to_string()]
    );
    assert_eq!(input_delta_abort_flags.lock().unwrap().as_slice(), &[true]);
    assert_eq!(
        available_inputs.lock().unwrap().as_slice(),
        &[json!({"mode": "rm"})]
    );
    assert_eq!(
        available_contexts.lock().unwrap()[0].get("requestId"),
        Some(&json!("req_stream"))
    );
    assert_eq!(available_abort_flags.lock().unwrap().as_slice(), &[true]);
}

#[tokio::test]
async fn test_generate_stream_owned_preprocesses_local_tool_approval_response() {
    let approval_id = "approval_stream_local";
    let messages = vec![
        ChatMessage::user("Approve local stream tool").build(),
        create_assistant_tool_approval_request_message(
            create_tool_call("stream_secure", json!({"path": "tmp"})),
            approval_id,
        ),
        create_tool_approval_response_message(approval_id, true, None, None),
    ];

    let model = MockChatModel::new(vec![create_text_response(
        "Stream local approval continued",
    )]);
    let resolver = MockToolResolver::new().with_result("stream_secure", json!({"ok": true}));

    let StreamOrchestration {
        mut stream, steps, ..
    } = generate_stream_owned(
        model.clone(),
        messages,
        Some(vec![create_tool("stream_secure")]),
        Some(resolver),
        OrchestratorStreamOptions::default(),
    )
    .await
    .unwrap();

    while let Some(event) = stream.next().await {
        event.unwrap();
    }

    let steps = steps.await.unwrap();
    assert_eq!(steps.len(), 1);

    let model_calls = model.get_calls();
    assert_eq!(model_calls.len(), 1);
    let tool_result = model_calls[0]
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .find_map(|message| match &message.content {
            MessageContent::MultiModal(parts) => parts.iter().find(|part| {
                matches!(
                    part,
                    ContentPart::ToolResult { tool_call_id, .. }
                        if tool_call_id == "call_stream_secure"
                )
            }),
            _ => None,
        })
        .expect("forwarded stream tool result");
    let ContentPart::ToolResult { input, output, .. } = tool_result else {
        panic!("expected tool result");
    };
    assert_eq!(input.as_ref(), Some(&json!({"path": "tmp"})));
    match output {
        ToolResultOutput::Json { value, .. } => assert_eq!(value, &json!({"ok": true})),
        _ => panic!("expected json output"),
    }
}

#[tokio::test]
async fn test_generate_stream_owned_preprocesses_local_tool_approval_response_preserves_runtime_options()
 {
    let approval_id = "approval_stream_runtime";
    let messages = vec![
        ChatMessage::user("Approve local stream tool with runtime context").build(),
        create_assistant_tool_approval_request_message(
            create_tool_call("stream_secure_runtime", json!({"path": "tmp"})),
            approval_id,
        ),
        create_tool_approval_response_message(approval_id, true, None, None),
    ];

    let model = MockChatModel::new(vec![create_text_response(
        "Stream local approval continued",
    )]);
    let resolver = RuntimeOptionsAwareToolResolver::new();

    let mut options = OrchestratorStreamOptions::default();
    options
        .context
        .insert("requestId", json!("req_approval_stream"));

    let StreamOrchestration {
        mut stream, steps, ..
    } = generate_stream_owned(
        model,
        messages,
        Some(vec![create_tool("stream_secure_runtime")]),
        Some(resolver.clone()),
        options,
    )
    .await
    .unwrap();

    while let Some(event) = stream.next().await {
        event.unwrap();
    }

    let steps = steps.await.unwrap();
    assert_eq!(steps.len(), 1);

    assert_eq!(
        resolver.get_calls(),
        vec![("stream_secure_runtime".to_string(), json!({"path": "tmp"}))]
    );

    let runtime_options = resolver.get_options();
    assert_eq!(runtime_options.len(), 1);
    assert_eq!(
        runtime_options[0].tool_call_id,
        "call_stream_secure_runtime"
    );
    assert_eq!(runtime_options[0].messages.len(), 3);
    assert!(matches!(
        runtime_options[0].messages.first(),
        Some(ModelMessage::User(_))
    ));
    assert!(matches!(
        runtime_options[0].messages.get(1),
        Some(ModelMessage::Assistant(_))
    ));
    assert!(matches!(
        runtime_options[0].messages.get(2),
        Some(ModelMessage::Tool(_))
    ));
    assert_eq!(
        runtime_options[0].context.get("requestId"),
        Some(&json!("req_approval_stream"))
    );
    assert!(runtime_options[0].abort_signal.is_some());
}

#[tokio::test]
async fn test_generate_stream_owned_forwards_provider_tool_approval_response_and_denied_result() {
    let approval_id = "approval_stream_provider";
    let messages = vec![
        ChatMessage::user("Deny provider stream tool").build(),
        create_assistant_tool_approval_request_message(
            create_provider_tool_call(
                "mcp-stream-call-1",
                "mcp_stream_tool",
                json!({"query": "test"}),
            ),
            approval_id,
        ),
        create_tool_approval_response_message(
            approval_id,
            false,
            Some("User denied the request"),
            Some(true),
        ),
    ];

    let model = MockChatModel::new(vec![create_text_response(
        "Stream provider denial continued",
    )]);

    let StreamOrchestration {
        mut stream, steps, ..
    } = generate_stream_owned(
        model.clone(),
        messages,
        Some(vec![create_deferred_provider_tool("mcp_stream_tool")]),
        None::<MockToolResolver>,
        OrchestratorStreamOptions::default(),
    )
    .await
    .unwrap();

    while let Some(event) = stream.next().await {
        event.unwrap();
    }

    let steps = steps.await.unwrap();
    assert_eq!(steps.len(), 1);

    let model_calls = model.get_calls();
    assert_eq!(model_calls.len(), 1);
    let approval_response_count = model_calls[0]
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .flat_map(|message| match &message.content {
            MessageContent::MultiModal(parts) => parts.iter().collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .filter(|part| {
            matches!(
                part,
                ContentPart::ToolApprovalResponse {
                    approval_id: current_approval_id,
                    provider_executed: Some(true),
                    ..
                } if current_approval_id == approval_id
            )
        })
        .count();
    assert_eq!(approval_response_count, 2);

    let tool_result = model_calls[0]
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .find_map(|message| match &message.content {
            MessageContent::MultiModal(parts) => parts.iter().find(|part| {
                matches!(
                    part,
                    ContentPart::ToolResult { tool_call_id, .. }
                        if tool_call_id == "mcp-stream-call-1"
                )
            }),
            _ => None,
        })
        .expect("provider stream execution denied tool result");
    let ContentPart::ToolResult { output, .. } = tool_result else {
        panic!("expected tool result");
    };
    match output {
        ToolResultOutput::ExecutionDenied {
            reason,
            provider_options,
        } => {
            assert_eq!(reason.as_deref(), Some("User denied the request"));
            assert_eq!(
                provider_options.get("openai"),
                Some(&json!({ "approvalId": approval_id }))
            );
        }
        _ => panic!("expected execution denied output"),
    }
}

// ============================================================================
// Tests for stop conditions (additional to existing tests)
// ============================================================================

#[test]
fn test_all_of_stop_condition() {
    let condition = all_of(vec![step_count_is(2), has_text_response()]);

    // Should not stop if only one condition is met
    let steps = vec![StepResult {
        call_id: test_call_id(),
        step_number: 0,
        model: test_step_model(),
        request: create_empty_step_request(),
        response: create_step_text_response("text"),
        raw_finish_reason: None,
        function_id: None,
        metadata: None,
        context: test_context(),
        content: vec![ContentPart::text("text")],
        messages: vec![ChatMessage::assistant("text").build()],
        finish_reason: None,
        usage: None,
        tool_calls: vec![],
        tool_results: vec![],
        warnings: None,
        provider_metadata: None,
    }];
    assert!(!condition.should_stop(&steps)); // has_text_response but not step_count

    // Should stop when both conditions are met
    let steps = vec![
        StepResult {
            call_id: test_call_id(),
            step_number: 0,
            model: test_step_model(),
            request: create_empty_step_request(),
            response: create_empty_step_response(),
            raw_finish_reason: None,
            function_id: None,
            metadata: None,
            context: test_context(),
            content: vec![],
            messages: vec![],
            finish_reason: None,
            usage: None,
            tool_calls: vec![ContentPart::tool_call("call_1", "tool1", json!({}), None)],
            tool_results: vec![],
            warnings: None,
            provider_metadata: None,
        },
        StepResult {
            call_id: test_call_id(),
            step_number: 1,
            model: test_step_model(),
            request: create_empty_step_request(),
            response: create_step_text_response("text"),
            raw_finish_reason: None,
            function_id: None,
            metadata: None,
            context: test_context(),
            content: vec![ContentPart::text("text")],
            messages: vec![ChatMessage::assistant("text").build()],
            finish_reason: None,
            usage: None,
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
            provider_metadata: None,
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

#[tokio::test]
async fn test_generate_propagates_context_through_prepare_step_tools_and_finish() {
    use super::prepare_step::{PrepareStepContext, PrepareStepResult, ToolChoice};

    let responses = vec![
        create_response_with_tools(vec![create_tool_call("search", json!({"query": "rust"}))]),
        create_text_response("Search completed"),
    ];
    let model = MockChatModel::new(responses);
    let resolver = ContextAwareToolResolver::new();

    let prepare_contexts = Arc::new(Mutex::new(Vec::<OrchestratorContext>::new()));
    let prepare_contexts_clone = prepare_contexts.clone();
    let finish_context = Arc::new(Mutex::new(None::<OrchestratorContext>));
    let finish_context_clone = finish_context.clone();

    let mut initial_context = OrchestratorContext::default();
    initial_context.insert("phase", json!("initial"));

    let prepare_fn: super::prepare_step::PrepareStepFn =
        Arc::new(move |ctx: PrepareStepContext| {
            prepare_contexts_clone
                .lock()
                .unwrap()
                .push(ctx.context.clone());

            if ctx.step_number == 0 {
                let mut next_context = ctx.context.clone();
                next_context.insert("phase", json!("prepared"));
                next_context.insert("step_marker", json!(ctx.step_number));

                PrepareStepResult::new()
                    .with_tool_choice(ToolChoice::Required)
                    .with_context(next_context)
            } else {
                PrepareStepResult::default()
            }
        });

    let options = OrchestratorOptions {
        context: initial_context,
        prepare_step: Some(prepare_fn),
        on_finish: Some(Arc::new(move |event| {
            *finish_context_clone.lock().unwrap() = Some(event.context.clone());
        })),
        ..Default::default()
    };

    let (_response, steps) = generate(
        &model,
        vec![ChatMessage::user("Search for Rust").build()],
        Some(vec![create_tool("search")]),
        Some(&resolver),
        &[],
        options,
    )
    .await
    .unwrap();

    let prepare_contexts = prepare_contexts.lock().unwrap().clone();
    let resolver_contexts = resolver.get_contexts();
    let finish_context = finish_context.lock().unwrap().clone().unwrap();

    assert_eq!(prepare_contexts.len(), 2);
    assert_eq!(prepare_contexts[0].get("phase"), Some(&json!("initial")));
    assert_eq!(prepare_contexts[1].get("phase"), Some(&json!("prepared")));

    assert_eq!(resolver_contexts.len(), 1);
    assert_eq!(resolver_contexts[0].get("phase"), Some(&json!("prepared")));
    assert_eq!(resolver_contexts[0].get("step_marker"), Some(&json!(0)));

    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].context.get("phase"), Some(&json!("prepared")));
    assert_eq!(steps[1].context.get("phase"), Some(&json!("prepared")));
    assert_eq!(finish_context.get("phase"), Some(&json!("prepared")));
}

#[tokio::test]
async fn test_generate_context_support_is_backward_compatible_with_legacy_resolver() {
    let responses = vec![
        create_response_with_tools(vec![create_tool_call("search", json!({"query": "compat"}))]),
        create_text_response("Legacy resolver completed"),
    ];
    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("search", json!({"ok": true}));

    let mut context = OrchestratorContext::default();
    context.insert("phase", json!("legacy"));

    let (_response, steps) = generate(
        &model,
        vec![ChatMessage::user("Compatibility").build()],
        Some(vec![create_tool("search")]),
        Some(&resolver),
        &[],
        OrchestratorOptions {
            context,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(resolver.get_calls().len(), 1);
    assert_eq!(steps[0].context.get("phase"), Some(&json!("legacy")));
    assert_eq!(steps[1].context.get("phase"), Some(&json!("legacy")));
}

#[tokio::test]
async fn test_generate_stream_owned_applies_prepare_step_context_and_tool_filters() {
    use super::prepare_step::{
        PrepareStepContext, PrepareStepResult, ToolChoice as PrepareToolChoice,
    };

    let responses = vec![
        create_response_with_tools(vec![create_tool_call("search", json!({"query": "stream"}))]),
        create_text_response("Stream search completed"),
    ];
    let model = MockChatModel::new(responses);
    let resolver = ContextAwareToolResolver::new();
    let resolver_handle = resolver.clone();

    let prepare_contexts = Arc::new(Mutex::new(Vec::<OrchestratorContext>::new()));
    let prepare_contexts_clone = prepare_contexts.clone();
    let finish_context = Arc::new(Mutex::new(None::<OrchestratorContext>));
    let finish_context_clone = finish_context.clone();

    let mut initial_context = OrchestratorContext::default();
    initial_context.insert("phase", json!("stream-initial"));

    let options = OrchestratorStreamOptions {
        context: initial_context,
        prepare_step: Some(Arc::new(move |ctx: PrepareStepContext| {
            prepare_contexts_clone
                .lock()
                .unwrap()
                .push(ctx.context.clone());

            if ctx.step_number == 0 {
                let mut next_context = ctx.context.clone();
                next_context.insert("phase", json!("stream-prepared"));
                next_context.insert("step_marker", json!(ctx.step_number));

                PrepareStepResult::new()
                    .with_tool_choice(PrepareToolChoice::Required)
                    .with_active_tools(vec!["search".to_string()])
                    .with_context(next_context)
            } else {
                PrepareStepResult::default()
            }
        })),
        on_finish: Some(Arc::new(move |event| {
            *finish_context_clone.lock().unwrap() = Some(event.context.clone());
        })),
        ..Default::default()
    };

    let StreamOrchestration {
        mut stream,
        steps,
        total_usage,
        ..
    } = generate_stream_owned(
        model,
        vec![ChatMessage::user("Search in stream").build()],
        Some(vec![create_tool("search"), create_tool("extra")]),
        Some(resolver),
        options,
    )
    .await
    .unwrap();

    while let Some(event) = stream.next().await {
        event.unwrap();
    }

    let steps = steps.await.unwrap();
    let _total_usage = total_usage.await.unwrap();

    let prepare_contexts = prepare_contexts.lock().unwrap().clone();
    let finish_context = finish_context.lock().unwrap().clone().unwrap();
    let resolver_contexts = resolver_handle.get_contexts();

    assert_eq!(prepare_contexts.len(), 2);
    assert_eq!(
        prepare_contexts[0].get("phase"),
        Some(&json!("stream-initial"))
    );
    assert_eq!(
        prepare_contexts[1].get("phase"),
        Some(&json!("stream-prepared"))
    );

    assert_eq!(steps.len(), 2);
    assert_eq!(resolver_contexts.len(), 1);
    assert_eq!(
        resolver_contexts[0].get("phase"),
        Some(&json!("stream-prepared"))
    );
    assert!(matches!(
        steps[0].request.tool_choice,
        Some(siumai::prelude::unified::ToolChoice::Required)
    ));
    assert_eq!(steps[0].request.tools.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        steps[0]
            .request
            .tools
            .as_ref()
            .and_then(|tools| tools.first())
            .map(|tool| match tool {
                Tool::Function { function } => function.name.as_str(),
                Tool::ProviderDefined(provider_tool) => provider_tool.name.as_str(),
            }),
        Some("search")
    );
    assert_eq!(
        steps[0].context.get("phase"),
        Some(&json!("stream-prepared"))
    );
    assert_eq!(
        steps[1].context.get("phase"),
        Some(&json!("stream-prepared"))
    );
    assert_eq!(finish_context.get("phase"), Some(&json!("stream-prepared")));
}

#[tokio::test]
async fn test_generate_stream_owned_prepare_step_can_swap_model() {
    let base_model = MockChatModel::new(vec![create_text_response("base-stream")]);
    let base_model_handle = base_model.clone();
    let override_model = MockChatModel::new(vec![create_text_response("override-stream")])
        .with_identity("stream-provider", "stream-model");
    let override_model_handle = override_model.clone();

    let override_model: StepLanguageModel = Arc::new(override_model);
    let options = OrchestratorStreamOptions {
        prepare_step: Some(Arc::new(move |ctx| {
            assert_eq!(ctx.model.provider_id(), "mock-provider");
            assert_eq!(ctx.model.model_id(), "mock-model");

            if ctx.step_number == 0 {
                PrepareStepResult::new().with_model(override_model.clone())
            } else {
                PrepareStepResult::default()
            }
        })),
        ..Default::default()
    };

    let StreamOrchestration {
        mut stream, steps, ..
    } = generate_stream_owned(
        base_model,
        vec![ChatMessage::user("swap stream").build()],
        None,
        None::<MockToolResolver>,
        options,
    )
    .await
    .unwrap();

    while let Some(event) = stream.next().await {
        event.unwrap();
    }

    let steps = steps.await.unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].text().as_deref(), Some("override-stream"));
    assert_eq!(steps[0].model.provider, "stream-provider");
    assert_eq!(steps[0].model.model_id, "stream-model");
    assert_eq!(base_model_handle.get_calls().len(), 0);
    assert_eq!(override_model_handle.get_calls().len(), 1);
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
// Tool-loop Contract Suite (no network)
// ============================================================================

/// Contract: tool results must be appended into the next model call history,
/// preserving the tool_call_id and tool_name, so the model can continue the loop.
#[tokio::test]
async fn contract_tool_loop_appends_tool_result_to_next_turn_history() {
    let tool_call_id = "call_get_weather";
    let tool_name = "get_weather";

    let tool_call = ContentPart::tool_call(
        tool_call_id.to_string(),
        tool_name.to_string(),
        json!({"city": "Tokyo"}),
        None,
    );

    let responses = vec![
        create_response_with_tools(vec![tool_call]),
        create_text_response("Tokyo is sunny."),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new()
        .with_result(tool_name, json!({"temperature": 25, "condition": "sunny"}));

    let messages = vec![ChatMessage::user("What's the weather in Tokyo?").build()];
    let tools = vec![Tool::function(
        tool_name.to_string(),
        "Get weather for a city".to_string(),
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    )];

    let (_response, _steps) = generate(
        &model,
        messages,
        Some(tools),
        Some(&resolver),
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    let calls = model.get_calls();
    assert_eq!(calls.len(), 2);

    let second_call = &calls[1];
    let tool_msg = second_call
        .iter()
        .find(|m| matches!(m.role, MessageRole::Tool))
        .expect("second call should include a tool message");

    let results = tool_msg.tool_results();
    assert_eq!(results.len(), 1);
    let info = results[0]
        .as_tool_result()
        .expect("expected tool result part");
    assert_eq!(info.tool_call_id, tool_call_id);
    assert_eq!(info.tool_name, tool_name);

    let out = serde_json::to_value(info.output).expect("tool output should be serializable");
    assert_eq!(
        out,
        json!({
            "type": "json",
            "value": { "temperature": 25, "condition": "sunny" }
        })
    );
}

/// Contract: invalid tool arguments must not call the resolver and must emit a tool error message.
#[tokio::test]
async fn contract_invalid_tool_args_skips_resolver_and_emits_tool_error() {
    let tool_call = create_tool_call("get_weather", json!({})); // missing required "city"
    let responses = vec![
        create_response_with_tools(vec![tool_call]),
        create_text_response("I could not call the tool."),
    ];

    let model = MockChatModel::new(responses);
    let resolver = MockToolResolver::new().with_result("get_weather", json!({"ok": true}));

    let messages = vec![ChatMessage::user("Weather?").build()];
    let tools = vec![Tool::function(
        "get_weather".to_string(),
        "Get weather for a city".to_string(),
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    )];

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

    // NOTE: JSON Schema validation is optional in siumai-extras.
    // - With `--features schema`, invalid args should be blocked before resolver execution.
    // - Without it, validation is a no-op and the resolver will still be called.
    #[cfg(feature = "schema")]
    assert_eq!(resolver.get_calls().len(), 0);
    #[cfg(not(feature = "schema"))]
    assert_eq!(resolver.get_calls().len(), 1);

    // The first step should contain a tool error message (role=Tool).
    let step0 = &steps[0];
    let tool_msg = step0
        .messages
        .iter()
        .find(|m| matches!(m.role, MessageRole::Tool))
        .expect("expected a tool error message");

    let results = tool_msg.tool_results();
    assert_eq!(results.len(), 1);
    let info = results[0]
        .as_tool_result()
        .expect("expected tool result part");
    assert_eq!(info.tool_name, "get_weather");

    let out = serde_json::to_value(info.output).expect("tool output should be serializable");
    #[cfg(feature = "schema")]
    {
        assert_eq!(out["type"], json!("error-json"));
        assert_eq!(out["value"]["error"], json!("invalid_args"));
    }

    #[cfg(not(feature = "schema"))]
    {
        assert_eq!(out["type"], json!("json"));
    }
}

/// Contract: provider metadata from a step response must be exposed through StepResult.
#[tokio::test]
async fn contract_provider_metadata_is_exposed_in_step_result() {
    let mut resp = create_text_response("ok");
    resp.provider_metadata = Some(std::collections::HashMap::from([(
        "openai".to_string(),
        serde_json::json!({ "foo": "bar" }),
    )]));

    let model = MockChatModel::new(vec![resp]);
    let messages = vec![ChatMessage::user("hi").build()];

    let (_response, steps) = generate(
        &model,
        messages,
        None,
        None,
        &[],
        OrchestratorOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(steps.len(), 1);
    let md = steps[0]
        .provider_metadata
        .as_ref()
        .expect("expected provider metadata");
    assert_eq!(md["openai"]["foo"], json!("bar"));
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
            call_id: test_call_id(),
            step_number: 0,
            model: test_step_model(),
            request: create_empty_step_request(),
            response: create_empty_step_response(),
            raw_finish_reason: None,
            function_id: None,
            metadata: None,
            context: test_context(),
            content: vec![],
            messages: vec![],
            finish_reason: None,
            usage: Some(
                Usage::builder()
                    .prompt_tokens(100)
                    .completion_tokens(50)
                    .total_tokens(150)
                    .with_raw_usage_value(serde_json::json!({
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "provider": "single-step"
                    }))
                    .build(),
            ),
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
            provider_metadata: None,
        },
        StepResult {
            call_id: test_call_id(),
            step_number: 1,
            model: test_step_model(),
            request: create_empty_step_request(),
            response: create_empty_step_response(),
            raw_finish_reason: None,
            function_id: None,
            metadata: None,
            context: test_context(),
            content: vec![],
            messages: vec![],
            finish_reason: None,
            usage: None, // No usage for this step
            tool_calls: vec![],
            tool_results: vec![],
            warnings: None,
            provider_metadata: None,
        },
    ];

    let merged = StepResult::merge_usage(&steps).unwrap();
    assert_eq!(merged.prompt_tokens(), Some(100));
    assert_eq!(merged.completion_tokens(), Some(50));
    // Simple addition: only first step has usage, so total_tokens = 150
    assert_eq!(merged.total_tokens(), Some(150));
    assert_eq!(merged.raw_usage_value(), None);
}
