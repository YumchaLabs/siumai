# Orchestrator Module

The orchestrator module provides advanced multi-step tool calling capabilities for building autonomous AI agents with flexible control flow.

## 🎯 Overview

The orchestrator enables AI models to:
- Execute multiple tool calls across several steps
- Make decisions based on tool execution results
- Iterate until a goal is achieved or a stop condition is met
- Track progress with callbacks and telemetry
- Control execution with approval workflows and dynamic configuration

## 🏗️ Architecture

The orchestrator is organized into focused modules:

```
orchestrator/
├── mod.rs              # Public API and module organization
├── types.rs            # Core types (StepResult, ToolApproval, etc.)
├── stop_condition.rs   # Flexible stop condition system
├── prepare_step.rs     # Dynamic step preparation
├── generate.rs         # Non-streaming orchestrator
├── stream.rs           # Streaming orchestrator
└── agent.rs            # Reusable agent abstraction
```

### Key Components

- **`generate()`**: Non-streaming multi-step execution
- **`generate_stream_owned()`**: Streaming multi-step execution with real-time progress
- **`ToolLoopAgent<M>`**: Reusable agent abstraction with builder pattern
- **`StopCondition`**: Flexible trait for controlling when to stop
- **`PrepareStep`**: Callback for dynamic step configuration
- **`ToolResolver`**: Trait for executing tools by name

## 🚀 Quick Start

### Basic Usage

```rust
use siumai_extras::orchestrator::{generate, step_count_is, OrchestratorOptions, ToolResolver};
use siumai::prelude::*;

// 1. Implement ToolResolver
struct MyResolver;

#[async_trait::async_trait]
impl ToolResolver for MyResolver {
    async fn call_tool(&self, name: &str, args: serde_json::Value) 
        -> Result<serde_json::Value, LlmError> {
        match name {
            "get_weather" => {
                let city = args["city"].as_str().unwrap();
                Ok(json!({"temperature": 72, "condition": "sunny"}))
            }
            _ => Err(LlmError::ToolExecutionError(format!("Unknown tool: {}", name)))
        }
    }
}

// 2. Set up orchestration
let client = Siumai::builder()
    .openai()
    .api_key(&api_key)
    .model("gpt-4")
    .build()?;

let tools = vec![
    Tool::function("get_weather", "Get weather for a city")
        .parameter("city", "string", "City name", true)
        .build(),
];

let messages = vec![user!("What's the weather in Tokyo?")];
let resolver = MyResolver;

// 3. Run orchestration
let stop_condition = step_count_is(10);
let (response, steps) = generate(
    &client,
    messages,
    Some(tools),
    Some(&resolver),
    &[&*stop_condition],
    OrchestratorOptions::default(),
).await?;

println!("Final response: {}", response.content_text().unwrap());
println!("Total steps: {}", steps.len());
```

### Using the Agent Pattern

```rust
use siumai_extras::orchestrator::{ToolLoopAgent, step_count_is};

// Create a reusable agent
let agent = ToolLoopAgent::new(client, tools, vec![step_count_is(10)])
    .with_system("You are a helpful weather assistant")
    .with_id("weather-agent")
    .on_step_finish(Arc::new(|step| {
        println!("Step {}: {} tool calls", step.step_number, step.tool_calls.len());
    }));

// Use it multiple times
let (response1, _) = agent.generate(messages1, &resolver).await?;
let (response2, _) = agent.generate(messages2, &resolver).await?;
```

## 📚 Core Concepts

### 1. Stop Conditions

Control when orchestration should stop using the `StopCondition` trait:

```rust
use siumai_extras::orchestrator::*;

// Built-in conditions
let c1 = step_count_is(10);              // Stop after N steps
let c2 = has_tool_call("finalAnswer");   // Stop when specific tool is called
let c3 = has_text_response();            // Stop when model generates text

// Combinators
let c4 = any_of(vec![                    // Stop if ANY condition is true
    step_count_is(20),
    has_tool_call("finalAnswer"),
]);

let c5 = all_of(vec![                    // Stop if ALL conditions are true
    has_text_response(),
    custom_condition(|steps| steps.len() > 3),
]);

// Custom conditions
let c6 = custom_condition(|steps: &[StepResult]| {
    steps.iter().any(|s| s.tool_calls.len() > 5)
});
```

### 2. Tool Approval

Control tool execution with approval workflows:

```rust
use siumai_extras::orchestrator::{OrchestratorOptions, ToolApproval};

let options = OrchestratorOptions {
    on_tool_approval: Some(Arc::new(|tool_name: &str, args: &serde_json::Value| {
        match tool_name {
            // Deny dangerous operations
            "delete_file" => ToolApproval::Deny {
                reason: "File deletion not allowed".to_string(),
            },
            
            // Modify arguments for safety
            "write_file" => {
                let mut modified = args.clone();
                if args["path"].as_str().unwrap().starts_with("/etc/") {
                    modified["path"] = json!("/tmp/safe_file.txt");
                    ToolApproval::Modify(modified)
                } else {
                    ToolApproval::Approve(args.clone())
                }
            },
            
            // Approve as-is
            _ => ToolApproval::Approve(args.clone()),
        }
    })),
    ..Default::default()
};
```

### 3. Dynamic Step Preparation

Modify orchestrator behavior before each step:

```rust
use siumai_extras::orchestrator::{PrepareStepContext, PrepareStepResult, ToolChoice};

let options = OrchestratorOptions {
    prepare_step: Some(Arc::new(|ctx: PrepareStepContext| {
        PrepareStepResult {
            // Control tool usage
            tool_choice: if ctx.step_number == 0 {
                Some(ToolChoice::Required)  // Force tool use on first step
            } else if ctx.step_number > 5 {
                Some(ToolChoice::Specific {
                    tool_name: "finalAnswer".to_string(),
                })
            } else {
                Some(ToolChoice::Auto)
            },
            
            // Filter available tools
            active_tools: if ctx.step_number > 5 {
                Some(vec!["finalAnswer".to_string()])
            } else {
                None  // All tools available
            },
            
            // Modify system message
            system: Some(format!(
                "Step {}/10. You must complete the task.",
                ctx.step_number + 1
            )),
            
            // Modify message history (optional)
            messages: None,
        }
    })),
    ..Default::default()
};
```

### 4. Callbacks

Track progress with callbacks:

```rust
let options = OrchestratorOptions {
    // Called after each step
    on_step_finish: Some(Arc::new(|step: &StepResult| {
        println!("Step completed:");
        println!("  Tool calls: {}", step.tool_calls.len());
        println!("  Finish reason: {:?}", step.finish_reason);
        if let Some(usage) = &step.usage {
            println!("  Tokens: {}", usage.total_tokens);
        }
    })),
    
    // Called when orchestration completes
    on_finish: Some(Arc::new(|steps: &[StepResult]| {
        let total_usage = StepResult::merge_usage(steps);
        println!("Orchestration complete:");
        println!("  Total steps: {}", steps.len());
        println!("  Total tokens: {}", total_usage.total_tokens);
    })),
    
    ..Default::default()
};
```

### 5. Streaming

Real-time multi-step execution:

```rust
use siumai_extras::orchestrator::{generate_stream_owned, OrchestratorStreamOptions};
use siumai::stream::StreamEvent;

let options = OrchestratorStreamOptions {
    max_steps: 10,
    on_chunk: Some(Arc::new(|chunk: &str| {
        print!("{}", chunk);
    })),
    on_step_finish: Some(Arc::new(|step: &StepResult| {
        println!("\n[Step {} complete]", step.step_number);
    })),
    on_finish: Some(Arc::new(|steps: &[StepResult]| {
        println!("\n[Orchestration complete: {} steps]", steps.len());
    })),
    ..Default::default()
};

let orchestration = generate_stream_owned(
    client,
    messages,
    Some(tools),
    resolver,
    vec![step_count_is(10)],
    options,
).await?;

// Consume stream
let mut stream = orchestration.stream;
while let Some(event) = stream.next().await {
    match event {
        Ok(StreamEvent::Delta { delta, .. }) => {
            if let Some(content) = delta.content {
                print!("{}", content);
            }
        }
        Ok(StreamEvent::End { .. }) => break,
        Err(e) => eprintln!("Stream error: {}", e),
        _ => {}
    }
}

// Get final steps
let steps = orchestration.steps.await?;
```

## 🎯 Use Cases

### Research Assistant

```rust
let research_agent = ToolLoopAgent::new(
    client,
    vec![
        Tool::function("search_web", "Search the web"),
        Tool::function("read_article", "Read article content"),
        Tool::function("finalAnswer", "Provide final answer"),
    ],
    vec![any_of(vec![
        has_tool_call("finalAnswer"),
        step_count_is(20),
    ])],
)
.with_system("You are a research assistant. Gather information and provide a comprehensive answer.");
```

### Code Generator with Safety

```rust
let options = OrchestratorOptions {
    on_tool_approval: Some(Arc::new(|tool, args| {
        if tool == "execute_code" {
            let code = args["code"].as_str().unwrap();
            if is_safe_code(code) {
                ToolApproval::Approve(args.clone())
            } else {
                ToolApproval::Deny {
                    reason: "Code contains unsafe operations".to_string(),
                }
            }
        } else {
            ToolApproval::Approve(args.clone())
        }
    })),
    ..Default::default()
};
```

### Customer Support Bot

```rust
let support_agent = ToolLoopAgent::new(client, tools, vec![step_count_is(15)])
    .with_system("You are a customer support agent. Help users with their issues.")
    .on_step_finish(Arc::new(|step| {
        // Log to analytics
        log_event("support_step", json!({
            "tool_calls": step.tool_calls.len(),
            "tokens": step.usage.as_ref().map(|u| u.total_tokens),
        }));
    }))
    .on_finish(Arc::new(|steps| {
        // Log conversation completion
        log_event("support_complete", json!({
            "total_steps": steps.len(),
            "total_tokens": StepResult::merge_usage(steps).total_tokens,
        }));
    }));
```

## 📖 API Reference

### Types

- **`StepResult`**: Result of a single orchestration step
  - `messages`: Messages exchanged in this step
  - `finish_reason`: Why the step ended
  - `usage`: Token usage for this step
  - `tool_calls`: Tools called in this step

- **`ToolApproval`**: Decision for tool execution
  - `Approve(args)`: Execute with given arguments
  - `Modify(args)`: Execute with modified arguments
  - `Deny { reason }`: Block execution

- **`ToolChoice`**: Tool selection strategy
  - `Auto`: Model decides
  - `Required`: Must call a tool
  - `None`: No tool calls allowed
  - `Specific { tool_name }`: Must call specific tool

### Traits

- **`ToolResolver`**: Execute tools by name
  ```rust
  #[async_trait::async_trait]
  pub trait ToolResolver: Send + Sync {
      async fn call_tool(&self, name: &str, arguments: Value) 
          -> Result<Value, LlmError>;
  }
  ```

- **`StopCondition`**: Determine when to stop
  ```rust
  pub trait StopCondition: Send + Sync {
      fn should_stop(&self, steps: &[StepResult]) -> bool;
  }
  ```

## 🔗 Examples

See the [examples directory](../../../examples/03-advanced-features/orchestrator/) for complete working examples:

- `basic-orchestrator.rs` - Multi-step tool calling basics
- `agent-pattern.rs` - Reusable agent abstraction
- `stop-conditions.rs` - Advanced stop condition usage
- `tool-approval.rs` - Tool approval workflow
- `streaming-orchestrator.rs` - Real-time streaming orchestration

## 📝 Migration from 0.10.x

The orchestrator is a new feature in 0.11.0. If you were using custom tool calling loops, you can now use the orchestrator for better ergonomics and features.

**Before (0.10.x):**
```rust
// Manual tool calling loop
loop {
    let response = client.chat_request(request).await?;
    if response.tool_calls.is_empty() {
        break;
    }
    // Execute tools manually
    // Add results to messages
    // Repeat
}
```

**After (0.11.0):**
```rust
use siumai_extras::orchestrator::{generate, step_count_is};

let (response, steps) = generate(
    &client,
    messages,
    Some(tools),
    Some(&resolver),
    &[&*step_count_is(10)],
    OrchestratorOptions::default(),
).await?;
```

## 🤝 Contributing

The orchestrator module is designed to be extensible. You can:
- Implement custom `StopCondition`s
- Create custom `ToolResolver`s
- Build higher-level abstractions on top of `ToolLoopAgent`

See the [contribution guide](../../../../CONTRIBUTING.md) for more details.
