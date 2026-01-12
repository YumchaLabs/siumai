# Orchestrator Examples

The orchestrator module provides advanced multi-step tool calling capabilities with flexible control flow, similar to Vercel AI SDK's agent system.

> Note: The orchestrator is implemented in the `siumai-extras` crate.
> As part of the beta.5 split refactor, the runnable orchestrator examples live under
> `siumai-extras/examples/*`. The `siumai` crate no longer ships duplicate orchestrator examples to
> avoid workspace target name collisions.

## 🎯 What is Orchestration?

Orchestration enables AI agents to:
- **Execute multiple tool calls** across several steps
- **Make decisions** based on tool results
- **Iterate** until a goal is achieved or a stop condition is met
- **Track progress** with callbacks and telemetry
- **Control execution** with approval workflows and dynamic configuration

## 📚 Examples Overview

| Example | Complexity | Features Demonstrated |
|---------|-----------|----------------------|
| `siumai-extras/examples/basic-orchestrator.rs` | ⭐ Beginner | Multi-step execution, callbacks, usage tracking |
| `siumai-extras/examples/agent-pattern.rs` | ⭐⭐ Intermediate | Reusable agents, builder pattern, multiple conversations |
| `siumai-extras/examples/stop-conditions.rs` | ⭐⭐ Intermediate | Built-in conditions, combinators, custom predicates |
| `siumai-extras/examples/tool-approval.rs` | ⭐⭐⭐ Advanced | Security workflows, argument modification, denial |
| `siumai-extras/examples/streaming-orchestrator.rs` | ⭐⭐⭐ Advanced | Real-time streaming, progress tracking, cancellation |
| `siumai-extras/examples/streaming-tool-execution.rs` | ⭐⭐⭐ Advanced | Streaming + provider tool execution |

## 🚀 Quick Start

Run any example via:

```bash
cargo run -p siumai-extras --example basic-orchestrator
```

### Basic Multi-Step Tool Calling

```rust
use siumai_extras::orchestrator::{generate, step_count_is, OrchestratorOptions, ToolResolver};
use siumai::prelude::*;

// Define your tool resolver
struct MyResolver;

#[async_trait::async_trait]
impl ToolResolver for MyResolver {
    async fn call_tool(&self, name: &str, args: serde_json::Value) 
        -> Result<serde_json::Value, LlmError> {
        // Execute tools here
        Ok(serde_json::json!({"result": "success"}))
    }
}

// Run orchestration
let stop_condition = step_count_is(10);
let (response, steps) = generate(
    &client,
    messages,
    Some(tools),
    Some(&resolver),
    &[&*stop_condition],
    OrchestratorOptions::default(),
).await?;
```

### Using the Agent Pattern

```rust
use siumai_extras::orchestrator::{ToolLoopAgent, step_count_is};

// Create a reusable agent
let agent = ToolLoopAgent::new(client, tools, vec![step_count_is(10)])
    .with_system("You are a helpful assistant")
    .with_id("my-agent")
    .on_step_finish(Arc::new(|step| {
        println!("Step completed: {} tool calls", step.tool_calls.len());
    }));

// Use it multiple times
let (response1, _) = agent.generate(messages1, &resolver).await?;
let (response2, _) = agent.generate(messages2, &resolver).await?;
```

## 🎓 Learning Path

### 1. Start with `basic-orchestrator.rs`
Learn the fundamentals:
- How to implement `ToolResolver`
- Setting up orchestrator options
- Using callbacks (`on_step_finish`, `on_finish`)
- Tracking token usage across steps

**Run:**
```bash
cargo run -p siumai-extras --example basic-orchestrator
```

### 2. Explore `agent-pattern.rs`
Understand reusable agents:
- Creating agents with `ToolLoopAgent::new()`
- Builder pattern configuration
- Using the same agent for multiple conversations
- Agent lifecycle management

**Run:**
```bash
cargo run -p siumai-extras --example agent-pattern
```

### 3. Master `stop-conditions.rs`
Control execution flow:
- Built-in conditions: `step_count_is()`, `has_tool_call()`, `has_text_response()`
- Combining conditions with `any_of()` and `all_of()`
- Creating custom conditions with predicates
- Designing complex stop logic

**Run:**
```bash
cargo run -p siumai-extras --example stop-conditions
```

### 4. Implement `tool-approval.rs`
Add security workflows:
- Approving tool calls before execution
- Modifying dangerous arguments
- Denying risky operations
- Building safe AI systems

**Run:**
```bash
cargo run -p siumai-extras --example tool-approval
```

### 5. Advanced: `streaming-orchestrator.rs`
Real-time execution:
- Streaming multi-step orchestration
- Progress tracking with callbacks
- Handling stream events
- Graceful cancellation

**Run:**
```bash
cargo run -p siumai-extras --example streaming-orchestrator
```

## 🔑 Key Concepts

### Stop Conditions

Control when orchestration should stop:

```rust
use siumai_extras::orchestrator::*;

// Stop after 10 steps
let condition1 = step_count_is(10);

// Stop when a specific tool is called
let condition2 = has_tool_call("finalAnswer");

// Stop when model generates text (no tool calls)
let condition3 = has_text_response();

// Combine conditions: stop if ANY is true
let combined = any_of(vec![
    step_count_is(20),
    has_tool_call("finalAnswer"),
]);

// Custom condition
let custom = custom_condition(|steps| {
    steps.iter().any(|s| s.tool_calls.len() > 5)
});
```

### Tool Approval

Control tool execution with approval workflows:

```rust
use siumai_extras::orchestrator::ToolApproval;

let options = OrchestratorOptions {
    on_tool_approval: Some(Arc::new(|tool_name, args| {
        match tool_name {
            // Deny dangerous operations
            "delete_file" => ToolApproval::Deny {
                reason: "File deletion not allowed".to_string(),
            },
            
            // Modify arguments
            "write_file" => {
                let mut modified = args.clone();
                // Redirect to safe directory
                modified["path"] = json!("/tmp/safe_file.txt");
                ToolApproval::Modify(modified)
            },
            
            // Approve as-is
            _ => ToolApproval::Approve(args.clone()),
        }
    })),
    ..Default::default()
};
```

### Dynamic Step Preparation

Modify orchestrator behavior before each step:

```rust
use siumai_extras::orchestrator::{PrepareStepContext, PrepareStepResult, ToolChoice};

let options = OrchestratorOptions {
    prepare_step: Some(Arc::new(|ctx: PrepareStepContext| {
        PrepareStepResult {
            // Force tool usage on first step
            tool_choice: if ctx.step_number == 0 {
                Some(ToolChoice::Required)
            } else {
                Some(ToolChoice::Auto)
            },
            
            // Filter tools based on step
            active_tools: if ctx.step_number > 5 {
                Some(vec!["finalAnswer".to_string()])
            } else {
                None
            },
            
            // Modify system message
            system: Some(format!(
                "Step {}/10. Focus on the goal.",
                ctx.step_number + 1
            )),
            
            messages: None,
        }
    })),
    ..Default::default()
};
```

### Agent Abstraction

Create reusable agents for common workflows:

```rust
use siumai_extras::orchestrator::ToolLoopAgent;

// Create agent
let weather_agent = ToolLoopAgent::new(
    client,
    weather_tools,
    vec![step_count_is(5)],
)
.with_system("You are a weather assistant")
.with_id("weather-agent")
.on_step_finish(Arc::new(|step| {
    println!("Weather agent step: {}", step.tool_calls.len());
}));

// Use for multiple queries
for city in &["Tokyo", "London", "New York"] {
    let messages = vec![user!(format!("Weather in {}", city))];
    let (response, _) = weather_agent.generate(messages, &resolver).await?;
    println!("{}: {}", city, response.content_text().unwrap());
}
```

## 🎯 Use Cases

### Research Assistant
```rust
// Stop when research is complete
let stop = any_of(vec![
    has_tool_call("finalAnswer"),
    step_count_is(20),
]);
```

### Code Generator
```rust
// Approve code execution with safety checks
on_tool_approval: Some(Arc::new(|tool, args| {
    if tool == "execute_code" {
        // Validate code is safe
        if is_safe_code(&args["code"]) {
            ToolApproval::Approve(args.clone())
        } else {
            ToolApproval::Deny { reason: "Unsafe code".into() }
        }
    } else {
        ToolApproval::Approve(args.clone())
    }
}))
```

### Customer Support Bot
```rust
// Track conversation progress
on_step_finish: Some(Arc::new(|step| {
    log_to_analytics("step_completed", step);
})),
on_finish: Some(Arc::new(|steps| {
    log_to_analytics("conversation_completed", steps);
}))
```

## 📊 Comparison with Vercel AI SDK

| Feature | Vercel AI SDK | Siumai Orchestrator |
|---------|---------------|---------------------|
| Multi-step execution | ✅ `generateText` with `stopWhen` | ✅ `generate()` with stop conditions |
| Stop conditions | ✅ `stepCountIs()`, custom | ✅ `step_count_is()`, `has_tool_call()`, `custom_condition()` |
| Step preparation | ✅ `prepareStep` callback | ✅ `PrepareStep` callback |
| Agent abstraction | ✅ `ToolLoopAgent` class | ✅ `ToolLoopAgent<M>` struct |
| Tool approval | ✅ Tool approvals | ✅ `ToolApproval` enum |
| Streaming | ✅ `streamText` | ✅ `generate_stream_owned()` |
| Telemetry | ✅ OpenTelemetry | ✅ `TelemetryConfig` |

## 🔗 Related Documentation

- [Orchestrator Module](../../../../siumai/src/orchestrator/README.md) - Module documentation
- [CHANGELOG](../../../../CHANGELOG.md#orchestrator-new) - What's new in 0.11.0
- [Main README](../../../../README.md) - Project overview

## 💡 Tips

1. **Start Simple**: Begin with `basic-orchestrator.rs` before exploring advanced features
2. **Use Agents**: For repeated workflows, `ToolLoopAgent` provides better ergonomics
3. **Monitor Usage**: Always track token usage with `StepResult::merge_usage()`
4. **Set Limits**: Use stop conditions to prevent infinite loops
5. **Test Tools**: Verify your `ToolResolver` implementation works correctly
6. **Handle Errors**: Implement proper error handling in tool execution
7. **Use Telemetry**: Enable telemetry for production debugging

## 🐛 Troubleshooting

**Orchestration never stops:**
- Check your stop conditions
- Add `step_count_is()` as a safety limit
- Verify tool calls are being detected correctly

**Tools not executing:**
- Ensure `ToolResolver` is implemented correctly
- Check tool names match between definition and resolver
- Verify tool arguments are valid JSON

**High token usage:**
- Use `on_step_finish` to monitor usage per step
- Set lower `max_steps` limit
- Consider using `has_text_response()` to stop early

**Stream not working:**
- Use `generate_stream_owned()` instead of `generate()`
- Ensure model supports streaming
- Check for errors in stream consumption
