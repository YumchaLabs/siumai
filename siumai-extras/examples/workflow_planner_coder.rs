//! Basic workflow example with planner + coder workers and in-memory memory.
//!
//! This example demonstrates how to:
//! - Build a `Workflow` on top of `Orchestrator` + `ToolLoopAgent`
//! - Register semantic worker roles (`planner`, `coder`)
//! - Attach an in-memory `WorkflowMemory`
//! - Run the workflow twice with the same session key and observe accumulated state
//!
//! Run with:
//! ```bash
//! cargo run -p siumai-extras --example workflow_planner_coder --features openai
//! ```

use std::sync::Arc;

use serde_json::json;
use siumai::error::LlmError;
use siumai::prelude::Siumai;
use siumai::types::{ChatMessage, OutputSchema, Tool};
use siumai_extras::orchestrator::{
    InMemoryWorkflowMemory, ToolLoopAgent, ToolResolver, WORKER_CODER, WORKER_PLANNER,
    WorkflowBuilder,
};

/// Dummy base tool resolver used for non-worker tools.
///
/// In this example we don't use any non-worker tools, so this resolver just
/// logs calls and returns a simple JSON payload.
struct DummyToolResolver;

#[async_trait::async_trait]
impl ToolResolver for DummyToolResolver {
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        println!("[base resolver] tool={name}, args={arguments}");
        Ok(json!({ "status": "ok" }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a unified client (OpenAI in this example).
    //
    // API key is read from OPENAI_API_KEY by default. You can also call
    // .api_key("...") explicitly if needed.
    let model = Siumai::builder()
        .openai()
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Define tools used by the top-level orchestrator. These are "worker tools"
    // that route into the corresponding workers via the Workflow.
    let planner_tool = Tool::function(
        format!("worker:{WORKER_PLANNER}"),
        "Call the planner worker to break down the task.",
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Instruction for the planner worker"
                }
            },
            "required": ["input"]
        }),
    );

    let coder_tool = Tool::function(
        format!("worker:{WORKER_CODER}"),
        "Call the coder worker to implement a plan as code.",
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Detailed coding task to implement"
                }
            },
            "required": ["input"]
        }),
    );

    let tools = vec![planner_tool, coder_tool];

    // Planner worker: produces a structured JSON plan (list of steps).
    let planner_schema = json!({
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": { "type": "string" },
                "description": "High-level steps to solve the task"
            }
        },
        "required": ["steps"]
    });

    let planner_agent = ToolLoopAgent::with_defaults(model.clone(), vec![])
        .with_system(
            "You are a planner. Break down the user request into clear, numbered steps. \
             Always respond with JSON only, matching the given schema.",
        )
        .with_output_schema(
            OutputSchema::new(planner_schema.clone())
                .with_name("plan")
                .with_description("Plan steps for the task"),
        )
        .with_temperature(0.2);

    // Coder worker: turns a task into Rust code (plain text output).
    let coder_agent = ToolLoopAgent::with_defaults(model.clone(), vec![])
        .with_system(
            "You are a senior Rust engineer. Given a task description, produce clear Rust code \
             with minimal comments. Focus on correctness and simplicity.",
        )
        .with_temperature(0.4);

    // Attach in-memory workflow memory so we can persist state across runs.
    let memory = Arc::new(InMemoryWorkflowMemory::new());

    // Build the workflow with semantic worker helpers and memory.
    let workflow = WorkflowBuilder::new(model.clone(), tools)
        .max_steps(8)
        .with_planner_agent(planner_agent)
        .with_coder_agent(coder_agent)
        .with_memory(memory.clone())
        .build();

    let resolver = DummyToolResolver;
    let session_key = "demo-session-1";

    println!("=== Workflow: planner + coder + in-memory memory ===\n");

    // First run: ask the workflow to implement a simple function.
    let messages_run1 = vec![
        ChatMessage::system(
            "You are an orchestrator. For complex tasks, first call the planner worker \
             to create a plan, then call the coder worker to implement it. \
             Finally, respond to the user with a helpful answer.",
        )
        .build(),
        ChatMessage::user(
            "Write a Rust function that sorts a list of integers and explain its complexity.",
        )
        .build(),
    ];

    let (resp1, steps1, state1) = workflow
        .run_with_memory(session_key, messages_run1, Some(&resolver))
        .await?;

    println!("--- Run 1: final answer ---");
    println!("{}", resp1.content_text().unwrap_or("<no text>"));
    println!("\nSteps taken (run 1): {}", steps1.len());
    if let Some(plan) = state1.worker_outputs.get(WORKER_PLANNER) {
        println!("\nPlanner structured output (run 1):\n{}", plan);
    }
    if let Some(planner_steps) = state1.worker_steps.get(WORKER_PLANNER) {
        println!("Planner steps recorded (run 1): {}", planner_steps.len());
    }

    // Second run: ask a different question using the same session key.
    let messages_run2 = vec![
        ChatMessage::system(
            "You are an orchestrator. Reuse prior knowledge from this session when helpful.",
        )
        .build(),
        ChatMessage::user("Now write a Rust function that filters even numbers from a list.")
            .build(),
    ];

    let (resp2, steps2, state2) = workflow
        .run_with_memory(session_key, messages_run2, Some(&resolver))
        .await?;

    println!("\n--- Run 2: final answer ---");
    println!("{}", resp2.content_text().unwrap_or("<no text>"));
    println!("\nSteps taken (run 2): {}", steps2.len());
    if let Some(plan) = state2.worker_outputs.get(WORKER_PLANNER) {
        println!("\nPlanner structured output (run 2):\n{}", plan);
    }
    if let Some(planner_steps) = state2.worker_steps.get(WORKER_PLANNER) {
        println!("Planner steps recorded (run 2): {}", planner_steps.len());
    }

    // Inspect accumulated state from the in-memory store.
    if let Some(acc_state) = memory.load(session_key).await? {
        if let Some(planner_steps) = acc_state.worker_steps.get(WORKER_PLANNER) {
            println!(
                "\n[Memory] Total planner steps across runs: {}",
                planner_steps.len()
            );
        }
        println!(
            "[Memory] Total orchestration steps across runs: {}",
            acc_state.orchestration_steps.len()
        );
    }

    Ok(())
}
