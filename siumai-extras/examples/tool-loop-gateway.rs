//! Multi-protocol Tool-Loop SSE Gateway (Gemini backend)
//!
//! This example demonstrates how to:
//! - Stream from an upstream provider (Gemini) using `siumai` unified streaming
//! - Detect tool calls in the upstream stream
//! - Execute tools in-process (gateway-side)
//! - Feed tool results back into the next model step
//! - Keep a single downstream SSE stream open across tool-loop rounds
//! - Expose multiple downstream protocol surfaces from the same upstream backend
//!   (OpenAI Responses / OpenAI Chat Completions / Anthropic Messages / Gemini GenerateContent)
//!
//! ## Setup
//! ```bash
//! export GOOGLE_API_KEY="your-key"
//! ```
//!
//! ## Run
//! ```bash
//! cargo run -p siumai-extras --example tool-loop-gateway --features "server,google,openai,anthropic"
//! ```
//!
//! ## Test
//! ```bash
//! # Tool-loop will typically emit tool-call + tool-result parts before the final answer.
//! curl -N "http://127.0.0.1:3001/v1/responses"
//! curl -N "http://127.0.0.1:3001/v1/chat/completions?lossy=1"
//! curl -N "http://127.0.0.1:3001/anthropic/messages"
//! curl -N "http://127.0.0.1:3001/gemini/generateContent?emit_function_response=1"
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    Router,
    extract::{Query, State},
    response::Response,
    routing::get,
};
use serde::Deserialize;
use serde_json::{Value, json};
use siumai::prelude::*;
use siumai_extras::orchestrator::ToolResolver;
use siumai_extras::server::axum::{
    TargetSseFormat, TranscodeSseOptions, to_transcoded_sse_response,
};
use siumai_extras::server::tool_loop::{ToolLoopGatewayOptions, tool_loop_chat_stream};

#[derive(Clone)]
struct AppState {
    model: Arc<Siumai>,
    tools: Vec<Tool>,
    resolver: Arc<dyn ToolResolver + Send + Sync>,
}

#[derive(Debug, Clone, Deserialize)]
struct GatewayQuery {
    /// Prompt override.
    prompt: Option<String>,
    /// Enable lossy downgrade for unsupported v3 parts.
    lossy: Option<bool>,
    /// For Gemini target only: emit gateway tool results as `functionResponse` frames.
    emit_function_response: Option<bool>,
}

struct LocalToolResolver;

#[async_trait]
impl ToolResolver for LocalToolResolver {
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
        match name {
            "get_weather" => {
                let city = arguments
                    .get("city")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");
                Ok(json!({
                    "city": city,
                    "temperature_c": 26,
                    "condition": "sunny"
                }))
            }
            _ => Err(LlmError::InternalError(format!("Unknown tool: {name}"))),
        }
    }
}

fn internal_error_response(error: &LlmError) -> Response {
    let body = format!("failed to start stream: {}", error.user_message());
    Response::builder()
        .status(500)
        .header("content-type", "text/plain")
        .body(axum::body::Body::from(body))
        .unwrap_or_else(|_| Response::new(axum::body::Body::from("internal error")))
}

fn transcode_opts(q: &GatewayQuery) -> TranscodeSseOptions {
    let mut opts = if q.lossy.unwrap_or(false) {
        TranscodeSseOptions::lossy_text()
    } else {
        TranscodeSseOptions::strict()
    };
    opts.gemini_emit_function_response_tool_results = q.emit_function_response.unwrap_or(false);
    opts
}

async fn start_tool_loop_stream(state: &AppState, prompt: String) -> Result<ChatStream, LlmError> {
    let messages = vec![ChatMessage::user(prompt).build()];
    tool_loop_chat_stream(
        state.model.clone(),
        messages,
        state.tools.clone(),
        state.resolver.clone(),
        ToolLoopGatewayOptions { max_steps: 6 },
    )
    .await
}

async fn responses(State(state): State<AppState>, Query(q): Query<GatewayQuery>) -> Response {
    let prompt = q.prompt.clone().unwrap_or_else(|| {
        "What's the weather in Guangzhou today? Call the tool if needed.".to_string()
    });

    let stream = match start_tool_loop_stream(&state, prompt).await {
        Ok(s) => s,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_sse_response(stream, TargetSseFormat::OpenAiResponses, transcode_opts(&q))
}

async fn chat_completions(
    State(state): State<AppState>,
    Query(q): Query<GatewayQuery>,
) -> Response {
    let prompt = q.prompt.clone().unwrap_or_else(|| {
        "Use the tool to get the weather for Guangzhou, then answer in one sentence.".to_string()
    });

    let stream = match start_tool_loop_stream(&state, prompt).await {
        Ok(s) => s,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::OpenAiChatCompletions,
        transcode_opts(&q),
    )
}

#[cfg(feature = "anthropic")]
async fn anthropic_messages(
    State(state): State<AppState>,
    Query(q): Query<GatewayQuery>,
) -> Response {
    let prompt = q.prompt.clone().unwrap_or_else(|| {
        "Use the tool to get weather for Guangzhou, then answer as JSON: {\"summary\":\"...\"}."
            .to_string()
    });

    let stream = match start_tool_loop_stream(&state, prompt).await {
        Ok(s) => s,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::AnthropicMessages,
        transcode_opts(&q),
    )
}

#[cfg(feature = "google")]
async fn gemini_generate_content(
    State(state): State<AppState>,
    Query(q): Query<GatewayQuery>,
) -> Response {
    let prompt = q.prompt.clone().unwrap_or_else(|| {
        "Use the tool to get the weather for Guangzhou, then answer briefly.".to_string()
    });

    let stream = match start_tool_loop_stream(&state, prompt).await {
        Ok(s) => s,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::GeminiGenerateContent,
        transcode_opts(&q),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GOOGLE_API_KEY")?;

    let model = Siumai::builder()
        .gemini()
        .api_key(&api_key)
        .model("gemini-2.0-flash-exp")
        .build()
        .await?;

    let tools = vec![Tool::function(
        "get_weather".to_string(),
        "Get current weather for a city".to_string(),
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }),
    )];

    let state = AppState {
        model: Arc::new(model),
        tools,
        resolver: Arc::new(LocalToolResolver),
    };

    let app = Router::<AppState>::new()
        .route("/v1/responses", get(responses))
        .route("/v1/chat/completions", get(chat_completions));

    #[cfg(feature = "anthropic")]
    let app = app.route("/anthropic/messages", get(anthropic_messages));

    #[cfg(feature = "google")]
    let app = app.route("/gemini/generateContent", get(gemini_generate_content));

    let app: Router = app.with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3001").await?;
    println!("Listening on http://127.0.0.1:3001/v1/responses");
    println!("Listening on http://127.0.0.1:3001/v1/chat/completions");
    println!("Listening on http://127.0.0.1:3001/anthropic/messages");
    println!("Listening on http://127.0.0.1:3001/gemini/generateContent");
    axum::serve(listener, app).await?;

    Ok(())
}
