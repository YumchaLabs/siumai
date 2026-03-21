//! Anthropic Messages -> OpenAI Responses Gateway
//!
//! This example demonstrates how to:
//! - accept Anthropic Messages request JSON from a downstream client
//! - normalize the request into `ChatRequest` via the explicit bridge API
//! - execute the normalized request on a fixed OpenAI model handle
//! - return OpenAI Responses JSON or SSE to the caller
//!
//! ## Setup
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export OPENAI_GATEWAY_MODEL="openai:gpt-4o-mini" # optional
//! ```
//!
//! ## Run
//! ```bash
//! cargo run -p siumai-extras --example anthropic-to-openai-responses-gateway --features "server,openai,anthropic"
//! ```
//!
//! ## Test
//! ```bash
//! curl -X POST "http://127.0.0.1:3002/bridge/anthropic-to-openai/json?prompt=Explain%20Rust%20ownership%20in%20one%20sentence."
//! curl -N -X POST "http://127.0.0.1:3002/bridge/anthropic-to-openai/sse?prompt=Explain%20Rust%20ownership%20in%20one%20sentence."
//! curl -X POST http://127.0.0.1:3002/bridge/anthropic-to-openai/json \
//!   -H "content-type: application/json" \
//!   -d '{"model":"claude-3-5-haiku-20241022","max_tokens":256,"messages":[{"role":"user","content":"Explain Rust ownership in one sentence."}]}'
//! curl -N -X POST http://127.0.0.1:3002/bridge/anthropic-to-openai/sse \
//!   -H "content-type: application/json" \
//!   -d '{"model":"claude-3-5-haiku-20241022","max_tokens":256,"stream":true,"messages":[{"role":"user","content":"Explain Rust ownership in one sentence."}]}'
//! ```

#[path = "common/gateway_bridge_common.rs"]
mod gateway_bridge_common;

use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    extract::{Query, Request, State},
    response::Response,
    routing::post,
};
use serde_json::{Value, json};
use siumai::prelude::unified::*;
use siumai_extras::server::{
    GatewayBridgePolicy,
    axum::{
        SourceRequestFormat, TargetJsonFormat, TargetSseFormat, TranscodeJsonOptions,
        TranscodeSseOptions, to_transcoded_json_response, to_transcoded_sse_response,
    },
};

use crate::gateway_bridge_common::{
    GatewayQuery, internal_error_response, normalize_source_request_for_backend,
    read_source_request_json_or_prompt,
};

#[derive(Clone)]
struct AppState {
    client: Arc<registry::LanguageModelHandle>,
    backend_model_id: String,
    policy: GatewayBridgePolicy,
}

fn gateway_policy() -> GatewayBridgePolicy {
    GatewayBridgePolicy::new(siumai::experimental::bridge::BridgeMode::BestEffort)
        .with_route_label("examples.gateway.anthropic-to-openai")
        .with_bridge_headers(true)
        .with_bridge_warning_headers(true)
        .with_request_body_limit_bytes(128 * 1024)
        .with_keepalive_interval(Duration::from_secs(15))
        .with_stream_idle_timeout(Duration::from_secs(60))
}

fn default_anthropic_request(prompt: String) -> Value {
    json!({
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    })
}

fn normalize_request(
    body: &Value,
    backend_model_id: &str,
    stream: bool,
    policy: &GatewayBridgePolicy,
) -> Result<ChatRequest, Response> {
    normalize_source_request_for_backend(
        body,
        SourceRequestFormat::AnthropicMessages,
        "Anthropic Messages",
        backend_model_id,
        stream,
        policy,
    )
}

async fn json_route(
    State(state): State<AppState>,
    Query(query): Query<GatewayQuery>,
    request: Request,
) -> Response {
    let prompt = query
        .prompt
        .unwrap_or_else(|| "Explain Rust ownership in one sentence.".to_string());
    let body = match read_source_request_json_or_prompt(
        request,
        &state.policy,
        prompt,
        default_anthropic_request,
    )
    .await
    {
        Ok(body) => body,
        Err(response) => return response,
    };

    let request = match normalize_request(&body, &state.backend_model_id, false, &state.policy) {
        Ok(request) => request,
        Err(response) => return response,
    };

    let response =
        match text::generate(&*state.client, request, text::GenerateOptions::default()).await {
            Ok(response) => response,
            Err(error) => return internal_error_response(&error),
        };

    to_transcoded_json_response(
        response,
        TargetJsonFormat::OpenAiResponses,
        TranscodeJsonOptions::default().with_policy(state.policy.clone()),
    )
}

async fn sse_route(
    State(state): State<AppState>,
    Query(query): Query<GatewayQuery>,
    request: Request,
) -> Response {
    let prompt = query
        .prompt
        .unwrap_or_else(|| "Explain Rust ownership in one sentence.".to_string());
    let body = match read_source_request_json_or_prompt(
        request,
        &state.policy,
        prompt,
        default_anthropic_request,
    )
    .await
    {
        Ok(body) => body,
        Err(response) => return response,
    };

    let request = match normalize_request(&body, &state.backend_model_id, true, &state.policy) {
        Ok(request) => request,
        Err(response) => return response,
    };

    let stream = match text::stream(&*state.client, request, text::StreamOptions::default()).await {
        Ok(stream) => stream,
        Err(error) => return internal_error_response(&error),
    };

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::OpenAiResponses,
        TranscodeSseOptions::default().with_policy(state.policy.clone()),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend_model =
        std::env::var("OPENAI_GATEWAY_MODEL").unwrap_or_else(|_| "openai:gpt-4o-mini".to_string());

    let reg = registry::global();
    let client = reg.language_model(&backend_model)?;
    let backend_model_id = client.model_id().to_string();

    let state = AppState {
        client: Arc::new(client),
        backend_model_id,
        policy: gateway_policy(),
    };

    let app: Router = Router::new()
        .route("/bridge/anthropic-to-openai/json", post(json_route))
        .route("/bridge/anthropic-to-openai/sse", post(sse_route))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3002").await?;
    println!("Listening on http://127.0.0.1:3002/bridge/anthropic-to-openai/json");
    println!("Listening on http://127.0.0.1:3002/bridge/anthropic-to-openai/sse");
    axum::serve(listener, app).await?;

    Ok(())
}
