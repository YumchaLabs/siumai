//! Gateway Loss Policy Example
//!
//! This example demonstrates how to:
//! - encode a known lossy response/stream bridge route under `BridgeMode::Strict`
//! - show the default mode-aware policy rejecting lossy conversion
//! - install a custom `BridgeLossPolicy` to allow only selected lossy fields
//! - keep the policy close to the bridge instead of scattering route-local `if` logic
//! - drive both JSON and SSE routes through the same `siumai-extras` gateway helper surface
//!
//! This example is fully local and does not require any provider API key.
//!
//! Route recipe:
//! - Recipe 3 from `docs/workstreams/protocol-bridge-gateway/route-recipes.md`
//! - source-aware target transcode with explicit strict-vs-custom-loss-policy behavior
//!
//! ## Run
//! ```bash
//! cargo run -p siumai-extras --example gateway-loss-policy --features "server,openai,anthropic"
//! ```
//!
//! ## Test
//! ```bash
//! curl -i http://127.0.0.1:3004/anthropic/messages.json
//! curl -i "http://127.0.0.1:3004/anthropic/messages.json?policy=allowlisted"
//! curl -i "http://127.0.0.1:3004/anthropic/messages.json?policy=continue"
//! curl -i http://127.0.0.1:3004/anthropic/messages.sse
//! curl -N "http://127.0.0.1:3004/anthropic/messages.sse?policy=allowlisted"
//! curl -N "http://127.0.0.1:3004/anthropic/messages.sse?policy=continue"
//! ```

use std::{collections::HashMap, sync::Arc};

use axum::{Router, extract::Query, response::Response, routing::get};
use futures::stream;
use serde::Deserialize;
use serde_json::json;
use siumai::experimental::bridge::{
    BridgeLossAction, BridgeLossPolicy, BridgeMode, BridgeOptionsOverride, BridgeReport,
    BridgeTarget, RequestBridgeContext, ResponseBridgeContext, StreamBridgeContext,
};
use siumai::prelude::unified::*;
use siumai_extras::server::{
    GatewayBridgePolicy,
    axum::{
        TargetJsonFormat, TargetSseFormat, TranscodeJsonOptions, TranscodeSseOptions,
        to_transcoded_json_response, to_transcoded_sse_response,
    },
};

#[derive(Debug, Clone, Deserialize)]
struct GatewayQuery {
    /// `default`, `allowlisted`, or `continue`.
    policy: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolicyKind {
    Default,
    Allowlisted,
    Continue,
}

impl PolicyKind {
    fn parse(value: Option<&str>) -> Self {
        match value {
            Some("allowlisted") => Self::Allowlisted,
            Some("continue") => Self::Continue,
            _ => Self::Default,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Allowlisted => "allowlisted",
            Self::Continue => "continue",
        }
    }
}

#[derive(Debug)]
struct ContinueLossyPolicy;

impl BridgeLossPolicy for ContinueLossyPolicy {
    fn request_action(
        &self,
        _ctx: &RequestBridgeContext,
        _report: &BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }

    fn response_action(
        &self,
        _ctx: &ResponseBridgeContext,
        _report: &BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }

    fn stream_action(
        &self,
        _ctx: &StreamBridgeContext,
        _report: &BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }
}

#[derive(Debug)]
struct AllowlistedLossyPolicy;

impl BridgeLossPolicy for AllowlistedLossyPolicy {
    fn request_action(
        &self,
        _ctx: &RequestBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        if report.is_exact() {
            BridgeLossAction::Continue
        } else {
            BridgeLossAction::Reject
        }
    }

    fn response_action(
        &self,
        ctx: &ResponseBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        if report.is_rejected() {
            return BridgeLossAction::Reject;
        }
        if ctx.target != BridgeTarget::AnthropicMessages {
            return BridgeLossAction::Reject;
        }

        let allowed_lossy_fields = [
            "usage.prompt_tokens_details",
            "usage.completion_tokens_details",
        ];
        let allowed_dropped_fields = ["provider_metadata.openai"];

        if report
            .lossy_fields
            .iter()
            .all(|field| allowed_lossy_fields.contains(&field.as_str()))
            && report
                .dropped_fields
                .iter()
                .all(|field| allowed_dropped_fields.contains(&field.as_str()))
            && report.unsupported_capabilities.is_empty()
        {
            BridgeLossAction::Continue
        } else {
            BridgeLossAction::Reject
        }
    }

    fn stream_action(&self, ctx: &StreamBridgeContext, report: &BridgeReport) -> BridgeLossAction {
        if report.is_rejected() {
            return BridgeLossAction::Reject;
        }
        if ctx.target != BridgeTarget::AnthropicMessages {
            return BridgeLossAction::Reject;
        }

        if report
            .lossy_fields
            .iter()
            .all(|field| field == "stream.protocol")
            && report.dropped_fields.is_empty()
            && report.unsupported_capabilities.is_empty()
        {
            BridgeLossAction::Continue
        } else {
            BridgeLossAction::Reject
        }
    }
}

fn synthetic_lossy_response() -> ChatResponse {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("searching"),
        ContentPart::tool_call("call_1", "web_search", json!({ "q": "rust" }), Some(true)),
    ]));
    response.id = Some("resp_1".to_string());
    response.model = Some("gpt-4o-mini".to_string());
    response.finish_reason = Some(FinishReason::ToolCalls);
    response.usage = Some(
        Usage::builder()
            .prompt_tokens(10)
            .completion_tokens(5)
            .total_tokens(15)
            .with_cached_tokens(3)
            .with_reasoning_tokens(2)
            .build(),
    );
    response.provider_metadata = Some(HashMap::from([(
        "openai".to_string(),
        HashMap::from([("responseId".to_string(), json!("resp_1"))]),
    )]));
    response
}

fn synthetic_cross_protocol_stream() -> ChatStream {
    let mut response = ChatResponse::new(MessageContent::Text("gateway ok".to_string()));
    response.id = Some("resp_1".to_string());
    response.model = Some("gpt-4o-mini".to_string());
    response.finish_reason = Some(FinishReason::Stop);

    Box::pin(stream::iter(vec![
        Ok(ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_1".to_string()),
                model: Some("gpt-4o-mini".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
            },
        }),
        Ok(ChatStreamEvent::ContentDelta {
            delta: "gateway ok".to_string(),
            index: None,
        }),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]))
}

fn bridge_options_override(policy: PolicyKind) -> Option<BridgeOptionsOverride> {
    match policy {
        PolicyKind::Default => None,
        PolicyKind::Allowlisted => Some(
            BridgeOptionsOverride::new()
                .with_route_label("examples.gateway-loss-policy.allowlisted")
                .with_loss_policy(Arc::new(AllowlistedLossyPolicy)),
        ),
        PolicyKind::Continue => Some(
            BridgeOptionsOverride::new()
                .with_route_label("examples.gateway-loss-policy.continue")
                .with_loss_policy(Arc::new(ContinueLossyPolicy)),
        ),
    }
}

fn gateway_policy(policy: PolicyKind) -> GatewayBridgePolicy {
    let mut gateway_policy = GatewayBridgePolicy::new(BridgeMode::Strict)
        .with_route_label(format!("examples.gateway-loss-policy.{}", policy.as_str()))
        .with_bridge_headers(true)
        .with_bridge_warning_headers(true);

    if let Some(override_options) = bridge_options_override(policy) {
        gateway_policy = gateway_policy.with_bridge_options_override(override_options);
    }

    gateway_policy
}

async fn anthropic_json(Query(query): Query<GatewayQuery>) -> Response {
    let policy = PolicyKind::parse(query.policy.as_deref());
    let response = synthetic_lossy_response();

    let mut http_response = to_transcoded_json_response(
        response,
        TargetJsonFormat::AnthropicMessages,
        TranscodeJsonOptions::default().with_policy(gateway_policy(policy)),
    );
    http_response.headers_mut().insert(
        "x-siumai-loss-policy",
        policy.as_str().parse().expect("header value"),
    );
    http_response
}

async fn anthropic_sse(Query(query): Query<GatewayQuery>) -> Response {
    let policy = PolicyKind::parse(query.policy.as_deref());
    let mut http_response = to_transcoded_sse_response(
        synthetic_cross_protocol_stream(),
        TargetSseFormat::AnthropicMessages,
        TranscodeSseOptions::default()
            .with_bridge_source(BridgeTarget::OpenAiResponses)
            .with_policy(gateway_policy(policy)),
    );
    http_response.headers_mut().insert(
        "x-siumai-loss-policy",
        policy.as_str().parse().expect("header value"),
    );
    http_response
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app: Router = Router::new()
        .route("/anthropic/messages.json", get(anthropic_json))
        .route("/anthropic/messages.sse", get(anthropic_sse));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3004").await?;
    println!("Listening on http://127.0.0.1:3004/anthropic/messages.json");
    println!("Listening on http://127.0.0.1:3004/anthropic/messages.sse");
    axum::serve(listener, app).await?;

    Ok(())
}
