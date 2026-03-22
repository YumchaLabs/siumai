//! Bridge Customization Example
//!
//! This example demonstrates the recommended reusable customization path for
//! request bridging:
//! - implement `BridgeCustomization`
//! - attach it via `BridgeOptions::with_customization(...)`
//! - branch on typed bridge context instead of patching raw route glue
//!
//! Run:
//! ```bash
//! cargo run -p siumai --example bridge-customization --features openai
//! ```

use std::sync::Arc;

use serde_json::json;
use siumai::experimental::bridge::{
    BridgeCustomization, BridgeMode, BridgeOptions, BridgePrimitiveContext, BridgeReport,
    BridgeTarget, BridgeWarning, BridgeWarningKind, RequestBridgeContext, RequestBridgePhase,
    bridge_chat_request_to_openai_responses_json_with_options,
};
use siumai::prelude::unified::{ChatMessage, ChatRequest, Tool, ToolChoice};

struct TenantRequestCustomization {
    tenant: &'static str,
}

impl BridgeCustomization for TenantRequestCustomization {
    fn transform_request(
        &self,
        ctx: &RequestBridgeContext,
        request: &mut ChatRequest,
        report: &mut BridgeReport,
    ) -> Result<(), siumai::prelude::unified::LlmError> {
        if ctx.phase == RequestBridgePhase::SerializeTarget
            && ctx.target == BridgeTarget::OpenAiResponses
        {
            request.common_params.max_tokens = Some(96);
            report.add_warning(BridgeWarning::new(
                BridgeWarningKind::Custom,
                "customization set max_tokens before target serialization",
            ));
        }
        Ok(())
    }

    fn transform_request_json(
        &self,
        ctx: &RequestBridgeContext,
        body: &mut serde_json::Value,
        report: &mut BridgeReport,
    ) -> Result<(), siumai::prelude::unified::LlmError> {
        if ctx.target != BridgeTarget::OpenAiResponses {
            return Ok(());
        }

        let root = body
            .as_object_mut()
            .expect("OpenAI Responses bridge should produce an object body");
        let metadata = root.entry("metadata").or_insert_with(|| json!({}));
        let metadata = metadata
            .as_object_mut()
            .expect("metadata should remain a JSON object");

        metadata.insert("tenant".to_string(), json!(self.tenant));
        metadata.insert("customized".to_string(), json!(true));
        metadata.insert(
            "path".to_string(),
            json!(ctx.path_label.as_deref().unwrap_or("unknown")),
        );

        report.add_warning(BridgeWarning::new(
            BridgeWarningKind::Custom,
            "customization injected OpenAI Responses metadata overlay",
        ));
        Ok(())
    }

    fn validate_request_json(
        &self,
        ctx: &RequestBridgeContext,
        body: &serde_json::Value,
        report: &mut BridgeReport,
    ) -> Result<(), siumai::prelude::unified::LlmError> {
        assert_eq!(ctx.phase, RequestBridgePhase::SerializeTarget);
        assert_eq!(body["metadata"]["tenant"], json!(self.tenant));
        assert_eq!(body["metadata"]["customized"], json!(true));

        report.add_warning(BridgeWarning::new(
            BridgeWarningKind::Custom,
            "customization validated target JSON after serialization",
        ));
        Ok(())
    }

    fn remap_tool_name(&self, ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        if ctx.target == BridgeTarget::OpenAiResponses {
            Some(format!("{}__{name}", self.tenant))
        } else {
            None
        }
    }
}

fn build_request() -> ChatRequest {
    ChatRequest::builder()
        .message(ChatMessage::user("Check the weather in Tokyo.").build())
        .tools(vec![Tool::function(
            "weather",
            "Get the weather for a city",
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::tool("weather"))
        .model("gpt-5-mini")
        .build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = build_request();
    let options = BridgeOptions::new(BridgeMode::BestEffort)
        .with_route_label("examples.bridge-customization")
        .with_customization(Arc::new(TenantRequestCustomization { tenant: "tenant_a" }));

    let bridged = bridge_chat_request_to_openai_responses_json_with_options(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        options,
    )?;

    let body = bridged.value.expect("bridge should produce JSON");

    println!("Bridge report:");
    println!("{}", serde_json::to_string_pretty(&bridged.report)?);
    println!();
    println!("Target JSON:");
    println!("{}", serde_json::to_string_pretty(&body)?);

    Ok(())
}
