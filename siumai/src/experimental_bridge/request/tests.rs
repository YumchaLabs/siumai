use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use siumai_core::bridge::{
    BridgeCustomization, BridgeLossAction, BridgeLossPolicy, BridgeMode, BridgeOptions,
    BridgePrimitiveContext, BridgePrimitiveRemapper, BridgeTarget, BridgeWarning,
    BridgeWarningKind, RequestBridgeContext, RequestBridgeHook, RequestBridgePhase,
    ResponseBridgeContext, StreamBridgeContext,
};
use siumai_core::types::{ChatMessage, ChatRequest, ContentPart, ProviderDefinedTool, Tool};

use crate::experimental_bridge::{ProviderToolRewriteCustomization, ProviderToolRewriteRule};

#[cfg(feature = "anthropic")]
use super::{
    bridge_anthropic_messages_json_to_chat_request, bridge_chat_request_to_anthropic_messages_json,
    bridge_chat_request_to_anthropic_messages_json_with_options,
};
#[cfg(feature = "google")]
use super::{
    bridge_chat_request_to_gemini_generate_content_json,
    bridge_gemini_generate_content_json_to_chat_request,
};
#[cfg(feature = "openai")]
use super::{
    bridge_chat_request_to_openai_chat_completions_json,
    bridge_chat_request_to_openai_chat_completions_json_with_options,
    bridge_chat_request_to_openai_responses_json,
    bridge_chat_request_to_openai_responses_json_with_options,
    bridge_openai_chat_completions_json_to_chat_request,
    bridge_openai_responses_json_to_chat_request,
    bridge_openai_responses_json_to_chat_request_with_options,
};

struct PrefixRemapper;

impl BridgePrimitiveRemapper for PrefixRemapper {
    fn remap_tool_name(&self, _ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        Some(format!("gw_{name}"))
    }
}

struct RejectLossyPolicy;

impl BridgeLossPolicy for RejectLossyPolicy {
    fn request_action(
        &self,
        _ctx: &RequestBridgeContext,
        report: &siumai_core::bridge::BridgeReport,
    ) -> BridgeLossAction {
        if report.is_lossy() || report.is_rejected() {
            BridgeLossAction::Reject
        } else {
            BridgeLossAction::Continue
        }
    }

    fn response_action(
        &self,
        _ctx: &ResponseBridgeContext,
        report: &siumai_core::bridge::BridgeReport,
    ) -> BridgeLossAction {
        if report.is_lossy() || report.is_rejected() {
            BridgeLossAction::Reject
        } else {
            BridgeLossAction::Continue
        }
    }

    fn stream_action(
        &self,
        _ctx: &StreamBridgeContext,
        report: &siumai_core::bridge::BridgeReport,
    ) -> BridgeLossAction {
        if report.is_lossy() || report.is_rejected() {
            BridgeLossAction::Reject
        } else {
            BridgeLossAction::Continue
        }
    }
}

struct RequestAuditHook;

impl RequestBridgeHook for RequestAuditHook {
    fn transform_request(
        &self,
        ctx: &RequestBridgeContext,
        request: &mut ChatRequest,
        report: &mut siumai_core::bridge::BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        assert_eq!(ctx.phase, RequestBridgePhase::SerializeTarget);
        assert_eq!(ctx.route_label.as_deref(), Some("tests.request.hook"));
        request.common_params.max_tokens = Some(77);
        report.add_warning(BridgeWarning::new(
            BridgeWarningKind::Custom,
            "request hook transformed normalized request",
        ));
        Ok(())
    }

    fn transform_json(
        &self,
        _ctx: &RequestBridgeContext,
        body: &mut serde_json::Value,
        _report: &mut siumai_core::bridge::BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        body["metadata"] = json!({
            "hooked": true,
        });
        Ok(())
    }

    fn validate_json(
        &self,
        _ctx: &RequestBridgeContext,
        body: &serde_json::Value,
        report: &mut siumai_core::bridge::BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        assert_eq!(body["metadata"]["hooked"], json!(true));
        report.add_warning(BridgeWarning::new(
            BridgeWarningKind::Custom,
            "request hook validated target json",
        ));
        Ok(())
    }
}

struct NormalizeAuditHook;

impl RequestBridgeHook for NormalizeAuditHook {
    fn transform_request(
        &self,
        ctx: &RequestBridgeContext,
        request: &mut ChatRequest,
        report: &mut siumai_core::bridge::BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        assert_eq!(ctx.phase, RequestBridgePhase::NormalizeSource);
        assert_eq!(ctx.source, Some(BridgeTarget::OpenAiResponses));
        assert_eq!(ctx.target, BridgeTarget::OpenAiResponses);
        assert_eq!(ctx.route_label.as_deref(), Some("tests.normalize.hook"));
        assert_eq!(ctx.path_label.as_deref(), Some("source-normalize"));
        request.common_params.max_tokens = Some(55);
        report.record_lossy_field(
            "normalize.custom",
            "custom normalize hook rewrote the normalized request",
        );
        Ok(())
    }
}

struct CompositeRequestCustomization;

impl BridgeCustomization for CompositeRequestCustomization {
    fn transform_request(
        &self,
        ctx: &RequestBridgeContext,
        request: &mut ChatRequest,
        report: &mut siumai_core::bridge::BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        assert_eq!(ctx.phase, RequestBridgePhase::SerializeTarget);
        assert_eq!(ctx.source, Some(BridgeTarget::AnthropicMessages));
        assert_eq!(ctx.target, BridgeTarget::OpenAiChatCompletions);
        assert_eq!(
            ctx.route_label.as_deref(),
            Some("tests.request.customization")
        );
        assert_eq!(ctx.path_label.as_deref(), Some("via-normalized"));

        request.common_params.max_tokens = Some(88);
        report.add_warning(BridgeWarning::new(
            BridgeWarningKind::Custom,
            "bundled customization rewrote normalized request",
        ));
        Ok(())
    }

    fn transform_request_json(
        &self,
        ctx: &RequestBridgeContext,
        body: &mut serde_json::Value,
        report: &mut siumai_core::bridge::BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        assert_eq!(ctx.target, BridgeTarget::OpenAiChatCompletions);
        body["metadata"] = json!({
            "customized": true,
            "target": ctx.target.as_str()
        });
        report.add_warning(BridgeWarning::new(
            BridgeWarningKind::Custom,
            "bundled customization rewrote target json",
        ));
        Ok(())
    }

    fn validate_request_json(
        &self,
        _ctx: &RequestBridgeContext,
        body: &serde_json::Value,
        report: &mut siumai_core::bridge::BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        assert_eq!(body["metadata"]["customized"], json!(true));
        report.add_warning(BridgeWarning::new(
            BridgeWarningKind::Custom,
            "bundled customization validated target json",
        ));
        Ok(())
    }

    fn remap_tool_name(&self, ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        assert_eq!(ctx.source, Some(BridgeTarget::AnthropicMessages));
        assert_eq!(ctx.target, BridgeTarget::OpenAiChatCompletions);
        Some(format!("bundle_{name}"))
    }
}

#[cfg(feature = "openai")]
#[test]
fn strict_openai_chat_bridge_rejects_reasoning_loss() {
    let mut request = ChatRequest::new(vec![
        ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("step by step"),
            ContentPart::text("final answer"),
        ])
        .build(),
    ]);
    request.common_params.model = "gpt-4o-mini".to_string();

    let bridged = bridge_chat_request_to_openai_chat_completions_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::Strict,
    )
    .expect("bridge");

    assert!(bridged.is_rejected());
    assert!(bridged.report.is_rejected());
    assert_eq!(bridged.report.lossy_fields.len(), 1);
}

#[cfg(feature = "openai")]
#[test]
fn best_effort_openai_chat_bridge_allows_reasoning_loss() {
    let mut request = ChatRequest::new(vec![
        ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("step by step"),
            ContentPart::text("final answer"),
        ])
        .build(),
    ]);
    request.common_params.model = "gpt-4o-mini".to_string();

    let bridged = bridge_chat_request_to_openai_chat_completions_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());
    assert_eq!(bridged.report.lossy_fields.len(), 1);
    assert!(bridged.value.is_some());
}

#[cfg(feature = "openai")]
#[test]
fn custom_request_loss_policy_can_reject_lossy_bridge_in_best_effort_mode() {
    let mut request = ChatRequest::new(vec![
        ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("step by step"),
            ContentPart::text("final answer"),
        ])
        .build(),
    ]);
    request.common_params.model = "gpt-4o-mini".to_string();

    let bridged = bridge_chat_request_to_openai_chat_completions_json_with_options(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeOptions::new(BridgeMode::BestEffort).with_loss_policy(Arc::new(RejectLossyPolicy)),
    )
    .expect("bridge");

    assert!(bridged.is_rejected());
    assert!(bridged.report.is_rejected());
    assert!(
        bridged.report.warnings.iter().any(|warning| warning
            .message
            .contains("bridge policy rejected request conversion")),
        "expected custom loss policy rejection"
    );
}

#[cfg(feature = "openai")]
#[test]
fn best_effort_openai_responses_bridge_returns_json_and_report() {
    let mut request = ChatRequest::new(vec![
        ChatMessage::assistant_with_content(vec![
            ContentPart::Reasoning {
                text: "hidden chain of thought".to_string(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    json!({ "itemId": "rs_1" }),
                )])),
            },
            ContentPart::text("visible answer"),
        ])
        .build(),
    ]);
    request.common_params.model = "gpt-4o-mini".to_string();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::GeminiGenerateContent),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());

    let value = bridged.value.expect("json body");
    let input = value
        .get("input")
        .and_then(|value| value.as_array())
        .expect("responses input array");
    assert!(
        input.iter().any(|item| {
            item.get("type")
                .and_then(|value| value.as_str())
                .is_some_and(|kind| kind == "item_reference")
                && item
                    .get("id")
                    .and_then(|value| value.as_str())
                    .is_some_and(|id| id == "rs_1")
        }),
        "expected reasoning item reference in responses input"
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn anthropic_direct_pair_bridge_flattens_system_and_forces_responses_store_policy() {
    let request = ChatRequest::builder()
        .message(ChatMessage::system("sys-1").build())
        .message(ChatMessage::developer("sys-2").build())
        .message(ChatMessage::user("hi").build())
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    assert_eq!(value["store"], serde_json::json!(false));

    let include = value
        .get("include")
        .and_then(|value| value.as_array())
        .expect("responses include array");
    assert!(
        include
            .iter()
            .any(|value| value.as_str() == Some("reasoning.encrypted_content")),
        "expected reasoning.encrypted_content include"
    );

    let input = value
        .get("input")
        .and_then(|value| value.as_array())
        .expect("responses input array");
    assert_eq!(input[0]["role"], serde_json::json!("system"));
    assert_eq!(input[0]["content"], serde_json::json!("sys-1\n\nsys-2"));
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn anthropic_direct_pair_bridge_translates_web_search_tool_and_required_choice() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("search rust").build())
        .tools(vec![
            crate::tools::anthropic::web_search_20250305().with_args(json!({
                "allowedDomains": ["example.com"],
                "maxUses": 2
            })),
        ])
        .tool_choice(siumai_core::types::ToolChoice::Required)
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    let tools = value
        .get("tools")
        .and_then(|value| value.as_array())
        .expect("responses tools array");

    assert_eq!(tools[0]["type"], serde_json::json!("web_search"));
    assert_eq!(
        tools[0]["filters"]["allowed_domains"],
        serde_json::json!(["example.com"])
    );
    assert_eq!(value["tool_choice"], serde_json::json!("required"));
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "tools[0].args.maxUses"),
        "expected dropped maxUses warning in bridge report"
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn anthropic_direct_pair_customization_can_rewrite_provider_tool_before_openai_translation() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("fetch docs").build())
        .tools(vec![Tool::ProviderDefined(
            ProviderDefinedTool::new("anthropic.web_fetch_20250910", "web_fetch").with_args(
                json!({
                    "allowedDomains": ["example.com"]
                }),
            ),
        )])
        .model("gpt-4.1-mini")
        .build();

    let customization = ProviderToolRewriteCustomization::new().with_rule(
        ProviderToolRewriteRule::new("anthropic.web_fetch_20250910", "openai.web_search")
            .with_args_mapper(Arc::new(|_ctx, tool, _report| {
                let allowed_domains = tool
                    .args
                    .get("allowedDomains")
                    .cloned()
                    .unwrap_or_else(|| json!([]));
                json!({
                    "filters": {
                        "allowedDomains": allowed_domains,
                    }
                })
            })),
    );

    let bridged = bridge_chat_request_to_openai_responses_json_with_options(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeOptions::new(BridgeMode::BestEffort).with_customization(Arc::new(customization)),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(
        bridged.report.warnings.iter().any(|warning| warning
            .message
            .contains("rewrote provider-defined tool `anthropic.web_fetch_20250910`")),
        "expected provider tool rewrite warning: {:?}",
        bridged.report.warnings
    );

    let value = bridged.value.expect("json body");
    let tools = value
        .get("tools")
        .and_then(|value| value.as_array())
        .expect("responses tools array");

    assert_eq!(tools[0]["type"], serde_json::json!("web_search"));
    assert_eq!(
        tools[0]["filters"]["allowed_domains"],
        serde_json::json!(["example.com"])
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn anthropic_direct_pair_bridge_replays_assistant_reasoning_without_openai_item_id() {
    let mut assistant = ChatMessage::assistant_with_content(vec![
        ContentPart::reasoning("hidden chain of thought"),
        ContentPart::text("final answer"),
    ])
    .build();
    assistant.metadata.custom.insert(
        "anthropic_redacted_thinking_data".to_string(),
        serde_json::json!("enc_payload"),
    );

    let request = ChatRequest::builder()
        .message(assistant)
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    let input = value
        .get("input")
        .and_then(|value| value.as_array())
        .expect("responses input array");

    let reasoning_item = input
        .iter()
        .find(|item| item.get("type").and_then(|value| value.as_str()) == Some("reasoning"))
        .expect("expected reasoning item");

    assert_eq!(
        reasoning_item["id"],
        serde_json::json!("anthropic_reasoning_0_0")
    );
    assert_eq!(
        reasoning_item["encrypted_content"],
        serde_json::json!("enc_payload")
    );
    assert_eq!(
        reasoning_item["summary"][0]["text"],
        serde_json::json!("hidden chain of thought")
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn anthropic_direct_pair_bridge_maps_tool_result_to_function_call_output() {
    let request = ChatRequest::builder()
        .message(ChatMessage::tool_result_text("call_1", "web_search", "done").build())
        .tools(vec![crate::tools::anthropic::web_search_20250305()])
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    let input = value
        .get("input")
        .and_then(|value| value.as_array())
        .expect("responses input array");

    let tool_output = input
        .iter()
        .find(|item| {
            item.get("type")
                .and_then(|value| value.as_str())
                .is_some_and(|kind| kind == "function_call_output")
        })
        .expect("expected function_call_output item");

    assert_eq!(tool_output["call_id"], serde_json::json!("call_1"));
    assert_eq!(tool_output["output"], serde_json::json!("done"));
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn openai_direct_pair_bridge_lifts_instructions_to_system() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .provider_option(
            "openai",
            json!({
                "responsesApi": {
                    "instructions": "follow system"
                }
            }),
        )
        .model("claude-3-5-sonnet-latest")
        .build();

    let bridged = bridge_chat_request_to_anthropic_messages_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    assert_eq!(value["system"], serde_json::json!("follow system"));
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn openai_direct_pair_bridge_maps_web_search_choice_and_parallel_policy() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("search rust").build())
        .tools(vec![crate::tools::openai::web_search().with_args(json!({
            "filters": {
                "allowedDomains": ["example.com"]
            }
        }))])
        .tool_choice(siumai_core::types::ToolChoice::tool("web_search"))
        .provider_option(
            "openai",
            json!({
                "parallelToolCalls": false
            }),
        )
        .model("claude-3-5-sonnet-latest")
        .build();

    let bridged = bridge_chat_request_to_anthropic_messages_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    let tools = value
        .get("tools")
        .and_then(|value| value.as_array())
        .expect("anthropic tools array");

    assert_eq!(tools[0]["type"], serde_json::json!("web_search_20250305"));
    assert_eq!(tools[0]["name"], serde_json::json!("web_search"));
    assert_eq!(
        tools[0]["allowed_domains"],
        serde_json::json!(["example.com"])
    );
    assert_eq!(
        value["tool_choice"],
        serde_json::json!({
            "type": "tool",
            "name": "web_search",
            "disable_parallel_tool_use": true
        })
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn openai_direct_pair_bridge_maps_mcp_tool_to_anthropic_mcp_servers() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("use docs mcp").build())
        .tools(vec![Tool::ProviderDefined(
            ProviderDefinedTool::new("openai.mcp", "MCP").with_args(json!({
                "serverLabel": "docs",
                "serverUrl": "https://example.com/mcp",
                "allowedTools": ["search_docs"],
                "headers": {
                    "Authorization": "Bearer secret_token"
                },
                "requireApproval": "always",
                "serverDescription": "Docs MCP server"
            })),
        )])
        .model("claude-3-5-sonnet-latest")
        .build();

    let bridged = bridge_chat_request_to_anthropic_messages_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    let servers = value
        .get("mcp_servers")
        .and_then(|value| value.as_array())
        .expect("anthropic mcp_servers array");

    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0]["name"], serde_json::json!("docs"));
    assert_eq!(
        servers[0]["url"],
        serde_json::json!("https://example.com/mcp")
    );
    assert_eq!(
        servers[0]["tool_configuration"]["allowed_tools"],
        serde_json::json!(["search_docs"])
    );
    assert_eq!(
        servers[0]["authorization_token"],
        serde_json::json!("secret_token")
    );
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "tools[0].args.requireApproval"),
        "expected dropped requireApproval warning"
    );
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "tools[0].args.serverDescription"),
        "expected dropped serverDescription warning"
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn openai_direct_pair_bridge_drops_specific_mcp_tool_choice() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("use docs mcp").build())
        .tools(vec![Tool::ProviderDefined(
            ProviderDefinedTool::new("openai.mcp", "MCP").with_args(json!({
                "serverLabel": "docs",
                "serverUrl": "https://example.com/mcp"
            })),
        )])
        .tool_choice(siumai_core::types::ToolChoice::tool("MCP"))
        .model("claude-3-5-sonnet-latest")
        .build();

    let bridged = bridge_chat_request_to_anthropic_messages_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    assert!(
        value.get("tool_choice").is_none(),
        "specific MCP tool choice should be dropped for anthropic mcp_servers: {value:?}"
    );
    assert!(
        bridged
            .report
            .lossy_fields
            .iter()
            .any(|field| field == "tool_choice"),
        "expected lossy tool_choice warning"
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn openai_direct_pair_customization_can_rewrite_provider_tool_before_anthropic_translation() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("search images").build())
        .tools(vec![Tool::ProviderDefined(ProviderDefinedTool::new(
            "openai.image_generation",
            "generateImage",
        ))])
        .model("claude-3-5-sonnet-latest")
        .build();

    let customization = ProviderToolRewriteCustomization::new().with_rule(
        ProviderToolRewriteRule::new("openai.image_generation", "anthropic.web_search_20250305")
            .with_args_mapper(Arc::new(|_ctx, _tool, _report| {
                json!({
                    "allowedDomains": ["example.com"]
                })
            })),
    );

    let bridged = bridge_chat_request_to_anthropic_messages_json_with_options(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeOptions::new(BridgeMode::BestEffort).with_customization(Arc::new(customization)),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(
        bridged.report.warnings.iter().any(|warning| warning
            .message
            .contains("rewrote provider-defined tool `openai.image_generation`")),
        "expected provider tool rewrite warning: {:?}",
        bridged.report.warnings
    );

    let value = bridged.value.expect("json body");
    let tools = value
        .get("tools")
        .and_then(|value| value.as_array())
        .expect("anthropic tools array");

    assert_eq!(tools[0]["type"], serde_json::json!("web_search_20250305"));
    assert_eq!(
        tools[0]["allowed_domains"],
        serde_json::json!(["example.com"])
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn openai_direct_pair_bridge_maps_reasoning_encrypted_content_to_redacted_thinking() {
    let request = ChatRequest::builder()
        .message(
            ChatMessage::assistant_with_content(vec![
                ContentPart::Reasoning {
                    text: "hidden chain of thought".to_string(),
                    provider_metadata: Some(HashMap::from([(
                        "openai".to_string(),
                        json!({
                            "itemId": "rs_1",
                            "reasoningEncryptedContent": "enc_payload"
                        }),
                    )])),
                },
                ContentPart::text("final answer"),
            ])
            .build(),
        )
        .model("claude-3-5-sonnet-latest")
        .build();

    let bridged = bridge_chat_request_to_anthropic_messages_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    let messages = value
        .get("messages")
        .and_then(|value| value.as_array())
        .expect("anthropic messages array");
    let content = messages[0]
        .get("content")
        .and_then(|value| value.as_array())
        .expect("assistant content");

    assert!(
        content.iter().any(|part| {
            part.get("type")
                .and_then(|value| value.as_str())
                .is_some_and(|kind| kind == "redacted_thinking")
                && part
                    .get("data")
                    .and_then(|value| value.as_str())
                    .is_some_and(|value| value == "enc_payload")
        }),
        "expected redacted thinking block"
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn openai_direct_pair_bridge_lifts_response_format_and_reasoning_effort() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .provider_option(
            "openai",
            json!({
                "reasoningEffort": "xhigh",
                "responsesApi": {
                    "responseFormat": {
                        "type": "json_schema",
                        "name": "result",
                        "strict": true,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" }
                            },
                            "required": ["value"],
                            "additionalProperties": false
                        }
                    }
                }
            }),
        )
        .model("claude-sonnet-4-5")
        .build();

    let bridged = bridge_chat_request_to_anthropic_messages_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    assert_eq!(
        value["output_format"]["type"],
        serde_json::json!("json_schema")
    );
    assert_eq!(value["output_config"]["effort"], serde_json::json!("high"));
    assert!(
        bridged
            .report
            .lossy_fields
            .iter()
            .any(|field| { field == "provider_options_map.openai.reasoning_effort" }),
        "expected lossy effort mapping warning"
    );
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_bridge_reports_cache_breakpoints_beyond_limit() {
    let mut request = ChatRequest::new(vec![
        ChatMessage::system("s1")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
        ChatMessage::system("s2")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
        ChatMessage::user("u1")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
        ChatMessage::assistant("a1")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
        ChatMessage::user("u2")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
    ]);
    request.common_params.model = "claude-3-5-sonnet-latest".to_string();

    let bridged = bridge_chat_request_to_anthropic_messages_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(bridged.report.is_lossy());
    assert_eq!(bridged.report.dropped_fields.len(), 1);
    assert_eq!(
        bridged.report.dropped_fields[0],
        "messages[4].metadata.cache_control"
    );
    assert!(bridged.value.is_some());
}

#[cfg(feature = "openai")]
#[test]
fn request_bridge_options_can_remap_tool_names_and_tool_choice() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .tools(vec![siumai_core::types::Tool::function(
            "weather",
            "Get weather",
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                }
            }),
        )])
        .tool_choice(siumai_core::types::ToolChoice::tool("weather"))
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_chat_completions_json_with_options(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.request.remap")
            .with_primitive_remapper(Arc::new(PrefixRemapper)),
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    assert_eq!(value["tools"][0]["function"]["name"], json!("gw_weather"));
    assert_eq!(
        value["tool_choice"]["function"]["name"],
        json!("gw_weather")
    );
}

#[cfg(feature = "openai")]
#[test]
fn request_bridge_hook_can_mutate_request_and_validate_target_json() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_chat_completions_json_with_options(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.request.hook")
            .with_request_hook(Arc::new(RequestAuditHook)),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_exact());
    assert!(
        bridged.report.warnings.iter().any(|warning| warning
            .message
            .contains("request hook validated target json")),
        "expected request hook validation warning"
    );

    let value = bridged.value.expect("json body");
    assert_eq!(value["max_tokens"], json!(77));
    assert_eq!(value["metadata"]["hooked"], json!(true));
}

#[cfg(feature = "openai")]
#[test]
fn bridge_customization_bundle_can_drive_request_json_validation_and_remap() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .tools(vec![siumai_core::types::Tool::function(
            "weather",
            "Get weather",
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                }
            }),
        )])
        .tool_choice(siumai_core::types::ToolChoice::tool("weather"))
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_chat_completions_json_with_options(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.request.customization")
            .with_customization(Arc::new(CompositeRequestCustomization)),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_exact());
    assert!(bridged.report.warnings.iter().any(|warning| {
        warning
            .message
            .contains("bundled customization validated target json")
    }));

    let value = bridged.value.expect("json body");
    assert_eq!(value["max_tokens"], json!(88));
    assert_eq!(value["metadata"]["customized"], json!(true));
    assert_eq!(
        value["metadata"]["target"],
        json!(BridgeTarget::OpenAiChatCompletions.as_str())
    );
    assert_eq!(
        value["tools"][0]["function"]["name"],
        json!("bundle_weather")
    );
    assert_eq!(
        value["tool_choice"]["function"]["name"],
        json!("bundle_weather")
    );
}

#[cfg(feature = "openai")]
#[test]
fn openai_chat_request_normalization_roundtrip_preserves_tools_and_response_format() {
    use siumai_core::types::ToolChoice;
    use siumai_core::types::chat::ResponseFormat;

    let schema = json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .message(ChatMessage::system("sys").build())
        .message(
            ChatMessage::user("hello")
                .with_content_parts(vec![ContentPart::image_url("https://example.com/a.png")])
                .build(),
        )
        .message(
            ChatMessage::assistant_with_content(vec![
                ContentPart::text("searching"),
                ContentPart::tool_call("call_1", "weather", json!({ "city": "Tokyo" }), None),
            ])
            .build(),
        )
        .message(
            ChatMessage::tool_result_json("call_1", "weather", json!({ "temperature": 18 }))
                .build(),
        )
        .tools(vec![siumai_core::types::Tool::function(
            "weather",
            "Get weather",
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        )])
        .tool_choice(ToolChoice::tool("weather"))
        .response_format(
            ResponseFormat::json_schema(schema)
                .with_name("result")
                .with_description("desc")
                .with_strict(false),
        )
        .model("gpt-4.1-mini")
        .temperature(0.2)
        .top_p(0.8)
        .max_tokens(128)
        .seed(7)
        .stream(true)
        .build();

    let value = bridge_chat_request_to_openai_chat_completions_json(
        &request,
        Some(BridgeTarget::OpenAiChatCompletions),
        BridgeMode::BestEffort,
    )
    .expect("bridge")
    .value
    .expect("json body");

    let normalized = bridge_openai_chat_completions_json_to_chat_request(&value).expect("parse");

    assert_eq!(normalized.common_params.model, "gpt-4.1-mini");
    assert_eq!(normalized.common_params.temperature, Some(0.2));
    assert_eq!(normalized.common_params.top_p, Some(0.8));
    assert_eq!(normalized.common_params.max_tokens, Some(128));
    assert_eq!(normalized.common_params.seed, Some(7));
    assert!(normalized.stream);
    assert_eq!(normalized.tool_choice, Some(ToolChoice::tool("weather")));
    assert_eq!(normalized.response_format, request.response_format);

    let tools = normalized.tools.expect("tools");
    assert_eq!(tools.len(), 1);
    let siumai_core::types::Tool::Function { function } = &tools[0] else {
        panic!("expected function tool");
    };
    assert_eq!(function.name, "weather");

    assert_eq!(normalized.messages.len(), 4);
    assert_eq!(normalized.messages[0].content_text(), Some("sys"));
    assert!(normalized.messages[1].content.as_multimodal().is_some());
    assert_eq!(normalized.messages[2].tool_calls().len(), 1);
    assert_eq!(normalized.messages[3].tool_results().len(), 1);
}

#[cfg(feature = "openai")]
#[test]
fn openai_responses_request_normalization_restores_instructions_and_options() {
    use siumai_core::types::ToolChoice;

    let value = json!({
        "model": "gpt-5-mini",
        "instructions": "follow system",
        "input": [
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "hi" }
                ]
            },
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [
                    { "type": "summary_text", "text": "step 1" }
                ],
                "encrypted_content": "enc_payload"
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "web_search",
                "arguments": "{\"q\":\"rust\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "{\"ok\":true}"
            },
            {
                "role": "assistant",
                "id": "msg_1",
                "content": [
                    { "type": "output_text", "text": "done" }
                ]
            }
        ],
        "tools": [
            {
                "type": "web_search",
                "filters": {
                    "allowed_domains": ["example.com"]
                }
            },
            {
                "type": "function",
                "name": "math",
                "description": "Math",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": { "type": "number" }
                    }
                },
                "strict": true
            }
        ],
        "tool_choice": { "type": "web_search" },
        "parallel_tool_calls": false,
        "store": false,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    },
                    "required": ["value"],
                    "additionalProperties": false
                }
            }
        },
        "reasoning": {
            "effort": "high"
        }
    });

    let normalized = bridge_openai_responses_json_to_chat_request(&value).expect("parse");

    assert_eq!(normalized.common_params.model, "gpt-5-mini");
    assert_eq!(normalized.tool_choice, Some(ToolChoice::tool("webSearch")));
    assert_eq!(
        normalized.messages[0].role,
        siumai_core::types::MessageRole::System
    );
    assert_eq!(normalized.messages[0].content_text(), Some("follow system"));

    let openai_options = normalized
        .provider_option("openai")
        .and_then(|value| value.as_object())
        .expect("openai options");
    assert_eq!(openai_options["parallelToolCalls"], json!(false));
    assert_eq!(openai_options["store"], json!(false));
    assert_eq!(openai_options["reasoningEffort"], json!("high"));

    let tools = normalized.tools.expect("tools");
    assert_eq!(tools.len(), 2);
    let siumai_core::types::Tool::ProviderDefined(provider_tool) = &tools[0] else {
        panic!("expected provider-defined tool");
    };
    assert_eq!(provider_tool.id, "openai.web_search");
    assert_eq!(provider_tool.name, "webSearch");

    let reasoning_message = &normalized.messages[2];
    let reasoning_parts = reasoning_message
        .content
        .as_multimodal()
        .expect("reasoning multimodal");
    let ContentPart::Reasoning {
        text,
        provider_metadata,
    } = &reasoning_parts[0]
    else {
        panic!("expected reasoning part");
    };
    assert_eq!(text, "step 1");
    assert_eq!(
        provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("openai"))
            .and_then(|meta| meta.get("itemId"))
            .and_then(|value| value.as_str()),
        Some("rs_1")
    );

    assert_eq!(normalized.messages[2].tool_calls().len(), 1);
    assert_eq!(normalized.messages[3].tool_results().len(), 1);
    assert_eq!(normalized.messages[4].metadata.id.as_deref(), Some("msg_1"));
    assert!(normalized.response_format.is_some());
}

#[cfg(feature = "openai")]
#[test]
fn openai_responses_request_normalization_with_options_applies_bridge_customization() {
    use siumai_core::types::ToolChoice;

    let value = json!({
        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "hi" }
                ]
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "math",
                "description": "Math",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": { "type": "number" }
                    }
                }
            }
        ],
        "tool_choice": {
            "type": "function",
            "name": "math"
        }
    });

    let bridged = bridge_openai_responses_json_to_chat_request_with_options(
        &value,
        BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.normalize.hook")
            .with_request_hook(Arc::new(NormalizeAuditHook))
            .with_primitive_remapper(Arc::new(PrefixRemapper)),
    )
    .expect("normalize");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());

    let normalized = bridged.value.expect("normalized request");
    assert_eq!(normalized.common_params.max_tokens, Some(55));
    assert_eq!(normalized.tool_choice, Some(ToolChoice::tool("gw_math")));

    let tools = normalized.tools.expect("tools");
    let siumai_core::types::Tool::Function { function } = &tools[0] else {
        panic!("expected function tool");
    };
    assert_eq!(function.name, "gw_math");
}

#[cfg(feature = "openai")]
#[test]
fn openai_responses_request_normalization_with_options_respects_loss_policy() {
    let value = json!({
        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "hi" }
                ]
            }
        ]
    });

    let bridged = bridge_openai_responses_json_to_chat_request_with_options(
        &value,
        BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.normalize.hook")
            .with_request_hook(Arc::new(NormalizeAuditHook))
            .with_loss_policy(Arc::new(RejectLossyPolicy)),
    )
    .expect("normalize");

    assert!(bridged.is_rejected());
    assert!(bridged.report.is_rejected());
    assert!(bridged.report.warnings.iter().any(|warning| {
        warning
            .message
            .contains("bridge policy rejected request normalization conversion")
    }));
}

#[cfg(feature = "openai")]
#[test]
fn openai_responses_request_normalization_restores_provider_tool_calls_and_outputs() {
    let value = json!({
        "model": "gpt-5-codex",
        "input": [
            {
                "type": "local_shell_call",
                "id": "lsh_1",
                "call_id": "call_shell_1",
                "action": {
                    "type": "exec",
                    "command": ["ls", "-a"]
                }
            },
            {
                "type": "local_shell_call_output",
                "call_id": "call_shell_1",
                "output": [
                    {
                        "stdout": "ok",
                        "stderr": "",
                        "outcome": { "type": "exit", "exitCode": 0 }
                    }
                ]
            },
            {
                "type": "apply_patch_call",
                "id": "apc_1",
                "call_id": "call_patch_1",
                "status": "completed",
                "operation": {
                    "type": "update_file",
                    "path": "src/lib.rs",
                    "diff": "*** Begin Patch\n*** End Patch\n"
                }
            },
            {
                "type": "apply_patch_call_output",
                "call_id": "call_patch_1",
                "status": "completed",
                "output": "applied"
            }
        ],
        "tools": [
            { "type": "local_shell" },
            { "type": "apply_patch" }
        ]
    });

    let normalized = bridge_openai_responses_json_to_chat_request(&value).expect("parse");

    let tools = normalized.tools.as_ref().expect("tools");
    let siumai_core::types::Tool::ProviderDefined(local_shell) = &tools[0] else {
        panic!("expected provider-defined local shell tool");
    };
    assert_eq!(local_shell.id, "openai.local_shell");
    assert_eq!(local_shell.name, "shell");

    let siumai_core::types::Tool::ProviderDefined(apply_patch) = &tools[1] else {
        panic!("expected provider-defined apply patch tool");
    };
    assert_eq!(apply_patch.id, "openai.apply_patch");
    assert_eq!(apply_patch.name, "apply_patch");

    assert_eq!(normalized.messages.len(), 4);

    let shell_call_parts = normalized.messages[0]
        .content
        .as_multimodal()
        .expect("shell call parts");
    let ContentPart::ToolCall {
        tool_call_id,
        tool_name,
        arguments,
        provider_metadata,
        ..
    } = &shell_call_parts[0]
    else {
        panic!("expected shell tool call");
    };
    assert_eq!(tool_call_id, "call_shell_1");
    assert_eq!(tool_name, "shell");
    assert_eq!(
        arguments,
        &json!({
            "action": {
                "type": "exec",
                "command": ["ls", "-a"]
            }
        })
    );
    assert_eq!(
        provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("openai"))
            .and_then(|meta| meta.get("itemId"))
            .and_then(|value| value.as_str()),
        Some("lsh_1")
    );

    let shell_output_parts = normalized.messages[1]
        .content
        .as_multimodal()
        .expect("shell output parts");
    let ContentPart::ToolResult {
        tool_call_id,
        tool_name,
        output,
        ..
    } = &shell_output_parts[0]
    else {
        panic!("expected shell tool result");
    };
    assert_eq!(tool_call_id, "call_shell_1");
    assert_eq!(tool_name, "shell");
    assert_eq!(
        output,
        &siumai_core::types::ToolResultOutput::json(json!({
            "output": [
                {
                    "stdout": "ok",
                    "stderr": "",
                    "outcome": { "type": "exit", "exitCode": 0 }
                }
            ]
        }))
    );

    let apply_patch_call_parts = normalized.messages[2]
        .content
        .as_multimodal()
        .expect("apply patch call parts");
    let ContentPart::ToolCall {
        tool_call_id,
        tool_name,
        arguments,
        provider_metadata,
        ..
    } = &apply_patch_call_parts[0]
    else {
        panic!("expected apply patch tool call");
    };
    assert_eq!(tool_call_id, "call_patch_1");
    assert_eq!(tool_name, "apply_patch");
    assert_eq!(
        arguments,
        &json!({
            "operation": {
                "type": "update_file",
                "path": "src/lib.rs",
                "diff": "*** Begin Patch\n*** End Patch\n"
            }
        })
    );
    assert_eq!(
        provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("openai"))
            .and_then(|meta| meta.get("itemId"))
            .and_then(|value| value.as_str()),
        Some("apc_1")
    );

    let apply_patch_output_parts = normalized.messages[3]
        .content
        .as_multimodal()
        .expect("apply patch output parts");
    let ContentPart::ToolResult {
        tool_call_id,
        tool_name,
        output,
        ..
    } = &apply_patch_output_parts[0]
    else {
        panic!("expected apply patch tool result");
    };
    assert_eq!(tool_call_id, "call_patch_1");
    assert_eq!(tool_name, "apply_patch");
    assert_eq!(
        output,
        &siumai_core::types::ToolResultOutput::json(json!({
            "status": "completed",
            "output": "applied"
        }))
    );
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_messages_request_normalization_restores_system_thinking_and_provider_options() {
    use siumai_core::types::MessageRole;
    use siumai_core::types::ToolChoice;

    let value = json!({
        "model": "claude-sonnet-4-5",
        "system": [
            {
                "type": "text",
                "text": "sys",
                "cache_control": { "type": "ephemeral" }
            },
            {
                "type": "text",
                "text": "Developer instructions: dev"
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "hi" },
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "Zm9v"
                        },
                        "title": "doc.pdf",
                        "context": "ctx",
                        "citations": { "enabled": true }
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "{\"done\":true}",
                        "is_error": false
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    { "type": "thinking", "thinking": "step", "signature": "sig" },
                    { "type": "tool_use", "id": "call_1", "name": "weather", "input": { "city": "Tokyo" } },
                    { "type": "text", "text": "ok" }
                ]
            }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 8,
        "max_tokens": 512,
        "stream": true,
        "tools": [
            {
                "name": "weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    }
                },
                "strict": true,
                "defer_loading": true
            },
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "allowed_domains": ["example.com"]
            }
        ],
        "tool_choice": {
            "type": "tool",
            "name": "weather",
            "disable_parallel_tool_use": true
        },
        "output_format": {
            "type": "json_schema",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                }
            }
        },
        "output_config": {
            "effort": "high"
        }
    });

    let normalized = bridge_anthropic_messages_json_to_chat_request(&value).expect("parse");

    assert_eq!(normalized.common_params.model, "claude-sonnet-4-5");
    assert_eq!(normalized.common_params.temperature, Some(0.2));
    assert_eq!(normalized.common_params.top_p, Some(0.9));
    assert_eq!(normalized.common_params.top_k, Some(8.0));
    assert_eq!(normalized.common_params.max_tokens, Some(512));
    assert!(normalized.stream);
    assert_eq!(normalized.tool_choice, Some(ToolChoice::tool("weather")));

    assert_eq!(normalized.messages[0].role, MessageRole::System);
    assert!(normalized.messages[0].metadata.cache_control.is_some());
    assert_eq!(normalized.messages[1].role, MessageRole::Developer);
    assert_eq!(normalized.messages[1].content_text(), Some("dev"));
    assert_eq!(normalized.messages[3].role, MessageRole::Tool);
    assert_eq!(normalized.messages[4].role, MessageRole::Assistant);
    assert_eq!(normalized.messages[4].tool_calls().len(), 1);

    let anthropic_options = normalized
        .provider_option("anthropic")
        .and_then(|value| value.as_object())
        .expect("anthropic options");
    assert_eq!(anthropic_options["disableParallelToolUse"], json!(true));
    assert_eq!(
        anthropic_options["structuredOutputMode"],
        json!("outputFormat")
    );
    assert_eq!(anthropic_options["effort"], json!("high"));

    let tools = normalized.tools.expect("tools");
    assert_eq!(tools.len(), 2);
    let siumai_core::types::Tool::Function { function } = &tools[0] else {
        panic!("expected function tool");
    };
    assert_eq!(function.name, "weather");
    assert_eq!(function.strict, Some(true));
    assert_eq!(
        function
            .provider_options_map
            .get("anthropic")
            .and_then(|value| value.get("deferLoading"))
            .and_then(|value| value.as_bool()),
        Some(true)
    );

    assert!(normalized.response_format.is_some());
}

#[cfg(feature = "google")]
#[test]
fn gemini_generate_content_request_normalization_roundtrip_preserves_core_projection() {
    use siumai_core::types::MessageRole;
    use siumai_core::types::ToolChoice;

    let value = json!({
        "model": "gemini-2.5-flash",
        "systemInstruction": {
            "parts": [
                { "text": "sys" }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    { "text": "hello" },
                    {
                        "fileData": {
                            "file_uri": "https://example.com/input.txt",
                            "mime_type": "text/plain"
                        }
                    }
                ]
            },
            {
                "role": "model",
                "parts": [
                    {
                        "text": "internal step",
                        "thought": true,
                        "thoughtSignature": "sig_1"
                    },
                    { "text": "Need tool" },
                    {
                        "functionCall": {
                            "name": "weather",
                            "args": { "city": "Tokyo" }
                        }
                    },
                    {
                        "executableCode": {
                            "language": "PYTHON",
                            "code": "print(1)"
                        }
                    }
                ]
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "weather",
                            "response": {
                                "name": "weather",
                                "content": { "ok": true }
                            }
                        }
                    },
                    {
                        "codeExecutionResult": {
                            "outcome": "OUTCOME_OK",
                            "output": "1"
                        }
                    }
                ]
            }
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": { "type": "string" }
                            }
                        }
                    }
                ]
            },
            { "googleSearch": {} },
            {
                "fileSearch": {
                    "fileSearchStoreNames": ["fileSearchStores/store-1"],
                    "topK": 3,
                    "metadataFilter": "scope = public"
                }
            },
            {
                "retrieval": {
                    "vertex_rag_store": {
                        "rag_resources": {
                            "rag_corpus": "projects/p/locations/l/ragCorpora/c"
                        },
                        "similarity_top_k": 4
                    }
                }
            }
        ],
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": ["weather"]
            },
            "retrievalConfig": {
                "latLng": {
                    "latitude": 35.0,
                    "longitude": 139.0
                }
            }
        },
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "topK": 8,
            "maxOutputTokens": 128,
            "seed": 7,
            "presencePenalty": 0.1,
            "frequencyPenalty": 0.2,
            "stopSequences": ["END"],
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                }
            },
            "thinkingConfig": {
                "thinkingBudget": 16
            },
            "responseModalities": ["TEXT"],
            "mediaResolution": "MEDIA_RESOLUTION_LOW",
            "responseLogprobs": true,
            "logprobs": 5
        },
        "cachedContent": "cachedContents/test-123",
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ],
        "labels": {
            "route": "bridge.test"
        }
    });

    let normalized = bridge_gemini_generate_content_json_to_chat_request(&value).expect("parse");

    assert_eq!(normalized.common_params.model, "gemini-2.5-flash");
    assert_eq!(normalized.common_params.temperature, Some(0.2));
    assert_eq!(normalized.common_params.top_p, Some(0.9));
    assert_eq!(normalized.common_params.top_k, Some(8.0));
    assert_eq!(normalized.common_params.max_tokens, Some(128));
    assert_eq!(normalized.common_params.seed, Some(7));
    assert_eq!(normalized.common_params.presence_penalty, Some(0.1));
    assert_eq!(normalized.common_params.frequency_penalty, Some(0.2));
    assert_eq!(
        normalized.common_params.stop_sequences,
        Some(vec!["END".to_string()])
    );
    assert_eq!(normalized.tool_choice, Some(ToolChoice::tool("weather")));
    assert!(normalized.response_format.is_some());

    assert_eq!(normalized.messages.len(), 4);
    assert_eq!(normalized.messages[0].role, MessageRole::System);
    assert_eq!(normalized.messages[0].content_text(), Some("sys"));
    assert_eq!(normalized.messages[1].role, MessageRole::User);
    assert_eq!(normalized.messages[2].role, MessageRole::Assistant);
    assert_eq!(normalized.messages[2].tool_calls().len(), 2);
    assert_eq!(normalized.messages[3].role, MessageRole::Tool);
    assert_eq!(normalized.messages[3].tool_results().len(), 2);

    let google_options = normalized
        .provider_option("google")
        .and_then(|value| value.as_object())
        .expect("google provider options");
    assert_eq!(
        google_options["cachedContent"],
        json!("cachedContents/test-123")
    );
    assert_eq!(google_options["structuredOutputs"], json!(false));
    assert_eq!(
        google_options["responseJsonSchema"]["type"],
        json!("object")
    );
    assert_eq!(
        google_options["thinkingConfig"]["thinkingBudget"],
        json!(16)
    );
    assert_eq!(
        google_options["retrievalConfig"]["latLng"]["latitude"],
        json!(35.0)
    );
    assert_eq!(google_options["labels"]["route"], json!("bridge.test"));

    let tools = normalized.tools.as_ref().expect("tools");
    assert_eq!(tools.len(), 4);

    let bridged = bridge_chat_request_to_gemini_generate_content_json(
        &normalized,
        Some(BridgeTarget::GeminiGenerateContent),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let bridged_value = bridged.value.expect("bridged json body");
    assert_eq!(bridged_value["model"], json!("gemini-2.5-flash"));
    assert_eq!(
        bridged_value["systemInstruction"]["parts"][0]["text"],
        json!("sys")
    );
    assert_eq!(
        bridged_value["contents"][1]["parts"][0]["thoughtSignature"],
        json!("sig_1")
    );
    assert_eq!(
        bridged_value["contents"][1]["parts"][2]["functionCall"]["name"],
        json!("weather")
    );
    assert_eq!(
        bridged_value["contents"][1]["parts"][3]["executableCode"]["code"],
        json!("print(1)")
    );
    assert_eq!(
        bridged_value["contents"][2]["parts"][0]["functionResponse"]["name"],
        json!("weather")
    );
    assert_eq!(
        bridged_value["contents"][2]["parts"][1]["codeExecutionResult"]["outcome"],
        json!("OUTCOME_OK")
    );
    assert_eq!(
        bridged_value["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"],
        json!(["weather"])
    );
    assert_eq!(
        bridged_value["toolConfig"]["retrievalConfig"]["latLng"]["longitude"],
        json!(139.0)
    );
    assert_eq!(
        bridged_value["generationConfig"]["responseJsonSchema"]["type"],
        json!("object")
    );
    assert_eq!(
        bridged_value["generationConfig"]["thinkingConfig"]["thinkingBudget"],
        json!(16)
    );
    assert_eq!(bridged_value["labels"]["route"], json!("bridge.test"));
}
