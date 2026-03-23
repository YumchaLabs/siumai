#![cfg(feature = "server")]

use std::sync::Arc;

use serde_json::json;
use siumai::experimental::bridge::{BridgeMode, BridgeTarget, ProviderToolRewriteCustomization};
use siumai::prelude::unified::{MessageRole, ResponseFormat, Tool, ToolChoice};
use siumai_extras::bridge::ClosureBridgeCustomization;
use siumai_extras::server::{
    GatewayBridgePolicy,
    axum::{
        NormalizeRequestOptions, SourceRequestFormat, normalize_request_json,
        normalize_request_json_with_options,
    },
};

#[cfg(feature = "openai")]
#[test]
fn request_normalize_smoke_restores_openai_responses_request() {
    let body = json!({
        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Explain ownership in one sentence."
                    }
                ]
            }
        ]
    });

    let bridged = normalize_request_json(&body, SourceRequestFormat::OpenAiResponses)
        .expect("normalize openai responses request");
    let (request, report) = bridged.into_result().expect("accepted");

    assert_eq!(request.common_params.model, "gpt-5-mini");
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].role, MessageRole::User);
    assert_eq!(
        request.messages[0].content_text(),
        Some("Explain ownership in one sentence.")
    );
    assert!(!report.is_rejected());
}

#[cfg(feature = "openai")]
#[test]
fn request_normalize_smoke_restores_openai_chat_request() {
    let body = json!({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are terse."
            },
            {
                "role": "user",
                "content": "Return JSON."
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": "{\"city\":\"Tokyo\"}"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"temperature\":18}"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" }
                        },
                        "required": ["city"]
                    }
                }
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "weather"
            }
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "ok": { "type": "boolean" }
                    },
                    "required": ["ok"]
                },
                "strict": false
            }
        }
    });

    let bridged = normalize_request_json(&body, SourceRequestFormat::OpenAiChatCompletions)
        .expect("normalize openai chat request");
    let (request, report) = bridged.into_result().expect("accepted");

    assert_eq!(request.common_params.model, "gpt-4o-mini");
    assert_eq!(request.messages.len(), 4);
    assert_eq!(request.messages[0].role, MessageRole::System);
    assert_eq!(request.messages[1].role, MessageRole::User);
    assert_eq!(request.messages[2].role, MessageRole::Assistant);
    assert_eq!(request.messages[3].role, MessageRole::Tool);
    assert_eq!(request.messages[2].tool_calls().len(), 1);
    assert_eq!(request.messages[3].tool_results().len(), 1);
    assert_eq!(request.tool_choice, Some(ToolChoice::tool("weather")));

    let tools = request.tools.as_ref().expect("function tools");
    let Tool::Function { function } = &tools[0] else {
        panic!("expected function tool");
    };
    assert_eq!(function.name, "weather");

    let format = request.response_format.expect("response format");
    let ResponseFormat::Json {
        name,
        strict,
        schema,
        ..
    } = format;
    assert_eq!(name.as_deref(), Some("response"));
    assert_eq!(strict, Some(false));
    assert_eq!(schema["properties"]["ok"]["type"], json!("boolean"));
    assert!(!report.is_rejected());
}

#[cfg(feature = "anthropic")]
#[test]
fn request_normalize_smoke_restores_anthropic_messages_request() {
    let body = json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 64000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello"
                    }
                ]
            }
        ],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 1,
                "user_location": {
                    "country": "US",
                    "type": "approximate"
                }
            }
        ],
        "output_format": {
            "type": "json_schema",
            "schema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                },
                "required": ["name"],
                "additionalProperties": false
            }
        }
    });

    let bridged = normalize_request_json(&body, SourceRequestFormat::AnthropicMessages)
        .expect("normalize anthropic request");
    let (request, report) = bridged.into_result().expect("accepted");

    assert_eq!(request.common_params.model, "claude-sonnet-4-5");
    assert_eq!(request.common_params.max_tokens, Some(64000));
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].role, MessageRole::User);
    assert_eq!(request.messages[0].content_text(), Some("Hello"));

    let tools = request.tools.as_ref().expect("tools");
    let provider_tool = tools
        .iter()
        .find_map(|tool| match tool {
            Tool::ProviderDefined(tool) => Some(tool),
            _ => None,
        })
        .expect("provider-defined tool");
    assert_eq!(provider_tool.id, "anthropic.web_search_20250305");
    assert_eq!(provider_tool.name, "web_search");
    assert_eq!(provider_tool.args["max_uses"], json!(1));

    let anthropic_options = request
        .provider_option("anthropic")
        .and_then(|value| value.as_object())
        .expect("anthropic options");
    assert_eq!(
        anthropic_options["structuredOutputMode"],
        json!("outputFormat")
    );

    let format = request.response_format.expect("response format");
    let ResponseFormat::Json { schema, .. } = format;
    assert_eq!(schema["properties"]["name"]["type"], json!("string"));
    assert!(!report.is_rejected());
}

#[cfg(feature = "google")]
#[test]
fn request_normalize_smoke_restores_gemini_generate_content_request() {
    let body = json!({
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
                    { "text": "hello" }
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
                    {
                        "functionCall": {
                            "name": "weather",
                            "args": { "city": "Tokyo" }
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
            }
        ],
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": ["weather"]
            }
        },
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 128,
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                }
            }
        },
        "cachedContent": "cachedContents/test-123",
        "labels": {
            "route": "smoke"
        }
    });

    let bridged = normalize_request_json(&body, SourceRequestFormat::GeminiGenerateContent)
        .expect("normalize gemini request");
    let (request, report) = bridged.into_result().expect("accepted");

    assert_eq!(request.common_params.model, "gemini-2.5-flash");
    assert_eq!(request.common_params.temperature, Some(0.2));
    assert_eq!(request.common_params.max_tokens, Some(128));
    assert_eq!(request.tool_choice, Some(ToolChoice::tool("weather")));

    assert_eq!(request.messages.len(), 4);
    assert_eq!(request.messages[0].role, MessageRole::System);
    assert_eq!(request.messages[0].content_text(), Some("sys"));
    assert_eq!(request.messages[1].role, MessageRole::User);
    assert_eq!(request.messages[2].role, MessageRole::Assistant);
    assert_eq!(request.messages[3].role, MessageRole::Tool);
    assert_eq!(request.messages[2].tool_calls().len(), 1);
    assert_eq!(request.messages[3].tool_results().len(), 1);

    let google_options = request
        .provider_option("google")
        .and_then(|value| value.as_object())
        .expect("google options");
    assert_eq!(
        google_options["cachedContent"],
        json!("cachedContents/test-123")
    );
    assert_eq!(google_options["labels"]["route"], json!("smoke"));
    assert_eq!(
        google_options["responseJsonSchema"]["type"],
        json!("object")
    );
    assert!(!report.is_rejected());
}

#[cfg(feature = "openai")]
#[test]
fn request_normalize_smoke_policy_override_can_reject_lossy_normalization() {
    let body = json!({
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

    let policy = GatewayBridgePolicy::new(BridgeMode::BestEffort)
        .with_route_label("tests.request-normalize.policy")
        .with_customization(Arc::new(
            ClosureBridgeCustomization::default().with_request(|ctx, request, report| {
                assert_eq!(ctx.source, Some(BridgeTarget::OpenAiResponses));
                assert_eq!(ctx.target, BridgeTarget::OpenAiResponses);
                assert_eq!(ctx.mode, BridgeMode::Strict);
                assert_eq!(
                    ctx.route_label.as_deref(),
                    Some("tests.request-normalize.policy")
                );
                assert_eq!(ctx.path_label.as_deref(), Some("source-normalize"));

                request.common_params.max_tokens = Some(55);
                report.record_lossy_field(
                    "normalize.custom",
                    "gateway policy marked normalization as lossy",
                );
                Ok(())
            }),
        ));

    let options = NormalizeRequestOptions::default()
        .with_policy(policy)
        .with_bridge_mode_override(BridgeMode::Strict);

    let bridged =
        normalize_request_json_with_options(&body, SourceRequestFormat::OpenAiResponses, &options)
            .expect("normalize request");

    assert!(bridged.is_rejected());
    assert!(bridged.report.is_rejected());
    assert!(bridged.report.warnings.iter().any(|warning| {
        warning
            .message
            .contains("bridge policy rejected request normalization conversion")
    }));
}

#[cfg(feature = "anthropic")]
#[test]
fn request_normalize_smoke_applies_provider_tool_rewrite_customization() {
    let body = json!({
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": "fetch docs"
            }
        ],
        "tools": [
            {
                "type": "web_fetch_20250910",
                "name": "web_fetch"
            }
        ]
    });

    let options = NormalizeRequestOptions::default().with_bridge_customization(Arc::new(
        ProviderToolRewriteCustomization::new()
            .map_provider_tool_id("anthropic.web_fetch_20250910", "openai.web_search"),
    ));

    let bridged = normalize_request_json_with_options(
        &body,
        SourceRequestFormat::AnthropicMessages,
        &options,
    )
    .expect("normalize anthropic request");
    let (request, report) = bridged.into_result().expect("accepted");

    let tools = request.tools.expect("rewritten tools");
    let Tool::ProviderDefined(provider_tool) = &tools[0] else {
        panic!("expected provider-defined tool");
    };
    assert_eq!(provider_tool.id, "openai.web_search");
    assert!(report.warnings.iter().any(|warning| {
        warning
            .message
            .contains("rewrote provider-defined tool `anthropic.web_fetch_20250910`")
    }));
}
