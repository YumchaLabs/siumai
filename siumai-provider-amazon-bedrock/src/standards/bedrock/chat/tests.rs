
use super::*;
use crate::provider_metadata::bedrock::BedrockChatResponseExt;
use crate::provider_options::{
    BedrockChatOptions, BedrockReasoningConfig, BedrockReasoningEffort, BedrockReasoningType,
    BedrockServiceTier,
};
use crate::providers::bedrock::ext::BedrockChatRequestExt;

fn test_transformers(uses_json_response_tool: bool) -> ChatTransformers {
    BedrockChatStandard::new().create_transformers(
        "bedrock",
        uses_json_response_tool,
        Some("anthropic.claude-3-sonnet".to_string()),
        vec![],
        false,
    )
}

fn test_transformers_with_model(
    model: &str,
    uses_json_response_tool: bool,
    warnings: Vec<Warning>,
) -> ChatTransformers {
    BedrockChatStandard::new().create_transformers(
        "bedrock",
        uses_json_response_tool,
        Some(model.to_string()),
        warnings,
        false,
    )
}

fn test_converter_with_options(
    uses_json_response_tool: bool,
    include_raw_chunks: bool,
    warnings: Vec<Warning>,
) -> BedrockEventConverter {
    BedrockEventConverter::new(
        "bedrock",
        uses_json_response_tool,
        Some("anthropic.claude-3-sonnet".to_string()),
        warnings,
        include_raw_chunks,
    )
}

fn test_converter(include_raw_chunks: bool, warnings: Vec<Warning>) -> BedrockEventConverter {
    test_converter_with_options(false, include_raw_chunks, warnings)
}

fn collected_parts(events: &[Result<ChatStreamEvent, LlmError>]) -> Vec<ChatStreamPart> {
    events
        .iter()
        .filter_map(|event| match event {
            Ok(ChatStreamEvent::Part { part }) => Some(part.clone()),
            _ => None,
        })
        .collect()
}

fn production_source_between(start_marker: &str, end_marker: &str) -> &'static str {
    let source = include_str!("../chat.rs");
    let (_, after_start) = source
        .split_once(start_marker)
        .expect("source start marker should exist");
    let (section, _) = after_start
        .split_once(end_marker)
        .expect("source end marker should exist");
    section
}

#[test]
fn request_conversion_source_does_not_read_legacy_provider_metadata_fields() {
    let request_source = production_source_between(
        "impl BedrockChatRequestTransformer",
        "impl RequestTransformer",
    );

    assert!(
        !request_source.contains("provider_metadata"),
        "Bedrock request transformer must not read legacy ContentPart::provider_metadata"
    );
    assert!(
        !request_source.contains("providerMetadata"),
        "Bedrock request transformer must not read legacy providerMetadata JSON fields"
    );
}

#[test]
fn response_and_stream_source_do_not_emit_request_provider_options() {
    let response_stream_source =
        production_source_between("struct BedrockChatResponseTransformer", "#[cfg(test)]");

    assert!(
        !response_stream_source.contains("providerOptions"),
        "Bedrock response/stream code must not emit request-side providerOptions"
    );

    let unexpected_provider_options_writes = response_stream_source
        .lines()
        .filter(|line| line.contains("provider_options"))
        .filter(|line| {
            !line.contains("provider_options: crate::types::ProviderOptionsMap::default()")
        })
        .collect::<Vec<_>>();

    assert!(
        unexpected_provider_options_writes.is_empty(),
        "Bedrock response/stream code may only initialize ContentPart::provider_options with the default map; unexpected lines: {unexpected_provider_options_writes:#?}"
    );
}

#[test]
fn request_injects_reserved_json_tool_for_response_format() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "value": { "type": "string" } },
        "required": ["value"],
        "additionalProperties": false
    });

    let req = ChatRequest::builder()
        .model("anthropic.claude-3-sonnet")
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build();

    let uses_json_tool = matches!(
        req.response_format.as_ref(),
        Some(ResponseFormat::Json { .. })
    );
    let tx = test_transformers(uses_json_tool);
    let body = tx.request.transform_chat(&req).expect("transform");

    let tool_cfg = body.get("toolConfig").expect("toolConfig should exist");
    let tools = tool_cfg
        .get("tools")
        .and_then(|v| v.as_array())
        .expect("tools should be an array");
    assert_eq!(tools.len(), 1);

    let tool_spec = tools[0]
        .get("toolSpec")
        .and_then(|v| v.as_object())
        .expect("toolSpec should exist");
    assert_eq!(tool_spec.get("name"), Some(&serde_json::json!("json")));
    assert_eq!(
        tool_spec
            .get("inputSchema")
            .and_then(|v| v.get("json"))
            .cloned(),
        Some(schema)
    );

    assert_eq!(
        tool_cfg
            .get("toolChoice")
            .and_then(|v| v.get("any"))
            .cloned(),
        Some(serde_json::json!({}))
    );
}

#[test]
fn request_maps_reasoning_config_service_tier_and_passthrough_fields() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-5-sonnet-20240620-v1:0")
        .max_tokens(100)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_bedrock_chat_options(
            BedrockChatOptions::new()
                .with_reasoning_config(
                    BedrockReasoningConfig::new()
                        .with_type(BedrockReasoningType::Enabled)
                        .with_budget_tokens(2000),
                )
                .with_anthropic_beta(["context-1m-2025-08-07"])
                .with_service_tier(BedrockServiceTier::Priority)
                .with_param("guardrailConfig", serde_json::json!({ "id": "gr-1" })),
        );

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");

    assert_eq!(
        body.get("additionalModelRequestFields"),
        Some(&serde_json::json!({
            "thinking": {
                "type": "enabled",
                "budget_tokens": 2000
            },
            "anthropic_beta": ["context-1m-2025-08-07"]
        }))
    );
    assert_eq!(
        body["inferenceConfig"]["maxTokens"],
        serde_json::json!(2100)
    );
    assert_eq!(
        body.get("serviceTier"),
        Some(&serde_json::json!({ "type": "priority" }))
    );
    assert_eq!(
        body.get("guardrailConfig"),
        Some(&serde_json::json!({ "id": "gr-1" }))
    );
    assert_eq!(
        body.get("additionalModelResponseFieldPaths"),
        Some(&serde_json::json!(["/delta/stop_sequence"]))
    );
    assert!(body.get("reasoningConfig").is_none());
}

#[test]
fn request_merges_user_additional_model_request_fields_with_derived_thinking() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-5-sonnet-20240620-v1:0")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_bedrock_chat_options(
            BedrockChatOptions::new()
                .with_additional_model_request_fields(
                    serde_json::json!({ "foo": "bar", "custom": 42 }),
                )
                .with_reasoning_config(
                    BedrockReasoningConfig::new()
                        .with_type(BedrockReasoningType::Enabled)
                        .with_budget_tokens(1234),
                ),
        );

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");

    assert_eq!(
        body.get("additionalModelRequestFields"),
        Some(&serde_json::json!({
            "foo": "bar",
            "custom": 42,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1234
            }
        }))
    );
}

#[test]
fn request_maps_max_reasoning_effort_for_anthropic_openai_and_other_models() {
    let anthropic_req = ChatRequest::builder()
        .model("anthropic.claude-3-5-sonnet-20240620-v1:0")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_bedrock_chat_options(
            BedrockChatOptions::new().with_reasoning_config(
                BedrockReasoningConfig::new()
                    .with_type(BedrockReasoningType::Enabled)
                    .with_max_reasoning_effort(BedrockReasoningEffort::Medium),
            ),
        );
    let anthropic_body = test_transformers(false)
        .request
        .transform_chat(&anthropic_req)
        .expect("transform anthropic");
    assert_eq!(
        anthropic_body["additionalModelRequestFields"]["output_config"],
        serde_json::json!({ "effort": "medium" })
    );
    assert!(
        anthropic_body["additionalModelRequestFields"]
            .get("reasoningConfig")
            .is_none()
    );

    let openai_req = ChatRequest::builder()
        .model("openai.gpt-oss-120b-1:0")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_bedrock_chat_options(BedrockChatOptions::new().with_reasoning_config(
            BedrockReasoningConfig::new().with_max_reasoning_effort(BedrockReasoningEffort::Medium),
        ));
    let openai_body = test_transformers(false)
        .request
        .transform_chat(&openai_req)
        .expect("transform openai");
    assert_eq!(
        openai_body["additionalModelRequestFields"]["reasoning_effort"],
        serde_json::json!("medium")
    );
    assert!(
        openai_body["additionalModelRequestFields"]
            .get("reasoningConfig")
            .is_none()
    );

    let nova_req = ChatRequest::builder()
        .model("us.amazon.nova-2-lite-v1:0")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_bedrock_chat_options(
            BedrockChatOptions::new().with_reasoning_config(
                BedrockReasoningConfig::new()
                    .with_type(BedrockReasoningType::Enabled)
                    .with_budget_tokens(2048)
                    .with_max_reasoning_effort(BedrockReasoningEffort::Medium),
            ),
        );
    let nova_body = test_transformers(false)
        .request
        .transform_chat(&nova_req)
        .expect("transform nova");
    assert_eq!(
        nova_body["additionalModelRequestFields"]["reasoningConfig"],
        serde_json::json!({
            "type": "enabled",
            "budgetTokens": 2048,
            "maxReasoningEffort": "medium"
        })
    );
    assert!(
        nova_body["additionalModelRequestFields"]
            .get("thinking")
            .is_none()
    );
}

#[test]
fn request_uses_native_structured_output_for_supported_anthropic_models() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
    });
    let req = ChatRequest::builder()
        .model("anthropic.claude-sonnet-4-6-v1")
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build();

    let plan = BedrockChatRequestTransformer::build_request_plan(&req).expect("plan");
    assert!(!plan.uses_json_response_tool);

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");
    assert!(body.get("toolConfig").is_none());
    assert_eq!(
        body["additionalModelRequestFields"]["output_config"],
        serde_json::json!({
            "format": {
                "type": "json_schema",
                "schema": schema
            }
        })
    );
}

#[test]
fn request_uses_native_structured_output_when_thinking_enabled_on_older_anthropic_models() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "recipe": { "type": "string" } },
        "required": ["recipe"]
    });
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-5-sonnet-20240620-v1:0")
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_bedrock_chat_options(
            BedrockChatOptions::new().with_reasoning_config(
                BedrockReasoningConfig::new()
                    .with_type(BedrockReasoningType::Enabled)
                    .with_budget_tokens(2000)
                    .with_max_reasoning_effort(BedrockReasoningEffort::Medium),
            ),
        );

    let plan = BedrockChatRequestTransformer::build_request_plan(&req).expect("plan");
    assert!(!plan.uses_json_response_tool);

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");
    assert!(body.get("toolConfig").is_none());
    assert_eq!(
        body["additionalModelRequestFields"]["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 2000
        })
    );
    assert_eq!(
        body["additionalModelRequestFields"]["output_config"],
        serde_json::json!({
            "effort": "medium",
            "format": {
                "type": "json_schema",
                "schema": schema
            }
        })
    );
}

#[test]
fn request_strips_sampling_knobs_when_anthropic_thinking_is_enabled() {
    let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
        .with_common_params(crate::types::CommonParams {
            model: "anthropic.claude-3-5-sonnet-20240620-v1:0".to_string(),
            temperature: Some(0.7),
            max_tokens: None,
            max_completion_tokens: None,
            top_p: Some(0.9),
            top_k: Some(5.0),
            stop_sequences: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
        })
        .with_bedrock_chat_options(
            BedrockChatOptions::new().with_reasoning_config(
                BedrockReasoningConfig::new()
                    .with_type(BedrockReasoningType::Enabled)
                    .with_budget_tokens(1024),
            ),
        );

    let plan = BedrockChatRequestTransformer::build_request_plan(&req).expect("plan");

    assert!(plan.inference_config.get("temperature").is_none());
    assert!(plan.inference_config.get("topP").is_none());
    assert!(plan.inference_config.get("topK").is_none());
    assert!(plan.warnings.contains(&Warning::unsupported(
        "temperature",
        Some("temperature is not supported when thinking is enabled"),
    )));
    assert!(plan.warnings.contains(&Warning::unsupported(
        "topP",
        Some("topP is not supported when thinking is enabled"),
    )));
    assert!(plan.warnings.contains(&Warning::unsupported(
        "topK",
        Some("topK is not supported when thinking is enabled"),
    )));
}

#[test]
fn request_rejects_non_object_additional_model_request_fields() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-5-sonnet-20240620-v1:0")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_bedrock_chat_options(
            BedrockChatOptions::new().with_additional_model_request_fields(serde_json::json!(true)),
        );

    let err = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect_err("expected invalid additionalModelRequestFields");

    assert!(matches!(
        err,
        LlmError::InvalidParameter(message)
            if message.contains("additionalModelRequestFields must be a JSON object")
    ));
}

#[test]
fn request_adds_message_level_cache_points_to_system_user_and_assistant_blocks() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-sonnet")
        .messages(vec![
            ChatMessage::system("system")
                .with_provider_option(
                    "bedrock",
                    serde_json::json!({
                        "cachePoint": { "type": "default", "ttl": "5m" }
                    }),
                )
                .build(),
            ChatMessage::user("user")
                .with_provider_option(
                    "bedrock",
                    serde_json::json!({
                        "cachePoint": { "type": "default" }
                    }),
                )
                .build(),
            ChatMessage::assistant("assistant")
                .with_provider_option(
                    "bedrock",
                    serde_json::json!({
                        "cache_point": { "type": "default", "ttl": "1h" }
                    }),
                )
                .build(),
        ])
        .build();

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");

    assert_eq!(
        body["system"],
        serde_json::json!([
            { "text": "system" },
            { "cachePoint": { "type": "default", "ttl": "5m" } }
        ])
    );
    assert_eq!(
        body["messages"],
        serde_json::json!([
            {
                "role": "user",
                "content": [
                    { "text": "user" },
                    { "cachePoint": { "type": "default" } }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    { "text": "assistant" },
                    { "cachePoint": { "type": "default", "ttl": "1h" } }
                ]
            }
        ])
    );
}

#[test]
fn request_converts_user_file_parts_to_documents_with_citations_and_strips_filename() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-sonnet")
        .messages(vec![
            ChatMessage::user("Hello")
                .with_content_parts(vec![
                    ContentPart::file_base64(
                        "AAECAw==",
                        "application/pdf",
                        Some("report.final.pdf".to_string()),
                    )
                    .with_provider_option(
                        "bedrock",
                        serde_json::json!({
                            "citations": { "enabled": true }
                        }),
                    ),
                ])
                .build(),
        ])
        .build();

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");

    assert_eq!(
        body["messages"][0],
        serde_json::json!({
            "role": "user",
            "content": [
                { "text": "Hello" },
                {
                    "document": {
                        "format": "pdf",
                        "name": "report",
                        "source": { "bytes": "AAECAw==" },
                        "citations": { "enabled": true }
                    }
                }
            ]
        })
    );
}

#[test]
fn request_converts_user_image_like_file_parts_to_bedrock_images() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-sonnet")
        .messages(vec![
            ChatMessage::user("look")
                .with_content_parts(vec![ContentPart::file_base64(
                    "AAECAw==",
                    "image/png",
                    Some("pixel.png".to_string()),
                )])
                .build(),
        ])
        .build();

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");

    assert_eq!(
        body["messages"][0]["content"][1],
        serde_json::json!({
            "image": {
                "format": "png",
                "source": { "bytes": "AAECAw==" }
            }
        })
    );
}

#[test]
fn request_converts_tool_result_content_image_data() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-sonnet")
        .messages(vec![ChatMessage {
            role: crate::types::MessageRole::Tool,
            content: MessageContent::MultiModal(vec![ContentPart::tool_result_content(
                "call-123",
                "image-generator",
                vec![
                    ToolResultContentPart::text("Generated image"),
                    ToolResultContentPart::image_data("base64data", "image/jpeg"),
                ],
            )]),
            provider_options: ProviderOptionsMap::default(),
            metadata: crate::types::MessageMetadata::default(),
        }])
        .build();

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");

    assert_eq!(
        body["messages"][0],
        serde_json::json!({
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "call-123",
                        "content": [
                            { "text": "Generated image" },
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": { "bytes": "base64data" }
                                }
                            }
                        ]
                    }
                }
            ]
        })
    );
}

#[test]
fn request_normalizes_mistral_tool_ids_for_tool_calls_and_results() {
    let original_id = "tooluse_bpe71yCfRu2b5i-nKGDr5g";
    let req = ChatRequest::builder()
        .model("mistral.mistral-7b-instruct-v0:2")
        .messages(vec![
            ChatMessage::assistant_with_content(vec![ContentPart::tool_call(
                original_id,
                "calculator",
                serde_json::json!({ "value": 42 }),
                None,
            )])
            .build(),
            ChatMessage::tool_result_text(original_id, "calculator", "ok").build(),
        ])
        .build();

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");

    assert_eq!(
        body["messages"],
        serde_json::json!([
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "toolusebp",
                            "name": "calculator",
                            "input": { "value": 42 }
                        }
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "toolusebp",
                            "content": [{ "text": "ok" }]
                        }
                    }
                ]
            }
        ])
    );
}

#[test]
fn request_preserves_signed_reasoning_and_trims_unsigned_last_reasoning() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-sonnet")
        .messages(vec![
            ChatMessage::user("Explain").build(),
            ChatMessage::assistant_with_content(vec![
                ContentPart::reasoning("signed reasoning   ")
                    .with_provider_option("bedrock", serde_json::json!({ "signature": "sig-1" })),
                ContentPart::text(""),
                ContentPart::reasoning("unsigned reasoning   "),
            ])
            .build(),
        ])
        .build();

    let body = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect("transform");

    assert_eq!(
        body["messages"][1],
        serde_json::json!({
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "signed reasoning   ",
                            "signature": "sig-1"
                        }
                    }
                },
                { "text": "" },
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "unsigned reasoning"
                        }
                    }
                }
            ]
        })
    );
}

#[test]
fn request_rejects_unsupported_user_file_mime_types() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-sonnet")
        .messages(vec![
            ChatMessage::user("check")
                .with_content_parts(vec![ContentPart::file_base64(
                    "base64data",
                    "application/rtf",
                    None,
                )])
                .build(),
        ])
        .build();

    let err = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect_err("expected unsupported mime type");

    assert!(matches!(
        err,
        LlmError::UnsupportedOperation(message)
            if message.contains("Unsupported file mime type: application/rtf")
    ));
}

#[test]
fn request_rejects_user_file_url_sources() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-sonnet")
        .messages(vec![
            ChatMessage::user("check")
                .with_content_parts(vec![ContentPart::file_url(
                    "https://example.com/report.pdf",
                    "application/pdf",
                )])
                .build(),
        ])
        .build();

    let err = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect_err("expected unsupported file url source");

    assert!(matches!(
        err,
        LlmError::UnsupportedOperation(message)
            if message.contains("file parts with URL sources")
    ));
}

#[test]
fn request_rejects_user_file_provider_references() {
    let req = ChatRequest::builder()
        .model("anthropic.claude-3-sonnet")
        .messages(vec![
            ChatMessage::user("check")
                .with_file_provider_reference(
                    crate::types::ProviderReference::single("openai", "file-123"),
                    "application/pdf",
                    Some("report.pdf".to_string()),
                )
                .build(),
        ])
        .build();

    let err = test_transformers(false)
        .request
        .transform_chat(&req)
        .expect_err("expected unsupported provider reference");

    assert!(matches!(
        err,
        LlmError::UnsupportedOperation(message)
            if message.contains("file parts with provider references")
    ));
}

#[test]
fn json_response_from_reserved_tool_is_emitted_as_text_and_finish_reason_stop() {
    let tx = test_transformers(true);

    let raw = serde_json::json!({
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "call_1",
                            "name": "json",
                            "input": { "value": "ok" }
                        }
                    }
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": { "inputTokens": 1, "outputTokens": 2, "totalTokens": 3 }
    });

    let resp = tx
        .response
        .transform_chat_response(&raw)
        .expect("transform");

    assert_eq!(resp.text().as_deref(), Some(r#"{"value":"ok"}"#));
    assert_eq!(resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(resp.raw_finish_reason.as_deref(), Some("tool_use"));
    assert_eq!(
        resp.bedrock_metadata()
            .and_then(|meta| meta.is_json_response_from_tool),
        Some(true)
    );
}

#[test]
fn response_preserves_reasoning_metadata_model_warnings_and_provider_metadata() {
    let warning = Warning::unsupported("seed", None::<String>);
    let tx =
        test_transformers_with_model("anthropic.claude-3-sonnet", false, vec![warning.clone()]);

    let raw = serde_json::json!({
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "thinking",
                                "signature": "sig-1"
                            }
                        }
                    },
                    { "text": "" },
                    {
                        "reasoningContent": {
                            "redactedReasoning": {
                                "data": "secret"
                            }
                        }
                    },
                    { "text": "answer" }
                ]
            }
        },
        "usage": {
            "inputTokens": 4,
            "outputTokens": 34,
            "totalTokens": 38,
            "cacheWriteInputTokens": 3,
            "cacheDetails": [{ "inputTokens": 100, "ttl": "T5M" }]
        },
        "trace": { "request": "trace-1" },
        "performanceConfig": { "latency": "optimized" },
        "serviceTier": { "type": "on-demand" },
        "stopReason": "end_turn"
    });

    let resp = tx
        .response
        .transform_chat_response(&raw)
        .expect("transform");

    assert_eq!(resp.model.as_deref(), Some("anthropic.claude-3-sonnet"));
    assert_eq!(resp.warnings, Some(vec![warning]));
    assert_eq!(resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        resp.usage
            .as_ref()
            .map(|usage| usage.normalized_input_tokens().total),
        Some(Some(7))
    );

    let MessageContent::MultiModal(parts) = &resp.content else {
        panic!("expected multimodal content");
    };

    assert!(matches!(
        parts.first(),
        Some(ContentPart::Reasoning { text, provider_metadata: Some(provider_metadata), .. })
            if text == "thinking"
                && provider_metadata.get("bedrock")
                    == Some(&serde_json::json!({"signature":"sig-1"}))
    ));
    assert!(matches!(
        parts.get(1),
        Some(ContentPart::Text { text, .. }) if text.is_empty()
    ));
    assert!(matches!(
        parts.get(2),
        Some(ContentPart::Reasoning { text, provider_metadata: Some(provider_metadata), .. })
            if text.is_empty()
                && provider_metadata.get("bedrock")
                    == Some(&serde_json::json!({"redactedData":"secret"}))
    ));
    assert!(matches!(
        parts.get(3),
        Some(ContentPart::Text { text, .. }) if text == "answer"
    ));

    let bedrock_metadata = resp.bedrock_metadata().expect("bedrock metadata");
    assert_eq!(
        bedrock_metadata.extra.get("trace"),
        Some(&serde_json::json!({ "request": "trace-1" }))
    );
    assert_eq!(
        bedrock_metadata.extra.get("performanceConfig"),
        Some(&serde_json::json!({ "latency": "optimized" }))
    );
    assert_eq!(
        bedrock_metadata.extra.get("serviceTier"),
        Some(&serde_json::json!({ "type": "on-demand" }))
    );
    assert_eq!(
        bedrock_metadata.extra.get("usage"),
        Some(&serde_json::json!({
            "cacheWriteInputTokens": 3,
            "cacheDetails": [{ "inputTokens": 100, "ttl": "T5M" }]
        }))
    );
}

#[test]
fn response_preserves_raw_usage_payload_with_unknown_fields() {
    let tx = test_transformers(false);

    let raw = serde_json::json!({
        "output": {
            "message": {
                "content": [
                    { "text": "answer" }
                ]
            }
        },
        "usage": {
            "inputTokens": 4,
            "outputTokens": 34,
            "totalTokens": 38,
            "cacheReadInputTokens": 2,
            "cacheWriteInputTokens": 3,
            "futureUsageField": { "x": 1 }
        },
        "stopReason": "end_turn"
    });

    let resp = tx
        .response
        .transform_chat_response(&raw)
        .expect("transform");

    let usage = resp.usage.as_ref().expect("usage");
    assert_eq!(usage.normalized_input_tokens().total, Some(9));
    assert_eq!(usage.normalized_input_tokens().no_cache, Some(4));
    assert_eq!(usage.normalized_input_tokens().cache_read, Some(2));
    assert_eq!(usage.normalized_input_tokens().cache_write, Some(3));
    assert_eq!(usage.normalized_output_tokens().total, Some(34));
    assert_eq!(usage.raw_usage_value(), raw.get("usage").cloned());
}

#[test]
fn response_normalizes_mistral_tool_call_ids() {
    let tx = test_transformers_with_model("mistral.mistral-7b-instruct-v0:2", false, vec![]);

    let raw = serde_json::json!({
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_bpe71yCfRu2b5i-nKGDr5g",
                            "name": "weather",
                            "input": { "location": "SF" }
                        }
                    }
                ]
            }
        },
        "stopReason": "tool_use"
    });

    let resp = tx
        .response
        .transform_chat_response(&raw)
        .expect("transform");

    assert_eq!(
        resp.model.as_deref(),
        Some("mistral.mistral-7b-instruct-v0:2")
    );
    let Some(ContentPart::ToolCall {
        tool_call_id,
        tool_name,
        ..
    }) = resp.tool_calls().first().copied()
    else {
        panic!("expected tool call");
    };
    assert_eq!(tool_call_id, "toolusebp");
    assert_eq!(tool_name, "weather");
}

#[tokio::test]
async fn bedrock_raw_chunks_follow_stream_start_and_response_metadata() {
    let converter = test_converter(true, vec![]);

    let events = converter
        .convert_json(r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"Hello"}}}"#)
        .await;

    assert_eq!(events.len(), 6);
    assert!(matches!(
        events.first(),
        Some(Ok(ChatStreamEvent::StreamStart { metadata }))
            if metadata.provider == "bedrock"
                && metadata.model.as_deref() == Some("anthropic.claude-3-sonnet")
    ));
    assert!(matches!(
        events.get(1),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::StreamStart { warnings }
        })) if warnings.is_empty()
    ));
    assert!(matches!(
        events.get(2),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::ResponseMetadata(metadata)
        })) if metadata.model.as_deref() == Some("anthropic.claude-3-sonnet")
    ));
    assert!(matches!(
        events.get(3),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Raw { raw_value }
        })) if raw_value["contentBlockDelta"]["delta"]["text"] == serde_json::json!("Hello")
    ));
    assert!(matches!(
        events.get(4),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::TextStart { id, .. }
        })) if id == "0"
    ));
    assert!(matches!(
        events.get(5),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta { id, delta, .. }
        })) if id == "0" && delta == "Hello"
    ));
}

#[tokio::test]
async fn bedrock_parse_error_emits_stream_start_and_response_metadata_before_error() {
    let converter = test_converter(false, vec![]);

    let events = converter.convert_json("{ not json").await;

    assert_eq!(events.len(), 4);
    assert!(matches!(
        events.first(),
        Some(Ok(ChatStreamEvent::StreamStart { metadata }))
            if metadata.provider == "bedrock"
                && metadata.model.as_deref() == Some("anthropic.claude-3-sonnet")
    ));
    assert!(matches!(
        events.get(1),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::StreamStart { warnings }
        })) if warnings.is_empty()
    ));
    assert!(matches!(
        events.get(2),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::ResponseMetadata(metadata)
        })) if metadata.model.as_deref() == Some("anthropic.claude-3-sonnet")
    ));
    assert!(matches!(
        events.get(3),
        Some(Err(LlmError::ParseError(message)))
            if message.contains("Failed to parse Bedrock JSON chunk")
    ));
}

#[tokio::test]
async fn bedrock_parse_error_with_raw_chunks_keeps_preamble_and_no_duplicate_later() {
    let converter = test_converter(true, vec![]);

    let invalid = converter.convert_json("{ not json").await;
    assert_eq!(invalid.len(), 5);
    assert!(matches!(
        invalid.get(3),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Raw { raw_value }
        })) if raw_value == &serde_json::Value::String("{ not json".to_string())
    ));
    assert!(matches!(
        invalid.get(4),
        Some(Err(LlmError::ParseError(message)))
            if message.contains("Failed to parse Bedrock JSON chunk")
    ));

    let later = converter
        .convert_json(r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"Later"}}}"#)
        .await;
    assert_eq!(later.len(), 3);
    assert!(matches!(
        later.first(),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Raw { raw_value }
        })) if raw_value["contentBlockDelta"]["delta"]["text"] == serde_json::json!("Later")
    ));
    assert!(matches!(
        later.get(1),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::TextStart { id, .. }
        })) if id == "0"
    ));
    assert!(matches!(
        later.get(2),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta { id, delta, .. }
        })) if id == "0" && delta == "Later"
    ));
}

#[tokio::test]
async fn bedrock_text_blocks_emit_stable_text_parts_and_finish() {
    let converter = test_converter(false, vec![]);

    let mut events = Vec::new();
    events.extend(
        converter
            .convert_json(
                r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"Hello"}}}"#,
            )
            .await,
    );
    events.extend(
        converter
            .convert_json(r#"{"contentBlockStop":{"contentBlockIndex":0}}"#)
            .await,
    );
    events.extend(
        converter
            .convert_json(r#"{"messageStop":{"stopReason":"end_turn"}}"#)
            .await,
    );

    let parts = collected_parts(&events);
    assert!(matches!(
        parts.get(2),
        Some(ChatStreamPart::TextStart { id, .. }) if id == "0"
    ));
    assert!(matches!(
        parts.get(3),
        Some(ChatStreamPart::TextDelta { id, delta, .. })
            if id == "0" && delta == "Hello"
    ));
    assert!(matches!(
        parts.get(4),
        Some(ChatStreamPart::TextEnd { id, .. }) if id == "0"
    ));
    assert!(matches!(
        parts.get(5),
        Some(ChatStreamPart::Finish { finish_reason, .. })
            if finish_reason.unified == FinishReason::Stop
                && finish_reason.raw.as_deref() == Some("end_turn")
    ));
}

#[tokio::test]
async fn bedrock_reasoning_blocks_emit_stable_reasoning_parts() {
    let converter = test_converter(false, vec![]);

    let mut events = Vec::new();
    events.extend(
            converter
                .convert_json(
                    r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"thinking"}}}}"#,
                )
                .await,
        );
    events.extend(
            converter
                .convert_json(
                    r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig-1"}}}}"#,
                )
                .await,
        );
    events.extend(
        converter
            .convert_json(r#"{"contentBlockStop":{"contentBlockIndex":0}}"#)
            .await,
    );
    events.extend(
        converter
            .convert_json(r#"{"messageStop":{"stopReason":"stop_sequence"}}"#)
            .await,
    );

    let parts = collected_parts(&events);
    assert!(matches!(
        parts.get(2),
        Some(ChatStreamPart::ReasoningStart { id, .. }) if id == "0"
    ));
    assert!(matches!(
        parts.get(3),
        Some(ChatStreamPart::ReasoningDelta { id, delta, .. })
            if id == "0" && delta == "thinking"
    ));
    assert!(matches!(
        parts.get(4),
        Some(ChatStreamPart::ReasoningDelta {
            id,
            delta,
            provider_metadata: Some(provider_metadata),
        })
            if id == "0"
                && delta.is_empty()
                && provider_metadata.get("bedrock")
                    == Some(&serde_json::json!({"signature":"sig-1"}))
    ));
    assert!(matches!(
        parts.get(5),
        Some(ChatStreamPart::ReasoningEnd { id, .. }) if id == "0"
    ));
}

#[tokio::test]
async fn bedrock_tool_blocks_emit_stable_tool_parts_and_finish() {
    let converter = test_converter(false, vec![]);

    let mut events = Vec::new();
    events.extend(
            converter
                .convert_json(
                    r#"{"contentBlockStart":{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"tool-1","name":"weather"}}}}"#,
                )
                .await,
        );
    events.extend(
            converter
                .convert_json(
                    r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"location\":\"SF\"}"}}}}"#,
                )
                .await,
        );
    events.extend(
        converter
            .convert_json(r#"{"contentBlockStop":{"contentBlockIndex":0}}"#)
            .await,
    );
    events.extend(
        converter
            .convert_json(r#"{"messageStop":{"stopReason":"tool_use"}}"#)
            .await,
    );

    let parts = collected_parts(&events);
    assert!(matches!(
        parts.get(2),
        Some(ChatStreamPart::ToolInputStart { id, tool_name, .. })
            if id == "tool-1" && tool_name == "weather"
    ));
    assert!(matches!(
        parts.get(3),
        Some(ChatStreamPart::ToolInputDelta { id, delta, .. })
            if id == "tool-1" && delta == "{\"location\":\"SF\"}"
    ));
    assert!(matches!(
        parts.get(4),
        Some(ChatStreamPart::ToolInputEnd { id, .. }) if id == "tool-1"
    ));
    assert!(matches!(
        parts.get(5),
        Some(ChatStreamPart::ToolCall(tool_call))
            if tool_call.tool_call_id == "tool-1"
                && tool_call.tool_name == "weather"
                && tool_call.input == "{\"location\":\"SF\"}"
    ));
    assert!(matches!(
        parts.get(6),
        Some(ChatStreamPart::Finish { finish_reason, .. })
            if finish_reason.unified == FinishReason::ToolCalls
                && finish_reason.raw.as_deref() == Some("tool_use")
    ));
}

#[tokio::test]
async fn bedrock_json_tool_blocks_emit_text_parts_and_stop_finish() {
    let converter = test_converter_with_options(true, false, vec![]);

    let mut events = Vec::new();
    events.extend(
            converter
                .convert_json(
                    r#"{"contentBlockStart":{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call_1","name":"json"}}}}"#,
                )
                .await,
        );
    events.extend(
            converter
                .convert_json(
                    r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"value\":\"ok\"}"}}}}"#,
                )
                .await,
        );
    events.extend(
        converter
            .convert_json(r#"{"contentBlockStop":{"contentBlockIndex":0}}"#)
            .await,
    );
    events.extend(
        converter
            .convert_json(r#"{"messageStop":{"stopReason":"tool_use"}}"#)
            .await,
    );

    let parts = collected_parts(&events);
    assert!(!parts.iter().any(|part| matches!(
        part,
        ChatStreamPart::ToolInputStart { .. }
            | ChatStreamPart::ToolInputDelta { .. }
            | ChatStreamPart::ToolInputEnd { .. }
            | ChatStreamPart::ToolCall(_)
    )));
    assert!(matches!(
        parts.get(2),
        Some(ChatStreamPart::TextStart { id, .. }) if id == "0"
    ));
    assert!(matches!(
        parts.get(3),
        Some(ChatStreamPart::TextDelta { id, delta, .. })
            if id == "0" && delta == "{\"value\":\"ok\"}"
    ));
    assert!(matches!(
        parts.get(4),
        Some(ChatStreamPart::TextEnd { id, .. }) if id == "0"
    ));
    assert!(matches!(
        parts.get(5),
        Some(ChatStreamPart::Finish {
            finish_reason,
            provider_metadata: Some(provider_metadata),
            ..
        })
            if finish_reason.unified == FinishReason::Stop
                && finish_reason.raw.as_deref() == Some("tool_use")
                && provider_metadata.get("bedrock")
                    == Some(&serde_json::json!({"isJsonResponseFromTool":true}))
    ));
}

#[tokio::test]
async fn bedrock_json_tool_clean_eof_stream_end_preserves_json_text_response() {
    let converter = test_converter_with_options(true, false, vec![]);

    let mut events = Vec::new();
    events.extend(
            converter
                .convert_json(
                    r#"{"contentBlockStart":{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call_1","name":"json"}}}}"#,
                )
                .await,
        );
    events.extend(
            converter
                .convert_json(
                    r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"value\":\"ok\"}"}}}}"#,
                )
                .await,
        );
    events.extend(
        converter
            .convert_json(r#"{"contentBlockStop":{"contentBlockIndex":0}}"#)
            .await,
    );
    events.extend(converter.handle_stream_end_events());

    let stream_end = events
        .into_iter()
        .find_map(|event| match event {
            Ok(ChatStreamEvent::StreamEnd { response }) => Some(response),
            _ => None,
        })
        .expect("stream end response");

    assert_eq!(stream_end.finish_reason, Some(FinishReason::Unknown));
    assert_eq!(stream_end.text().as_deref(), Some("{\"value\":\"ok\"}"));
    assert_eq!(
        stream_end
            .bedrock_metadata()
            .and_then(|metadata| metadata.is_json_response_from_tool),
        Some(true)
    );
}

#[tokio::test]
async fn bedrock_finish_part_preserves_usage_and_provider_metadata() {
    let converter = test_converter(false, vec![]);

    let mut events = Vec::new();
    events.extend(
            converter
                .convert_json(
                    r#"{"metadata":{"usage":{"inputTokens":4,"outputTokens":34,"totalTokens":38,"cacheReadInputTokens":2,"cacheWriteInputTokens":3,"cacheDetails":[{"inputTokens":100,"ttl":"T5M"}],"futureUsageField":{"x":1}},"trace":{"request":"trace-1"},"performanceConfig":{"latency":"optimized"},"serviceTier":{"type":"on-demand"}}}"#,
                )
                .await,
        );
    events.extend(
            converter
                .convert_json(
                    r#"{"messageStop":{"stopReason":"stop_sequence","additionalModelResponseFields":{"delta":{"stop_sequence":"STOP"}}}}"#,
                )
                .await,
        );

    let finish = collected_parts(&events)
        .into_iter()
        .find_map(|part| match part {
            ChatStreamPart::Finish {
                usage,
                finish_reason,
                provider_metadata,
            } => Some((usage, finish_reason, provider_metadata)),
            _ => None,
        })
        .expect("finish part");

    assert_eq!(finish.0.normalized_input_tokens().total, Some(9));
    assert_eq!(finish.0.normalized_input_tokens().no_cache, Some(4));
    assert_eq!(finish.0.normalized_input_tokens().cache_read, Some(2));
    assert_eq!(finish.0.normalized_input_tokens().cache_write, Some(3));
    assert_eq!(finish.0.normalized_output_tokens().total, Some(34));
    assert_eq!(
        finish.0.raw_usage_value(),
        Some(serde_json::json!({
            "inputTokens": 4,
            "outputTokens": 34,
            "totalTokens": 38,
            "cacheReadInputTokens": 2,
            "cacheWriteInputTokens": 3,
            "cacheDetails": [{ "inputTokens": 100, "ttl": "T5M" }],
            "futureUsageField": { "x": 1 }
        }))
    );
    assert_eq!(finish.1.unified, FinishReason::Stop);
    assert_eq!(finish.1.raw.as_deref(), Some("stop_sequence"));
    assert_eq!(
        finish
            .2
            .as_ref()
            .and_then(|metadata| metadata.get("bedrock"))
            .cloned(),
        Some(serde_json::json!({
            "usage": {
                "cacheWriteInputTokens": 3,
                "cacheDetails": [{ "inputTokens": 100, "ttl": "T5M" }]
            },
            "trace": { "request": "trace-1" },
            "performanceConfig": { "latency": "optimized" },
            "serviceTier": { "type": "on-demand" },
            "stopSequence": "STOP"
        }))
    );
}

#[tokio::test]
async fn bedrock_provider_error_chunk_emits_raw_and_error_after_preamble() {
    let converter = test_converter(true, vec![]);

    let events = converter
        .convert_json(r#"{"throttlingException":{"message":"slow down"}}"#)
        .await;

    assert_eq!(events.len(), 5);
    assert!(matches!(
        events.get(3),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Raw { raw_value }
        })) if raw_value["throttlingException"]["message"] == serde_json::json!("slow down")
    ));
    assert!(matches!(
        events.get(4),
        Some(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Error { error }
        })) if error["message"] == serde_json::json!("slow down")
    ));
    assert!(converter.handle_stream_end().is_none());
}

#[tokio::test]
async fn bedrock_stream_end_preserves_model_warnings_and_stop_sequence() {
    let warning = Warning::unsupported("topK", None::<String>);
    let converter = test_converter(false, vec![warning.clone()]);

    let events = converter
            .convert_json(
                r#"{"metadata":{"usage":{"inputTokens":1,"outputTokens":2,"totalTokens":3}},"messageStop":{"stopReason":"stop_sequence","additionalModelResponseFields":{"delta":{"stop_sequence":"END"}}}}"#,
            )
            .await;

    let response = events
        .iter()
        .find_map(|event| match event {
            Ok(ChatStreamEvent::StreamEnd { response }) => Some(response),
            _ => None,
        })
        .expect("stream end response");

    assert_eq!(response.model.as_deref(), Some("anthropic.claude-3-sonnet"));
    assert_eq!(response.raw_finish_reason.as_deref(), Some("stop_sequence"));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.warnings, Some(vec![warning]));
    assert_eq!(
        response
            .bedrock_metadata()
            .and_then(|meta| meta.stop_sequence.clone()),
        Some(serde_json::json!("END"))
    );
}
