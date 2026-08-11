use serde::{Deserialize, Serialize};
use serde_json::json;
use siumai_core::{
    ContentAnnotationTarget, ContentAnnotations, ContentPart, LanguageRequest, Message,
    MessagePart, MessageRole, ModelId, StructuredOutputSpec, ToolAnnotationTarget, ToolAnnotations,
    ToolChoice, ToolSpec, TypedProviderAnnotation,
};

use super::annotations::{
    AnthropicToolReference, CacheControl, CacheTtl, ContentNodeOptions, MessagesAnnotationResolver,
    MidConversationToolChange, ToolNodeOptions,
};
use super::options::{
    AdvisorToolOptions, AnthropicTool, ClearThinkingEdit, ClearThinkingKeep, ClearToolUsesEdit,
    CompactionEdit, ComputerToolOptions, ContainerSkill, ContextManagement, FallbackOutputConfig,
    InferenceGeo, InferenceSpeed, McpAuthorizationToken, McpServer, McpToolConfig,
    McpToolsetOptions, MessagesContainer, MessagesRequestOptions, MessagesServiceTierPreference,
    MessagesTokenCountOptions, OutputEffort, ResponseInclusion, ServerFallback, ServerFallbacks,
    TextEditorToolOptions, ThinkingConfig, ThinkingDisplay, TokenTaskBudget, ToolCaller,
    UserLocation, WebFetchToolOptions, WebSearchToolOptions,
};
use super::request::{
    anthropic_tool_anchor_schema, encode_count_tokens_request,
    encode_count_tokens_request_with_resolver_and_rules, encode_request,
    encode_request_with_resolver, encode_request_with_resolver_and_rules,
    encode_request_with_rules,
};
use super::rules::{CacheControlWireStyle, MessagesEncodingRules, TemperatureEncodingRule};
use super::{API_MODE_ID, MessagesCodecError};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestToolProjection {
    anthropic_tool: Option<AnthropicTool>,
    cache_control: Option<CacheControl>,
    allowed_callers: Vec<ToolCaller>,
    strict: Option<bool>,
    defer_loading: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestContentProjection {
    tool_change: Option<MidConversationToolChange>,
    cache_control: Option<CacheControl>,
}

impl TypedProviderAnnotation for TestContentProjection {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "test-anthropic-content";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

impl TypedProviderAnnotation for TestToolProjection {
    type Target = ToolAnnotationTarget;

    const NAMESPACE: &'static str = "test-anthropic-request";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

#[derive(Debug, Clone, Copy)]
struct TestResolver;

impl MessagesAnnotationResolver for TestResolver {
    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<ContentNodeOptions, MessagesCodecError> {
        let Some(annotation) = annotations
            .decode::<TestContentProjection>()
            .map_err(|source| MessagesCodecError::InvalidAnnotation {
                node: "test content",
                source,
            })?
        else {
            return Ok(ContentNodeOptions::default());
        };
        let mut options = ContentNodeOptions::default();
        if let Some(cache_control) = annotation.cache_control {
            options = options.with_cache_control(cache_control);
        }
        if let Some(tool_change) = annotation.tool_change {
            options = options.with_tool_change(tool_change);
        }
        Ok(options)
    }

    fn resolve_tool(
        &self,
        annotations: &ToolAnnotations,
    ) -> Result<ToolNodeOptions, MessagesCodecError> {
        let Some(annotation) = annotations
            .decode::<TestToolProjection>()
            .map_err(|source| MessagesCodecError::InvalidAnnotation {
                node: "test tool",
                source,
            })?
        else {
            return Ok(ToolNodeOptions::default());
        };

        let mut options =
            ToolNodeOptions::default().with_allowed_callers(annotation.allowed_callers);
        if let Some(anthropic_tool) = annotation.anthropic_tool {
            options = options.with_anthropic_tool(anthropic_tool);
        }
        if let Some(cache_control) = annotation.cache_control {
            options = options.with_cache_control(cache_control);
        }
        if let Some(strict) = annotation.strict {
            options = options.with_strict(strict);
        }
        if let Some(defer_loading) = annotation.defer_loading {
            options = options.with_defer_loading(defer_loading);
        }
        Ok(options)
    }
}

fn model() -> ModelId {
    ModelId::new("claude-fable-5").unwrap()
}

fn request() -> LanguageRequest {
    let mut request = LanguageRequest::new(vec![Message::text(MessageRole::User, "Hello")]);
    request.generation.max_output_tokens = Some(4_096);
    request
}

fn request_with_cache(ttl: CacheTtl) -> LanguageRequest {
    let part = MessagePart::new(ContentPart::Text {
        text: "Hello".to_string(),
    })
    .with_provider_annotation(&TestContentProjection {
        tool_change: None,
        cache_control: Some(CacheControl::new(ttl)),
    })
    .unwrap();
    let mut request = LanguageRequest::new(vec![Message::new(MessageRole::User, [part])]);
    request.generation.max_output_tokens = Some(4_096);
    request
}

#[test]
fn compatible_baseline_rejects_unverified_mid_conversation_system_messages() {
    let mut request = request();
    request
        .messages
        .push(Message::text(MessageRole::System, "Updated policy"));

    assert!(matches!(
        encode_request_with_rules(
            &model(),
            &request,
            &MessagesRequestOptions::default(),
            &MessagesEncodingRules::compatible_baseline(),
        ),
        Err(MessagesCodecError::Unsupported {
            feature: "mid-conversation system messages",
        })
    ));

    let native = encode_request(&model(), &request, &MessagesRequestOptions::default()).unwrap();
    assert_eq!(native["messages"][1]["role"], "system");
}

fn anthropic_tool(
    anthropic_tool: AnthropicTool,
    cache_control: Option<CacheControl>,
    allowed_callers: Vec<ToolCaller>,
    strict: Option<bool>,
    defer_loading: Option<bool>,
) -> ToolSpec {
    ToolSpec::new(
        anthropic_tool.canonical_name(),
        None,
        anthropic_tool_anchor_schema(),
    )
    .unwrap()
    .with_provider_annotation(&TestToolProjection {
        anthropic_tool: Some(anthropic_tool),
        cache_control,
        allowed_callers,
        strict,
        defer_loading,
    })
    .unwrap()
}

#[test]
fn merges_effort_with_canonical_structured_output_and_encodes_thinking_display() {
    let mut request = request();
    request.structured_output = Some(StructuredOutputSpec {
        name: "answer".to_string(),
        description: None,
        schema: json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        }),
        strict: true,
    });
    let options = MessagesRequestOptions::default()
        .with_thinking(ThinkingConfig::adaptive().with_display(ThinkingDisplay::Summarized))
        .with_output_effort(OutputEffort::XHigh);

    let encoded = encode_request(&model(), &request, &options).unwrap();
    assert_eq!(encoded["thinking"]["type"], "adaptive");
    assert_eq!(encoded["thinking"]["display"], "summarized");
    assert_eq!(encoded["output_config"]["effort"], "xhigh");
    assert_eq!(encoded["output_config"]["format"]["type"], "json_schema");
    assert_eq!(
        encoded["output_config"]["format"]["schema"]["required"],
        json!(["answer"])
    );
}

#[test]
fn encodes_typed_server_side_fallback_chain() {
    let fallback_format = StructuredOutputSpec {
        name: "fallback_answer".to_string(),
        description: None,
        schema: json!({"type": "object", "properties": {"ok": {"type": "boolean"}}}),
        strict: true,
    };
    let fallback = ServerFallback::new("claude-opus-5")
        .unwrap()
        .with_max_tokens(3_072)
        .with_thinking(ThinkingConfig::enabled(1_024).with_display(ThinkingDisplay::Omitted))
        .with_output_config(
            FallbackOutputConfig::new()
                .with_effort(OutputEffort::Max)
                .with_format(fallback_format),
        )
        .with_speed(InferenceSpeed::Fast);
    let options = MessagesRequestOptions::default()
        .with_fallbacks(ServerFallbacks::explicit(vec![fallback]).unwrap());

    let encoded = encode_request(&model(), &request(), &options).unwrap();
    let fallback = &encoded["fallbacks"][0];
    assert_eq!(fallback["model"], "claude-opus-5");
    assert_eq!(fallback["max_tokens"], 3_072);
    assert_eq!(fallback["thinking"]["display"], "omitted");
    assert_eq!(fallback["output_config"]["effort"], "max");
    assert_eq!(fallback["output_config"]["format"]["type"], "json_schema");
    assert_eq!(fallback["speed"], "fast");

    let default = encode_request(
        &model(),
        &request(),
        &MessagesRequestOptions::default().with_fallbacks(ServerFallbacks::Default),
    )
    .unwrap();
    assert_eq!(default["fallbacks"], "default");
}

#[test]
fn projects_current_anthropic_tools_from_language_request_tools() {
    let web_search = WebSearchToolOptions::default()
        .with_allowed_domains(["docs.rs", "rust-lang.org"])
        .with_max_uses(3)
        .with_response_inclusion(ResponseInclusion::Excluded)
        .with_user_location(
            UserLocation::new()
                .with_country("US")
                .with_region("California")
                .with_timezone("America/Los_Angeles"),
        );
    let web_fetch = WebFetchToolOptions::default()
        .with_blocked_domains(["example.invalid"])
        .with_citations(true)
        .with_max_content_tokens(8_192)
        .with_max_uses(2)
        .with_response_inclusion(ResponseInclusion::Full)
        .with_use_cache(false);
    let mcp = McpToolsetOptions::new("company_tools")
        .unwrap()
        .with_default_config(
            McpToolConfig::default()
                .with_enabled(true)
                .with_defer_loading(true),
        )
        .with_config(
            "dangerous_tool",
            McpToolConfig::default().with_enabled(false),
        );

    let mut request = request();
    request.tools = vec![
        anthropic_tool(
            AnthropicTool::web_search_20260318(web_search),
            Some(CacheControl::new(CacheTtl::OneHour)),
            vec![ToolCaller::Direct, ToolCaller::CodeExecution20260521],
            Some(true),
            Some(true),
        ),
        anthropic_tool(
            AnthropicTool::web_fetch_20260318(web_fetch),
            None,
            Vec::new(),
            None,
            None,
        ),
        anthropic_tool(
            AnthropicTool::CodeExecution20260521,
            None,
            Vec::new(),
            None,
            None,
        ),
        anthropic_tool(
            AnthropicTool::advisor_20260301(
                AdvisorToolOptions::new("claude-opus-5")
                    .unwrap()
                    .with_max_tokens(2_048)
                    .with_max_uses(2)
                    .with_caching(CacheControl::new(CacheTtl::FiveMinutes)),
            ),
            None,
            Vec::new(),
            None,
            None,
        ),
        anthropic_tool(
            AnthropicTool::ToolSearchRegex20251119,
            None,
            Vec::new(),
            None,
            None,
        ),
        anthropic_tool(
            AnthropicTool::ToolSearchBm25V20251119,
            None,
            Vec::new(),
            None,
            None,
        ),
        anthropic_tool(
            AnthropicTool::mcp_toolset(mcp),
            None,
            Vec::new(),
            None,
            None,
        ),
        anthropic_tool(AnthropicTool::Memory20250818, None, Vec::new(), None, None),
        anthropic_tool(AnthropicTool::Bash20250124, None, Vec::new(), None, None),
        anthropic_tool(
            AnthropicTool::text_editor_20250728(
                TextEditorToolOptions::new().with_max_characters(16_384),
            ),
            None,
            Vec::new(),
            None,
            None,
        ),
        anthropic_tool(
            AnthropicTool::computer_20251124(
                ComputerToolOptions::new(1_920, 1_080)
                    .with_display_number(1)
                    .with_enable_zoom(true),
            ),
            None,
            Vec::new(),
            None,
            None,
        ),
    ];

    let encoded = encode_request_with_resolver(
        &model(),
        &request,
        &MessagesRequestOptions::default().with_mcp_servers(vec![
            McpServer::new("company_tools", "https://mcp.example.test").unwrap(),
        ]),
        &TestResolver,
    )
    .unwrap();
    let tools = encoded["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 11);
    assert_eq!(tools[0]["type"], "web_search_20260318");
    assert_eq!(tools[0]["allowed_callers"][1], "code_execution_20260521");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(tools[0]["strict"], true);
    assert_eq!(tools[0]["defer_loading"], true);
    assert_eq!(tools[1]["type"], "web_fetch_20260318");
    assert_eq!(tools[1]["citations"]["enabled"], true);
    assert_eq!(tools[1]["use_cache"], false);
    assert_eq!(tools[2]["type"], "code_execution_20260521");
    assert_eq!(tools[3]["type"], "advisor_20260301");
    assert_eq!(tools[3]["max_tokens"], 2_048);
    assert_eq!(tools[4]["type"], "tool_search_tool_regex_20251119");
    assert_eq!(tools[5]["type"], "tool_search_tool_bm25_20251119");
    assert_eq!(tools[6]["type"], "mcp_toolset");
    assert_eq!(tools[6]["mcp_server_name"], "company_tools");
    assert_eq!(tools[6]["configs"]["dangerous_tool"]["enabled"], false);
    assert_eq!(tools[7]["type"], "memory_20250818");
    assert_eq!(tools[8]["type"], "bash_20250124");
    assert_eq!(tools[9]["type"], "text_editor_20250728");
    assert_eq!(tools[10]["type"], "computer_20251124");
    assert_eq!(tools[10]["enable_zoom"], true);
}

#[test]
fn anthropic_tools_fail_closed_on_noncanonical_anchors_and_unbounded_fields() {
    let annotation = TestToolProjection {
        anthropic_tool: Some(AnthropicTool::web_search_20260318(
            WebSearchToolOptions::default(),
        )),
        cache_control: None,
        allowed_callers: Vec::new(),
        strict: None,
        defer_loading: None,
    };
    let wrong_name = ToolSpec::new("search", None, anthropic_tool_anchor_schema())
        .unwrap()
        .with_provider_annotation(&annotation)
        .unwrap();
    let mut wrong_request = request();
    wrong_request.tools.push(wrong_name);
    assert!(matches!(
        encode_request_with_resolver(
            &model(),
            &wrong_request,
            &MessagesRequestOptions::default(),
            &TestResolver,
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "tools.anthropic_tool.anchor.name",
            ..
        })
    ));

    let invalid_domains = AnthropicTool::web_search_20260318(
        WebSearchToolOptions::default()
            .with_allowed_domains(["https://example.com"])
            .with_blocked_domains(["blocked.example"]),
    );
    let mut invalid_request = request();
    invalid_request.tools.push(anthropic_tool(
        invalid_domains,
        None,
        Vec::new(),
        None,
        None,
    ));
    assert!(matches!(
        encode_request_with_resolver(
            &model(),
            &invalid_request,
            &MessagesRequestOptions::default(),
            &TestResolver,
        ),
        Err(MessagesCodecError::InvalidOption { .. })
    ));
}

#[test]
fn ordinary_function_tools_remain_normal_tool_specs() {
    let mut request = request();
    request.tools.push(
        ToolSpec::new(
            "lookup",
            Some("Look up a record".to_string()),
            json!({
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"]
            }),
        )
        .unwrap()
        .with_provider_annotation(&TestToolProjection {
            anthropic_tool: None,
            cache_control: None,
            allowed_callers: vec![ToolCaller::Direct],
            strict: Some(true),
            defer_loading: Some(true),
        })
        .unwrap(),
    );

    let encoded = encode_request_with_resolver(
        &model(),
        &request,
        &MessagesRequestOptions::default(),
        &TestResolver,
    )
    .unwrap();
    assert_eq!(encoded["tools"][0]["name"], "lookup");
    assert_eq!(encoded["tools"][0]["description"], "Look up a record");
    assert_eq!(encoded["tools"][0]["input_schema"]["type"], "object");
    assert_eq!(encoded["tools"][0]["allowed_callers"][0], "direct");
    assert_eq!(encoded["tools"][0]["strict"], true);
    assert_eq!(encoded["tools"][0]["defer_loading"], true);
    assert!(encoded["tools"][0].get("type").is_none());
}

#[test]
fn encodes_mid_conversation_tool_changes_as_typed_system_blocks() {
    let tool = ToolSpec::new("lookup", None, json!({"type": "object"})).unwrap();
    let change = MidConversationToolChange::remove(AnthropicToolReference::tool("lookup").unwrap());
    let anchor = MessagePart::new(ContentPart::Text {
        text: String::new(),
    })
    .with_provider_annotation(&TestContentProjection {
        tool_change: Some(change),
        cache_control: None,
    })
    .unwrap();
    let mut request = request();
    request.tools.push(tool);
    request
        .messages
        .push(Message::new(MessageRole::System, [anchor]));

    let encoded = encode_request_with_resolver(
        &model(),
        &request,
        &MessagesRequestOptions::default(),
        &TestResolver,
    )
    .unwrap();
    assert_eq!(encoded["messages"][1]["role"], "system");
    assert_eq!(encoded["messages"][1]["content"][0]["type"], "tool_removal");
    assert_eq!(
        encoded["messages"][1]["content"][0]["tool"],
        json!({"type": "tool_reference", "name": "lookup"})
    );
}

#[test]
fn rejects_tool_changes_before_conversation_or_for_undeclared_tools() {
    let change = MidConversationToolChange::add(AnthropicToolReference::tool("lookup").unwrap());
    let anchor = MessagePart::new(ContentPart::Text {
        text: String::new(),
    })
    .with_provider_annotation(&TestContentProjection {
        tool_change: Some(change.clone()),
        cache_control: None,
    })
    .unwrap();
    let mut preamble =
        LanguageRequest::new(vec![Message::new(MessageRole::System, [anchor.clone()])]);
    preamble.generation.max_output_tokens = Some(2_048);
    assert!(matches!(
        encode_request_with_resolver(
            &model(),
            &preamble,
            &MessagesRequestOptions::default(),
            &TestResolver,
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "messages.tool_change",
            ..
        })
    ));

    let mut undeclared = request();
    undeclared
        .messages
        .push(Message::new(MessageRole::System, [anchor]));
    assert!(matches!(
        encode_request_with_resolver(
            &model(),
            &undeclared,
            &MessagesRequestOptions::default(),
            &TestResolver,
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "messages.tool_change.tool",
            ..
        })
    ));
}

#[test]
fn native_and_compatible_temperature_rules_enforce_their_exact_upper_bounds() {
    let mut native_request = request();
    native_request.generation.temperature = Some(1.01);
    assert!(matches!(
        encode_request(
            &model(),
            &native_request,
            &MessagesRequestOptions::default()
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "temperature",
            ..
        })
    ));

    let rules = MessagesEncodingRules::native().with_temperature(
        TemperatureEncodingRule::new(2.0).expect("two is a valid temperature maximum"),
    );
    let mut compatible_request = request();
    compatible_request.generation.temperature = Some(2.0);
    let encoded = encode_request_with_rules(
        &model(),
        &compatible_request,
        &MessagesRequestOptions::default(),
        &rules,
    )
    .unwrap();
    assert_eq!(encoded["temperature"], 2.0);

    compatible_request.generation.temperature = Some(2.01);
    assert!(matches!(
        encode_request_with_rules(
            &model(),
            &compatible_request,
            &MessagesRequestOptions::default(),
            &rules,
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "temperature",
            ..
        })
    ));
}

#[test]
fn temperature_rules_reject_invalid_maxima() {
    for maximum in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01] {
        assert!(TemperatureEncodingRule::new(maximum).is_err());
    }
    assert_eq!(TemperatureEncodingRule::new(0.0).unwrap().maximum(), 0.0);
}

#[test]
fn cache_control_wire_style_preserves_native_ttl_and_supports_five_minute_implicit_ttl() {
    let native = encode_request_with_resolver(
        &model(),
        &request_with_cache(CacheTtl::FiveMinutes),
        &MessagesRequestOptions::default(),
        &TestResolver,
    )
    .unwrap();
    assert_eq!(
        native["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral", "ttl": "5m"})
    );

    let rules = MessagesEncodingRules::native()
        .with_cache_control(CacheControlWireStyle::FiveMinutesImplicit);
    let compatible = encode_request_with_resolver_and_rules(
        &model(),
        &request_with_cache(CacheTtl::FiveMinutes),
        &MessagesRequestOptions::default(),
        &TestResolver,
        &rules,
    )
    .unwrap();
    assert_eq!(
        compatible["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
}

#[test]
fn implicit_cache_ttl_style_rejects_one_hour_requests() {
    let rules = MessagesEncodingRules::native()
        .with_cache_control(CacheControlWireStyle::FiveMinutesImplicit);
    assert!(matches!(
        encode_request_with_resolver_and_rules(
            &model(),
            &request_with_cache(CacheTtl::OneHour),
            &MessagesRequestOptions::default(),
            &TestResolver,
            &rules,
        ),
        Err(MessagesCodecError::Unsupported { .. })
    ));
}

#[test]
fn automatic_cache_coexists_with_same_ttl_explicit_target() {
    let request = request_with_cache(CacheTtl::FiveMinutes);
    let options = MessagesRequestOptions::default()
        .with_cache_control(CacheControl::new(CacheTtl::FiveMinutes));

    let encoded =
        encode_request_with_resolver(&model(), &request, &options, &TestResolver).unwrap();
    assert_eq!(
        encoded["cache_control"],
        json!({"type": "ephemeral", "ttl": "5m"})
    );
    assert_eq!(
        encoded["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral", "ttl": "5m"})
    );
}

#[test]
fn automatic_cache_consumes_a_breakpoint_unless_its_target_is_already_cached() {
    let mut parts = (0..4)
        .map(|index| {
            MessagePart::text(format!("cached-{index}"))
                .with_provider_annotation(&TestContentProjection {
                    tool_change: None,
                    cache_control: Some(CacheControl::new(CacheTtl::FiveMinutes)),
                })
                .unwrap()
        })
        .collect::<Vec<_>>();
    parts.push(MessagePart::text("automatic-target"));
    let mut request = LanguageRequest::new(vec![Message::new(MessageRole::User, parts)]);
    request.generation.max_output_tokens = Some(4_096);

    assert!(matches!(
        encode_request_with_resolver(
            &model(),
            &request,
            &MessagesRequestOptions::default()
                .with_cache_control(CacheControl::new(CacheTtl::FiveMinutes)),
            &TestResolver,
        ),
        Err(MessagesCodecError::TooManyCacheBreakpoints {
            actual: 5,
            maximum: 4,
        })
    ));
}

#[test]
fn automatic_cache_rejects_a_different_ttl_on_the_explicit_target() {
    assert!(matches!(
        encode_request_with_resolver(
            &model(),
            &request_with_cache(CacheTtl::FiveMinutes),
            &MessagesRequestOptions::default()
                .with_cache_control(CacheControl::new(CacheTtl::OneHour)),
            &TestResolver,
        ),
        Err(MessagesCodecError::ConflictingCacheAnnotation)
    ));
}

#[test]
fn serializes_request_only_messages_service_tier_preferences() {
    for (tier, expected) in [
        (MessagesServiceTierPreference::Auto, "auto"),
        (MessagesServiceTierPreference::StandardOnly, "standard_only"),
    ] {
        let encoded = encode_request(
            &model(),
            &request(),
            &MessagesRequestOptions::default().with_service_tier(tier),
        )
        .unwrap();
        assert_eq!(encoded["service_tier"], expected);
    }
}

#[test]
fn compaction_trigger_uses_the_versioned_fifty_thousand_token_minimum() {
    let future_model = ModelId::new("claude-future-compact").expect("valid future model");
    let options = MessagesRequestOptions::default().with_context_management(
        ContextManagement::new().with_edit(CompactionEdit::new().with_trigger_input_tokens(50_000)),
    );

    let encoded = encode_request(&future_model, &request(), &options).expect("encode compaction");
    assert_eq!(
        encoded["context_management"]["edits"][0],
        json!({
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 50_000}
        })
    );

    let below_minimum = MessagesRequestOptions::default().with_context_management(
        ContextManagement::new().with_edit(CompactionEdit::new().with_trigger_input_tokens(49_999)),
    );
    assert!(matches!(
        encode_request(&future_model, &request(), &below_minimum),
        Err(MessagesCodecError::InvalidOption {
            field: "context_management.edits[].trigger.value",
            ..
        })
    ));
}

#[test]
fn task_budget_accepts_positive_future_values_below_twenty_thousand() {
    let future_model = ModelId::new("claude-future-task-budget").expect("valid future model");
    let budget = TokenTaskBudget::new(8_000)
        .expect("positive budget")
        .with_remaining(3_000)
        .expect("remaining budget within total");
    let encoded = encode_request(
        &future_model,
        &request(),
        &MessagesRequestOptions::default().with_task_budget(budget),
    )
    .expect("encode future task budget");

    assert_eq!(
        encoded["output_config"]["task_budget"],
        json!({"type": "tokens", "total": 8_000, "remaining": 3_000})
    );
    assert!(matches!(
        TokenTaskBudget::new(0),
        Err(MessagesCodecError::InvalidOption {
            field: "output_config.task_budget.total",
            ..
        })
    ));
    assert!(matches!(
        TokenTaskBudget::new(1).and_then(|budget| budget.with_remaining(2)),
        Err(MessagesCodecError::InvalidOption {
            field: "output_config.task_budget.remaining",
            ..
        })
    ));
}

#[test]
fn encodes_current_typed_request_controls_and_redacts_mcp_tokens() {
    let container = MessagesContainer::configured()
        .with_id("container-1")
        .unwrap()
        .with_skill(ContainerSkill::anthropic("pdfs").unwrap())
        .unwrap();
    assert!(container.has_skills());
    assert_eq!(container.skills().len(), 1);
    assert_eq!(InferenceGeo::us().as_str(), "us");
    assert_eq!(InferenceGeo::global().as_str(), "global");
    let context_management = ContextManagement::new()
        .with_edit(
            ClearToolUsesEdit::new()
                .with_clear_at_least_input_tokens(8_000)
                .with_keep_tool_uses(2),
        )
        .with_edit(ClearThinkingEdit::new().with_keep(ClearThinkingKeep::ThinkingTurns(1)))
        .with_edit(
            CompactionEdit::new()
                .with_instructions("Keep decisions and open tasks")
                .with_pause_after_compaction(true)
                .with_trigger_input_tokens(100_000),
        );
    let mcp_server = McpServer::new("docs", "https://mcp.example.test")
        .unwrap()
        .with_authorization_token(McpAuthorizationToken::new("sentinel-secret").unwrap());
    assert!(!format!("{mcp_server:?}").contains("sentinel-secret"));

    let mut current_request = request();
    current_request.tools = vec![
        anthropic_tool(
            AnthropicTool::CodeExecution20260521,
            None,
            Vec::new(),
            None,
            None,
        ),
        anthropic_tool(
            AnthropicTool::mcp_toolset(McpToolsetOptions::new("docs").unwrap()),
            None,
            Vec::new(),
            None,
            None,
        ),
    ];

    let options = MessagesRequestOptions::default()
        .with_cache_control(CacheControl::new(CacheTtl::OneHour))
        .with_service_tier(MessagesServiceTierPreference::Auto)
        .with_speed(InferenceSpeed::Fast)
        .with_inference_geo(InferenceGeo::Us)
        .with_task_budget(TokenTaskBudget::new(20_000).unwrap())
        .with_container(container)
        .with_context_management(context_management)
        .with_mcp_servers(vec![mcp_server]);

    let encoded =
        encode_request_with_resolver(&model(), &current_request, &options, &TestResolver).unwrap();
    assert_eq!(
        encoded["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );
    assert_eq!(encoded["service_tier"], "auto");
    assert_eq!(encoded["speed"], "fast");
    assert_eq!(encoded["inference_geo"], "us");
    assert_eq!(
        encoded["output_config"]["task_budget"],
        json!({"type": "tokens", "total": 20_000})
    );
    assert_eq!(encoded["container"]["id"], "container-1");
    assert_eq!(encoded["container"]["skills"][0]["skill_id"], "pdfs");
    assert_eq!(
        encoded["context_management"]["edits"][2]["type"],
        "compact_20260112"
    );
    assert_eq!(encoded["mcp_servers"][0]["type"], "url");
    assert_eq!(
        encoded["mcp_servers"][0]["authorization_token"],
        "sentinel-secret"
    );
}

#[test]
fn protocol_rejects_incomplete_and_duplicate_mcp_bindings() {
    let count_options = MessagesTokenCountOptions::new().with_mcp_servers(vec![
        McpServer::new("docs", "https://mcp.example.test").unwrap(),
    ]);
    assert!(matches!(
        encode_count_tokens_request(
            &model(),
            &LanguageRequest::new(vec![Message::user("Count this")]),
            &count_options,
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "mcp_servers",
            ..
        })
    ));

    let mut duplicate_request = request();
    duplicate_request.tools.push(anthropic_tool(
        AnthropicTool::mcp_toolset(McpToolsetOptions::new("docs").unwrap()),
        None,
        Vec::new(),
        None,
        None,
    ));
    let duplicate_options = MessagesRequestOptions::default().with_mcp_servers(vec![
        McpServer::new("docs", "https://mcp.example.test/one").unwrap(),
        McpServer::new("docs", "https://mcp.example.test/two").unwrap(),
    ]);
    assert!(matches!(
        encode_request_with_resolver(
            &model(),
            &duplicate_request,
            &duplicate_options,
            &TestResolver,
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "mcp_servers[].name",
            ..
        })
    ));

    let mut duplicate_toolset_request = request();
    duplicate_toolset_request.tools = vec![
        anthropic_tool(
            AnthropicTool::mcp_toolset(
                McpToolsetOptions::with_anchor("docs_primary", "docs").unwrap(),
            ),
            None,
            Vec::new(),
            None,
            None,
        ),
        anthropic_tool(
            AnthropicTool::mcp_toolset(
                McpToolsetOptions::with_anchor("docs_secondary", "docs").unwrap(),
            ),
            None,
            Vec::new(),
            None,
            None,
        ),
    ];
    let duplicate_toolset_options = MessagesRequestOptions::default().with_mcp_servers(vec![
        McpServer::new("docs", "https://mcp.example.test").unwrap(),
    ]);
    assert!(matches!(
        encode_request_with_resolver(
            &model(),
            &duplicate_toolset_request,
            &duplicate_toolset_options,
            &TestResolver,
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "tools.mcp_toolset.mcp_server_name",
            ..
        })
    ));
}

#[test]
fn protocol_rejects_container_skills_without_code_execution() {
    let options = MessagesRequestOptions::default().with_container(
        MessagesContainer::configured()
            .with_skill(ContainerSkill::custom("skill_1").unwrap())
            .unwrap(),
    );
    assert!(matches!(
        encode_request(&model(), &request(), &options),
        Err(MessagesCodecError::InvalidOption {
            field: "container.skills",
            ..
        })
    ));
}

#[test]
fn mcp_servers_require_safe_https_urls_and_redact_debug_output() {
    let server = McpServer::new(
        "docs",
        "https://mcp.example.test/search?access=sentinel-query",
    )
    .unwrap()
    .with_authorization_token(McpAuthorizationToken::new("sentinel-token").unwrap());
    let debug = format!("{server:?}");
    assert!(debug.contains("docs"));
    assert!(debug.contains("has_authorization_token"));
    assert!(!debug.contains("mcp.example.test"));
    assert!(!debug.contains("sentinel-query"));
    assert!(!debug.contains("sentinel-token"));

    for invalid in [
        "http://mcp.example.test",
        "/relative",
        "https://user@example.test/mcp",
        "https://mcp.example.test/mcp#fragment",
    ] {
        assert!(matches!(
            McpServer::new("docs", invalid),
            Err(MessagesCodecError::InvalidOption {
                field: "mcp_servers[].url",
                ..
            })
        ));
    }
}

#[test]
fn zero_max_tokens_is_supported_only_for_non_generating_create_requests() {
    let mut zero = request();
    zero.generation.max_output_tokens = Some(0);
    let encoded = encode_request(&model(), &zero, &MessagesRequestOptions::default()).unwrap();
    assert_eq!(encoded["max_tokens"], 0);

    let mut invalid = zero;
    invalid
        .tools
        .push(ToolSpec::new("lookup", None, json!({"type": "object"})).expect("valid tool"));
    invalid.tool_choice = Some(ToolChoice::Required);
    invalid.structured_output = Some(StructuredOutputSpec {
        name: "answer".to_string(),
        description: None,
        schema: json!({"type": "object"}),
        strict: true,
    });
    let options = MessagesRequestOptions::new(true).with_thinking(ThinkingConfig::adaptive());
    assert!(matches!(
        encode_request(&model(), &invalid, &options),
        Err(MessagesCodecError::InvalidOption {
            field: "max_output_tokens",
            ..
        })
    ));
}

#[test]
fn token_count_encoder_emits_only_the_official_count_request_shape() {
    let mut count = LanguageRequest::new(vec![Message::text(MessageRole::User, "Count this")]);
    count.tools.push(anthropic_tool(
        AnthropicTool::mcp_toolset(McpToolsetOptions::new("docs").unwrap()),
        None,
        Vec::new(),
        None,
        None,
    ));
    count.tool_choice = Some(ToolChoice::Auto);
    count.structured_output = Some(StructuredOutputSpec {
        name: "answer".to_string(),
        description: None,
        schema: json!({"type": "object", "properties": {}}),
        strict: true,
    });
    let options = MessagesTokenCountOptions::new()
        .with_cache_control(CacheControl::new(CacheTtl::FiveMinutes))
        .with_thinking(ThinkingConfig::Disabled)
        .with_output_effort(OutputEffort::Low)
        .with_task_budget(TokenTaskBudget::new(20_000).unwrap())
        .with_speed(InferenceSpeed::Fast)
        .with_context_management(
            ContextManagement::new()
                .with_edit(ClearThinkingEdit::new().with_keep(ClearThinkingKeep::All)),
        )
        .with_mcp_servers(vec![
            McpServer::new("docs", "https://mcp.example.test").unwrap(),
        ]);

    let encoded = encode_count_tokens_request_with_resolver_and_rules(
        &model(),
        &count,
        &options,
        &TestResolver,
        &MessagesEncodingRules::native(),
    )
    .unwrap();
    assert_eq!(
        encoded,
        json!({
            "model": "claude-fable-5",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Count this"}]
            }],
            "tools": [{
                "type": "mcp_toolset",
                "mcp_server_name": "docs"
            }],
            "tool_choice": {"type": "auto"},
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "thinking": {"type": "disabled"},
            "output_config": {
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "properties": {}}
                },
                "task_budget": {"type": "tokens", "total": 20_000}
            },
            "speed": "fast",
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "mcp_servers": [{
                "type": "url",
                "name": "docs",
                "url": "https://mcp.example.test"
            }]
        })
    );
    for forbidden in [
        "max_tokens",
        "stream",
        "temperature",
        "top_p",
        "stop_sequences",
        "metadata",
        "service_tier",
        "inference_geo",
        "container",
        "fallbacks",
    ] {
        assert!(encoded.get(forbidden).is_none(), "unexpected {forbidden}");
    }
}

#[test]
fn token_count_encoder_rejects_generation_only_fields() {
    let mut count = LanguageRequest::new(vec![Message::text(MessageRole::User, "Count this")]);
    count.generation.temperature = Some(0.5);
    assert!(matches!(
        encode_count_tokens_request(&model(), &count, &MessagesTokenCountOptions::new()),
        Err(MessagesCodecError::Unsupported {
            feature: "temperature in token-count requests",
        })
    ));
}
