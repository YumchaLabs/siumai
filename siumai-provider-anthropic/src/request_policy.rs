use siumai_anthropic_compatible::{
    MessagesCallOptions, MessagesRequestPolicy, MessagesRequestRequirements,
};
use siumai_core::{Error, ErrorKind, LanguageRequest, MessageRole, ModelId};
use siumai_protocol_anthropic::messages::{
    AnthropicTool, ContextManagement, ContextManagementEdit, InferenceSpeed, OutputEffort,
    ServerFallback, ServerFallbacks, ThinkingConfig,
};

use crate::annotations::{AnthropicContentOptions, AnthropicToolOptions};
use crate::models::{
    CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5, CLAUDE_HAIKU_4_5_20251001, CLAUDE_MYTHOS_5,
    CLAUDE_MYTHOS_PREVIEW, CLAUDE_OPUS_4_1_20250805, CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7,
    CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5,
};

const SERVER_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const FAST_MODE_BETA: &str = "fast-mode-2026-02-01";
const TASK_BUDGET_BETA: &str = "task-budgets-2026-03-13";
const CONTEXT_MANAGEMENT_BETA: &str = "context-management-2025-06-27";
const COMPACTION_BETA: &str = "compact-2026-01-12";
const SKILLS_BETA: &str = "skills-2025-10-02";
const ADVISOR_TOOL_BETA: &str = "advisor-tool-2026-03-01";
const MCP_CLIENT_BETA: &str = "mcp-client-2025-11-20";
const COMPUTER_USE_BETA: &str = "computer-use-2025-11-24";
const MID_CONVERSATION_TOOL_CHANGES_BETA: &str = "mid-conversation-tool-changes-2026-07-01";
const MAX_OUTPUT_TOKENS_128K: u64 = 128_000;
const MAX_OUTPUT_TOKENS_64K: u64 = 64_000;

/// Anthropic-owned model defaults, compatibility rules, and beta contracts.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AnthropicRequestPolicy;

impl MessagesRequestPolicy for AnthropicRequestPolicy {
    fn prepare(
        &self,
        model: &ModelId,
        request: &LanguageRequest,
        options: &mut MessagesCallOptions,
    ) -> Result<MessagesRequestRequirements, Error> {
        validate_primary_request(model, request, options)?;

        let mut requirements = MessagesRequestRequirements::new();
        if validate_mid_conversation_messages(model.as_str(), request)? {
            requirements = requirements.with_beta_feature(MID_CONVERSATION_TOOL_CHANGES_BETA)?;
        }
        if let Some(fallbacks) = options.fallbacks() {
            requirements = requirements.with_beta_feature(SERVER_FALLBACK_BETA)?;
            if let ServerFallbacks::Explicit(fallbacks) = fallbacks {
                for fallback in fallbacks {
                    validate_fallback(fallback, request.generation.max_output_tokens)?;
                    if fallback.speed() == Some(InferenceSpeed::Fast) {
                        requirements = requirements.with_beta_feature(FAST_MODE_BETA)?;
                    }
                }
            }
        }

        if options.speed() == Some(InferenceSpeed::Fast) {
            requirements = requirements.with_beta_feature(FAST_MODE_BETA)?;
        }
        if options.task_budget().is_some() {
            requirements = requirements.with_beta_feature(TASK_BUDGET_BETA)?;
        }
        if let Some(context) = options.context_management() {
            for edit in context.edits() {
                requirements = match edit {
                    ContextManagementEdit::Compact(_) => {
                        requirements.with_beta_feature(COMPACTION_BETA)?
                    }
                    ContextManagementEdit::ClearToolUses(_)
                    | ContextManagementEdit::ClearThinking(_) => {
                        requirements.with_beta_feature(CONTEXT_MANAGEMENT_BETA)?
                    }
                    _ => requirements,
                };
            }
        }

        let container_uses_skills = options
            .container()
            .is_some_and(|container| container.has_skills());
        if container_uses_skills {
            requirements = requirements.with_beta_feature(SKILLS_BETA)?;
        }

        for tool in &request.tools {
            let annotation = tool
                .annotations()
                .decode::<AnthropicToolOptions>()
                .map_err(annotation_error)?;
            if let Some(anthropic_tool) = annotation
                .as_ref()
                .and_then(AnthropicToolOptions::anthropic_tool)
            {
                let beta = match anthropic_tool {
                    AnthropicTool::Advisor20260301(_) => Some(ADVISOR_TOOL_BETA),
                    AnthropicTool::McpToolset(_) => Some(MCP_CLIENT_BETA),
                    AnthropicTool::Computer20251124(_) => Some(COMPUTER_USE_BETA),
                    _ => None,
                };
                if let Some(beta) = beta {
                    requirements = requirements.with_beta_feature(beta)?;
                }
            }
        }

        if options.mcp_servers().is_some() {
            requirements = requirements.with_beta_feature(MCP_CLIENT_BETA)?;
        }

        Ok(requirements)
    }
}

fn validate_mid_conversation_messages(
    model: &str,
    request: &LanguageRequest,
) -> Result<bool, Error> {
    let mut conversation_started = false;
    let mut uses_tool_changes = false;
    for message in &request.messages {
        match message.role() {
            MessageRole::System if conversation_started => {
                if !supports_mid_conversation_messages(model) {
                    return Err(invalid(
                        "mid-conversation system messages are not verified for this model",
                    ));
                }
                for part in message.content() {
                    let annotation = part
                        .annotations()
                        .decode::<AnthropicContentOptions>()
                        .map_err(annotation_error)?;
                    uses_tool_changes |= annotation
                        .as_ref()
                        .and_then(AnthropicContentOptions::tool_change)
                        .is_some();
                }
            }
            MessageRole::User | MessageRole::Assistant | MessageRole::Tool => {
                conversation_started = true;
            }
            _ => {}
        }
    }
    Ok(uses_tool_changes)
}

fn supports_mid_conversation_messages(model: &str) -> bool {
    matches!(
        model,
        CLAUDE_FABLE_5 | CLAUDE_MYTHOS_5 | CLAUDE_OPUS_4_8 | CLAUDE_OPUS_5
    )
}

fn validate_primary_request(
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesCallOptions,
) -> Result<(), Error> {
    if uses_strict_sampling(model.as_str()) {
        if options.top_k().is_some() {
            return Err(invalid("Claude 4.7 and later do not accept top_k sampling"));
        }
        if request
            .generation
            .temperature
            .is_some_and(|temperature| temperature != 1.0)
        {
            return Err(invalid(
                "Claude 4.7 and later accept only the default temperature value of 1",
            ));
        }
        if request.generation.top_p.is_some_and(|top_p| top_p < 0.99) {
            return Err(invalid(
                "Claude 4.7 and later accept only the default top_p range of 0.99 to 1",
            ));
        }
    }
    validate_model_options(model.as_str(), options.thinking(), options.output_effort())?;
    validate_speed(model.as_str(), options.speed())?;
    validate_task_budget(model.as_str(), options.task_budget().is_some())?;
    validate_inference_geo(model.as_str(), options.inference_geo().is_some())?;
    validate_context_management(model.as_str(), options.context_management())?;
    validate_known_output_limit(model.as_str(), request.generation.max_output_tokens)
}

fn validate_fallback(
    fallback: &ServerFallback,
    request_max_output_tokens: Option<u64>,
) -> Result<(), Error> {
    validate_model_options(
        fallback.model_name(),
        fallback.thinking(),
        fallback.output_config().and_then(|output| output.effort()),
    )?;
    validate_speed(fallback.model_name(), fallback.speed())?;
    validate_known_output_limit(
        fallback.model_name(),
        fallback.max_tokens().or(request_max_output_tokens),
    )
}

fn validate_speed(model: &str, speed: Option<InferenceSpeed>) -> Result<(), Error> {
    if speed != Some(InferenceSpeed::Fast) || !is_known_model(model) {
        return Ok(());
    }
    if matches!(model, CLAUDE_OPUS_5 | CLAUDE_OPUS_4_8) {
        Ok(())
    } else {
        Err(invalid(
            "fast inference is currently verified only for Claude Opus 5 and Opus 4.8",
        ))
    }
}

fn validate_task_budget(model: &str, enabled: bool) -> Result<(), Error> {
    if !enabled || !is_known_model(model) {
        return Ok(());
    }
    if matches!(
        model,
        CLAUDE_OPUS_5 | CLAUDE_FABLE_5 | CLAUDE_MYTHOS_5 | CLAUDE_OPUS_4_8 | CLAUDE_OPUS_4_7
    ) {
        Ok(())
    } else {
        Err(invalid(
            "task budgets are not supported by this Claude model",
        ))
    }
}

fn validate_inference_geo(model: &str, configured: bool) -> Result<(), Error> {
    if !configured || !is_known_model(model) {
        return Ok(());
    }
    if matches!(
        model,
        CLAUDE_OPUS_4_1_20250805 | CLAUDE_HAIKU_4_5 | CLAUDE_HAIKU_4_5_20251001
    ) {
        Err(invalid(
            "inference geography is supported only by Claude 4.6 and later",
        ))
    } else {
        Ok(())
    }
}

fn validate_context_management(
    model: &str,
    context: Option<&ContextManagement>,
) -> Result<(), Error> {
    let uses_compaction = context.is_some_and(|context| {
        context
            .edits()
            .iter()
            .any(|edit| matches!(edit, ContextManagementEdit::Compact(_)))
    });
    if !uses_compaction || !is_known_model(model) {
        return Ok(());
    }
    if matches!(
        model,
        CLAUDE_HAIKU_4_5 | CLAUDE_HAIKU_4_5_20251001 | CLAUDE_OPUS_4_1_20250805
    ) {
        Err(invalid(
            "context compaction is not supported by this Claude model",
        ))
    } else {
        Ok(())
    }
}

fn is_known_model(model: &str) -> bool {
    matches!(
        model,
        CLAUDE_OPUS_5
            | CLAUDE_SONNET_5
            | CLAUDE_FABLE_5
            | CLAUDE_MYTHOS_5
            | CLAUDE_MYTHOS_PREVIEW
            | CLAUDE_OPUS_4_8
            | CLAUDE_OPUS_4_7
            | CLAUDE_OPUS_4_6
            | CLAUDE_SONNET_4_6
            | CLAUDE_HAIKU_4_5
            | CLAUDE_HAIKU_4_5_20251001
            | CLAUDE_OPUS_4_1_20250805
    )
}

fn validate_model_options(
    model: &str,
    thinking: Option<ThinkingConfig>,
    effort: Option<OutputEffort>,
) -> Result<(), Error> {
    match model {
        CLAUDE_OPUS_5 => {
            reject_manual_thinking(thinking)?;
            if thinking == Some(ThinkingConfig::Disabled)
                && matches!(effort, Some(OutputEffort::XHigh | OutputEffort::Max))
            {
                return Err(invalid(
                    "Claude Opus 5 cannot disable thinking at xhigh or max effort",
                ));
            }
        }
        CLAUDE_SONNET_5 => reject_manual_thinking(thinking)?,
        CLAUDE_FABLE_5 | CLAUDE_MYTHOS_5 => {
            if matches!(
                thinking,
                Some(ThinkingConfig::Disabled | ThinkingConfig::Enabled { .. })
            ) {
                return Err(invalid(
                    "this model requires adaptive thinking and cannot use disabled or legacy manual thinking",
                ));
            }
        }
        CLAUDE_MYTHOS_PREVIEW => {
            if thinking == Some(ThinkingConfig::Disabled) {
                return Err(invalid("Claude Mythos Preview cannot disable thinking"));
            }
            reject_manual_thinking(thinking)?;
            reject_xhigh_effort(effort)?;
        }
        CLAUDE_OPUS_4_7 | CLAUDE_OPUS_4_8 => reject_manual_thinking(thinking)?,
        CLAUDE_OPUS_4_6 | CLAUDE_SONNET_4_6 => reject_xhigh_effort(effort)?,
        CLAUDE_HAIKU_4_5 | CLAUDE_HAIKU_4_5_20251001 => {
            if effort.is_some() {
                return Err(invalid("Claude Haiku 4.5 does not support output effort"));
            }
            if matches!(thinking, Some(ThinkingConfig::Adaptive { .. })) {
                return Err(invalid(
                    "Claude Haiku 4.5 does not support adaptive thinking",
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

fn reject_xhigh_effort(effort: Option<OutputEffort>) -> Result<(), Error> {
    if effort == Some(OutputEffort::XHigh) {
        return Err(invalid(
            "this model supports low, medium, high, or max effort, not xhigh",
        ));
    }
    Ok(())
}

fn reject_manual_thinking(thinking: Option<ThinkingConfig>) -> Result<(), Error> {
    if matches!(thinking, Some(ThinkingConfig::Enabled { .. })) {
        return Err(invalid(
            "this model accepts adaptive or disabled thinking, not legacy manual thinking",
        ));
    }
    Ok(())
}

fn validate_known_output_limit(model: &str, max_output_tokens: Option<u64>) -> Result<(), Error> {
    let Some((maximum, message)) = known_output_limit(model) else {
        return Ok(());
    };
    if max_output_tokens.is_some_and(|requested| requested > maximum) {
        return Err(invalid(message));
    }
    Ok(())
}

fn known_output_limit(model: &str) -> Option<(u64, &'static str)> {
    match model {
        CLAUDE_OPUS_5
        | CLAUDE_SONNET_5
        | CLAUDE_FABLE_5
        | CLAUDE_MYTHOS_5
        | CLAUDE_MYTHOS_PREVIEW
        | CLAUDE_OPUS_4_8
        | CLAUDE_OPUS_4_7
        | CLAUDE_OPUS_4_6
        | CLAUDE_SONNET_4_6 => Some((
            MAX_OUTPUT_TOKENS_128K,
            "max_output_tokens exceeds this model's 128k output limit",
        )),
        CLAUDE_HAIKU_4_5 | CLAUDE_HAIKU_4_5_20251001 => Some((
            MAX_OUTPUT_TOKENS_64K,
            "max_output_tokens exceeds Claude Haiku 4.5's 64k output limit",
        )),
        _ => None,
    }
}

fn uses_strict_sampling(model: &str) -> bool {
    matches!(
        model,
        CLAUDE_OPUS_4_7
            | CLAUDE_OPUS_4_8
            | CLAUDE_OPUS_5
            | CLAUDE_SONNET_5
            | CLAUDE_FABLE_5
            | CLAUDE_MYTHOS_5
            | CLAUDE_MYTHOS_PREVIEW
    )
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn annotation_error(source: siumai_core::ProviderAnnotationError) -> Error {
    Error::new(ErrorKind::InvalidInput, "invalid Anthropic annotation").with_source(source)
}

#[cfg(test)]
mod tests {
    use siumai_core::{LanguageRequest, Message, MessageRole, ModelId};
    use siumai_protocol_anthropic::messages::{
        AdvisorToolOptions, AnthropicTool, AnthropicToolReference, CompactionEdit,
        ComputerToolOptions, ContainerSkill, InferenceSpeed, McpServer, McpToolsetOptions,
        MessagesContainer, MidConversationToolChange, ServerFallback, ServerFallbacks,
        ThinkingConfig, TokenTaskBudget,
    };

    use super::*;

    #[test]
    fn preparation_collects_fallback_and_anthropic_tool_beta_contracts() {
        let fallback = ServerFallback::new("fallback-model")
            .expect("fallback")
            .with_speed(InferenceSpeed::Fast);
        let mut options = MessagesCallOptions::new()
            .with_fallbacks(ServerFallbacks::explicit(vec![fallback]).expect("fallback chain"))
            .with_mcp_servers(vec![
                McpServer::new("company-tools", "https://mcp.example.test/events")
                    .expect("MCP server"),
            ]);
        let mut request = LanguageRequest::new(vec![Message::text(MessageRole::User, "tools")]);
        request.tools = vec![
            AnthropicToolOptions::for_tool(AnthropicTool::advisor_20260301(
                AdvisorToolOptions::new("advisor-model").expect("advisor"),
            ))
            .into_tool_spec()
            .expect("advisor tool"),
            AnthropicToolOptions::for_tool(AnthropicTool::mcp_toolset(
                McpToolsetOptions::new("company-tools").expect("MCP toolset"),
            ))
            .into_tool_spec()
            .expect("MCP tool"),
            AnthropicToolOptions::for_tool(AnthropicTool::computer_20251124(
                ComputerToolOptions::new(1_024, 768),
            ))
            .into_tool_spec()
            .expect("computer tool"),
        ];

        let requirements = AnthropicRequestPolicy
            .prepare(
                &ModelId::new("future-model").expect("model"),
                &request,
                &mut options,
            )
            .expect("requirements");
        assert_eq!(
            requirements.beta_features().collect::<Vec<_>>(),
            vec![
                ADVISOR_TOOL_BETA,
                COMPUTER_USE_BETA,
                FAST_MODE_BETA,
                MCP_CLIENT_BETA,
                SERVER_FALLBACK_BETA,
            ]
        );
    }

    #[test]
    fn preparation_collects_current_option_betas_once() {
        let mut options = MessagesCallOptions::new()
            .with_speed(InferenceSpeed::Fast)
            .with_task_budget(TokenTaskBudget::new(20_000).expect("task budget"))
            .with_context_management(
                siumai_protocol_anthropic::messages::ContextManagement::new()
                    .with_edit(CompactionEdit::new()),
            )
            .with_container(
                MessagesContainer::configured()
                    .with_skill(ContainerSkill::custom("skill_1").expect("skill"))
                    .expect("container"),
            )
            .with_mcp_servers(vec![
                McpServer::new("company-tools", "https://mcp.example.test/events")
                    .expect("MCP server"),
            ]);
        let mut request = LanguageRequest::new(vec![Message::user("tools")]);
        request.tools = vec![
            AnthropicToolOptions::for_tool(AnthropicTool::CodeExecution20260521)
                .into_tool_spec()
                .expect("code execution"),
            AnthropicToolOptions::for_tool(AnthropicTool::mcp_toolset(
                McpToolsetOptions::new("company-tools").expect("MCP toolset"),
            ))
            .into_tool_spec()
            .expect("MCP tool"),
        ];

        let requirements = AnthropicRequestPolicy
            .prepare(
                &ModelId::new(CLAUDE_OPUS_5).expect("model"),
                &request,
                &mut options,
            )
            .expect("requirements");
        assert_eq!(
            requirements.beta_features().collect::<Vec<_>>(),
            vec![
                COMPACTION_BETA,
                FAST_MODE_BETA,
                MCP_CLIENT_BETA,
                SKILLS_BETA,
                TASK_BUDGET_BETA,
            ]
        );
    }

    #[test]
    fn compaction_rejects_known_unsupported_models_without_closing_future_ids() {
        let context = || ContextManagement::new().with_edit(CompactionEdit::new());
        assert!(validate_context_management(CLAUDE_HAIKU_4_5, Some(&context())).is_err());
        assert!(validate_context_management(CLAUDE_OPUS_5, Some(&context())).is_ok());
        assert!(validate_context_management("future-claude", Some(&context())).is_ok());
    }

    #[test]
    fn current_effort_and_thinking_matrix_matches_documented_model_rules() {
        assert!(
            validate_model_options(
                CLAUDE_OPUS_5,
                Some(ThinkingConfig::Disabled),
                Some(OutputEffort::High),
            )
            .is_ok()
        );
        assert!(
            validate_model_options(
                CLAUDE_OPUS_5,
                Some(ThinkingConfig::Disabled),
                Some(OutputEffort::Max),
            )
            .is_err()
        );
        assert!(
            validate_model_options(
                CLAUDE_SONNET_5,
                Some(ThinkingConfig::Adaptive { display: None }),
                Some(OutputEffort::Max),
            )
            .is_ok()
        );
        assert!(
            validate_model_options(
                CLAUDE_FABLE_5,
                Some(ThinkingConfig::Enabled {
                    budget_tokens: 2_048,
                    display: None,
                }),
                Some(OutputEffort::High),
            )
            .is_err()
        );
        assert!(
            validate_model_options(
                CLAUDE_OPUS_4_6,
                Some(ThinkingConfig::Enabled {
                    budget_tokens: 2_048,
                    display: None,
                }),
                Some(OutputEffort::Max),
            )
            .is_ok()
        );
        assert!(
            validate_model_options(
                CLAUDE_OPUS_4_6,
                Some(ThinkingConfig::Adaptive { display: None }),
                Some(OutputEffort::XHigh),
            )
            .is_err()
        );
        assert!(
            validate_model_options(
                CLAUDE_HAIKU_4_5,
                Some(ThinkingConfig::Adaptive { display: None }),
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn known_output_limits_apply_to_primary_and_fallback_models_only() {
        assert!(validate_known_output_limit(CLAUDE_OPUS_5, Some(MAX_OUTPUT_TOKENS_128K)).is_ok());
        assert!(
            validate_known_output_limit(CLAUDE_OPUS_5, Some(MAX_OUTPUT_TOKENS_128K + 1)).is_err()
        );
        assert!(
            validate_known_output_limit(CLAUDE_HAIKU_4_5, Some(MAX_OUTPUT_TOKENS_64K + 1)).is_err()
        );
        assert!(validate_known_output_limit("future-claude", Some(u64::MAX)).is_ok());

        let inherited_limit = ServerFallback::new(CLAUDE_HAIKU_4_5).expect("fallback model");
        assert!(validate_fallback(&inherited_limit, Some(MAX_OUTPUT_TOKENS_64K + 1)).is_err());
        let overridden_limit = inherited_limit.with_max_tokens(MAX_OUTPUT_TOKENS_64K);
        assert!(validate_fallback(&overridden_limit, Some(MAX_OUTPUT_TOKENS_128K)).is_ok());
    }

    #[test]
    fn mid_conversation_tool_changes_are_typed_and_model_gated() {
        let anchor = AnthropicContentOptions::tool_change_part(MidConversationToolChange::remove(
            AnthropicToolReference::tool("lookup").expect("tool reference"),
        ))
        .expect("tool-change anchor");
        let mut request = LanguageRequest::new(vec![Message::text(MessageRole::User, "start")]);
        request
            .messages
            .push(Message::new(MessageRole::System, [anchor]));
        request.tools.push(
            siumai_core::ToolSpec::new("lookup", None, serde_json::json!({"type": "object"}))
                .expect("tool"),
        );
        let requirements = AnthropicRequestPolicy
            .prepare(
                &ModelId::new(CLAUDE_OPUS_5).expect("model"),
                &request,
                &mut MessagesCallOptions::new(),
            )
            .expect("supported tool changes");
        assert!(
            requirements
                .beta_features()
                .any(|feature| feature == MID_CONVERSATION_TOOL_CHANGES_BETA)
        );

        let error = AnthropicRequestPolicy
            .prepare(
                &ModelId::new(CLAUDE_SONNET_5).expect("model"),
                &request,
                &mut MessagesCallOptions::new(),
            )
            .expect_err("Sonnet 5 does not support mid-conversation system messages");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}
