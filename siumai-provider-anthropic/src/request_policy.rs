use siumai_anthropic_compatible::{
    MessagesCallOptions, MessagesRequestPolicy, MessagesRequestRequirements,
};
use siumai_core::{Error, ErrorKind, LanguageRequest, MessageRole, ModelId};
use siumai_protocol_anthropic::messages::{
    AnthropicTool, ContextManagementEdit, InferenceSpeed, ServerFallbacks,
};

use crate::annotations::{AnthropicContentOptions, AnthropicToolOptions};

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

/// Anthropic-owned feature-driven beta contracts.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AnthropicRequestPolicy;

impl MessagesRequestPolicy for AnthropicRequestPolicy {
    fn prepare(
        &self,
        _model: &ModelId,
        request: &LanguageRequest,
        options: &mut MessagesCallOptions,
    ) -> Result<MessagesRequestRequirements, Error> {
        let mut requirements = MessagesRequestRequirements::new();
        if has_mid_conversation_tool_changes(request)? {
            requirements = requirements.with_beta_feature(MID_CONVERSATION_TOOL_CHANGES_BETA)?;
        }
        if let Some(fallbacks) = options.fallbacks() {
            requirements = requirements.with_beta_feature(SERVER_FALLBACK_BETA)?;
            if let ServerFallbacks::Explicit(fallbacks) = fallbacks {
                for fallback in fallbacks {
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

fn has_mid_conversation_tool_changes(request: &LanguageRequest) -> Result<bool, Error> {
    let mut conversation_started = false;
    let mut has_tool_changes = false;
    for message in &request.messages {
        match message.role() {
            MessageRole::System if conversation_started => {
                for part in message.content() {
                    let annotation = part
                        .annotations()
                        .decode::<AnthropicContentOptions>()
                        .map_err(annotation_error)?;
                    has_tool_changes |= annotation
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
    Ok(has_tool_changes)
}

fn annotation_error(source: siumai_core::ProviderAnnotationError) -> Error {
    Error::new(ErrorKind::InvalidInput, "invalid Anthropic annotation").with_source(source)
}

#[cfg(test)]
mod tests {
    use siumai_core::{LanguageRequest, Message, MessageRole, ModelId};
    use siumai_protocol_anthropic::messages::{
        AdvisorToolOptions, AnthropicTool, AnthropicToolReference, CompactionEdit,
        ComputerToolOptions, ContainerSkill, ContextManagement, InferenceGeo, InferenceSpeed,
        McpServer, McpToolsetOptions, MessagesContainer, MidConversationToolChange, OutputEffort,
        ServerFallback, ServerFallbacks, ThinkingConfig, TokenTaskBudget,
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
    fn preparation_collects_current_option_betas_once_for_a_future_model() {
        let mut options = MessagesCallOptions::new()
            .with_speed(InferenceSpeed::Fast)
            .with_task_budget(TokenTaskBudget::new(1_024).expect("task budget"))
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
                &ModelId::new("future-claude").expect("model"),
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
    fn explicit_options_are_not_rejected_from_model_name_predictions() {
        let mut request = LanguageRequest::new(vec![Message::user("caller intent")]);
        request.generation.max_output_tokens = Some(200_000);
        request.generation.temperature = Some(0.4);
        request.generation.top_p = Some(0.5);
        let mut options = MessagesCallOptions::new()
            .with_thinking(ThinkingConfig::enabled(2_048))
            .with_output_effort(OutputEffort::Max)
            .with_task_budget(TokenTaskBudget::new(20_000).expect("task budget"))
            .with_top_k(32)
            .with_speed(InferenceSpeed::Fast)
            .with_inference_geo(InferenceGeo::Us)
            .with_context_management(ContextManagement::new().with_edit(CompactionEdit::new()));

        let requirements = AnthropicRequestPolicy
            .prepare(
                &ModelId::new(crate::models::CLAUDE_HAIKU_4_5).expect("model"),
                &request,
                &mut options,
            )
            .expect("request policy must preserve explicit caller intent");

        assert!(
            requirements
                .beta_features()
                .any(|feature| feature == FAST_MODE_BETA)
        );
        assert!(
            requirements
                .beta_features()
                .any(|feature| feature == COMPACTION_BETA)
        );
    }

    #[test]
    fn mid_conversation_system_message_needs_no_beta() {
        let request = LanguageRequest::new(vec![
            Message::text(MessageRole::User, "start"),
            Message::text(MessageRole::System, "updated policy"),
        ]);
        let requirements = AnthropicRequestPolicy
            .prepare(
                &ModelId::new("future-claude").expect("model"),
                &request,
                &mut MessagesCallOptions::new(),
            )
            .expect("native mid-conversation system message");

        assert!(requirements.beta_features().next().is_none());
    }

    #[test]
    fn future_model_tool_changes_keep_only_their_feature_beta() {
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
                &ModelId::new("future-claude").expect("model"),
                &request,
                &mut MessagesCallOptions::new(),
            )
            .expect("supported tool changes");
        assert_eq!(
            requirements.beta_features().collect::<Vec<_>>(),
            vec![MID_CONVERSATION_TOOL_CHANGES_BETA]
        );
    }
}
