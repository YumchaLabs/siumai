//! Response bridge inspection.

use siumai_core::bridge::{BridgeReport, BridgeTarget};
use siumai_core::types::{
    ChatResponse, ContentPart, FinishReason, ToolResultContentPart, ToolResultOutput,
};

use super::target_caps::{
    ResponseContentPartProviderMetadataMode, ResponseFinishReasonMode,
    ResponseProviderMetadataMode, ResponseTargetCapabilities, ResponseUsageMode,
    response_target_capabilities,
};

/// Inspect a normalized `ChatResponse` before bridging it into a target protocol.
pub fn inspect_chat_response_bridge(
    response: &ChatResponse,
    target: BridgeTarget,
    report: &mut BridgeReport,
) {
    let caps = response_target_capabilities(target);

    inspect_response_content(response, caps, report);
    inspect_response_usage(response, caps, report);
    inspect_response_finish_reason(response, caps, report);
    inspect_response_provider_metadata(response, caps, report);

    if response.audio.is_some() {
        report.record_dropped_field(
            "audio",
            format!(
                "{} response bridge does not preserve audio outputs",
                target.as_str()
            ),
        );
    }

    if response
        .warnings
        .as_ref()
        .is_some_and(|warnings| !warnings.is_empty())
    {
        report.record_dropped_field(
            "warnings",
            format!(
                "{} response bridge does not serialize provider warnings",
                target.as_str()
            ),
        );
    }

    if response.system_fingerprint.is_some() && !caps.supports_system_fingerprint {
        report.record_dropped_field(
            "system_fingerprint",
            format!(
                "{} response bridge does not preserve system_fingerprint",
                caps.target.as_str()
            ),
        );
    }

    if response.service_tier.is_some() && !caps.supports_service_tier {
        report.record_dropped_field(
            "service_tier",
            format!(
                "{} response bridge does not preserve service_tier",
                caps.target.as_str()
            ),
        );
    }
}

fn inspect_response_content(
    response: &ChatResponse,
    caps: ResponseTargetCapabilities,
    report: &mut BridgeReport,
) {
    let Some(parts) = response.content.as_multimodal() else {
        return;
    };

    for (index, part) in parts.iter().enumerate() {
        let path = format!("content[{index}]");
        inspect_response_content_part_provider_metadata(part, &path, caps, report);

        match part {
            ContentPart::Text { .. } | ContentPart::ToolCall { .. } => {}
            ContentPart::Reasoning { .. } => {
                if !caps.supports_reasoning_blocks {
                    report.record_dropped_field(
                        path,
                        format!(
                            "{} response bridge does not serialize reasoning blocks",
                            caps.target.as_str()
                        ),
                    );
                }
            }
            ContentPart::Image { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize image output parts",
                        caps.target.as_str()
                    ),
                );
            }
            ContentPart::Audio { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize audio output parts",
                        caps.target.as_str()
                    ),
                );
            }
            ContentPart::File { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize file output parts",
                        caps.target.as_str()
                    ),
                );
            }
            ContentPart::Source { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize source citation parts",
                        caps.target.as_str()
                    ),
                );
            }
            ContentPart::ToolResult {
                output,
                provider_executed,
                ..
            } if caps.supports_provider_executed_tool_results
                && *provider_executed == Some(true) =>
            {
                inspect_openai_response_tool_result(&path, output, report);
            }
            ContentPart::ToolResult { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize tool result parts",
                        caps.target.as_str()
                    ),
                );
            }
            ContentPart::ToolApprovalRequest { .. } if caps.supports_tool_approval_requests => {}
            ContentPart::ToolApprovalRequest { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize tool approval request parts",
                        caps.target.as_str()
                    ),
                );
            }
            ContentPart::ToolApprovalResponse { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize tool approval response parts",
                        caps.target.as_str()
                    ),
                );
            }
        }
    }
}

fn inspect_response_content_part_provider_metadata(
    part: &ContentPart,
    path: &str,
    caps: ResponseTargetCapabilities,
    report: &mut BridgeReport,
) {
    let Some(provider_metadata) = content_part_provider_metadata(part) else {
        return;
    };

    for (namespace, value) in provider_metadata {
        match (
            caps.content_part_provider_metadata_mode,
            namespace.as_str(),
            part,
        ) {
            (
                ResponseContentPartProviderMetadataMode::OpenAiResponses,
                "openai",
                ContentPart::Reasoning { .. },
            ) => {
                inspect_openai_reasoning_part_provider_metadata(path, value, report);
            }
            (
                ResponseContentPartProviderMetadataMode::OpenAiResponses,
                "openai",
                ContentPart::ToolCall { .. },
            ) => {
                inspect_openai_tool_call_part_provider_metadata(path, value, report);
            }
            (
                ResponseContentPartProviderMetadataMode::OpenAiResponses,
                "openai",
                ContentPart::ToolResult { .. },
            ) => {
                inspect_openai_tool_result_part_provider_metadata(path, value, report);
            }
            (
                ResponseContentPartProviderMetadataMode::AnthropicMessages,
                "anthropic",
                ContentPart::ToolCall { .. },
            ) => {
                inspect_anthropic_tool_call_part_provider_metadata(path, value, report);
            }
            _ => {
                report.record_dropped_field(
                    format!("{path}.provider_metadata.{namespace}"),
                    format!(
                        "{} response bridge does not serialize this content-part provider metadata namespace",
                        caps.target.as_str()
                    ),
                );
            }
        }
    }
}

fn inspect_response_usage(
    response: &ChatResponse,
    caps: ResponseTargetCapabilities,
    report: &mut BridgeReport,
) {
    let Some(usage) = &response.usage else {
        return;
    };

    match caps.usage_mode {
        ResponseUsageMode::PreserveAll => {}
        ResponseUsageMode::AnthropicAggregateOnly => {
            if usage.prompt_tokens_details.is_some() {
                report.record_lossy_field(
                    "usage.prompt_tokens_details",
                    "Anthropic Messages response encoding only preserves aggregate input/output token counts",
                );
            }
            if usage.completion_tokens_details.is_some() {
                report.record_lossy_field(
                    "usage.completion_tokens_details",
                    "Anthropic Messages response encoding only preserves aggregate input/output token counts",
                );
            }
        }
        ResponseUsageMode::GeminiPartialBreakdown => {
            if usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.audio_tokens)
                .is_some()
            {
                report.record_lossy_field(
                    "usage.prompt_tokens_details.audio_tokens",
                    "Gemini GenerateContent response encoding does not preserve prompt audio token breakdown",
                );
            }
            if usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .is_some()
            {
                report.record_lossy_field(
                    "usage.prompt_tokens_details.cached_tokens",
                    "Gemini GenerateContent response encoding does not preserve cached token breakdown",
                );
            }
            if usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.audio_tokens)
                .is_some()
            {
                report.record_lossy_field(
                    "usage.completion_tokens_details.audio_tokens",
                    "Gemini GenerateContent response encoding does not preserve completion audio token breakdown",
                );
            }
            if usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.accepted_prediction_tokens)
                .is_some()
            {
                report.record_lossy_field(
                    "usage.completion_tokens_details.accepted_prediction_tokens",
                    "Gemini GenerateContent response encoding does not preserve accepted prediction tokens",
                );
            }
            if usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.rejected_prediction_tokens)
                .is_some()
            {
                report.record_lossy_field(
                    "usage.completion_tokens_details.rejected_prediction_tokens",
                    "Gemini GenerateContent response encoding does not preserve rejected prediction tokens",
                );
            }
        }
    }
}

fn inspect_response_finish_reason(
    response: &ChatResponse,
    caps: ResponseTargetCapabilities,
    report: &mut BridgeReport,
) {
    let Some(reason) = response.finish_reason.as_ref() else {
        return;
    };

    match caps.finish_reason_mode {
        ResponseFinishReasonMode::OpenAiFamily => match reason {
            FinishReason::StopSequence => report.record_lossy_field(
                "finish_reason",
                "OpenAI response encoders collapse stop-sequence termination into `stop`",
            ),
            FinishReason::Unknown | FinishReason::Other(_) => report.record_lossy_field(
                "finish_reason",
                "OpenAI response encoders cannot preserve unknown finish reasons",
            ),
            _ => {}
        },
        ResponseFinishReasonMode::AnthropicMessages => match reason {
            FinishReason::StopSequence if anthropic_stop_sequence(response).is_none() => {
                report.record_lossy_field(
                    "finish_reason",
                    "Anthropic Messages response encoding does not preserve the concrete stop sequence value",
                )
            }
            FinishReason::ContentFilter => report.record_lossy_field(
                "finish_reason",
                "Anthropic Messages response encoding downgrades content filtering into `stop_sequence`",
            ),
            FinishReason::Error => report.record_lossy_field(
                "finish_reason",
                "Anthropic Messages response encoding downgrades errors into `end_turn`",
            ),
            FinishReason::Unknown | FinishReason::Other(_) => report.record_lossy_field(
                "finish_reason",
                "Anthropic Messages response encoding cannot preserve unknown finish reasons",
            ),
            _ => {}
        },
        ResponseFinishReasonMode::GeminiGenerateContent => match reason {
            FinishReason::StopSequence
            | FinishReason::ToolCalls
            | FinishReason::Error
            | FinishReason::Unknown
            | FinishReason::Other(_) => report.record_lossy_field(
                "finish_reason",
                "Gemini GenerateContent response encoding collapses this finish reason into a generic STOP state",
            ),
            _ => {}
        },
    }
}

fn inspect_response_provider_metadata(
    response: &ChatResponse,
    caps: ResponseTargetCapabilities,
    report: &mut BridgeReport,
) {
    let Some(provider_metadata) = &response.provider_metadata else {
        return;
    };

    for (namespace, metadata) in provider_metadata {
        match (caps.provider_metadata_mode, namespace.as_str()) {
            (ResponseProviderMetadataMode::OpenAiResponses, "openai") => {
                inspect_openai_response_provider_metadata(metadata, report);
            }
            (ResponseProviderMetadataMode::AnthropicMessages, "anthropic") => {
                inspect_anthropic_response_provider_metadata(response, metadata, report);
            }
            _ => {
                report.record_dropped_field(
                    format!("provider_metadata.{namespace}"),
                    format!(
                        "{} response bridge does not serialize top-level provider metadata namespaces",
                        caps.target.as_str()
                    ),
                );
            }
        }
    }
}

fn inspect_openai_response_provider_metadata(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
    report: &mut BridgeReport,
) {
    for key in metadata.keys() {
        match key.as_str() {
            "itemId" => report.record_carried_provider_metadata(
                "provider_metadata.openai.itemId",
                "OpenAI Responses response encoding preserves the assistant message item id",
            ),
            "sources" => report.record_lossy_field(
                "provider_metadata.openai.sources",
                "OpenAI Responses response encoding preserves tool linkage and typed file metadata where possible, but source ids are still regenerated and message citations are replayed via annotations",
            ),
            _ => report.record_dropped_field(
                format!("provider_metadata.openai.{key}"),
                "OpenAI Responses response encoding does not preserve this OpenAI provider metadata field",
            ),
        }
    }
}

fn inspect_anthropic_response_provider_metadata(
    response: &ChatResponse,
    metadata: &std::collections::HashMap<String, serde_json::Value>,
    report: &mut BridgeReport,
) {
    let has_reasoning = response_has_reasoning(response);

    for key in metadata.keys() {
        match key.as_str() {
            "thinking_signature" if has_reasoning => report.record_carried_provider_metadata(
                "provider_metadata.anthropic.thinking_signature",
                "Anthropic Messages response encoding replays thinking signatures on assistant thinking blocks",
            ),
            "thinking_signature" => report.record_dropped_field(
                "provider_metadata.anthropic.thinking_signature",
                "Anthropic Messages response encoding cannot replay a thinking signature without a reasoning block",
            ),
            "redacted_thinking_data" => report.record_carried_provider_metadata(
                "provider_metadata.anthropic.redacted_thinking_data",
                "Anthropic Messages response encoding replays redacted thinking blocks",
            ),
            "stopSequence" => report.record_carried_provider_metadata(
                "provider_metadata.anthropic.stopSequence",
                "Anthropic Messages response encoding preserves stop_sequence",
            ),
            _ => report.record_dropped_field(
                format!("provider_metadata.anthropic.{key}"),
                "Anthropic Messages response encoding does not preserve this Anthropic provider metadata field",
            ),
        }
    }
}

fn inspect_openai_reasoning_part_provider_metadata(
    path: &str,
    value: &serde_json::Value,
    report: &mut BridgeReport,
) {
    let Some(metadata) = value.as_object() else {
        report.record_dropped_field(
            format!("{path}.provider_metadata.openai"),
            "OpenAI Responses response encoding requires object-shaped reasoning provider metadata",
        );
        return;
    };

    for key in metadata.keys() {
        match key.as_str() {
            "itemId" => report.record_carried_provider_metadata(
                format!("{path}.provider_metadata.openai.itemId"),
                "OpenAI Responses response encoding preserves reasoning item ids",
            ),
            "reasoningEncryptedContent" => report.record_carried_provider_metadata(
                format!("{path}.provider_metadata.openai.reasoningEncryptedContent"),
                "OpenAI Responses response encoding preserves encrypted reasoning payloads",
            ),
            _ => report.record_dropped_field(
                format!("{path}.provider_metadata.openai.{key}"),
                "OpenAI Responses response encoding does not preserve this reasoning provider metadata field",
            ),
        }
    }
}

fn inspect_openai_tool_call_part_provider_metadata(
    path: &str,
    value: &serde_json::Value,
    report: &mut BridgeReport,
) {
    let Some(metadata) = value.as_object() else {
        report.record_dropped_field(
            format!("{path}.provider_metadata.openai"),
            "OpenAI Responses response encoding requires object-shaped tool-call provider metadata",
        );
        return;
    };

    for key in metadata.keys() {
        match key.as_str() {
            "itemId" => report.record_carried_provider_metadata(
                format!("{path}.provider_metadata.openai.itemId"),
                "OpenAI Responses response encoding preserves function and provider tool item ids",
            ),
            _ => report.record_dropped_field(
                format!("{path}.provider_metadata.openai.{key}"),
                "OpenAI Responses response encoding does not preserve this tool-call provider metadata field",
            ),
        }
    }
}

fn inspect_openai_tool_result_part_provider_metadata(
    path: &str,
    value: &serde_json::Value,
    report: &mut BridgeReport,
) {
    let Some(metadata) = value.as_object() else {
        report.record_dropped_field(
            format!("{path}.provider_metadata.openai"),
            "OpenAI Responses response encoding requires object-shaped tool-result provider metadata",
        );
        return;
    };

    for key in metadata.keys() {
        match key.as_str() {
            "itemId" => report.record_carried_provider_metadata(
                format!("{path}.provider_metadata.openai.itemId"),
                "OpenAI Responses response encoding preserves provider tool item ids on tool results",
            ),
            _ => report.record_dropped_field(
                format!("{path}.provider_metadata.openai.{key}"),
                "OpenAI Responses response encoding does not preserve this tool-result provider metadata field",
            ),
        }
    }
}

fn inspect_openai_response_tool_result(
    path: &str,
    output: &ToolResultOutput,
    report: &mut BridgeReport,
) {
    match output {
        ToolResultOutput::ExecutionDenied { .. } => report.record_lossy_field(
            path,
            "OpenAI Responses response encoding replays execution-denied tool results as generic custom tool output",
        ),
        ToolResultOutput::Content { value } if tool_result_content_has_binary_like_parts(value) => {
            report.record_lossy_field(
                path,
                "OpenAI Responses response encoding flattens non-text tool result content into coarse placeholders",
            );
        }
        _ => {}
    }
}

fn tool_result_content_has_binary_like_parts(parts: &[ToolResultContentPart]) -> bool {
    parts
        .iter()
        .any(|part| !matches!(part, ToolResultContentPart::Text { .. }))
}

fn inspect_anthropic_tool_call_part_provider_metadata(
    path: &str,
    value: &serde_json::Value,
    report: &mut BridgeReport,
) {
    let Some(metadata) = value.as_object() else {
        report.record_dropped_field(
            format!("{path}.provider_metadata.anthropic"),
            "Anthropic Messages response encoding requires object-shaped tool-call provider metadata",
        );
        return;
    };

    for key in metadata.keys() {
        match key.as_str() {
            "caller" => report.record_carried_provider_metadata(
                format!("{path}.provider_metadata.anthropic.caller"),
                "Anthropic Messages response encoding preserves tool caller metadata",
            ),
            _ => report.record_dropped_field(
                format!("{path}.provider_metadata.anthropic.{key}"),
                "Anthropic Messages response encoding does not preserve this tool-call provider metadata field",
            ),
        }
    }
}

fn content_part_provider_metadata(
    part: &ContentPart,
) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
    match part {
        ContentPart::Text {
            provider_metadata, ..
        }
        | ContentPart::Image {
            provider_metadata, ..
        }
        | ContentPart::Audio {
            provider_metadata, ..
        }
        | ContentPart::File {
            provider_metadata, ..
        }
        | ContentPart::ToolCall {
            provider_metadata, ..
        }
        | ContentPart::ToolResult {
            provider_metadata, ..
        }
        | ContentPart::Reasoning {
            provider_metadata, ..
        } => provider_metadata.as_ref(),
        ContentPart::Source { .. }
        | ContentPart::ToolApprovalRequest { .. }
        | ContentPart::ToolApprovalResponse { .. } => None,
    }
}

fn response_has_reasoning(response: &ChatResponse) -> bool {
    response.content.as_multimodal().is_some_and(|parts| {
        parts
            .iter()
            .any(|part| matches!(part, ContentPart::Reasoning { .. }))
    })
}

fn anthropic_stop_sequence(response: &ChatResponse) -> Option<&str> {
    response
        .provider_metadata
        .as_ref()?
        .get("anthropic")?
        .get("stopSequence")?
        .as_str()
}
