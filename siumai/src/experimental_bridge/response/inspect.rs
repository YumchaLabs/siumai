//! Response bridge inspection.

use siumai_core::bridge::{BridgeReport, BridgeTarget};
use siumai_core::types::{
    ChatResponse, ContentPart, FinishReason, MessageContent, ToolResultContentPart,
    ToolResultOutput,
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
            ContentPart::Source { .. } if caps.supports_source_parts_as_grounding => {}
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
                tool_name,
                output,
                provider_executed,
                ..
            } if caps.target == BridgeTarget::GeminiGenerateContent
                && *provider_executed == Some(true)
                && tool_name == "code_execution" =>
            {
                inspect_gemini_code_execution_tool_result(&path, output, report);
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
                ContentPart::Text { .. },
            ) => {
                inspect_anthropic_text_part_provider_metadata(path, value, report);
            }
            (
                ResponseContentPartProviderMetadataMode::AnthropicMessages,
                "anthropic",
                ContentPart::ToolCall { .. },
            ) => {
                inspect_anthropic_tool_call_part_provider_metadata(path, value, report);
            }
            (
                ResponseContentPartProviderMetadataMode::GeminiGenerateContent,
                namespace,
                ContentPart::Text { .. }
                | ContentPart::Reasoning { .. }
                | ContentPart::ToolCall { .. }
                | ContentPart::ToolResult { .. },
            ) if namespace == "google" || namespace == "vertex" => {
                inspect_gemini_content_part_provider_metadata(path, namespace, value, report);
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
            (ResponseProviderMetadataMode::GeminiGenerateContent, namespace)
                if namespace == "google" || namespace == "vertex" =>
            {
                inspect_gemini_response_provider_metadata(namespace, metadata, report);
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
                "OpenAI Responses response encoding preserves tool-scoped linkage, source ids, and typed file metadata when provider item call_id is carried, but message citations are still replayed via annotations",
            ),
            _ => report.record_dropped_field(
                format!("provider_metadata.openai.{key}"),
                "OpenAI Responses response encoding does not preserve this OpenAI provider metadata field",
            ),
        }
    }
}

fn inspect_gemini_response_provider_metadata(
    namespace: &str,
    metadata: &std::collections::HashMap<String, serde_json::Value>,
    report: &mut BridgeReport,
) {
    let has_grounding_metadata = metadata.contains_key("groundingMetadata");

    for key in metadata.keys() {
        match key.as_str() {
            "groundingMetadata" => report.record_carried_provider_metadata(
                format!("provider_metadata.{namespace}.groundingMetadata"),
                "Gemini GenerateContent response encoding preserves grounding metadata",
            ),
            "urlContextMetadata" => report.record_carried_provider_metadata(
                format!("provider_metadata.{namespace}.urlContextMetadata"),
                "Gemini GenerateContent response encoding preserves URL context metadata",
            ),
            "promptFeedback" => report.record_carried_provider_metadata(
                format!("provider_metadata.{namespace}.promptFeedback"),
                "Gemini GenerateContent response encoding preserves prompt feedback",
            ),
            "usageMetadata" => report.record_carried_provider_metadata(
                format!("provider_metadata.{namespace}.usageMetadata"),
                "Gemini GenerateContent response encoding preserves native usage metadata",
            ),
            "safetyRatings" => report.record_carried_provider_metadata(
                format!("provider_metadata.{namespace}.safetyRatings"),
                "Gemini GenerateContent response encoding preserves candidate safety ratings",
            ),
            "avgLogprobs" => report.record_carried_provider_metadata(
                format!("provider_metadata.{namespace}.avgLogprobs"),
                "Gemini GenerateContent response encoding preserves average log probabilities",
            ),
            "logprobsResult" => report.record_carried_provider_metadata(
                format!("provider_metadata.{namespace}.logprobsResult"),
                "Gemini GenerateContent response encoding preserves logprobs results",
            ),
            "sources" if has_grounding_metadata => report.record_carried_provider_metadata(
                format!("provider_metadata.{namespace}.sources"),
                "Gemini GenerateContent response encoding can recover normalized sources from preserved grounding metadata",
            ),
            "sources" => report.record_lossy_field(
                format!("provider_metadata.{namespace}.sources"),
                "Gemini GenerateContent response encoding cannot replay derived source lists exactly without grounding metadata",
            ),
            _ => report.record_dropped_field(
                format!("provider_metadata.{namespace}.{key}"),
                "Gemini GenerateContent response encoding does not preserve this Google provider metadata field",
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

    fn citation_payload_groups_from_top_level(
        value: &serde_json::Value,
    ) -> Vec<Vec<serde_json::Value>> {
        value
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|block| block.get("citations").and_then(serde_json::Value::as_array))
            .map(|citations| citations.to_vec())
            .collect()
    }

    fn citation_payload_groups_from_parts(response: &ChatResponse) -> Vec<Vec<serde_json::Value>> {
        let parts = match &response.content {
            MessageContent::MultiModal(parts) => parts,
            _ => return Vec::new(),
        };

        parts
            .iter()
            .filter_map(|part| {
                let ContentPart::Text {
                    provider_metadata: Some(provider_metadata),
                    ..
                } = part
                else {
                    return None;
                };

                provider_metadata
                    .get("anthropic")
                    .and_then(|value| value.get("citations"))
                    .and_then(serde_json::Value::as_array)
                    .map(|citations| citations.to_vec())
            })
            .collect()
    }

    fn serializable_text_part_count(response: &ChatResponse) -> usize {
        match &response.content {
            MessageContent::Text(text) => usize::from(!text.trim().is_empty()),
            MessageContent::MultiModal(parts) => parts
                .iter()
                .filter(|part| matches!(part, ContentPart::Text { .. }))
                .count(),
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(_) => 1,
        }
    }

    fn anthropic_citations_are_exactly_replayable(
        response: &ChatResponse,
        value: &serde_json::Value,
    ) -> bool {
        let top_level = citation_payload_groups_from_top_level(value);
        if top_level.is_empty() {
            return true;
        }

        let part_level = citation_payload_groups_from_parts(response);
        if !part_level.is_empty() {
            return part_level == top_level;
        }

        serializable_text_part_count(response) <= 1
    }

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
            "usage" => report.record_carried_provider_metadata(
                "provider_metadata.anthropic.usage",
                "Anthropic Messages response encoding preserves raw usage metadata when available",
            ),
            "cacheCreationInputTokens" => report.record_carried_provider_metadata(
                "provider_metadata.anthropic.cacheCreationInputTokens",
                "Anthropic Messages response encoding preserves cache-creation token metadata via the raw usage envelope",
            ),
            "container" => report.record_carried_provider_metadata(
                "provider_metadata.anthropic.container",
                "Anthropic Messages response encoding preserves container metadata",
            ),
            "contextManagement" => report.record_carried_provider_metadata(
                "provider_metadata.anthropic.contextManagement",
                "Anthropic Messages response encoding preserves context management metadata",
            ),
            "sources" => report.record_carried_provider_metadata(
                "provider_metadata.anthropic.sources",
                "Anthropic Messages response encoding can reconstruct provider-hosted web sources from serialized tool results",
            ),
            "citations" => {
                let value = metadata
                    .get("citations")
                    .expect("key from metadata iteration should exist");
                if anthropic_citations_are_exactly_replayable(response, value) {
                    report.record_carried_provider_metadata(
                        "provider_metadata.anthropic.citations",
                        "Anthropic Messages response encoding preserves citation payloads and text-block grouping when text-part metadata is available",
                    );
                } else {
                    report.record_lossy_field(
                        "provider_metadata.anthropic.citations",
                        "Anthropic Messages response encoding can only project top-level citations without matching text-part metadata",
                    );
                }
            }
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

fn inspect_gemini_code_execution_tool_result(
    path: &str,
    output: &ToolResultOutput,
    report: &mut BridgeReport,
) {
    match output {
        ToolResultOutput::Json { value } | ToolResultOutput::ErrorJson { value } => {
            let Some(obj) = value.as_object() else {
                report.record_lossy_field(
                    path,
                    "Gemini GenerateContent codeExecutionResult projects non-object tool results into outcome/output fields",
                );
                return;
            };

            if obj.keys().any(|key| key != "outcome" && key != "output") {
                report.record_lossy_field(
                    path,
                    "Gemini GenerateContent codeExecutionResult only preserves `outcome` and `output` fields",
                );
            }
        }
        ToolResultOutput::Text { .. } => report.record_lossy_field(
            path,
            "Gemini GenerateContent codeExecutionResult replays text-only tool results with an unspecified outcome",
        ),
        ToolResultOutput::ErrorText { .. } => report.record_lossy_field(
            path,
            "Gemini GenerateContent codeExecutionResult cannot preserve text-error classification exactly",
        ),
        ToolResultOutput::ExecutionDenied { .. } => report.record_lossy_field(
            path,
            "Gemini GenerateContent codeExecutionResult projects execution denial into generic failed output",
        ),
        ToolResultOutput::Content { value } if tool_result_content_has_binary_like_parts(value) => {
            report.record_lossy_field(
                path,
                "Gemini GenerateContent codeExecutionResult flattens non-text tool result content into a string summary",
            );
        }
        ToolResultOutput::Content { .. } => report.record_lossy_field(
            path,
            "Gemini GenerateContent codeExecutionResult flattens multimodal tool result content into a string summary",
        ),
    }
}

fn tool_result_content_has_binary_like_parts(parts: &[ToolResultContentPart]) -> bool {
    parts
        .iter()
        .any(|part| !matches!(part, ToolResultContentPart::Text { .. }))
}

fn inspect_gemini_content_part_provider_metadata(
    path: &str,
    namespace: &str,
    value: &serde_json::Value,
    report: &mut BridgeReport,
) {
    let Some(metadata) = value.as_object() else {
        report.record_dropped_field(
            format!("{path}.provider_metadata.{namespace}"),
            "Gemini GenerateContent response encoding requires object-shaped content-part provider metadata",
        );
        return;
    };

    for key in metadata.keys() {
        match key.as_str() {
            "thoughtSignature" => report.record_carried_provider_metadata(
                format!("{path}.provider_metadata.{namespace}.thoughtSignature"),
                "Gemini GenerateContent response encoding preserves content-part thought signatures",
            ),
            _ => report.record_dropped_field(
                format!("{path}.provider_metadata.{namespace}.{key}"),
                "Gemini GenerateContent response encoding does not preserve this content-part provider metadata field",
            ),
        }
    }
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
            "serverToolName" => report.record_carried_provider_metadata(
                format!("{path}.provider_metadata.anthropic.serverToolName"),
                "Anthropic Messages response encoding preserves provider-hosted raw server tool names when carried on tool-call metadata",
            ),
            "serverName" => report.record_carried_provider_metadata(
                format!("{path}.provider_metadata.anthropic.serverName"),
                "Anthropic Messages response encoding preserves MCP server names when carried on tool-call metadata",
            ),
            _ => report.record_dropped_field(
                format!("{path}.provider_metadata.anthropic.{key}"),
                "Anthropic Messages response encoding does not preserve this tool-call provider metadata field",
            ),
        }
    }
}

fn inspect_anthropic_text_part_provider_metadata(
    path: &str,
    value: &serde_json::Value,
    report: &mut BridgeReport,
) {
    let Some(metadata) = value.as_object() else {
        report.record_dropped_field(
            format!("{path}.provider_metadata.anthropic"),
            "Anthropic Messages response encoding requires object-shaped text provider metadata",
        );
        return;
    };

    for key in metadata.keys() {
        match key.as_str() {
            "citations" => report.record_carried_provider_metadata(
                format!("{path}.provider_metadata.anthropic.citations"),
                "Anthropic Messages response encoding preserves text-block citations",
            ),
            _ => report.record_dropped_field(
                format!("{path}.provider_metadata.anthropic.{key}"),
                "Anthropic Messages response encoding does not preserve this text-part provider metadata field",
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
