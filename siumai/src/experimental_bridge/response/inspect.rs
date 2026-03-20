//! Response bridge inspection.

use siumai_core::bridge::{BridgeReport, BridgeTarget};
use siumai_core::types::{ChatResponse, ContentPart, FinishReason};

/// Inspect a normalized `ChatResponse` before bridging it into a target protocol.
pub fn inspect_chat_response_bridge(
    response: &ChatResponse,
    target: BridgeTarget,
    report: &mut BridgeReport,
) {
    inspect_response_content(response, target, report);
    inspect_response_usage(response, target, report);
    inspect_response_finish_reason(response, target, report);

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

    if response.system_fingerprint.is_some() && !supports_system_fingerprint(target) {
        report.record_dropped_field(
            "system_fingerprint",
            format!(
                "{} response bridge does not preserve system_fingerprint",
                target.as_str()
            ),
        );
    }

    if response.service_tier.is_some() && !supports_service_tier(target) {
        report.record_dropped_field(
            "service_tier",
            format!(
                "{} response bridge does not preserve service_tier",
                target.as_str()
            ),
        );
    }

    if let Some(provider_metadata) = &response.provider_metadata {
        for namespace in provider_metadata.keys() {
            report.record_dropped_field(
                format!("provider_metadata.{namespace}"),
                format!(
                    "{} response bridge does not serialize top-level provider metadata namespaces",
                    target.as_str()
                ),
            );
        }
    }
}

fn inspect_response_content(
    response: &ChatResponse,
    target: BridgeTarget,
    report: &mut BridgeReport,
) {
    let Some(parts) = response.content.as_multimodal() else {
        return;
    };

    for (index, part) in parts.iter().enumerate() {
        let path = format!("content[{index}]");

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
            } => {
                if provider_metadata
                    .as_ref()
                    .is_some_and(|metadata| !metadata.is_empty())
                {
                    report.record_dropped_field(
                        format!("{path}.provider_metadata"),
                        format!(
                            "{} response bridge does not serialize content-part provider metadata",
                            target.as_str()
                        ),
                    );
                }
            }
            ContentPart::Source { .. }
            | ContentPart::ToolApprovalRequest { .. }
            | ContentPart::ToolApprovalResponse { .. } => {}
        }

        match part {
            ContentPart::Text { .. } | ContentPart::ToolCall { .. } => {}
            ContentPart::Reasoning { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize reasoning blocks",
                        target.as_str()
                    ),
                );
            }
            ContentPart::Image { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize image output parts",
                        target.as_str()
                    ),
                );
            }
            ContentPart::Audio { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize audio output parts",
                        target.as_str()
                    ),
                );
            }
            ContentPart::File { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize file output parts",
                        target.as_str()
                    ),
                );
            }
            ContentPart::Source { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize source citation parts",
                        target.as_str()
                    ),
                );
            }
            ContentPart::ToolResult { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize tool result parts",
                        target.as_str()
                    ),
                );
            }
            ContentPart::ToolApprovalRequest { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize tool approval request parts",
                        target.as_str()
                    ),
                );
            }
            ContentPart::ToolApprovalResponse { .. } => {
                report.record_dropped_field(
                    path,
                    format!(
                        "{} response bridge does not serialize tool approval response parts",
                        target.as_str()
                    ),
                );
            }
        }
    }
}

fn inspect_response_usage(
    response: &ChatResponse,
    target: BridgeTarget,
    report: &mut BridgeReport,
) {
    let Some(usage) = &response.usage else {
        return;
    };

    match target {
        BridgeTarget::OpenAiResponses | BridgeTarget::OpenAiChatCompletions => {}
        BridgeTarget::AnthropicMessages => {
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
        BridgeTarget::GeminiGenerateContent => {
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
    target: BridgeTarget,
    report: &mut BridgeReport,
) {
    let Some(reason) = response.finish_reason.as_ref() else {
        return;
    };

    match target {
        BridgeTarget::OpenAiResponses | BridgeTarget::OpenAiChatCompletions => match reason {
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
        BridgeTarget::AnthropicMessages => match reason {
            FinishReason::StopSequence => report.record_lossy_field(
                "finish_reason",
                "Anthropic Messages response encoding does not preserve the concrete stop sequence value",
            ),
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
        BridgeTarget::GeminiGenerateContent => match reason {
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

const fn supports_system_fingerprint(target: BridgeTarget) -> bool {
    matches!(target, BridgeTarget::OpenAiChatCompletions)
}

const fn supports_service_tier(target: BridgeTarget) -> bool {
    matches!(target, BridgeTarget::OpenAiChatCompletions)
}
