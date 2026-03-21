//! Reusable request bridge primitives.

use std::collections::BTreeSet;

use siumai_core::bridge::{BridgeReport, BridgeWarning, BridgeWarningKind};
use siumai_core::types::{ChatMessage, ChatRequest, ContentPart, MessageContent, MessageRole};

use super::target_caps::{
    RequestCacheControlMode, RequestReasoningMode, RequestTargetCapabilities,
};

pub(crate) fn inspect_reasoning_semantics(
    request: &ChatRequest,
    caps: RequestTargetCapabilities,
    report: &mut BridgeReport,
) {
    for (message_index, message) in request.messages.iter().enumerate() {
        let reasoning_indices = reasoning_part_indices(message);
        if reasoning_indices.is_empty() {
            continue;
        }

        match caps.reasoning_mode {
            RequestReasoningMode::OpenAiResponses => {
                for part_index in reasoning_indices {
                    report.record_lossy_field(
                        format!("messages[{message_index}].content[{part_index}].reasoning"),
                        if matches!(message.role, MessageRole::Assistant)
                            && assistant_reasoning_has_openai_item_id(message, part_index)
                            && openai_responses_store_enabled(request)
                        {
                            "OpenAI Responses replays assistant reasoning via item references when store is enabled"
                        } else if matches!(message.role, MessageRole::Assistant)
                            && assistant_reasoning_has_openai_item_id(message, part_index)
                        {
                            "OpenAI Responses normalizes assistant reasoning into reasoning summary items"
                        } else if matches!(message.role, MessageRole::Assistant) {
                            "OpenAI Responses drops assistant reasoning unless providerMetadata.openai.itemId is present"
                        } else {
                            "OpenAI Responses serializes non-assistant reasoning as tagged text input"
                        },
                    );
                }
            }
            RequestReasoningMode::OpenAiChatCompletions => {
                for part_index in reasoning_indices {
                    report.record_lossy_field(
                        format!("messages[{message_index}].content[{part_index}].reasoning"),
                        "OpenAI Chat Completions does not preserve structured reasoning parts",
                    );
                }
            }
            RequestReasoningMode::AnthropicMessages => {
                inspect_anthropic_reasoning_semantics(
                    message,
                    message_index,
                    &reasoning_indices,
                    report,
                );
            }
            RequestReasoningMode::Preserve => {}
        }
    }
}

pub(crate) fn inspect_cache_control_semantics(
    request: &ChatRequest,
    caps: RequestTargetCapabilities,
    report: &mut BridgeReport,
) {
    match caps.cache_control_mode {
        RequestCacheControlMode::AnthropicLimit4 => inspect_anthropic_cache_limit(request, report),
        RequestCacheControlMode::DropAnthropicControls => {
            inspect_non_anthropic_cache_controls(request, report)
        }
    }
}

pub(crate) fn inspect_tool_approval_semantics(
    request: &ChatRequest,
    caps: RequestTargetCapabilities,
    report: &mut BridgeReport,
) {
    for (message_index, message) in request.messages.iter().enumerate() {
        let MessageContent::MultiModal(parts) = &message.content else {
            continue;
        };

        for (part_index, part) in parts.iter().enumerate() {
            match part {
                ContentPart::ToolApprovalRequest { .. } => {
                    record_unsupported_path(
                        report,
                        "tool-approval-request",
                        format!("messages[{message_index}].content[{part_index}]"),
                        "request replay does not preserve pending approval requests",
                    );
                }
                ContentPart::ToolApprovalResponse { .. }
                    if !caps.preserves_tool_approval_responses =>
                {
                    record_unsupported_path(
                        report,
                        "tool-approval-response",
                        format!("messages[{message_index}].content[{part_index}]"),
                        "only OpenAI Responses preserves approval responses in request history",
                    );
                }
                _ => {}
            }
        }
    }
}

fn inspect_anthropic_reasoning_semantics(
    message: &ChatMessage,
    message_index: usize,
    reasoning_indices: &[usize],
    report: &mut BridgeReport,
) {
    if !matches!(message.role, MessageRole::Assistant) {
        for part_index in reasoning_indices {
            report.record_lossy_field(
                format!("messages[{message_index}].content[{part_index}].reasoning"),
                "Anthropic only replays structured thinking blocks on assistant messages",
            );
        }
        return;
    }

    let custom = &message.metadata.custom;
    let signature_global = custom
        .get("anthropic_thinking_signature")
        .and_then(|value| value.as_str());
    let signatures_by_index = custom
        .get("anthropic_thinking_signatures")
        .and_then(|value| value.as_object());
    let redacted_data = custom
        .get("anthropic_redacted_thinking_data")
        .and_then(|value| value.as_str());

    if signature_global.is_some() || redacted_data.is_some() {
        return;
    }

    for part_index in reasoning_indices {
        let has_part_signature = signatures_by_index
            .and_then(|map| map.get(&part_index.to_string()))
            .and_then(|value| value.as_str())
            .is_some();
        if !has_part_signature {
            report.record_lossy_field(
                format!("messages[{message_index}].content[{part_index}].reasoning"),
                "Anthropic assistant thinking blocks require a signature or redacted payload",
            );
        }
    }
}

fn inspect_non_anthropic_cache_controls(request: &ChatRequest, report: &mut BridgeReport) {
    for (message_index, message) in request.messages.iter().enumerate() {
        if message.metadata.cache_control.is_some() {
            report.record_dropped_field(
                format!("messages[{message_index}].metadata.cache_control"),
                "target protocol does not preserve Anthropic-style cache control metadata",
            );
        }

        for path in anthropic_part_cache_paths(message, message_index) {
            report.record_dropped_field(
                path,
                "target protocol does not preserve Anthropic content-level cache control metadata",
            );
        }
    }
}

fn inspect_anthropic_cache_limit(request: &ChatRequest, report: &mut BridgeReport) {
    let mut applied_paths = Vec::new();

    for (message_index, message) in request.messages.iter().enumerate() {
        applied_paths.extend(anthropic_cache_paths(message, message_index));
    }

    for path in applied_paths.into_iter().skip(4) {
        report.record_dropped_field(
            path,
            "Anthropic preserves at most 4 cache breakpoints per request",
        );
    }
}

fn anthropic_cache_paths(message: &ChatMessage, message_index: usize) -> Vec<String> {
    if matches!(message.role, MessageRole::System | MessageRole::Developer) {
        let mut out = Vec::new();
        if message.metadata.cache_control.is_some() {
            out.push(format!("messages[{message_index}].metadata.cache_control"));
        }
        return out;
    }

    let mut out = anthropic_part_cache_paths(message, message_index);
    if message.metadata.cache_control.is_some() {
        out.push(format!("messages[{message_index}].metadata.cache_control"));
    }
    out
}

fn anthropic_part_cache_paths(message: &ChatMessage, message_index: usize) -> Vec<String> {
    let mut indices = BTreeSet::new();
    let part_count = message_part_count(message);

    if let Some(obj) = message
        .metadata
        .custom
        .get("anthropic_content_cache_controls")
        .and_then(|value| value.as_object())
    {
        for key in obj.keys() {
            let Ok(index) = key.parse::<usize>() else {
                continue;
            };
            if index < part_count {
                indices.insert(index);
            }
        }
    }

    if let Some(values) = message
        .metadata
        .custom
        .get("anthropic_content_cache_indices")
        .and_then(|value| value.as_array())
    {
        for value in values {
            let Some(index) = value.as_u64().and_then(|raw| usize::try_from(raw).ok()) else {
                continue;
            };
            if index < part_count {
                indices.insert(index);
            }
        }
    }

    indices
        .into_iter()
        .map(|part_index| format!("messages[{message_index}].content[{part_index}].cache_control"))
        .collect()
}

fn record_unsupported_path(
    report: &mut BridgeReport,
    capability: &str,
    path: String,
    message: &str,
) {
    report.unsupported_capabilities.push(capability.to_string());
    report.add_warning(BridgeWarning::with_path(
        BridgeWarningKind::UnsupportedCapability,
        path,
        message,
    ));
}

fn reasoning_part_indices(message: &ChatMessage) -> Vec<usize> {
    match &message.content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .enumerate()
            .filter_map(|(index, part)| {
                matches!(part, ContentPart::Reasoning { .. }).then_some(index)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn message_part_count(message: &ChatMessage) -> usize {
    match &message.content {
        MessageContent::Text(_) => 1,
        MessageContent::MultiModal(parts) => parts.len(),
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(_) => 1,
    }
}

fn assistant_reasoning_has_openai_item_id(message: &ChatMessage, part_index: usize) -> bool {
    let MessageContent::MultiModal(parts) = &message.content else {
        return false;
    };
    let Some(ContentPart::Reasoning {
        provider_metadata, ..
    }) = parts.get(part_index)
    else {
        return false;
    };

    provider_metadata
        .as_ref()
        .and_then(|map| map.get("openai"))
        .and_then(|value| value.get("itemId").or_else(|| value.get("item_id")))
        .and_then(|value| value.as_str())
        .is_some()
}

fn openai_responses_store_enabled(request: &ChatRequest) -> bool {
    let openai = request.provider_options_map.get_object("openai");
    let store = openai
        .and_then(|map| map.get("store"))
        .and_then(|value| value.as_bool())
        .or_else(|| {
            openai
                .and_then(|map| map.get("responsesApi").or_else(|| map.get("responses_api")))
                .and_then(|value| value.as_object())
                .and_then(|map| map.get("store"))
                .and_then(|value| value.as_bool())
        });

    store != Some(false)
}
