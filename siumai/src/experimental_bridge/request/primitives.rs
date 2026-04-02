//! Reusable request bridge primitives.

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
                        } else if matches!(message.role, MessageRole::Assistant)
                            && assistant_reasoning_has_openai_encrypted_content(
                                message,
                                part_index,
                            )
                        {
                            "OpenAI Responses preserves assistant reasoning when providerOptions.openai.reasoningEncryptedContent is present"
                        } else if matches!(message.role, MessageRole::Assistant) {
                            "OpenAI Responses drops assistant reasoning unless providerOptions.openai.itemId is present"
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

    for part_index in reasoning_indices {
        if !assistant_reasoning_has_anthropic_replay_metadata(message, *part_index) {
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
    match &message.content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .enumerate()
            .filter_map(|(part_index, part)| {
                part.provider_options()
                    .and_then(|provider_options| provider_options.get_object("anthropic"))
                    .and_then(|anthropic| {
                        anthropic
                            .get("cacheControl")
                            .or_else(|| anthropic.get("cache_control"))
                    })
                    .map(|_| {
                        format!("messages[{message_index}].content[{part_index}].cache_control")
                    })
            })
            .collect(),
        _ => Vec::new(),
    }
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

fn assistant_reasoning_has_openai_item_id(message: &ChatMessage, part_index: usize) -> bool {
    let MessageContent::MultiModal(parts) = &message.content else {
        return false;
    };
    let Some(ContentPart::Reasoning {
        provider_options, ..
    }) = parts.get(part_index)
    else {
        return false;
    };

    provider_options
        .get_object("openai")
        .and_then(|value| value.get("itemId").or_else(|| value.get("item_id")))
        .and_then(|value| value.as_str())
        .is_some()
}

fn assistant_reasoning_has_openai_encrypted_content(
    message: &ChatMessage,
    part_index: usize,
) -> bool {
    let MessageContent::MultiModal(parts) = &message.content else {
        return false;
    };
    let Some(ContentPart::Reasoning {
        provider_options, ..
    }) = parts.get(part_index)
    else {
        return false;
    };

    provider_options
        .get_object("openai")
        .and_then(|value| {
            value
                .get("reasoningEncryptedContent")
                .or_else(|| value.get("reasoning_encrypted_content"))
        })
        .and_then(|value| value.as_str())
        .is_some()
}

fn assistant_reasoning_has_anthropic_replay_metadata(
    message: &ChatMessage,
    part_index: usize,
) -> bool {
    let MessageContent::MultiModal(parts) = &message.content else {
        return false;
    };
    let Some(part) = parts.get(part_index) else {
        return false;
    };
    let ContentPart::Reasoning { .. } = part else {
        return false;
    };

    let provider_options = part
        .provider_options()
        .and_then(|options| options.get_object("anthropic"));

    let has_signature = provider_options
        .and_then(|map| map.get("signature"))
        .and_then(|value| value.as_str())
        .is_some();

    let has_redacted = provider_options
        .and_then(|map| {
            map.get("redactedData")
                .or_else(|| map.get("redacted_data"))
                .or_else(|| map.get("redacted_thinking_data"))
        })
        .and_then(|value| value.as_str())
        .is_some();

    has_signature || has_redacted
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

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::bridge::{BridgeMode, BridgeTarget};
    use siumai_core::types::CacheControl;

    #[test]
    fn anthropic_part_cache_paths_follow_canonical_part_provider_options() {
        let request = ChatRequest::new(vec![
            ChatMessage::user("hi")
                .with_content_parts(vec![ContentPart::text("cached")])
                .cache_control_for_part(1, CacheControl::Ephemeral)
                .build(),
        ]);
        let mut report =
            BridgeReport::new(BridgeTarget::OpenAiChatCompletions, BridgeMode::BestEffort);

        inspect_cache_control_semantics(
            &request,
            RequestTargetCapabilities {
                reasoning_mode: RequestReasoningMode::OpenAiChatCompletions,
                cache_control_mode: RequestCacheControlMode::DropAnthropicControls,
                preserves_tool_approval_responses: false,
            },
            &mut report,
        );

        assert!(report.is_lossy());
        assert_eq!(
            report.dropped_fields,
            vec!["messages[0].content[1].cache_control".to_string()]
        );
    }
}
