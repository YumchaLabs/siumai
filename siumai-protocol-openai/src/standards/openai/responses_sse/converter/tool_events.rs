use super::*;

use crate::streaming::{
    LanguageModelV3StreamPart, LanguageModelV3ToolApprovalRequest, LanguageModelV3ToolCall,
    LanguageModelV3ToolResult, SharedV3ProviderMetadata,
};

fn normalize_tool_input_value(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s,
        serde_json::Value::Null => String::new(),
        other => serde_json::to_string(&other).unwrap_or_else(|_| "{}".to_string()),
    }
}

fn into_provider_metadata(value: Option<serde_json::Value>) -> Option<SharedV3ProviderMetadata> {
    match value {
        Some(serde_json::Value::Object(map)) => Some(map.into_iter().collect()),
        _ => None,
    }
}

impl OpenAiResponsesEventConverter {
    fn openai_stream_part_event(
        &self,
        part: LanguageModelV3StreamPart,
    ) -> crate::types::ChatStreamEvent {
        part.to_part_event()
    }

    fn attach_event_extras(
        &self,
        event: crate::types::ChatStreamEvent,
        extras: OpenAiResponsesEventExtras,
    ) -> crate::types::ChatStreamEvent {
        let Some(replay) =
            crate::types::ChatStreamReplay::openai_responses(extras.output_index, extras.raw_item)
        else {
            return event;
        };

        match event {
            crate::types::ChatStreamEvent::Part { part } => {
                crate::types::ChatStreamEvent::PartWithReplay { part, replay }
            }
            other => other,
        }
    }

    pub(super) fn openai_tool_input_start_event(
        &self,
        id: &str,
        tool_name: &str,
        provider_executed: Option<bool>,
        dynamic: Option<bool>,
        title: Option<String>,
        provider_metadata: Option<serde_json::Value>,
    ) -> crate::types::ChatStreamEvent {
        self.openai_stream_part_event(LanguageModelV3StreamPart::ToolInputStart {
            id: id.to_string(),
            tool_name: tool_name.to_string(),
            provider_metadata: into_provider_metadata(provider_metadata),
            provider_executed,
            dynamic,
            title,
        })
    }

    pub(super) fn openai_tool_input_delta_event(
        &self,
        id: &str,
        delta: &str,
    ) -> crate::types::ChatStreamEvent {
        self.openai_stream_part_event(LanguageModelV3StreamPart::ToolInputDelta {
            id: id.to_string(),
            delta: delta.to_string(),
            provider_metadata: None,
        })
    }

    pub(super) fn openai_tool_input_end_event(
        &self,
        id: &str,
        provider_metadata: Option<serde_json::Value>,
    ) -> crate::types::ChatStreamEvent {
        self.openai_stream_part_event(LanguageModelV3StreamPart::ToolInputEnd {
            id: id.to_string(),
            provider_metadata: into_provider_metadata(provider_metadata),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn openai_tool_call_event(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        input: serde_json::Value,
        provider_executed: Option<bool>,
        dynamic: Option<bool>,
        provider_metadata: Option<serde_json::Value>,
        extras: OpenAiResponsesEventExtras,
    ) -> crate::types::ChatStreamEvent {
        let event = self.openai_stream_part_event(LanguageModelV3StreamPart::ToolCall(
            LanguageModelV3ToolCall {
                tool_call_id: tool_call_id.to_string(),
                tool_name: tool_name.to_string(),
                input: normalize_tool_input_value(input),
                provider_executed,
                dynamic,
                provider_metadata: into_provider_metadata(provider_metadata),
            },
        ));
        self.attach_event_extras(event, extras)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn openai_tool_result_event(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        result: serde_json::Value,
        _provider_executed: Option<bool>,
        dynamic: Option<bool>,
        is_error: Option<bool>,
        provider_metadata: Option<serde_json::Value>,
        extras: OpenAiResponsesEventExtras,
    ) -> crate::types::ChatStreamEvent {
        self.openai_tool_result_event_with_preliminary(
            tool_call_id,
            tool_name,
            result,
            dynamic,
            is_error,
            None,
            provider_metadata,
            extras,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn openai_tool_result_event_with_preliminary(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        result: serde_json::Value,
        dynamic: Option<bool>,
        is_error: Option<bool>,
        preliminary: Option<bool>,
        provider_metadata: Option<serde_json::Value>,
        extras: OpenAiResponsesEventExtras,
    ) -> crate::types::ChatStreamEvent {
        let event = self.openai_stream_part_event(LanguageModelV3StreamPart::ToolResult(
            LanguageModelV3ToolResult {
                tool_call_id: tool_call_id.to_string(),
                tool_name: tool_name.to_string(),
                result,
                is_error,
                preliminary,
                dynamic,
                provider_metadata: into_provider_metadata(provider_metadata),
            },
        ));
        self.attach_event_extras(event, extras)
    }

    pub(super) fn openai_tool_approval_request_event(
        &self,
        approval_id: &str,
        tool_call_id: &str,
        provider_metadata: Option<serde_json::Value>,
        extras: OpenAiResponsesEventExtras,
    ) -> crate::types::ChatStreamEvent {
        let event = self.openai_stream_part_event(LanguageModelV3StreamPart::ToolApprovalRequest(
            LanguageModelV3ToolApprovalRequest {
                approval_id: approval_id.to_string(),
                tool_call_id: tool_call_id.to_string(),
                provider_metadata: into_provider_metadata(provider_metadata),
            },
        ));
        self.attach_event_extras(event, extras)
    }
}
