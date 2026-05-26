use crate::streaming::ChatStreamEvent;
use crate::types::{ChatStreamOpenAiResponsesReplay, ChatStreamReplay};

/// Private OpenAI Responses replay hints attached to stable stream parts.
#[derive(Debug, Default, Clone)]
pub(super) struct OpenAiResponsesEventExtras {
    pub(super) output_index: Option<u64>,
    pub(super) raw_item: Option<serde_json::Value>,
}

pub(super) fn attach_event_extras(
    event: ChatStreamEvent,
    extras: OpenAiResponsesEventExtras,
) -> ChatStreamEvent {
    let Some(replay) = ChatStreamReplay::openai_responses(extras.output_index, extras.raw_item)
    else {
        return event;
    };

    match event {
        ChatStreamEvent::Part { part } => ChatStreamEvent::PartWithReplay { part, replay },
        other => other,
    }
}

pub(super) fn openai_responses_replay(
    event: &ChatStreamEvent,
) -> Option<&ChatStreamOpenAiResponsesReplay> {
    event
        .replay_ref()
        .and_then(ChatStreamReplay::openai_responses_ref)
}

pub(super) fn apply_event_replay_to_custom_event(
    source: &ChatStreamEvent,
    custom_event: &mut ChatStreamEvent,
) {
    let Some(replay) = openai_responses_replay(source) else {
        return;
    };
    let ChatStreamEvent::Custom { data, .. } = custom_event else {
        return;
    };
    let Some(obj) = data.as_object_mut() else {
        return;
    };

    if let Some(output_index) = replay.output_index {
        obj.insert("outputIndex".to_string(), serde_json::json!(output_index));
    }
    if let Some(raw_item) = replay.raw_item.clone() {
        obj.insert("rawItem".to_string(), raw_item);
    }
}
