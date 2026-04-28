use super::*;

pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("end_turn") | Some("pause_turn") => Some(FinishReason::Stop),
        Some("max_tokens") | Some("model_context_window_exceeded") => Some(FinishReason::Length),
        Some("tool_use") => Some(FinishReason::ToolCalls),
        Some("stop_sequence") => Some(FinishReason::StopSequence),
        Some("refusal") => Some(FinishReason::ContentFilter),
        Some("compaction") => Some(FinishReason::Other("compaction".to_string())),
        Some(other) => Some(FinishReason::Other(other.to_string())),
        None => None,
    }
}

pub(crate) fn raw_anthropic_stop_reason(reason: Option<&str>) -> Option<&'static str> {
    match reason?.trim() {
        "end_turn" => Some("end_turn"),
        "max_tokens" => Some("max_tokens"),
        "stop_sequence" => Some("stop_sequence"),
        "tool_use" => Some("tool_use"),
        "refusal" => Some("refusal"),
        "pause_turn" => Some("pause_turn"),
        "model_context_window_exceeded" => Some("model_context_window_exceeded"),
        "compaction" => Some("compaction"),
        _ => None,
    }
}

pub(crate) fn finish_reason_to_anthropic_stop_reason(
    reason: Option<&FinishReason>,
) -> Option<&'static str> {
    match reason? {
        FinishReason::Stop => Some("end_turn"),
        FinishReason::Length => Some("max_tokens"),
        FinishReason::ToolCalls => Some("tool_use"),
        FinishReason::ContentFilter => Some("refusal"),
        FinishReason::StopSequence => Some("stop_sequence"),
        FinishReason::Other(value) => raw_anthropic_stop_reason(Some(value.as_str())),
        FinishReason::Error | FinishReason::Unknown => None,
    }
}

pub(crate) fn replay_anthropic_stop_reason(
    raw_reason: Option<&str>,
    finish_reason: Option<&FinishReason>,
) -> Option<&'static str> {
    raw_anthropic_stop_reason(raw_reason)
        .or_else(|| finish_reason_to_anthropic_stop_reason(finish_reason))
}

#[cfg(test)]
mod finish_reason_tests {
    use super::*;

    #[test]
    fn maps_stop_sequence_to_dedicated_finish_reason() {
        assert!(matches!(
            parse_finish_reason(Some("stop_sequence")),
            Some(FinishReason::StopSequence)
        ));
    }

    #[test]
    fn maps_ai_sdk_aligned_length_and_stop_reasons() {
        assert_eq!(
            parse_finish_reason(Some("model_context_window_exceeded")),
            Some(FinishReason::Length)
        );
        assert_eq!(
            parse_finish_reason(Some("pause_turn")),
            Some(FinishReason::Stop)
        );
        assert_eq!(
            parse_finish_reason(Some("compaction")),
            Some(FinishReason::Other("compaction".to_string()))
        );
    }

    #[test]
    fn replays_raw_anthropic_stop_reason_before_unified_projection() {
        assert_eq!(
            replay_anthropic_stop_reason(Some("tool_use"), Some(&FinishReason::Stop)),
            Some("tool_use")
        );
        assert_eq!(
            replay_anthropic_stop_reason(None, Some(&FinishReason::ContentFilter)),
            Some("refusal")
        );
        assert_eq!(
            replay_anthropic_stop_reason(
                Some("content_filter"),
                Some(&FinishReason::ContentFilter)
            ),
            Some("refusal")
        );
    }
}
