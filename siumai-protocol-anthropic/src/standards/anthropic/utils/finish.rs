use super::*;

pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("end_turn") => Some(FinishReason::Stop),
        Some("max_tokens") => Some(FinishReason::Length),
        Some("tool_use") => Some(FinishReason::ToolCalls),
        Some("stop_sequence") => Some(FinishReason::StopSequence),
        Some("pause_turn") => Some(FinishReason::Other("pause_turn".to_string())),
        Some("refusal") => Some(FinishReason::ContentFilter),
        Some(other) => Some(FinishReason::Other(other.to_string())),
        None => None,
    }
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
}
