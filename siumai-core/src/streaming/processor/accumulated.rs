use crate::compat::content::ContentPart;
use crate::streaming::processor::{StreamProcessor, ToolCallBuilder};

/// Compact snapshot of stream deltas needed by final response assembly.
#[derive(Debug, Clone, Default)]
pub(super) struct AccumulatedStreamRecord {
    pub(super) text: String,
    pub(super) reasoning: String,
    pub(super) tool_calls: Vec<ToolCallBuilder>,
    has_tool_call_builders: bool,
    pub(super) stream_parts: Vec<ContentPart>,
}

impl AccumulatedStreamRecord {
    pub(super) fn has_accumulated_content(&self) -> bool {
        !self.text.is_empty()
            || self.has_tool_call_builders
            || !self.reasoning.is_empty()
            || !self.stream_parts.is_empty()
    }

    pub(super) fn has_tool_call_builders(&self) -> bool {
        self.has_tool_call_builders
    }
}

impl StreamProcessor {
    pub(super) fn accumulated_stream_record(&self) -> AccumulatedStreamRecord {
        let tool_calls = self
            .tool_call_order
            .iter()
            .filter_map(|id| self.tool_calls.get(id).cloned())
            .collect();

        AccumulatedStreamRecord {
            text: self.buffer.clone(),
            reasoning: self.thinking_buffer.clone(),
            tool_calls,
            has_tool_call_builders: !self.tool_calls.is_empty(),
            stream_parts: self.stream_parts.clone(),
        }
    }
}
