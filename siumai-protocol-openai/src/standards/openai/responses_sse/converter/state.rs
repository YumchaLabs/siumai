#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSearchStreamMode {
    OpenAi,
    Xai,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPartsStyle {
    OpenAi,
    Xai,
}

#[derive(Debug, Default, Clone)]
pub(super) struct OpenAiResponsesFunctionCallSerializeState {
    pub(super) item_id: String,
    pub(super) output_index: u64,
    pub(super) name: Option<String>,
    pub(super) arguments: String,
    pub(super) arguments_done: bool,
}

#[derive(Debug, Default, Clone)]
pub(super) struct OpenAiResponsesSerializeState {
    pub(super) response_id: Option<String>,
    pub(super) model_id: Option<String>,
    pub(super) created_at: Option<i64>,
    pub(super) response_created_emitted: bool,
    pub(super) response_completed_emitted: bool,
    pub(super) next_sequence_number: u64,

    pub(super) used_output_indices: std::collections::HashSet<u64>,
    pub(super) next_output_index: u64,

    pub(super) emitted_output_item_added_ids: std::collections::HashSet<String>,
    pub(super) emitted_output_item_done_ids: std::collections::HashSet<String>,

    pub(super) message_item_id: Option<String>,
    pub(super) message_output_index: Option<u64>,
    pub(super) message_content_index: u64,
    pub(super) message_scaffold_emitted: bool,
    pub(super) message_text: String,
    pub(super) message_annotation_index: u64,

    pub(super) reasoning_item_id: Option<String>,
    pub(super) reasoning_output_index: Option<u64>,
    pub(super) reasoning_summary_index: u64,

    pub(super) function_calls_by_call_id:
        std::collections::HashMap<String, OpenAiResponsesFunctionCallSerializeState>,

    /// Stable output indices for provider-hosted tool calls/results when the caller does not
    /// provide an explicit `outputIndex` stream-part field.
    pub(super) provider_tool_output_index_by_tool_call_id: std::collections::HashMap<String, u64>,

    pub(super) latest_usage: Option<crate::types::Usage>,
}
