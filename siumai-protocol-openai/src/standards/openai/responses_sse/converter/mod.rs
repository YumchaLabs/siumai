use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use chrono::{SecondsFormat, TimeZone, Utc};

mod state;
use state::OpenAiResponsesSerializeState;
pub use state::{StreamPartsStyle, WebSearchStreamMode};

mod apply_patch;
mod code_interpreter;
mod convert;
mod custom_tools;
mod function_tool;
mod mcp;
mod pending;
mod provider_tools;
mod reasoning;
mod stream_meta;

mod serialize;
mod sse;

/// OpenAI Responses SSE event converter using unified streaming utilities
#[derive(Clone)]
pub struct OpenAiResponsesEventConverter {
    function_call_ids_by_output_index: Arc<Mutex<HashMap<u64, String>>>,
    function_call_meta_by_item_id: Arc<Mutex<HashMap<String, (String, String)>>>,
    emitted_function_tool_input_start_ids: Arc<Mutex<HashSet<String>>>,
    emitted_function_tool_input_end_ids: Arc<Mutex<HashSet<String>>>,
    provider_tool_name_by_item_type: Arc<Mutex<HashMap<String, String>>>,
    mcp_calls_by_item_id: Arc<Mutex<HashMap<String, (String, String)>>>,
    mcp_call_args_by_item_id: Arc<Mutex<HashMap<String, String>>>,
    emitted_mcp_call_ids: Arc<Mutex<HashSet<String>>>,
    emitted_mcp_result_ids: Arc<Mutex<HashSet<String>>>,
    mcp_approval_tool_call_id_by_approval_id: Arc<Mutex<HashMap<String, String>>>,
    next_mcp_approval_tool_call_index: Arc<Mutex<u64>>,
    emitted_mcp_approval_request_ids: Arc<Mutex<HashSet<String>>>,
    reasoning_encrypted_content_by_item_id: Arc<Mutex<HashMap<String, Option<String>>>>,
    emitted_reasoning_start_ids: Arc<Mutex<HashSet<String>>>,
    emitted_reasoning_end_ids: Arc<Mutex<HashSet<String>>>,
    apply_patch_call_id_by_item_id: Arc<Mutex<HashMap<String, String>>>,
    apply_patch_operation_by_item_id: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    emitted_apply_patch_tool_input_start_ids: Arc<Mutex<HashSet<String>>>,
    emitted_apply_patch_tool_input_end_ids: Arc<Mutex<HashSet<String>>>,
    code_interpreter_container_id_by_item_id: Arc<Mutex<HashMap<String, String>>>,
    emitted_code_interpreter_tool_input_start_ids: Arc<Mutex<HashSet<String>>>,
    emitted_code_interpreter_tool_input_end_ids: Arc<Mutex<HashSet<String>>>,
    emitted_web_search_tool_input_ids: Arc<Mutex<HashSet<String>>>,
    emitted_stream_start: Arc<Mutex<bool>>,
    emitted_response_metadata: Arc<Mutex<HashSet<String>>>,
    created_response_id: Arc<Mutex<Option<String>>>,
    created_model_id: Arc<Mutex<Option<String>>>,
    created_created_at: Arc<Mutex<Option<i64>>>,
    message_item_id_by_output_index: Arc<Mutex<HashMap<u64, String>>>,
    emitted_text_start_ids: Arc<Mutex<HashSet<String>>>,
    emitted_text_end_ids: Arc<Mutex<HashSet<String>>>,
    text_annotations_by_item_id: Arc<Mutex<HashMap<String, Vec<serde_json::Value>>>>,
    pending_stream_end_events: Arc<Mutex<VecDeque<crate::streaming::ChatStreamEvent>>>,

    /// Default tool input used for web search calls when the output item does not contain
    /// an explicit arguments payload.
    web_search_default_input: String,
    /// Whether to include `providerExecuted: true` on `tool-input-start` events for web search.
    include_web_search_provider_executed_in_tool_input: bool,
    /// Whether to emit `tool-input-delta` for web search calls (some vendors expect it).
    emit_web_search_tool_input_delta: bool,
    /// Whether to emit `tool-result` for web search calls (OpenAI does, xAI does not).
    emit_web_search_tool_result: bool,

    /// Controls the Vercel stream parts shape (ids / providerMetadata) emitted by this converter.
    stream_parts_style: StreamPartsStyle,

    /// Controls how the final `response.completed` payload is transformed into `ChatResponse`.
    responses_transform_style: super::super::transformers::ResponsesTransformStyle,

    /// Controls the providerMetadata key used in Vercel-aligned stream parts
    /// (e.g. "openai" vs "azure").
    provider_metadata_key: String,

    /// Maps custom tool call names (e.g. xAI internal tool names) to the user-facing tool name.
    custom_tool_name_by_call_name: Arc<Mutex<HashMap<String, String>>>,
    custom_tool_call_name_by_item_id: Arc<Mutex<HashMap<String, String>>>,
    custom_tool_tool_name_by_item_id: Arc<Mutex<HashMap<String, String>>>,
    emitted_custom_tool_input_start_ids: Arc<Mutex<HashSet<String>>>,
    emitted_custom_tool_input_end_ids: Arc<Mutex<HashSet<String>>>,
    emitted_custom_tool_call_ids: Arc<Mutex<HashSet<String>>>,

    serialize_state: Arc<Mutex<OpenAiResponsesSerializeState>>,
}

impl Default for OpenAiResponsesEventConverter {
    fn default() -> Self {
        Self {
            function_call_ids_by_output_index: Arc::new(Mutex::new(HashMap::new())),
            function_call_meta_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            emitted_function_tool_input_start_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_function_tool_input_end_ids: Arc::new(Mutex::new(HashSet::new())),
            provider_tool_name_by_item_type: Arc::new(Mutex::new(HashMap::new())),
            mcp_calls_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            mcp_call_args_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            emitted_mcp_call_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_mcp_result_ids: Arc::new(Mutex::new(HashSet::new())),
            mcp_approval_tool_call_id_by_approval_id: Arc::new(Mutex::new(HashMap::new())),
            next_mcp_approval_tool_call_index: Arc::new(Mutex::new(0)),
            emitted_mcp_approval_request_ids: Arc::new(Mutex::new(HashSet::new())),
            reasoning_encrypted_content_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            emitted_reasoning_start_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_reasoning_end_ids: Arc::new(Mutex::new(HashSet::new())),
            apply_patch_call_id_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            apply_patch_operation_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            emitted_apply_patch_tool_input_start_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_apply_patch_tool_input_end_ids: Arc::new(Mutex::new(HashSet::new())),
            code_interpreter_container_id_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            emitted_code_interpreter_tool_input_start_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_code_interpreter_tool_input_end_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_web_search_tool_input_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_stream_start: Arc::new(Mutex::new(false)),
            emitted_response_metadata: Arc::new(Mutex::new(HashSet::new())),
            created_response_id: Arc::new(Mutex::new(None)),
            created_model_id: Arc::new(Mutex::new(None)),
            created_created_at: Arc::new(Mutex::new(None)),
            message_item_id_by_output_index: Arc::new(Mutex::new(HashMap::new())),
            emitted_text_start_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_text_end_ids: Arc::new(Mutex::new(HashSet::new())),
            text_annotations_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            pending_stream_end_events: Arc::new(Mutex::new(VecDeque::new())),
            web_search_default_input: "{}".to_string(),
            include_web_search_provider_executed_in_tool_input: true,
            emit_web_search_tool_input_delta: false,
            emit_web_search_tool_result: true,
            stream_parts_style: StreamPartsStyle::OpenAi,
            responses_transform_style: super::super::transformers::ResponsesTransformStyle::OpenAi,
            provider_metadata_key: "openai".to_string(),
            custom_tool_name_by_call_name: Arc::new(Mutex::new(HashMap::new())),
            custom_tool_call_name_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            custom_tool_tool_name_by_item_id: Arc::new(Mutex::new(HashMap::new())),
            emitted_custom_tool_input_start_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_custom_tool_input_end_ids: Arc::new(Mutex::new(HashSet::new())),
            emitted_custom_tool_call_ids: Arc::new(Mutex::new(HashSet::new())),
            serialize_state: Arc::new(Mutex::new(OpenAiResponsesSerializeState::default())),
        }
    }
}

impl OpenAiResponsesEventConverter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_request_tools(self, tools: &[crate::types::Tool]) -> Self {
        self.seed_provider_tool_names_from_request_tools(tools);
        self
    }

    pub fn with_web_search_tool_input_provider_executed(mut self, enabled: bool) -> Self {
        self.include_web_search_provider_executed_in_tool_input = enabled;
        self
    }

    pub fn with_web_search_stream_mode(mut self, mode: WebSearchStreamMode) -> Self {
        match mode {
            WebSearchStreamMode::OpenAi => {
                self.web_search_default_input = "{}".to_string();
                self.include_web_search_provider_executed_in_tool_input = true;
                self.emit_web_search_tool_input_delta = false;
                self.emit_web_search_tool_result = true;
            }
            WebSearchStreamMode::Xai => {
                self.web_search_default_input = "".to_string();
                self.include_web_search_provider_executed_in_tool_input = false;
                self.emit_web_search_tool_input_delta = true;
                self.emit_web_search_tool_result = false;
            }
        }
        self
    }

    pub fn with_stream_parts_style(mut self, style: StreamPartsStyle) -> Self {
        self.stream_parts_style = style;
        self
    }

    pub fn with_responses_transform_style(
        mut self,
        style: super::super::transformers::ResponsesTransformStyle,
    ) -> Self {
        self.responses_transform_style = style;
        self
    }

    pub fn with_provider_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.provider_metadata_key = key.into();
        self
    }

    fn provider_metadata_json(&self, value: serde_json::Value) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        out.insert(self.provider_metadata_key.clone(), value);
        serde_json::Value::Object(out)
    }

    fn text_stream_part_id(&self, item_id: &str) -> String {
        match self.stream_parts_style {
            StreamPartsStyle::OpenAi => item_id.to_string(),
            StreamPartsStyle::Xai => format!("text-{item_id}"),
        }
    }

    fn reasoning_stream_part_id(&self, item_id: &str) -> String {
        match self.stream_parts_style {
            StreamPartsStyle::OpenAi => item_id.to_string(),
            StreamPartsStyle::Xai => format!("reasoning-{item_id}"),
        }
    }
}

impl OpenAiResponsesEventConverter {
    pub(super) fn serialize_event_impl(
        &self,
        event: &crate::streaming::ChatStreamEvent,
    ) -> Result<Vec<u8>, crate::error::LlmError> {
        serialize::serialize_event(self, event)
    }
}
