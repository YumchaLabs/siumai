//! OpenAI Responses SSE Event Converter (protocol layer)
//!
//! This module normalizes OpenAI Responses API SSE events into Siumai's unified
//! `ChatStreamEvent` sequence. It is intentionally part of the `standards::openai`
//! protocol implementation so that providers stay thin.
//!
//! Note: Providers may re-export this converter under historical module paths
//! (e.g. `providers::openai::responses::OpenAiResponsesEventConverter`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use chrono::{SecondsFormat, TimeZone, Utc};

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
struct OpenAiResponsesFunctionCallSerializeState {
    item_id: String,
    output_index: u64,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Default, Clone)]
struct OpenAiResponsesSerializeState {
    response_id: Option<String>,
    model_id: Option<String>,
    created_at: Option<i64>,
    next_sequence_number: u64,

    used_output_indices: std::collections::HashSet<u64>,
    next_output_index: u64,

    message_item_id: Option<String>,
    message_output_index: Option<u64>,
    message_content_index: u64,
    message_scaffold_emitted: bool,
    message_text: String,
    message_annotation_index: u64,

    reasoning_item_id: Option<String>,
    reasoning_output_index: Option<u64>,
    reasoning_summary_index: u64,

    function_calls_by_call_id:
        std::collections::HashMap<String, OpenAiResponsesFunctionCallSerializeState>,

    latest_usage: Option<crate::types::Usage>,
}

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
    responses_transform_style: super::transformers::ResponsesTransformStyle,

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
            responses_transform_style: super::transformers::ResponsesTransformStyle::OpenAi,
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
        style: super::transformers::ResponsesTransformStyle,
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

    fn clear_pending_stream_end_events(&self) {
        if let Ok(mut q) = self.pending_stream_end_events.lock() {
            q.clear();
        }
    }

    fn replace_pending_stream_end_events(&self, events: Vec<crate::streaming::ChatStreamEvent>) {
        if let Ok(mut q) = self.pending_stream_end_events.lock() {
            q.clear();
            q.extend(events);
        }
    }

    fn pop_pending_stream_end_event(&self) -> Option<crate::streaming::ChatStreamEvent> {
        self.pending_stream_end_events
            .lock()
            .ok()
            .and_then(|mut q| q.pop_front())
    }

    fn seed_provider_tool_names_from_request_tools(&self, tools: &[crate::types::Tool]) {
        use crate::types::Tool;

        let mut map = match self.provider_tool_name_by_item_type.lock() {
            Ok(m) => m,
            Err(_) => return,
        };

        let mut custom_tool_name_by_call_name = match self.custom_tool_name_by_call_name.lock() {
            Ok(m) => m,
            Err(_) => return,
        };

        for tool in tools {
            let Tool::ProviderDefined(t) = tool else {
                continue;
            };

            let tool_type = t.id.rsplit('.').next().unwrap_or("");
            if tool_type.is_empty() || t.name.is_empty() {
                continue;
            }

            match tool_type {
                // Responses built-ins (provider-defined tools)
                "web_search_preview" => {
                    map.insert("web_search_call".to_string(), t.name.clone());
                }
                "web_search" => {
                    map.entry("web_search_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                // xAI vendor mapping: code execution is exposed as `code_interpreter_call` items.
                "code_execution" => {
                    map.entry("code_interpreter_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                // xAI vendor mapping: x_search triggers internal `custom_tool_call` items
                // (e.g. `x_keyword_search`) that should map back to the client tool name.
                "x_search" => {
                    custom_tool_name_by_call_name
                        .entry("x_keyword_search".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "file_search" => {
                    map.entry("file_search_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "code_interpreter" => {
                    map.entry("code_interpreter_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "image_generation" => {
                    map.entry("image_generation_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "local_shell" => {
                    map.entry("local_shell_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "shell" => {
                    map.entry("shell_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "apply_patch" => {
                    map.entry("apply_patch_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "computer_use_preview" => {
                    map.insert("computer_call".to_string(), t.name.clone());
                }
                "computer_use" => {
                    map.entry("computer_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                _ => {}
            }
        }
    }

    fn custom_tool_name_for_call_name(&self, call_name: &str) -> String {
        if call_name.is_empty() {
            return String::new();
        }

        self.custom_tool_name_by_call_name
            .lock()
            .ok()
            .and_then(|m| m.get(call_name).cloned())
            .unwrap_or_else(|| call_name.to_string())
    }

    fn record_custom_tool_item(&self, item_id: &str, call_name: &str, tool_name: &str) {
        if item_id.is_empty() {
            return;
        }

        if let Ok(mut m) = self.custom_tool_call_name_by_item_id.lock() {
            m.insert(item_id.to_string(), call_name.to_string());
        }
        if let Ok(mut m) = self.custom_tool_tool_name_by_item_id.lock() {
            m.insert(item_id.to_string(), tool_name.to_string());
        }
    }

    fn custom_tool_name_by_item_id(&self, item_id: &str) -> Option<String> {
        self.custom_tool_tool_name_by_item_id
            .lock()
            .ok()
            .and_then(|m| m.get(item_id).cloned())
    }

    fn mark_custom_tool_input_start_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_custom_tool_input_start_ids.lock() else {
            return false;
        };
        set.insert(id.to_string())
    }

    fn mark_custom_tool_input_end_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_custom_tool_input_end_ids.lock() else {
            return false;
        };
        set.insert(id.to_string())
    }

    fn mark_custom_tool_call_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_custom_tool_call_ids.lock() else {
            return false;
        };
        set.insert(id.to_string())
    }

    fn convert_custom_tool_call_input_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str())?;
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");

        let mut out: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
        if !self.mark_custom_tool_input_start_emitted(item_id) {
            // If the start was already emitted, continue.
        } else if let Some(tool_name) = self.custom_tool_name_by_item_id(item_id)
            && !tool_name.is_empty()
        {
            out.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-start".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-start",
                    "id": item_id,
                    "toolName": tool_name,
                }),
            });
        }

        out.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": item_id,
                "delta": delta,
            }),
        });

        Some(out)
    }

    fn convert_custom_tool_call_input_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str())?;

        if !self.mark_custom_tool_input_end_emitted(item_id) {
            return None;
        }

        Some(vec![crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-end".to_string(),
            data: serde_json::json!({
                "type": "tool-input-end",
                "id": item_id,
            }),
        }])
    }

    fn update_provider_tool_names(&self, json: &serde_json::Value) {
        let Some(tools) = json
            .get("response")
            .and_then(|r| r.get("tools"))
            .and_then(|t| t.as_array())
        else {
            return;
        };

        let mut map = match self.provider_tool_name_by_item_type.lock() {
            Ok(m) => m,
            Err(_) => return,
        };

        for tool in tools {
            let Some(tool_type) = tool.get("type").and_then(|v| v.as_str()) else {
                continue;
            };

            match tool_type {
                // Fallback mapping: if request-level tool names were not provided,
                // use the configured tool type as `toolName`.
                "web_search_preview" => {
                    map.entry("web_search_call".to_string())
                        .or_insert_with(|| "web_search_preview".to_string());
                }
                "web_search" => {
                    map.entry("web_search_call".to_string())
                        .or_insert_with(|| "web_search".to_string());
                }
                "file_search" => {
                    map.entry("file_search_call".to_string())
                        .or_insert_with(|| "file_search".to_string());
                }
                "code_interpreter" => {
                    map.entry("code_interpreter_call".to_string())
                        .or_insert_with(|| "code_interpreter".to_string());
                }
                "image_generation" => {
                    map.entry("image_generation_call".to_string())
                        .or_insert_with(|| "image_generation".to_string());
                }
                "local_shell" => {
                    map.entry("local_shell_call".to_string())
                        .or_insert_with(|| "shell".to_string());
                }
                "shell" => {
                    map.entry("shell_call".to_string())
                        .or_insert_with(|| "shell".to_string());
                }
                "apply_patch" => {
                    map.entry("apply_patch_call".to_string())
                        .or_insert_with(|| "apply_patch".to_string());
                }
                "computer_use_preview" => {
                    map.entry("computer_call".to_string())
                        .or_insert_with(|| "computer_use_preview".to_string());
                }
                "computer_use" => {
                    map.entry("computer_call".to_string())
                        .or_insert_with(|| "computer_use".to_string());
                }
                _ => {}
            }
        }
    }

    fn provider_tool_name_for_item_type(&self, item_type: &str) -> Option<String> {
        let map = self.provider_tool_name_by_item_type.lock().ok()?;
        map.get(item_type).cloned()
    }

    fn record_mcp_call_added(&self, item_id: &str, name: &str, server_label: &str) {
        let Ok(mut map) = self.mcp_calls_by_item_id.lock() else {
            return;
        };
        map.insert(
            item_id.to_string(),
            (name.to_string(), server_label.to_string()),
        );
    }

    fn record_mcp_call_args(&self, item_id: &str, args: &str) {
        let Ok(mut map) = self.mcp_call_args_by_item_id.lock() else {
            return;
        };
        map.insert(item_id.to_string(), args.to_string());
    }

    fn mcp_call_meta(&self, item_id: &str) -> Option<(String, String)> {
        let map = self.mcp_calls_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    fn mcp_call_args(&self, item_id: &str) -> Option<String> {
        let map = self.mcp_call_args_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    fn mark_mcp_call_emitted(&self, item_id: &str) {
        if let Ok(mut set) = self.emitted_mcp_call_ids.lock() {
            set.insert(item_id.to_string());
        }
    }

    fn mark_mcp_result_emitted(&self, item_id: &str) {
        if let Ok(mut set) = self.emitted_mcp_result_ids.lock() {
            set.insert(item_id.to_string());
        }
    }

    fn has_emitted_mcp_call(&self, item_id: &str) -> bool {
        self.emitted_mcp_call_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(item_id))
    }

    fn has_emitted_mcp_result(&self, item_id: &str) -> bool {
        self.emitted_mcp_result_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(item_id))
    }

    fn mcp_approval_tool_call_id(&self, approval_id: &str) -> String {
        if approval_id.is_empty() {
            return "id-0".to_string();
        }

        if let Ok(mut map) = self.mcp_approval_tool_call_id_by_approval_id.lock() {
            if let Some(id) = map.get(approval_id) {
                return id.clone();
            }

            let idx = self
                .next_mcp_approval_tool_call_index
                .lock()
                .ok()
                .map(|v| *v)
                .unwrap_or(0);
            if let Ok(mut next) = self.next_mcp_approval_tool_call_index.lock() {
                *next = idx.saturating_add(1);
            }

            let id = format!("id-{idx}");
            map.insert(approval_id.to_string(), id.clone());
            return id;
        }

        "id-0".to_string()
    }

    fn mark_mcp_approval_request_emitted(&self, approval_id: &str) {
        if let Ok(mut set) = self.emitted_mcp_approval_request_ids.lock() {
            set.insert(approval_id.to_string());
        }
    }

    fn has_emitted_mcp_approval_request(&self, approval_id: &str) -> bool {
        self.emitted_mcp_approval_request_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(approval_id))
    }

    fn record_reasoning_encrypted_content(&self, item_id: &str, encrypted_content: Option<String>) {
        if item_id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.reasoning_encrypted_content_by_item_id.lock() {
            map.insert(item_id.to_string(), encrypted_content);
        }
    }

    fn reasoning_encrypted_content(&self, item_id: &str) -> Option<String> {
        let Ok(map) = self.reasoning_encrypted_content_by_item_id.lock() else {
            return None;
        };
        map.get(item_id).cloned().unwrap_or(None)
    }

    fn mark_reasoning_start_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_reasoning_start_ids.lock() {
            set.insert(id.to_string());
        }
    }

    fn has_emitted_reasoning_start(&self, id: &str) -> bool {
        self.emitted_reasoning_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn mark_reasoning_end_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_reasoning_end_ids.lock() {
            set.insert(id.to_string());
        }
    }

    fn has_emitted_reasoning_end(&self, id: &str) -> bool {
        self.emitted_reasoning_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn record_function_call_meta(&self, item_id: &str, call_id: &str, name: &str) {
        if item_id.is_empty() || call_id.is_empty() || name.is_empty() {
            return;
        }
        if let Ok(mut map) = self.function_call_meta_by_item_id.lock() {
            map.insert(item_id.to_string(), (call_id.to_string(), name.to_string()));
        }
    }

    fn function_call_meta(&self, item_id: &str) -> Option<(String, String)> {
        let map = self.function_call_meta_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    fn mark_function_tool_input_start_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_function_tool_input_start_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    #[allow(dead_code)]
    fn has_emitted_function_tool_input_start(&self, id: &str) -> bool {
        self.emitted_function_tool_input_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn mark_function_tool_input_end_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_function_tool_input_end_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    fn has_emitted_function_tool_input_end(&self, id: &str) -> bool {
        self.emitted_function_tool_input_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn record_apply_patch_call(&self, item_id: &str, call_id: &str, operation: serde_json::Value) {
        if item_id.is_empty() || call_id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.apply_patch_call_id_by_item_id.lock() {
            map.insert(item_id.to_string(), call_id.to_string());
        }
        if let Ok(mut map) = self.apply_patch_operation_by_item_id.lock() {
            map.insert(item_id.to_string(), operation);
        }
    }

    fn apply_patch_call_id(&self, item_id: &str) -> Option<String> {
        let map = self.apply_patch_call_id_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    #[allow(dead_code)]
    fn apply_patch_operation(&self, item_id: &str) -> Option<serde_json::Value> {
        let map = self.apply_patch_operation_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    fn mark_apply_patch_tool_input_start_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_apply_patch_tool_input_start_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    fn has_emitted_apply_patch_tool_input_start(&self, id: &str) -> bool {
        self.emitted_apply_patch_tool_input_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn mark_apply_patch_tool_input_end_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_apply_patch_tool_input_end_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    fn has_emitted_apply_patch_tool_input_end(&self, id: &str) -> bool {
        self.emitted_apply_patch_tool_input_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn record_code_interpreter_container_id(&self, item_id: &str, container_id: &str) {
        if item_id.is_empty() || container_id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.code_interpreter_container_id_by_item_id.lock() {
            map.insert(item_id.to_string(), container_id.to_string());
        }
    }

    fn code_interpreter_container_id(&self, item_id: &str) -> Option<String> {
        let map = self.code_interpreter_container_id_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    fn mark_code_interpreter_tool_input_start_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_code_interpreter_tool_input_start_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    fn has_emitted_code_interpreter_tool_input_start(&self, id: &str) -> bool {
        self.emitted_code_interpreter_tool_input_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn mark_code_interpreter_tool_input_end_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_code_interpreter_tool_input_end_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    fn has_emitted_code_interpreter_tool_input_end(&self, id: &str) -> bool {
        self.emitted_code_interpreter_tool_input_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn mark_web_search_tool_input_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_web_search_tool_input_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    fn mark_stream_start_emitted(&self) -> bool {
        let Ok(mut emitted) = self.emitted_stream_start.lock() else {
            return false;
        };
        if *emitted {
            return false;
        }
        *emitted = true;
        true
    }

    fn mark_response_metadata_emitted(&self, response_id: &str) -> bool {
        if response_id.is_empty() {
            return false;
        }
        let Ok(mut emitted) = self.emitted_response_metadata.lock() else {
            return false;
        };
        if emitted.contains(response_id) {
            return false;
        }
        emitted.insert(response_id.to_string());
        true
    }

    fn record_created_response_metadata(&self, response_id: &str, model_id: &str, created_at: i64) {
        if let Ok(mut id) = self.created_response_id.lock() {
            *id = if response_id.is_empty() {
                None
            } else {
                Some(response_id.to_string())
            };
        }
        if let Ok(mut model) = self.created_model_id.lock() {
            *model = if model_id.is_empty() {
                None
            } else {
                Some(model_id.to_string())
            };
        }
        if let Ok(mut created) = self.created_created_at.lock() {
            *created = Some(created_at);
        }
    }

    fn created_response_id(&self) -> Option<String> {
        self.created_response_id.lock().ok().and_then(|v| v.clone())
    }

    fn created_model_id(&self) -> Option<String> {
        self.created_model_id.lock().ok().and_then(|v| v.clone())
    }

    fn created_timestamp_rfc3339_millis(&self) -> Option<String> {
        let created_at = self.created_created_at.lock().ok().and_then(|v| *v)?;
        Utc.timestamp_opt(created_at, 0)
            .single()
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Millis, true))
    }

    fn record_message_item_id(&self, output_index: u64, item_id: &str) {
        if item_id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.message_item_id_by_output_index.lock() {
            map.insert(output_index, item_id.to_string());
        }
    }

    fn message_item_id_for_output_index(&self, output_index: u64) -> Option<String> {
        let map = self.message_item_id_by_output_index.lock().ok()?;
        map.get(&output_index).cloned()
    }

    fn mark_text_start_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_text_start_ids.lock() {
            set.insert(id.to_string());
        }
    }

    fn has_emitted_text_start(&self, id: &str) -> bool {
        self.emitted_text_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn mark_text_end_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_text_end_ids.lock() {
            set.insert(id.to_string());
        }
    }

    fn has_emitted_text_end(&self, id: &str) -> bool {
        self.emitted_text_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn record_text_annotation(&self, item_id: &str, annotation: serde_json::Value) {
        if item_id.is_empty() {
            return;
        }
        let Ok(mut map) = self.text_annotations_by_item_id.lock() else {
            return;
        };
        map.entry(item_id.to_string())
            .or_insert_with(Vec::new)
            .push(annotation);
    }

    fn take_text_annotations(&self, item_id: &str) -> Vec<serde_json::Value> {
        let Ok(mut map) = self.text_annotations_by_item_id.lock() else {
            return Vec::new();
        };
        map.remove(item_id).unwrap_or_default()
    }

    fn convert_responses_event(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // Handle delta as plain text or delta.content
        if let Some(delta) = json.get("delta") {
            // Case 1: delta is a plain string (response.output_text.delta)
            if let Some(s) = delta.as_str()
                && !s.is_empty()
            {
                return Some(crate::streaming::ChatStreamEvent::ContentDelta {
                    delta: s.to_string(),
                    index: None,
                });
            }
            // Case 2: delta.content is a string (message.delta simplified)
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                return Some(crate::streaming::ChatStreamEvent::ContentDelta {
                    delta: content.to_string(),
                    index: None,
                });
            }

            // Handle tool_calls delta (first item only; downstream can coalesce)
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array())
                && let Some((index, tool_call)) = tool_calls.iter().enumerate().next()
            {
                let id = tool_call
                    .get("id")
                    .and_then(|id| id.as_str())
                    .unwrap_or("")
                    .to_string();

                let function_name = tool_call
                    .get("function")
                    .and_then(|func| func.get("name"))
                    .and_then(|n| n.as_str())
                    .map(std::string::ToString::to_string);

                let arguments_delta = tool_call
                    .get("function")
                    .and_then(|func| func.get("arguments"))
                    .and_then(|a| a.as_str())
                    .map(std::string::ToString::to_string);

                return Some(crate::streaming::ChatStreamEvent::ToolCallDelta {
                    id,
                    function_name,
                    arguments_delta,
                    index: Some(index),
                });
            }
        }

        // Handle usage updates with both snake_case and camelCase fields
        if let Some(usage) = json
            .get("usage")
            .or_else(|| json.get("response")?.get("usage"))
        {
            let prompt_tokens = usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
                .or_else(|| usage.get("inputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let completion_tokens = usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
                .or_else(|| usage.get("outputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let total_tokens = usage
                .get("total_tokens")
                .or_else(|| usage.get("totalTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let reasoning_tokens = usage
                .get("reasoning_tokens")
                .or_else(|| usage.get("reasoningTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32);

            let usage_info = crate::types::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                #[allow(deprecated)]
                reasoning_tokens,
                #[allow(deprecated)]
                cached_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: reasoning_tokens.map(|r| {
                    crate::types::CompletionTokensDetails {
                        reasoning_tokens: Some(r),
                        audio_tokens: None,
                        accepted_prediction_tokens: None,
                        rejected_prediction_tokens: None,
                    }
                }),
            };
            return Some(crate::streaming::ChatStreamEvent::UsageUpdate { usage: usage_info });
        }

        None
    }

    fn convert_message_output_item_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // response.output_item.added (message)
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("message") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let output_index = json
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        self.record_message_item_id(output_index, item_id);

        let id = self.text_stream_part_id(item_id);

        if self.has_emitted_text_start(&id) {
            return None;
        }
        self.mark_text_start_emitted(&id);

        match self.stream_parts_style {
            StreamPartsStyle::OpenAi => Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:text-start".to_string(),
                data: serde_json::json!({
                    "type": "text-start",
                    "id": id,
                    "providerMetadata": self.provider_metadata_json(serde_json::json!({
                        "itemId": item_id,
                    })),
                }),
            }),
            StreamPartsStyle::Xai => Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:text-start".to_string(),
                data: serde_json::json!({
                    "type": "text-start",
                    "id": id,
                }),
            }),
        }
    }

    fn convert_output_text_delta_events(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        // response.output_text.delta
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() {
            return None;
        }
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if delta.is_empty() {
            return None;
        }

        let id = self.text_stream_part_id(item_id);

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if !self.has_emitted_text_start(&id) {
            self.mark_text_start_emitted(&id);
            match self.stream_parts_style {
                StreamPartsStyle::OpenAi => {
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:text-start".to_string(),
                        data: serde_json::json!({
                            "type": "text-start",
                            "id": id,
                            "providerMetadata": self.provider_metadata_json(serde_json::json!({
                                "itemId": item_id,
                            })),
                        }),
                    });
                }
                StreamPartsStyle::Xai => {
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:text-start".to_string(),
                        data: serde_json::json!({
                            "type": "text-start",
                            "id": id,
                        }),
                    });
                }
            }
        }

        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": id,
                "delta": delta,
            }),
        });

        Some(events)
    }

    fn convert_message_output_item_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // response.output_item.done (message)
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("message") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let id = self.text_stream_part_id(item_id);

        if self.has_emitted_text_end(&id) {
            return None;
        }
        self.mark_text_end_emitted(&id);

        if self.stream_parts_style == StreamPartsStyle::Xai {
            // xAI Vercel-aligned stream parts omit providerMetadata and annotations.
            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:text-end".to_string(),
                data: serde_json::json!({
                    "type": "text-end",
                    "id": id,
                }),
            });
        }

        let mut annotations = self.take_text_annotations(item_id);

        // If the message id changes between added/deltas and done, try to carry over annotations
        // captured under the original output_index message id.
        if annotations.is_empty()
            && let Some(output_index) = json.get("output_index").and_then(|v| v.as_u64())
            && let Some(original_id) = self.message_item_id_for_output_index(output_index)
            && original_id != item_id
        {
            annotations = self.take_text_annotations(&original_id);
        }

        if annotations.is_empty() {
            // Best-effort fallback: extract final annotations from completed message content.
            if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                for part in content {
                    if let Some(arr) = part.get("annotations").and_then(|v| v.as_array())
                        && !arr.is_empty()
                    {
                        annotations.extend(arr.iter().cloned());
                    }
                }
            }
        }

        let provider_metadata_openai = if annotations.is_empty() {
            serde_json::json!({
                "itemId": item_id,
            })
        } else {
            serde_json::json!({
                "itemId": item_id,
                "annotations": annotations,
            })
        };

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-end".to_string(),
            data: serde_json::json!({
                "type": "text-end",
                "id": id,
                "providerMetadata": self.provider_metadata_json(provider_metadata_openai),
            }),
        })
    }

    fn convert_finish_event(
        &self,
        completed_json: &serde_json::Value,
        response: &crate::types::ChatResponse,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let usage = completed_json
            .get("response")
            .and_then(|r| r.get("usage"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let input_tokens = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cached_tokens = usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let reasoning_tokens = usage
            .get("output_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let input_cache_read = cached_tokens.min(input_tokens);
        let input_no_cache = input_tokens.saturating_sub(input_cache_read);
        let output_reasoning = reasoning_tokens.min(output_tokens);
        let output_text = output_tokens.saturating_sub(output_reasoning);

        let unified = response.finish_reason.as_ref().map(|r| match r {
            crate::types::FinishReason::Stop => "stop".to_string(),
            crate::types::FinishReason::StopSequence => "stop".to_string(),
            crate::types::FinishReason::Length => "length".to_string(),
            crate::types::FinishReason::ToolCalls => "tool-calls".to_string(),
            crate::types::FinishReason::ContentFilter => "content-filter".to_string(),
            crate::types::FinishReason::Error => "error".to_string(),
            crate::types::FinishReason::Unknown => "unknown".to_string(),
            crate::types::FinishReason::Other(s) => s.clone(),
        });

        if self.stream_parts_style == StreamPartsStyle::Xai {
            let raw_finish = completed_json
                .get("response")
                .and_then(|r| r.get("status"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "completed".to_string());

            let input_tokens_obj = serde_json::json!({
                "total": input_tokens,
                "cacheRead": input_cache_read,
                "noCache": input_no_cache,
            });
            let output_tokens_obj = serde_json::json!({
                "total": output_tokens,
                "reasoning": output_reasoning,
                "text": output_text,
            });

            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:finish".to_string(),
                data: serde_json::json!({
                    "type": "finish",
                    "finishReason": {
                        "raw": raw_finish,
                        "unified": unified,
                    },
                    "usage": {
                        "inputTokens": input_tokens_obj,
                        "outputTokens": output_tokens_obj,
                        "raw": usage,
                    },
                }),
            });
        }

        let response_id = self.created_response_id().or_else(|| {
            completed_json
                .get("response")?
                .get("id")?
                .as_str()
                .map(|s| s.to_string())
        });

        let service_tier = completed_json
            .get("response")
            .and_then(|r| r.get("service_tier"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let raw_finish_reason = completed_json
            .get("response")
            .and_then(|r| r.get("incomplete_details"))
            .and_then(|d| d.get("reason"))
            .and_then(|v| v.as_str())
            .map(|s| serde_json::Value::String(s.to_string()))
            .unwrap_or(serde_json::Value::Null);

        let mut provider_metadata = serde_json::Map::new();
        if let Some(id) = response_id {
            provider_metadata.insert("responseId".to_string(), serde_json::Value::String(id));
        }
        if let Some(tier) = service_tier {
            provider_metadata.insert("serviceTier".to_string(), serde_json::Value::String(tier));
        }
        if let Some(resp) = completed_json.get("response")
            && let Some(logprobs) =
                super::transformers::response::extract_responses_output_text_logprobs(resp)
        {
            provider_metadata.insert("logprobs".to_string(), logprobs);
        }
        let provider_metadata = serde_json::Value::Object(provider_metadata);

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:finish".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "finishReason": {
                    "raw": raw_finish_reason,
                    "unified": unified,
                },
                "providerMetadata": self.provider_metadata_json(provider_metadata),
                "usage": {
                    "inputTokens": {
                        "total": input_tokens,
                        "cacheRead": input_cache_read,
                        "cacheWrite": serde_json::Value::Null,
                        "noCache": input_no_cache,
                    },
                    "outputTokens": {
                        "total": output_tokens,
                        "reasoning": output_reasoning,
                        "text": output_text,
                    },
                    "raw": usage,
                },
            }),
        })
    }

    fn convert_reasoning_output_item_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("reasoning") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let encrypted_content = item
            .get("encrypted_content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        self.record_reasoning_encrypted_content(item_id, encrypted_content.clone());

        let id = match self.stream_parts_style {
            // Vercel alignment: a reasoning item implies at least one block (`:0`), even when summary is empty.
            StreamPartsStyle::OpenAi => format!("{item_id}:0"),
            // xAI Vercel stream parts use a single id without block suffix.
            StreamPartsStyle::Xai => self.reasoning_stream_part_id(item_id),
        };
        if self.has_emitted_reasoning_start(&id) {
            return None;
        }
        self.mark_reasoning_start_emitted(&id);

        match self.stream_parts_style {
            StreamPartsStyle::OpenAi => Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-start".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-start",
                    "id": id,
                    "providerMetadata": self.provider_metadata_json(serde_json::json!({
                        "itemId": item_id,
                        // Vercel alignment: always include the key for start events.
                        "reasoningEncryptedContent": encrypted_content,
                    })),
                }),
            }),
            StreamPartsStyle::Xai => Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-start".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-start",
                    "id": id,
                }),
            }),
        }
    }

    fn convert_reasoning_summary_part_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // response.reasoning_summary_part.added
        let item_id = json.get("item_id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        if self.stream_parts_style == StreamPartsStyle::Xai {
            // xAI Vercel stream parts do not expose block indices for reasoning summaries.
            // Ensure a start event exists and otherwise ignore this event.
            let id = self.reasoning_stream_part_id(item_id);
            if self.has_emitted_reasoning_start(&id) {
                return None;
            }
            self.mark_reasoning_start_emitted(&id);
            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-start".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-start",
                    "id": id,
                }),
            });
        }
        let summary_index = json
            .get("summary_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let id = format!("{item_id}:{summary_index}");
        if self.has_emitted_reasoning_start(&id) {
            return None;
        }
        self.mark_reasoning_start_emitted(&id);

        let encrypted_content = self.reasoning_encrypted_content(item_id);

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:reasoning-start".to_string(),
            data: serde_json::json!({
                "type": "reasoning-start",
                "id": id,
                "providerMetadata": self.provider_metadata_json(serde_json::json!({
                    "itemId": item_id,
                    // Vercel alignment: always include the key for start events.
                    "reasoningEncryptedContent": encrypted_content,
                })),
            }),
        })
    }

    fn convert_reasoning_summary_text_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // response.reasoning_summary_text.delta
        let item_id = json.get("item_id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if delta.is_empty() {
            return None;
        }

        match self.stream_parts_style {
            StreamPartsStyle::OpenAi => {
                let summary_index = json
                    .get("summary_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let id = format!("{item_id}:{summary_index}");

                // Ensure a start event exists for this block.
                if !self.has_emitted_reasoning_start(&id) {
                    self.mark_reasoning_start_emitted(&id);
                }

                Some(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:reasoning-delta".to_string(),
                    data: serde_json::json!({
                        "type": "reasoning-delta",
                        "id": id,
                        "delta": delta,
                        "providerMetadata": self.provider_metadata_json(serde_json::json!({
                            "itemId": item_id,
                        })),
                    }),
                })
            }
            StreamPartsStyle::Xai => {
                let id = self.reasoning_stream_part_id(item_id);
                if !self.has_emitted_reasoning_start(&id) {
                    self.mark_reasoning_start_emitted(&id);
                }

                Some(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:reasoning-delta".to_string(),
                    data: serde_json::json!({
                        "type": "reasoning-delta",
                        "id": id,
                        "delta": delta,
                    }),
                })
            }
        }
    }

    fn convert_reasoning_output_item_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("reasoning") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let encrypted_content = item
            .get("encrypted_content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        self.record_reasoning_encrypted_content(item_id, encrypted_content.clone());

        if self.stream_parts_style == StreamPartsStyle::Xai {
            let id = self.reasoning_stream_part_id(item_id);
            if self.has_emitted_reasoning_end(&id) {
                return None;
            }
            self.mark_reasoning_end_emitted(&id);

            return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-end".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-end",
                    "id": id,
                }),
            }]);
        }

        let summary_len = item
            .get("summary")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let blocks = std::cmp::max(1, summary_len);

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
        for i in 0..blocks {
            let id = format!("{item_id}:{i}");
            if self.has_emitted_reasoning_end(&id) {
                continue;
            }
            self.mark_reasoning_end_emitted(&id);

            // Vercel alignment: omit reasoningEncryptedContent when it is null/absent.
            let provider_metadata = if let Some(enc) = encrypted_content.as_ref() {
                self.provider_metadata_json(serde_json::json!({
                    "itemId": item_id,
                    "reasoningEncryptedContent": enc,
                }))
            } else {
                self.provider_metadata_json(serde_json::json!({
                    "itemId": item_id,
                }))
            };

            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-end".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-end",
                    "id": id,
                    "providerMetadata": provider_metadata,
                }),
            });
        }

        Some(events)
    }

    fn convert_output_text_annotation_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let annotation = json.get("annotation")?;

        if !item_id.is_empty() {
            self.record_text_annotation(item_id, annotation.clone());
        }

        let ann_type = annotation.get("type")?.as_str()?;

        if ann_type == "url_citation" {
            let url = annotation.get("url")?.as_str()?;
            let title = annotation.get("title").and_then(|v| v.as_str());
            let start_index = annotation.get("start_index").and_then(|v| v.as_u64());
            let id = start_index
                .map(|s| format!("ann:url:{s}"))
                .unwrap_or_else(|| format!("ann:url:{url}"));

            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:source".to_string(),
                data: serde_json::json!({
                    "type": "source",
                    "sourceType": "url",
                    "id": id,
                    "url": url,
                    "title": title,
                }),
            });
        }

        if matches!(
            ann_type,
            "file_citation" | "container_file_citation" | "file_path"
        ) {
            let file_id = annotation.get("file_id")?.as_str()?;
            let filename = annotation
                .get("filename")
                .and_then(|v| v.as_str())
                .unwrap_or(file_id);
            let quote = annotation.get("quote").and_then(|v| v.as_str());

            let media_type = if ann_type == "file_path" {
                "application/octet-stream"
            } else {
                "text/plain"
            };

            let title = quote.unwrap_or(filename);
            let start_index = annotation.get("start_index").and_then(|v| v.as_u64());
            let id = start_index
                .map(|s| format!("ann:doc:{s}"))
                .unwrap_or_else(|| format!("ann:doc:{file_id}"));

            let provider_metadata = match ann_type {
                "file_citation" => {
                    self.provider_metadata_json(serde_json::json!({ "fileId": file_id }))
                }
                "container_file_citation" => self.provider_metadata_json(serde_json::json!({
                    "fileId": file_id,
                    "containerId": annotation.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
                    "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
                })),
                "file_path" => self.provider_metadata_json(serde_json::json!({
                    "fileId": file_id,
                    "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
                })),
                _ => serde_json::Value::Null,
            };

            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:source".to_string(),
                data: serde_json::json!({
                    "type": "source",
                    "sourceType": "document",
                    "id": id,
                    "url": file_id,
                    "title": title,
                    "mediaType": media_type,
                    "filename": filename,
                    "providerMetadata": provider_metadata,
                }),
            });
        }

        None
    }

    fn convert_function_call_arguments_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // Handle response.function_call_arguments.delta events
        let delta = json.get("delta").and_then(|d| d.as_str())?;
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0);

        let id = self
            .function_call_ids_by_output_index
            .lock()
            .ok()
            .and_then(|map| map.get(&output_index).cloned())
            .or_else(|| {
                json.get("item_id")
                    .and_then(|id| id.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_default();

        Some(crate::streaming::ChatStreamEvent::ToolCallDelta {
            id,
            function_name: None, // Function name is set in the initial item.added event
            arguments_delta: Some(delta.to_string()),
            index: Some(output_index as usize),
        })
    }

    fn convert_output_item_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // Handle response.output_item.added events for function calls
        let item = json.get("item")?;
        if item.get("type").and_then(|t| t.as_str()) != Some("function_call") {
            return None;
        }

        let id = item.get("call_id").and_then(|id| id.as_str()).unwrap_or("");
        let function_name = item.get("name").and_then(|name| name.as_str());
        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0);

        if !item_id.is_empty()
            && !id.is_empty()
            && let Some(name) = function_name
        {
            self.record_function_call_meta(item_id, id, name);
        }

        if !id.is_empty()
            && let Ok(mut map) = self.function_call_ids_by_output_index.lock()
        {
            map.insert(output_index, id.to_string());
        }

        Some(crate::streaming::ChatStreamEvent::ToolCallDelta {
            id: id.to_string(),
            function_name: function_name.map(|s| s.to_string()),
            arguments_delta: None, // Arguments will come in subsequent delta events
            index: Some(output_index as usize),
        })
    }

    fn convert_function_call_output_item_added_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item = json.get("item")?;
        if item.get("type").and_then(|t| t.as_str()) != Some("function_call") {
            return None;
        }

        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
        let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");

        if !item_id.is_empty() && !call_id.is_empty() && !tool_name.is_empty() {
            self.record_function_call_meta(item_id, call_id, tool_name);
        }

        if call_id.is_empty() || tool_name.is_empty() {
            return None;
        }

        if !self.mark_function_tool_input_start_emitted(call_id) {
            return None;
        }

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-start".to_string(),
            data: serde_json::json!({
                "type": "tool-input-start",
                "id": call_id,
                "toolName": tool_name,
            }),
        })
    }

    fn convert_function_call_arguments_delta_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let delta = json.get("delta").and_then(|d| d.as_str())?;
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");

        let (call_id, _tool_name) = self.function_call_meta(item_id)?;
        if call_id.is_empty() || delta.is_empty() {
            return None;
        }

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": call_id,
                "delta": delta,
            }),
        })
    }

    fn convert_function_call_arguments_done_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let args = json.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() {
            return None;
        }

        let (call_id, tool_name) = self.function_call_meta(item_id)?;
        if call_id.is_empty() || tool_name.is_empty() {
            return None;
        }

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if !self.has_emitted_function_tool_input_end(call_id.as_str())
            && self.mark_function_tool_input_end_emitted(call_id.as_str())
        {
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-end".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-end",
                    "id": call_id,
                }),
            });
        }

        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": call_id,
                "toolName": tool_name,
                "input": args,
                "providerMetadata": self.provider_metadata_json(serde_json::json!({
                    "itemId": item_id,
                })),
            }),
        });

        Some(events)
    }

    fn convert_apply_patch_output_item_added_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("apply_patch_call") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
        let operation = item
            .get("operation")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        if call_id.is_empty() {
            return None;
        }
        self.record_apply_patch_call(item_id, call_id, operation.clone());

        if !self.mark_apply_patch_tool_input_start_emitted(call_id) {
            return None;
        }

        let tool_name = self
            .provider_tool_name_for_item_type("apply_patch_call")
            .unwrap_or_else(|| "apply_patch".to_string());

        let mut events: Vec<crate::streaming::ChatStreamEvent> =
            vec![crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-start".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-start",
                    "id": call_id,
                    "toolName": tool_name,
                }),
            }];

        let op_type = operation.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let path = operation.get("path").and_then(|v| v.as_str());

        let call_id_json = serde_json::to_string(call_id).unwrap_or_else(|_| "\"\"".to_string());
        let op_type_json = serde_json::to_string(op_type).unwrap_or_else(|_| "\"\"".to_string());
        let path_json = path
            .and_then(|p| serde_json::to_string(p).ok())
            .unwrap_or_else(|| "null".to_string());

        if op_type == "delete_file" {
            let input = format!(
                "{{\"callId\":{call_id_json},\"operation\":{{\"type\":{op_type_json},\"path\":{path_json}}}}}"
            );

            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-delta".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-delta",
                    "id": call_id,
                    "delta": input,
                }),
            });

            if self.mark_apply_patch_tool_input_end_emitted(call_id) {
                events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-input-end".to_string(),
                    data: serde_json::json!({
                        "type": "tool-input-end",
                        "id": call_id,
                    }),
                });
            }

            return Some(events);
        }

        let prefix = format!(
            "{{\"callId\":{call_id_json},\"operation\":{{\"type\":{op_type_json},\"path\":{path_json},\"diff\":\""
        );
        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": call_id,
                "delta": prefix,
            }),
        });

        Some(events)
    }

    fn convert_apply_patch_operation_diff_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() || delta.is_empty() {
            return None;
        }
        let call_id = self.apply_patch_call_id(item_id)?;
        if call_id.is_empty() {
            return None;
        }
        if !self.has_emitted_apply_patch_tool_input_start(call_id.as_str()) {
            return None;
        }

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": call_id,
                "delta": delta,
            }),
        })
    }

    fn convert_apply_patch_operation_diff_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() {
            return None;
        }
        let call_id = self.apply_patch_call_id(item_id)?;
        if call_id.is_empty() || self.has_emitted_apply_patch_tool_input_end(call_id.as_str()) {
            return None;
        }

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        // Close the open `diff` string and the surrounding objects: `"}}`
        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": call_id,
                "delta": "\"}}",
            }),
        });

        if self.mark_apply_patch_tool_input_end_emitted(call_id.as_str()) {
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-end".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-end",
                    "id": call_id,
                }),
            });
        }

        Some(events)
    }

    fn convert_code_interpreter_output_item_added_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("code_interpreter_call") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let tool_name = self
            .provider_tool_name_for_item_type("code_interpreter_call")
            .unwrap_or_else(|| "code_interpreter".to_string());

        let container_id = item
            .get("container_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !container_id.is_empty() {
            self.record_code_interpreter_container_id(item_id, container_id);
        }

        if !self.mark_code_interpreter_tool_input_start_emitted(item_id) {
            return None;
        }

        let container_id_json =
            serde_json::to_string(container_id).unwrap_or_else(|_| "\"\"".to_string());
        let prefix = format!("{{\"containerId\":{container_id_json},\"code\":\"");

        Some(vec![
            crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-start".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-start",
                    "id": item_id,
                    "toolName": tool_name,
                    "providerExecuted": true,
                }),
            },
            crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-delta".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-delta",
                    "id": item_id,
                    "delta": prefix,
                }),
            },
        ])
    }

    fn convert_code_interpreter_code_delta_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() || delta.is_empty() {
            return None;
        }

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if !self.has_emitted_code_interpreter_tool_input_start(item_id)
            && self.mark_code_interpreter_tool_input_start_emitted(item_id)
        {
            let tool_name = self
                .provider_tool_name_for_item_type("code_interpreter_call")
                .unwrap_or_else(|| "code_interpreter".to_string());
            let container_id = self
                .code_interpreter_container_id(item_id)
                .unwrap_or_default();
            let container_id_json =
                serde_json::to_string(container_id.as_str()).unwrap_or_else(|_| "\"\"".to_string());
            let prefix = format!("{{\"containerId\":{container_id_json},\"code\":\"");

            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-start".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-start",
                    "id": item_id,
                    "toolName": tool_name,
                    "providerExecuted": true,
                }),
            });
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-delta".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-delta",
                    "id": item_id,
                    "delta": prefix,
                }),
            });
        }

        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": item_id,
                "delta": delta,
            }),
        });

        Some(events)
    }

    fn convert_code_interpreter_code_done_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() || self.has_emitted_code_interpreter_tool_input_end(item_id) {
            return None;
        }

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        // Close the open `code` string and the object: `"}`
        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": item_id,
                "delta": "\"}",
            }),
        });

        if self.mark_code_interpreter_tool_input_end_emitted(item_id) {
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-end".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-end",
                    "id": item_id,
                }),
            });
        }

        Some(events)
    }

    fn convert_provider_tool_output_item_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        let item_type = item.get("type")?.as_str()?;
        let output_index = json.get("output_index").and_then(|v| v.as_u64());

        let (default_tool_name, input) = match item_type {
            "mcp_call" => {
                // MCP tool calls stream arguments separately. Record metadata here,
                // emit tool-call when arguments are available.
                let item_id = item.get("id")?.as_str()?;
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let server_label = item
                    .get("server_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                self.record_mcp_call_added(item_id, name, server_label);
                return None;
            }
            "mcp_approval_request" => {
                // Vercel alignment: represent approval request as a dynamic tool-call
                // followed by a tool-approval-request (emitted on output_item.done).
                let approval_id = item.get("id")?.as_str()?;
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let tool_call_id = self.mcp_approval_tool_call_id(approval_id);
                let tool_name = format!("mcp.{name}");

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "dynamic": true,
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "input": args,
                        "providerExecuted": true,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
            }
            "custom_tool_call" => {
                // xAI x_search emits internal custom tool calls (e.g. `x_keyword_search`) and streams
                // their input via `response.custom_tool_call_input.*` events.
                let tool_call_id = item.get("id")?.as_str()?;
                let call_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let tool_name = self.custom_tool_name_for_call_name(call_name);
                self.record_custom_tool_item(tool_call_id, call_name, &tool_name);

                if self.mark_custom_tool_input_start_emitted(tool_call_id) {
                    return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-input-start".to_string(),
                        data: serde_json::json!({
                            "type": "tool-input-start",
                            "id": tool_call_id,
                            "toolName": tool_name,
                        }),
                    }]);
                }

                return None;
            }
            "web_search_call" => {
                // xAI streams `arguments` on the output item; Vercel aligns web_search tool input
                // to this JSON string when present. OpenAI does not always include arguments.
                let args = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
                if args.is_empty() {
                    (
                        "web_search",
                        serde_json::Value::String(self.web_search_default_input.clone()),
                    )
                } else {
                    ("web_search", serde_json::Value::String(args.to_string()))
                }
            }
            "file_search_call" => ("file_search", serde_json::json!("{}")),
            "computer_call" => ("computer_use", serde_json::json!("")),
            "code_interpreter_call" => {
                let container_id = item.get("container_id").and_then(|v| v.as_str());
                let tool_call_id = item.get("id")?.as_str()?;
                if let Some(cid) = container_id {
                    self.record_code_interpreter_container_id(tool_call_id, cid);
                }

                // Vercel alignment: code interpreter tool-call is emitted after tool-input-end,
                // once the full code is known (at output_item.done).
                return None;
            }
            "image_generation_call" => ("image_generation", serde_json::json!("{}")),
            _ => return None,
        };

        let tool_name = self
            .provider_tool_name_for_item_type(item_type)
            .unwrap_or_else(|| default_tool_name.to_string());

        let tool_call_id = item.get("id")?.as_str()?;

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        // Vercel alignment: webSearch emits tool-input-start/end even with empty input.
        if item_type == "web_search_call" && self.mark_web_search_tool_input_emitted(tool_call_id) {
            let mut data = serde_json::json!({
                "type": "tool-input-start",
                "id": tool_call_id,
                "toolName": tool_name,
            });
            if self.include_web_search_provider_executed_in_tool_input {
                data["providerExecuted"] = serde_json::json!(true);
            }
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-start".to_string(),
                data,
            });

            if self.emit_web_search_tool_input_delta
                && let serde_json::Value::String(delta) = &input
            {
                events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-input-delta".to_string(),
                    data: serde_json::json!({
                        "type": "tool-input-delta",
                        "id": tool_call_id,
                        "delta": delta,
                    }),
                });
            }

            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-end".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-end",
                    "id": tool_call_id,
                }),
            });
        }

        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "input": input,
                "providerExecuted": true,
                "outputIndex": output_index,
                "rawItem": serde_json::Value::Object(item.clone()),
            }),
        });

        Some(events)
    }

    fn convert_provider_tool_output_item_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        let item_type = item.get("type")?.as_str()?;
        let output_index = json.get("output_index").and_then(|v| v.as_u64());

        let tool_call_id = item.get("id")?.as_str()?;

        let mut extra_events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if item_type == "custom_tool_call" {
            if !self.mark_custom_tool_call_emitted(tool_call_id) {
                return None;
            }

            let call_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let tool_name = self.custom_tool_name_for_call_name(call_name);
            let input = item.get("input").and_then(|v| v.as_str()).unwrap_or("");

            return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-call".to_string(),
                data: serde_json::json!({
                    "type": "tool-call",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "input": input,
                    "providerExecuted": true,
                    "outputIndex": output_index,
                    "rawItem": serde_json::Value::Object(item.clone()),
                }),
            }]);
        }

        let (default_tool_name, result) = match item_type {
            "mcp_approval_request" => {
                let approval_id = item.get("id")?.as_str()?;
                if self.has_emitted_mcp_approval_request(approval_id) {
                    return None;
                }

                let tool_call_id = self.mcp_approval_tool_call_id(approval_id);
                extra_events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-approval-request".to_string(),
                    data: serde_json::json!({
                        "type": "tool-approval-request",
                        "approvalId": approval_id,
                        "toolCallId": tool_call_id,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                });

                self.mark_mcp_approval_request_emitted(approval_id);
                return Some(extra_events);
            }
            "mcp_call" => {
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let server_label = item
                    .get("server_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| self.mcp_call_args(tool_call_id))
                    .unwrap_or_else(|| "{}".to_string());
                let output = item
                    .get("output")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                let tool_name = format!("mcp.{name}");
                let tool_name_for_result = tool_name.clone();
                let args_for_result = args.clone();

                if !self.has_emitted_mcp_call(tool_call_id) {
                    extra_events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-call".to_string(),
                        data: serde_json::json!({
                            "type": "tool-call",
                            "dynamic": true,
                            "toolCallId": tool_call_id,
                            "toolName": tool_name,
                            "input": args,
                            "providerExecuted": true,
                            "outputIndex": output_index,
                            "rawItem": serde_json::Value::Object(item.clone()),
                        }),
                    });
                    self.mark_mcp_call_emitted(tool_call_id);
                }

                if self.has_emitted_mcp_result(tool_call_id) {
                    return Some(extra_events);
                }
                self.mark_mcp_result_emitted(tool_call_id);

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-result".to_string(),
                    data: serde_json::json!({
                        "type": "tool-result",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name_for_result,
                        "result": {
                            "type": "call",
                            "serverLabel": server_label,
                            "name": name,
                            "arguments": args_for_result,
                            "output": output,
                        },
                        "providerExecuted": true,
                        "outputIndex": output_index,
                        "providerMetadata": self.provider_metadata_json(serde_json::json!({
                            "itemId": tool_call_id,
                        })),
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
            }
            "web_search_call" => {
                if !self.emit_web_search_tool_result {
                    return None;
                }

                // xAI web search fixtures do not include `action`/`results` payloads in `output_item.done`.
                // When there is no result payload, Vercel does not emit a `tool-result` stream part.
                if item.get("action").is_none() && item.get("results").is_none() {
                    return None;
                }

                // Include results if present (align with non-streaming transformer).
                let results = item
                    .get("results")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                // Emit Vercel-aligned sources for web search results.
                if let Some(arr) = item.get("results").and_then(|v| v.as_array()) {
                    for (i, r) in arr.iter().enumerate() {
                        let Some(obj) = r.as_object() else {
                            continue;
                        };
                        let Some(url) = obj.get("url").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let title = obj.get("title").and_then(|v| v.as_str());

                        extra_events.push(crate::streaming::ChatStreamEvent::Custom {
                            event_type: "openai:source".to_string(),
                            data: serde_json::json!({
                                "type": "source",
                                "sourceType": "url",
                                "id": format!("{tool_call_id}:{i}"),
                                "url": url,
                                "title": title,
                                "toolCallId": tool_call_id,
                            }),
                        });
                    }
                }

                (
                    "web_search",
                    serde_json::json!({
                        "action": item.get("action").cloned().unwrap_or(serde_json::Value::Null),
                        "results": results,
                        "status": item.get("status").cloned().unwrap_or_else(|| serde_json::json!("completed")),
                    }),
                )
            }
            "file_search_call" => (
                "file_search",
                serde_json::json!({
                    "results": item.get("results").cloned().unwrap_or(serde_json::Value::Null),
                    "status": item.get("status").cloned().unwrap_or_else(|| serde_json::json!("completed")),
                }),
            ),
            "code_interpreter_call" => {
                // Vercel alignment: codeExecution streams tool input, then emits tool-call, then tool-result.
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "code_interpreter".to_string());

                let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

                if !self.has_emitted_code_interpreter_tool_input_end(tool_call_id)
                    && self.mark_code_interpreter_tool_input_end_emitted(tool_call_id)
                {
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-input-delta".to_string(),
                        data: serde_json::json!({
                            "type": "tool-input-delta",
                            "id": tool_call_id,
                            "delta": "\"}",
                        }),
                    });
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-input-end".to_string(),
                        data: serde_json::json!({
                            "type": "tool-input-end",
                            "id": tool_call_id,
                        }),
                    });
                }

                let container_id = item.get("container_id").and_then(|v| v.as_str());
                let code = item.get("code").and_then(|v| v.as_str()).unwrap_or("");

                let code_json = serde_json::to_string(code).unwrap_or_else(|_| "\"\"".to_string());
                let container_id_json = container_id
                    .and_then(|cid| serde_json::to_string(cid).ok())
                    .unwrap_or_else(|| "null".to_string());

                let input = format!("{{\"code\":{code_json},\"containerId\":{container_id_json}}}");

                events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "input": input,
                        "providerExecuted": true,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                });

                events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-result".to_string(),
                    data: serde_json::json!({
                        "type": "tool-result",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "result": {
                            "outputs": item.get("outputs").cloned().unwrap_or_else(|| serde_json::json!([])),
                        },
                        "providerExecuted": true,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                });

                return Some(events);
            }
            "image_generation_call" => (
                "image_generation",
                serde_json::json!({
                    "result": item.get("result").cloned().unwrap_or(serde_json::Value::Null),
                }),
            ),
            "local_shell_call" => {
                let call_id = item.get("call_id").and_then(|v| v.as_str())?;
                let action = item
                    .get("action")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "shell".to_string());

                let input = serde_json::json!({ "action": action }).to_string();

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "toolCallId": call_id,
                        "toolName": tool_name,
                        "input": input,
                        "providerExecuted": false,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
            }
            "shell_call" => {
                let call_id = item.get("call_id").and_then(|v| v.as_str())?;
                let action = item
                    .get("action")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "shell".to_string());

                // Vercel alignment: only expose the commands list to the shell executor.
                let commands = action
                    .as_object()
                    .and_then(|m| m.get("commands"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let input = serde_json::json!({
                    "action": {
                        "commands": commands,
                    }
                })
                .to_string();

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "toolCallId": call_id,
                        "toolName": tool_name,
                        "input": input,
                        "providerExecuted": false,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
            }
            "apply_patch_call" => {
                let call_id = item.get("call_id").and_then(|v| v.as_str())?;
                let operation = item
                    .get("operation")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "apply_patch".to_string());

                let input = serde_json::json!({
                    "callId": call_id,
                    "operation": operation,
                })
                .to_string();

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "toolCallId": call_id,
                        "toolName": tool_name,
                        "input": input,
                        "providerExecuted": false,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
            }
            "computer_call" => (
                "computer_use",
                serde_json::json!({
                    "action": item.get("action").cloned().unwrap_or(serde_json::Value::Null),
                    "status": item.get("status").cloned().unwrap_or_else(|| serde_json::json!("completed")),
                }),
            ),
            _ => return None,
        };

        let tool_name = self
            .provider_tool_name_for_item_type(item_type)
            .unwrap_or_else(|| default_tool_name.to_string());

        let mut events = vec![crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "result": result,
                "providerExecuted": true,
                "outputIndex": output_index,
                "rawItem": serde_json::Value::Object(item.clone()),
            }),
        }];

        events.extend(extra_events);
        Some(events)
    }

    fn convert_mcp_call_arguments_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item_id = json.get("item_id").and_then(|v| v.as_str())?;
        let args = json.get("arguments").and_then(|v| v.as_str())?;
        self.record_mcp_call_args(item_id, args);

        if self.has_emitted_mcp_call(item_id) {
            return None;
        }

        let (name, _server_label) = self.mcp_call_meta(item_id)?;
        self.mark_mcp_call_emitted(item_id);

        let tool_name = format!("mcp.{name}");
        let output_index = json.get("output_index").and_then(|v| v.as_u64());

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "dynamic": true,
                "toolCallId": item_id,
                "toolName": tool_name,
                "input": args,
                "providerExecuted": true,
                "outputIndex": output_index,
            }),
        })
    }

    fn convert_mcp_items_from_completed(
        &self,
        json: &serde_json::Value,
    ) -> Vec<crate::streaming::ChatStreamEvent> {
        let Some(output) = json
            .get("response")
            .and_then(|r| r.get("output"))
            .and_then(|v| v.as_array())
        else {
            return Vec::new();
        };

        let mut events = Vec::new();

        for item in output {
            let Some(item_type) = item.get("type").and_then(|v| v.as_str()) else {
                continue;
            };

            match item_type {
                "mcp_call" => {
                    let Some(tool_call_id) = item.get("id").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    if self.has_emitted_mcp_result(tool_call_id) {
                        continue;
                    }

                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let server_label = item
                        .get("server_label")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let args = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let output = item
                        .get("output")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let tool_name = format!("mcp.{name}");
                    let tool_name_for_result = tool_name.clone();

                    if !self.has_emitted_mcp_call(tool_call_id) {
                        events.push(crate::streaming::ChatStreamEvent::Custom {
                            event_type: "openai:tool-call".to_string(),
                            data: serde_json::json!({
                                "type": "tool-call",
                                "dynamic": true,
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                                "input": args,
                                "providerExecuted": true,
                            }),
                        });
                        self.mark_mcp_call_emitted(tool_call_id);
                    }

                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-result".to_string(),
                        data: serde_json::json!({
                            "type": "tool-result",
                            "toolCallId": tool_call_id,
                            "toolName": tool_name_for_result,
                            "result": {
                                "type": "call",
                                "serverLabel": server_label,
                                "name": name,
                                "arguments": args,
                                "output": output,
                            },
                            "providerExecuted": true,
                            "providerMetadata": self.provider_metadata_json(serde_json::json!({
                                "itemId": tool_call_id,
                            })),
                        }),
                    });
                    self.mark_mcp_result_emitted(tool_call_id);
                }
                "mcp_approval_request" => {
                    let Some(approval_id) = item.get("id").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    if self.has_emitted_mcp_approval_request(approval_id) {
                        continue;
                    }
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let tool_call_id = self.mcp_approval_tool_call_id(approval_id);
                    let tool_call_id_for_approval = tool_call_id.clone();
                    let tool_name = format!("mcp.{name}");

                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-call".to_string(),
                        data: serde_json::json!({
                            "type": "tool-call",
                            "dynamic": true,
                            "toolCallId": tool_call_id,
                            "toolName": tool_name,
                            "input": args,
                            "providerExecuted": true,
                        }),
                    });
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-approval-request".to_string(),
                        data: serde_json::json!({
                            "type": "tool-approval-request",
                            "approvalId": approval_id,
                            "toolCallId": tool_call_id_for_approval,
                        }),
                    });

                    self.mark_mcp_approval_request_emitted(approval_id);
                }
                _ => {}
            }
        }

        events
    }
}

impl crate::streaming::SseEventConverter for OpenAiResponsesEventConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>>,
                > + Send
                + Sync
                + '_,
        >,
    > {
        Box::pin(async move {
            let data_raw = event.data.trim();
            if data_raw.is_empty() {
                return vec![];
            }
            // Consider explicit completed events
            let event_name = event.event.as_str();

            if data_raw == "[DONE]" {
                // [DONE] events should not generate any events in the new architecture
                return vec![];
            }

            // Parse JSON once; most Responses API SSE chunks use `data: {...}` with a `type` field.
            let json = match serde_json::from_str::<serde_json::Value>(data_raw) {
                Ok(v) => v,
                Err(e) => {
                    return vec![Err(crate::error::LlmError::ParseError(format!(
                        "Failed to parse SSE JSON: {e}"
                    )))];
                }
            };

            self.update_provider_tool_names(&json);

            let chunk_type = if !event_name.is_empty() {
                event_name
            } else {
                json.get("type").and_then(|t| t.as_str()).unwrap_or("")
            };

            if chunk_type == "response.created" {
                // A new response in the same SSE connection means any previously buffered
                // StreamEnd is not terminal for the overall stream.
                self.clear_pending_stream_end_events();

                if let Some(resp) = json.get("response") {
                    let response_id = resp.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let model_id = resp.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    let created_at = resp.get("created_at").and_then(|v| v.as_i64()).unwrap_or(0);
                    self.record_created_response_metadata(response_id, model_id, created_at);
                }

                let mut out: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

                if self.mark_stream_start_emitted() {
                    out.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:stream-start".to_string(),
                        data: serde_json::json!({
                            "type": "stream-start",
                            "warnings": [],
                        }),
                    });
                }

                if let (Some(id), Some(model_id), Some(ts)) = (
                    self.created_response_id(),
                    self.created_model_id(),
                    self.created_timestamp_rfc3339_millis(),
                ) && self.mark_response_metadata_emitted(&id)
                {
                    out.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:response-metadata".to_string(),
                        data: serde_json::json!({
                            "type": "response-metadata",
                            "id": id,
                            "modelId": model_id,
                            "timestamp": ts,
                        }),
                    });
                }

                return out.into_iter().map(Ok).collect();
            }

            if chunk_type == "response.failed" {
                // Vercel alignment: failed responses should not error out the converter;
                // instead they emit an error part (from the preceding `error` chunk) and
                // then a finish part with `unified: "other"` and `providerMetadata.responseId`.
                //
                // The OpenAI Responses API does not provide stable usage totals on failures.
                // Mirror the Vercel stream shape by emitting `null` token fields.
                let extra_events = self.convert_mcp_items_from_completed(&json);

                let response_id = self.created_response_id().or_else(|| {
                    json.get("response")?
                        .get("id")?
                        .as_str()
                        .map(|s| s.to_string())
                });

                let mut provider_metadata = serde_json::Map::new();
                if let Some(id) = response_id {
                    provider_metadata
                        .insert("responseId".to_string(), serde_json::Value::String(id));
                }

                let finish_evt = crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:finish".to_string(),
                    data: serde_json::json!({
                        "type": "finish",
                        "finishReason": {
                            "raw": serde_json::Value::Null,
                            "unified": "other",
                        },
                        "providerMetadata": self.provider_metadata_json(serde_json::Value::Object(provider_metadata)),
                        "usage": {
                            "inputTokens": {
                                "total": serde_json::Value::Null,
                                "cacheRead": serde_json::Value::Null,
                                "cacheWrite": serde_json::Value::Null,
                                "noCache": serde_json::Value::Null,
                            },
                            "outputTokens": {
                                "total": serde_json::Value::Null,
                                "reasoning": serde_json::Value::Null,
                                "text": serde_json::Value::Null,
                            },
                            "raw": serde_json::Value::Null,
                        },
                    }),
                };

                self.replace_pending_stream_end_events(vec![finish_evt]);

                return extra_events.into_iter().map(Ok).collect();
            }

            if chunk_type == "response.completed" || chunk_type == "response.incomplete" {
                let extra_events = self.convert_mcp_items_from_completed(&json);

                // The completed event often contains the full response payload.
                // Delegate to centralized ResponseTransformer for final ChatResponse.
                let resp_tx = super::transformers::OpenAiResponsesResponseTransformer::new()
                    .with_style(self.responses_transform_style)
                    .with_provider_metadata_key(self.provider_metadata_key.clone());
                match crate::execution::transformers::response::ResponseTransformer::transform_chat_response(
                    &resp_tx, &json,
                ) {
                    Ok(response) => {
                        // Buffer the final finish + StreamEnd until the stream ends.
                        // OpenAI Responses can emit multiple `response.created` / `response.completed`
                        // pairs on a single SSE connection (e.g., built-in tools), and only the last
                        // completed response should terminate the stream.
                        let mut pending: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
                        if let Some(finish_evt) = self.convert_finish_event(&json, &response) {
                            pending.push(finish_evt);
                        }
                        pending.push(crate::streaming::ChatStreamEvent::StreamEnd { response });
                        self.replace_pending_stream_end_events(pending);

                        return extra_events.into_iter().map(Ok).collect();
                    }
                    Err(err) => return vec![Err(err)],
                }
            }

            // Route by event name first
            match chunk_type {
                "response.output_text.delta" => {
                    let mut out: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

                    if let Some(mut extra) = self.convert_output_text_delta_events(&json) {
                        out.append(&mut extra);
                    }
                    if let Some(evt) = self.convert_responses_event(&json) {
                        out.push(evt);
                    }

                    return out.into_iter().map(Ok).collect();
                }
                "response.tool_call.delta" | "response.function_call.delta" | "response.usage" => {
                    if let Some(evt) = self.convert_responses_event(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.output_text.annotation.added" => {
                    if let Some(evt) = self.convert_output_text_annotation_added(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.error" | "error" => {
                    // Vercel alignment: some fixtures use a legacy top-level `"type": "error"` chunk,
                    // while newer streams use `"type": "response.error"`.
                    //
                    // Emit both:
                    // - a structured custom error part (matches Vercel stream parts)
                    // - a standard ChatStreamEvent::Error for generic consumers
                    let msg = json
                        .get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("Unknown error")
                        .to_string();

                    return vec![
                        Ok(crate::streaming::ChatStreamEvent::Custom {
                            event_type: "openai:error".to_string(),
                            data: serde_json::json!({
                                "type": "error",
                                "error": json,
                            }),
                        }),
                        Ok(crate::streaming::ChatStreamEvent::Error { error: msg }),
                    ];
                }
                "response.function_call_arguments.delta" => {
                    let mut out: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
                    if let Some(evt) = self.convert_function_call_arguments_delta_tool_input(&json)
                    {
                        out.push(evt);
                    }
                    if let Some(evt) = self.convert_function_call_arguments_delta(&json) {
                        out.push(evt);
                    }
                    if !out.is_empty() {
                        return out.into_iter().map(Ok).collect();
                    }
                }
                "response.function_call_arguments.done" => {
                    if let Some(events) =
                        self.convert_function_call_arguments_done_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.apply_patch_call_operation_diff.delta" => {
                    if let Some(evt) = self.convert_apply_patch_operation_diff_delta(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.apply_patch_call_operation_diff.done" => {
                    if let Some(events) = self.convert_apply_patch_operation_diff_done(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.code_interpreter_call_code.delta" => {
                    if let Some(events) = self.convert_code_interpreter_code_delta_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.code_interpreter_call_code.done" => {
                    if let Some(events) = self.convert_code_interpreter_code_done_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.mcp_call_arguments.delta" => {
                    if let Some(item_id) = json.get("item_id").and_then(|v| v.as_str())
                        && let Some(delta) = json.get("delta").and_then(|v| v.as_str())
                    {
                        self.record_mcp_call_args(item_id, delta);
                    }
                }
                "response.mcp_call_arguments.done" => {
                    if let Some(evt) = self.convert_mcp_call_arguments_done(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.custom_tool_call_input.delta" => {
                    if let Some(events) = self.convert_custom_tool_call_input_delta(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.custom_tool_call_input.done" => {
                    if let Some(events) = self.convert_custom_tool_call_input_done(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.output_item.added" => {
                    if let Some(evt) = self.convert_message_output_item_added(&json) {
                        return vec![Ok(evt)];
                    }
                    if let Some(evt) = self.convert_reasoning_output_item_added(&json) {
                        return vec![Ok(evt)];
                    }
                    if let Some(events) =
                        self.convert_apply_patch_output_item_added_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                    if let Some(events) =
                        self.convert_code_interpreter_output_item_added_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                    if let Some(events) = self.convert_provider_tool_output_item_added(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }

                    let mut extra: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
                    if let Some(evt) =
                        self.convert_function_call_output_item_added_tool_input(&json)
                    {
                        extra.push(evt);
                    }
                    if let Some(evt) = self.convert_output_item_added(&json) {
                        extra.push(evt);
                    }
                    if !extra.is_empty() {
                        return extra.into_iter().map(Ok).collect();
                    }
                }
                "response.output_item.done" => {
                    if let Some(events) = self.convert_reasoning_output_item_done(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                    if let Some(events) = self.convert_provider_tool_output_item_done(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                    if let Some(evt) = self.convert_message_output_item_done(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.reasoning_summary_part.added" => {
                    if let Some(evt) = self.convert_reasoning_summary_part_added(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.reasoning_summary_text.delta" => {
                    if let Some(evt) = self.convert_reasoning_summary_text_delta(&json) {
                        return vec![Ok(evt)];
                    }
                }
                _ => {
                    if let Some(evt) = self.convert_responses_event(&json) {
                        return vec![Ok(evt)];
                    }
                }
            }

            vec![]
        })
    }

    fn handle_stream_end(
        &self,
    ) -> Option<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>> {
        self.pop_pending_stream_end_event().map(Ok)
    }

    fn handle_stream_end_events(
        &self,
    ) -> Vec<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>> {
        let Ok(mut q) = self.pending_stream_end_events.lock() else {
            return Vec::new();
        };
        q.drain(..).map(Ok).collect()
    }

    fn finalize_on_disconnect(&self) -> bool {
        true
    }

    fn serialize_event(
        &self,
        event: &crate::streaming::ChatStreamEvent,
    ) -> Result<Vec<u8>, crate::error::LlmError> {
        use crate::error::LlmError;

        fn sse_event_frame(event: &str, value: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
            let data = serde_json::to_vec(value)
                .map_err(|e| LlmError::JsonError(format!("Failed to serialize SSE JSON: {e}")))?;
            let mut out = Vec::with_capacity(data.len() + event.len() + 16);
            out.extend_from_slice(b"event: ");
            out.extend_from_slice(event.as_bytes());
            out.extend_from_slice(b"\n");
            out.extend_from_slice(b"data: ");
            out.extend_from_slice(&data);
            out.extend_from_slice(b"\n\n");
            Ok(out)
        }

        fn sse_done_frame() -> Vec<u8> {
            // OpenAI-style end marker; some clients rely on it, others just close the connection.
            b"data: [DONE]\n\n".to_vec()
        }

        fn now_epoch_seconds() -> i64 {
            chrono::Utc::now().timestamp()
        }

        fn next_sequence_number(state: &mut OpenAiResponsesSerializeState) -> u64 {
            let n = state.next_sequence_number;
            state.next_sequence_number = state.next_sequence_number.saturating_add(1);
            n
        }

        fn ensure_response_metadata(
            this: &OpenAiResponsesEventConverter,
            state: &mut OpenAiResponsesSerializeState,
        ) -> (String, String, i64) {
            let response_id = state
                .response_id
                .clone()
                .or_else(|| this.created_response_id())
                .unwrap_or_else(|| "resp_siumai_0".to_string());
            let model_id = state
                .model_id
                .clone()
                .or_else(|| this.created_model_id())
                .unwrap_or_else(|| "unknown".to_string());
            let created_at = state
                .created_at
                .or_else(|| this.created_created_at.lock().ok().and_then(|v| *v))
                .unwrap_or_else(now_epoch_seconds);

            state.response_id = Some(response_id.clone());
            state.model_id = Some(model_id.clone());
            state.created_at = Some(created_at);

            (response_id, model_id, created_at)
        }

        fn alloc_output_index(state: &mut OpenAiResponsesSerializeState) -> u64 {
            let mut candidate = state.next_output_index;
            loop {
                if !state.used_output_indices.contains(&candidate) {
                    state.used_output_indices.insert(candidate);
                    state.next_output_index = candidate.saturating_add(1);
                    return candidate;
                }
                candidate = candidate.saturating_add(1);
            }
        }

        let mut state = self
            .serialize_state
            .lock()
            .map_err(|_| LlmError::InternalError("serialize_state lock poisoned".to_string()))?;

        match event {
            crate::streaming::ChatStreamEvent::StreamStart { metadata } => {
                *state = OpenAiResponsesSerializeState::default();

                let response_id = metadata
                    .id
                    .clone()
                    .unwrap_or_else(|| "resp_siumai_0".to_string());
                let model_id = metadata
                    .model
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                let created_at = metadata
                    .created
                    .map(|dt| dt.timestamp())
                    .unwrap_or_else(now_epoch_seconds);

                // Persist metadata for subsequent `StreamEnd` frames (best-effort).
                self.record_created_response_metadata(&response_id, &model_id, created_at);
                state.response_id = Some(response_id.clone());
                state.model_id = Some(model_id.clone());
                state.created_at = Some(created_at);

                let payload = serde_json::json!({
                    "type": "response.created",
                    "sequence_number": next_sequence_number(&mut state),
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_at,
                        "status": "in_progress",
                        "model": model_id,
                        "output": [],
                        "usage": serde_json::Value::Null,
                        "metadata": {},
                    }
                });
                sse_event_frame("response.created", &payload)
            }
            crate::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
                let (response_id, _, _) = ensure_response_metadata(self, &mut state);

                if state.message_output_index.is_none() {
                    state.message_output_index = Some(alloc_output_index(&mut state));
                }
                let output_index = state.message_output_index.unwrap_or(0);

                if state.message_item_id.is_none() {
                    state.message_item_id = Some(format!("msg_{response_id}_0"));
                }
                let item_id = state
                    .message_item_id
                    .clone()
                    .unwrap_or_else(|| "msg_siumai_0".to_string());

                // Emit message scaffolding (output_item.added + content_part.added) once.
                let mut out = Vec::new();
                if !state.message_scaffold_emitted {
                    let added = serde_json::json!({
                        "type": "response.output_item.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        }
                    });
                    out.extend_from_slice(&sse_event_frame("response.output_item.added", &added)?);

                    let part_added = serde_json::json!({
                        "type": "response.content_part.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message_content_index,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                            "logprobs": [],
                        }
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.content_part.added",
                        &part_added,
                    )?);
                    state.message_scaffold_emitted = true;
                }

                state.message_text.push_str(delta);
                let payload = serde_json::json!({
                    "type": "response.output_text.delta",
                    "sequence_number": next_sequence_number(&mut state),
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": state.message_content_index,
                    "delta": delta,
                    "logprobs": [],
                });
                out.extend_from_slice(&sse_event_frame("response.output_text.delta", &payload)?);
                Ok(out)
            }
            crate::streaming::ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            } => {
                ensure_response_metadata(self, &mut state);

                // Map ToolCallDelta into Responses-style function_call item + arguments deltas.
                // This aligns better with real `response.output_item.added` + `response.function_call_arguments.delta`.
                let mut emit_added: Option<(String, u64, String, String)> = None;
                let mut emit_args_delta: Option<(String, u64, String)> = None;

                {
                    let output_index_fallback = index
                        .map(|i| i as u64)
                        .filter(|idx| !state.used_output_indices.contains(idx));

                    if !state.function_calls_by_call_id.contains_key(id) {
                        let output_index =
                            output_index_fallback.unwrap_or_else(|| alloc_output_index(&mut state));
                        state.function_calls_by_call_id.insert(
                            id.clone(),
                            OpenAiResponsesFunctionCallSerializeState {
                                item_id: format!("fc_siumai_{output_index}"),
                                output_index,
                                name: None,
                                arguments: String::new(),
                            },
                        );
                    }

                    let call = state
                        .function_calls_by_call_id
                        .get_mut(id)
                        .unwrap_or_else(|| unreachable!("function call state must exist"));

                    if let Some(name) = function_name.clone()
                        && call.name.is_none()
                    {
                        call.name = Some(name);
                        emit_added = Some((
                            call.item_id.clone(),
                            call.output_index,
                            id.clone(),
                            call.name.clone().unwrap_or_else(|| "tool".to_string()),
                        ));
                    }

                    if let Some(delta) = arguments_delta.clone() {
                        call.arguments.push_str(&delta);
                        emit_args_delta = Some((call.item_id.clone(), call.output_index, delta));
                    }
                }

                let mut out = Vec::new();
                if let Some((item_id, output_index, call_id, name)) = emit_added {
                    let added = serde_json::json!({
                        "type": "response.output_item.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "arguments": "",
                            "call_id": call_id,
                            "name": name,
                        }
                    });
                    out.extend_from_slice(&sse_event_frame("response.output_item.added", &added)?);
                }

                if let Some((item_id, output_index, delta)) = emit_args_delta {
                    let payload = serde_json::json!({
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": delta,
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.function_call_arguments.delta",
                        &payload,
                    )?);
                }

                Ok(out)
            }
            crate::streaming::ChatStreamEvent::ThinkingDelta { delta } => {
                // Best-effort mapping into Responses reasoning summary text deltas.
                ensure_response_metadata(self, &mut state);

                if state.reasoning_output_index.is_none() {
                    state.reasoning_output_index = Some(alloc_output_index(&mut state));
                }
                let output_index = state.reasoning_output_index.unwrap_or(0);

                let emit_added = state.reasoning_item_id.is_none();
                if emit_added {
                    state.reasoning_item_id = Some(format!("rs_siumai_{output_index}"));
                }
                let item_id = state
                    .reasoning_item_id
                    .clone()
                    .unwrap_or_else(|| "rs_siumai_0".to_string());

                let mut out = Vec::new();
                if emit_added {
                    let added = serde_json::json!({
                        "type": "response.output_item.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "reasoning",
                            "summary": [],
                        }
                    });
                    out.extend_from_slice(&sse_event_frame("response.output_item.added", &added)?);
                }

                let payload = serde_json::json!({
                    "type": "response.reasoning_summary_text.delta",
                    "sequence_number": next_sequence_number(&mut state),
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": state.reasoning_summary_index,
                    "delta": delta,
                });
                out.extend_from_slice(&sse_event_frame(
                    "response.reasoning_summary_text.delta",
                    &payload,
                )?);
                Ok(out)
            }
            crate::streaming::ChatStreamEvent::UsageUpdate { usage } => {
                state.latest_usage = Some(usage.clone());
                let payload = serde_json::json!({
                    "type": "response.usage",
                    "sequence_number": next_sequence_number(&mut state),
                    "usage": {
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                });
                sse_event_frame("response.usage", &payload)
            }
            crate::streaming::ChatStreamEvent::StreamEnd { response } => {
                let (response_id, model_id, created_at) =
                    ensure_response_metadata(self, &mut state);

                let mut out = Vec::new();

                // Close open function calls (best-effort): emit done + output_item.done.
                let mut function_outputs: Vec<serde_json::Value> = Vec::new();
                let calls: Vec<(String, OpenAiResponsesFunctionCallSerializeState)> =
                    state.function_calls_by_call_id.drain().collect();
                for (call_id, call) in calls {
                    if !call.arguments.is_empty() {
                        let done = serde_json::json!({
                            "type": "response.function_call_arguments.done",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": call.item_id,
                            "output_index": call.output_index,
                            "arguments": call.arguments,
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.function_call_arguments.done",
                            &done,
                        )?);
                    }

                    let item_done = serde_json::json!({
                        "type": "response.output_item.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": call.output_index,
                        "item": {
                            "id": call.item_id,
                            "type": "function_call",
                            "status": "completed",
                            "arguments": call.arguments,
                            "call_id": call_id,
                            "name": call.name.clone().unwrap_or_else(|| "tool".to_string()),
                        }
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.output_item.done",
                        &item_done,
                    )?);

                    function_outputs.push(item_done["item"].clone());
                }

                // Close message output (best-effort): emit output_text.done + content_part.done + output_item.done.
                let final_text = {
                    let txt = response.content.all_text();
                    if txt.is_empty() {
                        state.message_text.clone()
                    } else {
                        txt
                    }
                };

                let mut output: Vec<serde_json::Value> = Vec::new();
                output.extend(function_outputs);

                if let (Some(item_id), Some(output_index)) =
                    (state.message_item_id.clone(), state.message_output_index)
                {
                    let output_text_done = serde_json::json!({
                        "type": "response.output_text.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message_content_index,
                        "text": final_text,
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.output_text.done",
                        &output_text_done,
                    )?);

                    let part_done = serde_json::json!({
                        "type": "response.content_part.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message_content_index,
                        "part": {
                            "type": "output_text",
                            "text": final_text,
                            "annotations": [],
                        }
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.content_part.done",
                        &part_done,
                    )?);

                    let item_done = serde_json::json!({
                        "type": "response.output_item.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                { "type": "output_text", "text": final_text, "annotations": [] }
                            ]
                        }
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.output_item.done",
                        &item_done,
                    )?);
                    output.push(item_done["item"].clone());
                } else if !final_text.is_empty() {
                    output.push(serde_json::json!({ "type": "output_text", "text": final_text }));
                }

                let usage = response
                    .usage
                    .clone()
                    .or_else(|| state.latest_usage.clone());
                let usage_json = usage.as_ref().map(|u| {
                    serde_json::json!({
                        "input_tokens": u.prompt_tokens,
                        "output_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                    })
                });

                let payload = serde_json::json!({
                    "type": "response.completed",
                    "sequence_number": next_sequence_number(&mut state),
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_at,
                        "status": "completed",
                        "model": model_id,
                        "output": output,
                        "usage": usage_json.unwrap_or(serde_json::Value::Null),
                        "metadata": {},
                    }
                });

                out.extend_from_slice(&sse_event_frame("response.completed", &payload)?);
                out.extend_from_slice(&sse_done_frame());
                Ok(out)
            }
            crate::streaming::ChatStreamEvent::Error { error } => {
                let payload = serde_json::json!({
                    "type": "response.error",
                    "sequence_number": next_sequence_number(&mut state),
                    "error": { "message": error },
                });
                sse_event_frame("response.error", &payload)
            }
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                fn provider_metadata_value(
                    metadata: &serde_json::Value,
                ) -> Option<&serde_json::Value> {
                    let obj = metadata.as_object()?;
                    obj.get("openai").or_else(|| obj.values().next())
                }

                match event_type.as_str() {
                    "openai:text-delta" => {
                        let Some(delta) = data.get("delta").and_then(|v| v.as_str()) else {
                            return Ok(Vec::new());
                        };
                        if delta.is_empty() {
                            return Ok(Vec::new());
                        }

                        let (response_id, _, _) = ensure_response_metadata(self, &mut state);
                        if state.message_output_index.is_none() {
                            state.message_output_index = Some(alloc_output_index(&mut state));
                        }
                        let output_index = state.message_output_index.unwrap_or(0);

                        if state.message_item_id.is_none() {
                            state.message_item_id = Some(format!("msg_{response_id}_0"));
                        }
                        let item_id = state
                            .message_item_id
                            .clone()
                            .unwrap_or_else(|| "msg_siumai_0".to_string());

                        let mut out = Vec::new();
                        if !state.message_scaffold_emitted {
                            let added = serde_json::json!({
                                "type": "response.output_item.added",
                                "sequence_number": next_sequence_number(&mut state),
                                "output_index": output_index,
                                "item": {
                                    "id": item_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                }
                            });
                            out.extend_from_slice(&sse_event_frame(
                                "response.output_item.added",
                                &added,
                            )?);

                            let part_added = serde_json::json!({
                                "type": "response.content_part.added",
                                "sequence_number": next_sequence_number(&mut state),
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": state.message_content_index,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                    "logprobs": [],
                                }
                            });
                            out.extend_from_slice(&sse_event_frame(
                                "response.content_part.added",
                                &part_added,
                            )?);
                            state.message_scaffold_emitted = true;
                        }

                        state.message_text.push_str(delta);
                        let payload = serde_json::json!({
                            "type": "response.output_text.delta",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": state.message_content_index,
                            "delta": delta,
                            "logprobs": [],
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_text.delta",
                            &payload,
                        )?);
                        Ok(out)
                    }
                    "openai:reasoning-delta" => {
                        let Some(delta) = data.get("delta").and_then(|v| v.as_str()) else {
                            return Ok(Vec::new());
                        };
                        if delta.is_empty() {
                            return Ok(Vec::new());
                        }

                        ensure_response_metadata(self, &mut state);

                        if state.reasoning_output_index.is_none() {
                            state.reasoning_output_index = Some(alloc_output_index(&mut state));
                        }
                        let output_index = state.reasoning_output_index.unwrap_or(0);

                        let emit_added = state.reasoning_item_id.is_none();
                        if emit_added {
                            state.reasoning_item_id = Some(format!("rs_siumai_{output_index}"));
                        }
                        let item_id = state
                            .reasoning_item_id
                            .clone()
                            .unwrap_or_else(|| "rs_siumai_0".to_string());

                        let summary_index = data
                            .get("id")
                            .and_then(|v| v.as_str())
                            .and_then(|id| id.rsplit_once(':').map(|(_, n)| n))
                            .and_then(|n| n.parse::<u64>().ok())
                            .unwrap_or(state.reasoning_summary_index);

                        let mut out = Vec::new();
                        if emit_added {
                            let added = serde_json::json!({
                                "type": "response.output_item.added",
                                "sequence_number": next_sequence_number(&mut state),
                                "output_index": output_index,
                                "item": {
                                    "id": item_id,
                                    "type": "reasoning",
                                    "summary": [],
                                }
                            });
                            out.extend_from_slice(&sse_event_frame(
                                "response.output_item.added",
                                &added,
                            )?);
                        }

                        let payload = serde_json::json!({
                            "type": "response.reasoning_summary_text.delta",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": item_id,
                            "output_index": output_index,
                            "summary_index": summary_index,
                            "delta": delta,
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.reasoning_summary_text.delta",
                            &payload,
                        )?);
                        Ok(out)
                    }
                    "openai:source" => {
                        // Tool result sources (e.g. web_search_call) should stay as Vercel parts and
                        // must not be mapped into output_text annotations.
                        if data.get("toolCallId").is_some() {
                            return Ok(Vec::new());
                        }

                        let Some(source_type) = data.get("sourceType").and_then(|v| v.as_str())
                        else {
                            return Ok(Vec::new());
                        };

                        let (response_id, _, _) = ensure_response_metadata(self, &mut state);

                        if state.message_output_index.is_none() {
                            state.message_output_index = Some(alloc_output_index(&mut state));
                        }
                        let output_index = state.message_output_index.unwrap_or(0);

                        if state.message_item_id.is_none() {
                            state.message_item_id = Some(format!("msg_{response_id}_0"));
                        }
                        let item_id = state
                            .message_item_id
                            .clone()
                            .unwrap_or_else(|| "msg_siumai_0".to_string());

                        let mut out = Vec::new();
                        if !state.message_scaffold_emitted {
                            let added = serde_json::json!({
                                "type": "response.output_item.added",
                                "sequence_number": next_sequence_number(&mut state),
                                "output_index": output_index,
                                "item": {
                                    "id": item_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                }
                            });
                            out.extend_from_slice(&sse_event_frame(
                                "response.output_item.added",
                                &added,
                            )?);

                            let part_added = serde_json::json!({
                                "type": "response.content_part.added",
                                "sequence_number": next_sequence_number(&mut state),
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": state.message_content_index,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                    "logprobs": [],
                                }
                            });
                            out.extend_from_slice(&sse_event_frame(
                                "response.content_part.added",
                                &part_added,
                            )?);
                            state.message_scaffold_emitted = true;
                        }

                        let annotation = match source_type {
                            "url" => {
                                let Some(url) = data.get("url").and_then(|v| v.as_str()) else {
                                    return Ok(Vec::new());
                                };
                                let title = data.get("title").and_then(|v| v.as_str());
                                let mut ann = serde_json::Map::new();
                                ann.insert("type".to_string(), serde_json::json!("url_citation"));
                                ann.insert("url".to_string(), serde_json::json!(url));
                                if let Some(t) = title {
                                    ann.insert("title".to_string(), serde_json::json!(t));
                                }
                                serde_json::Value::Object(ann)
                            }
                            "document" => {
                                let Some(file_id) = data.get("url").and_then(|v| v.as_str()) else {
                                    return Ok(Vec::new());
                                };
                                let filename = data
                                    .get("filename")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(file_id);
                                let quote = data.get("title").and_then(|v| v.as_str());
                                let media_type = data.get("mediaType").and_then(|v| v.as_str());

                                let provider_meta = data
                                    .get("providerMetadata")
                                    .and_then(provider_metadata_value);
                                let container_id = provider_meta
                                    .and_then(|v| v.get("containerId"))
                                    .filter(|v| !v.is_null())
                                    .cloned();
                                let index = provider_meta.and_then(|v| v.get("index")).cloned();

                                let ann_type = if media_type == Some("application/octet-stream") {
                                    "file_path"
                                } else if container_id.is_some() {
                                    "container_file_citation"
                                } else {
                                    "file_citation"
                                };

                                let mut ann = serde_json::Map::new();
                                ann.insert("type".to_string(), serde_json::json!(ann_type));
                                ann.insert("file_id".to_string(), serde_json::json!(file_id));
                                ann.insert("filename".to_string(), serde_json::json!(filename));
                                if let Some(q) = quote {
                                    ann.insert("quote".to_string(), serde_json::json!(q));
                                }
                                if ann_type == "container_file_citation" {
                                    if let Some(cid) = container_id {
                                        ann.insert("container_id".to_string(), cid);
                                    }
                                    if let Some(idx) = index {
                                        ann.insert("index".to_string(), idx);
                                    }
                                } else if ann_type == "file_path"
                                    && let Some(idx) = index
                                {
                                    ann.insert("index".to_string(), idx);
                                }
                                serde_json::Value::Object(ann)
                            }
                            _ => return Ok(Vec::new()),
                        };

                        let annotation_index = state.message_annotation_index;
                        state.message_annotation_index =
                            state.message_annotation_index.saturating_add(1);

                        let payload = serde_json::json!({
                            "type": "response.output_text.annotation.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": state.message_content_index,
                            "annotation_index": annotation_index,
                            "annotation": annotation,
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_text.annotation.added",
                            &payload,
                        )?);
                        Ok(out)
                    }
                    _ => Ok(Vec::new()),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::SseEventConverter;
    use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};

    fn parse_sse_frames(bytes: &[u8]) -> Vec<(String, serde_json::Value)> {
        let text = String::from_utf8_lossy(bytes);
        text.split("\n\n")
            .filter_map(|chunk| {
                let mut event_name: Option<String> = None;
                let mut data_line: Option<String> = None;
                for line in chunk.lines() {
                    if let Some(v) = line.strip_prefix("event: ") {
                        event_name = Some(v.trim().to_string());
                    } else if let Some(v) = line.strip_prefix("data: ") {
                        data_line = Some(v.trim().to_string());
                    }
                }
                let event_name = event_name?;
                let data_line = data_line?;
                if data_line == "[DONE]" {
                    return None;
                }
                let json = serde_json::from_str::<serde_json::Value>(&data_line).ok()?;
                Some((event_name, json))
            })
            .collect()
    }

    #[test]
    fn test_responses_event_converter_content_delta() {
        let conv = OpenAiResponsesEventConverter::new();
        let event = eventsource_stream::Event {
            event: "message".to_string(),
            data: r#"{"delta":{"content":"hello"}}"#.to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let fut = conv.convert_event(event);
        let events = futures::executor::block_on(fut);
        assert!(!events.is_empty());
        let ev = events.first().unwrap().as_ref().unwrap();
        match ev {
            crate::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
                assert_eq!(delta, "hello")
            }
            _ => panic!("expected ContentDelta"),
        }
    }

    #[test]
    fn test_responses_event_converter_tool_call_delta() {
        let conv = OpenAiResponsesEventConverter::new();
        let event = eventsource_stream::Event {
            event: "message".to_string(),
            data: r#"{"delta":{"tool_calls":[{"id":"t1","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]}}"#.to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let fut = conv.convert_event(event);
        let events = futures::executor::block_on(fut);
        assert!(!events.is_empty());
        let ev = events.first().unwrap().as_ref().unwrap();
        match ev {
            crate::streaming::ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            } => {
                assert_eq!(id, "t1");
                assert_eq!(function_name.clone().unwrap(), "lookup");
                assert_eq!(arguments_delta.clone().unwrap(), "{\"q\":\"x\"}");
            }
            _ => panic!("expected ToolCallDelta"),
        }
    }

    #[test]
    fn test_responses_event_converter_usage_update() {
        let conv = OpenAiResponsesEventConverter::new();
        let event = eventsource_stream::Event {
            event: "message".to_string(),
            data: r#"{"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}"#
                .to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let fut = conv.convert_event(event);
        let events = futures::executor::block_on(fut);
        assert!(!events.is_empty());
        let ev = events.first().unwrap().as_ref().unwrap();
        match ev {
            crate::streaming::ChatStreamEvent::UsageUpdate { usage } => {
                assert_eq!(usage.prompt_tokens, 3);
                assert_eq!(usage.completion_tokens, 5);
                assert_eq!(usage.total_tokens, 8);
            }
            _ => panic!("expected UsageUpdate"),
        }
    }

    #[test]
    fn test_responses_event_converter_done() {
        let conv = OpenAiResponsesEventConverter::new();
        let event = eventsource_stream::Event {
            event: "message".to_string(),
            data: "[DONE]".to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let fut = conv.convert_event(event);
        let events = futures::executor::block_on(fut);
        // [DONE] events should not generate any events in our new architecture
        assert!(events.is_empty());
    }

    #[test]
    fn test_sse_named_events_routing() {
        let conv = OpenAiResponsesEventConverter::new();
        use crate::streaming::SseEventConverter;

        // content delta via named event
        let ev1 = eventsource_stream::Event {
            event: "response.output_text.delta".to_string(),
            data: r#"{"delta":{"content":"abc"}}"#.to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let events1 = futures::executor::block_on(conv.convert_event(ev1));
        assert!(!events1.is_empty());
        let out1 = events1.first().unwrap().as_ref().unwrap();
        match out1 {
            crate::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
                assert_eq!(delta, "abc")
            }
            _ => panic!("expected ContentDelta"),
        }

        // tool call delta via named event
        let ev2 = eventsource_stream::Event {
            event: "response.tool_call.delta".to_string(),
            data: r#"{"delta":{"tool_calls":[{"id":"t1","function":{"name":"fn","arguments":"{}"}}]}}"#.to_string(),
            id: "2".to_string(),
            retry: None,
        };
        let events2 = futures::executor::block_on(conv.convert_event(ev2));
        assert!(!events2.is_empty());
        let out2 = events2.first().unwrap().as_ref().unwrap();
        match out2 {
            crate::streaming::ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            } => {
                assert_eq!(id, "t1");
                assert_eq!(function_name.clone().unwrap(), "fn");
                assert_eq!(arguments_delta.clone().unwrap(), "{}");
            }
            _ => panic!("expected ToolCallDelta"),
        }

        // usage via named event camelCase
        let ev3 = eventsource_stream::Event {
            event: "response.usage".to_string(),
            data: r#"{"usage":{"inputTokens":4,"outputTokens":6,"totalTokens":10}}"#.to_string(),
            id: "3".to_string(),
            retry: None,
        };
        let events3 = futures::executor::block_on(conv.convert_event(ev3));
        assert!(!events3.is_empty());
        let out3 = events3.first().unwrap().as_ref().unwrap();
        match out3 {
            crate::streaming::ChatStreamEvent::UsageUpdate { usage } => {
                assert_eq!(usage.prompt_tokens, 4);
                assert_eq!(usage.completion_tokens, 6);
                assert_eq!(usage.total_tokens, 10);
            }
            _ => panic!("expected UsageUpdate"),
        }

        // provider tool output_item.added emits custom tool-call event
        let ev_added = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"}}"#.to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let out_added = futures::executor::block_on(conv.convert_event(ev_added));
        assert_eq!(out_added.len(), 3);
        match out_added[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-input-start");
                assert_eq!(data["id"], serde_json::json!("ws_1"));
                assert_eq!(data["toolName"], serde_json::json!("web_search"));
                assert_eq!(data["providerExecuted"], serde_json::json!(true));
            }
            other => panic!("expected Custom tool-input-start, got {other:?}"),
        }
        match out_added[1].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-input-end");
                assert_eq!(data["id"], serde_json::json!("ws_1"));
            }
            other => panic!("expected Custom tool-input-end, got {other:?}"),
        }
        match out_added[2].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-call");
                assert_eq!(data["toolCallId"], serde_json::json!("ws_1"));
                assert_eq!(data["toolName"], serde_json::json!("web_search"));
                assert_eq!(data["providerExecuted"], serde_json::json!(true));
            }
            other => panic!("expected Custom tool-call, got {other:?}"),
        }

        let ev_done = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"rust"}}}"#.to_string(),
            id: "2".to_string(),
            retry: None,
        };
        let out_done = futures::executor::block_on(conv.convert_event(ev_done));
        assert_eq!(out_done.len(), 1);
        match out_done[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("ws_1"));
                assert_eq!(data["toolName"], serde_json::json!("web_search"));
                assert_eq!(data["providerExecuted"], serde_json::json!(true));
                assert_eq!(data["result"]["action"]["query"], serde_json::json!("rust"));
            }
            other => panic!("expected Custom tool-result, got {other:?}"),
        }

        // If the payload includes results, we also emit Vercel-aligned sources.
        let ev_done_with_results = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"rust"},"results":[{"url":"https://www.rust-lang.org","title":"Rust"}]}}"#.to_string(),
            id: "3".to_string(),
            retry: None,
        };
        let out_done = futures::executor::block_on(conv.convert_event(ev_done_with_results));
        assert_eq!(out_done.len(), 2);
        match out_done[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("ws_1"));
            }
            other => panic!("expected Custom tool-result, got {other:?}"),
        }

        match out_done[1].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:source");
                assert_eq!(data["url"], serde_json::json!("https://www.rust-lang.org"));
                assert_eq!(data["sourceType"], serde_json::json!("url"));
            }
            other => panic!("expected Custom source, got {other:?}"),
        }
    }

    #[test]
    fn responses_provider_tool_name_uses_configured_web_search_preview() {
        let conv = OpenAiResponsesEventConverter::new();

        let ev_created = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.created","response":{"tools":[{"type":"web_search_preview","search_context_size":"low","user_location":{"type":"approximate"}}]}}"#
                .to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let _ = futures::executor::block_on(conv.convert_event(ev_created));

        let ev_added = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"}}"#
                .to_string(),
            id: "2".to_string(),
            retry: None,
        };
        let out_added = futures::executor::block_on(conv.convert_event(ev_added));
        // Vercel alignment: web search emits tool-input-start/end even with empty input.
        assert_eq!(out_added.len(), 3);
        match out_added[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-input-start");
                assert_eq!(data["toolName"], serde_json::json!("web_search_preview"));
            }
            other => panic!("expected Custom tool-input-start, got {other:?}"),
        }
        match out_added[1].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, .. } => {
                assert_eq!(event_type, "openai:tool-input-end");
            }
            other => panic!("expected Custom tool-input-end, got {other:?}"),
        }
        match out_added[2].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-call");
                assert_eq!(data["toolName"], serde_json::json!("web_search_preview"));
            }
            other => panic!("expected Custom tool-call, got {other:?}"),
        }

        let ev_done = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"rust"}}}"#
                .to_string(),
            id: "3".to_string(),
            retry: None,
        };
        let out_done = futures::executor::block_on(conv.convert_event(ev_done));
        assert_eq!(out_done.len(), 1);
        match out_done[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-result");
                assert_eq!(data["toolName"], serde_json::json!("web_search_preview"));
            }
            other => panic!("expected Custom tool-result, got {other:?}"),
        }
    }

    #[test]
    fn responses_output_text_annotation_added_emits_source() {
        let conv = OpenAiResponsesEventConverter::new();

        let ev = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","url":"https://www.rust-lang.org","title":"Rust","start_index":1,"end_index":2}}"#
                .to_string(),
            id: "1".to_string(),
            retry: None,
        };

        let out = futures::executor::block_on(conv.convert_event(ev));
        assert_eq!(out.len(), 1);
        match out[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:source");
                assert_eq!(data["sourceType"], serde_json::json!("url"));
                assert_eq!(data["url"], serde_json::json!("https://www.rust-lang.org"));
            }
            other => panic!("expected Custom source, got {other:?}"),
        }

        let ev = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_text.annotation.added","annotation":{"type":"file_citation","file_id":"file_123","filename":"notes.txt","quote":"Document","start_index":10,"end_index":20}}"#
                .to_string(),
            id: "2".to_string(),
            retry: None,
        };

        let out = futures::executor::block_on(conv.convert_event(ev));
        assert_eq!(out.len(), 1);
        match out[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:source");
                assert_eq!(data["sourceType"], serde_json::json!("document"));
                assert_eq!(data["url"], serde_json::json!("file_123"));
                assert_eq!(data["filename"], serde_json::json!("notes.txt"));
            }
            other => panic!("expected Custom source, got {other:?}"),
        }
    }

    #[test]
    fn responses_stream_proxy_serializes_basic_text_deltas() {
        let conv = OpenAiResponsesEventConverter::new();

        let start_bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("resp_test".to_string()),
                    model: Some("gpt-test".to_string()),
                    created: None,
                    provider: "openai".to_string(),
                    request_id: None,
                },
            })
            .expect("serialize start");
        let start_frames = parse_sse_frames(&start_bytes);
        assert_eq!(start_frames.len(), 1);
        assert_eq!(start_frames[0].0, "response.created");
        assert_eq!(
            start_frames[0].1["type"],
            serde_json::json!("response.created")
        );

        let delta_bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            })
            .expect("serialize delta");
        let delta_frames = parse_sse_frames(&delta_bytes);
        assert!(
            delta_frames
                .iter()
                .any(|(ev, v)| ev == "response.output_text.delta"
                    && v["delta"] == serde_json::json!("Hello"))
        );

        let end_bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("resp_test".to_string()),
                    model: Some("gpt-test".to_string()),
                    content: MessageContent::Text("Hello".to_string()),
                    usage: Some(
                        Usage::builder()
                            .prompt_tokens(3)
                            .completion_tokens(5)
                            .total_tokens(8)
                            .build(),
                    ),
                    finish_reason: Some(FinishReason::Stop),
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: None,
                },
            })
            .expect("serialize end");
        let end_frames = parse_sse_frames(&end_bytes);
        assert!(end_frames.iter().any(|(ev, v)| {
            ev == "response.completed" && v["type"] == serde_json::json!("response.completed")
        }));
    }

    #[test]
    fn responses_stream_proxy_serializes_openai_text_stream_part_delta() {
        let conv = OpenAiResponsesEventConverter::new();

        let _ = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("resp_test".to_string()),
                    model: Some("gpt-test".to_string()),
                    created: None,
                    provider: "openai".to_string(),
                    request_id: None,
                },
            })
            .expect("serialize start");

        let bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:text-delta".to_string(),
                data: serde_json::json!({
                    "type": "text-delta",
                    "id": "msg_1",
                    "delta": "Hello",
                }),
            })
            .expect("serialize stream part");

        let frames = parse_sse_frames(&bytes);
        assert!(frames.iter().any(|(ev, v)| {
            ev == "response.output_text.delta" && v["delta"] == serde_json::json!("Hello")
        }));
    }

    #[test]
    fn responses_stream_proxy_serializes_openai_reasoning_stream_part_delta() {
        let conv = OpenAiResponsesEventConverter::new();

        let bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-delta".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-delta",
                    "id": "rs_1:0",
                    "delta": "think",
                }),
            })
            .expect("serialize reasoning stream part");

        let frames = parse_sse_frames(&bytes);
        assert!(frames.iter().any(|(ev, v)| {
            ev == "response.reasoning_summary_text.delta"
                && v["delta"] == serde_json::json!("think")
        }));
    }

    #[test]
    fn responses_stream_proxy_serializes_openai_source_stream_part_as_annotation_added() {
        let conv = OpenAiResponsesEventConverter::new();

        let _ = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("resp_test".to_string()),
                    model: Some("gpt-test".to_string()),
                    created: None,
                    provider: "openai".to_string(),
                    request_id: None,
                },
            })
            .expect("serialize start");

        let _ = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:text-delta".to_string(),
                data: serde_json::json!({
                    "type": "text-delta",
                    "id": "msg_1",
                    "delta": "Hello",
                }),
            })
            .expect("serialize text delta");

        let url_bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:source".to_string(),
                data: serde_json::json!({
                    "type": "source",
                    "sourceType": "url",
                    "id": "ann:url:https://www.rust-lang.org",
                    "url": "https://www.rust-lang.org",
                    "title": "Rust",
                }),
            })
            .expect("serialize url source");
        let url_frames = parse_sse_frames(&url_bytes);
        assert!(url_frames.iter().any(|(ev, v)| {
            ev == "response.output_text.annotation.added"
                && v["annotation"]["type"] == serde_json::json!("url_citation")
                && v["annotation"]["url"] == serde_json::json!("https://www.rust-lang.org")
        }));

        let doc_bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:source".to_string(),
                data: serde_json::json!({
                    "type": "source",
                    "sourceType": "document",
                    "id": "ann:doc:file_123",
                    "url": "file_123",
                    "title": "Document",
                    "mediaType": "text/plain",
                    "filename": "notes.txt",
                    "providerMetadata": { "openai": { "fileId": "file_123" } },
                }),
            })
            .expect("serialize doc source");
        let doc_frames = parse_sse_frames(&doc_bytes);
        assert!(doc_frames.iter().any(|(ev, v)| {
            ev == "response.output_text.annotation.added"
                && v["annotation"]["type"] == serde_json::json!("file_citation")
                && v["annotation"]["file_id"] == serde_json::json!("file_123")
                && v["annotation"]["filename"] == serde_json::json!("notes.txt")
        }));
    }

    #[test]
    fn responses_stream_proxy_serializes_tool_call_delta() {
        let conv = OpenAiResponsesEventConverter::new();

        let bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                function_name: Some("lookup".to_string()),
                arguments_delta: Some("{\"q\":\"rust\"}".to_string()),
                index: None,
            })
            .expect("serialize tool delta");

        let frames = parse_sse_frames(&bytes);
        assert!(frames.iter().any(|(ev, v)| {
            ev == "response.output_item.added"
                && v["item"]["type"] == serde_json::json!("function_call")
                && v["item"]["call_id"] == serde_json::json!("call_1")
                && v["item"]["name"] == serde_json::json!("lookup")
        }));
        assert!(frames.iter().any(|(ev, v)| {
            ev == "response.function_call_arguments.delta"
                && v["delta"] == serde_json::json!("{\"q\":\"rust\"}")
        }));
    }

    #[test]
    fn responses_stream_proxy_serializes_response_error() {
        let conv = OpenAiResponsesEventConverter::new();

        let bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::Error {
                error: "Upstream failure".to_string(),
            })
            .expect("serialize error");
        let frames = parse_sse_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, "response.error");
        assert_eq!(
            frames[0].1["error"]["message"],
            serde_json::json!("Upstream failure")
        );
    }

    #[test]
    fn responses_stream_proxy_serializes_reasoning_delta() {
        let conv = OpenAiResponsesEventConverter::new();

        let bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::ThinkingDelta {
                delta: "think".to_string(),
            })
            .expect("serialize reasoning delta");
        let frames = parse_sse_frames(&bytes);

        assert!(frames.iter().any(|(ev, v)| {
            ev == "response.output_item.added"
                && v["item"]["type"] == serde_json::json!("reasoning")
        }));
        assert!(frames.iter().any(|(ev, v)| {
            ev == "response.reasoning_summary_text.delta"
                && v["delta"] == serde_json::json!("think")
        }));
    }

    #[test]
    fn responses_stream_proxy_closes_function_call_on_stream_end() {
        let conv = OpenAiResponsesEventConverter::new();

        let _ = conv.serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
            },
        });

        let _ = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                function_name: Some("lookup".to_string()),
                arguments_delta: Some("{\"q\":\"rust".to_string()),
                index: Some(0),
            })
            .expect("serialize tool delta 1");
        let _ = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                function_name: None,
                arguments_delta: Some("\"}".to_string()),
                index: Some(0),
            })
            .expect("serialize tool delta 2");

        let end_bytes = conv
            .serialize_event(&crate::streaming::ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("resp_test".to_string()),
                    model: Some("gpt-test".to_string()),
                    content: MessageContent::Text(String::new()),
                    usage: None,
                    finish_reason: Some(FinishReason::Stop),
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: None,
                },
            })
            .expect("serialize end");

        let frames = parse_sse_frames(&end_bytes);
        assert!(frames.iter().any(|(ev, v)| {
            ev == "response.function_call_arguments.done"
                && v["arguments"] == serde_json::json!("{\"q\":\"rust\"}")
        }));
        assert!(frames.iter().any(|(ev, v)| {
            ev == "response.completed"
                && v["response"]["output"]
                    .as_array()
                    .is_some_and(|arr| arr.iter().any(|it| it["type"] == "function_call"))
        }));
    }
}
