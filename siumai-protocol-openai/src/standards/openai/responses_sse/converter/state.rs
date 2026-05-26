use std::sync::{Arc, Mutex, MutexGuard};

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
pub(super) struct OpenAiResponsesReasoningItemSerializeState {
    pub(super) output_index: u64,
}

#[derive(Debug, Default, Clone)]
pub(super) struct OpenAiResponsesMessageSerializeState {
    pub(super) item_id: Option<String>,
    pub(super) output_index: Option<u64>,
    pub(super) content_index: u64,
    pub(super) scaffold_emitted: bool,
    pub(super) text: String,
    pub(super) annotation_index: u64,
    pub(super) annotations: Vec<serde_json::Value>,
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

    pub(super) message: OpenAiResponsesMessageSerializeState,

    pub(super) reasoning_items_by_item_id:
        std::collections::HashMap<String, OpenAiResponsesReasoningItemSerializeState>,
    pub(super) fallback_reasoning_item_id: Option<String>,
    pub(super) latest_reasoning_item_id: Option<String>,

    pub(super) function_calls_by_call_id:
        std::collections::HashMap<String, OpenAiResponsesFunctionCallSerializeState>,

    /// Stable output indices for provider-hosted tool calls/results when the caller does not
    /// provide an explicit `outputIndex` stream-part field.
    pub(super) provider_tool_output_index_by_tool_call_id: std::collections::HashMap<String, u64>,
    pub(super) provider_tool_item_type_by_tool_call_id: std::collections::HashMap<String, String>,

    pub(super) latest_usage: Option<crate::types::Usage>,
    pub(super) latest_error_message: Option<String>,
}

/// Shared serializer state cell for OpenAI Responses SSE replay.
#[derive(Debug, Clone, Default)]
pub(super) struct OpenAiResponsesSerializeStateCell {
    inner: Arc<Mutex<OpenAiResponsesSerializeState>>,
}

impl OpenAiResponsesSerializeStateCell {
    pub(super) fn lock(
        &self,
    ) -> Result<MutexGuard<'_, OpenAiResponsesSerializeState>, crate::error::LlmError> {
        self.inner.lock().map_err(|_| {
            crate::error::LlmError::InternalError("serialize_state lock poisoned".to_string())
        })
    }
}

impl OpenAiResponsesSerializeState {
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(super) fn next_sequence_number(&mut self) -> u64 {
        let n = self.next_sequence_number;
        self.next_sequence_number = self.next_sequence_number.saturating_add(1);
        n
    }

    pub(super) fn alloc_output_index(&mut self) -> u64 {
        let mut candidate = self.next_output_index;
        loop {
            if !self.used_output_indices.contains(&candidate) {
                self.used_output_indices.insert(candidate);
                self.next_output_index = candidate.saturating_add(1);
                return candidate;
            }
            candidate = candidate.saturating_add(1);
        }
    }

    pub(super) fn alloc_or_reuse_output_index(&mut self, requested: Option<u64>) -> u64 {
        if let Some(idx) = requested
            && !self.used_output_indices.contains(&idx)
        {
            self.used_output_indices.insert(idx);
            self.next_output_index = std::cmp::max(self.next_output_index, idx.saturating_add(1));
            return idx;
        }
        self.alloc_output_index()
    }

    pub(super) fn provider_tool_output_index(
        &mut self,
        tool_call_id: Option<&str>,
        requested: Option<u64>,
    ) -> u64 {
        if let Some(req) = requested {
            let idx = self.alloc_or_reuse_output_index(Some(req));
            if let Some(id) = tool_call_id
                && !id.is_empty()
            {
                self.provider_tool_output_index_by_tool_call_id
                    .insert(id.to_string(), idx);
            }
            return idx;
        }

        if let Some(id) = tool_call_id
            && !id.is_empty()
        {
            if let Some(idx) = self.provider_tool_output_index_by_tool_call_id.get(id) {
                return *idx;
            }

            let idx = self.alloc_output_index();
            self.provider_tool_output_index_by_tool_call_id
                .insert(id.to_string(), idx);
            return idx;
        }

        self.alloc_output_index()
    }

    pub(super) fn ensure_reasoning_item(
        &mut self,
        requested_item_id: Option<&str>,
    ) -> (String, u64) {
        if let Some(item_id) = requested_item_id.filter(|s| !s.is_empty()) {
            if let Some(existing) = self.reasoning_items_by_item_id.get(item_id) {
                self.latest_reasoning_item_id = Some(item_id.to_string());
                return (item_id.to_string(), existing.output_index);
            }

            let output_index = self.alloc_output_index();
            self.reasoning_items_by_item_id.insert(
                item_id.to_string(),
                OpenAiResponsesReasoningItemSerializeState { output_index },
            );
            self.latest_reasoning_item_id = Some(item_id.to_string());
            return (item_id.to_string(), output_index);
        }

        if let Some(item_id) = self.latest_reasoning_item_id.clone()
            && let Some(existing) = self.reasoning_items_by_item_id.get(&item_id)
        {
            return (item_id, existing.output_index);
        }

        if let Some(item_id) = self.fallback_reasoning_item_id.clone() {
            if let Some(existing) = self.reasoning_items_by_item_id.get(&item_id) {
                self.latest_reasoning_item_id = Some(item_id.clone());
                return (item_id, existing.output_index);
            }

            let output_index = self.alloc_output_index();
            self.reasoning_items_by_item_id.insert(
                item_id.clone(),
                OpenAiResponsesReasoningItemSerializeState { output_index },
            );
            self.latest_reasoning_item_id = Some(item_id.clone());
            return (item_id, output_index);
        }

        let output_index = self.alloc_output_index();
        let item_id = format!("rs_siumai_{output_index}");
        self.reasoning_items_by_item_id.insert(
            item_id.clone(),
            OpenAiResponsesReasoningItemSerializeState { output_index },
        );
        self.fallback_reasoning_item_id = Some(item_id.clone());
        self.latest_reasoning_item_id = Some(item_id.clone());
        (item_id, output_index)
    }

    pub(super) fn ensure_function_call_state(
        &mut self,
        call_id: &str,
        output_index_seed: Option<u64>,
        default_name: Option<&str>,
    ) -> &mut OpenAiResponsesFunctionCallSerializeState {
        if self.function_calls_by_call_id.contains_key(call_id) {
            return self
                .function_calls_by_call_id
                .get_mut(call_id)
                .unwrap_or_else(|| unreachable!("function call state must exist"));
        }

        let output_index = output_index_seed.unwrap_or_else(|| self.alloc_output_index());
        self.function_calls_by_call_id.insert(
            call_id.to_string(),
            OpenAiResponsesFunctionCallSerializeState {
                item_id: format!("fc_siumai_{output_index}"),
                output_index,
                name: default_name.map(ToString::to_string),
                arguments: String::new(),
                arguments_done: false,
            },
        );
        self.function_calls_by_call_id
            .get_mut(call_id)
            .unwrap_or_else(|| unreachable!("function call state must exist"))
    }

    pub(super) fn ensure_message_item(
        &mut self,
        requested_item_id: Option<&str>,
        response_id_fallback: Option<&str>,
        prefer_requested_item_id: bool,
    ) -> (String, u64) {
        if self.message.output_index.is_none() {
            self.message.output_index = Some(self.alloc_output_index());
        }
        let output_index = self.message.output_index.unwrap_or(0);

        let requested_item_id = requested_item_id.filter(|item_id| !item_id.is_empty());
        if prefer_requested_item_id {
            if let Some(item_id) = requested_item_id {
                self.message.item_id = Some(item_id.to_string());
            }
        } else if self.message.item_id.is_none()
            && let Some(item_id) = requested_item_id
        {
            self.message.item_id = Some(item_id.to_string());
        }

        if self.message.item_id.is_none() {
            let fallback_id = response_id_fallback
                .filter(|response_id| !response_id.is_empty())
                .map(|response_id| format!("msg_{response_id}_0"))
                .unwrap_or_else(|| format!("msg_siumai_{output_index}"));
            self.message.item_id = Some(fallback_id);
        }

        (
            self.message
                .item_id
                .clone()
                .unwrap_or_else(|| format!("msg_siumai_{output_index}")),
            output_index,
        )
    }
}
