use super::*;

impl OpenAiResponsesEventConverter {
    pub(super) fn custom_tool_name_for_call_name(&self, call_name: &str) -> String {
        if call_name.is_empty() {
            return String::new();
        }

        self.custom_tool_name_by_call_name
            .lock()
            .ok()
            .and_then(|m| m.get(call_name).cloned())
            .unwrap_or_else(|| call_name.to_string())
    }

    pub(super) fn record_custom_tool_item(&self, item_id: &str, call_name: &str, tool_name: &str) {
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

    pub(super) fn custom_tool_name_by_item_id(&self, item_id: &str) -> Option<String> {
        self.custom_tool_tool_name_by_item_id
            .lock()
            .ok()
            .and_then(|m| m.get(item_id).cloned())
    }

    pub(super) fn record_custom_tool_input_delta(&self, item_id: &str, delta: &str) {
        if item_id.is_empty() {
            return;
        }

        let Ok(mut map) = self.custom_tool_input_by_item_id.lock() else {
            return;
        };

        map.entry(item_id.to_string())
            .and_modify(|input| input.push_str(delta))
            .or_insert_with(|| delta.to_string());
    }

    pub(super) fn record_custom_tool_input_done(&self, item_id: &str, input: Option<&str>) {
        if item_id.is_empty() {
            return;
        }

        let Some(input) = input else {
            return;
        };

        let Ok(mut map) = self.custom_tool_input_by_item_id.lock() else {
            return;
        };

        map.insert(item_id.to_string(), input.to_string());
    }

    pub(super) fn take_custom_tool_input_by_item_id(&self, item_id: &str) -> Option<String> {
        self.custom_tool_input_by_item_id
            .lock()
            .ok()
            .and_then(|mut m| m.remove(item_id))
    }

    pub(super) fn mark_custom_tool_input_start_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_custom_tool_input_start_ids.lock() else {
            return false;
        };
        set.insert(id.to_string())
    }

    pub(super) fn mark_custom_tool_input_end_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_custom_tool_input_end_ids.lock() else {
            return false;
        };
        set.insert(id.to_string())
    }

    pub(super) fn mark_custom_tool_call_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_custom_tool_call_ids.lock() else {
            return false;
        };
        set.insert(id.to_string())
    }

    pub(super) fn convert_custom_tool_call_input_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str())?;
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");

        self.record_custom_tool_input_delta(item_id, delta);

        if self.stream_parts_style == StreamPartsStyle::Xai {
            return None;
        }

        let mut out: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
        if !self.mark_custom_tool_input_start_emitted(item_id) {
            // If the start was already emitted, continue.
        } else if let Some(tool_name) = self.custom_tool_name_by_item_id(item_id)
            && !tool_name.is_empty()
        {
            out.push(
                self.openai_tool_input_start_event(item_id, &tool_name, None, None, None, None),
            );
        }

        out.push(self.openai_tool_input_delta_event(item_id, delta));

        Some(out)
    }

    pub(super) fn convert_custom_tool_call_input_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str())?;
        let input = json.get("input").and_then(|v| v.as_str());

        self.record_custom_tool_input_done(item_id, input);

        if self.stream_parts_style == StreamPartsStyle::Xai {
            return None;
        }

        if !self.mark_custom_tool_input_end_emitted(item_id) {
            return None;
        }

        Some(vec![self.openai_tool_input_end_event(item_id, None)])
    }
}
