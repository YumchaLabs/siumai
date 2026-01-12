use super::*;

impl OpenAiResponsesEventConverter {
    pub(super) fn record_function_call_meta(&self, item_id: &str, call_id: &str, name: &str) {
        if item_id.is_empty() || call_id.is_empty() || name.is_empty() {
            return;
        }
        if let Ok(mut map) = self.function_call_meta_by_item_id.lock() {
            map.insert(item_id.to_string(), (call_id.to_string(), name.to_string()));
        }
    }

    pub(super) fn function_call_meta(&self, item_id: &str) -> Option<(String, String)> {
        let map = self.function_call_meta_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    pub(super) fn mark_function_tool_input_start_emitted(&self, id: &str) -> bool {
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
    pub(super) fn has_emitted_function_tool_input_start(&self, id: &str) -> bool {
        self.emitted_function_tool_input_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    pub(super) fn mark_function_tool_input_end_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_function_tool_input_end_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    pub(super) fn has_emitted_function_tool_input_end(&self, id: &str) -> bool {
        self.emitted_function_tool_input_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }
}
