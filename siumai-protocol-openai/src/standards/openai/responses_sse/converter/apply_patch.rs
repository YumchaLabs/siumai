use super::*;

impl OpenAiResponsesEventConverter {
    pub(super) fn record_apply_patch_call(
        &self,
        item_id: &str,
        call_id: &str,
        operation: serde_json::Value,
    ) {
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

    pub(super) fn apply_patch_call_id(&self, item_id: &str) -> Option<String> {
        let map = self.apply_patch_call_id_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    #[allow(dead_code)]
    pub(super) fn apply_patch_operation(&self, item_id: &str) -> Option<serde_json::Value> {
        let map = self.apply_patch_operation_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    pub(super) fn mark_apply_patch_tool_input_start_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_apply_patch_tool_input_start_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    pub(super) fn has_emitted_apply_patch_tool_input_start(&self, id: &str) -> bool {
        self.emitted_apply_patch_tool_input_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    pub(super) fn mark_apply_patch_tool_input_end_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_apply_patch_tool_input_end_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    pub(super) fn has_emitted_apply_patch_tool_input_end(&self, id: &str) -> bool {
        self.emitted_apply_patch_tool_input_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }
}
