use super::*;

impl OpenAiResponsesEventConverter {
    pub(super) fn record_mcp_call_added(&self, item_id: &str, name: &str, server_label: &str) {
        let Ok(mut map) = self.mcp_calls_by_item_id.lock() else {
            return;
        };
        map.insert(
            item_id.to_string(),
            (name.to_string(), server_label.to_string()),
        );
    }

    pub(super) fn record_mcp_call_args(&self, item_id: &str, args: &str) {
        let Ok(mut map) = self.mcp_call_args_by_item_id.lock() else {
            return;
        };
        map.insert(item_id.to_string(), args.to_string());
    }

    pub(super) fn mcp_call_meta(&self, item_id: &str) -> Option<(String, String)> {
        let map = self.mcp_calls_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    pub(super) fn mcp_call_args(&self, item_id: &str) -> Option<String> {
        let map = self.mcp_call_args_by_item_id.lock().ok()?;
        map.get(item_id).cloned()
    }

    pub(super) fn mark_mcp_call_emitted(&self, item_id: &str) {
        if let Ok(mut set) = self.emitted_mcp_call_ids.lock() {
            set.insert(item_id.to_string());
        }
    }

    pub(super) fn mark_mcp_result_emitted(&self, item_id: &str) {
        if let Ok(mut set) = self.emitted_mcp_result_ids.lock() {
            set.insert(item_id.to_string());
        }
    }

    pub(super) fn has_emitted_mcp_call(&self, item_id: &str) -> bool {
        self.emitted_mcp_call_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(item_id))
    }

    pub(super) fn has_emitted_mcp_result(&self, item_id: &str) -> bool {
        self.emitted_mcp_result_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(item_id))
    }

    pub(super) fn mcp_approval_tool_call_id(&self, approval_id: &str) -> String {
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

    pub(super) fn mark_mcp_approval_request_emitted(&self, approval_id: &str) {
        if let Ok(mut set) = self.emitted_mcp_approval_request_ids.lock() {
            set.insert(approval_id.to_string());
        }
    }

    pub(super) fn has_emitted_mcp_approval_request(&self, approval_id: &str) -> bool {
        self.emitted_mcp_approval_request_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(approval_id))
    }
}
