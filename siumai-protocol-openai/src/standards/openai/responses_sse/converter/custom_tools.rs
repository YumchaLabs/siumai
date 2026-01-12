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

    pub(super) fn convert_custom_tool_call_input_done(
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
}
