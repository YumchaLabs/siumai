use super::*;

impl OpenAiResponsesEventConverter {
    pub(super) fn mark_web_search_tool_input_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.emitted_web_search_tool_input_ids.lock() else {
            return false;
        };
        if set.contains(id) {
            return false;
        }
        set.insert(id.to_string());
        true
    }

    pub(super) fn mark_stream_start_emitted(&self) -> bool {
        let Ok(mut emitted) = self.emitted_stream_start.lock() else {
            return false;
        };
        if *emitted {
            return false;
        }
        *emitted = true;
        true
    }

    pub(super) fn mark_response_metadata_emitted(&self, response_id: &str) -> bool {
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

    pub(super) fn record_created_response_metadata(
        &self,
        response_id: &str,
        model_id: &str,
        created_at: i64,
    ) {
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

    pub(super) fn created_response_id(&self) -> Option<String> {
        self.created_response_id.lock().ok().and_then(|v| v.clone())
    }

    pub(super) fn created_model_id(&self) -> Option<String> {
        self.created_model_id.lock().ok().and_then(|v| v.clone())
    }

    pub(super) fn created_timestamp_rfc3339_millis(&self) -> Option<String> {
        let created_at = self.created_created_at.lock().ok().and_then(|v| *v)?;
        Utc.timestamp_opt(created_at, 0)
            .single()
            .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Millis, true))
    }

    pub(super) fn record_message_item_id(&self, output_index: u64, item_id: &str) {
        if item_id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.message_item_id_by_output_index.lock() {
            map.insert(output_index, item_id.to_string());
        }
    }

    pub(super) fn message_item_id_for_output_index(&self, output_index: u64) -> Option<String> {
        let map = self.message_item_id_by_output_index.lock().ok()?;
        map.get(&output_index).cloned()
    }

    pub(super) fn mark_text_start_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_text_start_ids.lock() {
            set.insert(id.to_string());
        }
    }

    pub(super) fn has_emitted_text_start(&self, id: &str) -> bool {
        self.emitted_text_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    pub(super) fn mark_text_end_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_text_end_ids.lock() {
            set.insert(id.to_string());
        }
    }

    pub(super) fn has_emitted_text_end(&self, id: &str) -> bool {
        self.emitted_text_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    pub(super) fn record_text_annotation(&self, item_id: &str, annotation: serde_json::Value) {
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

    pub(super) fn take_text_annotations(&self, item_id: &str) -> Vec<serde_json::Value> {
        let Ok(mut map) = self.text_annotations_by_item_id.lock() else {
            return Vec::new();
        };
        map.remove(item_id).unwrap_or_default()
    }
}
