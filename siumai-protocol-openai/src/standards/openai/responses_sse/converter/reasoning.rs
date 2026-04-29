use super::*;

impl OpenAiResponsesEventConverter {
    pub(super) fn record_reasoning_encrypted_content(
        &self,
        item_id: &str,
        encrypted_content: Option<String>,
    ) {
        if item_id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.reasoning_encrypted_content_by_item_id.lock() {
            map.insert(item_id.to_string(), encrypted_content);
        }
    }

    pub(super) fn reasoning_encrypted_content(&self, item_id: &str) -> Option<String> {
        let Ok(map) = self.reasoning_encrypted_content_by_item_id.lock() else {
            return None;
        };
        map.get(item_id).cloned().unwrap_or(None)
    }

    pub(super) fn mark_reasoning_start_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_reasoning_start_ids.lock() {
            set.insert(id.to_string());
        }
    }

    pub(super) fn record_reasoning_part_id(&self, item_id: &str, id: &str) {
        if item_id.is_empty() || id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.reasoning_part_ids_by_item_id.lock() {
            map.entry(item_id.to_string())
                .or_insert_with(std::collections::HashSet::new)
                .insert(id.to_string());
        }
    }

    pub(super) fn take_reasoning_part_ids(&self, item_id: &str) -> Vec<String> {
        if item_id.is_empty() {
            return Vec::new();
        }
        let Ok(mut map) = self.reasoning_part_ids_by_item_id.lock() else {
            return Vec::new();
        };
        let Some(ids) = map.remove(item_id) else {
            return Vec::new();
        };
        let mut ids: Vec<String> = ids.into_iter().collect();
        ids.sort();
        ids
    }

    pub(super) fn has_emitted_reasoning_start(&self, id: &str) -> bool {
        self.emitted_reasoning_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    pub(super) fn mark_reasoning_end_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_reasoning_end_ids.lock() {
            set.insert(id.to_string());
        }
    }

    pub(super) fn has_emitted_reasoning_end(&self, id: &str) -> bool {
        self.emitted_reasoning_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    pub(super) fn mark_reasoning_part_can_conclude(&self, item_id: &str, id: &str) {
        if item_id.is_empty() || id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.can_conclude_reasoning_part_ids_by_item_id.lock() {
            map.entry(item_id.to_string())
                .or_insert_with(std::collections::HashSet::new)
                .insert(id.to_string());
        }
    }

    pub(super) fn take_reasoning_parts_can_conclude(&self, item_id: &str) -> Vec<String> {
        if item_id.is_empty() {
            return Vec::new();
        }
        let Ok(mut map) = self.can_conclude_reasoning_part_ids_by_item_id.lock() else {
            return Vec::new();
        };
        let Some(ids) = map.remove(item_id) else {
            return Vec::new();
        };
        let mut ids: Vec<String> = ids.into_iter().collect();
        ids.sort();
        ids
    }
}
