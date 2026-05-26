use super::OpenAiResponsesEventConverter;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Owns the Responses reasoning item lifecycle across split SSE events.
#[derive(Debug, Clone, Default)]
pub(super) struct ReasoningLifecycleState {
    encrypted_content_by_item_id: Arc<Mutex<HashMap<String, Option<String>>>>,
    part_ids_by_item_id: Arc<Mutex<HashMap<String, HashSet<String>>>>,
    emitted_start_ids: Arc<Mutex<HashSet<String>>>,
    emitted_end_ids: Arc<Mutex<HashSet<String>>>,
    can_conclude_part_ids_by_item_id: Arc<Mutex<HashMap<String, HashSet<String>>>>,
}

impl ReasoningLifecycleState {
    fn record_encrypted_content(&self, item_id: &str, encrypted_content: Option<String>) {
        if item_id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.encrypted_content_by_item_id.lock() {
            map.insert(item_id.to_string(), encrypted_content);
        }
    }

    fn encrypted_content(&self, item_id: &str) -> Option<String> {
        let Ok(map) = self.encrypted_content_by_item_id.lock() else {
            return None;
        };
        map.get(item_id).cloned().unwrap_or(None)
    }

    fn mark_start_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_start_ids.lock() {
            set.insert(id.to_string());
        }
    }

    fn has_emitted_start(&self, id: &str) -> bool {
        self.emitted_start_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn record_part_id(&self, item_id: &str, id: &str) {
        if item_id.is_empty() || id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.part_ids_by_item_id.lock() {
            map.entry(item_id.to_string())
                .or_insert_with(HashSet::new)
                .insert(id.to_string());
        }
    }

    fn take_part_ids(&self, item_id: &str) -> Vec<String> {
        Self::take_sorted_item_ids(&self.part_ids_by_item_id, item_id)
    }

    fn mark_end_emitted(&self, id: &str) {
        if let Ok(mut set) = self.emitted_end_ids.lock() {
            set.insert(id.to_string());
        }
    }

    fn has_emitted_end(&self, id: &str) -> bool {
        self.emitted_end_ids
            .lock()
            .ok()
            .is_some_and(|set| set.contains(id))
    }

    fn mark_part_can_conclude(&self, item_id: &str, id: &str) {
        if item_id.is_empty() || id.is_empty() {
            return;
        }
        if let Ok(mut map) = self.can_conclude_part_ids_by_item_id.lock() {
            map.entry(item_id.to_string())
                .or_insert_with(HashSet::new)
                .insert(id.to_string());
        }
    }

    fn take_parts_can_conclude(&self, item_id: &str) -> Vec<String> {
        Self::take_sorted_item_ids(&self.can_conclude_part_ids_by_item_id, item_id)
    }

    fn take_sorted_item_ids(
        map: &Mutex<HashMap<String, HashSet<String>>>,
        item_id: &str,
    ) -> Vec<String> {
        if item_id.is_empty() {
            return Vec::new();
        }
        let Ok(mut map) = map.lock() else {
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

impl OpenAiResponsesEventConverter {
    pub(super) fn record_reasoning_encrypted_content(
        &self,
        item_id: &str,
        encrypted_content: Option<String>,
    ) {
        self.reasoning_lifecycle
            .record_encrypted_content(item_id, encrypted_content);
    }

    pub(super) fn reasoning_encrypted_content(&self, item_id: &str) -> Option<String> {
        self.reasoning_lifecycle.encrypted_content(item_id)
    }

    pub(super) fn mark_reasoning_start_emitted(&self, id: &str) {
        self.reasoning_lifecycle.mark_start_emitted(id);
    }

    pub(super) fn record_reasoning_part_id(&self, item_id: &str, id: &str) {
        self.reasoning_lifecycle.record_part_id(item_id, id);
    }

    pub(super) fn take_reasoning_part_ids(&self, item_id: &str) -> Vec<String> {
        self.reasoning_lifecycle.take_part_ids(item_id)
    }

    pub(super) fn has_emitted_reasoning_start(&self, id: &str) -> bool {
        self.reasoning_lifecycle.has_emitted_start(id)
    }

    pub(super) fn mark_reasoning_end_emitted(&self, id: &str) {
        self.reasoning_lifecycle.mark_end_emitted(id);
    }

    pub(super) fn has_emitted_reasoning_end(&self, id: &str) -> bool {
        self.reasoning_lifecycle.has_emitted_end(id)
    }

    pub(super) fn mark_reasoning_part_can_conclude(&self, item_id: &str, id: &str) {
        self.reasoning_lifecycle.mark_part_can_conclude(item_id, id);
    }

    pub(super) fn take_reasoning_parts_can_conclude(&self, item_id: &str) -> Vec<String> {
        self.reasoning_lifecycle.take_parts_can_conclude(item_id)
    }
}
