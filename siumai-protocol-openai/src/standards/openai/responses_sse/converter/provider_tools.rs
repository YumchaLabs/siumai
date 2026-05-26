use super::OpenAiResponsesEventConverter;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

pub(super) fn shell_environment_is_provider_executed(value: &serde_json::Value) -> bool {
    let environment = value.get("environment").unwrap_or(value);
    let Some(environment_type) = environment.get("type").and_then(|value| value.as_str()) else {
        return false;
    };

    matches!(
        environment_type,
        "containerAuto" | "containerReference" | "container_auto" | "container_reference"
    )
}

/// Owns provider-defined tool naming, execution ownership, and hosted tool-search pairing.
#[derive(Debug, Clone, Default)]
pub(super) struct ProviderToolState {
    name_by_item_type: Arc<Mutex<HashMap<String, String>>>,
    shell_call_provider_executed: Arc<Mutex<bool>>,
    web_search_tool_input_emitted_ids: Arc<Mutex<HashSet<String>>>,
    hosted_tool_search_call_ids: Arc<Mutex<VecDeque<String>>>,
    emitted_tool_search_input_start_ids: Arc<Mutex<HashSet<String>>>,
}

impl ProviderToolState {
    fn set_tool_name(&self, item_type: &str, name: &str) {
        if item_type.is_empty() || name.is_empty() {
            return;
        }
        if let Ok(mut map) = self.name_by_item_type.lock() {
            map.insert(item_type.to_string(), name.to_string());
        }
    }

    fn set_tool_name_if_absent(&self, item_type: &str, name: &str) {
        if item_type.is_empty() || name.is_empty() {
            return;
        }
        if let Ok(mut map) = self.name_by_item_type.lock() {
            map.entry(item_type.to_string())
                .or_insert_with(|| name.to_string());
        }
    }

    fn tool_name_for_item_type(&self, item_type: &str) -> Option<String> {
        let map = self.name_by_item_type.lock().ok()?;
        map.get(item_type).cloned()
    }

    fn mark_shell_call_provider_executed(&self) {
        if let Ok(mut value) = self.shell_call_provider_executed.lock() {
            *value = true;
        }
    }

    fn shell_call_provider_executed(&self) -> bool {
        self.shell_call_provider_executed
            .lock()
            .ok()
            .is_some_and(|value| *value)
    }

    fn mark_web_search_tool_input_emitted(&self, id: &str) -> bool {
        let Ok(mut set) = self.web_search_tool_input_emitted_ids.lock() else {
            return false;
        };
        set.insert(id.to_string())
    }

    fn mark_tool_search_input_start_emitted(&self, id: &str) -> bool {
        let Ok(mut ids) = self.emitted_tool_search_input_start_ids.lock() else {
            return false;
        };
        ids.insert(id.to_string())
    }

    fn has_tool_search_input_start_emitted(&self, id: &str) -> bool {
        self.emitted_tool_search_input_start_ids
            .lock()
            .ok()
            .is_some_and(|ids| ids.contains(id))
    }

    fn push_hosted_tool_search_call_id(&self, id: &str) {
        if id.is_empty() {
            return;
        }
        if let Ok(mut ids) = self.hosted_tool_search_call_ids.lock() {
            ids.push_back(id.to_string());
        }
    }

    fn pop_hosted_tool_search_call_id(&self) -> Option<String> {
        self.hosted_tool_search_call_ids
            .lock()
            .ok()
            .and_then(|mut ids| ids.pop_front())
    }
}

impl OpenAiResponsesEventConverter {
    pub(super) fn mark_shell_call_provider_executed(&self) {
        self.provider_tools.mark_shell_call_provider_executed();
    }

    pub(super) fn shell_call_provider_executed(&self) -> bool {
        self.provider_tools.shell_call_provider_executed()
    }

    pub(super) fn seed_provider_tool_names_from_request_tools(&self, tools: &[crate::types::Tool]) {
        use crate::types::Tool;

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
                    self.provider_tools
                        .set_tool_name("web_search_call", &t.name);
                }
                "web_search" => {
                    self.provider_tools
                        .set_tool_name_if_absent("web_search_call", &t.name);
                }
                // xAI vendor mapping: code execution is exposed as `code_interpreter_call` items.
                "code_execution" => {
                    self.provider_tools
                        .set_tool_name_if_absent("code_interpreter_call", &t.name);
                }
                // xAI vendor mapping: x_search triggers internal `custom_tool_call` items
                // (e.g. `x_keyword_search`) that should map back to the client tool name.
                "x_search" => {
                    self.record_custom_tool_name_for_call_name("x_keyword_search", &t.name);
                }
                "file_search" => {
                    self.provider_tools
                        .set_tool_name_if_absent("file_search_call", &t.name);
                }
                "code_interpreter" => {
                    self.provider_tools
                        .set_tool_name_if_absent("code_interpreter_call", &t.name);
                }
                "image_generation" => {
                    self.provider_tools
                        .set_tool_name_if_absent("image_generation_call", &t.name);
                }
                "local_shell" => {
                    self.provider_tools
                        .set_tool_name_if_absent("local_shell_call", &t.name);
                    self.provider_tools
                        .set_tool_name_if_absent("local_shell_call_output", &t.name);
                }
                "shell" => {
                    self.provider_tools
                        .set_tool_name_if_absent("shell_call", &t.name);
                    self.provider_tools
                        .set_tool_name_if_absent("shell_call_output", &t.name);
                    if shell_environment_is_provider_executed(&t.args) {
                        self.mark_shell_call_provider_executed();
                    }
                }
                "apply_patch" => {
                    self.provider_tools
                        .set_tool_name_if_absent("apply_patch_call", &t.name);
                    self.provider_tools
                        .set_tool_name_if_absent("apply_patch_call_output", &t.name);
                }
                "tool_search" => {
                    self.provider_tools
                        .set_tool_name_if_absent("tool_search_call", &t.name);
                    self.provider_tools
                        .set_tool_name_if_absent("tool_search_output", &t.name);
                }
                "computer_use_preview" => {
                    self.provider_tools.set_tool_name("computer_call", &t.name);
                }
                "computer_use" => {
                    self.provider_tools
                        .set_tool_name_if_absent("computer_call", &t.name);
                }
                _ => {}
            }
        }
    }

    pub(super) fn update_provider_tool_names(&self, json: &serde_json::Value) {
        let Some(tools) = json
            .get("response")
            .and_then(|r| r.get("tools"))
            .and_then(|t| t.as_array())
        else {
            return;
        };

        for tool in tools {
            let Some(tool_type) = tool.get("type").and_then(|v| v.as_str()) else {
                continue;
            };

            match tool_type {
                // Fallback mapping: if request-level tool names were not provided,
                // use the configured tool type as `toolName`.
                "web_search_preview" => {
                    self.provider_tools
                        .set_tool_name_if_absent("web_search_call", "web_search_preview");
                }
                "web_search" => {
                    self.provider_tools
                        .set_tool_name_if_absent("web_search_call", "web_search");
                }
                "file_search" => {
                    self.provider_tools
                        .set_tool_name_if_absent("file_search_call", "file_search");
                }
                "code_interpreter" => {
                    self.provider_tools
                        .set_tool_name_if_absent("code_interpreter_call", "code_interpreter");
                }
                "image_generation" => {
                    self.provider_tools
                        .set_tool_name_if_absent("image_generation_call", "image_generation");
                }
                "local_shell" => {
                    self.provider_tools
                        .set_tool_name_if_absent("local_shell_call", "shell");
                    self.provider_tools
                        .set_tool_name_if_absent("local_shell_call_output", "shell");
                }
                "shell" => {
                    self.provider_tools
                        .set_tool_name_if_absent("shell_call", "shell");
                    self.provider_tools
                        .set_tool_name_if_absent("shell_call_output", "shell");
                    if shell_environment_is_provider_executed(tool) {
                        self.mark_shell_call_provider_executed();
                    }
                }
                "apply_patch" => {
                    self.provider_tools
                        .set_tool_name_if_absent("apply_patch_call", "apply_patch");
                    self.provider_tools
                        .set_tool_name_if_absent("apply_patch_call_output", "apply_patch");
                }
                "tool_search" => {
                    self.provider_tools
                        .set_tool_name_if_absent("tool_search_call", "toolSearch");
                    self.provider_tools
                        .set_tool_name_if_absent("tool_search_output", "toolSearch");
                }
                "computer_use_preview" => {
                    self.provider_tools
                        .set_tool_name_if_absent("computer_call", "computer_use_preview");
                }
                "computer_use" => {
                    self.provider_tools
                        .set_tool_name_if_absent("computer_call", "computer_use");
                }
                _ => {}
            }
        }
    }

    pub(super) fn provider_tool_name_for_item_type(&self, item_type: &str) -> Option<String> {
        self.provider_tools.tool_name_for_item_type(item_type)
    }

    pub(super) fn mark_web_search_tool_input_emitted(&self, id: &str) -> bool {
        self.provider_tools.mark_web_search_tool_input_emitted(id)
    }

    pub(super) fn mark_tool_search_input_start_emitted(&self, id: &str) -> bool {
        self.provider_tools.mark_tool_search_input_start_emitted(id)
    }

    pub(super) fn has_tool_search_input_start_emitted(&self, id: &str) -> bool {
        self.provider_tools.has_tool_search_input_start_emitted(id)
    }

    pub(super) fn push_hosted_tool_search_call_id(&self, id: &str) {
        self.provider_tools.push_hosted_tool_search_call_id(id);
    }

    pub(super) fn pop_hosted_tool_search_call_id(&self) -> Option<String> {
        self.provider_tools.pop_hosted_tool_search_call_id()
    }
}
