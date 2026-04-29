use super::*;

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

impl OpenAiResponsesEventConverter {
    pub(super) fn mark_shell_call_provider_executed(&self) {
        if let Ok(mut value) = self.shell_call_provider_executed.lock() {
            *value = true;
        }
    }

    pub(super) fn shell_call_provider_executed(&self) -> bool {
        self.shell_call_provider_executed
            .lock()
            .ok()
            .is_some_and(|value| *value)
    }

    pub(super) fn seed_provider_tool_names_from_request_tools(&self, tools: &[crate::types::Tool]) {
        use crate::types::Tool;

        let mut map = match self.provider_tool_name_by_item_type.lock() {
            Ok(m) => m,
            Err(_) => return,
        };

        let mut custom_tool_name_by_call_name = match self.custom_tool_name_by_call_name.lock() {
            Ok(m) => m,
            Err(_) => return,
        };

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
                    map.insert("web_search_call".to_string(), t.name.clone());
                }
                "web_search" => {
                    map.entry("web_search_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                // xAI vendor mapping: code execution is exposed as `code_interpreter_call` items.
                "code_execution" => {
                    map.entry("code_interpreter_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                // xAI vendor mapping: x_search triggers internal `custom_tool_call` items
                // (e.g. `x_keyword_search`) that should map back to the client tool name.
                "x_search" => {
                    custom_tool_name_by_call_name
                        .entry("x_keyword_search".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "file_search" => {
                    map.entry("file_search_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "code_interpreter" => {
                    map.entry("code_interpreter_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "image_generation" => {
                    map.entry("image_generation_call".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "local_shell" => {
                    map.entry("local_shell_call".to_string())
                        .or_insert_with(|| t.name.clone());
                    map.entry("local_shell_call_output".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "shell" => {
                    map.entry("shell_call".to_string())
                        .or_insert_with(|| t.name.clone());
                    map.entry("shell_call_output".to_string())
                        .or_insert_with(|| t.name.clone());
                    if shell_environment_is_provider_executed(&t.args) {
                        self.mark_shell_call_provider_executed();
                    }
                }
                "apply_patch" => {
                    map.entry("apply_patch_call".to_string())
                        .or_insert_with(|| t.name.clone());
                    map.entry("apply_patch_call_output".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "tool_search" => {
                    map.entry("tool_search_call".to_string())
                        .or_insert_with(|| t.name.clone());
                    map.entry("tool_search_output".to_string())
                        .or_insert_with(|| t.name.clone());
                }
                "computer_use_preview" => {
                    map.insert("computer_call".to_string(), t.name.clone());
                }
                "computer_use" => {
                    map.entry("computer_call".to_string())
                        .or_insert_with(|| t.name.clone());
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

        let mut map = match self.provider_tool_name_by_item_type.lock() {
            Ok(m) => m,
            Err(_) => return,
        };

        for tool in tools {
            let Some(tool_type) = tool.get("type").and_then(|v| v.as_str()) else {
                continue;
            };

            match tool_type {
                // Fallback mapping: if request-level tool names were not provided,
                // use the configured tool type as `toolName`.
                "web_search_preview" => {
                    map.entry("web_search_call".to_string())
                        .or_insert_with(|| "web_search_preview".to_string());
                }
                "web_search" => {
                    map.entry("web_search_call".to_string())
                        .or_insert_with(|| "web_search".to_string());
                }
                "file_search" => {
                    map.entry("file_search_call".to_string())
                        .or_insert_with(|| "file_search".to_string());
                }
                "code_interpreter" => {
                    map.entry("code_interpreter_call".to_string())
                        .or_insert_with(|| "code_interpreter".to_string());
                }
                "image_generation" => {
                    map.entry("image_generation_call".to_string())
                        .or_insert_with(|| "image_generation".to_string());
                }
                "local_shell" => {
                    map.entry("local_shell_call".to_string())
                        .or_insert_with(|| "shell".to_string());
                    map.entry("local_shell_call_output".to_string())
                        .or_insert_with(|| "shell".to_string());
                }
                "shell" => {
                    map.entry("shell_call".to_string())
                        .or_insert_with(|| "shell".to_string());
                    map.entry("shell_call_output".to_string())
                        .or_insert_with(|| "shell".to_string());
                    if shell_environment_is_provider_executed(tool) {
                        self.mark_shell_call_provider_executed();
                    }
                }
                "apply_patch" => {
                    map.entry("apply_patch_call".to_string())
                        .or_insert_with(|| "apply_patch".to_string());
                    map.entry("apply_patch_call_output".to_string())
                        .or_insert_with(|| "apply_patch".to_string());
                }
                "tool_search" => {
                    map.entry("tool_search_call".to_string())
                        .or_insert_with(|| "toolSearch".to_string());
                    map.entry("tool_search_output".to_string())
                        .or_insert_with(|| "toolSearch".to_string());
                }
                "computer_use_preview" => {
                    map.entry("computer_call".to_string())
                        .or_insert_with(|| "computer_use_preview".to_string());
                }
                "computer_use" => {
                    map.entry("computer_call".to_string())
                        .or_insert_with(|| "computer_use".to_string());
                }
                _ => {}
            }
        }
    }

    pub(super) fn provider_tool_name_for_item_type(&self, item_type: &str) -> Option<String> {
        let map = self.provider_tool_name_by_item_type.lock().ok()?;
        map.get(item_type).cloned()
    }
}
