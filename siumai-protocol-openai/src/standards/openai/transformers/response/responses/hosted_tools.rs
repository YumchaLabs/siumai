//! Hosted and dynamic OpenAI Responses output item helpers.

pub(super) fn xai_file_search_queries(item: &serde_json::Value) -> serde_json::Value {
    item.get("queries")
        .filter(|value| !value.is_null())
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Array(Vec::new()))
}

pub(super) fn file_search_results(item: &serde_json::Value) -> serde_json::Value {
    let Some(results) = item.get("results") else {
        return serde_json::Value::Null;
    };
    let Some(results) = results.as_array() else {
        return results.clone();
    };

    serde_json::Value::Array(
        results
            .iter()
            .map(|result| {
                let mut out = serde_json::Map::new();
                if let Some(file_id) = result.get("file_id").or_else(|| result.get("fileId"))
                    && !file_id.is_null()
                {
                    out.insert("fileId".to_string(), file_id.clone());
                }
                if let Some(filename) = result.get("filename")
                    && !filename.is_null()
                {
                    out.insert("filename".to_string(), filename.clone());
                }
                if let Some(attributes) = result.get("attributes")
                    && !attributes.is_null()
                {
                    out.insert("attributes".to_string(), attributes.clone());
                }
                if let Some(score) = result.get("score")
                    && !score.is_null()
                {
                    out.insert("score".to_string(), score.clone());
                }
                if let Some(text) = result.get("text")
                    && !text.is_null()
                {
                    out.insert("text".to_string(), text.clone());
                }
                serde_json::Value::Object(out)
            })
            .collect(),
    )
}

fn json_string(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn ordered_object_json(
    object: &serde_json::Map<String, serde_json::Value>,
    ordered_keys: &[&str],
) -> String {
    let mut fields = Vec::new();
    for key in ordered_keys {
        if let Some(value) = object.get(*key) {
            fields.push(format!("\"{key}\":{}", json_string(value)));
        }
    }
    for (key, value) in object {
        if !ordered_keys.contains(&key.as_str()) {
            let key = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
            fields.push(format!("{key}:{}", json_string(value)));
        }
    }

    format!("{{{}}}", fields.join(","))
}

pub(super) fn local_shell_generate_input(item: &serde_json::Value) -> String {
    const ACTION_KEYS: [&str; 6] = [
        "type",
        "command",
        "timeout_ms",
        "user",
        "working_directory",
        "env",
    ];

    let action = item.get("action").unwrap_or(&serde_json::Value::Null);
    let Some(action_obj) = action.as_object() else {
        return format!("{{\"action\":{}}}", json_string(action));
    };

    format!(
        "{{\"action\":{}}}",
        ordered_object_json(action_obj, &ACTION_KEYS)
    )
}

pub(super) fn apply_patch_generate_input(call_id: &str, item: &serde_json::Value) -> String {
    const OPERATION_KEYS: [&str; 3] = ["type", "path", "diff"];

    let call_id_json = serde_json::to_string(call_id).unwrap_or_else(|_| "\"\"".to_string());
    let operation = item.get("operation").unwrap_or(&serde_json::Value::Null);
    let operation_json = operation
        .as_object()
        .map(|object| ordered_object_json(object, &OPERATION_KEYS))
        .unwrap_or_else(|| json_string(operation));

    format!("{{\"callId\":{call_id_json},\"operation\":{operation_json}}}")
}

fn shell_environment_is_provider_executed(value: &serde_json::Value) -> bool {
    let environment = value.get("environment").unwrap_or(value);
    let Some(environment_type) = environment.get("type").and_then(|value| value.as_str()) else {
        return false;
    };

    matches!(
        environment_type,
        "containerAuto" | "containerReference" | "container_auto" | "container_reference"
    )
}

pub(super) fn response_shell_call_provider_executed(root: &serde_json::Value) -> bool {
    root.get("tools")
        .and_then(|value| value.as_array())
        .is_some_and(|tools| {
            tools.iter().any(|tool| {
                tool.get("type").and_then(|value| value.as_str()) == Some("shell")
                    && shell_environment_is_provider_executed(tool)
            })
        })
}
