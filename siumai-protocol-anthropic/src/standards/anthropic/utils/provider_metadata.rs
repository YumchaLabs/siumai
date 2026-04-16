pub fn map_usage_iterations_provider_metadata(usage: &serde_json::Value) -> serde_json::Value {
    let Some(iterations) = usage.get("iterations").and_then(|value| value.as_array()) else {
        return serde_json::Value::Null;
    };

    serde_json::Value::Array(
        iterations
            .iter()
            .map(|iteration| {
                let Some(iteration) = iteration.as_object() else {
                    return iteration.clone();
                };

                let mut mapped = serde_json::Map::new();
                if let Some(kind) = iteration.get("type") {
                    mapped.insert("type".to_string(), kind.clone());
                }
                mapped.insert(
                    "inputTokens".to_string(),
                    iteration
                        .get("input_tokens")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                );
                mapped.insert(
                    "outputTokens".to_string(),
                    iteration
                        .get("output_tokens")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                );
                serde_json::Value::Object(mapped)
            })
            .collect(),
    )
}

pub fn map_container_provider_metadata(v: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = v.as_object()?;
    let mut out = serde_json::Map::new();

    if let Some(id) = obj.get("id") {
        out.insert("id".to_string(), id.clone());
    }

    out.insert(
        "expiresAt".to_string(),
        obj.get("expires_at")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    );

    let skills = obj
        .get("skills")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let mapped_skills = match skills {
        serde_json::Value::Array(arr) => {
            let mut out_arr: Vec<serde_json::Value> = Vec::new();
            for item in arr {
                let Some(skill) = item.as_object() else {
                    out_arr.push(item);
                    continue;
                };
                let mut out_skill = serde_json::Map::new();
                if let Some(t) = skill.get("type") {
                    out_skill.insert("type".to_string(), t.clone());
                }
                out_skill.insert(
                    "skillId".to_string(),
                    skill
                        .get("skill_id")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                );
                if let Some(v) = skill.get("version") {
                    out_skill.insert("version".to_string(), v.clone());
                }
                out_arr.push(serde_json::Value::Object(out_skill));
            }
            serde_json::Value::Array(out_arr)
        }
        other => other,
    };
    out.insert("skills".to_string(), mapped_skills);

    Some(serde_json::Value::Object(out))
}

pub fn map_context_management_provider_metadata(
    v: &serde_json::Value,
) -> Option<serde_json::Value> {
    let obj = v.as_object()?;
    let applied_edits = obj.get("applied_edits")?.as_array()?;

    let mut out_edits: Vec<serde_json::Value> = Vec::new();
    for edit in applied_edits {
        let Some(edit_obj) = edit.as_object() else {
            continue;
        };
        let Some(ty) = edit_obj.get("type").and_then(|v| v.as_str()) else {
            continue;
        };

        match ty {
            "clear_tool_uses_20250919" => {
                let mut mapped = serde_json::Map::new();
                mapped.insert(
                    "type".to_string(),
                    serde_json::Value::String(ty.to_string()),
                );
                mapped.insert(
                    "clearedToolUses".to_string(),
                    edit_obj
                        .get("cleared_tool_uses")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                );
                mapped.insert(
                    "clearedInputTokens".to_string(),
                    edit_obj
                        .get("cleared_input_tokens")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                );
                out_edits.push(serde_json::Value::Object(mapped));
            }
            "clear_thinking_20251015" => {
                let mut mapped = serde_json::Map::new();
                mapped.insert(
                    "type".to_string(),
                    serde_json::Value::String(ty.to_string()),
                );
                mapped.insert(
                    "clearedThinkingTurns".to_string(),
                    edit_obj
                        .get("cleared_thinking_turns")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                );
                mapped.insert(
                    "clearedInputTokens".to_string(),
                    edit_obj
                        .get("cleared_input_tokens")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                );
                out_edits.push(serde_json::Value::Object(mapped));
            }
            "compact_20260112" => {
                let mut mapped = serde_json::Map::new();
                mapped.insert(
                    "type".to_string(),
                    serde_json::Value::String(ty.to_string()),
                );
                out_edits.push(serde_json::Value::Object(mapped));
            }
            _ => {}
        }
    }

    Some(serde_json::json!({ "appliedEdits": out_edits }))
}

pub fn raw_container_from_provider_metadata(
    value: &serde_json::Value,
) -> Option<serde_json::Value> {
    let obj = value.as_object()?;
    let mut out = serde_json::Map::new();

    if let Some(id) = obj.get("id") {
        out.insert("id".to_string(), id.clone());
    }
    if let Some(expires_at) = obj.get("expiresAt") {
        out.insert("expires_at".to_string(), expires_at.clone());
    }
    if let Some(skills) = obj.get("skills").and_then(|value| value.as_array()) {
        let mut out_skills = Vec::with_capacity(skills.len());
        for item in skills {
            let Some(skill) = item.as_object() else {
                out_skills.push(item.clone());
                continue;
            };

            let mut out_skill = serde_json::Map::new();
            if let Some(skill_type) = skill.get("type") {
                out_skill.insert("type".to_string(), skill_type.clone());
            }
            if let Some(skill_id) = skill.get("skillId") {
                out_skill.insert("skill_id".to_string(), skill_id.clone());
            }
            if let Some(version) = skill.get("version") {
                out_skill.insert("version".to_string(), version.clone());
            }
            out_skills.push(serde_json::Value::Object(out_skill));
        }
        out.insert("skills".to_string(), serde_json::Value::Array(out_skills));
    }

    Some(serde_json::Value::Object(out))
}

pub fn raw_context_management_from_provider_metadata(
    value: &serde_json::Value,
) -> Option<serde_json::Value> {
    let obj = value.as_object()?;
    let edits = obj.get("appliedEdits")?.as_array()?;
    let mut out_edits = Vec::with_capacity(edits.len());

    for edit in edits {
        let Some(edit_obj) = edit.as_object() else {
            continue;
        };
        let Some(edit_type) = edit_obj.get("type").and_then(|value| value.as_str()) else {
            continue;
        };

        match edit_type {
            "clear_tool_uses_20250919" => {
                let mut out = serde_json::Map::new();
                out.insert(
                    "type".to_string(),
                    serde_json::Value::String(edit_type.to_string()),
                );
                if let Some(value) = edit_obj.get("clearedToolUses") {
                    out.insert("cleared_tool_uses".to_string(), value.clone());
                }
                if let Some(value) = edit_obj.get("clearedInputTokens") {
                    out.insert("cleared_input_tokens".to_string(), value.clone());
                }
                out_edits.push(serde_json::Value::Object(out));
            }
            "clear_thinking_20251015" => {
                let mut out = serde_json::Map::new();
                out.insert(
                    "type".to_string(),
                    serde_json::Value::String(edit_type.to_string()),
                );
                if let Some(value) = edit_obj.get("clearedThinkingTurns") {
                    out.insert("cleared_thinking_turns".to_string(), value.clone());
                }
                if let Some(value) = edit_obj.get("clearedInputTokens") {
                    out.insert("cleared_input_tokens".to_string(), value.clone());
                }
                out_edits.push(serde_json::Value::Object(out));
            }
            "compact_20260112" => {
                let mut out = serde_json::Map::new();
                out.insert(
                    "type".to_string(),
                    serde_json::Value::String(edit_type.to_string()),
                );
                out_edits.push(serde_json::Value::Object(out));
            }
            _ => {}
        }
    }

    Some(serde_json::json!({ "applied_edits": out_edits }))
}

#[cfg(test)]
mod tests {
    use super::{
        map_context_management_provider_metadata, raw_context_management_from_provider_metadata,
    };

    #[test]
    fn context_management_provider_metadata_preserves_compact_edit() {
        let mapped = map_context_management_provider_metadata(&serde_json::json!({
            "applied_edits": [
                { "type": "compact_20260112" }
            ]
        }))
        .expect("mapped context management");

        assert_eq!(
            mapped,
            serde_json::json!({
                "appliedEdits": [
                    { "type": "compact_20260112" }
                ]
            })
        );
    }

    #[test]
    fn raw_context_management_roundtrip_preserves_compact_edit() {
        let raw = raw_context_management_from_provider_metadata(&serde_json::json!({
            "appliedEdits": [
                { "type": "compact_20260112" }
            ]
        }))
        .expect("raw context management");

        assert_eq!(
            raw,
            serde_json::json!({
                "applied_edits": [
                    { "type": "compact_20260112" }
                ]
            })
        );
    }
}
