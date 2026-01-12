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
            _ => {}
        }
    }

    Some(serde_json::json!({ "appliedEdits": out_edits }))
}
