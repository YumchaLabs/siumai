#![allow(clippy::collapsible_if)]
//! JSON path read/write helpers for request transformation
//!
//! Internal utilities used by the declarative mapping in `request.rs`.

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PathSeg {
    Key(String),
    Index(usize),
}

/// Parse a dotted/array path like `a.b[0].c[2]` into segments
pub(super) fn parse_path(path: &str) -> Vec<PathSeg> {
    let mut segs = Vec::new();
    for part in path.split('.') {
        if part.is_empty() {
            continue;
        }
        // Parse key up to the first '['
        let mut key = String::new();
        let mut chars = part.chars().peekable();
        while let Some(&ch) = chars.peek() {
            if ch == '[' {
                break;
            }
            key.push(ch);
            chars.next();
        }
        if !key.is_empty() {
            segs.push(PathSeg::Key(key.clone()));
        }
        // Zero or more [number]
        while let Some(&ch) = chars.peek() {
            if ch != '[' {
                break;
            }
            chars.next(); // '[`
            let mut num = String::new();
            while let Some(&d) = chars.peek() {
                if d == ']' {
                    break;
                }
                num.push(d);
                chars.next();
            }
            let _ = chars.next(); // ']'
            if let Ok(idx) = num.parse::<usize>() {
                segs.push(PathSeg::Index(idx));
            }
        }
    }
    segs
}

/// Get immutable reference by path
pub(super) fn get_path<'a>(v: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut cur = v;
    for seg in parse_path(path) {
        match (seg, cur) {
            (PathSeg::Key(k), serde_json::Value::Object(map)) => {
                cur = map.get(&k)?;
            }
            (PathSeg::Index(i), serde_json::Value::Array(arr)) => {
                cur = arr.get(i)?;
            }
            _ => return None,
        }
    }
    Some(cur)
}

/// Get mutable reference by path (does not create intermediate structures)
pub(super) fn get_path_mut<'a>(
    v: &'a mut serde_json::Value,
    path: &str,
) -> Option<&'a mut serde_json::Value> {
    let segs = parse_path(path);
    let mut cur = v;
    for seg in segs {
        match seg {
            PathSeg::Key(k) => match cur {
                serde_json::Value::Object(map) => {
                    cur = map.get_mut(&k)?;
                }
                _ => return None,
            },
            PathSeg::Index(i) => match cur {
                serde_json::Value::Array(arr) => {
                    cur = arr.get_mut(i)?;
                }
                _ => return None,
            },
        }
    }
    Some(cur)
}

/// Ensure the parent is an object, return it; creates intermediate structures as needed
pub(super) fn ensure_parent_object<'a>(
    v: &'a mut serde_json::Value,
    path: &str,
) -> Option<&'a mut serde_json::Map<String, serde_json::Value>> {
    let segs = parse_path(path);
    if segs.is_empty() {
        return None;
    }
    let parent_segs = &segs[..segs.len() - 1];

    let mut cur = v;
    for (idx, seg) in parent_segs.iter().enumerate() {
        let next = parent_segs.get(idx + 1);
        match seg {
            PathSeg::Key(k) => {
                match cur {
                    serde_json::Value::Null => {
                        *cur = serde_json::Value::Object(serde_json::Map::new());
                    }
                    serde_json::Value::Object(_) => {}
                    _ => return None,
                }
                if let serde_json::Value::Object(map) = cur {
                    let entry = map.entry(k.clone()).or_insert(serde_json::Value::Null);
                    match next {
                        Some(PathSeg::Index(_)) => {
                            if !entry.is_array() {
                                *entry = serde_json::Value::Array(Vec::new());
                            }
                        }
                        Some(PathSeg::Key(_)) | None => {
                            if !entry.is_object() {
                                *entry = serde_json::Value::Object(serde_json::Map::new());
                            }
                        }
                    }
                    cur = entry;
                }
            }
            PathSeg::Index(i) => {
                match cur {
                    serde_json::Value::Null => {
                        *cur = serde_json::Value::Array(Vec::new());
                    }
                    serde_json::Value::Array(_) => {}
                    _ => return None,
                }
                if let serde_json::Value::Array(arr) = cur {
                    if arr.len() <= *i {
                        arr.resize(i + 1, serde_json::Value::Null);
                    }
                    match next {
                        Some(PathSeg::Index(_)) => {
                            if !arr[*i].is_array() {
                                arr[*i] = serde_json::Value::Array(Vec::new());
                            }
                        }
                        Some(PathSeg::Key(_)) | None => {
                            if !arr[*i].is_object() {
                                arr[*i] = serde_json::Value::Object(serde_json::Map::new());
                            }
                        }
                    }
                    cur = &mut arr[*i];
                }
            }
        }
    }

    if let serde_json::Value::Object(map) = cur {
        Some(map)
    } else {
        None
    }
}

/// Move a value from one path to another, removing the source
pub(super) fn move_field(body: &mut serde_json::Value, from: &str, to: &str) {
    let val = get_path(body, from).cloned();
    if val.is_none() || val.as_ref().unwrap().is_null() {
        return;
    }
    drop_field(body, from);
    if let Some(parent) = ensure_parent_object(body, to) {
        let leaf = to.split('.').next_back().unwrap();
        parent.insert(leaf.to_string(), val.unwrap());
    }
}

/// Drop a field at path (supports array index removals)
pub(super) fn drop_field(body: &mut serde_json::Value, field: &str) {
    let segs = parse_path(field);
    if segs.is_empty() {
        return;
    }
    if segs.len() == 1 {
        match (&segs[0], body) {
            (PathSeg::Key(k), serde_json::Value::Object(map)) => {
                map.remove(k);
            }
            (PathSeg::Index(i), serde_json::Value::Array(arr)) => {
                if *i < arr.len() {
                    arr.remove(*i);
                }
            }
            _ => {}
        }
        return;
    }
    // Reconstruct parent path string for get_path_mut
    let mut parent = String::new();
    for (idx, seg) in segs.iter().enumerate() {
        if idx == segs.len() - 1 {
            break;
        }
        match seg {
            PathSeg::Key(k) => {
                if !parent.is_empty() {
                    parent.push('.');
                }
                parent.push_str(k);
            }
            PathSeg::Index(i) => {
                parent.push('[');
                parent.push_str(&i.to_string());
                parent.push(']');
            }
        }
    }
    if let Some(p) = get_path_mut(body, &parent) {
        match (segs.last().unwrap(), p) {
            (PathSeg::Key(k), serde_json::Value::Object(map)) => {
                map.remove(k);
            }
            (PathSeg::Index(i), serde_json::Value::Array(arr)) => {
                if *i < arr.len() {
                    arr.remove(*i);
                }
            }
            _ => {}
        }
    }
}

/// Apply default if field missing or null
pub(super) fn apply_default(body: &mut serde_json::Value, field: &str, value: serde_json::Value) {
    let exists_and_non_null = get_path(body, field).map(|v| !v.is_null()).unwrap_or(false);
    if !exists_and_non_null {
        if let Some(parent) = ensure_parent_object(body, field) {
            let leaf = field.split('.').next_back().unwrap();
            parent.insert(leaf.to_string(), value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn parse_mixed_path() {
        let segs = parse_path("a.b[2].c[0][1]");
        assert!(matches!(&segs[0], PathSeg::Key(k) if k == "a"));
        assert!(matches!(&segs[1], PathSeg::Key(k) if k == "b"));
        assert!(matches!(segs[2], PathSeg::Index(2)));
        assert!(matches!(&segs[3], PathSeg::Key(k) if k == "c"));
        assert!(matches!(segs[4], PathSeg::Index(0)));
        assert!(matches!(segs[5], PathSeg::Index(1)));
    }

    #[test]
    fn move_from_array_to_object() {
        let mut v = serde_json::json!({ "messages": [{"content": "hello"},{"content":"x"}] });
        move_field(&mut v, "messages[0].content", "payload.first");
        assert_eq!(v["payload"]["first"], serde_json::json!("hello"));
        assert!(v["messages"][0].get("content").is_none());
    }

    #[test]
    fn default_creates_nested_array() {
        let mut v = serde_json::json!({});
        apply_default(&mut v, "params.options[0].name", serde_json::json!("x"));
        assert_eq!(v["params"]["options"][0]["name"], serde_json::json!("x"));
    }

    proptest! {
        // Property: applying default on a fresh object for random positive index
        // should not panic and should create either arrays or objects as needed.
        #[test]
        fn prop_default_any_index_creates_structure(i in 0usize..8) {
            let mut v = serde_json::json!({});
            let path = format!("root.items[{}].flag", i);
            apply_default(&mut v, &path, serde_json::json!(true));
            // Read back; it should be true
            let got = get_path(&v, &path).and_then(|x| x.as_bool()).unwrap_or(false);
            prop_assert!(got);
        }

        // Property: moving from one nested path to another either moves value or keeps target unset when source missing
        #[test]
        fn prop_move_field_idempotent_when_source_missing(j in 0usize..4) {
            let mut v = serde_json::json!({});
            let from = format!("missing[{}].value", j);
            let to = "dst.here";
            move_field(&mut v, &from, to);
            // dst.here should not exist
            prop_assert!(get_path(&v, to).is_none());
        }
    }
}
