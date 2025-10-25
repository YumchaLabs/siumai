//! Request transformation traits
//!
//! Converts unified request structs into provider-specific JSON bodies or structs.
//! This matches Cherry Studio's RequestTransformer concept. For now this is an
//! interface definition used by upcoming executors.

use crate::error::LlmError;
use crate::types::{
    ChatRequest, EmbeddingRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
    ModerationRequest, RerankRequest,
};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
enum PathSeg {
    Key(String),
    Index(usize),
}

fn parse_path(path: &str) -> Vec<PathSeg> {
    let mut segs = Vec::new();
    for part in path.split('.') {
        if part.is_empty() {
            continue;
        }
        // Parse key plus optional [idx][idx]...
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
        // parse zero or more [number]
        while let Some(&ch) = chars.peek() {
            if ch != '[' {
                break;
            }
            // consume '['
            chars.next();
            // read digits
            let mut num = String::new();
            while let Some(&d) = chars.peek() {
                if d == ']' {
                    break;
                }
                num.push(d);
                chars.next();
            }
            // consume ']'
            let _ = chars.next();
            if let Ok(idx) = num.parse::<usize>() {
                segs.push(PathSeg::Index(idx));
            }
        }
    }
    segs
}

/// Body for image HTTP requests
pub enum ImageHttpBody {
    Json(serde_json::Value),
    Multipart(reqwest::multipart::Form),
}

/// Transform unified chat request into provider-specific payload
pub trait RequestTransformer: Send + Sync {
    /// Provider identifier (e.g., "openai", "anthropic", "gemini", or compat id)
    fn provider_id(&self) -> &str;

    /// Transform a unified ChatRequest into a provider-specific JSON body
    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError>;

    /// Transform an EmbeddingRequest into a provider-specific JSON body
    fn transform_embedding(&self, _req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement embedding transformer",
            self.provider_id()
        )))
    }

    /// Transform an ImageGenerationRequest into a provider-specific JSON body
    fn transform_image(
        &self,
        _req: &ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement image transformer",
            self.provider_id()
        )))
    }

    /// Transform an ImageEditRequest into a provider-specific body (JSON or multipart)
    fn transform_image_edit(&self, _req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement image edit transformer",
            self.provider_id()
        )))
    }

    /// Transform an ImageVariationRequest into a provider-specific body (JSON or multipart)
    fn transform_image_variation(
        &self,
        _req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement image variation transformer",
            self.provider_id()
        )))
    }

    /// Transform a RerankRequest into a provider-specific JSON body
    fn transform_rerank(&self, _req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement rerank transformer",
            self.provider_id()
        )))
    }

    /// Transform a ModerationRequest into a provider-specific JSON body
    fn transform_moderation(
        &self,
        _req: &ModerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement moderation transformer",
            self.provider_id()
        )))
    }
}

// === Generic request transformation (declarative mapping) ===

/// Strategy for merging provider_params into the final body
#[derive(Clone, Debug)]
pub enum ProviderParamsMergeStrategy {
    /// Insert all key/value pairs at the top-level of the JSON object
    Flatten,
    /// Insert under a namespace object (e.g., "openai": { ... })
    Namespace(&'static str),
}

/// Range validation mode
#[derive(Clone, Debug, Copy)]
pub enum RangeMode {
    /// Return an error if out of range
    Error,
    /// Clamp the value into [min, max]
    Clamp,
}

/// Condition for conditional rules
#[derive(Clone, Debug)]
pub enum Condition {
    /// Match when model starts with given prefix (e.g., "o1-")
    ModelPrefix(&'static str),
}

/// Declarative mapping rule
#[derive(Clone, Debug)]
pub enum Rule {
    /// Rename or move a value from one field path to another
    Move {
        from: &'static str,
        to: &'static str,
    },
    /// Drop a field if present
    Drop { field: &'static str },
    /// Set default value if field is missing or null
    Default {
        field: &'static str,
        value: serde_json::Value,
    },
    /// Validate a numeric field is within [min, max]
    Range {
        field: &'static str,
        min: f64,
        max: f64,
        mode: RangeMode,
        message: Option<&'static str>,
    },
    /// Forbid a field when condition holds
    ForbidWhen {
        field: &'static str,
        condition: Condition,
        message: &'static str,
    },
    /// Map a discrete string value from one field to another field according to a map
    /// For example: map {"low"->"lite", "high"->"pro"}
    EnumMap {
        from: &'static str,
        to: &'static str,
        map: Vec<(String, serde_json::Value)>,
        default: Option<serde_json::Value>,
    },
    /// Conditionally apply a list of rules when a condition holds
    When {
        condition: Condition,
        rules: Vec<Rule>,
    },
    /// Validate array field maximum length
    MaxLen {
        field: &'static str,
        max: usize,
        message: &'static str,
    },
}

/// Provider request hooks for complex mappings not easily expressed via rules
pub trait ProviderRequestHooks: Send + Sync {
    /// Build the base chat request body from a unified ChatRequest
    fn build_base_chat_body(
        &self,
        _req: &crate::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "chat transformer not provided by hooks".to_string(),
        ))
    }

    /// Optional post-processing after rules are applied
    fn post_process_chat(
        &self,
        _req: &crate::types::ChatRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Build the base embedding request body (optional). Default: unsupported.
    fn build_base_embedding_body(
        &self,
        _req: &crate::types::EmbeddingRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "embedding transformer not provided by hooks".to_string(),
        ))
    }

    /// Optional post-processing for embedding request bodies
    fn post_process_embedding(
        &self,
        _req: &crate::types::EmbeddingRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Build the base image generation request body (optional). Default: unsupported.
    fn build_base_image_body(
        &self,
        _req: &crate::types::ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "image transformer not provided by hooks".to_string(),
        ))
    }

    /// Optional post-processing for image generation request bodies
    fn post_process_image(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// Mapping profile for a provider
#[derive(Clone)]
pub struct MappingProfile {
    pub provider_id: &'static str,
    pub rules: Vec<Rule>,
    pub merge_strategy: ProviderParamsMergeStrategy,
}

/// Generic request transformer driven by MappingProfile + Hooks
pub struct GenericRequestTransformer<H: ProviderRequestHooks> {
    pub profile: MappingProfile,
    pub hooks: H,
}

impl<H: ProviderRequestHooks> GenericRequestTransformer<H> {
    /// Merge a key/value map into the request body using the configured strategy
    fn merge_map(
        strategy: &ProviderParamsMergeStrategy,
        body: &mut serde_json::Value,
        params: &HashMap<String, serde_json::Value>,
    ) {
        if params.is_empty() {
            return;
        }

        match strategy {
            ProviderParamsMergeStrategy::Flatten => {
                if let Some(obj) = body.as_object_mut() {
                    for (k, v) in params {
                        obj.insert(k.clone(), v.clone());
                    }
                }
            }
            ProviderParamsMergeStrategy::Namespace(ns) => {
                if let Some(root) = body.as_object_mut() {
                    let entry = root
                        .entry(ns.to_string())
                        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                    if let Some(obj) = entry.as_object_mut() {
                        for (k, v) in params {
                            obj.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
        }
    }

    /// Remove top-level nulls for a cleaner payload
    fn clean_top_level_nulls(body: &mut serde_json::Value) {
        if let serde_json::Value::Object(obj) = body {
            let keys: Vec<String> = obj
                .iter()
                .filter_map(|(k, v)| if v.is_null() { Some(k.clone()) } else { None })
                .collect();
            for k in keys {
                obj.remove(&k);
            }
        }
    }
    fn get_path<'a>(v: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
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

    fn get_path_mut<'a>(
        v: &'a mut serde_json::Value,
        path: &str,
    ) -> Option<&'a mut serde_json::Value> {
        let mut cur = v as *mut serde_json::Value; // raw pointer to satisfy borrow checker during iteration
        for seg in parse_path(path) {
            unsafe {
                match seg {
                    PathSeg::Key(k) => match &mut *cur {
                        serde_json::Value::Object(map) => {
                            let next = map.get_mut(&k)? as *mut serde_json::Value;
                            cur = next;
                        }
                        _ => return None,
                    },
                    PathSeg::Index(i) => match &mut *cur {
                        serde_json::Value::Array(arr) => {
                            let next = arr.get_mut(i)? as *mut serde_json::Value;
                            cur = next;
                        }
                        _ => return None,
                    },
                }
            }
        }
        unsafe { Some(&mut *cur) }
    }

    fn ensure_parent_object<'a>(
        v: &'a mut serde_json::Value,
        path: &str,
    ) -> Option<&'a mut serde_json::Map<String, serde_json::Value>> {
        // Parent of the final key; path must end with a key
        let segments = parse_path(path);
        if segments.is_empty() {
            return None;
        }
        let (parent_segs, leaf_is_key) = match &segments[segments.len() - 1] {
            PathSeg::Key(_) => (&segments[..segments.len() - 1], true),
            PathSeg::Index(_) => (&segments[..segments.len()], false),
        };
        if !leaf_is_key {
            return None;
        }

        let mut cur: *mut serde_json::Value = v;
        for (idx, seg) in parent_segs.iter().enumerate() {
            let next = parent_segs.get(idx + 1);
            unsafe {
                match seg {
                    PathSeg::Key(k) => {
                        // Ensure current is an object
                        match &mut *cur {
                            serde_json::Value::Null => {
                                *cur = serde_json::Value::Object(serde_json::Map::new());
                            }
                            serde_json::Value::Object(_) => {}
                            _ => return None,
                        }
                        if let serde_json::Value::Object(map) = &mut *cur {
                            // Insert or get child
                            let entry = map.entry(k.clone()).or_insert(serde_json::Value::Null);
                            // Shape child according to next segment
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
                            cur = entry as *mut serde_json::Value;
                        }
                    }
                    PathSeg::Index(i) => {
                        // Ensure current is an array
                        match &mut *cur {
                            serde_json::Value::Null => {
                                *cur = serde_json::Value::Array(Vec::new());
                            }
                            serde_json::Value::Array(_) => {}
                            _ => return None,
                        }
                        if let serde_json::Value::Array(arr) = &mut *cur {
                            if arr.len() <= *i {
                                arr.resize(i + 1, serde_json::Value::Null);
                            }
                            // Shape the indexed child according to next segment
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
                            cur = &mut arr[*i] as *mut serde_json::Value;
                        }
                    }
                }
            }
        }
        unsafe {
            if let serde_json::Value::Object(map) = &mut *cur {
                Some(map)
            } else {
                None
            }
        }
    }

    fn move_field(body: &mut serde_json::Value, from: &str, to: &str) {
        // Read value
        let val = Self::get_path(body, from).cloned();
        if val.is_none() || val.as_ref().unwrap().is_null() {
            return;
        }
        // Remove source
        Self::drop_field(body, from);
        // Set destination
        if let Some(parent) = Self::ensure_parent_object(body, to) {
            let leaf = to.split('.').next_back().unwrap();
            parent.insert(leaf.to_string(), val.unwrap());
        }
    }

    fn drop_field(body: &mut serde_json::Value, field: &str) {
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
        let parent_path = {
            // reconstruct parent string path for get_path_mut, which now supports arrays
            let mut s = String::new();
            for (idx, seg) in segs.iter().enumerate() {
                if idx == segs.len() - 1 {
                    break;
                }
                match seg {
                    PathSeg::Key(k) => {
                        if !s.is_empty() {
                            s.push('.');
                        }
                        s.push_str(k);
                    }
                    PathSeg::Index(i) => {
                        s.push('[');
                        s.push_str(&i.to_string());
                        s.push(']');
                    }
                }
            }
            s
        };
        if let Some(parent) = Self::get_path_mut(body, &parent_path) {
            match (segs.last().unwrap(), parent) {
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

    fn apply_default(body: &mut serde_json::Value, field: &str, value: serde_json::Value) {
        let exists_and_non_null = Self::get_path(body, field)
            .map(|v| !v.is_null())
            .unwrap_or(false);
        if !exists_and_non_null && let Some(parent) = Self::ensure_parent_object(body, field) {
            let leaf = field.split('.').next_back().unwrap();
            parent.insert(leaf.to_string(), value);
        }
    }

    fn validate_range(
        body: &mut serde_json::Value,
        field: &str,
        min: f64,
        max: f64,
        mode: RangeMode,
        message: Option<&'static str>,
    ) -> Result<(), LlmError> {
        if let Some(v) = Self::get_path_mut(body, field) {
            let n_opt = match v {
                serde_json::Value::Number(n) => n.as_f64(),
                serde_json::Value::Null => None,
                _ => None,
            };
            if let Some(n) = n_opt
                && (n < min || n > max)
            {
                match mode {
                    RangeMode::Error => {
                        let msg = message.map(|s| s.to_string()).unwrap_or_else(|| {
                            format!("{} must be between {} and {}", field, min, max)
                        });
                        return Err(LlmError::InvalidParameter(msg));
                    }
                    RangeMode::Clamp => {
                        let clamped = if n < min { min } else { max };
                        *v = serde_json::json!(clamped);
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_max_len(
        body: &serde_json::Value,
        field: &str,
        max: usize,
        message: &'static str,
    ) -> Result<(), LlmError> {
        if let Some(v) = Self::get_path(body, field)
            && let serde_json::Value::Array(arr) = v
            && arr.len() > max
        {
            return Err(LlmError::InvalidParameter(message.to_string()));
        }
        Ok(())
    }

    fn apply_rule_list(
        &self,
        req: &crate::types::ChatRequest,
        body: &mut serde_json::Value,
        rules: &[Rule],
    ) -> Result<(), LlmError> {
        for r in rules {
            match r {
                Rule::Move { from, to } => Self::move_field(body, from, to),
                Rule::Drop { field } => Self::drop_field(body, field),
                Rule::Default { field, value } => Self::apply_default(body, field, value.clone()),
                Rule::Range {
                    field,
                    min,
                    max,
                    mode,
                    message,
                } => Self::validate_range(body, field, *min, *max, *mode, *message)?,
                Rule::ForbidWhen {
                    field,
                    condition,
                    message,
                } => {
                    let violated = match condition {
                        Condition::ModelPrefix(prefix) => {
                            req.common_params.model.starts_with(prefix)
                        }
                    };
                    if violated {
                        // Only error if the field is effectively set
                        let present = Self::get_path(body, field)
                            .map(|v| !v.is_null())
                            .unwrap_or(false);
                        if present {
                            return Err(LlmError::InvalidParameter((*message).to_string()));
                        }
                    }
                }
                Rule::EnumMap {
                    from,
                    to,
                    map,
                    default,
                } => {
                    let current =
                        Self::get_path(body, from).and_then(|v| v.as_str().map(|s| s.to_string()));
                    if let Some(key) = current {
                        // Find mapping for key
                        if let Some((_, mapped)) = map.iter().find(|(k, _)| k == &key) {
                            if let Some(parent) = Self::ensure_parent_object(body, to) {
                                let leaf = to.split('.').next_back().unwrap();
                                parent.insert(leaf.to_string(), mapped.clone());
                            }
                        } else if let Some(d) = default.clone() {
                            if let Some(parent) = Self::ensure_parent_object(body, to) {
                                let leaf = to.split('.').next_back().unwrap();
                                parent.insert(leaf.to_string(), d);
                            }
                        }
                    } else if let Some(d) = default.clone() {
                        if let Some(parent) = Self::ensure_parent_object(body, to) {
                            let leaf = to.split('.').next_back().unwrap();
                            parent.insert(leaf.to_string(), d);
                        }
                    }
                }
                Rule::When { condition, rules } => {
                    let cond = match condition {
                        Condition::ModelPrefix(prefix) => {
                            req.common_params.model.starts_with(prefix)
                        }
                    };
                    if cond {
                        self.apply_rule_list(req, body, rules)?;
                    }
                }
                Rule::MaxLen {
                    field,
                    max,
                    message,
                } => Self::validate_max_len(body, field, *max, message)?,
            }
        }
        Ok(())
    }

    fn apply_rules(
        &self,
        req: &crate::types::ChatRequest,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        // First pass rules
        self.apply_rule_list(req, body, &self.profile.rules)?;

        // Clean nulls at the top level only
        if let serde_json::Value::Object(obj) = body {
            let keys: Vec<String> = obj
                .iter()
                .filter_map(|(k, v)| if v.is_null() { Some(k.clone()) } else { None })
                .collect();
            for k in keys {
                obj.remove(&k);
            }
        }

        Ok(())
    }
}

impl<H: ProviderRequestHooks> RequestTransformer for GenericRequestTransformer<H> {
    fn provider_id(&self) -> &str {
        self.profile.provider_id
    }

    fn transform_chat(
        &self,
        req: &crate::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        // Build base via hooks
        let mut body = self.hooks.build_base_chat_body(req)?;
        // Apply rules
        self.apply_rules(req, &mut body)?;
        // Post process
        self.hooks.post_process_chat(req, &mut body)?;
        Ok(body)
    }

    fn transform_embedding(
        &self,
        req: &crate::types::EmbeddingRequest,
    ) -> Result<serde_json::Value, LlmError> {
        // Build via hooks
        let mut body = self.hooks.build_base_embedding_body(req)?;

        // Merge provider-specific params according to profile strategy
        Self::merge_map(
            &self.profile.merge_strategy,
            &mut body,
            &req.provider_params,
        );

        // Post-process
        self.hooks.post_process_embedding(req, &mut body)?;

        // Clean top-level nulls
        Self::clean_top_level_nulls(&mut body);

        Ok(body)
    }

    fn transform_image(
        &self,
        req: &crate::types::ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = self.hooks.build_base_image_body(req)?;

        // Merge extra params according to profile strategy
        Self::merge_map(&self.profile.merge_strategy, &mut body, &req.extra_params);

        // Post-process image
        self.hooks.post_process_image(req, &mut body)?;

        // Clean top-level nulls
        Self::clean_top_level_nulls(&mut body);

        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyHooks;
    impl ProviderRequestHooks for DummyHooks {
        fn build_base_embedding_body(
            &self,
            req: &crate::types::EmbeddingRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({
                "model": req.model.clone().unwrap_or_else(|| "m".to_string()),
                "input": req.input,
                "nullable": serde_json::Value::Null,
            }))
        }

        fn build_base_image_body(
            &self,
            req: &crate::types::ImageGenerationRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({
                "prompt": req.prompt,
                "size": req.size,
                "nullable": serde_json::Value::Null,
            }))
        }
    }

    #[test]
    fn merge_embedding_params_flatten_and_cleanup() {
        let profile = MappingProfile {
            provider_id: "test",
            rules: vec![],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        let tx = GenericRequestTransformer {
            profile,
            hooks: DummyHooks,
        };

        let mut req = EmbeddingRequest::new(vec!["hello".into()]).with_model("m");
        req = req.with_provider_param("foo", serde_json::json!("bar"));
        let body = tx.transform_embedding(&req).unwrap();
        assert_eq!(body["foo"], serde_json::json!("bar"));
        // top-level nulls removed
        assert!(body.get("nullable").is_none());
    }

    #[test]
    fn merge_embedding_params_namespace() {
        let profile = MappingProfile {
            provider_id: "test",
            rules: vec![],
            merge_strategy: ProviderParamsMergeStrategy::Namespace("ns"),
        };
        let tx = GenericRequestTransformer {
            profile,
            hooks: DummyHooks,
        };

        let mut req = EmbeddingRequest::new(vec!["hello".into()]).with_model("m");
        req = req.with_provider_param("alpha", serde_json::json!(1));
        let body = tx.transform_embedding(&req).unwrap();
        assert_eq!(body["ns"]["alpha"], serde_json::json!(1));
        assert!(body.get("alpha").is_none());
    }

    #[test]
    fn merge_image_extra_params_flatten() {
        let profile = MappingProfile {
            provider_id: "test",
            rules: vec![],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        let tx = GenericRequestTransformer {
            profile,
            hooks: DummyHooks,
        };

        let mut req = ImageGenerationRequest::default();
        req.prompt = "draw cat".into();
        req.extra_params
            .insert("style".into(), serde_json::json!("anime"));
        let body = tx.transform_image(&req).unwrap();
        assert_eq!(body["style"], serde_json::json!("anime"));
    }

    #[test]
    fn enum_map_applies_to_chat_body() {
        // Build a transformer with EnumMap rule
        let profile = MappingProfile {
            provider_id: "test",
            rules: vec![Rule::EnumMap {
                from: "service",
                to: "service_tier",
                map: vec![
                    ("premium".to_string(), serde_json::json!("pro")),
                    ("basic".to_string(), serde_json::json!("lite")),
                ],
                default: Some(serde_json::json!("standard")),
            }],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        struct ChatHooks;
        impl ProviderRequestHooks for ChatHooks {
            fn build_base_chat_body(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({ "service": "premium" }))
            }
        }
        let tx = GenericRequestTransformer {
            profile,
            hooks: ChatHooks,
        };

        let mut req = crate::types::ChatRequest::new(vec![]);
        req.common_params.model = "gpt-4".to_string();
        let out = tx.transform_chat(&req).unwrap();
        assert_eq!(out["service_tier"], serde_json::json!("pro"));
    }

    #[test]
    fn when_condition_model_prefix_applies_rules() {
        // When model starts with o1-, set max_completion_tokens default to 100
        let profile = MappingProfile {
            provider_id: "test",
            rules: vec![Rule::When {
                condition: Condition::ModelPrefix("o1-"),
                rules: vec![Rule::Default {
                    field: "max_completion_tokens",
                    value: serde_json::json!(100),
                }],
            }],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        struct ChatHooks;
        impl ProviderRequestHooks for ChatHooks {
            fn build_base_chat_body(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({ "model": "ignored" }))
            }
        }
        let tx = GenericRequestTransformer {
            profile,
            hooks: ChatHooks,
        };

        let mut req = crate::types::ChatRequest::new(vec![]);
        req.common_params.model = "o1-mini".to_string();
        let out = tx.transform_chat(&req).unwrap();
        assert_eq!(out["max_completion_tokens"], serde_json::json!(100));

        let mut req2 = crate::types::ChatRequest::new(vec![]);
        req2.common_params.model = "gpt-4o".to_string();
        let out2 = tx.transform_chat(&req2).unwrap();
        assert!(out2.get("max_completion_tokens").is_none());
    }

    #[test]
    fn move_from_array_path_into_object_key() {
        // messages[0].content -> payload.first
        let profile = MappingProfile {
            provider_id: "test",
            rules: vec![Rule::Move {
                from: "messages[0].content",
                to: "payload.first",
            }],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        struct ChatHooks;
        impl ProviderRequestHooks for ChatHooks {
            fn build_base_chat_body(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({
                    "messages": [
                        {"content": "hello"},
                        {"content": "world"}
                    ]
                }))
            }
        }
        let tx = GenericRequestTransformer {
            profile,
            hooks: ChatHooks,
        };
        let req = crate::types::ChatRequest::new(vec![]);
        let out = tx.transform_chat(&req).unwrap();
        assert_eq!(out["payload"]["first"], serde_json::json!("hello"));
        assert!(out["messages"][0].get("content").is_none());
    }

    #[test]
    fn default_into_nested_array_path_creates_structure() {
        let profile = MappingProfile {
            provider_id: "test",
            rules: vec![Rule::Default {
                field: "params.options[0].name",
                value: serde_json::json!("x"),
            }],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        struct ChatHooks;
        impl ProviderRequestHooks for ChatHooks {
            fn build_base_chat_body(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({}))
            }
        }
        let tx = GenericRequestTransformer {
            profile,
            hooks: ChatHooks,
        };
        let req = crate::types::ChatRequest::new(vec![]);
        let out = tx.transform_chat(&req).unwrap();
        assert_eq!(out["params"]["options"][0]["name"], serde_json::json!("x"));
    }
}
