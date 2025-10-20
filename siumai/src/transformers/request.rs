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
    fn get_path<'a>(v: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
        let mut cur = v;
        for seg in path.split('.') {
            match cur {
                serde_json::Value::Object(map) => {
                    cur = map.get(seg)?;
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
        let mut cur = v;
        for seg in path.split('.') {
            match cur {
                serde_json::Value::Object(map) => {
                    cur = map.get_mut(seg)?;
                }
                _ => return None,
            }
        }
        Some(cur)
    }

    fn ensure_parent_object<'a>(
        v: &'a mut serde_json::Value,
        path: &str,
    ) -> Option<&'a mut serde_json::Map<String, serde_json::Value>> {
        let mut cur = v;
        let mut it = path.split('.').peekable();
        while let Some(seg) = it.next() {
            if it.peek().is_none() {
                // parent level reached
                if let serde_json::Value::Object(map) = cur {
                    return Some(map);
                } else {
                    return None;
                }
            }
            match cur {
                serde_json::Value::Object(map) => {
                    cur = map
                        .entry(seg.to_string())
                        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                }
                _ => return None,
            }
        }
        None
    }

    fn move_field(body: &mut serde_json::Value, from: &str, to: &str) {
        // Read value
        let val = Self::get_path(body, from).cloned();
        if val.is_none() || val.as_ref().unwrap().is_null() {
            return;
        }
        // Remove source
        // Simple removal supports only top-level; for nested, rebuild along the way
        // We'll do a best-effort delete by walking parents
        let segments: Vec<&str> = from.split('.').collect();
        if segments.len() == 1 {
            if let serde_json::Value::Object(map) = body {
                map.remove(segments[0]);
            }
        } else {
            // Navigate to parent
            let (parent_path, leaf) = (
                segments[..segments.len() - 1].join("."),
                segments[segments.len() - 1],
            );
            if let Some(parent) = Self::get_path_mut(body, &parent_path)
                && let serde_json::Value::Object(map) = parent
            {
                map.remove(leaf);
            }
        }
        // Set destination
        if let Some(parent) = Self::ensure_parent_object(body, to) {
            let leaf = to.split('.').next_back().unwrap();
            parent.insert(leaf.to_string(), val.unwrap());
        }
    }

    fn drop_field(body: &mut serde_json::Value, field: &str) {
        let segments: Vec<&str> = field.split('.').collect();
        if segments.len() == 1 {
            if let serde_json::Value::Object(map) = body {
                map.remove(segments[0]);
            }
        } else {
            let (parent_path, leaf) = (
                segments[..segments.len() - 1].join("."),
                segments[segments.len() - 1],
            );
            if let Some(parent) = Self::get_path_mut(body, &parent_path)
                && let serde_json::Value::Object(map) = parent
            {
                map.remove(leaf);
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

    fn apply_rules(
        &self,
        req: &crate::types::ChatRequest,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        // First pass rules
        for r in &self.profile.rules {
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
                Rule::MaxLen {
                    field,
                    max,
                    message,
                } => Self::validate_max_len(body, field, *max, message)?,
            }
        }

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

        // Post-process
        self.hooks.post_process_embedding(req, &mut body)?;

        // Clean top-level nulls
        if let serde_json::Value::Object(obj) = &mut body {
            let keys: Vec<String> = obj
                .iter()
                .filter_map(|(k, v)| if v.is_null() { Some(k.clone()) } else { None })
                .collect();
            for k in keys {
                obj.remove(&k);
            }
        }

        Ok(body)
    }

    fn transform_image(
        &self,
        req: &crate::types::ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = self.hooks.build_base_image_body(req)?;

        // Post-process image
        self.hooks.post_process_image(req, &mut body)?;

        // Clean top-level nulls
        if let serde_json::Value::Object(obj) = &mut body {
            let keys: Vec<String> = obj
                .iter()
                .filter_map(|(k, v)| if v.is_null() { Some(k.clone()) } else { None })
                .collect();
            for k in keys {
                obj.remove(&k);
            }
        }

        Ok(body)
    }
}
