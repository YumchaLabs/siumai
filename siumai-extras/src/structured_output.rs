//! Unified structured output *decoding* configuration and helpers used by
//! high-level object generation and orchestrator/agent utilities.
//!
//! This module focuses on how to parse and validate the model's text output
//! into JSON / typed values. It is intentionally separate from provider-level
//! "structured output" request configuration (e.g. OpenAI JSON schema APIs).
//!
//! Responsibilities:
//! - Output shape hints (`OutputKind`)
//! - Mode hints for providers (`GenerateMode`)
//! - JSON repair and optional schema validation
//! - Typed / untyped JSON decoding from raw model text

use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::prelude::unified::{LlmError, OutputSchema};

/// Type alias for JSON repair function used in structured output APIs.
pub type RepairFn = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Output kind hints for structured object generation.
#[derive(Debug, Clone, Default)]
pub enum OutputKind {
    /// Expect a JSON object value.
    #[default]
    Object,
    /// Expect a JSON array value.
    Array,
    /// Expect one of enumerated values (string/number/bool).
    Enum(Vec<Value>),
    /// Do not apply schema validation; free-form JSON.
    NoSchema,
}

/// Mode hint for providers to select structured output strategy.
///
/// Provider adapters may use this to decide whether to use JSON-native modes,
/// tool-based JSON, or let the provider decide automatically.
#[derive(Debug, Clone, Copy, Default)]
pub enum GenerateMode {
    /// Let provider / adapter decide the best strategy.
    #[default]
    Auto,
    /// Prefer JSON-native structured output (e.g. OpenAI Responses API, Gemini JSON).
    Json,
    /// Prefer tool-based structured output (JSON via tool calls).
    Tool,
}

/// Configuration for decoding structured output, inspired by Vercel AI SDK's
/// `experimental_output` parameter.
///
/// This configuration is used by:
/// - High-level object helpers (`highlevel::object`)
/// - Agents (`ToolLoopAgent`) for extracting structured output
#[derive(Clone)]
pub struct OutputDecodeConfig {
    /// Optional JSON schema + metadata (name, description).
    pub schema: Option<OutputSchema>,
    /// Output shape hint (object/array/enum/free-form).
    pub kind: OutputKind,
    /// Provider strategy hint (auto/json/tool).
    pub mode: GenerateMode,
    /// Whether to attempt partial JSON parsing when streaming.
    pub emit_partial: bool,
    /// Optional repair function to turn imperfect text into valid JSON.
    pub repair_text: Option<RepairFn>,
    /// Maximum number of repair rounds to try when parsing/validation fails.
    pub max_repair_rounds: usize,
}

impl Default for OutputDecodeConfig {
    fn default() -> Self {
        Self {
            schema: None,
            kind: OutputKind::default(),
            mode: GenerateMode::default(),
            emit_partial: true,
            repair_text: None,
            max_repair_rounds: 1,
        }
    }
}

impl OutputDecodeConfig {
    /// Convenience constructor for a simple object schema.
    ///
    /// This mirrors the common pattern of "just give me an object matching this schema".
    pub fn from_schema(schema: OutputSchema) -> Self {
        Self {
            schema: Some(schema),
            kind: OutputKind::Object,
            mode: GenerateMode::Auto,
            emit_partial: true,
            repair_text: None,
            max_repair_rounds: 1,
        }
    }
}

/// Decode a raw JSON-like string into a `serde_json::Value` according to the
/// structured output configuration.
///
/// Responsibilities:
/// - Parse JSON from text (with optional repair loops)
/// - Enforce `OutputKind` shape
/// - Optionally validate against JSON Schema (when `schema` feature is enabled)
pub fn decode_json_value(text: &str, cfg: &OutputDecodeConfig) -> Result<Value, LlmError> {
    let mut current = text.to_string();
    let mut rounds = 0usize;

    loop {
        match parse_once(&current, cfg) {
            Ok(v) => return Ok(v),
            Err(e) => {
                if rounds >= cfg.max_repair_rounds {
                    return Err(e);
                }
                if let Some(repair) = &cfg.repair_text {
                    if let Some(next) = repair(&current) {
                        current = next;
                        rounds += 1;
                        continue;
                    }
                } else if let Some(next) = default_repair_text(&current) {
                    current = next;
                    rounds += 1;
                    continue;
                }
                return Err(e);
            }
        }
    }
}

/// Decode a raw JSON-like string into a typed value `T` using the same rules
/// as [`decode_json_value`].
pub fn decode_typed<T: DeserializeOwned>(
    text: &str,
    cfg: &OutputDecodeConfig,
) -> Result<T, LlmError> {
    let value = decode_json_value(text, cfg)?;
    serde_json::from_value::<T>(value)
        .map_err(|e| LlmError::ParseError(format!("Failed to deserialize object: {}", e)))
}

/// Internal single-attempt parse + shape + schema validation.
fn parse_once(text: &str, cfg: &OutputDecodeConfig) -> Result<Value, LlmError> {
    let value: Value = serde_json::from_str(text)
        .map_err(|e| LlmError::ParseError(format!("Failed to parse JSON: {}", e)))?;

    // Shape check from OutputKind.
    match &cfg.kind {
        OutputKind::Object => {
            if !value.is_object() {
                return Err(LlmError::InvalidParameter("Expected a JSON object".into()));
            }
        }
        OutputKind::Array => {
            if !value.is_array() {
                return Err(LlmError::InvalidParameter("Expected a JSON array".into()));
            }
        }
        OutputKind::Enum(allowed) => {
            if !allowed.is_empty() && !allowed.contains(&value) {
                return Err(LlmError::InvalidParameter(format!(
                    "Value not in enum set: {}",
                    value
                )));
            }
        }
        OutputKind::NoSchema => {}
    }

    // Optional JSON Schema validation via `siumai-extras::schema` when enabled.
    if let Some(out_schema) = &cfg.schema {
        #[cfg(feature = "schema")]
        {
            if let Err(e) = crate::schema::validate_json(&out_schema.schema, &value) {
                return Err(LlmError::ParseError(format!(
                    "Schema validation failed: {}",
                    e
                )));
            }
        }
        #[cfg(not(feature = "schema"))]
        {
            let _ = out_schema; // avoid unused warning without schema feature
        }
    }

    Ok(value)
}

/// Default lightweight repair: strip markdown fences, trim to a balanced JSON
/// slice, and remove trailing commas before `}`/`]`.
pub(crate) fn default_repair_text(text: &str) -> Option<String> {
    // Remove common fenced code wrappers like ```json ... ```
    let mut s = text.trim().to_string();
    if s.starts_with("```") {
        // remove first line fence
        if let Some(pos) = s.find('\n') {
            s = s[pos + 1..].to_string();
        }
    }
    if let Some(idx) = s.rfind("```") {
        s = s[..idx].to_string();
    }
    // Try balanced slice
    if let Some(slice) = extract_balanced_json_slice(&s) {
        let cand = strip_trailing_commas(slice);
        return Some(cand);
    }
    None
}

/// Extract a balanced JSON substring from the given text if possible.
///
/// This scans for the first '{' or '[' and then tracks brace/bracket balance,
/// ignoring occurrences within string literals. When balance returns to zero,
/// returns the substring covering that balanced JSON block.
pub(crate) fn extract_balanced_json_slice(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut start = None;
    let mut brace: i32 = 0;
    let mut bracket: i32 = 0;
    let mut i = 0;
    let mut in_str = false;
    let mut escape = false;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if start.is_none() {
            if c == '{' || c == '[' {
                start = Some(i);
                if c == '{' {
                    brace = 1;
                } else {
                    bracket = 1;
                }
                i += 1;
                continue;
            }
        } else if in_str {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_str = false;
            }
        } else {
            match c {
                '"' => in_str = true,
                '{' => brace += 1,
                '}' => brace -= 1,
                '[' => bracket += 1,
                ']' => bracket -= 1,
                _ => {}
            }
            if brace < 0 || bracket < 0 {
                // malformed; abort current detection
                start = None;
                brace = 0;
                bracket = 0;
                in_str = false;
                escape = false;
            } else if brace == 0 && bracket == 0 {
                let s = start.unwrap();
                let e = i; // inclusive char at i
                return text.get(s..=e);
            }
        }
        i += 1;
    }
    None
}

/// Remove trailing commas immediately before '}' or ']' to increase JSON
/// parse tolerance.
pub(crate) fn strip_trailing_commas(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == ',' {
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            if j < bytes.len() {
                let nc = bytes[j] as char;
                if nc == '}' || nc == ']' {
                    i += 1; // skip this comma
                    continue;
                }
            }
            out.push(',');
            i += 1;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}
