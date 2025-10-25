//! High-level structured object generation APIs
//!
//! Provides a minimal, provider-agnostic wrapper
//! to generate typed JSON objects using any ChatCapability model. The function
//! performs optional JSON Schema validation and optional text repair before
//! deserializing into `T`.

use std::sync::Arc;

use crate::error::LlmError;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, Tool, Usage};
use futures::Stream;
use serde::de::DeserializeOwned;
use std::pin::Pin;

/// Output kind hints for object generation.
#[derive(Debug, Clone)]
pub enum OutputKind {
    /// Expect a JSON object value
    Object,
    /// Expect a JSON array value
    Array,
    /// Expect one of enumerated values (string/number/bool)
    Enum(Vec<serde_json::Value>),
    /// Do not apply schema validation; free-form JSON
    NoSchema,
}

impl Default for OutputKind {
    fn default() -> Self {
        OutputKind::Object
    }
}

/// Mode hint for providers to select structured output strategy.
/// Currently informational at the high-level API; provider mappers may use it.
#[derive(Debug, Clone, Copy)]
pub enum GenerateMode {
    Auto,
    Json,
    Tool,
}

impl Default for GenerateMode {
    fn default() -> Self {
        GenerateMode::Auto
    }
}

/// Options for object generation.
pub struct GenerateObjectOptions {
    /// Optional JSON Schema used to validate the parsed JSON value before
    /// deserialization.
    pub schema: Option<serde_json::Value>,
    /// Optional schema name for provider hints.
    pub schema_name: Option<String>,
    /// Optional schema description for provider hints.
    pub schema_description: Option<String>,
    /// Output kind hint to apply simple shape checks before deserialization.
    pub output: OutputKind,
    /// Mode hint for downstream providers.
    pub mode: GenerateMode,
    /// Optional repair function to turn imperfect model text into valid JSON.
    /// Return Some(repaired) to retry parsing/validation, None to stop.
    pub repair_text: Option<Arc<dyn Fn(&str) -> Option<String> + Send + Sync>>,
    /// Number of repair attempts to try when parsing/validation fails.
    pub max_repair_rounds: usize,
}

impl Default for GenerateObjectOptions {
    fn default() -> Self {
        Self {
            schema: None,
            schema_name: None,
            schema_description: None,
            output: OutputKind::default(),
            mode: GenerateMode::default(),
            repair_text: None,
            max_repair_rounds: 1,
        }
    }
}

/// Generate a typed object `T` using a chat model and optional tools.
///
/// Flow:
/// - Calls `model.chat_with_tools(messages, tools)`.
/// - Extracts text content and attempts to parse as JSON.
/// - Optionally validates against a JSON Schema.
/// - Deserializes JSON into `T` and returns along with the raw ChatResponse.
pub async fn generate_object<T: DeserializeOwned>(
    model: &impl ChatCapability,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    opts: GenerateObjectOptions,
) -> Result<(T, ChatResponse), LlmError> {
    // Build ChatRequest to allow passing provider hints for structured outputs
    let request = build_chat_request_with_hints(messages, tools.clone(), &opts, false);
    let resp = model.chat_request(request).await?;

    // Prefer tool call arguments if present (Tool mode or provider chose tools)
    let mut text = {
        let tool_calls = resp.tool_calls();
        if let Some(first) = tool_calls.first() {
            if let crate::types::ContentPart::ToolCall { arguments, .. } = first {
                serde_json::to_string(arguments).unwrap_or_default()
            } else {
                resp.content_text()
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            }
        } else {
            resp.content_text()
                .map(|s| s.to_string())
                .unwrap_or_default()
        }
    };
    if text.is_empty() {
        return Err(LlmError::ParseError(
            "No content for object generation".into(),
        ));
    }

    // Attempt to parse + validate + deserialize, with optional repair rounds.
    let mut rounds = 0usize;
    loop {
        match try_parse_validate_deserialize::<T>(&text, opts.schema.as_ref(), &opts.output) {
            Ok(obj) => return Ok((obj, resp)),
            Err(e) => {
                if rounds >= opts.max_repair_rounds {
                    return Err(e);
                }
                if let Some(repair) = &opts.repair_text {
                    if let Some(next) = repair(&text) {
                        text = next;
                        rounds += 1;
                        continue;
                    }
                } else if let Some(next) = default_repair_text(&text) {
                    text = next;
                    rounds += 1;
                    continue;
                }
                return Err(e);
            }
        }
    }
}

fn try_parse_validate_deserialize<T: DeserializeOwned>(
    text: &str,
    schema: Option<&serde_json::Value>,
    output: &OutputKind,
) -> Result<T, LlmError> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| LlmError::ParseError(format!("Failed to parse JSON: {}", e)))?;
    // Output kind simple validation
    match output {
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

    // Schema validation has been moved to siumai-extras
    // If you need schema validation, use siumai-extras::schema::validate_json
    if schema.is_some() {
        tracing::warn!(
            "Schema validation is no longer built-in. Use siumai-extras::schema::validate_json for validation."
        );
    }
    serde_json::from_value::<T>(value)
        .map_err(|e| LlmError::ParseError(format!("Failed to deserialize object: {}", e)))
}

/// Stream options for `stream_object`.
pub struct StreamObjectOptions {
    pub schema: Option<serde_json::Value>,
    pub schema_name: Option<String>,
    pub schema_description: Option<String>,
    pub output: OutputKind,
    pub mode: GenerateMode,
    pub repair_text: Option<Arc<dyn Fn(&str) -> Option<String> + Send + Sync>>,
    pub max_repair_rounds: usize,
    /// Whether to attempt partial JSON parsing on each delta and emit partial updates.
    pub emit_partial_object: bool,
}

impl Default for StreamObjectOptions {
    fn default() -> Self {
        Self {
            schema: None,
            schema_name: None,
            schema_description: None,
            output: OutputKind::default(),
            mode: GenerateMode::default(),
            repair_text: None,
            max_repair_rounds: 1,
            emit_partial_object: true,
        }
    }
}

/// Streaming events for object generation.
pub enum StreamObjectEvent<T> {
    /// Raw text delta from the model.
    TextDelta { delta: String },
    /// Parsed partial object update if current buffer is valid JSON.
    PartialObject { partial: serde_json::Value },
    /// Usage update passthrough.
    UsageUpdate { usage: Usage },
    /// Final parsed and validated object with the underlying response.
    Final { object: T, response: ChatResponse },
}

/// Stream a typed object `T` from a chat model.
///
/// Minimal strategy: accumulate text deltas; on stream end attempt parse + optional
/// schema validation + optional repair rounds; yield Final event when successful.
pub async fn stream_object<T: DeserializeOwned + Send + 'static>(
    model: &impl ChatCapability,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    opts: StreamObjectOptions,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamObjectEvent<T>, LlmError>> + Send>>, LlmError> {
    let req = build_chat_request_with_hints(
        messages,
        tools.clone(),
        &GenerateObjectOptions {
            schema: opts.schema.clone(),
            schema_name: opts.schema_name.clone(),
            schema_description: opts.schema_description.clone(),
            output: opts.output.clone(),
            mode: opts.mode,
            repair_text: opts.repair_text.clone(),
            max_repair_rounds: opts.max_repair_rounds,
        },
        true,
    );
    let mut stream = model.chat_stream_request(req).await?;
    let schema = opts.schema.clone();
    let output = opts.output.clone();
    let repair = opts.repair_text.clone();
    let max_rounds = opts.max_repair_rounds;
    let emit_partial = opts.emit_partial_object;

    let s = async_stream::try_stream! {
        use futures::StreamExt;
        let mut acc = String::new();
        let mut final_resp: Option<ChatResponse> = None;
        let mut tool_args_acc = String::new();
        let mut last_partial: Option<serde_json::Value> = None;
        while let Some(item) = stream.next().await {
            match item? {
                crate::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
                    acc.push_str(&delta);
                    yield StreamObjectEvent::TextDelta { delta };
                    if emit_partial {
                        // Try parse a balanced JSON slice from the accumulated text.
                        if let Some(slice) = extract_balanced_json_slice(&acc) {
                            let cand = strip_trailing_commas(slice);
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&cand) {
                                let changed = match &last_partial {
                                    Some(prev) => prev != &v,
                                    None => true,
                                };
                                if changed {
                                    last_partial = Some(v.clone());
                                    yield StreamObjectEvent::PartialObject { partial: v };
                                }
                            }
                        }
                    }
                }
                crate::streaming::ChatStreamEvent::ToolCallDelta { arguments_delta, .. } => {
                    if let Some(d) = arguments_delta { tool_args_acc.push_str(&d); }
                }
                crate::streaming::ChatStreamEvent::UsageUpdate { usage } => {
                    yield StreamObjectEvent::UsageUpdate { usage };
                }
                crate::streaming::ChatStreamEvent::StreamEnd { response } => {
                    final_resp = Some(response);
                    break;
                }
                _ => {}
            }
        }
        let resp = final_resp.unwrap_or_else(|| ChatResponse::new(crate::types::MessageContent::Text(acc.clone())));
        // Try parse/validate/deserialize with optional repair
        // Prefer tool arguments if present
        let mut text = if !tool_args_acc.is_empty() { tool_args_acc } else { acc };
        let mut rounds = 0usize;
        loop {
            match try_parse_validate_deserialize::<T>(&text, schema.as_ref(), &output) {
                Ok(obj) => {
                    yield StreamObjectEvent::Final { object: obj, response: resp };
                    break;
                }
                Err(e) => {
                    let mut err_opt = Some(e);
                    if rounds < max_rounds {
                        if let Some(cb) = &repair {
                            if let Some(next) = cb(&text) { text = next; rounds += 1; err_opt = None; }
                        } else if let Some(next) = default_repair_text(&text) {
                            text = next; rounds += 1; err_opt = None;
                        }
                    }
                    if let Some(err) = err_opt { Err::<(), LlmError>(err)?; }
                }
            }
        }
    };
    Ok(Box::pin(s))
}

/// Extract a balanced JSON substring from the given text if possible.
///
/// This scans for the first '{' or '[' and then tracks brace/bracket balance,
/// ignoring occurrences within string literals. When balance returns to zero,
/// returns the substring covering that balanced JSON block.
fn extract_balanced_json_slice(text: &str) -> Option<&str> {
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
        } else {
            if in_str {
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
        }
        i += 1;
    }
    None
}

/// Default lightweight repair: strip code fences, trim to balanced JSON slice.
fn default_repair_text(text: &str) -> Option<String> {
    // Remove common fenced code wrappers
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

/// Remove trailing commas immediately before '}' or ']'.
fn strip_trailing_commas(input: &str) -> String {
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

/// Remove trailing commas before '}' and ']' to increase JSON parse tolerance.
#[allow(dead_code)]
fn remove_trailing_commas(input: &str) -> String {
    strip_trailing_commas(input)
    /*
    // Simple regex approach; safe fallback to original on regex build failure.
    // Pattern: ,\s*} and ,\s*]
    let re1 = Regex::new(",\\s*}\").ok();
    let re2 = Regex::new(",\\s*]\\").ok();
    let mut out = input.to_string();
    if let Some(r) = re1.as_ref() { out = r.replace_all(&out, "}").to_string(); }
    if let Some(r) = re2.as_ref() { out = r.replace_all(&out, "]").to_string(); }
    */
}

/// Build a ChatRequest carrying structured_output hints.
fn build_chat_request_with_hints(
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    opts: &GenerateObjectOptions,
    stream: bool,
) -> ChatRequest {
    // Optionally inject JSON-only/system hints and a tool definition when using specific modes with schema
    let mut final_messages = messages;
    let mut tools_vec = tools.clone().unwrap_or_default();
    match opts.mode {
        GenerateMode::Tool => {
            if let Some(schema) = opts.schema.clone() {
                let tool_name = opts
                    .schema_name
                    .clone()
                    .unwrap_or_else(|| "submit_object".to_string());
                let desc = opts.schema_description.clone().unwrap_or_else(|| {
                    "Return the final JSON object by calling this tool".to_string()
                });
                tools_vec.push(Tool::function(tool_name.clone(), desc, schema));
                final_messages.insert(
                    0,
                    ChatMessage::system(format!(
                        "When your result is ready, call the tool `{}` with the exact JSON object.",
                        tool_name
                    ))
                    .build(),
                );
            }
        }
        GenerateMode::Json => {
            // Add a JSON-only instruction as a separate system hint, does not overwrite user content
            final_messages.insert(
                0,
                ChatMessage::system(
                    "Return ONLY a valid JSON value. No prose, no markdown fences, no backticks.",
                )
                .build(),
            );
        }
        GenerateMode::Auto => {
            // For auto + schema, provide a gentle JSON-only hint; optionally add a fallback tool when user didn't provide any tool.
            if opts.schema.is_some() {
                final_messages.insert(
                    0,
                    ChatMessage::system(
                        "Prefer returning a valid JSON value only; avoid extra text or markdown.",
                    )
                    .build(),
                );
                if tools_vec.is_empty() {
                    let tool_name = opts
                        .schema_name
                        .clone()
                        .unwrap_or_else(|| "submit_object".to_string());
                    let desc = opts.schema_description.clone().unwrap_or_else(|| {
                        "Return the final JSON object by calling this tool".to_string()
                    });
                    // Use a minimal, lenient schema as fallback if none was provided (but only when Some exists we use it)
                    if let Some(schema) = opts.schema.clone() {
                        tools_vec.push(Tool::function(tool_name.clone(), desc, schema));
                        final_messages.insert(
                            0,
                            ChatMessage::system(format!(
                                "If needed, call the tool `{}` with the exact final JSON.",
                                tool_name
                            ))
                            .build(),
                        );
                    }
                }
            }
        }
    }

    let mut req = ChatRequest::new(final_messages);
    if let Some(t) = tools {
        req = req.with_tools(t);
    }
    if !tools_vec.is_empty() {
        req = req.with_tools(tools_vec);
    }
    // Build structured_output hint
    let mut hint = serde_json::Map::new();
    // mode
    let mode_str = match opts.mode {
        GenerateMode::Auto => "auto",
        GenerateMode::Json => "json",
        GenerateMode::Tool => "tool",
    };
    hint.insert("mode".into(), serde_json::Value::String(mode_str.into()));
    // schema
    if let Some(schema) = opts.schema.clone() {
        hint.insert("schema".into(), schema);
    }
    if let Some(name) = opts.schema_name.clone() {
        hint.insert("schema_name".into(), serde_json::Value::String(name));
    }
    if let Some(desc) = opts.schema_description.clone() {
        hint.insert("schema_description".into(), serde_json::Value::String(desc));
    }
    // output
    match &opts.output {
        OutputKind::Object => {
            hint.insert("output".into(), serde_json::Value::String("object".into()));
        }
        OutputKind::Array => {
            hint.insert("output".into(), serde_json::Value::String("array".into()));
        }
        OutputKind::Enum(vals) => {
            hint.insert("output".into(), serde_json::Value::String("enum".into()));
            hint.insert("enum".into(), serde_json::Value::Array(vals.clone()));
        }
        OutputKind::NoSchema => {
            hint.insert(
                "output".into(),
                serde_json::Value::String("no-schema".into()),
            );
        }
    }
    // TODO: Migrate to provider_options
    // This highlevel API needs to be refactored to use provider-specific options
    // instead of the deprecated provider_params
    req = req.with_streaming(stream);
    req
}

/// Generic auto convenience using provider params hints (may be ignored by some providers).
pub async fn generate_object_auto<T: DeserializeOwned>(
    model: &impl ChatCapability,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    opts: GenerateObjectOptions,
) -> Result<(T, ChatResponse), LlmError> {
    generate_object(model, messages, tools, opts).await
}

/// Generic auto streaming convenience using provider options/content hints.
pub async fn stream_object_auto<T: DeserializeOwned + Send + 'static>(
    model: &impl ChatCapability,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    opts: StreamObjectOptions,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamObjectEvent<T>, LlmError>> + Send>>, LlmError> {
    stream_object(model, messages, tools, opts).await
}

// === Provider-specific convenience (OpenAI) ===

#[cfg(feature = "openai")]
/// Generate a typed object using OpenAI Responses API structured outputs when possible.
/// This helper creates a ChatRequest with appropriate provider_options for structured output.
pub async fn generate_object_openai<T: DeserializeOwned>(
    client: &crate::providers::openai::OpenAiClient,
    messages: Vec<crate::types::ChatMessage>,
    tools: Option<Vec<crate::types::Tool>>,
    opts: GenerateObjectOptions,
) -> Result<(T, crate::types::ChatResponse), crate::error::LlmError> {
    // Build a ChatRequest with provider_options for structured output
    use crate::types::{ChatRequest, OpenAiOptions, ResponsesApiConfig};

    let mut request = ChatRequest::new(messages);
    if let Some(t) = tools {
        request.tools = Some(t);
    }

    if let Some(schema) = opts.schema.clone() {
        // Build response_format JSON
        let response_format = if let Some(name) = opts.schema_name.clone() {
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": true,
                    "schema": schema
                }
            })
        } else {
            serde_json::json!({
                "type": "json_object",
                "json_schema": {
                    "name": "response",
                    "strict": true,
                    "schema": schema
                }
            })
        };

        request =
            request.with_openai_options(OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new().with_response_format(response_format),
            ));
    }

    // Call chat_request and parse the response
    let resp = client.chat_request(request).await?;

    // Extract text content (prefer tool call arguments if present)
    let mut text = {
        let tool_calls = resp.tool_calls();
        if let Some(first) = tool_calls.first() {
            if let crate::types::ContentPart::ToolCall { arguments, .. } = first {
                serde_json::to_string(arguments).unwrap_or_default()
            } else {
                resp.content_text()
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            }
        } else {
            resp.content_text()
                .map(|s| s.to_string())
                .unwrap_or_default()
        }
    };

    if text.is_empty() {
        return Err(crate::error::LlmError::ParseError(
            "No content for object generation".into(),
        ));
    }

    // Parse and repair with retries
    let mut rounds = 0usize;
    loop {
        match try_parse_validate_deserialize::<T>(&text, opts.schema.as_ref(), &opts.output) {
            Ok(obj) => return Ok((obj, resp)),
            Err(e) => {
                if rounds >= opts.max_repair_rounds {
                    return Err(e);
                }
                if let Some(repair) = &opts.repair_text {
                    if let Some(next) = repair(&text) {
                        text = next;
                        rounds += 1;
                        continue;
                    }
                } else if let Some(next) = default_repair_text(&text) {
                    text = next;
                    rounds += 1;
                    continue;
                }
                return Err(e);
            }
        }
    }
}

#[cfg(feature = "openai")]
/// Stream a typed object using OpenAI Responses API structured outputs when possible.
/// Creates a ChatRequest with appropriate provider_options for structured output.
pub async fn stream_object_openai<T: DeserializeOwned + Send + 'static>(
    client: &crate::providers::openai::OpenAiClient,
    messages: Vec<crate::types::ChatMessage>,
    tools: Option<Vec<crate::types::Tool>>,
    opts: StreamObjectOptions,
) -> Result<
    Pin<Box<dyn Stream<Item = Result<StreamObjectEvent<T>, crate::error::LlmError>> + Send>>,
    crate::error::LlmError,
> {
    // Build a ChatRequest with provider_options for structured output
    use crate::types::{ChatRequest, OpenAiOptions, ResponsesApiConfig};

    let mut request = ChatRequest::new(messages);
    if let Some(t) = tools {
        request.tools = Some(t);
    }
    request.stream = true;

    if let Some(schema) = opts.schema.clone() {
        // Build response_format JSON
        let response_format = if let Some(name) = opts.schema_name.clone() {
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": true,
                    "schema": schema
                }
            })
        } else {
            serde_json::json!({
                "type": "json_object",
                "json_schema": {
                    "name": "response",
                    "strict": true,
                    "schema": schema
                }
            })
        };

        request =
            request.with_openai_options(OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new().with_response_format(response_format),
            ));
    }

    // Stream and process events
    let mut stream = client.chat_stream_request(request).await?;
    let schema = opts.schema.clone();
    let output = opts.output.clone();
    let repair = opts.repair_text.clone();
    let max_rounds = opts.max_repair_rounds;
    let emit_partial = opts.emit_partial_object;

    let s = async_stream::try_stream! {
        use futures::StreamExt;
        let mut acc = String::new();
        let mut final_resp: Option<ChatResponse> = None;
        let mut last_partial: Option<serde_json::Value> = None;

        while let Some(item) = stream.next().await {
            match item? {
                crate::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
                    acc.push_str(&delta);
                    yield StreamObjectEvent::TextDelta { delta };
                    if emit_partial {
                        if let Some(slice) = extract_balanced_json_slice(&acc) {
                            let cand = strip_trailing_commas(slice);
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&cand) {
                                let changed = match &last_partial {
                                    Some(prev) => prev != &v,
                                    None => true,
                                };
                                if changed {
                                    last_partial = Some(v.clone());
                                    yield StreamObjectEvent::PartialObject { partial: v };
                                }
                            }
                        }
                    }
                }
                crate::streaming::ChatStreamEvent::StreamEnd { response } => {
                    final_resp = Some(response);
                    break;
                }
                _ => {}
            }
        }

        let resp = final_resp.ok_or_else(|| LlmError::ParseError("No final response".into()))?;
        let mut text = resp.content_text().map(|s| s.to_string()).unwrap_or_else(|| acc.clone());
        if text.is_empty() {
            text = acc;
        }

        // Parse and repair
        let mut rounds = 0usize;
        loop {
            match try_parse_validate_deserialize::<T>(&text, schema.as_ref(), &output) {
                Ok(obj) => {
                    yield StreamObjectEvent::Final { object: obj, response: resp };
                    break;
                }
                Err(e) => {
                    if rounds >= max_rounds {
                        Err(e)?;
                        break;
                    }
                    if let Some(repair_fn) = &repair {
                        if let Some(next) = repair_fn(&text) {
                            text = next;
                            rounds += 1;
                            continue;
                        }
                    } else if let Some(next) = default_repair_text(&text) {
                        text = next;
                        rounds += 1;
                        continue;
                    }
                    Err(e)?;
                    break;
                }
            }
        }
    };
    Ok(Box::pin(s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct User {
        name: String,
        age: u32,
    }

    struct StreamOnlyModel {
        deltas: Vec<&'static str>,
    }
    #[async_trait]
    impl ChatCapability for StreamOnlyModel {
        async fn chat_with_tools(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::types::ChatResponse, LlmError> {
            Err(LlmError::UnsupportedOperation("non-stream".into()))
        }
        async fn chat_stream(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::streaming::ChatStream, LlmError> {
            let chunks = self.deltas.clone();
            let s = async_stream::try_stream! {
                for d in chunks {
                    yield crate::streaming::ChatStreamEvent::ContentDelta { delta: d.to_string(), index: None };
                }
                yield crate::streaming::ChatStreamEvent::StreamEnd { response: crate::types::ChatResponse::new(crate::types::MessageContent::Text(String::new())) };
            };
            Ok(Box::pin(s))
        }
    }

    #[tokio::test]
    async fn stream_object_emits_partial_on_balanced_block() {
        // JSON appears across multiple deltas; partial should emit once balanced
        let model = StreamOnlyModel {
            deltas: vec!["prefix ", "{", "\"id\":1", "}", " suffix"],
        };
        let mut s = stream_object::<serde_json::Value>(&model, vec![], None, Default::default())
            .await
            .expect("stream");
        use futures::StreamExt;
        let mut saw_partial = false;
        let mut saw_final = false;
        while let Some(ev) = s.next().await {
            match ev.expect("ok") {
                StreamObjectEvent::PartialObject { partial } => {
                    saw_partial = true;
                    assert_eq!(partial.get("id").and_then(|v| v.as_u64()), Some(1));
                }
                StreamObjectEvent::Final { object, .. } => {
                    saw_final = true;
                    assert_eq!(object.get("id").and_then(|v| v.as_u64()), Some(1));
                }
                _ => {}
            }
        }
        assert!(saw_partial && saw_final);
    }

    #[tokio::test]
    async fn stream_object_repairs_trailing_comma() {
        // Trailing comma appears before closing brace; repair should handle
        let model = StreamOnlyModel {
            deltas: vec!["{\"a\":1,", "}\n"],
        };
        let mut s = stream_object::<serde_json::Value>(&model, vec![], None, Default::default())
            .await
            .expect("stream");
        use futures::StreamExt;
        let mut final_obj: Option<serde_json::Value> = None;
        while let Some(ev) = s.next().await {
            if let StreamObjectEvent::Final { object, .. } = ev.expect("ok") {
                final_obj = Some(object);
            }
        }
        let obj = final_obj.expect("final");
        assert_eq!(obj.get("a").and_then(|v| v.as_u64()), Some(1));
    }

    struct MockModel;
    #[async_trait]
    impl ChatCapability for MockModel {
        async fn chat_with_tools(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::types::ChatResponse, LlmError> {
            Ok(crate::types::ChatResponse::new(
                crate::types::MessageContent::Text("{\"name\":\"Ada\",\"age\":36}".to_string()),
            ))
        }

        async fn chat_stream(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::streaming::ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation("no stream".into()))
        }
    }

    #[tokio::test]
    async fn generate_object_happy_path() {
        let model = MockModel;
        let schema = serde_json::json!({
            "type":"object",
            "properties":{
                "name":{"type":"string"},
                "age":{"type":"integer","minimum":0}
            },
            "required":["name","age"]
        });
        let (user, _resp): (User, _) = generate_object(
            &model,
            vec![crate::types::ChatMessage::user("give me user json").build()],
            None,
            GenerateObjectOptions {
                schema: Some(schema),
                output: OutputKind::Object,
                ..Default::default()
            },
        )
        .await
        .expect("object");
        assert_eq!(
            user,
            User {
                name: "Ada".into(),
                age: 36
            }
        );
    }

    #[tokio::test]
    async fn stream_object_emits_partial_updates() {
        use futures::StreamExt;

        struct MockStreamModel;
        #[async_trait]
        impl ChatCapability for MockStreamModel {
            async fn chat_with_tools(
                &self,
                _messages: Vec<crate::types::ChatMessage>,
                _tools: Option<Vec<crate::types::Tool>>,
            ) -> Result<crate::types::ChatResponse, LlmError> {
                Err(LlmError::UnsupportedOperation("no sync".into()))
            }

            async fn chat_stream(
                &self,
                _messages: Vec<crate::types::ChatMessage>,
                _tools: Option<Vec<crate::types::Tool>>,
            ) -> Result<crate::streaming::ChatStream, LlmError> {
                let s = async_stream::try_stream! {
                    yield crate::types::ChatStreamEvent::ContentDelta { delta: "{".into(), index: None };
                    yield crate::types::ChatStreamEvent::ContentDelta { delta: "\"name\"".into(), index: None };
                    yield crate::types::ChatStreamEvent::ContentDelta { delta: ":\"Ada\",".into(), index: None };
                    yield crate::types::ChatStreamEvent::ContentDelta { delta: "\"age\":36}".into(), index: None };
                    yield crate::types::ChatStreamEvent::StreamEnd { response: crate::types::ChatResponse::new(crate::types::MessageContent::Text("".into())) };
                };
                Ok(Box::pin(s))
            }
        }

        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct U {
            name: String,
            age: u32,
        }

        let model = MockStreamModel;
        let mut s = stream_object::<U>(
            &model,
            vec![crate::types::ChatMessage::user("user").build()],
            None,
            StreamObjectOptions {
                emit_partial_object: true,
                ..Default::default()
            },
        )
        .await
        .expect("stream");
        let mut saw_partial = false;
        let mut saw_final = false;
        while let Some(ev) = s.next().await {
            match ev.expect("ok") {
                StreamObjectEvent::PartialObject { partial } => {
                    // eventually becomes object with both fields
                    if partial.get("name").is_some() {
                        saw_partial = true;
                    }
                }
                StreamObjectEvent::Final { object, .. } => {
                    assert_eq!(
                        object,
                        U {
                            name: "Ada".into(),
                            age: 36
                        }
                    );
                    saw_final = true;
                }
                _ => {}
            }
        }
        assert!(saw_partial, "should emit at least one partial object");
        assert!(saw_final, "should emit final object");
    }
}

#[cfg(all(test, feature = "openai"))]
mod openai_integration_tests {
    use super::*;

    struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
    impl crate::execution::http::interceptor::HttpInterceptor for Capture {
        fn on_before_send(
            &self,
            _ctx: &crate::execution::http::interceptor::HttpRequestContext,
            _rb: reqwest::RequestBuilder,
            body: &serde_json::Value,
            _headers: &reqwest::header::HeaderMap,
        ) -> Result<reqwest::RequestBuilder, crate::error::LlmError> {
            *self.0.lock().unwrap() = Some(body.clone());
            Err(crate::error::LlmError::InvalidParameter("stop".into()))
        }
    }

    #[test]
    fn openai_generate_object_injects_response_format_object() {
        // Prepare OpenAI client
        let cfg =
            crate::providers::openai::OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let client = crate::providers::openai::OpenAiClient::new(cfg, reqwest::Client::new());
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        // A simple object schema without name triggers json_object format
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        });
        let messages = vec![crate::types::ChatMessage::user("hi").build()];
        let opts = GenerateObjectOptions {
            schema: Some(schema),
            ..Default::default()
        };
        // Invoke; interceptor will abort before network
        let _ = futures::executor::block_on(async {
            let _ =
                generate_object_openai::<serde_json::Value>(&client, messages, None, opts).await;
        });
        let body = captured.lock().unwrap().clone().expect("captured body");
        let rf = body.get("response_format").cloned().expect("format");
        assert_eq!(rf.get("type").and_then(|v| v.as_str()), Some("json_object"));
    }

    #[test]
    fn openai_stream_object_injects_response_format_named_schema() {
        let cfg =
            crate::providers::openai::OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
        let client = crate::providers::openai::OpenAiClient::new(cfg, reqwest::Client::new());
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

        let schema = serde_json::json!({
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"]
        });
        let messages = vec![crate::types::ChatMessage::user("hi").build()];
        let opts = StreamObjectOptions {
            schema: Some(schema),
            schema_name: Some("User".into()),
            ..Default::default()
        };
        // Invoke streaming helper; interceptor will abort before HTTP
        let _ = futures::executor::block_on(async {
            let _ = stream_object_openai::<serde_json::Value>(&client, messages, None, opts).await;
        });
        let body = captured.lock().unwrap().clone().expect("captured body");
        let rf = body.get("response_format").cloned().expect("format");
        assert_eq!(rf.get("type").and_then(|v| v.as_str()), Some("json_schema"));
        assert_eq!(
            rf.get("json_schema")
                .and_then(|o| o.get("name"))
                .and_then(|v| v.as_str()),
            Some("User")
        );
    }
}
