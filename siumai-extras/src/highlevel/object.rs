#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::large_enum_variant)]
//! High-level structured object generation APIs
//!
//! Provides a minimal, provider-agnostic wrapper
//! to generate typed JSON objects using any ChatCapability model. The function
//! performs optional JSON Schema validation and optional text repair before
//! deserializing into `T`.

use futures::Stream;
use serde::de::DeserializeOwned;
use siumai::error::LlmError;
use siumai::traits::ChatCapability;
use siumai::types::{ChatMessage, ChatRequest, ChatResponse, Tool, Usage};
use std::pin::Pin;

use crate::structured_output::{
    GenerateMode, OutputDecodeConfig, OutputKind, RepairFn, decode_typed,
};

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
    pub repair_text: Option<RepairFn>,
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
    let text = {
        let tool_calls = resp.tool_calls();
        if let Some(first) = tool_calls.first() {
            if let siumai::types::ContentPart::ToolCall { arguments, .. } = first {
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

    let cfg = OutputDecodeConfig {
        schema: opts
            .schema
            .clone()
            .map(|schema| siumai::types::OutputSchema {
                schema,
                name: opts.schema_name.clone(),
                description: opts.schema_description.clone(),
            }),
        kind: opts.output.clone(),
        mode: opts.mode,
        emit_partial: false,
        repair_text: opts.repair_text.clone(),
        max_repair_rounds: opts.max_repair_rounds,
    };

    let obj = decode_typed::<T>(&text, &cfg)?;
    Ok((obj, resp))
}

/// Stream options for `stream_object`.
pub struct StreamObjectOptions {
    /// Optional JSON Schema definition used for validation hints.
    pub schema: Option<serde_json::Value>,
    /// Optional human-readable schema name passed as a hint to the model.
    pub schema_name: Option<String>,
    /// Optional human-readable schema description passed as a hint to the model.
    pub schema_description: Option<String>,
    /// Desired output shape (object/array/enum) used to guide decoding.
    pub output: OutputKind,
    /// Output generation mode (auto/strict) controlling how strongly to enforce shape.
    pub mode: GenerateMode,
    /// Optional custom text repair function applied before JSON parsing.
    pub repair_text: Option<RepairFn>,
    /// Maximum number of repair attempts before giving up.
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
    TextDelta {
        /// The raw text chunk emitted by the model.
        delta: String,
    },
    /// Parsed partial object update if current buffer is valid JSON.
    PartialObject {
        /// Latest parsed JSON value built from the accumulated stream buffer.
        partial: serde_json::Value,
    },
    /// Usage update passthrough.
    UsageUpdate {
        /// Updated token usage statistics.
        usage: Usage,
    },
    /// Final parsed and validated object with the underlying response.
    Final {
        /// The final typed object decoded from the model output.
        object: T,
        /// The underlying raw chat response used to build the object.
        response: ChatResponse,
    },
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
    let output_kind = opts.output.clone();
    let mode = opts.mode;
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
                siumai::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
                    acc.push_str(&delta);
                    yield StreamObjectEvent::TextDelta { delta };
                    if emit_partial {
                        // Try parse a balanced JSON slice from the accumulated text.
                        if let Some(slice) = crate::structured_output::extract_balanced_json_slice(&acc) {
                            let cand = crate::structured_output::strip_trailing_commas(slice);
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
                siumai::streaming::ChatStreamEvent::ToolCallDelta { arguments_delta: Some(d), .. } => {
                    tool_args_acc.push_str(&d);
                }
                siumai::streaming::ChatStreamEvent::UsageUpdate { usage } => {
                    yield StreamObjectEvent::UsageUpdate { usage };
                }
                siumai::streaming::ChatStreamEvent::StreamEnd { response } => {
                    final_resp = Some(response);
                    break;
                }
                _ => {}
            }
        }
        let resp = final_resp
            .unwrap_or_else(|| ChatResponse::new(siumai::types::MessageContent::Text(acc.clone())));
        // Try parse/validate/deserialize with optional repair
        // Prefer tool arguments if present
        let text = if !tool_args_acc.is_empty() { tool_args_acc } else { acc };
        let cfg = OutputDecodeConfig {
            schema: opts.schema.clone().map(|schema| siumai::types::OutputSchema {
                schema,
                name: opts.schema_name.clone(),
                description: opts.schema_description.clone(),
            }),
            kind: output_kind,
            mode,
            emit_partial: emit_partial,
            repair_text: repair,
            max_repair_rounds: max_rounds,
        };

        let obj = decode_typed::<T>(&text, &cfg)?;
        yield StreamObjectEvent::Final { object: obj, response: resp };
    };
    Ok(Box::pin(s))
}

/// Extract a balanced JSON substring from the given text if possible.
///
/// This scans for the first '{' or '[' and then tracks brace/bracket balance,
/// ignoring occurrences within string literals. When balance returns to zero,
/// returns the substring covering that balanced JSON block.
// Balanced-slice helpers now live in `crate::structured_output` and are reused
// here for computing partial JSON objects.

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
    // If a JSON schema is provided, attach OpenAI Responses API options as a best‑effort hint.
    // Non‑OpenAI providers will simply ignore these options via ProviderSpec logic.
    if let Some(schema) = opts.schema.clone() {
        #[cfg(feature = "openai")]
        {
            use siumai::provider_ext::openai::{OpenAiOptions, ResponsesApiConfig};
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
            req = req.with_openai_options(OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new().with_response_format(response_format),
            ));
        }

        #[cfg(feature = "anthropic")]
        {
            use siumai::provider_ext::anthropic::AnthropicOptions;
            if let Some(name) = opts.schema_name.clone() {
                let opts_an = AnthropicOptions::new().with_json_schema(name, schema.clone(), true);
                req = req.with_anthropic_options(opts_an);
            } else {
                let opts_an = AnthropicOptions::new().with_json_object();
                req = req.with_anthropic_options(opts_an);
            }
        }

        #[cfg(feature = "google")]
        {
            use siumai::provider_ext::gemini::GeminiOptions;
            // Ask Gemini to return JSON by setting the response MIME type
            let opts_g = GeminiOptions::new().with_response_mime_type("application/json");
            req = req.with_gemini_options(opts_g);
        }
    }
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
    client: &siumai::providers::openai::OpenAiClient,
    messages: Vec<siumai::types::ChatMessage>,
    tools: Option<Vec<siumai::types::Tool>>,
    opts: GenerateObjectOptions,
) -> Result<(T, siumai::types::ChatResponse), LlmError> {
    // Build a ChatRequest with provider_options for structured output
    use siumai::provider_ext::openai::{OpenAiOptions, ResponsesApiConfig};
    use siumai::types::ChatRequest;

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
    let text = {
        let tool_calls = resp.tool_calls();
        if let Some(first) = tool_calls.first() {
            if let siumai::types::ContentPart::ToolCall { arguments, .. } = first {
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

    let cfg = OutputDecodeConfig {
        schema: opts
            .schema
            .clone()
            .map(|schema| siumai::types::OutputSchema {
                schema,
                name: opts.schema_name.clone(),
                description: opts.schema_description.clone(),
            }),
        kind: opts.output.clone(),
        mode: opts.mode,
        emit_partial: false,
        repair_text: opts.repair_text.clone(),
        max_repair_rounds: opts.max_repair_rounds,
    };

    let obj = decode_typed::<T>(&text, &cfg)?;
    Ok((obj, resp))
}

#[cfg(feature = "openai")]
/// Stream a typed object using OpenAI Responses API structured outputs when possible.
/// Creates a ChatRequest with appropriate provider_options for structured output.
pub async fn stream_object_openai<T: DeserializeOwned + Send + 'static>(
    client: &siumai::providers::openai::OpenAiClient,
    messages: Vec<siumai::types::ChatMessage>,
    tools: Option<Vec<siumai::types::Tool>>,
    opts: StreamObjectOptions,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamObjectEvent<T>, LlmError>> + Send>>, LlmError> {
    // Build a ChatRequest with provider_options for structured output
    use siumai::provider_ext::openai::{OpenAiOptions, ResponsesApiConfig};
    use siumai::types::ChatRequest;

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
                siumai::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
                    acc.push_str(&delta);
                    yield StreamObjectEvent::TextDelta { delta };
                    if emit_partial {
                        if let Some(slice) = crate::structured_output::extract_balanced_json_slice(&acc) {
                            let cand = crate::structured_output::strip_trailing_commas(slice);
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
                siumai::streaming::ChatStreamEvent::StreamEnd { response } => {
                    final_resp = Some(response);
                    break;
                }
                _ => {}
            }
        }

        let resp = final_resp.ok_or_else(|| LlmError::ParseError("No final response".into()))?;
        let text = resp
            .content_text()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or(acc);

        let cfg = OutputDecodeConfig {
            schema: schema.clone().map(|schema| siumai::types::OutputSchema {
                schema,
                name: opts.schema_name.clone(),
                description: opts.schema_description.clone(),
            }),
            kind: output.clone(),
            mode: opts.mode,
            emit_partial,
            repair_text: repair.clone(),
            max_repair_rounds: max_rounds,
        };

        let obj = decode_typed::<T>(&text, &cfg)?;
        yield StreamObjectEvent::Final { object: obj, response: resp };
    };
    Ok(Box::pin(s))
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "openai"))]
mod openai_integration_tests;
