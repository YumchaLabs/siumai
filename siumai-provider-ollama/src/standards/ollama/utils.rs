//! Ollama utility functions (protocol layer)
//!
//! Common utility functions for building Ollama request/response payloads.

use super::params::OllamaParams;
use super::types::*;
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::provider_options::OllamaOptions;
use crate::types::{
    ChatMessage, ChatRequest, CommonParams, EmbeddingRequest, EmbeddingResponse, Tool,
};
use base64::Engine;
use reqwest::header::HeaderMap;
use std::collections::HashMap;

/// Build HTTP headers for Ollama requests
pub fn build_headers(additional_headers: &HashMap<String, String>) -> Result<HeaderMap, LlmError> {
    let version = env!("CARGO_PKG_VERSION");
    let builder = HttpHeaderBuilder::new()
        .with_json_content_type()
        .with_user_agent(&format!("siumai-provider-ollama/{version}"))?
        .with_custom_headers(additional_headers)?;
    Ok(builder.build())
}

#[cfg(test)]
mod header_tests {
    use super::*;

    #[test]
    fn build_headers_sets_json_and_user_agent() {
        let headers = build_headers(&HashMap::new()).unwrap();
        assert_eq!(
            headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        assert!(headers.contains_key(reqwest::header::USER_AGENT));
    }
}

/// Convert common `ChatMessage` to Ollama format
pub fn convert_chat_message(message: &ChatMessage) -> OllamaChatMessage {
    let role_str = match message.role {
        crate::types::MessageRole::System => "system",
        crate::types::MessageRole::User => "user",
        crate::types::MessageRole::Assistant => "assistant",
        crate::types::MessageRole::Developer => "system", // Map developer to system
        crate::types::MessageRole::Tool => "tool",
    }
    .to_string();

    let content_str = match &message.content {
        crate::types::MessageContent::Text(text) => text.clone(),
        crate::types::MessageContent::MultiModal(parts) => {
            // Extract text from multimodal content
            parts
                .iter()
                .filter_map(|part| {
                    if let crate::types::ContentPart::Text { text, .. } = part {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        #[cfg(feature = "structured-messages")]
        crate::types::MessageContent::Json(v) => serde_json::to_string(v).unwrap_or_default(),
    };

    let mut ollama_message = OllamaChatMessage {
        role: role_str,
        content: content_str,
        images: None,
        tool_calls: None,
        thinking: None,
    };

    // Extract images from multimodal content
    if let crate::types::MessageContent::MultiModal(parts) = &message.content {
        let images: Vec<String> = parts
            .iter()
            .filter_map(|part| {
                if let crate::types::ContentPart::Image { source, .. } = part {
                    match source {
                        crate::types::chat::MediaSource::Url { url } => Some(url.clone()),
                        crate::types::chat::MediaSource::Base64 { data } => {
                            Some(format!("data:image/jpeg;base64,{}", data))
                        }
                        crate::types::chat::MediaSource::Binary { data } => {
                            let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                            Some(format!("data:image/jpeg;base64,{}", encoded))
                        }
                    }
                } else {
                    None
                }
            })
            .collect();

        if !images.is_empty() {
            ollama_message.images = Some(images);
        }
    }

    // Convert tool calls if present
    let tool_calls = message.tool_calls();
    if !tool_calls.is_empty() {
        ollama_message.tool_calls = Some(
            tool_calls
                .iter()
                .filter_map(|part| {
                    if let crate::types::ContentPart::ToolCall {
                        tool_name,
                        arguments,
                        ..
                    } = part
                    {
                        Some(OllamaToolCall {
                            function: OllamaFunctionCall {
                                name: tool_name.clone(),
                                arguments: arguments.clone(),
                            },
                        })
                    } else {
                        None
                    }
                })
                .collect(),
        );
    }

    ollama_message
}

/// Convert common Tool to Ollama format
pub fn convert_tool(tool: &Tool) -> Option<OllamaTool> {
    match tool {
        Tool::Function { function } => Some(OllamaTool {
            tool_type: "function".to_string(),
            function: OllamaFunction {
                name: function.name.clone(),
                description: function.description.clone(),
                parameters: function.parameters.clone(),
            },
        }),
        Tool::ProviderDefined(_) => {
            // Ollama doesn't support provider-defined tools
            // Return None to skip them
            None
        }
    }
}

/// Convert Ollama chat message to common format
pub fn convert_from_ollama_message(message: &OllamaChatMessage) -> ChatMessage {
    let role = match message.role.as_str() {
        "system" => crate::types::MessageRole::System,
        "user" => crate::types::MessageRole::User,
        "assistant" => crate::types::MessageRole::Assistant,
        "tool" => crate::types::MessageRole::Tool,
        _ => crate::types::MessageRole::Assistant, // Default fallback
    };

    let mut parts = vec![crate::types::ContentPart::Text {
        text: message.content.clone(),
        provider_metadata: None,
    }];

    // Add images if present
    if let Some(images) = &message.images {
        for image_url in images {
            parts.push(crate::types::ContentPart::Image {
                source: crate::types::chat::MediaSource::Url {
                    url: image_url.clone(),
                },
                detail: None,
                provider_metadata: None,
            });
        }
    }

    // Add tool calls if present
    if let Some(tool_calls) = &message.tool_calls {
        for tc in tool_calls {
            parts.push(crate::types::ContentPart::tool_call(
                format!("call_{}", chrono::Utc::now().timestamp_millis()),
                tc.function.name.clone(),
                tc.function.arguments.clone(),
                None,
            ));
        }
    }

    // Add thinking content if present
    if let Some(thinking) = &message.thinking {
        parts.push(crate::types::ContentPart::reasoning(thinking));
    }

    // Determine final content
    let content = if parts.len() == 1 && parts[0].is_text() {
        crate::types::MessageContent::Text(message.content.clone())
    } else {
        crate::types::MessageContent::MultiModal(parts)
    };

    ChatMessage {
        role,
        content,
        metadata: crate::types::MessageMetadata::default(),
    }
}

// Deprecated ToolCall conversions removed; use ContentPart::ToolCall helpers instead.

/// Parse streaming response line
pub fn parse_streaming_line(line: &str) -> Result<Option<serde_json::Value>, LlmError> {
    let line = line.trim();

    // Skip empty lines and comments
    if line.is_empty() || line.starts_with(':') {
        return Ok(None);
    }

    // Remove "data: " prefix if present
    let json_str = if let Some(stripped) = line.strip_prefix("data: ") {
        stripped
    } else {
        line
    };

    // Skip [DONE] marker
    if json_str == "[DONE]" {
        return Ok(None);
    }

    // Parse JSON
    serde_json::from_str(json_str)
        .map(Some)
        .map_err(|e| LlmError::ParseError(format!("Failed to parse streaming response: {e}")))
}

/// Extract model name from model string (handles model:tag format)
pub fn extract_model_name(model: &str) -> String {
    // Ollama models can be in format "model:tag" or just "model"
    // We keep the full format as Ollama expects it
    model.to_string()
}

/// Validate model name format
pub fn validate_model_name(model: &str) -> Result<(), LlmError> {
    if model.is_empty() {
        return Err(LlmError::ConfigurationError(
            "Model name cannot be empty".to_string(),
        ));
    }

    // Basic validation - model names should not contain invalid characters
    if model.contains(' ') || model.contains('\n') || model.contains('\t') {
        return Err(LlmError::ConfigurationError(
            "Model name contains invalid characters".to_string(),
        ));
    }

    Ok(())
}

/// Build model options from common parameters
pub fn build_model_options(
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    additional_options: Option<&HashMap<String, serde_json::Value>>,
) -> HashMap<String, serde_json::Value> {
    let mut options = HashMap::new();

    if let Some(temp) = temperature {
        options.insert(
            "temperature".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(temp).unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
    }

    if let Some(max_tokens) = max_tokens {
        options.insert(
            "num_predict".to_string(),
            serde_json::Value::Number(serde_json::Number::from(max_tokens)),
        );
    }

    if let Some(top_p) = top_p {
        options.insert(
            "top_p".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(top_p).unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
    }

    if let Some(freq_penalty) = frequency_penalty {
        options.insert(
            "frequency_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(freq_penalty as f64)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
    }

    if let Some(pres_penalty) = presence_penalty {
        options.insert(
            "presence_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(pres_penalty as f64)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
    }

    // Add additional options
    if let Some(additional) = additional_options {
        for (key, value) in additional {
            options.insert(key.clone(), value.clone());
        }
    }

    options
}

fn parse_format_value(format_str: &str) -> Option<serde_json::Value> {
    if format_str.is_empty() {
        return None;
    }
    if format_str == "json" {
        return Some(serde_json::Value::String("json".to_string()));
    }
    match serde_json::from_str(format_str) {
        Ok(schema) => Some(schema),
        Err(_) => Some(serde_json::Value::String(format_str.to_string())),
    }
}

fn apply_ollama_runtime_options(
    options: &mut HashMap<String, serde_json::Value>,
    params: &OllamaParams,
) {
    if let Some(numa) = params.numa {
        options.insert("numa".to_string(), serde_json::Value::Bool(numa));
    }
    if let Some(num_ctx) = params.num_ctx {
        options.insert(
            "num_ctx".to_string(),
            serde_json::Value::Number(serde_json::Number::from(num_ctx)),
        );
    }
    if let Some(num_batch) = params.num_batch {
        options.insert(
            "num_batch".to_string(),
            serde_json::Value::Number(serde_json::Number::from(num_batch)),
        );
    }
    if let Some(num_gpu) = params.num_gpu {
        options.insert(
            "num_gpu".to_string(),
            serde_json::Value::Number(serde_json::Number::from(num_gpu)),
        );
    }
    if let Some(main_gpu) = params.main_gpu {
        options.insert(
            "main_gpu".to_string(),
            serde_json::Value::Number(serde_json::Number::from(main_gpu)),
        );
    }
    if let Some(use_mmap) = params.use_mmap {
        options.insert("use_mmap".to_string(), serde_json::Value::Bool(use_mmap));
    }
    if let Some(num_thread) = params.num_thread {
        options.insert(
            "num_thread".to_string(),
            serde_json::Value::Number(serde_json::Number::from(num_thread)),
        );
    }
}

fn apply_common_params_options(
    options: &mut HashMap<String, serde_json::Value>,
    common: &CommonParams,
) {
    if let Some(stop) = &common.stop_sequences {
        options.insert(
            "stop".to_string(),
            serde_json::Value::Array(
                stop.iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
    }
    if let Some(seed) = common.seed {
        options.insert(
            "seed".to_string(),
            serde_json::Value::Number(serde_json::Number::from(seed)),
        );
    }
}

/// Build an Ollama `/api/chat` request from a unified `ChatRequest` + client defaults.
///
/// This function merges:
/// - `ChatRequest.common_params` (temperature/max_tokens/top_p/stop/seed)
/// - client-level `OllamaParams` (keep_alive/format/think + runtime options)
/// - per-request `providerOptions["ollama"]` (keep_alive/format + extra_params)
pub fn build_chat_request(
    request: &ChatRequest,
    default_params: &OllamaParams,
) -> Result<OllamaChatRequest, LlmError> {
    let model = request.common_params.model.clone();
    if model.is_empty() {
        return Err(LlmError::ConfigurationError(
            "Model is required".to_string(),
        ));
    }
    validate_model_name(&model)?;

    // Convert messages
    let messages: Vec<OllamaChatMessage> =
        request.messages.iter().map(convert_chat_message).collect();

    // Convert tools
    let tools = request
        .tools
        .as_ref()
        .map(|tools| tools.iter().filter_map(convert_tool).collect());

    // Merge provider options
    let mut keep_alive = default_params.keep_alive.clone();
    let mut format_str = default_params.format.clone();
    let mut think_override: Option<bool> = None;
    let mut extra_params: HashMap<String, serde_json::Value> = HashMap::new();

    let options_value = request.provider_options_map.get("ollama").cloned();

    if let Some(val) = options_value
        && let Ok(opts) = serde_json::from_value::<OllamaOptions>(val)
    {
        if opts.keep_alive.is_some() {
            keep_alive = opts.keep_alive.clone();
        }
        if opts.format.is_some() {
            format_str = opts.format.clone();
        }
        think_override = opts.extra_params.get("think").and_then(|v| v.as_bool());
        extra_params = opts.extra_params.clone();
        extra_params.remove("think");
    }

    // Build options: common params + runtime options + custom maps
    let mut options = build_model_options(
        request.common_params.temperature,
        request.common_params.max_tokens,
        request.common_params.top_p,
        None,
        None,
        None,
    );
    apply_common_params_options(&mut options, &request.common_params);
    apply_ollama_runtime_options(&mut options, default_params);
    if let Some(map) = &default_params.stop {
        options.entry("stop".to_string()).or_insert_with(|| {
            serde_json::Value::Array(map.iter().cloned().map(serde_json::Value::String).collect())
        });
    }
    if let Some(custom) = &default_params.options {
        for (k, v) in custom {
            options.insert(k.clone(), v.clone());
        }
    }
    for (k, v) in extra_params.into_iter() {
        options.insert(k, v);
    }

    // Determine thinking behavior
    let think = think_override.or(default_params.think).or_else(|| {
        let m = model.to_lowercase();
        if m.contains("deepseek-r1") || m.contains("qwen3") {
            Some(true)
        } else {
            None
        }
    });

    Ok(OllamaChatRequest {
        model,
        messages,
        tools,
        stream: Some(request.stream),
        format: format_str.as_deref().and_then(parse_format_value),
        options: if options.is_empty() {
            None
        } else {
            Some(options)
        },
        keep_alive,
        think,
    })
}

/// Build an Ollama `/api/embed` request from a unified `EmbeddingRequest` + client defaults.
///
/// This function merges:
/// - `EmbeddingRequest.model` (fallback to `default_model`)
/// - client-level `OllamaParams.keep_alive` and `OllamaParams.options`
/// - per-request `providerOptions["ollama"]` (`keep_alive`, plus `extra_params`)
///
/// Notes:
/// - `truncate` is carried via `providerOptions["ollama"].extra_params["truncate"]`.
/// - Other `extra_params` are merged into `options`.
pub fn build_embedding_request(
    request: &EmbeddingRequest,
    default_model: &str,
    default_params: &OllamaParams,
) -> Result<OllamaEmbeddingRequest, LlmError> {
    if request.input.is_empty() {
        return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
    }

    let model = request
        .model
        .clone()
        .unwrap_or_else(|| default_model.to_string());
    validate_model_name(&model)?;

    // Convert input to appropriate format
    let input_value = if request.input.len() == 1 {
        serde_json::Value::String(request.input[0].clone())
    } else {
        serde_json::Value::Array(
            request
                .input
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        )
    };

    // Defaults (client-level)
    let mut keep_alive = default_params.keep_alive.clone();
    let mut truncate: Option<bool> = Some(true);
    let mut extra_params: HashMap<String, serde_json::Value> = HashMap::new();

    // Per-request overrides
    let options_value = request.provider_options_map.get("ollama").cloned();

    if let Some(val) = options_value
        && let Ok(opts) = serde_json::from_value::<OllamaOptions>(val)
    {
        if opts.keep_alive.is_some() {
            keep_alive = opts.keep_alive.clone();
        }
        if let Some(b) = opts.extra_params.get("truncate").and_then(|v| v.as_bool()) {
            truncate = Some(b);
        }
        extra_params = opts.extra_params.clone();
        extra_params.remove("truncate");
    }

    // Build options: runtime + client options + per-request extra params
    let mut options: HashMap<String, serde_json::Value> = HashMap::new();
    apply_ollama_runtime_options(&mut options, default_params);
    if let Some(custom) = &default_params.options {
        for (k, v) in custom {
            options.insert(k.clone(), v.clone());
        }
    }
    for (k, v) in extra_params.into_iter() {
        options.insert(k, v);
    }

    Ok(OllamaEmbeddingRequest {
        model,
        input: input_value,
        truncate,
        options: if options.is_empty() {
            None
        } else {
            Some(options)
        },
        keep_alive,
    })
}

pub fn convert_embedding_response(response: OllamaEmbeddingResponse) -> EmbeddingResponse {
    // Convert f64 to f32
    let embeddings: Vec<Vec<f32>> = response
        .embeddings
        .into_iter()
        .map(|embedding| embedding.into_iter().map(|x| x as f32).collect())
        .collect();

    let usage = response
        .prompt_eval_count
        .map(|count| crate::types::EmbeddingUsage::new(count, count));

    let mut result = EmbeddingResponse::new(embeddings, response.model);
    if let Some(usage) = usage {
        result = result.with_usage(usage);
    }

    if let Some(total_duration) = response.total_duration {
        result = result.with_metadata(
            "total_duration_ns".to_string(),
            serde_json::Value::Number(serde_json::Number::from(total_duration)),
        );
    }
    if let Some(load_duration) = response.load_duration {
        result = result.with_metadata(
            "load_duration_ns".to_string(),
            serde_json::Value::Number(serde_json::Number::from(load_duration)),
        );
    }

    result
}

/// Calculate tokens per second from Ollama response metrics
pub fn calculate_tokens_per_second(
    eval_count: Option<u32>,
    eval_duration: Option<u64>,
) -> Option<f64> {
    match (eval_count, eval_duration) {
        (Some(count), Some(duration)) if duration > 0 => {
            // Convert nanoseconds to seconds and calculate tokens/second
            let duration_seconds = duration as f64 / 1_000_000_000.0;
            Some(count as f64 / duration_seconds)
        }
        _ => None,
    }
}

/// Convert an Ollama chat response to the unified `ChatResponse` (includes provider metadata).
pub fn convert_chat_response(response: OllamaChatResponse) -> crate::types::ChatResponse {
    let message = convert_from_ollama_message(&response.message);

    // Usage
    let usage = if response.prompt_eval_count.is_some() || response.eval_count.is_some() {
        let prompt = response.prompt_eval_count.unwrap_or(0);
        let completion = response.eval_count.unwrap_or(0);
        Some(
            crate::types::Usage::builder()
                .prompt_tokens(prompt)
                .completion_tokens(completion)
                .total_tokens(prompt + completion)
                .build(),
        )
    } else {
        None
    };

    // Finish reason
    let finish_reason = response
        .done_reason
        .as_deref()
        .map(|reason| match reason {
            "stop" => crate::types::FinishReason::Stop,
            "length" => crate::types::FinishReason::Length,
            _ => crate::types::FinishReason::Other(reason.to_string()),
        })
        .or({
            if response.done {
                Some(crate::types::FinishReason::Stop)
            } else {
                None
            }
        });

    // Provider metadata
    let mut ollama_metadata = HashMap::new();
    if let Some(tokens_per_second) =
        calculate_tokens_per_second(response.eval_count, response.eval_duration)
    {
        ollama_metadata.insert(
            "tokens_per_second".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(tokens_per_second)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
    }
    if let Some(total_duration) = response.total_duration {
        ollama_metadata.insert(
            "total_duration_ms".to_string(),
            serde_json::Value::Number(serde_json::Number::from(total_duration / 1_000_000)),
        );
    }

    let provider_metadata = if !ollama_metadata.is_empty() {
        let mut meta = HashMap::new();
        meta.insert("ollama".to_string(), ollama_metadata);
        Some(meta)
    } else {
        None
    };

    crate::types::ChatResponse {
        id: Some(format!("ollama-{}", chrono::Utc::now().timestamp_millis())),
        content: message.content,
        model: Some(response.model),
        usage,
        finish_reason,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        provider_metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{CONTENT_TYPE, USER_AGENT};

    #[test]
    fn test_build_headers() {
        let additional = HashMap::new();
        let headers = build_headers(&additional).unwrap();

        assert!(headers.contains_key(CONTENT_TYPE));
        assert!(headers.contains_key(USER_AGENT));
    }

    #[test]
    fn test_convert_chat_message() {
        let message = ChatMessage {
            role: crate::types::MessageRole::User,
            content: crate::types::MessageContent::MultiModal(vec![
                crate::types::ContentPart::Text {
                    text: "Hello".to_string(),
                    provider_metadata: None,
                },
                crate::types::ContentPart::Image {
                    source: crate::types::chat::MediaSource::Url {
                        url: "image1".to_string(),
                    },
                    detail: None,
                    provider_metadata: None,
                },
            ]),
            metadata: crate::types::MessageMetadata::default(),
        };

        let ollama_message = convert_chat_message(&message);
        assert_eq!(ollama_message.role, "user");
        assert_eq!(ollama_message.content, "Hello");
        assert_eq!(ollama_message.images, Some(vec!["image1".to_string()]));
    }

    #[test]
    fn test_validate_model_name() {
        assert!(validate_model_name("llama3.2").is_ok());
        assert!(validate_model_name("llama3.2:latest").is_ok());
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name("model with spaces").is_err());
    }

    #[test]
    fn test_calculate_tokens_per_second() {
        assert_eq!(
            calculate_tokens_per_second(Some(100), Some(1_000_000_000)),
            Some(100.0)
        );
        assert_eq!(
            calculate_tokens_per_second(Some(50), Some(500_000_000)),
            Some(100.0)
        );
        assert_eq!(calculate_tokens_per_second(None, Some(1_000_000_000)), None);
        assert_eq!(calculate_tokens_per_second(Some(100), None), None);
        assert_eq!(calculate_tokens_per_second(Some(100), Some(0)), None);
    }
}
