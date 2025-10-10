//! `OpenAI` Responses API Implementation
//!
//! This module implements the `OpenAI` Responses API which combines the simplicity
//! of Chat Completions with the tool-use capabilities of the Assistants API.
//! It supports built-in tools like web search, file search, and computer use.
//!
//! The Responses API provides:
//! - Stateful conversations with automatic context management
//! - Background processing for long-running tasks
//! - Built-in tools (web search, file search, computer use)
//! - Response lifecycle management (create, get, cancel, list)
//!
//! API Reference: <https://platform.openai.com/docs/api-reference/responses>

use async_trait::async_trait;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, OpenAiBuiltInTool, Tool};
use crate::web_search::{WebSearchCapability, WebSearchProvider};

use super::config::OpenAiConfig;

/// Response status for background processing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponseStatus {
    /// Response is being processed
    InProgress,
    /// Response completed successfully
    Completed,
    /// Response failed with an error
    Failed,
    /// Response was cancelled
    Cancelled,
    /// Response is queued for processing
    Queued,
    /// Response is incomplete
    Incomplete,
}

/// Response metadata for Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Response ID
    pub id: String,
    /// Response status
    pub status: ResponseStatus,
    /// Creation timestamp
    pub created_at: u64,
    /// Completion timestamp (if completed)
    pub completed_at: Option<u64>,
    /// Model used
    pub model: String,
    /// Whether this was a background request
    pub background: bool,
    /// Previous response ID (if chained)
    pub previous_response_id: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// List responses query parameters
#[derive(Debug, Clone, Default)]
pub struct ListResponsesQuery {
    /// Limit number of results
    pub limit: Option<u32>,
    /// Pagination cursor
    pub after: Option<String>,
    /// Filter by status
    pub status: Option<ResponseStatus>,
    /// Sort order (asc/desc)
    pub order: Option<crate::params::openai::SortOrder>,
}

/// `OpenAI` Responses API client
#[allow(dead_code)]
pub struct OpenAiResponses {
    /// HTTP client
    http_client: reqwest::Client,
    /// API configuration
    config: OpenAiConfig,
    /// Web search provider
    web_search: Option<WebSearchProvider>,
}

impl OpenAiResponses {
    /// Create a new Responses API client
    pub fn new(http_client: reqwest::Client, config: OpenAiConfig) -> Self {
        let web_search = if config.web_search_config.enabled {
            Some(WebSearchProvider::new(
                "openai".to_string(),
                config.web_search_config.clone(),
            ))
        } else {
            None
        };

        Self {
            http_client,
            config,
            web_search,
        }
    }

    /// Get the responses endpoint
    fn responses_endpoint(&self) -> String {
        format!("{}/responses", self.config.base_url)
    }

    /// Get a specific response endpoint
    fn response_endpoint(&self, response_id: &str) -> String {
        format!("{}/responses/{}", self.config.base_url, response_id)
    }

    /// Get response cancel endpoint
    fn response_cancel_endpoint(&self, response_id: &str) -> String {
        format!("{}/responses/{}/cancel", self.config.base_url, response_id)
    }

    /// Delete a response by ID
    pub async fn delete_response(&self, response_id: &str) -> Result<bool, LlmError> {
        let response = self
            .http_client
            .delete(self.response_endpoint(response_id))
            .headers({
                let mut hm = reqwest::header::HeaderMap::new();
                for (k, v) in self.config.get_headers() {
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
                    let val = reqwest::header::HeaderValue::from_str(&v)
                        .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
                    hm.insert(name, val);
                }
                hm
            })
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI delete response error: {error_text}"),
            ));
        }

        Ok(true)
    }

    /// List items within a response (if supported)
    pub async fn list_response_items(
        &self,
        response_id: &str,
    ) -> Result<serde_json::Value, LlmError> {
        let url = format!("{}/responses/{}/items", self.config.base_url, response_id);
        let response = self
            .http_client
            .get(url)
            .headers({
                let mut hm = reqwest::header::HeaderMap::new();
                for (k, v) in self.config.get_headers() {
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
                    let val = reqwest::header::HeaderValue::from_str(&v)
                        .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
                    hm.insert(name, val);
                }
                hm
            })
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI list response items error: {error_text}"),
            ));
        }

        response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))
    }

    /// Create a response with background processing
    pub async fn create_response_background(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        built_in_tools: Option<Vec<OpenAiBuiltInTool>>,
        previous_response_id: Option<String>,
    ) -> Result<ResponseMetadata, LlmError> {
        let request_body = self.build_request_body_with_options(
            &messages,
            tools.as_deref(),
            built_in_tools.as_deref(),
            false, // stream = false for background
            true,  // background = true
            previous_response_id,
        )?;

        let response = self
            .http_client
            .post(self.responses_endpoint())
            .header(
                "Authorization",
                format!("Bearer {}", self.config.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI Responses API background error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response_metadata(response_data)
    }

    /// Get a response by ID
    pub async fn get_response(&self, response_id: &str) -> Result<ChatResponse, LlmError> {
        let response = self
            .http_client
            .get(self.response_endpoint(response_id))
            .headers({
                let mut hm = reqwest::header::HeaderMap::new();
                for (k, v) in self.config.get_headers() {
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
                    let val = reqwest::header::HeaderValue::from_str(&v)
                        .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
                    hm.insert(name, val);
                }
                hm
            })
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI get response error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response(response_data)
    }

    /// Cancel a background response
    pub async fn cancel_response(&self, response_id: &str) -> Result<ResponseMetadata, LlmError> {
        let response = self
            .http_client
            .post(self.response_cancel_endpoint(response_id))
            .headers({
                let mut hm = reqwest::header::HeaderMap::new();
                for (k, v) in self.config.get_headers() {
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
                    let val = reqwest::header::HeaderValue::from_str(&v)
                        .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
                    hm.insert(name, val);
                }
                hm
            })
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI cancel response error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response_metadata(response_data)
    }

    /// List responses with optional filtering
    pub async fn list_responses(
        &self,
        query: Option<ListResponsesQuery>,
    ) -> Result<Vec<ResponseMetadata>, LlmError> {
        let mut url = self.responses_endpoint();

        if let Some(q) = query {
            let mut params = Vec::new();

            if let Some(limit) = q.limit {
                params.push(format!("limit={limit}"));
            }
            if let Some(after) = q.after {
                params.push(format!("after={after}"));
            }
            if let Some(status) = q.status {
                let status_str = match status {
                    ResponseStatus::InProgress => "in_progress",
                    ResponseStatus::Completed => "completed",
                    ResponseStatus::Failed => "failed",
                    ResponseStatus::Cancelled => "cancelled",
                    ResponseStatus::Queued => "queued",
                    ResponseStatus::Incomplete => "incomplete",
                };
                params.push(format!("status={status_str}"));
            }
            if let Some(order) = q.order {
                let order_str = match order {
                    crate::params::openai::SortOrder::Asc => "asc",
                    crate::params::openai::SortOrder::Desc => "desc",
                };
                params.push(format!("order={order_str}"));
            }

            if !params.is_empty() {
                url.push('?');
                url.push_str(&params.join("&"));
            }
        }

        let response = self
            .http_client
            .get(&url)
            .headers({
                let mut hm = reqwest::header::HeaderMap::new();
                for (k, v) in self.config.get_headers() {
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
                    let val = reqwest::header::HeaderValue::from_str(&v)
                        .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
                    hm.insert(name, val);
                }
                hm
            })
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI list responses error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        // Parse the list of responses
        let responses = response_data
            .get("data")
            .and_then(|data| data.as_array())
            .ok_or_else(|| LlmError::ParseError("Invalid response format".to_string()))?;

        let mut result = Vec::new();
        for response_item in responses {
            result.push(self.parse_response_metadata(response_item.clone())?);
        }

        Ok(result)
    }

    /// Build request body for Responses API
    fn build_request_body(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        built_in_tools: Option<&[OpenAiBuiltInTool]>,
        stream: bool,
        background: bool,
    ) -> Result<serde_json::Value, LlmError> {
        self.build_request_body_with_options(
            messages,
            tools,
            built_in_tools,
            stream,
            background,
            self.config.previous_response_id.clone(),
        )
    }

    /// Build request body for Responses API with additional options
    fn build_request_body_with_options(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        built_in_tools: Option<&[OpenAiBuiltInTool]>,
        stream: bool,
        background: bool,
        previous_response_id: Option<String>,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::json!({
            "model": self.config.common_params.model,
            "stream": stream,
            "background": self.config.openai_params.background.unwrap_or(background),
        });

        // Add previous response ID for chaining
        if let Some(prev_id) = previous_response_id {
            body["previous_response_id"] = serde_json::Value::String(prev_id);
        }

        // Build Responses API input items from messages
        // - Tool result messages become function_call_output items
        // - Other messages become role-based message objects
        let mut input_items: Vec<serde_json::Value> = Vec::new();
        for msg in messages.iter() {
            match msg.role {
                crate::types::MessageRole::Tool => {
                    // Convert tool message to function_call_output item
                    let call_id = msg.tool_call_id.as_ref().ok_or_else(|| {
                        LlmError::provider_error(
                            "OpenAI",
                            "Tool message missing tool_call_id for function_call_output",
                        )
                    })?;
                    let output_text = match &msg.content {
                        crate::types::MessageContent::Text(t) => t.clone(),
                        _ => String::new(),
                    };
                    input_items.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_text,
                    }));
                }
                _ => {
                    // Convert non-tool messages
                    input_items.push(self.convert_message_to_api_format(msg)?);
                }
            }
        }

        // Responses API expects `input` to be an array of input items
        body["input"] = serde_json::Value::Array(input_items);

        // Add instructions if provided
        if let Some(ref instructions) = self.config.openai_params.instructions {
            body["instructions"] = serde_json::Value::String(instructions.clone());
        }

        // Add include array if provided
        if let Some(ref include) = self.config.openai_params.include {
            body["include"] = serde_json::to_value(include)
                .map_err(|e| LlmError::ParseError(format!("Failed to serialize include: {}", e)))?;
        }

        // Add truncation strategy if provided
        if let Some(ref truncation) = self.config.openai_params.truncation {
            body["truncation"] = serde_json::to_value(truncation).map_err(|e| {
                LlmError::ParseError(format!("Failed to serialize truncation: {}", e))
            })?;
        }

        // Add reasoning configuration if provided
        if let Some(ref reasoning) = self.config.openai_params.reasoning {
            body["reasoning"] = reasoning.clone();
        }

        // Add text configuration if provided
        if let Some(ref text) = self.config.openai_params.text {
            body["text"] = text.clone();
        }

        // Add prompt configuration if provided
        if let Some(ref prompt) = self.config.openai_params.prompt {
            body["prompt"] = prompt.clone();
        }

        // Add stream options if provided
        if let Some(ref stream_options) = self.config.openai_params.stream_options {
            body["stream_options"] = stream_options.clone();
        }

        // Add optional parameters
        if let Some(temp) = self.config.common_params.temperature
            && let Some(num) = serde_json::Number::from_f64(temp as f64)
        {
            body["temperature"] = serde_json::Value::Number(num);
        }

        // Prefer Responses API specific max_output_tokens, then max_completion_tokens, then fallback to common max_tokens
        if let Some(max_output) = self.config.openai_params.max_output_tokens {
            body["max_output_tokens"] = serde_json::Value::Number(max_output.into());
        } else if let Some(max_comp) = self.config.openai_params.max_completion_tokens {
            body["max_output_tokens"] = serde_json::Value::Number(max_comp.into());
        } else if let Some(max_tokens) = self.config.common_params.max_tokens {
            body["max_output_tokens"] = serde_json::Value::Number(max_tokens.into());
        }

        // Add max_tool_calls if provided
        if let Some(max_tool_calls) = self.config.openai_params.max_tool_calls {
            body["max_tool_calls"] = serde_json::Value::Number(max_tool_calls.into());
        }

        // Seed for reproducibility (if supported by model)
        if let Some(seed) = self.config.common_params.seed {
            body["seed"] = serde_json::Value::Number(seed.into());
        }

        if let Some(top_p) = self.config.common_params.top_p
            && let Some(num) = serde_json::Number::from_f64(top_p as f64)
        {
            body["top_p"] = serde_json::Value::Number(num);
        }

        // Stop sequences -> OpenAI uses "stop" for arrays
        if let Some(stops) = &self.config.common_params.stop_sequences {
            body["stop"] = serde_json::Value::Array(
                stops
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            );
        }

        // OpenAI-specific parameters for Responses API
        if let Some(ref rf) = self.config.openai_params.response_format
            && let Ok(val) = serde_json::to_value(rf)
        {
            body["response_format"] = val;
        }
        // Include tool_choice when configured
        if let Some(tool_choice) = &self.config.openai_params.tool_choice
            && let Ok(val) = serde_json::to_value(tool_choice)
        {
            // Keep string forms as strings ("auto"|"required"|"none"); keep object for specific function
            body["tool_choice"] = val;
        }
        if let Some(parallel) = self.config.openai_params.parallel_tool_calls {
            body["parallel_tool_calls"] = serde_json::Value::Bool(parallel);
        }
        if let Some(store) = self.config.openai_params.store {
            body["store"] = serde_json::Value::Bool(store);
        }
        if let Some(ref meta) = self.config.openai_params.metadata {
            // Convert HashMap<String, String> to JSON object
            let mut obj = serde_json::Map::new();
            for (k, v) in meta.iter() {
                obj.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            body["metadata"] = serde_json::Value::Object(obj);
        }
        if let Some(ref user) = self.config.openai_params.user {
            body["user"] = serde_json::Value::String(user.clone());
        }
        if let Some(ref safety_id) = self.config.openai_params.safety_identifier {
            body["safety_identifier"] = serde_json::Value::String(safety_id.clone());
        }
        if let Some(ref cache_key) = self.config.openai_params.prompt_cache_key {
            body["prompt_cache_key"] = serde_json::Value::String(cache_key.clone());
        }

        // Build tools array
        let mut all_tools = Vec::new();

        // Add function tools
        if let Some(tools) = tools {
            for tool in tools {
                all_tools.push(self.convert_tool_to_responses_format(tool)?);
            }
        }

        // Add built-in tools (deduplicated by 'type')
        use std::collections::HashSet;
        let mut seen_types: HashSet<String> = HashSet::new();
        if let Some(built_in_tools) = built_in_tools {
            for tool in built_in_tools {
                let json = tool.to_json();
                let t = json
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if seen_types.insert(t) {
                    all_tools.push(json);
                }
            }
        }

        // Add web search tool if enabled (skip if already present)
        if self.config.web_search_config.enabled {
            let json = OpenAiBuiltInTool::WebSearch.to_json();
            let t = json
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if seen_types.insert(t) {
                all_tools.push(json);
            }
        }

        if !all_tools.is_empty() {
            body["tools"] = serde_json::Value::Array(all_tools);
            // Note: OpenAI Responses API doesn't support tool_choice="auto" for custom function tools
            // Only set tool_choice if explicitly configured by the user
            // Built-in tools (file_search, web_search, etc.) are handled differently
        }

        Ok(body)
    }

    /// Parse response metadata from API response
    fn parse_response_metadata(
        &self,
        response_data: serde_json::Value,
    ) -> Result<ResponseMetadata, LlmError> {
        let id = response_data
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::ParseError("Missing response ID".to_string()))?
            .to_string();

        let status_str = response_data
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("in_progress");

        let status = match status_str {
            "in_progress" => ResponseStatus::InProgress,
            "completed" => ResponseStatus::Completed,
            "failed" => ResponseStatus::Failed,
            "cancelled" => ResponseStatus::Cancelled,
            "queued" => ResponseStatus::Queued,
            "incomplete" => ResponseStatus::Incomplete,
            _ => ResponseStatus::InProgress,
        };

        let created_at = response_data
            .get("created_at")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let completed_at = response_data.get("completed_at").and_then(|v| v.as_u64());

        let model = response_data
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.config.common_params.model)
            .to_string();

        let background = response_data
            .get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let previous_response_id = response_data
            .get("previous_response_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let error = response_data
            .get("error")
            .and_then(|v| v.get("message"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(ResponseMetadata {
            id,
            status,
            created_at,
            completed_at,
            model,
            background,
            previous_response_id,
            error,
        })
    }

    /// Convert `ChatMessage` to API format
    fn convert_message_to_api_format(
        &self,
        message: &ChatMessage,
    ) -> Result<serde_json::Value, LlmError> {
        // Responses API does not accept role "tool" in input; map tool messages to user content with tool_result blocks
        if matches!(message.role, crate::types::MessageRole::Tool) {
            let mut api_message = serde_json::json!({
                "role": "user",
            });
            let result_text = match &message.content {
                crate::types::MessageContent::Text(t) => t.clone(),
                _ => String::new(),
            };
            let mut content_parts: Vec<serde_json::Value> = Vec::new();
            let mut tool_result = serde_json::json!({
                "type": "tool_result",
                "content": result_text,
            });
            if let Some(id) = &message.tool_call_id {
                tool_result["tool_call_id"] = serde_json::Value::String(id.clone());
            }
            content_parts.push(tool_result);
            api_message["content"] = serde_json::Value::Array(content_parts);
            return Ok(api_message);
        }

        let mut api_message = serde_json::json!({
            "role": match message.role {
                crate::types::MessageRole::System => "system",
                crate::types::MessageRole::User => "user",
                crate::types::MessageRole::Assistant => "assistant",
                crate::types::MessageRole::Developer => "developer",
                crate::types::MessageRole::Tool => "user", // handled above, default to user if reached
            }
        });

        // Special handling for assistant tool calls: convert to Responses API tool_use content
        if matches!(message.role, crate::types::MessageRole::Assistant)
            && message.tool_calls.is_some()
        {
            let mut content_parts: Vec<serde_json::Value> = Vec::new();

            // Include any assistant text as a separate content part
            if let crate::types::MessageContent::Text(text) = &message.content {
                if !text.is_empty() {
                    content_parts.push(serde_json::json!({
                        "type": "input_text",
                        "text": text
                    }));
                }
            } else if let crate::types::MessageContent::MultiModal(parts) = &message.content {
                // Preserve multimodal text parts if present
                for part in parts {
                    if let crate::types::ContentPart::Text { text } = part {
                        content_parts
                            .push(serde_json::json!({ "type": "input_text", "text": text }));
                    }
                }
            }

            // Add tool_use items for each tool call
            if let Some(tool_calls) = &message.tool_calls {
                for call in tool_calls {
                    let (name, args_str) = if let Some(func) = &call.function {
                        (func.name.clone(), func.arguments.clone())
                    } else {
                        (String::new(), String::new())
                    };
                    if !name.is_empty() {
                        // Arguments must be a JSON object; fall back to empty object on parse error
                        let input_json = serde_json::from_str::<serde_json::Value>(&args_str)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        content_parts.push(serde_json::json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": name,
                            "input": input_json
                        }));
                    }
                }
            }

            api_message["content"] = serde_json::Value::Array(content_parts);

            // Do not include legacy chat.completions-style tool_calls field in Responses API
            if api_message.get("tool_calls").is_some() {
                api_message.as_object_mut().unwrap().remove("tool_calls");
            }

            // Handle tool call ID (not typically used on assistant messages)
            if let Some(tool_call_id) = &message.tool_call_id {
                api_message["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
            }

            return Ok(api_message);
        }

        // Default content handling (no assistant tool calls)
        match &message.content {
            crate::types::MessageContent::Text(text) => {
                // Responses API expects content as array with proper type
                // All input messages (including assistant messages from conversation history) use "input_text"
                api_message["content"] = serde_json::Value::Array(vec![serde_json::json!({
                    "type": "input_text",
                    "text": text
                })]);
            }
            crate::types::MessageContent::MultiModal(parts) => {
                let mut content_parts = Vec::new();
                // All input messages use "input_text" for text content
                for part in parts {
                    match part {
                        crate::types::ContentPart::Text { text } => {
                            content_parts.push(serde_json::json!({
                                "type": "input_text",
                                "text": text
                            }));
                        }
                        crate::types::ContentPart::Image { image_url, detail } => {
                            let mut image_part = serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            });
                            if let Some(detail) = detail {
                                image_part["image_url"]["detail"] =
                                    serde_json::Value::String(detail.clone());
                            }
                            content_parts.push(image_part);
                        }
                        crate::types::ContentPart::Audio { audio_url, format } => {
                            content_parts.push(serde_json::json!({
                                "type": "audio",
                                "audio_url": audio_url,
                                "format": format
                            }));
                        }
                    }
                }
                api_message["content"] = serde_json::Value::Array(content_parts);
            }
        }

        // For tool role messages, include tool_call_id when present
        if let Some(tool_call_id) = &message.tool_call_id {
            api_message["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
        }

        Ok(api_message)
    }

    /// Convert Tool to Responses API format
    fn convert_tool_to_responses_format(&self, tool: &Tool) -> Result<serde_json::Value, LlmError> {
        // OpenAI Responses API expects flattened tool format (no nested `function` object)
        Ok(serde_json::json!({
            "type": tool.r#type,
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters
        }))
    }

    /// Parse response from Responses API
    fn parse_response(&self, response_data: serde_json::Value) -> Result<ChatResponse, LlmError> {
        // Helper: map finish reason strings to our enum
        fn map_finish_reason(s: Option<&str>) -> Option<crate::types::FinishReason> {
            match s {
                Some("stop") => Some(crate::types::FinishReason::Stop),
                Some("length") | Some("max_tokens") => Some(crate::types::FinishReason::Length),
                Some("tool_calls") | Some("tool_use") | Some("function_call") => {
                    Some(crate::types::FinishReason::ToolCalls)
                }
                Some("content_filter") | Some("safety") => {
                    Some(crate::types::FinishReason::ContentFilter)
                }
                Some(other) => Some(crate::types::FinishReason::Other(other.to_string())),
                None => None,
            }
        }

        // Some Responses payloads wrap data under { "response": { ... } }
        let root = response_data.get("response").unwrap_or(&response_data);

        // Extract full text content: traverse output[*].content[*].text or string fallbacks
        let mut text_content = String::new();
        let mut tool_calls_acc: Vec<crate::types::ToolCall> = Vec::new();

        if let Some(output_items) = root.get("output").and_then(|o| o.as_array()) {
            for item in output_items {
                // Case A: tool calls embedded as array on the item (old shape)
                if let Some(calls) = item.get("tool_calls").and_then(|tc| tc.as_array()) {
                    for call in calls {
                        let id = call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let r#type = call
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("function")
                            .to_string();
                        // Support both nested { function: { name, arguments } } and flattened { name, arguments }
                        let (name, arguments) = if let Some(f) = call.get("function") {
                            (
                                f.get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                f.get("arguments")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                            )
                        } else {
                            (
                                call.get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                call.get("arguments")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                            )
                        };
                        tool_calls_acc.push(crate::types::ToolCall {
                            id,
                            r#type,
                            function: Some(crate::types::FunctionCall { name, arguments }),
                        });
                    }
                }

                // Case B: Responses API function call item
                match item.get("type").and_then(|v| v.as_str()) {
                    Some("tool_call") | Some("function_call") => {
                        let id = item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let r#type = "function".to_string();
                        let name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if !name.is_empty() {
                            tool_calls_acc.push(crate::types::ToolCall {
                                id,
                                r#type,
                                function: Some(crate::types::FunctionCall { name, arguments }),
                            });
                        }
                    }
                    _ => {}
                }

                // Extract text content
                match item.get("content") {
                    Some(serde_json::Value::String(s)) => {
                        if !text_content.is_empty() {
                            text_content.push('\n');
                        }
                        text_content.push_str(s);
                    }
                    Some(serde_json::Value::Array(parts)) => {
                        for part in parts {
                            // Common shapes: {"type":"output_text","text":"..."} or {"type":"text","text":"..."}
                            if let Some(txt) = part.get("text").and_then(|v| v.as_str()) {
                                if !text_content.is_empty() {
                                    text_content.push('\n');
                                }
                                text_content.push_str(txt);
                            } else if let Some(s) = part.as_str() {
                                if !text_content.is_empty() {
                                    text_content.push('\n');
                                }
                                text_content.push_str(s);
                            }
                        }
                    }
                    _ => {}
                }
            }
        } else if let Some(s) = root.get("output_text").and_then(|v| v.as_str()) {
            // Some schemas provide aggregated output_text
            text_content.push_str(s);
        }

        // Case C: tool calls at the root of the response
        if let Some(root_calls) = root.get("tool_calls").and_then(|tc| tc.as_array()) {
            for call in root_calls {
                let id = call
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let r#type = call
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("function")
                    .to_string();
                let (name, arguments) = if let Some(f) = call.get("function") {
                    (
                        f.get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        f.get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    )
                } else {
                    (
                        call.get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        call.get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    )
                };
                if !name.is_empty() {
                    tool_calls_acc.push(crate::types::ToolCall {
                        id,
                        r#type,
                        function: Some(crate::types::FunctionCall { name, arguments }),
                    });
                }
            }
        }

        // Extract usage information (support snake_case and camelCase)
        let usage = root.get("usage").map(|usage_data| crate::types::Usage {
            prompt_tokens: usage_data
                .get("input_tokens")
                .or_else(|| usage_data.get("prompt_tokens"))
                .or_else(|| usage_data.get("inputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0),
            completion_tokens: usage_data
                .get("output_tokens")
                .or_else(|| usage_data.get("completion_tokens"))
                .or_else(|| usage_data.get("outputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0),
            total_tokens: usage_data
                .get("total_tokens")
                .or_else(|| usage_data.get("totalTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0),
            reasoning_tokens: usage_data
                .get("reasoning_tokens")
                .or_else(|| usage_data.get("reasoningTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32),
            cached_tokens: None,
        });

        // Provider-specific data: include reasoning/thinking if present
        let mut provider_data = HashMap::new();
        if let Some(reasoning) = root.get("reasoning") {
            provider_data.insert("reasoning".to_string(), reasoning.clone());
        }
        if let Some(annotations) = root.get("annotations") {
            provider_data.insert("annotations".to_string(), annotations.clone());
        }

        // Determine finish reason from common fields
        let finish_reason = map_finish_reason(
            root.get("finish_reason")
                .or_else(|| root.get("stop_reason"))
                .and_then(|v| v.as_str()),
        );

        Ok(ChatResponse {
            id: root
                .get("id")
                .and_then(|v| v.as_str())
                .map(std::string::ToString::to_string),
            content: crate::types::MessageContent::Text(text_content),
            model: Some(self.config.common_params.model.clone()),
            usage,
            finish_reason,
            tool_calls: if tool_calls_acc.is_empty() {
                None
            } else {
                Some(tool_calls_acc)
            },
            thinking: provider_data
                .get("reasoning")
                .and_then(|v| v.as_str())
                .map(std::string::ToString::to_string),
            metadata: provider_data,
        })
    }
}

#[async_trait]
impl ChatCapability for OpenAiResponses {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let request_body = self.build_request_body(
            &messages,
            tools.as_deref(),
            Some(&self.config.built_in_tools),
            false,
            false,
        )?;

        let response = self
            .http_client
            .post(self.responses_endpoint())
            .headers({
                let mut hm = reqwest::header::HeaderMap::new();
                for (k, v) in self.config.get_headers() {
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
                    let val = reqwest::header::HeaderValue::from_str(&v)
                        .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
                    hm.insert(name, val);
                }
                hm
            })
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI Responses API error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response(response_data)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let request_body = self.build_request_body(
            &messages,
            tools.as_deref(),
            Some(&self.config.built_in_tools),
            true, // stream = true
            false,
        )?;

        // Build request
        let request_builder = self
            .http_client
            .post(self.responses_endpoint())
            .headers({
                let mut hm = reqwest::header::HeaderMap::new();
                for (k, v) in self.config.get_headers() {
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
                    let val = reqwest::header::HeaderValue::from_str(&v)
                        .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
                    hm.insert(name, val);
                }
                hm.insert(
                    reqwest::header::ACCEPT,
                    reqwest::header::HeaderValue::from_static("text/event-stream"),
                );
                hm
            })
            .json(&request_body);

        // Use unified EventSource-based stream processor for reliability
        let converter = OpenAiResponsesEventConverter::new(self.config.common_params.model.clone());
        crate::utils::streaming::StreamFactory::create_eventsource_stream(
            request_builder,
            converter,
        )
        .await
    }
}

/// OpenAI Responses SSE event converter using unified streaming utilities
#[derive(Clone)]
pub struct OpenAiResponsesEventConverter {
    model: String,
}

impl OpenAiResponsesEventConverter {
    pub fn new(model: String) -> Self {
        Self { model }
    }

    fn convert_responses_event(
        &self,
        json: serde_json::Value,
    ) -> Option<crate::stream::ChatStreamEvent> {
        // Handle delta.content as plain text
        if let Some(delta) = json.get("delta") {
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                return Some(crate::stream::ChatStreamEvent::ContentDelta {
                    delta: content.to_string(),
                    index: None,
                });
            }

            // Handle tool_calls delta
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array())
                && let Some((index, tool_call)) = tool_calls.iter().enumerate().next()
            {
                let id = tool_call
                    .get("id")
                    .and_then(|id| id.as_str())
                    .unwrap_or("")
                    .to_string();

                let function_name = tool_call
                    .get("function")
                    .and_then(|func| func.get("name"))
                    .and_then(|n| n.as_str())
                    .map(std::string::ToString::to_string);

                let arguments_delta = tool_call
                    .get("function")
                    .and_then(|func| func.get("arguments"))
                    .and_then(|a| a.as_str())
                    .map(std::string::ToString::to_string);

                return Some(crate::stream::ChatStreamEvent::ToolCallDelta {
                    id,
                    function_name,
                    arguments_delta,
                    index: Some(index),
                });
            }
        }

        // Handle usage updates with both snake_case and camelCase fields
        if let Some(usage) = json
            .get("usage")
            .or_else(|| json.get("response")?.get("usage"))
        {
            let prompt_tokens = usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
                .or_else(|| usage.get("inputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let completion_tokens = usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
                .or_else(|| usage.get("outputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let total_tokens = usage
                .get("total_tokens")
                .or_else(|| usage.get("totalTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let reasoning_tokens = usage
                .get("reasoning_tokens")
                .or_else(|| usage.get("reasoningTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32);

            let usage_info = crate::types::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                reasoning_tokens,
                cached_tokens: None,
            };
            return Some(crate::stream::ChatStreamEvent::UsageUpdate { usage: usage_info });
        }

        None
    }

    fn convert_function_call_arguments_delta(
        &self,
        json: serde_json::Value,
    ) -> Option<crate::stream::ChatStreamEvent> {
        // Handle response.function_call_arguments.delta events
        let delta = json.get("delta").and_then(|d| d.as_str())?;
        let item_id = json.get("item_id").and_then(|id| id.as_str()).unwrap_or("");
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0);

        Some(crate::stream::ChatStreamEvent::ToolCallDelta {
            id: item_id.to_string(),
            function_name: None, // Function name is set in the initial item.added event
            arguments_delta: Some(delta.to_string()),
            index: Some(output_index as usize),
        })
    }

    fn convert_output_item_added(
        &self,
        json: serde_json::Value,
    ) -> Option<crate::stream::ChatStreamEvent> {
        // Handle response.output_item.added events for function calls
        let item = json.get("item")?;
        if item.get("type").and_then(|t| t.as_str()) != Some("function_call") {
            return None;
        }

        let id = item.get("call_id").and_then(|id| id.as_str()).unwrap_or("");
        let function_name = item.get("name").and_then(|name| name.as_str());
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0);

        Some(crate::stream::ChatStreamEvent::ToolCallDelta {
            id: id.to_string(),
            function_name: function_name.map(|s| s.to_string()),
            arguments_delta: None, // Arguments will come in subsequent delta events
            index: Some(output_index as usize),
        })
    }
}

impl crate::utils::streaming::SseEventConverter for OpenAiResponsesEventConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<Result<crate::stream::ChatStreamEvent, crate::error::LlmError>>,
                > + Send
                + Sync
                + '_,
        >,
    > {
        Box::pin(async move {
            let data_raw = event.data.trim();
            if data_raw.is_empty() {
                return vec![];
            }
            // Consider explicit completed events
            let event_name = event.event.as_str();

            // Debug logging for tool-related events (can be enabled for debugging)
            // println!("🔍 SSE Event: '{}' | Data: '{}'", event_name, data_raw);

            if data_raw == "[DONE]" {
                // [DONE] events should not generate any events in the new architecture
                // StreamEnd events are generated by actual completion events
                return vec![];
            }
            if event_name == "response.completed" {
                // The completed event often contains the full response payload
                // Try to parse it into a final ChatResponse instead of returning empty
                let json = match serde_json::from_str::<serde_json::Value>(data_raw) {
                    Ok(v) => v,
                    Err(e) => {
                        return vec![Err(crate::error::LlmError::ParseError(format!(
                            "Failed to parse completed event JSON: {e}"
                        )))];
                    }
                };
                let root = json.get("response").unwrap_or(&json);

                // Aggregate text content
                let mut text_content = String::new();
                if let Some(output_items) = root.get("output").and_then(|o| o.as_array()) {
                    for item in output_items {
                        match item.get("content") {
                            Some(serde_json::Value::String(s)) => {
                                if !text_content.is_empty() {
                                    text_content.push('\n');
                                }
                                text_content.push_str(s);
                            }
                            Some(serde_json::Value::Array(parts)) => {
                                for part in parts {
                                    if let Some(txt) = part.get("text").and_then(|v| v.as_str()) {
                                        if !text_content.is_empty() {
                                            text_content.push('\n');
                                        }
                                        text_content.push_str(txt);
                                    } else if let Some(s) = part.as_str() {
                                        if !text_content.is_empty() {
                                            text_content.push('\n');
                                        }
                                        text_content.push_str(s);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                } else if let Some(s) = root.get("output_text").and_then(|v| v.as_str()) {
                    text_content.push_str(s);
                }

                // Collect tool calls from output items and root
                let mut tool_calls: Vec<crate::types::ToolCall> = Vec::new();
                if let Some(output_items) = root.get("output").and_then(|o| o.as_array()) {
                    for item in output_items {
                        if let Some(calls) = item.get("tool_calls").and_then(|tc| tc.as_array()) {
                            for call in calls {
                                let id = call
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let r#type = call
                                    .get("type")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("function")
                                    .to_string();
                                let (name, arguments) = if let Some(f) = call.get("function") {
                                    (
                                        f.get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        f.get("arguments")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                    )
                                } else {
                                    (
                                        call.get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        call.get("arguments")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                    )
                                };
                                tool_calls.push(crate::types::ToolCall {
                                    id,
                                    r#type,
                                    function: Some(crate::types::FunctionCall { name, arguments }),
                                });
                            }
                        }
                        if item.get("type").and_then(|v| v.as_str()) == Some("tool_call") {
                            let id = item
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let r#type = item
                                .get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("function")
                                .to_string();
                            let name = item
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let arguments = item
                                .get("arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            if !name.is_empty() {
                                tool_calls.push(crate::types::ToolCall {
                                    id,
                                    r#type,
                                    function: Some(crate::types::FunctionCall { name, arguments }),
                                });
                            }
                        }
                    }
                }
                if let Some(root_calls) = root.get("tool_calls").and_then(|tc| tc.as_array()) {
                    for call in root_calls {
                        let id = call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let r#type = call
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("function")
                            .to_string();
                        let (name, arguments) = if let Some(f) = call.get("function") {
                            (
                                f.get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                f.get("arguments")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                            )
                        } else {
                            (
                                call.get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                call.get("arguments")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                            )
                        };
                        if !name.is_empty() {
                            tool_calls.push(crate::types::ToolCall {
                                id,
                                r#type,
                                function: Some(crate::types::FunctionCall { name, arguments }),
                            });
                        }
                    }
                }

                let response = crate::types::ChatResponse {
                    id: root
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                    content: crate::types::MessageContent::Text(text_content),
                    model: Some(self.model.clone()),
                    usage: None,
                    finish_reason: Some(crate::types::FinishReason::Stop),
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    thinking: None,
                    metadata: std::collections::HashMap::new(),
                };
                return vec![Ok(crate::stream::ChatStreamEvent::StreamEnd { response })];
            }

            // Parse JSON, allow both top-level and nested under {"response": ...}
            let json = match serde_json::from_str::<serde_json::Value>(data_raw) {
                Ok(v) => v,
                Err(e) => {
                    return vec![Err(crate::error::LlmError::ParseError(format!(
                        "Failed to parse SSE JSON: {e}"
                    )))];
                }
            };

            // Route by event name first (cover both tool_call and function_call variants)
            match event_name {
                "response.output_text.delta"
                | "response.tool_call.delta"
                | "response.function_call.delta"
                | "response.usage" => {
                    if let Some(evt) = self.convert_responses_event(json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.function_call_arguments.delta" => {
                    // Handle function call arguments delta from OpenAI Responses API
                    if let Some(evt) = self.convert_function_call_arguments_delta(json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.output_item.added" => {
                    // Handle function call item added
                    if let Some(evt) = self.convert_output_item_added(json) {
                        return vec![Ok(evt)];
                    }
                }
                _ => {
                    // Fallback to generic delta/usage extraction
                    if let Some(evt) = self.convert_responses_event(json) {
                        return vec![Ok(evt)];
                    }
                }
            }

            vec![]
        })
    }

    fn handle_stream_end(
        &self,
    ) -> Option<Result<crate::stream::ChatStreamEvent, crate::error::LlmError>> {
        Some(Ok(crate::stream::ChatStreamEvent::StreamEnd {
            response: crate::types::ChatResponse {
                id: None,
                content: crate::types::MessageContent::Text(String::new()),
                model: Some(self.model.clone()),
                usage: None,
                finish_reason: Some(crate::types::FinishReason::Stop),
                tool_calls: None,
                thinking: None,
                metadata: std::collections::HashMap::new(),
            },
        }))
    }
}

impl OpenAiResponses {
    /// Parse SSE chunk into stream events (static version)
    fn parse_sse_chunk_static(
        chunk: &str,
        _model_name: &str,
    ) -> Vec<crate::stream::ChatStreamEvent> {
        let mut events = Vec::new();

        for line in chunk.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                // Remove "data: " prefix

                if data == "[DONE]" {
                    // Stream end event
                    events.push(crate::stream::ChatStreamEvent::StreamEnd {
                        response: crate::types::ChatResponse {
                            id: None,
                            content: crate::types::MessageContent::Text(String::new()),
                            model: None,
                            usage: None,
                            finish_reason: Some(crate::types::FinishReason::Stop),
                            tool_calls: None,
                            thinking: None,
                            metadata: std::collections::HashMap::new(),
                        },
                    });
                    continue;
                }

                // Try to parse JSON data
                if let Ok(json_data) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(delta) = json_data.get("delta") {
                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                            events.push(crate::stream::ChatStreamEvent::ContentDelta {
                                delta: content.to_string(),
                                index: None,
                            });
                        }

                        if let Some(tool_calls) =
                            delta.get("tool_calls").and_then(|tc| tc.as_array())
                        {
                            for (index, tool_call) in tool_calls.iter().enumerate() {
                                let id = tool_call
                                    .get("id")
                                    .and_then(|id| id.as_str())
                                    .map(std::string::ToString::to_string)
                                    .unwrap_or_default();

                                let function_name = tool_call
                                    .get("function")
                                    .and_then(|func| func.get("name"))
                                    .and_then(|n| n.as_str())
                                    .map(std::string::ToString::to_string);

                                let arguments_delta = tool_call
                                    .get("function")
                                    .and_then(|func| func.get("arguments"))
                                    .and_then(|a| a.as_str())
                                    .map(std::string::ToString::to_string);

                                events.push(crate::stream::ChatStreamEvent::ToolCallDelta {
                                    id,
                                    function_name,
                                    arguments_delta,
                                    index: Some(index),
                                });
                            }
                        }
                    }

                    // Handle usage updates
                    if let Some(usage) = json_data.get("usage") {
                        let usage_info = crate::types::Usage {
                            prompt_tokens: usage
                                .get("prompt_tokens")
                                .and_then(serde_json::Value::as_u64)
                                .map(|v| v as u32)
                                .unwrap_or(0),
                            completion_tokens: usage
                                .get("completion_tokens")
                                .and_then(serde_json::Value::as_u64)
                                .map(|v| v as u32)
                                .unwrap_or(0),
                            total_tokens: usage
                                .get("total_tokens")
                                .and_then(serde_json::Value::as_u64)
                                .map(|v| v as u32)
                                .unwrap_or(0),
                            reasoning_tokens: usage
                                .get("reasoning_tokens")
                                .and_then(serde_json::Value::as_u64)
                                .map(|v| v as u32),
                            cached_tokens: None,
                        };
                        events.push(crate::stream::ChatStreamEvent::UsageUpdate {
                            usage: usage_info,
                        });
                    }
                }
            }
        }

        events
    }

    /// Parse SSE chunk into stream events
    #[allow(dead_code)]
    fn parse_sse_chunk(&self, chunk: &str) -> Vec<crate::stream::ChatStreamEvent> {
        Self::parse_sse_chunk_static(chunk, &self.config.common_params.model)
    }
}

#[async_trait]
impl WebSearchCapability for OpenAiResponses {
    async fn web_search(
        &self,
        query: String,
        _config: Option<crate::types::WebSearchConfig>,
    ) -> Result<Vec<crate::types::WebSearchResult>, LlmError> {
        // Use the built-in web search tool
        let messages = vec![crate::types::ChatMessage {
            role: crate::types::MessageRole::User,
            content: crate::types::MessageContent::Text(query),
            metadata: crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        let built_in_tools = vec![OpenAiBuiltInTool::WebSearch];

        let _request_body =
            self.build_request_body(&messages, None, Some(&built_in_tools), false, false)?;

        // Web search via Responses API is not implemented yet.
        // Return an explicit error instead of a silent success to avoid misleading callers.
        Err(LlmError::UnsupportedOperation(
            "OpenAI Responses API web_search not implemented".to_string(),
        ))
    }

    fn supports_web_search(&self) -> bool {
        true
    }

    fn web_search_strategy(&self) -> crate::types::WebSearchStrategy {
        crate::types::WebSearchStrategy::BuiltIn
    }
}

/// Trait for OpenAI Responses API specific functionality
///
/// This trait extends beyond basic chat capabilities to provide
/// stateful conversation management, background processing, and
/// response lifecycle management specific to OpenAI's Responses API.
#[async_trait]
pub trait ResponsesApiCapability {
    /// Create a response with background processing
    async fn create_response_background(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        built_in_tools: Option<Vec<OpenAiBuiltInTool>>,
        previous_response_id: Option<String>,
    ) -> Result<ResponseMetadata, LlmError>;

    /// Get a response by ID
    async fn get_response(&self, response_id: &str) -> Result<ChatResponse, LlmError>;

    /// Cancel a background response
    async fn cancel_response(&self, response_id: &str) -> Result<ResponseMetadata, LlmError>;

    /// List responses with optional filtering
    async fn list_responses(
        &self,
        query: Option<ListResponsesQuery>,
    ) -> Result<Vec<ResponseMetadata>, LlmError>;

    /// Create a response that continues from a previous response
    async fn continue_conversation(
        &self,
        previous_response_id: String,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        background: bool,
    ) -> Result<ChatResponse, LlmError>;

    /// Check if a response is ready (for background responses)
    async fn is_response_ready(&self, response_id: &str) -> Result<bool, LlmError>;
}

#[async_trait]
impl ResponsesApiCapability for OpenAiResponses {
    async fn create_response_background(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        built_in_tools: Option<Vec<OpenAiBuiltInTool>>,
        previous_response_id: Option<String>,
    ) -> Result<ResponseMetadata, LlmError> {
        self.create_response_background(messages, tools, built_in_tools, previous_response_id)
            .await
    }

    async fn get_response(&self, response_id: &str) -> Result<ChatResponse, LlmError> {
        self.get_response(response_id).await
    }

    async fn cancel_response(&self, response_id: &str) -> Result<ResponseMetadata, LlmError> {
        self.cancel_response(response_id).await
    }

    async fn list_responses(
        &self,
        query: Option<ListResponsesQuery>,
    ) -> Result<Vec<ResponseMetadata>, LlmError> {
        self.list_responses(query).await
    }

    async fn continue_conversation(
        &self,
        previous_response_id: String,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        background: bool,
    ) -> Result<ChatResponse, LlmError> {
        let request_body = self.build_request_body_with_options(
            &messages,
            tools.as_deref(),
            None,
            false,
            background,
            Some(previous_response_id),
        )?;

        let response = self
            .http_client
            .post(self.responses_endpoint())
            .headers({
                let mut hm = reqwest::header::HeaderMap::new();
                for (k, v) in self.config.get_headers() {
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
                    let val = reqwest::header::HeaderValue::from_str(&v)
                        .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
                    hm.insert(name, val);
                }
                hm
            })
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI continue conversation error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response(response_data)
    }

    async fn is_response_ready(&self, response_id: &str) -> Result<bool, LlmError> {
        let metadata = self.get_response_metadata(response_id).await?;
        Ok(matches!(
            metadata.status,
            ResponseStatus::Completed
                | ResponseStatus::Failed
                | ResponseStatus::Cancelled
                | ResponseStatus::Incomplete
        ))
    }
}

impl OpenAiResponses {
    /// Get response metadata without full response content
    pub async fn get_response_metadata(
        &self,
        response_id: &str,
    ) -> Result<ResponseMetadata, LlmError> {
        let response = self
            .http_client
            .get(self.response_endpoint(response_id))
            .header(
                "Authorization",
                format!("Bearer {}", self.config.api_key.expose_secret()),
            )
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI get response metadata error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response_metadata(response_data)
    }
}

#[cfg(test)]
use crate::utils::streaming::SseEventConverter;

#[test]
fn test_responses_event_converter_content_delta() {
    let conv = OpenAiResponsesEventConverter::new("gpt-5-mini".to_string());
    let event = eventsource_stream::Event {
        event: "message".to_string(),
        data: r#"{"delta":{"content":"hello"}}"#.to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let fut = conv.convert_event(event);
    let events = futures::executor::block_on(fut);
    assert!(!events.is_empty());
    let ev = events.first().unwrap().as_ref().unwrap();
    match ev {
        crate::stream::ChatStreamEvent::ContentDelta { delta, .. } => assert_eq!(delta, "hello"),
        _ => panic!("expected ContentDelta"),
    }
}

#[test]
fn test_responses_event_converter_tool_call_delta() {
    let conv = OpenAiResponsesEventConverter::new("gpt-5".to_string());
    let event = eventsource_stream::Event {
            event: "message".to_string(),
            data: r#"{"delta":{"tool_calls":[{"id":"t1","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]}}"#.to_string(),
            id: "1".to_string(),
            retry: None,
        };
    let fut = conv.convert_event(event);
    let events = futures::executor::block_on(fut);
    assert!(!events.is_empty());
    let ev = events.first().unwrap().as_ref().unwrap();
    match ev {
        crate::stream::ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            ..
        } => {
            assert_eq!(id, "t1");
            assert_eq!(function_name.clone().unwrap(), "lookup");
            assert_eq!(arguments_delta.clone().unwrap(), "{\"q\":\"x\"}");
        }
        _ => panic!("expected ToolCallDelta"),
    }
}

#[test]
fn test_responses_event_converter_usage_update() {
    let conv = OpenAiResponsesEventConverter::new("gpt-5".to_string());
    let event = eventsource_stream::Event {
        event: "message".to_string(),
        data: r#"{"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}"#.to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let fut = conv.convert_event(event);
    let events = futures::executor::block_on(fut);
    assert!(!events.is_empty());
    let ev = events.first().unwrap().as_ref().unwrap();
    match ev {
        crate::stream::ChatStreamEvent::UsageUpdate { usage } => {
            assert_eq!(usage.prompt_tokens, 3);
            assert_eq!(usage.completion_tokens, 5);
            assert_eq!(usage.total_tokens, 8);
        }
        _ => panic!("expected UsageUpdate"),
    }
}

#[test]
fn test_responses_event_converter_done() {
    let conv = OpenAiResponsesEventConverter::new("gpt-5".to_string());
    let event = eventsource_stream::Event {
        event: "message".to_string(),
        data: "[DONE]".to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let fut = conv.convert_event(event);
    let events = futures::executor::block_on(fut);
    // [DONE] events should not generate any events in our new architecture
    assert!(events.is_empty());
}

mod tests {
    use crate::providers::openai::config::OpenAiConfig;
    use crate::types::{
        ChatMessage, MessageContent, MessageMetadata, MessageRole, OpenAiBuiltInTool,
    };

    #[allow(dead_code)]
    fn create_test_config() -> OpenAiConfig {
        OpenAiConfig::new("test-key")
            .with_model("gpt-4o")
            .with_responses_api(true)
            .with_built_in_tool(OpenAiBuiltInTool::WebSearch)
    }

    #[allow(dead_code)]
    fn create_test_message() -> ChatMessage {
        ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello, world!".to_string()),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn test_responses_client_creation() {
        let config = create_test_config();
        let client = super::OpenAiResponses::new(reqwest::Client::new(), config);

        // Test that the client was created successfully
        assert_eq!(client.config.common_params.model, "gpt-4o");
        assert!(client.config.use_responses_api);
        assert_eq!(client.config.built_in_tools.len(), 1);
    }

    #[test]
    fn test_responses_endpoint() {
        let config = create_test_config();
        let client = super::OpenAiResponses::new(reqwest::Client::new(), config);

        assert_eq!(
            client.responses_endpoint(),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn test_build_request_body_basic() {
        let config = create_test_config();
        let client = super::OpenAiResponses::new(reqwest::Client::new(), config);
        let messages = vec![create_test_message()];

        let body = client
            .build_request_body(&messages, None, None, false, false)
            .unwrap();

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);
        assert_eq!(body["background"], false);

        // Check input format: Responses input is always an array of input items
        assert!(body["input"].is_array());

        // Check that user message content uses "input_text" type
        let input_array = body["input"].as_array().unwrap();
        assert!(!input_array.is_empty());
        let first_message = &input_array[0];
        if let Some(content) = first_message.get("content")
            && let Some(content_array) = content.as_array()
            && let Some(first_content) = content_array.first()
        {
            assert_eq!(first_content["type"], "input_text");
        }
    }

    #[test]
    fn test_content_types_by_role() {
        let config = create_test_config();
        let client = super::OpenAiResponses::new(reqwest::Client::new(), config);

        // Test user message uses input_text
        let user_message = ChatMessage {
            role: crate::types::MessageRole::User,
            content: crate::types::MessageContent::Text("User message".to_string()),
            metadata: crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        };

        // Test assistant message uses output_text
        let assistant_message = ChatMessage {
            role: crate::types::MessageRole::Assistant,
            content: crate::types::MessageContent::Text("Assistant message".to_string()),
            metadata: crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        };

        let messages = vec![user_message, assistant_message];
        let body = client
            .build_request_body(&messages, None, None, false, false)
            .unwrap();

        let input_array = body["input"].as_array().unwrap();
        assert_eq!(input_array.len(), 2);

        // Check user message content type
        let user_msg = &input_array[0];
        if let Some(content) = user_msg.get("content")
            && let Some(content_array) = content.as_array()
            && let Some(first_content) = content_array.first()
        {
            assert_eq!(first_content["type"], "input_text");
        }

        // Check assistant message content type (also uses input_text in Responses API)
        let assistant_msg = &input_array[1];
        if let Some(content) = assistant_msg.get("content")
            && let Some(content_array) = content.as_array()
            && let Some(first_content) = content_array.first()
        {
            assert_eq!(first_content["type"], "input_text");
        }
    }

    #[test]
    fn test_responses_api_specific_parameters() {
        use crate::params::OpenAiParams;

        let mut config = create_test_config();

        // Set Responses API specific parameters
        config.openai_params = OpenAiParams {
            instructions: Some("You are a helpful assistant.".to_string()),
            include: Some(vec![
                crate::params::openai::IncludableItem::MessageOutputTextLogprobs,
            ]),
            truncation: Some(crate::params::openai::TruncationStrategy::Auto),
            reasoning: Some(serde_json::json!({"effort": "high"})),
            max_output_tokens: Some(1000),
            max_tool_calls: Some(5),
            text: Some(serde_json::json!({"format": {"type": "text"}})),
            background: Some(true),
            ..Default::default()
        };

        let client = super::OpenAiResponses::new(reqwest::Client::new(), config);
        let message = create_test_message();

        let body = client
            .build_request_body(&[message], None, None, false, false)
            .unwrap();

        // Check that Responses API specific parameters are included
        assert_eq!(body["instructions"], "You are a helpful assistant.");
        assert_eq!(
            body["include"],
            serde_json::json!(["message.output_text.logprobs"])
        );
        assert_eq!(body["truncation"], "auto");
        assert_eq!(body["reasoning"], serde_json::json!({"effort": "high"}));
        assert_eq!(body["max_output_tokens"], 1000);
        assert_eq!(body["max_tool_calls"], 5);
        assert_eq!(
            body["text"],
            serde_json::json!({"format": {"type": "text"}})
        );
        assert_eq!(body["background"], true); // Should override the method parameter
    }

    #[test]
    fn test_enum_serialization() {
        use crate::params::OpenAiParams;
        use crate::params::openai::{IncludableItem, TruncationStrategy};

        let mut config = create_test_config();

        // Test enum serialization
        config.openai_params = OpenAiParams {
            include: Some(vec![
                IncludableItem::MessageOutputTextLogprobs,
                IncludableItem::FileSearchCallResults,
            ]),
            truncation: Some(TruncationStrategy::Auto),
            ..Default::default()
        };

        let client = super::OpenAiResponses::new(reqwest::Client::new(), config);
        let message = create_test_message();

        let body = client
            .build_request_body(&[message], None, None, false, false)
            .unwrap();

        // Check that enums are serialized correctly
        assert_eq!(
            body["include"],
            serde_json::json!(["message.output_text.logprobs", "file_search_call.results"])
        );
        assert_eq!(body["truncation"], "auto");
    }

    #[test]
    fn test_parse_response_metadata() {
        let config = create_test_config();
        let client = super::OpenAiResponses::new(reqwest::Client::new(), config);

        let response_data = serde_json::json!({
            "id": "resp_123",
            "status": "completed",
            "created_at": 1234567890,
            "completed_at": 1234567900,
            "model": "gpt-4o",
            "background": true,
            "previous_response_id": "prev_resp_456"
        });

        let metadata = client.parse_response_metadata(response_data).unwrap();

        assert_eq!(metadata.id, "resp_123");
        assert!(matches!(metadata.status, super::ResponseStatus::Completed));
        assert_eq!(metadata.created_at, 1234567890);
        assert_eq!(metadata.completed_at, Some(1234567900));
        assert_eq!(metadata.model, "gpt-4o");
        assert!(metadata.background);
        assert_eq!(
            metadata.previous_response_id,
            Some("prev_resp_456".to_string())
        );
        assert!(metadata.error.is_none());
    }
}

// Additional unit tests for parse_response and request body building
#[cfg(test)]
fn create_test_config() -> OpenAiConfig {
    OpenAiConfig::new("test-key")
        .with_model("gpt-5-mini")
        .with_responses_api(true)
}

#[cfg(test)]
fn create_test_message() -> ChatMessage {
    ChatMessage {
        role: crate::types::MessageRole::User,
        content: crate::types::MessageContent::Text("Hello, world!".to_string()),
        metadata: crate::types::MessageMetadata::default(),
        tool_calls: None,
        tool_call_id: None,
    }
}

#[test]
fn test_parse_response_text_tool_calls_finish_reason_stop() {
    let config = create_test_config();
    let client = super::OpenAiResponses::new(reqwest::Client::new(), config);

    let response_data = serde_json::json!({
        "id": "resp_abc",
        "output": [
            {
                "content": [
                    {"type":"output_text","text":"Hello"},
                    {"type":"output_text","text":"World"}
                ],
                "tool_calls": [
                    {"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}
                ]
            }
        ],
        "finish_reason": "stop",
        "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5}
    });

    let resp = client.parse_response(response_data).unwrap();
    assert_eq!(resp.id.as_ref().unwrap(), "resp_abc");
    assert_eq!(resp.content_text().unwrap(), "Hello\nWorld");
    assert!(matches!(
        resp.finish_reason,
        Some(crate::types::FinishReason::Stop)
    ));
    let tc = resp.tool_calls.unwrap();
    assert_eq!(tc.len(), 1);
    let c0 = &tc[0];
    assert_eq!(c0.id, "call_1");
    assert_eq!(c0.function.as_ref().unwrap().name, "lookup");
    assert_eq!(c0.function.as_ref().unwrap().arguments, "{\"q\":\"x\"}");
    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 2);
    assert_eq!(usage.completion_tokens, 3);
    assert_eq!(usage.total_tokens, 5);
}

#[test]
fn test_parse_response_finish_reason_length_via_stop_reason() {
    let config = create_test_config();
    let client = super::OpenAiResponses::new(reqwest::Client::new(), config);

    let response_data = serde_json::json!({
        "id": "resp_len",
        "output": [{"content": "Partial"}],
        "stop_reason": "max_tokens",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });

    let resp = client.parse_response(response_data).unwrap();
    assert!(matches!(
        resp.finish_reason,
        Some(crate::types::FinishReason::Length)
    ));
    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 1);
    assert_eq!(usage.completion_tokens, 1);
    assert_eq!(usage.total_tokens, 2);
}

#[test]
fn test_sse_named_events_routing() {
    use crate::utils::streaming::SseEventConverter;
    let conv = crate::providers::openai::responses::OpenAiResponsesEventConverter::new(
        "gpt-5".to_string(),
    );

    // content delta via named event
    let ev1 = eventsource_stream::Event {
        event: "response.output_text.delta".to_string(),
        data: r#"{"delta":{"content":"abc"}}"#.to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let events1 = futures::executor::block_on(conv.convert_event(ev1));
    assert!(!events1.is_empty());
    let out1 = events1.first().unwrap().as_ref().unwrap();
    match out1 {
        crate::stream::ChatStreamEvent::ContentDelta { delta, .. } => assert_eq!(delta, "abc"),
        _ => panic!("expected ContentDelta"),
    }

    // tool call delta via named event
    let ev2 = eventsource_stream::Event {
        event: "response.tool_call.delta".to_string(),
        data: r#"{"delta":{"tool_calls":[{"id":"t1","function":{"name":"fn","arguments":"{}"}}]}}"#
            .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let events2 = futures::executor::block_on(conv.convert_event(ev2));
    assert!(!events2.is_empty());
    let out2 = events2.first().unwrap().as_ref().unwrap();
    match out2 {
        crate::stream::ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            ..
        } => {
            assert_eq!(id, "t1");
            assert_eq!(function_name.clone().unwrap(), "fn");
            assert_eq!(arguments_delta.clone().unwrap(), "{}");
        }
        _ => panic!("expected ToolCallDelta"),
    }

    // usage via named event camelCase
    let ev3 = eventsource_stream::Event {
        event: "response.usage".to_string(),
        data: r#"{"usage":{"inputTokens":4,"outputTokens":6,"totalTokens":10}}"#.to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let events3 = futures::executor::block_on(conv.convert_event(ev3));
    assert!(!events3.is_empty());
    let out3 = events3.first().unwrap().as_ref().unwrap();
    match out3 {
        crate::stream::ChatStreamEvent::UsageUpdate { usage } => {
            assert_eq!(usage.prompt_tokens, 4);
            assert_eq!(usage.completion_tokens, 6);
            assert_eq!(usage.total_tokens, 10);
        }
        _ => panic!("expected UsageUpdate"),
    }

    // completed
    let ev4 = eventsource_stream::Event {
        event: "response.completed".to_string(),
        data: "{}".to_string(),
        id: "4".to_string(),
        retry: None,
    };
    let events4 = futures::executor::block_on(conv.convert_event(ev4));
    assert!(!events4.is_empty());
    let out4 = events4.first().unwrap().as_ref().unwrap();
    match out4 {
        crate::stream::ChatStreamEvent::StreamEnd { .. } => {}
        _ => panic!("expected StreamEnd"),
    }
}

#[test]
fn test_build_request_body_with_openai_specific_params() {
    use crate::params::{FunctionChoice, OpenAiParamsBuilder, ResponseFormat, ToolChoice};
    let mut config = create_test_config();
    // Common params
    config.common_params.max_tokens = Some(555); // should be overridden by max_completion_tokens below
    config.common_params.stop_sequences = Some(vec!["STOP1".to_string(), "STOP2".to_string()]);
    // OpenAI params
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("a".to_string(), "b".to_string());
    let openai_params = OpenAiParamsBuilder::new()
        .response_format(ResponseFormat::JsonSchema {
            schema: serde_json::json!({"type":"object"}),
        })
        .tool_choice(ToolChoice::Function {
            choice_type: "function".to_string(),
            function: FunctionChoice {
                name: "doit".to_string(),
            },
        })
        .parallel_tool_calls(true)
        .store(true)
        .metadata(metadata.clone())
        .user("user-1".to_string())
        .max_completion_tokens(321)
        .build()
        .unwrap();
    config.openai_params = openai_params;

    let client = super::OpenAiResponses::new(reqwest::Client::new(), config);
    let messages = vec![create_test_message()];
    let body = client
        .build_request_body(&messages, None, None, false, false)
        .unwrap();

    assert_eq!(body["model"], "gpt-5-mini");
    assert_eq!(body["stream"], false);
    assert_eq!(body["background"], false);
    assert_eq!(body["max_output_tokens"], 321);
    assert_eq!(body["stop"], serde_json::json!(["STOP1", "STOP2"]));
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(
        body["response_format"]["schema"].get("type").unwrap(),
        "object"
    );
    assert_eq!(body["tool_choice"]["type"], "function");
    assert_eq!(body["tool_choice"]["function"]["name"], "doit");
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(body["store"], true);
    assert_eq!(body["metadata"], serde_json::json!({"a":"b"}));
    assert_eq!(body["user"], "user-1");
}

#[test]
fn test_build_request_body_max_tokens_fallback() {
    // When openai_params.max_completion_tokens is not set, use common_params.max_tokens
    let mut config = create_test_config();
    config.common_params.max_tokens = Some(777);
    // Ensure openai_params has no max_completion_tokens override
    // (builder default None)
    let openai_params = crate::params::OpenAiParamsBuilder::new().build().unwrap();
    config.openai_params = openai_params;

    let client = super::OpenAiResponses::new(reqwest::Client::new(), config);
    let messages = vec![create_test_message()];
    let body = client
        .build_request_body(&messages, None, None, false, false)
        .unwrap();
    assert_eq!(body["max_output_tokens"], 777);
}
