//! Non-streaming JSON gateway helpers (Axum).
//!
//! English-only comments in code as requested.

use axum::{body::Body, http::header, response::Response};

use siumai::prelude::unified::{ChatResponse, LlmError};

/// Target JSON wire format for non-streaming response transcoding helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetJsonFormat {
    /// OpenAI Responses API JSON response.
    OpenAiResponses,
    /// OpenAI Chat Completions JSON response.
    OpenAiChatCompletions,
    /// Anthropic Messages API JSON response.
    AnthropicMessages,
    /// Gemini/Vertex GenerateContent JSON response.
    GeminiGenerateContent,
}

/// Options for transcoding a `ChatResponse` into a provider JSON response body.
#[derive(Debug, Clone)]
pub struct TranscodeJsonOptions {
    /// Whether to pretty-print the JSON output.
    pub pretty: bool,
}

impl Default for TranscodeJsonOptions {
    fn default() -> Self {
        Self { pretty: false }
    }
}

/// Convert a unified `ChatResponse` into a provider-native JSON response body (best-effort).
///
/// Note: This helper parses the encoded bytes back into `serde_json::Value` and is intended
/// mainly for debugging or JSON post-processing.
pub fn transcode_chat_response_to_json(
    response: &ChatResponse,
    target: TargetJsonFormat,
) -> serde_json::Value {
    use siumai::experimental::encoding::JsonEncodeOptions;

    let bytes: Result<Vec<u8>, LlmError> = match target {
        TargetJsonFormat::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                use siumai::protocol::openai::json_response::OpenAiResponsesJsonResponseConverter;
                siumai::experimental::encoding::encode_chat_response_as_json(
                    response,
                    OpenAiResponsesJsonResponseConverter::new(),
                    JsonEncodeOptions::default(),
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        TargetJsonFormat::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                use siumai::protocol::openai::json_response::OpenAiChatCompletionsJsonResponseConverter;
                siumai::experimental::encoding::encode_chat_response_as_json(
                    response,
                    OpenAiChatCompletionsJsonResponseConverter::new(),
                    JsonEncodeOptions::default(),
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        TargetJsonFormat::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                use siumai::protocol::anthropic::json_response::AnthropicMessagesJsonResponseConverter;
                siumai::experimental::encoding::encode_chat_response_as_json(
                    response,
                    AnthropicMessagesJsonResponseConverter::new(),
                    JsonEncodeOptions::default(),
                )
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "anthropic feature is disabled".to_string(),
                ))
            }
        }
        TargetJsonFormat::GeminiGenerateContent => {
            #[cfg(feature = "google")]
            {
                use siumai::protocol::gemini::json_response::GeminiGenerateContentJsonResponseConverter;
                siumai::experimental::encoding::encode_chat_response_as_json(
                    response,
                    GeminiGenerateContentJsonResponseConverter::new(),
                    JsonEncodeOptions::default(),
                )
            }
            #[cfg(not(feature = "google"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "google feature is disabled".to_string(),
                ))
            }
        }
    };

    match bytes {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| serde_json::json!({ "error": "failed to parse encoded json" })),
        Err(e) => serde_json::json!({ "error": e.user_message() }),
    }
}

/// Convert a unified `ChatResponse` into an Axum JSON response for the selected provider format.
pub fn to_transcoded_json_response(
    response: ChatResponse,
    target: TargetJsonFormat,
    opts: TranscodeJsonOptions,
) -> Response<Body> {
    use siumai::experimental::encoding::JsonEncodeOptions;

    let json_opts = JsonEncodeOptions {
        pretty: opts.pretty,
    };
    let bytes: Result<Vec<u8>, LlmError> = match target {
        TargetJsonFormat::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                use siumai::protocol::openai::json_response::OpenAiResponsesJsonResponseConverter;
                siumai::experimental::encoding::encode_chat_response_as_json(
                    &response,
                    OpenAiResponsesJsonResponseConverter::new(),
                    json_opts,
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        TargetJsonFormat::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                use siumai::protocol::openai::json_response::OpenAiChatCompletionsJsonResponseConverter;
                siumai::experimental::encoding::encode_chat_response_as_json(
                    &response,
                    OpenAiChatCompletionsJsonResponseConverter::new(),
                    json_opts,
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        TargetJsonFormat::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                use siumai::protocol::anthropic::json_response::AnthropicMessagesJsonResponseConverter;
                siumai::experimental::encoding::encode_chat_response_as_json(
                    &response,
                    AnthropicMessagesJsonResponseConverter::new(),
                    json_opts,
                )
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "anthropic feature is disabled".to_string(),
                ))
            }
        }
        TargetJsonFormat::GeminiGenerateContent => {
            #[cfg(feature = "google")]
            {
                use siumai::protocol::gemini::json_response::GeminiGenerateContentJsonResponseConverter;
                siumai::experimental::encoding::encode_chat_response_as_json(
                    &response,
                    GeminiGenerateContentJsonResponseConverter::new(),
                    json_opts,
                )
            }
            #[cfg(not(feature = "google"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "google feature is disabled".to_string(),
                ))
            }
        }
    };

    match bytes {
        Ok(bytes) => {
            let mut resp = Response::new(Body::from(bytes));
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
            resp
        }
        Err(e) => Response::builder()
            .status(501)
            .header("content-type", "text/plain")
            .body(Body::from(e.user_message()))
            .unwrap_or_else(|_| Response::new(Body::from("internal error"))),
    }
}

/// Convert a unified `ChatResponse` into an Axum JSON response for the selected provider format,
/// with a caller-provided post-processing hook.
pub fn to_transcoded_json_response_with_transform<F>(
    response: ChatResponse,
    target: TargetJsonFormat,
    opts: TranscodeJsonOptions,
    transform: F,
) -> Response<Body>
where
    F: FnOnce(serde_json::Value) -> serde_json::Value + Send + Sync + 'static,
{
    let json = transcode_chat_response_to_json(&response, target);
    let json = transform(json);
    let body = if opts.pretty {
        serde_json::to_vec_pretty(&json)
    } else {
        serde_json::to_vec(&json)
    };

    match body {
        Ok(bytes) => {
            let mut resp = Response::new(Body::from(bytes));
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
            resp
        }
        Err(_) => Response::builder()
            .status(500)
            .header("content-type", "text/plain")
            .body(Body::from("failed to serialize response"))
            .unwrap_or_else(|_| Response::new(Body::from("internal error"))),
    }
}

/// Convert a unified `ChatResponse` into an Axum JSON response for the selected provider format,
/// with a caller-provided response transform hook (more efficient than JSON post-processing).
pub fn to_transcoded_json_response_with_response_transform<F>(
    mut response: ChatResponse,
    target: TargetJsonFormat,
    opts: TranscodeJsonOptions,
    transform: F,
) -> Response<Body>
where
    F: FnOnce(&mut ChatResponse) + Send + Sync + 'static,
{
    transform(&mut response);
    to_transcoded_json_response(response, target, opts)
}

#[cfg(test)]
mod json_transcode_tests {
    use super::*;
    use serde_json::json;
    use siumai::prelude::unified::{ContentPart, MessageContent};

    #[test]
    #[cfg(feature = "openai")]
    fn openai_chat_completions_json_includes_tool_calls() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "get_weather", json!({"city":"GZ"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::OpenAiChatCompletions);
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
    }

    #[test]
    #[cfg(feature = "anthropic")]
    fn anthropic_messages_json_emits_tool_use_blocks() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("thinking"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::AnthropicMessages);
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"][1]["type"], "tool_use");
        assert_eq!(v["content"][1]["id"], "call_1");
        assert_eq!(v["content"][1]["name"], "search");
    }

    #[test]
    #[cfg(feature = "google")]
    fn gemini_generate_content_json_emits_function_call_parts() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::GeminiGenerateContent);
        assert_eq!(v["candidates"][0]["content"]["role"], "model");
        assert_eq!(
            v["candidates"][0]["content"]["parts"][1]["functionCall"]["name"],
            "search"
        );
    }

    #[test]
    #[cfg(feature = "openai")]
    fn openai_responses_json_includes_output_items() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::OpenAiResponses);
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][1]["type"], "function_call");
        assert_eq!(v["output"][1]["call_id"], "call_1");
    }
}
