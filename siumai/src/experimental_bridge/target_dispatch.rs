//! Shared target-specific dispatch helpers for experimental bridges.

use futures_util::Stream;
use siumai_core::LlmError;
use siumai_core::bridge::BridgeTarget;
use siumai_core::encoding::JsonEncodeOptions;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex"
))]
use siumai_core::encoding::encode_chat_response_as_json;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex"
))]
use siumai_core::execution::transformers::request::RequestTransformer;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex"
))]
use siumai_core::streaming::encode_chat_stream_as_sse;
use siumai_core::streaming::{ChatByteStream, ChatStreamEvent};
use siumai_core::types::{ChatRequest, ChatResponse};

pub(crate) fn transform_chat_request_to_json(
    request: &ChatRequest,
    target: BridgeTarget,
) -> Result<serde_json::Value, LlmError> {
    #[cfg(not(any(
        feature = "openai",
        feature = "anthropic",
        feature = "google",
        feature = "google-vertex"
    )))]
    let _ = request;

    match target {
        BridgeTarget::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                let tx =
                    siumai_protocol_openai::standards::openai::transformers::request::OpenAiResponsesRequestTransformer;
                tx.transform_chat(request)
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                let tx =
                    siumai_protocol_openai::standards::openai::transformers::request::OpenAiRequestTransformer;
                tx.transform_chat(request)
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                let tx =
                    siumai_protocol_anthropic::standards::anthropic::transformers::AnthropicRequestTransformer::default();
                tx.transform_chat(request)
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "anthropic feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::GeminiGenerateContent => {
            #[cfg(any(feature = "google", feature = "google-vertex"))]
            {
                let tx =
                    siumai_protocol_gemini::standards::gemini::transformers::GeminiRequestTransformer {
                        config:
                            siumai_protocol_gemini::standards::gemini::types::GeminiConfig::default(),
                    };
                tx.transform_chat(request)
            }
            #[cfg(not(any(feature = "google", feature = "google-vertex")))]
            {
                Err(LlmError::UnsupportedOperation(
                    "google/google-vertex feature is disabled".to_string(),
                ))
            }
        }
    }
}

pub(crate) fn encode_chat_response_to_json_bytes(
    response: &ChatResponse,
    target: BridgeTarget,
    opts: JsonEncodeOptions,
) -> Result<Vec<u8>, LlmError> {
    #[cfg(not(any(
        feature = "openai",
        feature = "anthropic",
        feature = "google",
        feature = "google-vertex"
    )))]
    let _ = (response, opts);

    match target {
        BridgeTarget::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                encode_chat_response_as_json(
                    response,
                    siumai_protocol_openai::standards::openai::json_response::OpenAiResponsesJsonResponseConverter::new(),
                    opts,
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                encode_chat_response_as_json(
                    response,
                    siumai_protocol_openai::standards::openai::json_response::OpenAiChatCompletionsJsonResponseConverter::new(),
                    opts,
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                encode_chat_response_as_json(
                    response,
                    siumai_protocol_anthropic::standards::anthropic::json_response::AnthropicMessagesJsonResponseConverter::new(),
                    opts,
                )
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "anthropic feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::GeminiGenerateContent => {
            #[cfg(any(feature = "google", feature = "google-vertex"))]
            {
                encode_chat_response_as_json(
                    response,
                    siumai_protocol_gemini::standards::gemini::json_response::GeminiGenerateContentJsonResponseConverter::new(),
                    opts,
                )
            }
            #[cfg(not(any(feature = "google", feature = "google-vertex")))]
            {
                Err(LlmError::UnsupportedOperation(
                    "google/google-vertex feature is disabled".to_string(),
                ))
            }
        }
    }
}

pub(crate) fn encode_chat_stream_for_target<S>(
    stream: S,
    target: BridgeTarget,
) -> Result<ChatByteStream, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    #[cfg(not(any(
        feature = "openai",
        feature = "anthropic",
        feature = "google",
        feature = "google-vertex"
    )))]
    let _ = stream;

    match target {
        BridgeTarget::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                Ok(encode_chat_stream_as_sse(
                    stream,
                    siumai_protocol_openai::standards::openai::responses_sse::OpenAiResponsesEventConverter::new(),
                ))
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                let adapter = std::sync::Arc::new(
                    siumai_core::standards::openai::compat::adapter::OpenAiStandardAdapter {
                        base_url: String::new(),
                    },
                );
                let cfg = siumai_core::standards::openai::compat::openai_config::OpenAiCompatibleConfig::new(
                    "openai",
                    "",
                    "",
                    adapter.clone(),
                );
                Ok(encode_chat_stream_as_sse(
                    stream,
                    siumai_core::standards::openai::compat::streaming::OpenAiCompatibleEventConverter::new(
                        cfg,
                        adapter,
                    ),
                ))
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                Ok(encode_chat_stream_as_sse(
                    stream,
                    siumai_protocol_anthropic::standards::anthropic::streaming::AnthropicEventConverter::new(
                        siumai_protocol_anthropic::standards::anthropic::params::AnthropicParams::default(),
                    ),
                ))
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "anthropic feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::GeminiGenerateContent => {
            #[cfg(any(feature = "google", feature = "google-vertex"))]
            {
                Ok(encode_chat_stream_as_sse(
                    stream,
                    siumai_protocol_gemini::standards::gemini::streaming::GeminiEventConverter::new(
                        siumai_protocol_gemini::standards::gemini::types::GeminiConfig::default(),
                    ),
                ))
            }
            #[cfg(not(any(feature = "google", feature = "google-vertex")))]
            {
                Err(LlmError::UnsupportedOperation(
                    "google/google-vertex feature is disabled".to_string(),
                ))
            }
        }
    }
}
