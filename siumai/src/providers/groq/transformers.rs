//! Audio transformers for Groq (TTS/STT)
use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::transformers::audio::{AudioHttpBody, AudioTransformer};
use crate::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::types::{ChatRequest, ChatResponse, ContentPart, MessageContent, Usage};
use eventsource_stream::Event;
use std::future::Future;
use std::pin::Pin;

#[derive(Clone)]
pub struct GroqAudioTransformer;

impl AudioTransformer for GroqAudioTransformer {
    fn provider_id(&self) -> &str {
        "groq"
    }

    fn build_tts_body(&self, req: &crate::types::TtsRequest) -> Result<AudioHttpBody, LlmError> {
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| "playai-tts".to_string());
        let voice = req
            .voice
            .clone()
            .unwrap_or_else(|| "Fritz-PlayAI".to_string());
        let format = req.format.clone().unwrap_or_else(|| "wav".to_string());
        let speed = req.speed.unwrap_or(1.0);
        let json = serde_json::json!({
            "model": model,
            "input": req.text,
            "voice": voice,
            "response_format": format,
            "speed": speed
        });
        Ok(AudioHttpBody::Json(json))
    }

    fn build_stt_body(&self, req: &crate::types::SttRequest) -> Result<AudioHttpBody, LlmError> {
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| "whisper-large-v3".to_string());
        let audio = req
            .audio_data
            .clone()
            .ok_or_else(|| LlmError::InvalidInput("audio_data required for STT".to_string()))?;
        let form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(audio).file_name("audio.wav"),
            )
            .text("model", model);
        Ok(AudioHttpBody::Multipart(form))
    }

    fn tts_endpoint(&self) -> &str {
        "/audio/speech"
    }
    fn stt_endpoint(&self) -> &str {
        "/audio/transcriptions"
    }

    fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError> {
        let text = json
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::ParseError("missing 'text' field".to_string()))?;
        Ok(text.to_string())
    }
}

/// Request transformer for Groq Chat
#[derive(Clone)]
pub struct GroqRequestTransformer;

impl RequestTransformer for GroqRequestTransformer {
    fn provider_id(&self) -> &str {
        "groq"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }
        struct GroqChatHooks;
        impl crate::transformers::request::ProviderRequestHooks for GroqChatHooks {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                let mut body = serde_json::json!({ "model": req.common_params.model });
                if let Some(t) = req.common_params.temperature {
                    body["temperature"] = serde_json::json!(t);
                }
                if let Some(tp) = req.common_params.top_p {
                    body["top_p"] = serde_json::json!(tp);
                }
                if let Some(max) = req.common_params.max_tokens {
                    body["max_tokens"] = serde_json::json!(max);
                }
                if let Some(stops) = &req.common_params.stop_sequences {
                    body["stop"] = serde_json::json!(stops);
                }
                let messages = super::utils::convert_messages(&req.messages)?;
                body["messages"] = serde_json::to_value(messages)?;
                if let Some(tools) = &req.tools
                    && !tools.is_empty()
                {
                    body["tools"] = serde_json::to_value(tools)?;
                }
                body["stream"] = serde_json::json!(req.stream);
                Ok(body)
            }
            fn post_process_chat(
                &self,
                _req: &ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // All provider-specific features are now handled via provider_options
                // in ProviderSpec::chat_before_send()
                super::utils::validate_groq_params(body)?;
                Ok(())
            }
        }
        let hooks = GroqChatHooks;
        let profile = crate::transformers::request::MappingProfile {
            provider_id: "groq",
            rules: vec![
                crate::transformers::request::Rule::Range {
                    field: "temperature",
                    min: 0.0,
                    max: 2.0,
                    mode: crate::transformers::request::RangeMode::Error,
                    message: None,
                },
                crate::transformers::request::Rule::Range {
                    field: "top_p",
                    min: 0.0,
                    max: 1.0,
                    mode: crate::transformers::request::RangeMode::Error,
                    message: None,
                },
            ],
            merge_strategy: crate::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = crate::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_chat(req)
    }
}

/// Response transformer for Groq Chat
#[derive(Clone)]
pub struct GroqResponseTransformer;

impl ResponseTransformer for GroqResponseTransformer {
    fn provider_id(&self) -> &str {
        "groq"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        let response: super::types::GroqChatResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Groq response: {e}")))?;

        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ApiError {
                code: 500,
                message: "No choices in response".to_string(),
                details: None,
            })?;

        let content = if let Some(content) = choice.message.content {
            match content {
                serde_json::Value::String(text) => MessageContent::Text(text),
                serde_json::Value::Array(parts) => {
                    let mut content_parts = Vec::new();
                    for part in parts {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            content_parts.push(crate::types::ContentPart::Text {
                                text: text.to_string(),
                            });
                        }
                    }
                    MessageContent::MultiModal(content_parts)
                }
                _ => MessageContent::Text(String::new()),
            }
        } else {
            MessageContent::Text(String::new())
        };

        // Add tool calls to content if present
        let mut final_content = content;
        if let Some(calls) = choice.message.tool_calls {
            let mut parts = match final_content {
                MessageContent::Text(ref text) if !text.is_empty() => vec![ContentPart::text(text)],
                MessageContent::MultiModal(ref parts) => parts.clone(),
                _ => Vec::new(),
            };

            for call in calls {
                if let Some(function) = call.function {
                    // Parse arguments string to JSON Value
                    let arguments = serde_json::from_str(&function.arguments)
                        .unwrap_or_else(|_| serde_json::Value::String(function.arguments.clone()));

                    parts.push(ContentPart::tool_call(
                        call.id,
                        function.name,
                        arguments,
                        None,
                    ));
                }
            }

            final_content = if parts.len() == 1 && parts[0].is_text() {
                MessageContent::Text(parts[0].as_text().unwrap_or_default().to_string())
            } else {
                MessageContent::MultiModal(parts)
            };
        }

        let finish_reason = Some(super::utils::parse_finish_reason(
            choice.finish_reason.as_deref(),
        ));

        let usage = response.usage.map(|u| {
            Usage::builder()
                .prompt_tokens(u.prompt_tokens.unwrap_or(0))
                .completion_tokens(u.completion_tokens.unwrap_or(0))
                .total_tokens(u.total_tokens.unwrap_or(0))
                .build()
        });

        Ok(ChatResponse {
            id: Some(response.id),
            content: final_content,
            model: Some(response.model),
            usage,
            finish_reason,
            audio: None, // Groq doesn't support audio output
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Stream transformer wrapper for Groq
#[derive(Clone)]
pub struct GroqStreamChunkTransformer {
    pub provider_id: String,
    pub inner: super::streaming::GroqEventConverter,
}

impl StreamChunkTransformer for GroqStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<
        Box<
            dyn Future<Output = Vec<Result<crate::streaming::ChatStreamEvent, LlmError>>>
                + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }
    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end()
    }
}

// Tests for structured_output via provider_params have been removed
// as this functionality is now handled via provider_options in ProviderSpec::chat_before_send()
