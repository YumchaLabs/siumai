//! DeepSeek warning parity middleware.

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Warning};
use siumai_provider_openai_compatible::providers::openai_compatible::{
    RequestBodyTransformer, RequestType,
};

#[derive(Debug, Default)]
pub(crate) struct DeepSeekRequestBodyTransformer;

impl DeepSeekRequestBodyTransformer {
    pub const fn new() -> Self {
        Self
    }

    fn response_format_prompt(response_format: &serde_json::Value) -> Option<String> {
        let obj = response_format.as_object()?;
        match obj.get("type").and_then(|value| value.as_str()) {
            Some("json_object") => Some("Return JSON.".to_string()),
            Some("json_schema") => {
                let schema = obj
                    .get("json_schema")
                    .and_then(|value| value.as_object())
                    .and_then(|json_schema| json_schema.get("schema"))?;
                let schema = serde_json::to_string(schema).unwrap_or_else(|_| schema.to_string());
                Some(format!(
                    "Return JSON that conforms to the following schema: {schema}"
                ))
            }
            _ => None,
        }
    }

    fn inject_prompt(body: &mut serde_json::Value, prompt: String) {
        let Some(messages) = body
            .get_mut("messages")
            .and_then(|value| value.as_array_mut())
        else {
            return;
        };

        messages.insert(
            0,
            serde_json::json!({
                "role": "system",
                "content": prompt,
            }),
        );
    }
}

impl RequestBodyTransformer for DeepSeekRequestBodyTransformer {
    fn transform_request_body(
        &self,
        body: &mut serde_json::Value,
        _model: &str,
        request_type: RequestType,
    ) -> Result<(), LlmError> {
        if request_type != RequestType::Chat {
            return Ok(());
        }

        let Some(response_format) = body.get("response_format").cloned() else {
            return Ok(());
        };
        let prompt = Self::response_format_prompt(&response_format);

        if let Some(body_obj) = body.as_object_mut() {
            body_obj.insert(
                "response_format".to_string(),
                serde_json::json!({ "type": "json_object" }),
            );
        }

        if let Some(prompt) = prompt {
            Self::inject_prompt(body, prompt);
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
pub(crate) struct DeepSeekWarningsMiddleware;

impl DeepSeekWarningsMiddleware {
    pub const fn new() -> Self {
        Self
    }

    fn compute_warnings(req: &ChatRequest) -> Vec<Warning> {
        if matches!(
            req.response_format,
            Some(crate::types::chat::ResponseFormat::Json { .. })
        ) {
            vec![Warning::compatibility(
                "responseFormat JSON schema",
                Some("JSON response schema is injected into the system message."),
            )]
        } else {
            Vec::new()
        }
    }

    fn merge_warnings(mut resp: ChatResponse, additional: Vec<Warning>) -> ChatResponse {
        if additional.is_empty() {
            return resp;
        }

        match resp.warnings.as_mut() {
            Some(existing) => existing.extend(additional),
            None => resp.warnings = Some(additional),
        }

        resp
    }
}

impl LanguageModelMiddleware for DeepSeekWarningsMiddleware {
    fn post_generate(
        &self,
        req: &ChatRequest,
        resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        Ok(Self::merge_warnings(resp, Self::compute_warnings(req)))
    }

    fn on_stream_event(
        &self,
        req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        match ev {
            ChatStreamEvent::StreamEnd { response } => {
                let response = Self::merge_warnings(response, Self::compute_warnings(req));
                Ok(vec![ChatStreamEvent::StreamEnd { response }])
            }
            other => Ok(vec![other]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, MessageContent};

    fn dummy_resp() -> ChatResponse {
        ChatResponse::new(MessageContent::Text("ok".to_string()))
    }

    #[test]
    fn warns_when_json_schema_is_injected_into_system_message() {
        let req = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({ "type": "object" }),
            ))
            .build();

        let out = DeepSeekWarningsMiddleware::new()
            .post_generate(&req, dummy_resp())
            .expect("post generate");

        assert_eq!(
            out.warnings,
            Some(vec![Warning::compatibility(
                "responseFormat JSON schema",
                Some("JSON response schema is injected into the system message."),
            )])
        );
    }

    #[test]
    fn does_not_warn_for_schema_less_json_response_format() {
        let req = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(crate::types::chat::ResponseFormat::json_object())
            .build();

        let out = DeepSeekWarningsMiddleware::new()
            .post_generate(&req, dummy_resp())
            .expect("post generate");

        assert_eq!(out.warnings, None);
    }

    #[test]
    fn request_transformer_maps_json_schema_to_json_object_and_injects_prompt() {
        let mut body = serde_json::json!({
            "model": "deepseek-chat",
            "messages": [{ "role": "user", "content": "hi" }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": { "answer": { "type": "string" } }
                    },
                    "strict": true
                }
            }
        });

        DeepSeekRequestBodyTransformer::new()
            .transform_request_body(&mut body, "deepseek-chat", RequestType::Chat)
            .expect("transform");

        assert_eq!(
            body["response_format"],
            serde_json::json!({ "type": "json_object" })
        );
        assert_eq!(
            body["messages"][0],
            serde_json::json!({
                "role": "system",
                "content": "Return JSON that conforms to the following schema: {\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}"
            })
        );
    }
}
