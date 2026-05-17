//! Shared completion request shaping helpers for the OpenAI protocol family.

use crate::LlmError;
use crate::types::{
    ChatMessage, CompletionRequest, ContentPart, MessageContent, MessageRole, Warning,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionPromptMaterialization {
    pub prompt: String,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CompletionBodyOptions {
    pub stream: bool,
    pub include_usage: bool,
    pub deprecated_openai_compatible_key_warning: Option<&'static str>,
    pub provider_options: serde_json::Map<String, serde_json::Value>,
}

impl CompletionBodyOptions {
    pub fn new(stream: bool) -> Self {
        Self {
            stream,
            include_usage: false,
            deprecated_openai_compatible_key_warning: None,
            provider_options: serde_json::Map::new(),
        }
    }

    pub fn with_include_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = include_usage;
        self
    }

    pub fn with_deprecated_openai_compatible_key_warning(
        mut self,
        warning: Option<&'static str>,
    ) -> Self {
        self.deprecated_openai_compatible_key_warning = warning;
        self
    }

    pub fn with_provider_options(
        mut self,
        provider_options: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        self.provider_options = provider_options;
        self
    }
}

#[allow(unreachable_patterns)]
pub fn completion_message_text(
    content: &MessageContent,
    role_name: &str,
) -> Result<String, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::MultiModal(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    ContentPart::Text {
                        text: part_text, ..
                    } => text.push_str(part_text),
                    ContentPart::ToolCall { .. } => {
                        return Err(LlmError::UnsupportedOperation(format!(
                            "Completion prompts do not support tool-call parts in {role_name} messages"
                        )));
                    }
                    ContentPart::ToolResult { .. } => {
                        return Err(LlmError::UnsupportedOperation(
                            "Completion prompts do not support tool messages".to_string(),
                        ));
                    }
                    _ => {
                        return Err(LlmError::UnsupportedOperation(format!(
                            "Completion prompts only support text content in {role_name} messages"
                        )));
                    }
                }
            }
            Ok(text)
        }
        _ => Err(LlmError::UnsupportedOperation(format!(
            "Completion prompts do not support structured JSON content in {role_name} messages"
        ))),
    }
}

pub fn materialize_completion_prompt(
    prompt: &[ChatMessage],
) -> Result<CompletionPromptMaterialization, LlmError> {
    if prompt.is_empty() {
        return Err(LlmError::InvalidParameter(
            "Completion prompt cannot be empty".to_string(),
        ));
    }

    let mut text = String::new();
    let mut remaining = prompt;

    if let Some(first) = prompt.first()
        && first.role == MessageRole::System
    {
        text.push_str(&completion_message_text(&first.content, "system")?);
        text.push_str("\n\n");
        remaining = &prompt[1..];
    }

    for message in remaining {
        match message.role {
            MessageRole::System => {
                return Err(LlmError::InvalidParameter(
                    "Unexpected system message in completion prompt".to_string(),
                ));
            }
            MessageRole::Developer => {
                return Err(LlmError::UnsupportedOperation(
                    "Completion prompts do not support developer messages".to_string(),
                ));
            }
            MessageRole::Tool => {
                return Err(LlmError::UnsupportedOperation(
                    "Completion prompts do not support tool messages".to_string(),
                ));
            }
            MessageRole::User => {
                text.push_str("user:\n");
                text.push_str(&completion_message_text(&message.content, "user")?);
                text.push_str("\n\n");
            }
            MessageRole::Assistant => {
                text.push_str("assistant:\n");
                text.push_str(&completion_message_text(&message.content, "assistant")?);
                text.push_str("\n\n");
            }
        }
    }

    text.push_str("assistant:\n");

    Ok(CompletionPromptMaterialization {
        prompt: text,
        stop_sequences: vec!["\nuser:".to_string()],
    })
}

pub fn completion_warnings(
    request: &CompletionRequest,
    deprecated_openai_compatible_key_warning: Option<&'static str>,
) -> Vec<Warning> {
    let mut warnings = Vec::new();

    if let Some(warning) = deprecated_openai_compatible_key_warning
        && request
            .provider_options_map
            .get("openai-compatible")
            .is_some()
    {
        warnings.push(Warning::other(warning));
    }
    if request.common_params.top_k.is_some() {
        warnings.push(Warning::unsupported("topK", None::<String>));
    }
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        warnings.push(Warning::unsupported("tools", None::<String>));
    }
    if request.tool_choice.is_some() {
        warnings.push(Warning::unsupported("toolChoice", None::<String>));
    }
    if request.response_format.is_some() {
        warnings.push(Warning::unsupported(
            "responseFormat",
            Some("JSON response format is not supported."),
        ));
    }

    warnings
}

pub fn build_completion_body(
    request: &CompletionRequest,
    options: CompletionBodyOptions,
) -> Result<(serde_json::Value, Vec<Warning>), LlmError> {
    let prompt = materialize_completion_prompt(&request.prompt)?;
    let warnings = completion_warnings(request, options.deprecated_openai_compatible_key_warning);

    let stop = prompt
        .stop_sequences
        .into_iter()
        .chain(
            request
                .common_params
                .stop_sequences
                .clone()
                .unwrap_or_default(),
        )
        .collect::<Vec<_>>();

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::Value::String(request.common_params.model.clone()),
    );

    if let Some(max_tokens) = request
        .common_params
        .max_completion_tokens
        .or(request.common_params.max_tokens)
    {
        body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
    }
    if let Some(temperature) = request.common_params.temperature {
        body.insert("temperature".to_string(), serde_json::json!(temperature));
    }
    if let Some(top_p) = request.common_params.top_p {
        body.insert("top_p".to_string(), serde_json::json!(top_p));
    }
    if let Some(frequency_penalty) = request.common_params.frequency_penalty {
        body.insert(
            "frequency_penalty".to_string(),
            serde_json::json!(frequency_penalty),
        );
    }
    if let Some(presence_penalty) = request.common_params.presence_penalty {
        body.insert(
            "presence_penalty".to_string(),
            serde_json::json!(presence_penalty),
        );
    }
    if let Some(seed) = request.common_params.seed {
        body.insert("seed".to_string(), serde_json::json!(seed));
    }

    body.extend(options.provider_options);
    body.insert(
        "prompt".to_string(),
        serde_json::Value::String(prompt.prompt),
    );
    if !stop.is_empty() {
        body.insert("stop".to_string(), serde_json::json!(stop));
    }
    if options.stream {
        body.insert("stream".to_string(), serde_json::json!(true));
        if options.include_usage {
            body.insert(
                "stream_options".to_string(),
                serde_json::json!({ "include_usage": true }),
            );
        }
    }

    Ok((serde_json::Value::Object(body), warnings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CommonParams, ResponseFormat};

    #[test]
    fn materialize_completion_prompt_matches_ai_sdk_completion_rules() {
        let prompt = vec![
            ChatMessage::system("System prelude").build(),
            ChatMessage::user("Hello").build(),
            ChatMessage::assistant("Hi").build(),
            ChatMessage::user("Next").build(),
        ];

        let materialized = materialize_completion_prompt(&prompt).expect("prompt");

        assert_eq!(
            materialized.prompt,
            "System prelude\n\nuser:\nHello\n\nassistant:\nHi\n\nuser:\nNext\n\nassistant:\n"
        );
        assert_eq!(materialized.stop_sequences, vec!["\nuser:"]);
    }

    #[test]
    fn materialize_completion_prompt_rejects_unsupported_roles() {
        let err = materialize_completion_prompt(&[
            ChatMessage::user("Hello").build(),
            ChatMessage::system("late system").build(),
        ])
        .expect_err("late system should fail");

        assert!(err.to_string().contains("Unexpected system message"));
    }

    #[test]
    fn build_completion_body_maps_shared_params_and_stream_usage() {
        let mut request = CompletionRequest::new("Hello");
        request.common_params = CommonParams {
            model: "text-model".to_string(),
            temperature: Some(0.3),
            max_tokens: Some(7),
            max_completion_tokens: Some(11),
            top_p: Some(0.9),
            top_k: Some(2.0),
            stop_sequences: Some(vec!["END".to_string()]),
            seed: Some(42),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.2),
        };
        request.response_format = Some(ResponseFormat::json_object());
        let mut provider_options = serde_json::Map::new();
        provider_options.insert("logprobs".to_string(), serde_json::json!(2));

        let (body, warnings) = build_completion_body(
            &request,
            CompletionBodyOptions::new(true)
                .with_include_usage(true)
                .with_provider_options(provider_options),
        )
        .expect("body");

        assert_eq!(body["model"], serde_json::json!("text-model"));
        assert_eq!(
            body["prompt"],
            serde_json::json!("user:\nHello\n\nassistant:\n")
        );
        assert_eq!(body["max_tokens"], serde_json::json!(11));
        assert_eq!(body["temperature"], serde_json::json!(0.3));
        assert_eq!(body["top_p"], serde_json::json!(0.9));
        assert_eq!(body["frequency_penalty"], serde_json::json!(0.1));
        assert_eq!(body["presence_penalty"], serde_json::json!(0.2));
        assert_eq!(body["seed"], serde_json::json!(42));
        assert_eq!(body["stop"], serde_json::json!(["\nuser:", "END"]));
        assert_eq!(body["logprobs"], serde_json::json!(2));
        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(
            body["stream_options"],
            serde_json::json!({ "include_usage": true })
        );
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn build_completion_body_emits_deprecated_openai_compatible_warning() {
        let request = CompletionRequest::new("Hello")
            .with_model("text-model")
            .with_provider_option("openai-compatible", serde_json::json!({ "user": "legacy" }));

        let (_, warnings) = build_completion_body(
            &request,
            CompletionBodyOptions::new(false)
                .with_deprecated_openai_compatible_key_warning(Some("deprecated compat key")),
        )
        .expect("body");

        assert_eq!(warnings.len(), 1);
        assert!(format!("{:?}", warnings[0]).contains("deprecated compat key"));
    }
}
