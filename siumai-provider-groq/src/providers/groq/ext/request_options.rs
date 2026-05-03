use crate::types::ChatRequest;

fn merge_provider_option_object(mut request: ChatRequest, value: serde_json::Value) -> ChatRequest {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = request
            .provider_options_map
            .get("groq")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        request
            .provider_options_map
            .insert("groq", serde_json::Value::Object(merged));
        request
    } else {
        request.with_provider_option("groq", value)
    }
}

/// Groq request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait GroqChatRequestExt {
    /// Convenience: attach Groq-specific options to `provider_options_map["groq"]`.
    fn with_groq_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl GroqChatRequestExt for ChatRequest {
    fn with_groq_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        merge_provider_option_object(self, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::{
        GroqOptions, GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier,
    };
    use crate::types::ChatMessage;

    #[test]
    fn chat_request_ext_merges_existing_groq_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option("groq", serde_json::json!({ "existing": true }))
            .with_groq_options(
                GroqOptions::new()
                    .with_service_tier(GroqServiceTier::Performance)
                    .with_reasoning_effort(GroqReasoningEffort::High)
                    .with_reasoning_format(GroqReasoningFormat::Parsed)
                    .with_parallel_tool_calls(false)
                    .with_user("groq-user-1"),
            );

        let value = request
            .provider_options_map
            .get("groq")
            .expect("groq options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["serviceTier"], serde_json::json!("performance"));
        assert_eq!(value["reasoningEffort"], serde_json::json!("high"));
        assert_eq!(value["reasoningFormat"], serde_json::json!("parsed"));
        assert_eq!(value["parallelToolCalls"], serde_json::json!(false));
        assert_eq!(value["user"], serde_json::json!("groq-user-1"));
    }
}
