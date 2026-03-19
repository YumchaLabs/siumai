use crate::types::ChatRequest;

/// Azure request option helpers for `ChatRequest`.
pub trait AzureOpenAiChatRequestExt {
    /// Convenience: attach Azure-specific options to `provider_options_map["azure"]`.
    fn with_azure_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl AzureOpenAiChatRequestExt for ChatRequest {
    fn with_azure_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("azure", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::azure::{AzureOpenAiOptions, AzureReasoningEffort};
    use crate::types::ChatMessage;

    #[test]
    fn chat_request_ext_attaches_azure_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_azure_options(
            AzureOpenAiOptions::new()
                .with_force_reasoning(true)
                .with_reasoning_effort(AzureReasoningEffort::High),
        );

        let value = request
            .provider_options_map
            .get("azure")
            .expect("azure options present");
        assert_eq!(value["force_reasoning"], serde_json::json!(true));
        assert_eq!(value["reasoning_effort"], serde_json::json!("high"));
    }
}
