use crate::types::ChatRequest;

fn merge_provider_option_object(mut request: ChatRequest, value: serde_json::Value) -> ChatRequest {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = request
            .provider_options_map
            .get("xai")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        request
            .provider_options_map
            .insert("xai", serde_json::Value::Object(merged));
        request
    } else {
        request.with_provider_option("xai", value)
    }
}

/// xAI request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait XaiChatRequestExt {
    /// Convenience: attach xAI-specific options to `provider_options_map["xai"]`.
    fn with_xai_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl XaiChatRequestExt for ChatRequest {
    fn with_xai_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        merge_provider_option_object(self, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::{SearchMode, XaiOptions, XaiSearchParameters};
    use crate::types::ChatMessage;

    #[test]
    fn chat_request_ext_merges_existing_xai_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option("xai", serde_json::json!({ "existing": true }))
            .with_xai_options(
                XaiOptions::new()
                    .with_reasoning_effort("high")
                    .with_parallel_function_calling(false)
                    .with_search(XaiSearchParameters {
                        mode: SearchMode::On,
                        return_citations: Some(true),
                        max_search_results: Some(3),
                        from_date: None,
                        to_date: None,
                        sources: None,
                    }),
            );

        let value = request
            .provider_options_map
            .get("xai")
            .expect("xai options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["reasoningEffort"], serde_json::json!("high"));
        assert_eq!(value["parallel_function_calling"], serde_json::json!(false));
        assert_eq!(value["searchParameters"]["mode"], serde_json::json!("on"));
    }
}
