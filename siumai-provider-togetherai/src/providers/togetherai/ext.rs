use crate::provider_options::{TogetherAiImageOptions, TogetherAiRerankOptions};

fn merge_provider_option_object<T>(
    map: &mut crate::types::ProviderOptionsMap,
    value: T,
    error_label: &str,
) where
    T: serde::Serialize,
{
    let value = serde_json::to_value(value).expect(error_label);
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = map
            .get("togetherai")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        map.insert("togetherai", serde_json::Value::Object(merged));
    } else {
        map.insert("togetherai", value);
    }
}

/// Typed rerank request helpers for TogetherAI.
pub trait TogetherAiRerankRequestExt {
    /// Store typed options under `provider_options_map["togetherai"]`.
    fn with_togetherai_options(self, options: TogetherAiRerankOptions) -> Self;
}

impl TogetherAiRerankRequestExt for crate::types::RerankRequest {
    fn with_togetherai_options(mut self, options: TogetherAiRerankOptions) -> Self {
        merge_provider_option_object(
            &mut self.provider_options_map,
            options,
            "serialize TogetherAiRerankOptions",
        );
        self
    }
}

/// Typed image request helpers for TogetherAI.
pub trait TogetherAiImageRequestExt {
    /// Store typed image options under `provider_options_map["togetherai"]`.
    fn with_togetherai_image_options(self, options: TogetherAiImageOptions) -> Self;
}

impl TogetherAiImageRequestExt for crate::types::ImageGenerationRequest {
    fn with_togetherai_image_options(mut self, options: TogetherAiImageOptions) -> Self {
        merge_provider_option_object(
            &mut self.provider_options_map,
            options,
            "serialize TogetherAiImageOptions",
        );
        self
    }
}

impl TogetherAiImageRequestExt for crate::types::ImageEditRequest {
    fn with_togetherai_image_options(mut self, options: TogetherAiImageOptions) -> Self {
        merge_provider_option_object(
            &mut self.provider_options_map,
            options,
            "serialize TogetherAiImageOptions",
        );
        self
    }
}

impl TogetherAiImageRequestExt for crate::types::ImageVariationRequest {
    fn with_togetherai_image_options(mut self, options: TogetherAiImageOptions) -> Self {
        merge_provider_option_object(
            &mut self.provider_options_map,
            options,
            "serialize TogetherAiImageOptions",
        );
        self
    }
}

impl TogetherAiImageRequestExt for crate::types::GenerateImageRequest {
    fn with_togetherai_image_options(mut self, options: TogetherAiImageOptions) -> Self {
        merge_provider_option_object(
            &mut self.provider_options_map,
            options,
            "serialize TogetherAiImageOptions",
        );
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn togetherai_image_request_ext_merges_image_options() {
        let request = crate::types::ImageGenerationRequest::default()
            .with_provider_option(
                "togetherai",
                serde_json::json!({
                    "existing": true,
                    "steps": 4
                }),
            )
            .with_togetherai_image_options(
                TogetherAiImageOptions::new()
                    .with_steps(28)
                    .with_guidance(3.0),
            );

        let value = request
            .provider_options_map
            .get("togetherai")
            .expect("togetherai options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["steps"], serde_json::json!(28));
        assert_eq!(value["guidance"], serde_json::json!(3.0));
    }

    #[test]
    fn togetherai_generate_image_request_ext_merges_image_options() {
        let request = crate::types::GenerateImageRequest::new("draw a robot")
            .with_provider_option(
                "togetherai",
                serde_json::json!({
                    "existing": true,
                    "disable_safety_checker": false
                }),
            )
            .with_togetherai_image_options(
                TogetherAiImageOptions::new()
                    .with_steps(28)
                    .with_disable_safety_checker(true),
            );

        let value = request
            .provider_options_map
            .get("togetherai")
            .expect("togetherai options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["steps"], serde_json::json!(28));
        assert_eq!(value["disable_safety_checker"], serde_json::json!(true));
    }
}
