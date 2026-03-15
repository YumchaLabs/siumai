//! OpenRouter provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["openrouter"]`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenRouter prompt transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenRouterTransform {
    /// OpenRouter middle-out transform.
    #[serde(rename = "middle-out")]
    MiddleOut,
}

/// OpenRouter-specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenRouterOptions {
    /// Request transforms applied by OpenRouter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transforms: Option<Vec<OpenRouterTransform>>,
    /// Additional OpenRouter-specific parameters.
    #[serde(flatten)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl OpenRouterOptions {
    /// Create new OpenRouter options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a single OpenRouter transform.
    pub fn with_transform(mut self, transform: OpenRouterTransform) -> Self {
        self.transforms.get_or_insert_with(Vec::new).push(transform);
        self
    }

    /// Replace the OpenRouter transforms list.
    pub fn with_transforms<I>(mut self, transforms: I) -> Self
    where
        I: IntoIterator<Item = OpenRouterTransform>,
    {
        let transforms = transforms.into_iter().collect::<Vec<_>>();
        self.transforms = if transforms.is_empty() {
            None
        } else {
            Some(transforms)
        };
        self
    }

    /// Add a custom OpenRouter parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_options_serialize_transforms_and_extra_params() {
        let value = serde_json::to_value(
            OpenRouterOptions::new()
                .with_transform(OpenRouterTransform::MiddleOut)
                .with_param("someVendorParam", serde_json::json!(true)),
        )
        .expect("options serialize");

        assert_eq!(value["transforms"], serde_json::json!(["middle-out"]));
        assert_eq!(value["someVendorParam"], serde_json::json!(true));
    }
}
