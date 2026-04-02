//! `MiniMaxi` video provider options.
//!
//! These typed option structs are owned by the MiniMaxi provider crate and are serialized into
//! `providerOptions["minimaxi"]`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MiniMaxi-specific video-generation options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinimaxiVideoOptions {
    /// Whether to automatically optimize the prompt.
    #[serde(skip_serializing_if = "Option::is_none", alias = "promptOptimizer")]
    pub prompt_optimizer: Option<bool>,
    /// Whether to use faster prompt pretreatment.
    #[serde(skip_serializing_if = "Option::is_none", alias = "fastPretreatment")]
    pub fast_pretreatment: Option<bool>,
    /// Callback URL for task status updates.
    #[serde(skip_serializing_if = "Option::is_none", alias = "callbackUrl")]
    pub callback_url: Option<String>,
    /// Whether to add an AIGC watermark to the generated video.
    #[serde(skip_serializing_if = "Option::is_none", alias = "aigcWatermark")]
    pub aigc_watermark: Option<bool>,
    /// Forward-compatible provider-owned escape hatch for newly introduced options.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl MinimaxiVideoOptions {
    /// Create new MiniMaxi video options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set prompt optimization.
    pub fn with_prompt_optimizer(mut self, enabled: bool) -> Self {
        self.prompt_optimizer = Some(enabled);
        self
    }

    /// Set fast pretreatment.
    pub fn with_fast_pretreatment(mut self, enabled: bool) -> Self {
        self.fast_pretreatment = Some(enabled);
        self
    }

    /// Set callback URL.
    pub fn with_callback_url(mut self, url: impl Into<String>) -> Self {
        self.callback_url = Some(url.into());
        self
    }

    /// Set watermark preference.
    pub fn with_aigc_watermark(mut self, enabled: bool) -> Self {
        self.aigc_watermark = Some(enabled);
        self
    }

    /// Backwards-friendly alias for watermark preference.
    pub fn with_watermark(self, enabled: bool) -> Self {
        self.with_aigc_watermark(enabled)
    }

    /// Add a custom MiniMaxi parameter.
    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }

    /// Return true when no typed or extra option is set.
    pub fn is_empty(&self) -> bool {
        self.prompt_optimizer.is_none()
            && self.fast_pretreatment.is_none()
            && self.callback_url.is_none()
            && self.aigc_watermark.is_none()
            && self.extra_fields.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimaxi_video_options_serialize_known_fields() {
        let value = serde_json::to_value(
            MinimaxiVideoOptions::new()
                .with_prompt_optimizer(true)
                .with_fast_pretreatment(false)
                .with_callback_url("https://example.com/callback")
                .with_aigc_watermark(false),
        )
        .expect("serialize options");

        assert_eq!(value["prompt_optimizer"], serde_json::json!(true));
        assert_eq!(value["fast_pretreatment"], serde_json::json!(false));
        assert_eq!(
            value["callback_url"],
            serde_json::json!("https://example.com/callback")
        );
        assert_eq!(value["aigc_watermark"], serde_json::json!(false));
    }

    #[test]
    fn minimaxi_video_options_serialize_sparse_shape() {
        let value =
            serde_json::to_value(MinimaxiVideoOptions::new()).expect("serialize empty options");
        assert_eq!(value, serde_json::json!({}));
        assert!(MinimaxiVideoOptions::new().is_empty());
    }
}
