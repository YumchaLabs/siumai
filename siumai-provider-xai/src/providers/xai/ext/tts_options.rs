use crate::provider_options::XaiTtsOptions;
use crate::types::TtsRequest;

/// xAI request option helpers for `TtsRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait XaiTtsRequestExt {
    /// Convenience: attach xAI-specific TTS options to `provider_options_map["xai"]`.
    fn with_xai_tts_options(self, options: XaiTtsOptions) -> Self;
}

impl XaiTtsRequestExt for TtsRequest {
    fn with_xai_tts_options(mut self, options: XaiTtsOptions) -> Self {
        if options.is_empty() {
            return self;
        }

        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        match (self.provider_options_map.get("xai").cloned(), value) {
            (Some(serde_json::Value::Object(mut base)), serde_json::Value::Object(overrides)) => {
                for (key, value) in overrides {
                    base.insert(key, value);
                }
                self.provider_options_map
                    .insert("xai", serde_json::Value::Object(base));
            }
            (_, value) => {
                self.provider_options_map.insert("xai", value);
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xai_tts_request_ext_merges_with_existing_xai_provider_options() {
        let request = TtsRequest::new("hello".to_string())
            .with_provider_option("xai", serde_json::json!({ "foo": "bar" }))
            .with_xai_tts_options(XaiTtsOptions::new().with_sample_rate(44_100));

        assert_eq!(
            request.provider_options_map.get("xai"),
            Some(&serde_json::json!({
                "foo": "bar",
                "sample_rate": 44_100
            }))
        );
    }
}
