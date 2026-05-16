//! MiniMaxi TTS options (extension API).
//!
//! These typed options are provider-owned and are carried via the open `providerOptions` map.

pub use crate::provider_options::MinimaxiTtsOptions;
use crate::types::TtsRequest;

/// Extension trait for `TtsRequest` to attach MiniMaxi-specific vendor options.
pub trait MinimaxiTtsRequestExt {
    fn with_minimaxi_tts_options(self, options: MinimaxiTtsOptions) -> Self;
}

impl MinimaxiTtsRequestExt for TtsRequest {
    fn with_minimaxi_tts_options(mut self, options: MinimaxiTtsOptions) -> Self {
        if options.is_empty() {
            return self;
        }

        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        match (self.provider_options_map.get("minimaxi").cloned(), value) {
            (Some(serde_json::Value::Object(mut base)), serde_json::Value::Object(overrides)) => {
                for (k, v) in overrides {
                    base.insert(k, v);
                }
                self.provider_options_map
                    .insert("minimaxi", serde_json::Value::Object(base));
            }
            (_, v) => {
                self.provider_options_map.insert("minimaxi", v);
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn minimaxi_tts_request_extension_source_does_not_read_response_metadata() {
        let source = include_str!("tts_options.rs");
        let request_source =
            source_section(source, "pub trait MinimaxiTtsRequestExt", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "MiniMaxi TTS request extension helpers must stay request-only"
            );
        }
    }
}
