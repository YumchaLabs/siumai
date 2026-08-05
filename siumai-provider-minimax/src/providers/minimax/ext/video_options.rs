//! MiniMax video options (extension API).
//!
//! These typed options are provider-owned and are carried via the open `providerOptions` map.

pub use crate::provider_options::MinimaxVideoOptions;
use crate::types::video::VideoGenerationRequest;

/// Extension trait for `VideoGenerationRequest` to attach MiniMax-specific video options.
pub trait MinimaxVideoRequestExt {
    fn with_minimax_video_options(self, options: MinimaxVideoOptions) -> Self;
}

impl MinimaxVideoRequestExt for VideoGenerationRequest {
    fn with_minimax_video_options(mut self, options: MinimaxVideoOptions) -> Self {
        if options.is_empty() {
            return self;
        }

        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        match (self.provider_options_map.get("minimax").cloned(), value) {
            (Some(serde_json::Value::Object(mut base)), serde_json::Value::Object(overrides)) => {
                for (key, value) in overrides {
                    base.insert(key, value);
                }
                self.provider_options_map
                    .insert("minimax", serde_json::Value::Object(base));
            }
            (_, value) => {
                self.provider_options_map.insert("minimax", value);
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn minimax_video_request_extension_source_does_not_read_response_metadata() {
        let source = include_str!("video_options.rs");
        let request_source =
            source_section(source, "pub trait MinimaxVideoRequestExt", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "MiniMax video request extension helpers must stay request-only"
            );
        }
    }

    #[test]
    fn video_request_ext_attaches_minimax_video_options() {
        let request = VideoGenerationRequest::new("hailuo-2.3", "hi").with_minimax_video_options(
            MinimaxVideoOptions::new()
                .with_prompt_optimizer(true)
                .with_callback_url("https://example.com/callback"),
        );

        let value = request
            .provider_options_map
            .get("minimax")
            .expect("minimax video options present");
        assert_eq!(value["prompt_optimizer"], serde_json::json!(true));
        assert_eq!(
            value["callback_url"],
            serde_json::json!("https://example.com/callback")
        );
    }
}
