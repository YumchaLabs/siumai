use crate::types::{CacheControl, ChatMessage, ContentPart, MessageContent};

/// Anthropic request option helpers for `ChatMessage`.
///
/// This provider-owned extension keeps Anthropic prompt-cache and document-part options out of the
/// spec chat builder. Historical `ChatMessageBuilder` Anthropic helper methods were removed from
/// `siumai-spec`; migrate those call sites to this trait.
pub trait AnthropicChatMessageExt {
    /// Attach request-level Anthropic prompt cache control.
    fn with_anthropic_cache_control(self, cache: CacheControl) -> Self;

    /// Attach Anthropic prompt cache control to one multimodal content part.
    fn with_anthropic_part_cache_control(self, index: usize, cache: CacheControl) -> Self;

    /// Attach Anthropic prompt cache control to multiple multimodal content parts.
    fn with_anthropic_parts_cache_control<I: IntoIterator<Item = usize>>(
        self,
        indices: I,
        cache: CacheControl,
    ) -> Self;

    /// Enable or disable Anthropic document citations on one content part.
    fn with_anthropic_document_citations_for_part(self, index: usize, enabled: bool) -> Self;

    /// Set Anthropic document title/context metadata on one content part.
    fn with_anthropic_document_metadata_for_part(
        self,
        index: usize,
        title: Option<String>,
        context: Option<String>,
    ) -> Self;
}

fn cache_control_to_json(cache: &CacheControl) -> serde_json::Value {
    match cache {
        CacheControl::Ephemeral => serde_json::json!({ "type": "ephemeral" }),
        CacheControl::Persistent { ttl } => {
            let mut obj = serde_json::json!({ "type": "ephemeral" });
            if let Some(duration) = ttl {
                obj["ttl"] = serde_json::json!(duration.as_secs());
            }
            obj
        }
    }
}

fn with_anthropic_provider_options(
    part: &mut ContentPart,
    update: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) {
    let Some(provider_options) = part.provider_options_mut() else {
        return;
    };

    let mut anthropic = provider_options
        .get("anthropic")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();

    update(&mut anthropic);

    if anthropic.is_empty() {
        provider_options.0.remove("anthropic");
    } else {
        provider_options.insert("anthropic", serde_json::Value::Object(anthropic));
    }
}

fn update_part(
    message: &mut ChatMessage,
    index: usize,
    update: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) {
    let MessageContent::MultiModal(parts) = &mut message.content else {
        return;
    };
    let Some(part) = parts.get_mut(index) else {
        return;
    };

    with_anthropic_provider_options(part, update);
}

impl AnthropicChatMessageExt for ChatMessage {
    fn with_anthropic_cache_control(mut self, cache: CacheControl) -> Self {
        let cache_json = cache_control_to_json(&cache);
        self.metadata.cache_control = Some(cache);
        self.provider_options_mut().insert(
            "anthropic",
            serde_json::json!({ "cacheControl": cache_json }),
        );
        self
    }

    fn with_anthropic_part_cache_control(mut self, index: usize, cache: CacheControl) -> Self {
        update_part(&mut self, index, |anthropic| {
            anthropic.insert("cacheControl".to_string(), cache_control_to_json(&cache));
        });
        self
    }

    fn with_anthropic_parts_cache_control<I: IntoIterator<Item = usize>>(
        mut self,
        indices: I,
        cache: CacheControl,
    ) -> Self {
        for index in indices {
            self = self.with_anthropic_part_cache_control(index, cache.clone());
        }
        self
    }

    fn with_anthropic_document_citations_for_part(mut self, index: usize, enabled: bool) -> Self {
        update_part(&mut self, index, |anthropic| {
            if enabled {
                anthropic.insert(
                    "citations".to_string(),
                    serde_json::json!({ "enabled": true }),
                );
            } else {
                anthropic.remove("citations");
            }
        });
        self
    }

    fn with_anthropic_document_metadata_for_part(
        mut self,
        index: usize,
        title: Option<String>,
        context: Option<String>,
    ) -> Self {
        update_part(&mut self, index, |anthropic| {
            match title.as_ref() {
                Some(title) if !title.is_empty() => {
                    anthropic.insert(
                        "title".to_string(),
                        serde_json::Value::String(title.clone()),
                    );
                }
                _ => {
                    anthropic.remove("title");
                }
            }

            match context.as_ref() {
                Some(context) if !context.is_empty() => {
                    anthropic.insert(
                        "context".to_string(),
                        serde_json::Value::String(context.clone()),
                    );
                }
                _ => {
                    anthropic.remove("context");
                }
            }
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ContentPart};

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn anthropic_chat_message_extension_source_stays_request_side() {
        let source = include_str!("chat_message.rs");
        let request_source =
            source_section(source, "pub trait AnthropicChatMessageExt", "#[cfg(test)]");

        for forbidden in [
            "provider_metadata",
            "ProviderMetadata",
            "ChatResponse",
            "siumai_protocol_",
            "reqwest",
            "tokio",
            "async fn",
            ".await",
        ] {
            assert!(
                !request_source.contains(forbidden),
                "Anthropic chat message extension helpers must stay request-only"
            );
        }
    }

    #[test]
    fn cache_control_writes_request_provider_options() {
        let message = ChatMessage::system("cached")
            .build()
            .with_anthropic_cache_control(CacheControl::Ephemeral);

        assert!(message.metadata.cache_control.is_some());
        let anthropic = message
            .provider_options
            .get_object("anthropic")
            .expect("anthropic provider options");
        assert_eq!(
            anthropic["cacheControl"],
            serde_json::json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn document_helpers_write_part_provider_options() {
        let message = ChatMessage::user("hello")
            .with_file_url("https://example.com/doc.pdf", "application/pdf")
            .build()
            .with_anthropic_part_cache_control(1, CacheControl::Ephemeral)
            .with_anthropic_document_citations_for_part(1, true)
            .with_anthropic_document_metadata_for_part(
                1,
                Some("Doc Title".to_string()),
                Some("Doc Context".to_string()),
            );

        let MessageContent::MultiModal(parts) = message.content else {
            panic!("expected multimodal content");
        };
        let ContentPart::File {
            provider_options, ..
        } = &parts[1]
        else {
            panic!("expected file part");
        };

        let anthropic = provider_options
            .get_object("anthropic")
            .expect("anthropic provider options");
        assert_eq!(
            anthropic["cacheControl"],
            serde_json::json!({ "type": "ephemeral" })
        );
        assert_eq!(anthropic["citations"]["enabled"], serde_json::json!(true));
        assert_eq!(anthropic["title"], serde_json::json!("Doc Title"));
        assert_eq!(anthropic["context"], serde_json::json!("Doc Context"));
    }
}
