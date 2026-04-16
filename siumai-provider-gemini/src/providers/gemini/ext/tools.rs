//! Gemini provider-hosted tools helpers (streaming).
//!
//! Gemini can return grounding metadata containing references to web / retrieved context.
//! Siumai normalizes these into Vercel-aligned stable `source` parts, and this helper also keeps
//! accepting the historical `gemini:source` custom-event shadow for compatibility.

use crate::streaming::ChatStreamEvent;

/// Gemini streaming custom events emitted by Siumai.
///
/// Current extension events:
/// - stable `source` parts extracted from grounding chunks
/// - legacy `gemini:source` custom events for backward compatibility
#[derive(Debug, Clone)]
pub enum GeminiCustomEvent {
    Source(GeminiSourceEvent),
}

impl GeminiCustomEvent {
    pub fn from_stream_event(event: &ChatStreamEvent) -> Option<Self> {
        if let Some(source) = event.part_ref().and_then(GeminiSourceEvent::from_part) {
            return Some(GeminiCustomEvent::Source(source));
        }

        let ChatStreamEvent::Custom { event_type, data } = event else {
            return None;
        };

        if event_type == GeminiSourceEvent::EVENT_TYPE {
            return GeminiSourceEvent::from_custom(event_type, data).map(GeminiCustomEvent::Source);
        }

        None
    }
}

/// Source event (`gemini:source`), emitted from grounding metadata (Vercel-aligned).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiSourceEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub source_type: String,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

impl GeminiSourceEvent {
    pub const EVENT_TYPE: &'static str = "gemini:source";

    pub fn from_custom(event_type: &str, data: &serde_json::Value) -> Option<Self> {
        if event_type != Self::EVENT_TYPE {
            return None;
        }
        serde_json::from_value(data.clone()).ok()
    }

    pub fn from_part(part: &crate::types::ChatStreamPart) -> Option<Self> {
        let crate::types::ChatStreamPart::Source { id, source, .. } = part else {
            return None;
        };

        let (url, title, media_type, filename) = match source {
            crate::types::SourcePart::Url { url, title } => {
                (Some(url.clone()), title.clone(), None, None)
            }
            crate::types::SourcePart::Document {
                media_type,
                title,
                filename,
            } => (
                None,
                Some(title.clone()),
                Some(media_type.clone()),
                filename.clone(),
            ),
        };

        Some(Self {
            kind: "source".to_string(),
            source_type: source.source_type().to_string(),
            id: id.clone(),
            url,
            title,
            media_type,
            filename,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gemini_source_runtime_part() {
        let source = ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::Source {
                id: "src_0".to_string(),
                source: crate::types::SourcePart::Url {
                    url: "https://www.rust-lang.org".to_string(),
                    title: Some("Rust".to_string()),
                },
                provider_metadata: None,
            },
        };

        match GeminiCustomEvent::from_stream_event(&source).unwrap() {
            GeminiCustomEvent::Source(ev) => {
                assert_eq!(ev.kind, "source");
                assert_eq!(ev.source_type, "url");
                assert_eq!(ev.id, "src_0");
                assert_eq!(ev.url.as_deref(), Some("https://www.rust-lang.org"));
                assert_eq!(ev.title.as_deref(), Some("Rust"));
            }
        }
    }

    #[test]
    fn parses_gemini_source_custom_event() {
        let source = ChatStreamEvent::Custom {
            event_type: "gemini:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "url",
                "id": "src_0",
                "url": "https://www.rust-lang.org",
                "title": "Rust",
            }),
        };

        match GeminiCustomEvent::from_stream_event(&source).unwrap() {
            GeminiCustomEvent::Source(ev) => {
                assert_eq!(ev.kind, "source");
                assert_eq!(ev.source_type, "url");
                assert_eq!(ev.id, "src_0");
                assert_eq!(ev.url.as_deref(), Some("https://www.rust-lang.org"));
                assert_eq!(ev.title.as_deref(), Some("Rust"));
            }
        }
    }
}
