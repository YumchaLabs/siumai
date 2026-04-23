//! Gemini grounding → Vercel-aligned sources.

use super::types::{GroundingChunk, GroundingMetadata};
use serde::{Deserialize, Serialize};

/// A normalized "source" entry (Vercel-aligned), extracted from grounding chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GeminiSource {
    /// Source identifier (stable within a response).
    pub id: String,

    /// Source type ("url" or "document").
    pub source_type: String,

    /// Source URL (only for `source_type = "url"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Optional title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Media type for document sources (e.g. "application/pdf").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,

    /// Filename for document sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

fn filename_from_path(path: &str) -> Option<String> {
    path.split('/')
        .rfind(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn media_type_and_filename_from_uri(uri: &str) -> (String, Option<String>) {
    let lower = uri.to_ascii_lowercase();
    let filename = filename_from_path(uri);

    if lower.ends_with(".pdf") {
        return ("application/pdf".to_string(), filename);
    }
    if lower.ends_with(".txt") {
        return ("text/plain".to_string(), filename);
    }
    if lower.ends_with(".docx") {
        return (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string(),
            filename,
        );
    }
    if lower.ends_with(".doc") {
        return ("application/msword".to_string(), filename);
    }
    if lower.ends_with(".md") || lower.ends_with(".markdown") {
        return ("text/markdown".to_string(), filename);
    }

    ("application/octet-stream".to_string(), filename)
}

pub(crate) fn source_key(source: &GeminiSource) -> String {
    if source.source_type == "url" {
        return format!("url:{}", source.url.as_deref().unwrap_or_default());
    }
    format!(
        "doc:{}:{}:{}",
        source.media_type.as_deref().unwrap_or_default(),
        source.filename.as_deref().unwrap_or_default(),
        source.title.as_deref().unwrap_or_default()
    )
}

pub(crate) fn extract_sources_with_generate_id<F>(
    grounding_metadata: Option<&GroundingMetadata>,
    mut generate_id: F,
) -> Vec<GeminiSource>
where
    F: FnMut() -> String,
{
    let Some(grounding_metadata) = grounding_metadata else {
        return Vec::new();
    };
    let Some(chunks) = grounding_metadata.grounding_chunks.as_ref() else {
        return Vec::new();
    };

    let mut out: Vec<GeminiSource> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut push_unique = |mut source: GeminiSource| {
        let key = source_key(&source);
        if !seen.insert(key) {
            return;
        }
        source.id = generate_id();
        out.push(source);
    };

    for chunk in chunks {
        match chunk {
            GroundingChunk::Web { web } => {
                let Some(uri) = web.uri.as_deref() else {
                    continue;
                };
                if uri.is_empty() {
                    continue;
                }
                push_unique(GeminiSource {
                    id: String::new(),
                    source_type: "url".to_string(),
                    url: Some(uri.to_string()),
                    title: web.title.clone(),
                    media_type: None,
                    filename: None,
                });
            }
            GroundingChunk::RetrievedContext { retrieved_context } => {
                let uri = retrieved_context.uri.as_deref();
                let file_search_store = retrieved_context.file_search_store.as_deref();

                if let Some(uri) = uri
                    && !uri.is_empty()
                    && (uri.starts_with("http://") || uri.starts_with("https://"))
                {
                    push_unique(GeminiSource {
                        id: String::new(),
                        source_type: "url".to_string(),
                        url: Some(uri.to_string()),
                        title: retrieved_context.title.clone(),
                        media_type: None,
                        filename: None,
                    });
                    continue;
                }

                if let Some(uri) = uri
                    && !uri.is_empty()
                {
                    let title = retrieved_context
                        .title
                        .clone()
                        .unwrap_or_else(|| "Unknown Document".to_string());
                    let (media_type, filename) = media_type_and_filename_from_uri(uri);
                    push_unique(GeminiSource {
                        id: String::new(),
                        source_type: "document".to_string(),
                        url: None,
                        title: Some(title),
                        media_type: Some(media_type),
                        filename,
                    });
                    continue;
                }

                if let Some(store) = file_search_store
                    && !store.is_empty()
                {
                    let title = retrieved_context
                        .title
                        .clone()
                        .unwrap_or_else(|| "Unknown Document".to_string());
                    let filename = filename_from_path(store);
                    push_unique(GeminiSource {
                        id: String::new(),
                        source_type: "document".to_string(),
                        url: None,
                        title: Some(title),
                        media_type: Some("application/octet-stream".to_string()),
                        filename,
                    });
                }
            }
            GroundingChunk::Maps { maps } => {
                let Some(uri) = maps.uri.as_deref() else {
                    continue;
                };
                if uri.is_empty() {
                    continue;
                }
                push_unique(GeminiSource {
                    id: String::new(),
                    source_type: "url".to_string(),
                    url: Some(uri.to_string()),
                    title: maps.title.clone(),
                    media_type: None,
                    filename: None,
                });
            }
        }
    }

    out
}
