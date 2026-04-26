//! URL Utility Functions
//!
//! This module provides utility functions for safe URL construction and manipulation.

use std::collections::BTreeMap;

/// Safely join a base URL with a path, handling trailing/leading slashes correctly
///
/// This function ensures that there's exactly one slash between the base URL and path,
/// regardless of whether the base URL ends with a slash or the path starts with one.
///
/// # Examples
/// ```rust
/// use siumai::experimental::utils::url::join_url;
///
/// assert_eq!(join_url("https://api.example.com", "v1/chat"), "https://api.example.com/v1/chat");
/// assert_eq!(join_url("https://api.example.com/", "v1/chat"), "https://api.example.com/v1/chat");
/// assert_eq!(join_url("https://api.example.com", "/v1/chat"), "https://api.example.com/v1/chat");
/// assert_eq!(join_url("https://api.example.com/", "/v1/chat"), "https://api.example.com/v1/chat");
/// ```
pub fn join_url(base: &str, path: &str) -> String {
    let path = path.trim_start_matches('/');

    if path.is_empty() {
        return base.trim_end_matches('/').to_string();
    }

    let (path_with_query, fragment) = match path.split_once('#') {
        Some((path, fragment)) => (path, Some(fragment)),
        None => (path, None),
    };
    let (path, query) = match path_with_query.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (path_with_query, None),
    };

    if let Ok(mut url) = reqwest::Url::parse(base) {
        let current_path = url.path().trim_end_matches('/');
        let joined_path = if current_path.is_empty() || current_path == "/" {
            format!("/{path}")
        } else {
            format!("{current_path}/{path}")
        };
        url.set_path(&joined_path);
        if let Some(query) = query {
            url.set_query(Some(query));
        }
        if let Some(fragment) = fragment {
            url.set_fragment(Some(fragment));
        }
        return url.to_string();
    }

    let base = base.trim_end_matches('/');
    if query.is_some() || fragment.is_some() {
        let mut joined = format!("{base}/{path}");
        if let Some(query) = query {
            joined.push('?');
            joined.push_str(query);
        }
        if let Some(fragment) = fragment {
            joined.push('#');
            joined.push_str(fragment);
        }
        return joined;
    }
    format!("{base}/{path}")
}

/// Replace the query string of a URL using a deterministic parameter map.
///
/// This mirrors AI SDK's provider-level `queryParams` behavior where the final URL search string
/// is derived from the configured map rather than merged incrementally.
pub fn with_query_params(url: &str, query_params: &BTreeMap<String, String>) -> String {
    if query_params.is_empty() {
        return url.to_string();
    }

    let Ok(mut parsed) = reqwest::Url::parse(url) else {
        return url.to_string();
    };

    parsed.set_query(None);
    {
        let mut pairs = parsed.query_pairs_mut();
        for (key, value) in query_params {
            pairs.append_pair(key, value);
        }
    }

    parsed.to_string()
}

/// Remove a single trailing slash from an optional URL string.
///
/// This mirrors AI SDK `withoutTrailingSlash`: missing input stays missing, and only one final
/// slash is removed.
pub fn without_trailing_slash(url: Option<&str>) -> Option<String> {
    url.map(|url| url.strip_suffix('/').unwrap_or(url).to_string())
}

/// Join multiple URL segments safely
///
/// # Examples
/// ```rust
/// use siumai::experimental::utils::url::join_url_segments;
///
/// assert_eq!(
///     join_url_segments(&["https://api.example.com", "v1", "models", "gpt-4"]),
///     "https://api.example.com/v1/models/gpt-4"
/// );
/// assert_eq!(
///     join_url_segments(&["https://api.example.com/", "/v1/", "/models/", "/gpt-4"]),
///     "https://api.example.com/v1/models/gpt-4"
/// );
/// ```
pub fn join_url_segments(segments: &[&str]) -> String {
    if segments.is_empty() {
        return String::new();
    }

    let mut result = segments[0].trim_end_matches('/').to_string();

    for segment in &segments[1..] {
        let clean_segment = segment.trim_start_matches('/').trim_end_matches('/');
        if !clean_segment.is_empty() {
            result.push('/');
            result.push_str(clean_segment);
        }
    }

    result
}

/// Normalize a URL by removing duplicate slashes (except after protocol)
///
/// # Examples
/// ```rust
/// use siumai::experimental::utils::url::normalize_url;
///
/// assert_eq!(normalize_url("https://api.example.com//v1//chat"), "https://api.example.com/v1/chat");
/// assert_eq!(normalize_url("http://localhost:11434//api//chat"), "http://localhost:11434/api/chat");
/// ```
pub fn normalize_url(url: &str) -> String {
    if let Some(protocol_end) = url.find("://") {
        let protocol_part = &url[..protocol_end + 3];
        let path_part = &url[protocol_end + 3..];

        // Replace multiple slashes with single slash in the path part
        let normalized_path = path_part
            .split('/')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("/");

        if normalized_path.is_empty() {
            protocol_part.to_string()
        } else {
            format!("{protocol_part}{normalized_path}")
        }
    } else {
        // No protocol, just normalize slashes
        url.split('/')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("/")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_url() {
        // Basic cases
        assert_eq!(
            join_url("https://api.example.com", "v1/chat"),
            "https://api.example.com/v1/chat"
        );
        assert_eq!(
            join_url("https://api.example.com/", "v1/chat"),
            "https://api.example.com/v1/chat"
        );
        assert_eq!(
            join_url("https://api.example.com", "/v1/chat"),
            "https://api.example.com/v1/chat"
        );
        assert_eq!(
            join_url("https://api.example.com/", "/v1/chat"),
            "https://api.example.com/v1/chat"
        );
        assert_eq!(
            join_url(
                "https://api.example.com/v1?api-version=2025-04-01",
                "/audio/speech"
            ),
            "https://api.example.com/v1/audio/speech?api-version=2025-04-01"
        );
        assert_eq!(
            join_url(
                "https://api.example.com/v1",
                "files?purpose=assistants&limit=10"
            ),
            "https://api.example.com/v1/files?purpose=assistants&limit=10"
        );
        assert_eq!(
            join_url("https://api.example.com/v1", "responses#stream-end"),
            "https://api.example.com/v1/responses#stream-end"
        );

        // Empty path
        assert_eq!(
            join_url("https://api.example.com", ""),
            "https://api.example.com"
        );
        assert_eq!(
            join_url("https://api.example.com/", ""),
            "https://api.example.com"
        );

        // Multiple slashes
        assert_eq!(
            join_url("https://api.example.com///", "///v1/chat"),
            "https://api.example.com/v1/chat"
        );
    }

    #[test]
    fn test_join_url_segments() {
        assert_eq!(
            join_url_segments(&["https://api.example.com", "v1", "models", "gpt-4"]),
            "https://api.example.com/v1/models/gpt-4"
        );
        assert_eq!(
            join_url_segments(&["https://api.example.com/", "/v1/", "/models/", "/gpt-4"]),
            "https://api.example.com/v1/models/gpt-4"
        );
        assert_eq!(
            join_url_segments(&["https://api.example.com"]),
            "https://api.example.com"
        );
        assert_eq!(join_url_segments(&[]), "");
    }

    #[test]
    fn test_normalize_url() {
        assert_eq!(
            normalize_url("https://api.example.com//v1//chat"),
            "https://api.example.com/v1/chat"
        );
        assert_eq!(
            normalize_url("http://localhost:11434//api//chat"),
            "http://localhost:11434/api/chat"
        );
        assert_eq!(
            normalize_url("https://api.example.com"),
            "https://api.example.com"
        );
        assert_eq!(
            normalize_url("https://api.example.com/"),
            "https://api.example.com"
        );
        assert_eq!(
            normalize_url("https://api.example.com/v1/chat"),
            "https://api.example.com/v1/chat"
        );
    }

    #[test]
    fn test_real_world_cases() {
        // OpenAI
        assert_eq!(
            join_url("https://api.openai.com/v1", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            join_url("https://api.openai.com/v1/", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );

        // Anthropic
        assert_eq!(
            join_url("https://api.anthropic.com", "v1/messages"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            join_url("https://api.anthropic.com/", "v1/messages"),
            "https://api.anthropic.com/v1/messages"
        );

        // Ollama
        assert_eq!(
            join_url("http://localhost:11434", "api/chat"),
            "http://localhost:11434/api/chat"
        );
        assert_eq!(
            join_url("http://localhost:11434/", "api/chat"),
            "http://localhost:11434/api/chat"
        );

        // Custom proxy with trailing slash
        assert_eq!(
            join_url("https://api1.oaipro.com/", "v1/messages"),
            "https://api1.oaipro.com/v1/messages"
        );
    }

    #[test]
    fn test_with_query_params_replaces_search_string() {
        let params = BTreeMap::from([
            ("api-version".to_string(), "2025-04-01".to_string()),
            ("tenant".to_string(), "acme".to_string()),
        ]);

        assert_eq!(
            with_query_params("https://api.example.com/v1/chat/completions", &params),
            "https://api.example.com/v1/chat/completions?api-version=2025-04-01&tenant=acme"
        );
        assert_eq!(
            with_query_params(
                "https://api.example.com/v1/chat/completions?legacy=true",
                &params
            ),
            "https://api.example.com/v1/chat/completions?api-version=2025-04-01&tenant=acme"
        );
    }

    #[test]
    fn test_without_trailing_slash_matches_ai_sdk_semantics() {
        assert_eq!(
            without_trailing_slash(Some("https://api.example.com/")),
            Some("https://api.example.com".to_string())
        );
        assert_eq!(
            without_trailing_slash(Some("https://api.example.com//")),
            Some("https://api.example.com/".to_string())
        );
        assert_eq!(
            without_trailing_slash(Some("https://api.example.com")),
            Some("https://api.example.com".to_string())
        );
        assert_eq!(without_trailing_slash(None), None);
    }
}
