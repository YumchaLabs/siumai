//! URL Utility Functions
//!
//! This module provides utility functions for safe URL construction and manipulation.

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
    let base = base.trim_end_matches('/');
    let path = path.trim_start_matches('/');

    if path.is_empty() {
        base.to_string()
    } else {
        format!("{base}/{path}")
    }
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
}
