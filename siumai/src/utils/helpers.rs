//! Utility helpers
//!
//! This module contains small utility functions for common tasks like
//! MIME type detection and Vertex AI URL construction.

// ============================================================================
// MIME Type Detection
// ============================================================================

/// Guess MIME by inspecting bytes (magic numbers)
pub fn guess_mime_from_bytes(bytes: &[u8]) -> Option<String> {
    infer::get(bytes).map(|k| k.mime_type().to_string())
}

/// Guess MIME by file path or URL (extension-based)
pub fn guess_mime_from_path_or_url(path_or_url: &str) -> Option<String> {
    mime_guess::from_path(path_or_url)
        .first_raw()
        .map(|s| s.to_string())
}

/// Combined guess: prefer bytes, fall back to extension, otherwise octet-stream
pub fn guess_mime(bytes: Option<&[u8]>, path_or_url: Option<&str>) -> String {
    if let Some(b) = bytes
        && let Some(m) = guess_mime_from_bytes(b)
    {
        return m;
    }
    if let Some(p) = path_or_url
        && let Some(m) = guess_mime_from_path_or_url(p)
    {
        return m;
    }
    "application/octet-stream".to_string()
}

// ============================================================================
// Vertex AI Helpers
// ============================================================================

/// Build a Vertex AI base URL given project, location and publisher.
///
/// Example:
/// - publisher "google" for Gemini
/// - publisher "anthropic" for Claude on Vertex
pub fn vertex_base_url(project: &str, location: &str, publisher: &str) -> String {
    // Prefer the global host; regional hosts are also valid but not necessary here.
    // https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/{publisher}
    format!(
        "https://aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/{}",
        project, location, publisher
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_base_url() {
        let url = vertex_base_url("myproj", "us-central1", "google");
        assert_eq!(
            url,
            "https://aiplatform.googleapis.com/v1/projects/myproj/locations/us-central1/publishers/google"
        );
        let url2 = vertex_base_url("myproj", "global", "anthropic");
        assert_eq!(
            url2,
            "https://aiplatform.googleapis.com/v1/projects/myproj/locations/global/publishers/anthropic"
        );
    }
}
