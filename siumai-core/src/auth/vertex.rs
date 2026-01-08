//! Vertex AI Utilities
//!
//! This module provides utility functions for working with Google Cloud Vertex AI,
//! including URL construction for different publishers (Google, Anthropic, etc.).

/// Build a Vertex AI base URL given project, location and publisher.
///
/// Example:
/// - publisher "google" for Gemini
/// - publisher "anthropic" for Claude on Vertex
///
/// # Arguments
///
/// * `project` - The Google Cloud project ID
/// * `location` - The region/location (e.g., "us-central1", "global")
/// * `publisher` - The model publisher (e.g., "google", "anthropic")
///
/// # Returns
///
/// A fully qualified Vertex AI API base URL
///
/// # Example
///
/// ```rust,ignore
/// use siumai::experimental::auth::vertex::vertex_base_url;
///
/// let url = vertex_base_url("my-project", "us-central1", "google");
/// assert_eq!(
///     url,
///     "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google"
/// );
/// ```
pub fn vertex_base_url(project: &str, location: &str, publisher: &str) -> String {
    // Prefer the regional host (official docs use `https://{location}-aiplatform.googleapis.com`).
    // For `global`, fall back to the global host.
    // https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/{publisher}
    // https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/{publisher}
    let host = if location == "global" {
        "aiplatform.googleapis.com".to_string()
    } else {
        format!("{location}-aiplatform.googleapis.com")
    };
    format!(
        "https://{}/v1/projects/{}/locations/{}/publishers/{}",
        host, project, location, publisher
    )
}

/// Build a Google Vertex provider base URL (Vercel AI SDK aligned).
///
/// This uses the `v1beta1` API prefix and the Google publisher namespace:
/// `https://{host}/v1beta1/projects/{project}/locations/{location}/publishers/google`
///
/// Note: Vertex also supports an "express mode" base URL that does not require project/location:
/// `https://aiplatform.googleapis.com/v1/publishers/google`.
pub fn google_vertex_base_url(project: &str, location: &str) -> String {
    let host = if location == "global" {
        "aiplatform.googleapis.com".to_string()
    } else {
        format!("{location}-aiplatform.googleapis.com")
    };
    format!(
        "https://{}/v1beta1/projects/{}/locations/{}/publishers/google",
        host, project, location
    )
}

/// Express mode base URL for the Google Vertex provider (Vercel AI SDK aligned).
pub const GOOGLE_VERTEX_EXPRESS_BASE_URL: &str =
    "https://aiplatform.googleapis.com/v1/publishers/google";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_base_url() {
        let url = vertex_base_url("myproj", "us-central1", "google");
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/myproj/locations/us-central1/publishers/google"
        );
        let url2 = vertex_base_url("myproj", "global", "anthropic");
        assert_eq!(
            url2,
            "https://aiplatform.googleapis.com/v1/projects/myproj/locations/global/publishers/anthropic"
        );
    }

    #[test]
    fn test_google_vertex_base_url_v1beta1() {
        let url = google_vertex_base_url("test-project", "us-central1");
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google"
        );

        let url2 = google_vertex_base_url("test-project", "global");
        assert_eq!(
            url2,
            "https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/publishers/google"
        );
    }
}
