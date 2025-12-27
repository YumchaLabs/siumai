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
/// use siumai::utils::vertex_base_url;
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
}
