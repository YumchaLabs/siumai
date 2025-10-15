//! Vertex AI endpoint helpers.
//! Build base URLs for publisher-specific model endpoints.

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

#[cfg(test)]
mod tests {
    #[test]
    fn test_vertex_base_url() {
        let url = super::vertex_base_url("myproj", "us-central1", "google");
        assert_eq!(
            url,
            "https://aiplatform.googleapis.com/v1/projects/myproj/locations/us-central1/publishers/google"
        );
        let url2 = super::vertex_base_url("myproj", "global", "anthropic");
        assert_eq!(
            url2,
            "https://aiplatform.googleapis.com/v1/projects/myproj/locations/global/publishers/anthropic"
        );
    }
}
