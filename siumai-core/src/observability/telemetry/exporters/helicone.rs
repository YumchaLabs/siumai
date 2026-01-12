#![allow(clippy::collapsible_if)]
//! Helicone Exporter
//!
//! Export telemetry via Helicone headers for request tracking.
//!
//! Helicone works by intercepting HTTP requests to LLM providers and adding
//! tracking headers. This exporter provides utilities for adding Helicone headers.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use siumai::experimental::observability::telemetry::exporters::helicone::HeliconeExporter;
//! use std::collections::HashMap;
//!
//! let exporter = HeliconeExporter::new("your-api-key");
//!
//! // Get headers to add to LLM requests
//! let mut headers = HashMap::new();
//! exporter.add_headers(&mut headers, Some("session-123"), Some("user-456"));
//! ```

use crate::error::LlmError;
use crate::observability::telemetry::events::TelemetryEvent;
use crate::observability::telemetry::exporters::TelemetryExporter;
use std::collections::HashMap;

/// Helicone exporter
///
/// Helicone works by adding headers to HTTP requests to LLM providers.
/// This exporter provides utilities for adding those headers.
pub struct HeliconeExporter {
    api_key: String,
    base_url: String,
}

impl HeliconeExporter {
    /// Create a new Helicone exporter
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://oai.hconeai.com".to_string(),
        }
    }

    /// Create a new Helicone exporter with custom base URL
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Get the Helicone base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Add Helicone headers to a request
    ///
    /// # Arguments
    ///
    /// * `headers` - HashMap to add headers to
    /// * `session_id` - Optional session ID for grouping requests
    /// * `user_id` - Optional user ID for tracking user-specific metrics
    pub fn add_headers(
        &self,
        headers: &mut HashMap<String, String>,
        session_id: Option<&str>,
        user_id: Option<&str>,
    ) {
        // Required: API key
        headers.insert(
            "Helicone-Auth".to_string(),
            format!("Bearer {}", self.api_key),
        );

        // Optional: Session ID
        if let Some(session) = session_id {
            headers.insert("Helicone-Session-Id".to_string(), session.to_string());
        }

        // Optional: User ID
        if let Some(user) = user_id {
            headers.insert("Helicone-User-Id".to_string(), user.to_string());
        }
    }

    /// Add Helicone headers with additional properties
    pub fn add_headers_with_properties(
        &self,
        headers: &mut HashMap<String, String>,
        session_id: Option<&str>,
        user_id: Option<&str>,
        properties: &HashMap<String, String>,
    ) {
        self.add_headers(headers, session_id, user_id);

        // Add custom properties
        if !properties.is_empty()
            && let Ok(json) = serde_json::to_string(properties)
        {
            headers.insert("Helicone-Property".to_string(), json);
        }
    }

    /// Get headers as a Vec of tuples (for reqwest)
    pub fn get_headers_vec(
        &self,
        session_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Vec<(String, String)> {
        let mut headers = HashMap::new();
        self.add_headers(&mut headers, session_id, user_id);
        headers.into_iter().collect()
    }
}

#[async_trait::async_trait]
impl TelemetryExporter for HeliconeExporter {
    async fn export(&self, _event: &TelemetryEvent) -> Result<(), LlmError> {
        // Helicone doesn't need to export events directly
        // It works by adding headers to HTTP requests
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helicone_exporter_creation() {
        let exporter = HeliconeExporter::new("test-api-key");
        assert_eq!(exporter.api_key, "test-api-key");
        assert_eq!(exporter.base_url, "https://oai.hconeai.com");
    }

    #[test]
    fn test_helicone_headers() {
        let exporter = HeliconeExporter::new("test-api-key");
        let mut headers = HashMap::new();

        exporter.add_headers(&mut headers, Some("session-123"), Some("user-456"));

        assert_eq!(
            headers.get("Helicone-Auth"),
            Some(&"Bearer test-api-key".to_string())
        );
        assert_eq!(
            headers.get("Helicone-Session-Id"),
            Some(&"session-123".to_string())
        );
        assert_eq!(
            headers.get("Helicone-User-Id"),
            Some(&"user-456".to_string())
        );
    }

    #[test]
    fn test_helicone_headers_with_properties() {
        let exporter = HeliconeExporter::new("test-api-key");
        let mut headers = HashMap::new();
        let mut properties = HashMap::new();
        properties.insert("environment".to_string(), "production".to_string());

        exporter.add_headers_with_properties(&mut headers, Some("session-123"), None, &properties);

        assert!(headers.contains_key("Helicone-Auth"));
        assert!(headers.contains_key("Helicone-Property"));
    }

    #[test]
    fn test_get_headers_vec() {
        let exporter = HeliconeExporter::new("test-api-key");
        let headers = exporter.get_headers_vec(Some("session-123"), None);

        assert!(!headers.is_empty());
        assert!(headers.iter().any(|(k, _)| k == "Helicone-Auth"));
        assert!(headers.iter().any(|(k, _)| k == "Helicone-Session-Id"));
    }
}
