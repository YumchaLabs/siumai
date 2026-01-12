//! File management and upload types

use std::collections::HashMap;

use super::HttpConfig;

/// File upload request
#[derive(Debug, Clone)]
pub struct FileUploadRequest {
    /// File content as bytes
    pub content: Vec<u8>,
    /// Original filename
    pub filename: String,
    /// MIME type
    pub mime_type: Option<String>,
    /// Purpose of the file (e.g., "assistants", "fine-tune")
    pub purpose: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Per-request HTTP configuration (headers, timeout, etc.)
    pub http_config: Option<HttpConfig>,
}

/// File object metadata
#[derive(Debug, Clone)]
pub struct FileObject {
    /// File ID
    pub id: String,
    /// Original filename
    pub filename: String,
    /// File size in bytes
    pub bytes: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// File purpose
    pub purpose: String,
    /// File status
    pub status: String,
    /// MIME type
    pub mime_type: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// File list query parameters
#[derive(Debug, Clone, Default)]
pub struct FileListQuery {
    /// Filter by purpose
    pub purpose: Option<String>,
    /// Limit number of results
    pub limit: Option<u32>,
    /// Pagination cursor
    pub after: Option<String>,
    /// Sort order
    pub order: Option<String>,
    /// Per-request HTTP configuration (headers, timeout, etc.)
    pub http_config: Option<HttpConfig>,
}

impl FileListQuery {
    /// Add per-request HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }

    /// Add a custom header to the request
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut config = self.http_config.take().unwrap_or_default();
        config.headers.insert(key.into(), value.into());
        self.http_config = Some(config);
        self
    }
}

/// File list response
#[derive(Debug, Clone)]
pub struct FileListResponse {
    /// List of files
    pub files: Vec<FileObject>,
    /// Whether there are more results
    pub has_more: bool,
    /// Next page cursor
    pub next_cursor: Option<String>,
}

/// File deletion response
#[derive(Debug, Clone)]
pub struct FileDeleteResponse {
    /// File ID that was deleted
    pub id: String,
    /// Whether deletion was successful
    pub deleted: bool,
}
