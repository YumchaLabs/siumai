//! Files transformers (upload/list/retrieve/delete/content)
//!
//! This module defines generic file transformation traits to unify how
//! providers build requests and map responses for file management APIs.

use crate::error::LlmError;
use crate::types::{FileListQuery, FileListResponse, FileObject, FileUploadRequest};

/// Output body for files HTTP requests
pub enum FilesHttpBody {
    Json(serde_json::Value),
    Multipart(reqwest::multipart::Form),
}

/// Files transformer interface for providers
pub trait FilesTransformer: Send + Sync {
    fn provider_id(&self) -> &str;

    /// Build upload HTTP body (JSON or multipart)
    fn build_upload_body(&self, req: &FileUploadRequest) -> Result<FilesHttpBody, LlmError>;

    /// Endpoint builders (relative to base URL)
    fn upload_endpoint(&self, _req: &FileUploadRequest) -> String {
        "/files".to_string()
    }
    fn list_endpoint(&self, query: &Option<FileListQuery>) -> String;
    fn retrieve_endpoint(&self, file_id: &str) -> String;
    fn delete_endpoint(&self, file_id: &str) -> String;

    /// Transformers for responses
    fn transform_file_object(&self, raw: &serde_json::Value) -> Result<FileObject, LlmError>;
    fn transform_list_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<FileListResponse, LlmError>;

    /// Extract a content URL (if provider returns one) from a FileObject
    fn content_url_from_file_object(&self, _file: &FileObject) -> Option<String> {
        None
    }

    /// Provide API content endpoint when provider serves content via API
    fn content_endpoint(&self, _file_id: &str) -> Option<String> {
        None
    }
}
