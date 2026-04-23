//! OpenAI Files API Standard
//!
//! This module provides a reusable `FilesTransformer` implementation for providers
//! that follow OpenAI's files endpoints:
//! - `POST /files` (upload)
//! - `GET /files` (list)
//! - `GET /files/{id}` (retrieve)
//! - `DELETE /files/{id}` (delete)
//! - `GET /files/{id}/content` (content)

use crate::error::LlmError;
use crate::execution::transformers::files::{FilesHttpBody, FilesTransformer};

fn upload_options_object<'a>(
    req: &'a crate::types::FileUploadRequest,
    provider_id: &str,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    match provider_id {
        "azure" => req
            .provider_options
            .get_object("azure")
            .or_else(|| req.provider_options.get_object("openai")),
        _ => req.provider_options.get_object(provider_id),
    }
}

fn upload_purpose(req: &crate::types::FileUploadRequest, provider_id: &str) -> String {
    upload_options_object(req, provider_id)
        .and_then(|options| options.get("purpose"))
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| req.purpose.clone())
}

fn upload_expires_after(
    req: &crate::types::FileUploadRequest,
    provider_id: &str,
) -> Option<String> {
    upload_options_object(req, provider_id)
        .and_then(|options| {
            options
                .get("expiresAfter")
                .or_else(|| options.get("expires_after"))
        })
        .and_then(|value| {
            value
                .as_i64()
                .map(|number| number.to_string())
                .or_else(|| value.as_u64().map(|number| number.to_string()))
                .or_else(|| value.as_str().map(ToOwned::to_owned))
        })
}

fn build_upload_body_impl(
    req: &crate::types::FileUploadRequest,
    provider_id: &str,
) -> Result<FilesHttpBody, LlmError> {
    let detected = req
        .mime_type
        .clone()
        .unwrap_or_else(|| crate::utils::guess_mime(Some(&req.content), req.filename.as_deref()));
    let mut part = reqwest::multipart::Part::bytes(req.content.clone());
    if let Some(filename) = req.filename.clone() {
        part = part.file_name(filename);
    }
    let part = part
        .mime_str(&detected)
        .map_err(|e| LlmError::InvalidParameter(format!("Invalid MIME type '{detected}': {e}")))?;
    let mut form = reqwest::multipart::Form::new()
        .text("purpose", upload_purpose(req, provider_id))
        .part("file", part);
    if let Some(expires_after) = upload_expires_after(req, provider_id) {
        form = form.text("expires_after", expires_after);
    }
    Ok(FilesHttpBody::Multipart(form))
}

fn list_endpoint_impl(query: &Option<crate::types::FileListQuery>) -> String {
    let mut endpoint = "files".to_string();
    if let Some(q) = query {
        let mut params = Vec::new();
        if let Some(purpose) = &q.purpose {
            params.push(format!("purpose={}", urlencoding::encode(purpose)));
        }
        if let Some(limit) = q.limit {
            params.push(format!("limit={limit}"));
        }
        if let Some(after) = &q.after {
            params.push(format!("after={}", urlencoding::encode(after)));
        }
        if let Some(order) = &q.order {
            params.push(format!("order={}", urlencoding::encode(order)));
        }
        if !params.is_empty() {
            endpoint.push('?');
            endpoint.push_str(&params.join("&"));
        }
    }
    endpoint
}

fn transform_file_object_impl(
    raw: &serde_json::Value,
) -> Result<crate::types::FileObject, LlmError> {
    #[derive(serde::Deserialize)]
    struct OpenAiFileResponse {
        id: String,
        object: String,
        bytes: u64,
        created_at: u64,
        filename: Option<String>,
        purpose: String,
        status: String,
        status_details: Option<String>,
    }
    let f: OpenAiFileResponse = serde_json::from_value(raw.clone())
        .map_err(|e| LlmError::ParseError(format!("Failed to parse file: {e}")))?;
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("object".to_string(), serde_json::json!(f.object));
    metadata.insert("status".to_string(), serde_json::json!(f.status));
    if let Some(d) = f.status_details {
        metadata.insert("status_details".to_string(), serde_json::json!(d));
    }
    Ok(crate::types::FileObject {
        id: f.id,
        filename: f.filename,
        bytes: f.bytes,
        created_at: f.created_at,
        purpose: f.purpose,
        status: f.status,
        mime_type: None,
        metadata,
    })
}

fn transform_list_response_impl(
    raw: &serde_json::Value,
) -> Result<crate::types::FileListResponse, LlmError> {
    #[derive(serde::Deserialize)]
    struct OpenAiFileResponse {
        id: String,
        object: String,
        bytes: u64,
        created_at: u64,
        filename: Option<String>,
        purpose: String,
        status: String,
        status_details: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct OpenAiFileListResponse {
        data: Vec<OpenAiFileResponse>,
        has_more: Option<bool>,
    }
    let r: OpenAiFileListResponse = serde_json::from_value(raw.clone())
        .map_err(|e| LlmError::ParseError(format!("Failed to parse list: {e}")))?;
    let files = r
        .data
        .into_iter()
        .map(|f| {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("object".to_string(), serde_json::json!(f.object));
            metadata.insert("status".to_string(), serde_json::json!(f.status));
            if let Some(d) = f.status_details.clone() {
                metadata.insert("status_details".to_string(), serde_json::json!(d));
            }
            crate::types::FileObject {
                id: f.id,
                filename: f.filename,
                bytes: f.bytes,
                created_at: f.created_at,
                purpose: f.purpose,
                status: f.status,
                mime_type: None,
                metadata,
            }
        })
        .collect();
    Ok(crate::types::FileListResponse {
        files,
        has_more: r.has_more.unwrap_or(false),
        next_cursor: None,
    })
}

#[derive(Clone)]
pub struct OpenAiFilesTransformer;

impl FilesTransformer for OpenAiFilesTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn build_upload_body(
        &self,
        req: &crate::types::FileUploadRequest,
    ) -> Result<FilesHttpBody, LlmError> {
        build_upload_body_impl(req, self.provider_id())
    }

    fn list_endpoint(&self, query: &Option<crate::types::FileListQuery>) -> String {
        list_endpoint_impl(query)
    }

    fn retrieve_endpoint(&self, file_id: &str) -> String {
        format!("files/{file_id}")
    }
    fn delete_endpoint(&self, file_id: &str) -> String {
        format!("files/{file_id}")
    }
    fn transform_file_object(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileObject, LlmError> {
        transform_file_object_impl(raw)
    }

    fn transform_list_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        transform_list_response_impl(raw)
    }

    fn content_endpoint(&self, file_id: &str) -> Option<String> {
        Some(format!("files/{file_id}/content"))
    }
}

/// OpenAI files transformer with configurable provider id.
///
/// This is useful for OpenAI-compatible providers that expose OpenAI's files API
/// but should report their own provider id in error messages and metadata.
#[derive(Debug, Clone)]
pub struct OpenAiFilesTransformerWithProviderId {
    provider_id: std::borrow::Cow<'static, str>,
}

impl OpenAiFilesTransformerWithProviderId {
    pub fn new(provider_id: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        Self {
            provider_id: provider_id.into(),
        }
    }
}

impl FilesTransformer for OpenAiFilesTransformerWithProviderId {
    fn provider_id(&self) -> &str {
        self.provider_id.as_ref()
    }

    fn build_upload_body(
        &self,
        req: &crate::types::FileUploadRequest,
    ) -> Result<FilesHttpBody, LlmError> {
        build_upload_body_impl(req, self.provider_id())
    }

    fn list_endpoint(&self, query: &Option<crate::types::FileListQuery>) -> String {
        list_endpoint_impl(query)
    }

    fn retrieve_endpoint(&self, file_id: &str) -> String {
        format!("files/{file_id}")
    }
    fn delete_endpoint(&self, file_id: &str) -> String {
        format!("files/{file_id}")
    }

    fn transform_file_object(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileObject, LlmError> {
        transform_file_object_impl(raw)
    }

    fn transform_list_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        transform_list_response_impl(raw)
    }

    fn content_endpoint(&self, file_id: &str) -> Option<String> {
        Some(format!("files/{file_id}/content"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_files_endpoints() {
        let tx = OpenAiFilesTransformer;
        // list endpoint with params
        let q = crate::types::FileListQuery {
            purpose: Some("assistants".to_string()),
            limit: Some(10),
            after: Some("cursor123".to_string()),
            order: Some("desc".to_string()),
            http_config: None,
        };
        let ep = tx.list_endpoint(&Some(q));
        assert!(ep.starts_with("files?"));
        assert!(ep.contains("purpose=assistants"));
        assert!(ep.contains("limit=10"));
        assert!(ep.contains("after=cursor123"));
        assert!(ep.contains("order=desc"));

        assert_eq!(tx.retrieve_endpoint("file_1"), "files/file_1");
        assert_eq!(tx.delete_endpoint("file_2"), "files/file_2");
        assert_eq!(
            tx.content_endpoint("file_3"),
            Some("files/file_3/content".to_string())
        );
    }

    #[test]
    fn test_openai_files_upload_and_parse() {
        let tx = OpenAiFilesTransformer;
        let req = crate::types::FileUploadRequest {
            content: b"hello".to_vec(),
            filename: Some("hello.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
            purpose: "assistants".to_string(),
            metadata: std::collections::HashMap::new(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            http_config: None,
        };
        match tx.build_upload_body(&req).unwrap() {
            FilesHttpBody::Multipart(_) => {}
            _ => panic!("expected multipart form for OpenAI upload"),
        }

        // file object
        let json = serde_json::json!({
            "id": "file_123",
            "object": "file",
            "bytes": 12,
            "created_at": 1710000000u64,
            "filename": "hello.txt",
            "purpose": "assistants",
            "status": "uploaded",
            "status_details": null
        });
        let fo = tx.transform_file_object(&json).unwrap();
        assert_eq!(fo.id, "file_123");
        assert_eq!(fo.filename.as_deref(), Some("hello.txt"));
        assert_eq!(fo.purpose, "assistants");
        assert_eq!(fo.status, "uploaded");

        // list response
        let list = serde_json::json!({
            "object": "list",
            "data": [json],
            "has_more": false
        });
        let lr = tx.transform_list_response(&list).unwrap();
        assert_eq!(lr.files.len(), 1);
        assert_eq!(lr.files[0].status, "uploaded");
        assert!(!lr.has_more);
    }

    #[test]
    fn files_status_is_preserved_from_api() {
        let tx = OpenAiFilesTransformer;
        let json = serde_json::json!({
            "id": "file_999",
            "object": "file",
            "bytes": 12,
            "created_at": 1710000000u64,
            "filename": "hello.txt",
            "purpose": "assistants",
            "status": "processed",
            "status_details": null
        });
        let fo = tx.transform_file_object(&json).unwrap();
        assert_eq!(fo.status, "processed");
    }

    #[test]
    fn configurable_provider_id_reports_custom_id() {
        let tx = OpenAiFilesTransformerWithProviderId::new("openai-compatible");
        assert_eq!(tx.provider_id(), "openai-compatible");
    }
}
