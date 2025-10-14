//! Files transformer for OpenAI provider

use crate::error::LlmError;
use crate::transformers::files::{FilesHttpBody, FilesTransformer};

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
        let detected = req.mime_type.clone().unwrap_or_else(|| {
            crate::utils::mime::guess_mime(Some(&req.content), Some(&req.filename))
        });
        let part = reqwest::multipart::Part::bytes(req.content.clone())
            .file_name(req.filename.clone())
            .mime_str(&detected)
            .map_err(|e| LlmError::HttpError(format!("Invalid MIME type: {e}")))?;
        let form = reqwest::multipart::Form::new()
            .text("purpose", req.purpose.clone())
            .part("file", part);
        Ok(FilesHttpBody::Multipart(form))
    }

    fn list_endpoint(&self, query: &Option<crate::types::FileListQuery>) -> String {
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
        #[derive(serde::Deserialize)]
        struct OpenAiFileResponse {
            id: String,
            object: String,
            bytes: u64,
            created_at: u64,
            filename: String,
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
            status: "uploaded".to_string(),
            mime_type: None,
            metadata,
        })
    }

    fn transform_list_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        #[derive(serde::Deserialize)]
        struct OpenAiFileResponse {
            id: String,
            object: String,
            bytes: u64,
            created_at: u64,
            filename: String,
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
                    status: "uploaded".to_string(),
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

    fn content_endpoint(&self, file_id: &str) -> Option<String> {
        Some(format!("files/{file_id}/content"))
    }
}

#[cfg(test)]
mod files_tests {
    use super::*;
    use crate::transformers::files::{FilesHttpBody, FilesTransformer};

    #[test]
    fn test_openai_files_endpoints() {
        let tx = OpenAiFilesTransformer;
        // list endpoint with params
        let q = crate::types::FileListQuery {
            purpose: Some("assistants".to_string()),
            limit: Some(10),
            after: Some("cursor123".to_string()),
            order: Some("desc".to_string()),
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
            filename: "hello.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            purpose: "assistants".to_string(),
            metadata: std::collections::HashMap::new(),
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
        assert_eq!(fo.filename, "hello.txt");
        assert_eq!(fo.purpose, "assistants");

        // list response
        let list = serde_json::json!({
            "object": "list",
            "data": [json],
            "has_more": false
        });
        let lr = tx.transform_list_response(&list).unwrap();
        assert_eq!(lr.files.len(), 1);
        assert!(!lr.has_more);
    }
}
