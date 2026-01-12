use super::*;

/// Files transformer for Gemini
#[derive(Clone)]
pub struct GeminiFilesTransformer {
    pub config: GeminiConfig,
}

impl GeminiFilesTransformer {
    fn convert_file(&self, gemini_file: &GeminiFile) -> crate::types::FileObject {
        let id = gemini_file
            .name
            .as_ref()
            .and_then(|n| n.strip_prefix("files/"))
            .unwrap_or("")
            .to_string();
        let bytes = gemini_file
            .size_bytes
            .as_ref()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let created_at = gemini_file
            .create_time
            .as_ref()
            .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
            .map(|dt| dt.timestamp() as u64)
            .unwrap_or(0);
        let status = match gemini_file.state {
            Some(GeminiFileState::Active) => "active".to_string(),
            Some(GeminiFileState::Processing) => "processing".to_string(),
            Some(GeminiFileState::Failed) => "failed".to_string(),
            _ => "unknown".to_string(),
        };
        let filename = gemini_file
            .display_name
            .clone()
            .unwrap_or_else(|| format!("file_{id}"));
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("provider".to_string(), serde_json::json!("gemini"));
        if let Some(uri) = &gemini_file.uri {
            metadata.insert("uri".to_string(), serde_json::json!(uri));
        }
        if let Some(hash) = &gemini_file.sha256_hash {
            metadata.insert("sha256_hash".to_string(), serde_json::json!(hash));
        }
        if let Some(exp) = &gemini_file.expiration_time {
            metadata.insert("expiration_time".to_string(), serde_json::json!(exp));
        }
        crate::types::FileObject {
            id,
            filename,
            bytes,
            created_at,
            purpose: "general".to_string(),
            status,
            mime_type: gemini_file.mime_type.clone(),
            metadata,
        }
    }
}

impl FilesTransformer for GeminiFilesTransformer {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    fn build_upload_body(
        &self,
        req: &crate::types::FileUploadRequest,
    ) -> Result<FilesHttpBody, LlmError> {
        let detected = req
            .mime_type
            .clone()
            .unwrap_or_else(|| crate::utils::guess_mime(Some(&req.content), Some(&req.filename)));
        let part = reqwest::multipart::Part::bytes(req.content.clone())
            .file_name(req.filename.clone())
            .mime_str(&detected)
            .map_err(|e| LlmError::HttpError(format!("Invalid MIME type: {e}")))?;
        let mut form = reqwest::multipart::Form::new().part("file", part);
        if let Some(name) = req.metadata.get("display_name") {
            form = form.text("display_name", name.clone());
        }
        Ok(FilesHttpBody::Multipart(form))
    }

    fn list_endpoint(&self, query: &Option<crate::types::FileListQuery>) -> String {
        let mut endpoint = "files".to_string();
        let mut params = Vec::new();
        if let Some(q) = query {
            if let Some(limit) = q.limit {
                params.push(format!("pageSize={limit}"));
            }
            if let Some(after) = &q.after {
                params.push(format!("pageToken={after}"));
            }
        }
        if !params.is_empty() {
            endpoint.push('?');
            endpoint.push_str(&params.join("&"));
        }
        endpoint
    }

    fn retrieve_endpoint(&self, file_id: &str) -> String {
        if file_id.starts_with("files/") {
            file_id.to_string()
        } else {
            format!("files/{file_id}")
        }
    }

    fn delete_endpoint(&self, file_id: &str) -> String {
        if file_id.starts_with("files/") {
            file_id.to_string()
        } else {
            format!("files/{file_id}")
        }
    }

    fn transform_file_object(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileObject, LlmError> {
        // Gemini upload returns CreateFileResponse; retrieve returns GeminiFile directly
        if raw.get("file").is_some() {
            let resp: CreateFileResponse = serde_json::from_value(raw.clone()).map_err(|e| {
                LlmError::ParseError(format!("Failed to parse upload response: {e}"))
            })?;
            let file = resp
                .file
                .ok_or_else(|| LlmError::ParseError("No file in upload response".to_string()))?;
            Ok(self.convert_file(&file))
        } else {
            let file: GeminiFile = serde_json::from_value(raw.clone())
                .map_err(|e| LlmError::ParseError(format!("Failed to parse file response: {e}")))?;
            Ok(self.convert_file(&file))
        }
    }

    fn transform_list_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        let resp: ListFilesResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Failed to parse list response: {e}")))?;
        let files: Vec<crate::types::FileObject> = resp
            .files
            .into_iter()
            .map(|f| self.convert_file(&f))
            .collect();
        Ok(crate::types::FileListResponse {
            files,
            has_more: resp.next_page_token.is_some(),
            next_cursor: resp.next_page_token,
        })
    }

    fn content_url_from_file_object(&self, file: &crate::types::FileObject) -> Option<String> {
        file.metadata
            .get("uri")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}

#[cfg(test)]
mod files_tests {
    use super::*;
    use crate::execution::transformers::files::{FilesHttpBody, FilesTransformer};

    fn sample_config() -> GeminiConfig {
        use secrecy::SecretString;
        GeminiConfig {
            api_key: SecretString::from("x".to_string()),
            base_url: "https://example.com".into(),
            model: "gemini-1.5-flash".into(),
            common_params: crate::types::CommonParams::default(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: crate::types::HttpConfig::default(),
            token_provider: None,
            http_transport: None,
            provider_metadata_key: None,
        }
    }

    #[test]
    fn test_gemini_files_endpoints() {
        let tx = GeminiFilesTransformer {
            config: sample_config(),
        };
        let q = crate::types::FileListQuery {
            limit: Some(20),
            after: Some("page123".into()),
            ..Default::default()
        };
        let ep = tx.list_endpoint(&Some(q));
        assert!(ep.starts_with("files?"));
        assert!(ep.contains("pageSize=20"));
        assert!(ep.contains("pageToken=page123"));
        assert_eq!(tx.retrieve_endpoint("abc"), "files/abc");
        assert_eq!(tx.delete_endpoint("files/def"), "files/def");
        assert!(tx.content_endpoint("ignored").is_none());
    }

    #[test]
    fn test_gemini_files_upload_and_parse() {
        let tx = GeminiFilesTransformer {
            config: sample_config(),
        };
        let req = crate::types::FileUploadRequest {
            content: b"hi".to_vec(),
            filename: "hi.txt".into(),
            mime_type: Some("text/plain".into()),
            purpose: "general".into(),
            metadata: std::collections::HashMap::new(),
            http_config: None,
        };
        match tx.build_upload_body(&req).unwrap() {
            FilesHttpBody::Multipart(_) => {}
            _ => panic!("expected multipart form for Gemini upload"),
        }

        // upload response shape (CreateFileResponse)
        let upload_json = serde_json::json!({
            "file": {
                "name": "files/abc",
                "display_name": "hi.txt",
                "mime_type": "text/plain",
                "size_bytes": "2",
                "create_time": "2024-01-01T00:00:00Z",
                "state": "ACTIVE",
                "uri": "https://content.example/abc"
            }
        });
        let fo = tx.transform_file_object(&upload_json).unwrap();
        assert_eq!(fo.id, "abc");
        assert_eq!(fo.filename, "hi.txt");
        assert_eq!(
            tx.content_url_from_file_object(&fo).as_deref(),
            Some("https://content.example/abc")
        );

        // list response
        let list_json = serde_json::json!({
            "files": [ upload_json["file"].clone() ],
            "next_page_token": null
        });
        let lr = tx.transform_list_response(&list_json).unwrap();
        assert_eq!(lr.files.len(), 1);
        assert!(!lr.has_more);
    }
}
