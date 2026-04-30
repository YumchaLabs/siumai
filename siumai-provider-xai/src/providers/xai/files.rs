use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::common::{
    execute_delete_request, execute_get_binary, execute_get_request, execute_multipart_request,
};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::wiring::HttpExecutionWiring;
use crate::provider_options::XaiFilesOptions;
use crate::retry_api::RetryOptions;
use crate::traits::{FileManagementCapability, ProviderCapabilities};
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};
use async_trait::async_trait;
use reqwest::header::HeaderMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const PROVIDER_ID: &str = "xai";

#[derive(Clone, Copy, Default)]
struct XaiFilesSpec;

impl ProviderSpec for XaiFilesSpec {
    fn id(&self) -> &'static str {
        PROVIDER_ID
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_file_management()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        siumai_protocol_openai::standards::openai::headers::build_openai_compatible_json_headers(
            ctx,
        )
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        siumai_protocol_openai::standards::openai::errors::classify_openai_compatible_http_error(
            PROVIDER_ID,
            status,
            body_text,
        )
    }
}

#[derive(Clone)]
pub(crate) struct XaiFiles {
    provider_context: ProviderContext,
    http_client: reqwest::Client,
    http_transport: Option<Arc<dyn HttpTransport>>,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    retry_options: Option<RetryOptions>,
}

impl XaiFiles {
    pub(crate) fn new(
        provider_context: ProviderContext,
        http_client: reqwest::Client,
        http_transport: Option<Arc<dyn HttpTransport>>,
        http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            provider_context,
            http_client,
            http_transport,
            http_interceptors,
            retry_options,
        }
    }

    fn build_http_config(&self) -> crate::execution::executors::common::HttpExecutionConfig {
        let mut wiring = HttpExecutionWiring::new(
            PROVIDER_ID,
            self.http_client.clone(),
            self.provider_context.clone(),
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(Arc::new(XaiFilesSpec))
    }

    fn base_url(&self) -> String {
        self.provider_context
            .base_url
            .trim_end_matches('/')
            .to_string()
    }

    fn parse_upload_options(
        &self,
        request: &FileUploadRequest,
    ) -> Result<Option<XaiFilesOptions>, LlmError> {
        let Some(value) = request.provider_options.get("xai").cloned() else {
            return Ok(None);
        };

        serde_json::from_value(value).map(Some).map_err(|err| {
            LlmError::InvalidParameter(format!(
                "Invalid xAI file options in providerOptions.xai: {err}"
            ))
        })
    }

    fn build_upload_form(
        request: &FileUploadRequest,
        options: Option<&XaiFilesOptions>,
    ) -> Result<reqwest::multipart::Form, LlmError> {
        let mime_type = request.mime_type.clone().unwrap_or_else(|| {
            crate::utils::guess_mime(Some(&request.content), request.filename.as_deref())
        });

        let mut part = reqwest::multipart::Part::bytes(request.content.clone());
        if let Some(filename) = request.filename.clone() {
            part = part.file_name(filename);
        }
        let part = part.mime_str(&mime_type).map_err(|err| {
            LlmError::InvalidParameter(format!("Invalid MIME type '{mime_type}': {err}"))
        })?;

        let mut form = reqwest::multipart::Form::new().part("file", part);

        if let Some(team_id) = options.and_then(|options| options.team_id.as_deref()) {
            form = form.text("team_id", team_id.to_string());
        }

        if let Some(file_path) = options.and_then(|options| options.file_path.as_deref()) {
            form = form.text("file_path", file_path.to_string());
        }

        Ok(form)
    }
}

#[async_trait]
impl FileManagementCapability for XaiFiles {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        let config = self.build_http_config();
        let url = crate::utils::url::join_url(&self.base_url(), "files");
        let options = self.parse_upload_options(&request)?;
        let request_clone = request.clone();

        let response = execute_multipart_request(
            &config,
            &url,
            move || Self::build_upload_form(&request_clone, options.as_ref()),
            request.http_config.as_ref(),
        )
        .await?;

        map_xai_file_object(&response.json, Some(&request))
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let config = self.build_http_config();
        let mut url = reqwest::Url::parse(&crate::utils::url::join_url(&self.base_url(), "files"))
            .map_err(|err| LlmError::InvalidInput(format!("Invalid xAI files URL: {err}")))?;

        if let Some(query) = &query {
            let mut pairs = url.query_pairs_mut();
            if let Some(limit) = query.limit {
                pairs.append_pair("limit", &limit.to_string());
            }
            if let Some(after) = query
                .after
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                pairs.append_pair("next_token", after);
            }
            if let Some(order) = query.order.as_deref().and_then(normalize_list_order) {
                pairs.append_pair("order", order);
            }
        }

        let response = execute_get_request(
            &config,
            url.as_str(),
            query.as_ref().and_then(|q| q.http_config.as_ref()),
        )
        .await?;
        map_xai_file_list_response(&response.json)
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        let config = self.build_http_config();
        let url = crate::utils::url::join_url(&self.base_url(), &format!("files/{file_id}"));
        let response = execute_get_request(&config, &url, None).await?;
        map_xai_file_object(&response.json, None)
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let config = self.build_http_config();
        let url = crate::utils::url::join_url(&self.base_url(), &format!("files/{file_id}"));
        let response = execute_delete_request(&config, &url, None).await?;
        Ok(map_xai_file_delete_response(&response.json, &file_id))
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let config = self.build_http_config();
        let url =
            crate::utils::url::join_url(&self.base_url(), &format!("files/{file_id}/content"));
        let response = execute_get_binary(&config, &url, None).await?;
        Ok(response.bytes)
    }
}

fn normalize_list_order(order: &str) -> Option<&'static str> {
    let normalized = order.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" => None,
        "asc" | "ascending" => Some("ASCENDING"),
        "desc" | "descending" => Some("DESCENDING"),
        _ => None,
    }
}

fn get_string_field(value: &serde_json::Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match value.get(*key) {
        Some(serde_json::Value::String(inner)) if !inner.trim().is_empty() => Some(inner.clone()),
        Some(other) if other.is_number() || other.is_boolean() => Some(other.to_string()),
        _ => None,
    })
}

fn get_u64_field(value: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| match value.get(*key) {
        Some(serde_json::Value::Number(inner)) => inner.as_u64(),
        Some(serde_json::Value::String(inner)) => inner.trim().parse::<u64>().ok(),
        _ => None,
    })
}

fn get_timestamp_field(value: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| match value.get(*key) {
        Some(serde_json::Value::Number(inner)) => inner.as_u64(),
        Some(serde_json::Value::String(inner)) => inner.trim().parse::<u64>().ok().or_else(|| {
            chrono::DateTime::parse_from_rfc3339(inner)
                .ok()
                .and_then(|timestamp| u64::try_from(timestamp.timestamp()).ok())
        }),
        _ => None,
    })
}

fn get_bool_field(value: &serde_json::Value, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| match value.get(*key) {
        Some(serde_json::Value::Bool(inner)) => Some(*inner),
        Some(serde_json::Value::String(inner)) => {
            match inner.trim().to_ascii_lowercase().as_str() {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            }
        }
        _ => None,
    })
}

fn map_xai_file_object(
    value: &serde_json::Value,
    request: Option<&FileUploadRequest>,
) -> Result<FileObject, LlmError> {
    let id = get_string_field(value, &["id", "file_id"]).ok_or_else(|| {
        LlmError::ParseError("Failed to parse xAI file response: missing file id".to_string())
    })?;

    let filename = get_string_field(value, &["filename", "name"]);

    let bytes = get_u64_field(value, &["bytes", "size_bytes"])
        .or_else(|| request.map(|request| request.content.len() as u64))
        .unwrap_or(0);

    let created_at = get_timestamp_field(value, &["created_at"]).unwrap_or(0);

    let purpose = get_string_field(value, &["purpose"])
        .or_else(|| request.map(|request| request.purpose.clone()))
        .unwrap_or_default();

    let status = get_string_field(value, &["status", "processing_status"])
        .unwrap_or_else(|| "uploaded".to_string());

    let mime_type = get_string_field(value, &["mime_type", "content_type"]);

    let known_fields = HashSet::from([
        "id",
        "file_id",
        "object",
        "bytes",
        "size_bytes",
        "created_at",
        "filename",
        "name",
        "purpose",
        "status",
        "processing_status",
        "mime_type",
        "content_type",
    ]);

    let mut metadata = value
        .as_object()
        .map(|object| {
            object
                .iter()
                .filter(|(key, _)| !known_fields.contains(key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    if let Some(filename) = &filename {
        metadata.insert("filename".to_string(), serde_json::json!(filename));
    }
    if let Some(bytes) = get_u64_field(value, &["bytes", "size_bytes"]) {
        metadata.insert("bytes".to_string(), serde_json::json!(bytes));
    }
    if let Some(created_at) = get_timestamp_field(value, &["created_at"]) {
        metadata.insert("createdAt".to_string(), serde_json::json!(created_at));
    }

    Ok(FileObject {
        id,
        filename,
        bytes,
        created_at,
        purpose,
        status,
        mime_type,
        metadata,
    })
}

fn map_xai_file_list_response(value: &serde_json::Value) -> Result<FileListResponse, LlmError> {
    let files_value = value
        .get("data")
        .or_else(|| value.get("files"))
        .or_else(|| value.as_array().map(|_| value))
        .ok_or_else(|| {
            LlmError::ParseError(
                "Failed to parse xAI file list response: missing files array".to_string(),
            )
        })?;

    let files_array = files_value.as_array().ok_or_else(|| {
        LlmError::ParseError(
            "Failed to parse xAI file list response: files is not an array".to_string(),
        )
    })?;

    let mut files = Vec::with_capacity(files_array.len());
    for file in files_array {
        files.push(map_xai_file_object(file, None)?);
    }

    let next_cursor = get_string_field(value, &["next_cursor", "next_token", "after"]);
    let has_more = get_bool_field(value, &["has_more"]).unwrap_or_else(|| next_cursor.is_some());

    Ok(FileListResponse {
        files,
        has_more,
        next_cursor,
    })
}

fn map_xai_file_delete_response(
    value: &serde_json::Value,
    requested_id: &str,
) -> FileDeleteResponse {
    let id =
        get_string_field(value, &["id", "file_id"]).unwrap_or_else(|| requested_id.to_string());
    let deleted = get_bool_field(value, &["deleted"]).unwrap_or(true);

    FileDeleteResponse { id, deleted }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use std::collections::VecDeque;
    use std::sync::Mutex;

    #[derive(Clone)]
    struct CaptureTransport {
        multipart_requests: Arc<Mutex<Vec<HttpTransportMultipartRequest>>>,
        responses: Arc<Mutex<VecDeque<HttpTransportResponse>>>,
    }

    impl CaptureTransport {
        fn new(responses: Vec<HttpTransportResponse>) -> Self {
            Self {
                multipart_requests: Arc::new(Mutex::new(Vec::new())),
                responses: Arc::new(Mutex::new(responses.into_iter().collect())),
            }
        }

        fn take_multipart_requests(&self) -> Vec<HttpTransportMultipartRequest> {
            std::mem::take(&mut *self.multipart_requests.lock().expect("multipart lock"))
        }
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "json transport should not be used in xai files tests".to_string(),
            ))
        }

        async fn execute_multipart(
            &self,
            request: HttpTransportMultipartRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.multipart_requests
                .lock()
                .expect("multipart lock")
                .push(request);
            self.responses
                .lock()
                .expect("responses lock")
                .pop_front()
                .ok_or_else(|| LlmError::HttpError("missing multipart response".to_string()))
        }
    }

    fn make_json_response(body: serde_json::Value) -> HttpTransportResponse {
        HttpTransportResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: serde_json::to_vec(&body).expect("serialize response body"),
        }
    }

    fn make_test_files(transport: Arc<dyn HttpTransport>) -> XaiFiles {
        XaiFiles::new(
            ProviderContext::new(
                PROVIDER_ID,
                "https://api.x.ai/v1",
                Some("test-key".to_string()),
                HashMap::new(),
            ),
            reqwest::Client::new(),
            Some(transport),
            Vec::new(),
            None,
        )
    }

    #[tokio::test]
    async fn upload_file_uses_multipart_and_forwards_team_id_and_file_path() {
        let transport = CaptureTransport::new(vec![make_json_response(serde_json::json!({
            "id": "file-123",
            "bytes": 3,
            "created_at": 1,
            "filename": "hello.txt",
            "status": "uploaded"
        }))]);

        let files = make_test_files(Arc::new(transport.clone()));
        let mut provider_options = crate::types::ProviderOptionsMap::default();
        provider_options.insert(
            "xai",
            serde_json::json!({
                "teamId": "team-123",
                "filePath": "/uploads/hello.txt"
            }),
        );
        let request = FileUploadRequest {
            content: b"hey".to_vec(),
            filename: Some("hello.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
            purpose: "assistants".to_string(),
            metadata: HashMap::new(),
            provider_options,
            http_config: None,
        };

        let result = files.upload_file(request).await.expect("upload result");
        assert_eq!(result.id, "file-123");
        assert_eq!(result.filename.as_deref(), Some("hello.txt"));
        assert_eq!(
            result.metadata.get("filename"),
            Some(&serde_json::json!("hello.txt"))
        );
        assert_eq!(result.metadata.get("bytes"), Some(&serde_json::json!(3)));
        assert_eq!(
            result.metadata.get("createdAt"),
            Some(&serde_json::json!(1))
        );

        let requests = transport.take_multipart_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url, "https://api.x.ai/v1/files");
        let body = String::from_utf8_lossy(&requests[0].body);
        assert!(body.contains("name=\"team_id\""));
        assert!(body.contains("team-123"));
        assert!(body.contains("name=\"file_path\""));
        assert!(body.contains("/uploads/hello.txt"));
        assert!(body.contains("name=\"file\"; filename=\"hello.txt\""));
    }

    #[test]
    fn file_object_parser_accepts_docs_shape() {
        let file = map_xai_file_object(
            &serde_json::json!({
                "file_id": "file-123",
                "name": "hello.txt",
                "size_bytes": 12,
                "created_at": "2026-04-15T12:00:00Z",
                "processing_status": "completed",
                "content_type": "text/plain",
                "file_path": "/uploads/hello.txt"
            }),
            None,
        )
        .expect("parsed xai docs-shaped file");

        assert_eq!(file.id, "file-123");
        assert_eq!(file.filename.as_deref(), Some("hello.txt"));
        assert_eq!(file.bytes, 12);
        assert_eq!(file.status, "completed");
        assert_eq!(file.mime_type.as_deref(), Some("text/plain"));
        assert_eq!(
            file.metadata.get("file_path"),
            Some(&serde_json::json!("/uploads/hello.txt"))
        );
    }
}
