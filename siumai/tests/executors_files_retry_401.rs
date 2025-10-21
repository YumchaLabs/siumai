use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use serde_json::json;
use siumai::error::LlmError;
use siumai::executors::files::{FilesExecutor, HttpFilesExecutor};
use siumai::transformers::files::{FilesHttpBody, FilesTransformer};
use siumai::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};
use wiremock::matchers::{header, method, path, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct TestFilesTransformer;
impl FilesTransformer for TestFilesTransformer {
    fn provider_id(&self) -> &str {
        "test"
    }

    fn build_upload_body(&self, req: &FileUploadRequest) -> Result<FilesHttpBody, LlmError> {
        let body = json!({
            "filename": req.filename,
            "purpose": req.purpose,
            "bytes": req.content.len(),
        });
        Ok(FilesHttpBody::Json(body))
    }

    fn list_endpoint(&self, _query: &Option<FileListQuery>) -> String {
        "/files".to_string()
    }
    fn retrieve_endpoint(&self, file_id: &str) -> String {
        format!("/files/{}", file_id)
    }
    fn delete_endpoint(&self, file_id: &str) -> String {
        format!("/files/{}", file_id)
    }

    fn transform_file_object(&self, raw: &serde_json::Value) -> Result<FileObject, LlmError> {
        Ok(FileObject {
            id: raw["id"].as_str().unwrap_or("f_1").to_string(),
            filename: raw["filename"].as_str().unwrap_or("test.txt").to_string(),
            bytes: raw["bytes"].as_u64().unwrap_or(0),
            created_at: raw["created_at"].as_u64().unwrap_or(0),
            purpose: raw["purpose"].as_str().unwrap_or("test").to_string(),
            status: raw["status"].as_str().unwrap_or("processed").to_string(),
            mime_type: Some("text/plain".to_string()),
            metadata: Default::default(),
        })
    }

    fn transform_list_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<FileListResponse, LlmError> {
        let files = raw["files"].as_array().cloned().unwrap_or_default();
        let files = files
            .into_iter()
            .map(|v| self.transform_file_object(&v).unwrap())
            .collect();
        Ok(FileListResponse {
            files,
            has_more: false,
            next_cursor: None,
        })
    }

    fn content_endpoint(&self, file_id: &str) -> Option<String> {
        Some(format!("/files/{}/content", file_id))
    }
}

fn headers_builder_factory(
    counter: Arc<AtomicUsize>,
) -> impl Fn() -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, LlmError>> + Send>,
> + Send
+ Sync
+ 'static {
    move || {
        let counter = counter.clone();
        Box::pin(async move {
            use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
            let n = counter.fetch_add(1, Ordering::SeqCst);
            let token = if n == 0 { "bad" } else { "ok" };
            let mut h = HeaderMap::new();
            h.insert(
                HeaderName::from_static("authorization"),
                HeaderValue::from_str(&format!("Bearer {}", token)).unwrap(),
            );
            h.insert(
                HeaderName::from_static("content-type"),
                HeaderValue::from_static("application/json"),
            );
            Ok(h)
        })
    }
}

#[tokio::test]
async fn files_executor_retries_on_401_list_retrieve_delete_upload_content() {
    let server = MockServer::start().await;

    // 200 when Authorization = Bearer ok, else 401
    let unauthorized = ResponseTemplate::new(401)
        .set_body_json(json!({"error": {"code":401,"message":"unauthorized"}}));
    let ok_list = ResponseTemplate::new(200).set_body_json(json!({
        "files": [{"id":"f1","filename":"a.txt","bytes":1,"created_at":1,"purpose":"test","status":"processed"}]
    }));
    let ok_retrieve = ResponseTemplate::new(200).set_body_json(json!({
        "id":"f1","filename":"a.txt","bytes":1,"created_at":1,"purpose":"test","status":"processed"
    }));
    let ok_upload = ResponseTemplate::new(200).set_body_json(json!({
        "id":"f2","filename":"b.txt","bytes":2,"created_at":2,"purpose":"test","status":"processed"
    }));
    let ok_content = ResponseTemplate::new(200).set_body_string("HELLO");

    // List
    Mock::given(method("GET"))
        .and(path("/files"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ok_list.clone())
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/files"))
        .respond_with(unauthorized.clone())
        .mount(&server)
        .await;

    // Retrieve
    Mock::given(method("GET"))
        .and(path("/files/f1"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ok_retrieve.clone())
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/files/f1"))
        .respond_with(unauthorized.clone())
        .mount(&server)
        .await;

    // Delete (200/204 both fine; we return success regardless body)
    Mock::given(method("DELETE"))
        .and(path("/files/f1"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/files/f1"))
        .respond_with(unauthorized.clone())
        .mount(&server)
        .await;

    // Upload
    Mock::given(method("POST"))
        .and(path("/files"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ok_upload.clone())
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/files"))
        .respond_with(unauthorized.clone())
        .mount(&server)
        .await;

    // Content
    Mock::given(method("GET"))
        .and(path("/files/f1/content"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ok_content.clone())
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path_regex("^/files/.+/content$"))
        .respond_with(unauthorized.clone())
        .mount(&server)
        .await;

    let counter = Arc::new(AtomicUsize::new(0));
    let files = HttpFilesExecutor {
        provider_id: "test".to_string(),
        http_client: reqwest::Client::new(),
        transformer: Arc::new(TestFilesTransformer),
        build_base_url: Box::new({
            let base = server.uri();
            move || base.clone()
        }),
        build_headers: Box::new(headers_builder_factory(counter.clone())),
    };

    // list
    let listed = files.list(None).await.unwrap();
    assert_eq!(listed.files.len(), 1);

    // retrieve
    let obj = files.retrieve("f1".to_string()).await.unwrap();
    assert_eq!(obj.id, "f1");

    // delete
    let FileDeleteResponse { id, deleted } = files.delete("f1".to_string()).await.unwrap();
    assert_eq!(id, "f1");
    assert!(deleted);

    // upload
    let up = files
        .upload(FileUploadRequest {
            content: vec![1, 2],
            filename: "b.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            purpose: "test".to_string(),
            metadata: Default::default(),
        })
        .await
        .unwrap();
    assert_eq!(up.id, "f2");

    // get_content
    let bytes = files.get_content("f1".to_string()).await.unwrap();
    assert_eq!(bytes, b"HELLO");
}
