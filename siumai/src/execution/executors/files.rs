//! Files executor traits

use crate::error::LlmError;
use crate::execution::transformers::files::{FilesHttpBody, FilesTransformer};
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};
use std::sync::Arc;

#[async_trait::async_trait]
pub trait FilesExecutor: Send + Sync {
    async fn upload(&self, req: FileUploadRequest) -> Result<FileObject, LlmError>;
    async fn list(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError>;
    async fn retrieve(&self, file_id: String) -> Result<FileObject, LlmError>;
    async fn delete(&self, file_id: String) -> Result<FileDeleteResponse, LlmError>;
    async fn get_content(&self, file_id: String) -> Result<Vec<u8>, LlmError>;
}

/// Generic HTTP-based Files executor
pub struct HttpFilesExecutor {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub transformer: Arc<dyn FilesTransformer>,
    pub provider_spec: Arc<dyn crate::core::ProviderSpec>,
    pub provider_context: crate::core::ProviderContext,
    /// Execution policy
    pub policy: crate::execution::ExecutionPolicy,
}

#[async_trait::async_trait]
impl FilesExecutor for HttpFilesExecutor {
    async fn upload(&self, req: FileUploadRequest) -> Result<FileObject, LlmError> {
        // Capability guard
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File management is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, self.transformer.upload_endpoint(&req));

        // 2. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 3. Transform request and execute based on body type
        let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
        let body = self.transformer.build_upload_body(&req)?;
        let result = match body {
            FilesHttpBody::Json(mut json) => {
                // Apply before_send if present
                if let Some(cb) = &self.policy.before_send {
                    json = cb(&json)?;
                }
                // Use JSON request path
                crate::execution::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(json),
                    per_request_headers,
                    false, // stream = false
                )
                .await?
            }
            FilesHttpBody::Multipart(_) => {
                // Use multipart request path
                let req_clone = req.clone();
                crate::execution::executors::common::execute_multipart_request(
                    &config,
                    &url,
                    || {
                        self.transformer
                            .build_upload_body(&req_clone)
                            .and_then(|body| match body {
                                FilesHttpBody::Multipart(form) => Ok(form),
                                _ => Err(LlmError::InvalidParameter(
                                    "Expected multipart body".into(),
                                )),
                            })
                    },
                    per_request_headers,
                )
                .await?
            }
        };

        // 4. Transform response
        self.transformer.transform_file_object(&result.json)
    }

    async fn list(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File listing is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL from transformer
        let endpoint = self.transformer.list_endpoint(&query);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, endpoint);

        // 2. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 3. Extract per-request headers from query
        let per_request_headers = query
            .as_ref()
            .and_then(|q| q.http_config.as_ref())
            .map(|hc| &hc.headers);

        // 4. Execute GET request using common HTTP layer
        let result = crate::execution::executors::common::execute_get_request(
            &config,
            &url,
            per_request_headers,
        )
        .await?;

        // 5. Transform response
        self.transformer.transform_list_response(&result.json)
    }

    async fn retrieve(&self, file_id: String) -> Result<FileObject, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File retrieve is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL from transformer
        let endpoint = self.transformer.retrieve_endpoint(&file_id);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, endpoint);

        // 2. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 3. Execute GET request using common HTTP layer
        let result = crate::execution::executors::common::execute_get_request(
            &config, &url, None, // No per-request headers for retrieve
        )
        .await?;

        // 4. Transform response
        self.transformer.transform_file_object(&result.json)
    }

    async fn delete(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File delete is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL from transformer
        let endpoint = self.transformer.delete_endpoint(&file_id);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, endpoint);

        // 2. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 3. Execute DELETE request using common HTTP layer
        let _result = crate::execution::executors::common::execute_delete_request(
            &config, &url, None, // No per-request headers for delete
        )
        .await?;

        // 4. Return success response
        // Some providers may return an empty body or a small JSON; we just acknowledge success
        let id = file_id.trim_start_matches("files/").to_string();
        Ok(FileDeleteResponse { id, deleted: true })
    }

    async fn get_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File content download is not supported by this provider".to_string(),
            ));
        }
        // 1. Determine URL (prefer API endpoint if provided; otherwise fall back to URL from file object)
        let mut maybe_endpoint = self.transformer.content_endpoint(&file_id);
        let url = if let Some(ep) = maybe_endpoint.take() {
            let base_url = self.provider_spec.files_base_url(&self.provider_context);
            format!("{}{}", base_url, ep)
        } else {
            let file = self.retrieve(file_id.clone()).await?;
            self.transformer
                .content_url_from_file_object(&file)
                .ok_or_else(|| {
                    LlmError::UnsupportedOperation("File download URI not available".to_string())
                })?
        };

        // 2. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 3. Execute GET request for binary content using common HTTP layer
        let result = crate::execution::executors::common::execute_get_binary(
            &config, &url, None, // No per-request headers for get_content
        )
        .await?;

        // 4. Return binary content
        Ok(result.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
    use crate::types::{FileObject, HttpConfig};
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    use std::sync::{Arc, Mutex};

    // Minimal FilesTransformer for tests
    struct TestFilesTx {
        json: bool,
    }
    impl crate::execution::transformers::files::FilesTransformer for TestFilesTx {
        fn provider_id(&self) -> &str {
            "test"
        }
        fn build_upload_body(
            &self,
            _req: &crate::types::FileUploadRequest,
        ) -> Result<FilesHttpBody, LlmError> {
            if self.json {
                Ok(FilesHttpBody::Json(serde_json::json!({"name":"orig"})))
            } else {
                let form = reqwest::multipart::Form::new()
                    .text("field", "value")
                    .text("filename", "a.txt");
                Ok(FilesHttpBody::Multipart(form))
            }
        }
        fn list_endpoint(&self, _q: &Option<crate::types::FileListQuery>) -> String {
            "/files".into()
        }
        fn retrieve_endpoint(&self, _id: &str) -> String {
            "/files/x".into()
        }
        fn delete_endpoint(&self, _id: &str) -> String {
            "/files/x".into()
        }
        fn transform_file_object(&self, raw: &serde_json::Value) -> Result<FileObject, LlmError> {
            Ok(FileObject {
                id: raw["id"].as_str().unwrap_or("x").to_string(),
                filename: "a.txt".into(),
                bytes: 0,
                created_at: 0,
                purpose: "assistants".into(),
                status: "active".into(),
                mime_type: None,
                metadata: Default::default(),
            })
        }
        fn transform_list_response(
            &self,
            _raw: &serde_json::Value,
        ) -> Result<crate::types::FileListResponse, LlmError> {
            Ok(crate::types::FileListResponse {
                files: vec![],
                has_more: false,
                next_cursor: None,
            })
        }
    }

    // Minimal ProviderSpec for files testing
    #[derive(Clone, Copy)]
    struct TestSpec;
    impl crate::core::ProviderSpec for TestSpec {
        fn id(&self) -> &'static str {
            "test"
        }
        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            crate::traits::ProviderCapabilities::new().with_file_management()
        }
        fn build_headers(
            &self,
            _ctx: &crate::core::ProviderContext,
        ) -> Result<HeaderMap, LlmError> {
            let mut h = HeaderMap::new();
            h.insert(
                HeaderName::from_static("x-base"),
                HeaderValue::from_static("B"),
            );
            Ok(h)
        }
        fn chat_url(
            &self,
            _stream: bool,
            _req: &crate::types::ChatRequest,
            ctx: &crate::core::ProviderContext,
        ) -> String {
            format!("{}/never", ctx.base_url)
        }
        fn choose_chat_transformers(
            &self,
            _req: &crate::types::ChatRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> crate::core::ChatTransformers {
            panic!("not used in files tests")
        }
    }

    // Interceptor to capture headers and abort
    struct CaptureHeaders {
        seen: Arc<Mutex<Option<HeaderMap>>>,
    }
    impl HttpInterceptor for CaptureHeaders {
        fn on_before_send(
            &self,
            _ctx: &HttpRequestContext,
            _rb: reqwest::RequestBuilder,
            _body: &serde_json::Value,
            headers: &HeaderMap,
        ) -> Result<reqwest::RequestBuilder, LlmError> {
            *self.seen.lock().unwrap() = Some(headers.clone());
            Err(LlmError::InvalidParameter("abort".into()))
        }
    }

    #[tokio::test]
    async fn json_upload_applies_before_send_and_aborts() {
        let http = reqwest::Client::new();
        let spec = Arc::new(TestSpec);
        let ctx =
            crate::core::ProviderContext::new("test", "http://127.0.0.1", None, Default::default());
        let tx = Arc::new(TestFilesTx { json: true });

        let seen_name = Arc::new(Mutex::new(None::<String>));
        let seen_clone = seen_name.clone();
        let before: crate::execution::executors::BeforeSendHook =
            Arc::new(move |body: &serde_json::Value| {
                let name = body.get("name").and_then(|v| v.as_str()).unwrap_or("");
                *seen_clone.lock().unwrap() = Some(name.to_string());
                Err(LlmError::InvalidParameter("abort".into()))
            });

        let exec = HttpFilesExecutor {
            provider_id: "test".into(),
            http_client: http,
            transformer: tx,
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new().with_before_send(before),
        };

        let req = crate::types::FileUploadRequest {
            content: vec![1, 2, 3],
            filename: "a.txt".into(),
            mime_type: Some("text/plain".into()),
            purpose: "assistants".into(),
            metadata: Default::default(),
            http_config: None,
        };
        let err = exec.upload(req).await.unwrap_err();
        match err {
            LlmError::InvalidParameter(msg) => assert_eq!(msg, "abort"),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(seen_name.lock().unwrap().clone().unwrap(), "orig");
    }

    #[tokio::test]
    async fn multipart_upload_merges_headers_and_aborts() {
        let http = reqwest::Client::new();
        let spec = Arc::new(TestSpec);
        let ctx =
            crate::core::ProviderContext::new("test", "http://127.0.0.1", None, Default::default());
        let tx = Arc::new(TestFilesTx { json: false });

        let seen = Arc::new(Mutex::new(None::<HeaderMap>));
        let interceptor = Arc::new(CaptureHeaders { seen: seen.clone() });

        let exec = HttpFilesExecutor {
            provider_id: "test".into(),
            http_client: http,
            transformer: tx,
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new().with_interceptors(vec![interceptor]),
        };

        let mut hc = HttpConfig::default();
        hc.headers.insert("x-req".into(), "R".into());
        let req = crate::types::FileUploadRequest {
            content: vec![1, 2, 3],
            filename: "a.txt".into(),
            mime_type: Some("text/plain".into()),
            purpose: "assistants".into(),
            metadata: Default::default(),
            http_config: Some(hc),
        };

        let err = exec.upload(req).await.unwrap_err();
        match err {
            LlmError::InvalidParameter(msg) => assert_eq!(msg, "abort"),
            other => panic!("unexpected {other:?}"),
        }
        let headers = seen.lock().unwrap().clone().unwrap();
        assert_eq!(
            headers.get("x-base").unwrap(),
            &HeaderValue::from_static("B")
        );
        assert_eq!(
            headers.get("x-req").unwrap(),
            &HeaderValue::from_static("R")
        );
    }
}
