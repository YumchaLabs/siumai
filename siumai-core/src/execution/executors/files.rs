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

/// Builder for creating HttpFilesExecutor instances
pub struct FilesExecutorBuilder {
    provider_id: String,
    http_client: reqwest::Client,
    spec: Option<Arc<dyn crate::core::ProviderSpec>>,
    context: Option<crate::core::ProviderContext>,
    transformer: Option<Arc<dyn crate::execution::transformers::files::FilesTransformer>>,
    policy: crate::execution::ExecutionPolicy,
}

impl FilesExecutorBuilder {
    pub fn new(provider_id: impl Into<String>, http_client: reqwest::Client) -> Self {
        Self {
            provider_id: provider_id.into(),
            http_client,
            spec: None,
            context: None,
            transformer: None,
            policy: crate::execution::ExecutionPolicy::new(),
        }
    }

    pub fn with_spec(mut self, spec: Arc<dyn crate::core::ProviderSpec>) -> Self {
        self.spec = Some(spec);
        self
    }

    pub fn with_context(mut self, context: crate::core::ProviderContext) -> Self {
        self.context = Some(context);
        self
    }

    pub fn with_transformer(
        mut self,
        transformer: Arc<dyn crate::execution::transformers::files::FilesTransformer>,
    ) -> Self {
        self.transformer = Some(transformer);
        self
    }

    pub fn with_before_send(mut self, hook: crate::execution::executors::BeforeSendHook) -> Self {
        self.policy.before_send = Some(hook);
        self
    }

    pub fn with_interceptors(
        mut self,
        interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    ) -> Self {
        self.policy.interceptors = interceptors;
        self
    }

    pub fn with_retry_options(mut self, retry_options: crate::retry_api::RetryOptions) -> Self {
        self.policy.retry_options = Some(retry_options);
        self
    }

    pub fn build(self) -> Arc<HttpFilesExecutor> {
        let spec = self.spec.expect("provider_spec is required");
        let context = self.context.expect("provider_context is required");
        let transformer = match self.transformer {
            Some(t) => t,
            None => spec.choose_files_transformer(&context).transformer,
        };
        Arc::new(HttpFilesExecutor {
            provider_id: self.provider_id,
            http_client: self.http_client,
            transformer,
            provider_spec: spec,
            provider_context: context,
            policy: self.policy,
        })
    }
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
        let endpoint = self.transformer.upload_endpoint(&req);
        let url = crate::utils::url::join_url(&base_url, &endpoint);

        let provider_id = self.provider_id.clone();
        let http_client = self.http_client.clone();
        let transformer = self.transformer.clone();
        let provider_spec = self.provider_spec.clone();
        let provider_context = self.provider_context.clone();
        let interceptors = self.policy.interceptors.clone();
        let before_send = self.policy.before_send.clone();
        let retry_wrapper_opts = self.policy.retry_options.clone();
        let retry_options_for_http = self.policy.retry_options.clone();

        let req_for_attempts = req;

        let run_once = move || {
            let provider_id = provider_id.clone();
            let http_client = http_client.clone();
            let transformer = transformer.clone();
            let provider_spec = provider_spec.clone();
            let provider_context = provider_context.clone();
            let interceptors = interceptors.clone();
            let before_send = before_send.clone();
            let retry_options_for_http = retry_options_for_http.clone();
            let url = url.clone();
            let req = req_for_attempts.clone();

            async move {
                // Build execution config for common HTTP layer (401 rebuild).
                let config = crate::execution::executors::common::HttpExecutionConfig {
                    provider_id,
                    http_client,
                    provider_spec,
                    provider_context,
                    interceptors,
                    retry_options: retry_options_for_http.clone(),
                };

                let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
                let body = transformer.build_upload_body(&req)?;

                let result = match body {
                    FilesHttpBody::Json(mut json) => {
                        // Apply before_send if present
                        if let Some(cb) = &before_send {
                            json = cb(&json)?;
                        }
                        crate::execution::executors::common::execute_json_request(
                            &config,
                            &url,
                            crate::execution::executors::common::HttpBody::Json(json),
                            per_request_headers,
                            false,
                        )
                        .await?
                    }
                    FilesHttpBody::Multipart(_) => {
                        let req_clone = req.clone();
                        let transformer_for_form = transformer.clone();
                        crate::execution::executors::common::execute_multipart_request(
                            &config,
                            &url,
                            move || {
                                transformer_for_form.build_upload_body(&req_clone).and_then(
                                    |body| match body {
                                        FilesHttpBody::Multipart(form) => Ok(form),
                                        _ => Err(LlmError::InvalidParameter(
                                            "Expected multipart body".into(),
                                        )),
                                    },
                                )
                            },
                            per_request_headers,
                        )
                        .await?
                    }
                };

                transformer.transform_file_object(&result.json)
            }
        };

        if let Some(opts) = retry_wrapper_opts {
            crate::retry_api::retry_with(run_once, opts).await
        } else {
            run_once().await
        }
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
        let url = crate::utils::url::join_url(&base_url, &endpoint);

        let provider_id = self.provider_id.clone();
        let http_client = self.http_client.clone();
        let transformer = self.transformer.clone();
        let provider_spec = self.provider_spec.clone();
        let provider_context = self.provider_context.clone();
        let interceptors = self.policy.interceptors.clone();
        let retry_wrapper_opts = self.policy.retry_options.clone();
        let retry_options_for_http = self.policy.retry_options.clone();

        let query_for_attempts = query;

        let run_once = move || {
            let provider_id = provider_id.clone();
            let http_client = http_client.clone();
            let transformer = transformer.clone();
            let provider_spec = provider_spec.clone();
            let provider_context = provider_context.clone();
            let interceptors = interceptors.clone();
            let retry_options_for_http = retry_options_for_http.clone();
            let url = url.clone();
            let query = query_for_attempts.clone();

            async move {
                let config = crate::execution::executors::common::HttpExecutionConfig {
                    provider_id,
                    http_client,
                    provider_spec,
                    provider_context,
                    interceptors,
                    retry_options: retry_options_for_http.clone(),
                };

                let per_request_headers = query
                    .as_ref()
                    .and_then(|q| q.http_config.as_ref())
                    .map(|hc| &hc.headers);

                let result = crate::execution::executors::http_request::execute_get_request(
                    &config,
                    &url,
                    per_request_headers,
                )
                .await?;

                transformer.transform_list_response(&result.json)
            }
        };

        if let Some(opts) = retry_wrapper_opts {
            crate::retry_api::retry_with(run_once, opts).await
        } else {
            run_once().await
        }
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
        let url = crate::utils::url::join_url(&base_url, &endpoint);

        let provider_id = self.provider_id.clone();
        let http_client = self.http_client.clone();
        let transformer = self.transformer.clone();
        let provider_spec = self.provider_spec.clone();
        let provider_context = self.provider_context.clone();
        let interceptors = self.policy.interceptors.clone();
        let retry_wrapper_opts = self.policy.retry_options.clone();
        let retry_options_for_http = self.policy.retry_options.clone();

        let run_once = move || {
            let provider_id = provider_id.clone();
            let http_client = http_client.clone();
            let transformer = transformer.clone();
            let provider_spec = provider_spec.clone();
            let provider_context = provider_context.clone();
            let interceptors = interceptors.clone();
            let retry_options_for_http = retry_options_for_http.clone();
            let url = url.clone();

            async move {
                let config = crate::execution::executors::common::HttpExecutionConfig {
                    provider_id,
                    http_client,
                    provider_spec,
                    provider_context,
                    interceptors,
                    retry_options: retry_options_for_http.clone(),
                };

                let result = crate::execution::executors::http_request::execute_get_request(
                    &config, &url, None,
                )
                .await?;

                transformer.transform_file_object(&result.json)
            }
        };

        if let Some(opts) = retry_wrapper_opts {
            crate::retry_api::retry_with(run_once, opts).await
        } else {
            run_once().await
        }
    }

    async fn delete(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File delete is not supported by this provider".to_string(),
            ));
        }
        let id = file_id.trim_start_matches("files/").to_string();
        // 1. Get URL from transformer
        let endpoint = self.transformer.delete_endpoint(&file_id);
        let base_url = self.provider_spec.files_base_url(&self.provider_context);
        let url = crate::utils::url::join_url(&base_url, &endpoint);

        let provider_id = self.provider_id.clone();
        let http_client = self.http_client.clone();
        let provider_spec = self.provider_spec.clone();
        let provider_context = self.provider_context.clone();
        let interceptors = self.policy.interceptors.clone();
        let retry_wrapper_opts = self.policy.retry_options.clone();
        let retry_options_for_http = self.policy.retry_options.clone();

        let run_once = move || {
            let provider_id = provider_id.clone();
            let http_client = http_client.clone();
            let provider_spec = provider_spec.clone();
            let provider_context = provider_context.clone();
            let interceptors = interceptors.clone();
            let retry_options_for_http = retry_options_for_http.clone();
            let url = url.clone();
            let id = id.clone();

            async move {
                let config = crate::execution::executors::common::HttpExecutionConfig {
                    provider_id,
                    http_client,
                    provider_spec,
                    provider_context,
                    interceptors,
                    retry_options: retry_options_for_http.clone(),
                };

                crate::execution::executors::http_request::execute_delete_request(
                    &config, &url, None,
                )
                .await?;

                Ok(FileDeleteResponse { id, deleted: true })
            }
        };

        if let Some(opts) = retry_wrapper_opts {
            crate::retry_api::retry_with(run_once, opts).await
        } else {
            run_once().await
        }
    }

    async fn get_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("file_management") {
            return Err(LlmError::UnsupportedOperation(
                "File content download is not supported by this provider".to_string(),
            ));
        }
        let provider_id = self.provider_id.clone();
        let http_client = self.http_client.clone();
        let transformer = self.transformer.clone();
        let provider_spec = self.provider_spec.clone();
        let provider_context = self.provider_context.clone();
        let interceptors = self.policy.interceptors.clone();
        let retry_wrapper_opts = self.policy.retry_options.clone();
        let retry_options_for_http = self.policy.retry_options.clone();

        let file_id_for_attempts = file_id;

        let run_once = move || {
            let provider_id = provider_id.clone();
            let http_client = http_client.clone();
            let transformer = transformer.clone();
            let provider_spec = provider_spec.clone();
            let provider_context = provider_context.clone();
            let interceptors = interceptors.clone();
            let retry_options_for_http = retry_options_for_http.clone();
            let file_id = file_id_for_attempts.clone();

            async move {
                let config = crate::execution::executors::common::HttpExecutionConfig {
                    provider_id,
                    http_client,
                    provider_spec: provider_spec.clone(),
                    provider_context: provider_context.clone(),
                    interceptors,
                    retry_options: retry_options_for_http.clone(),
                };

                // Determine URL (prefer API endpoint if provided; otherwise fall back to URL from file object)
                let url = if let Some(ep) = transformer.content_endpoint(&file_id) {
                    let base_url = provider_spec.files_base_url(&provider_context);
                    crate::utils::url::join_url(&base_url, &ep)
                } else {
                    let endpoint = transformer.retrieve_endpoint(&file_id);
                    let base_url = provider_spec.files_base_url(&provider_context);
                    let retrieve_url = crate::utils::url::join_url(&base_url, &endpoint);
                    let result = crate::execution::executors::http_request::execute_get_request(
                        &config,
                        &retrieve_url,
                        None,
                    )
                    .await?;
                    let file = transformer.transform_file_object(&result.json)?;
                    transformer
                        .content_url_from_file_object(&file)
                        .ok_or_else(|| {
                            LlmError::UnsupportedOperation(
                                "File download URI not available".to_string(),
                            )
                        })?
                };

                let result = crate::execution::executors::http_request::execute_get_binary(
                    &config, &url, None,
                )
                .await?;

                Ok(result.bytes)
            }
        };

        if let Some(opts) = retry_wrapper_opts {
            crate::retry_api::retry_with(run_once, opts).await
        } else {
            run_once().await
        }
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
