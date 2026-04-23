use base64::Engine;
use siumai::files::{self, UploadFileOptions};
use siumai::prelude::unified::*;
use siumai_core::client::LlmClient;
use siumai_core::traits::FileManagementCapability;
use siumai_core::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct MockFilesClient {
    provider_id: String,
    last_upload: Arc<Mutex<Option<FileUploadRequest>>>,
}

impl MockFilesClient {
    fn new(provider_id: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            last_upload: Arc::new(Mutex::new(None)),
        }
    }

    fn take_last_upload(&self) -> FileUploadRequest {
        self.last_upload
            .lock()
            .expect("lock upload")
            .take()
            .expect("captured upload request")
    }
}

#[async_trait::async_trait]
impl FileManagementCapability for MockFilesClient {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        *self.last_upload.lock().expect("lock upload") = Some(request.clone());
        let file_id_suffix = request.filename.as_deref().unwrap_or("generated");

        Ok(FileObject {
            id: format!("file_{file_id_suffix}"),
            filename: request.filename,
            bytes: request.content.len() as u64,
            created_at: 1,
            purpose: request.purpose,
            status: "uploaded".to_string(),
            mime_type: None,
            metadata: HashMap::from([("serverChecksum".to_string(), serde_json::json!("abc123"))]),
        })
    }

    async fn list_files(
        &self,
        _query: Option<FileListQuery>,
    ) -> Result<FileListResponse, LlmError> {
        unreachable!("not used in upload helper tests")
    }

    async fn retrieve_file(&self, _file_id: String) -> Result<FileObject, LlmError> {
        unreachable!("not used in upload helper tests")
    }

    async fn delete_file(&self, _file_id: String) -> Result<FileDeleteResponse, LlmError> {
        unreachable!("not used in upload helper tests")
    }

    async fn get_file_content(&self, _file_id: String) -> Result<Vec<u8>, LlmError> {
        unreachable!("not used in upload helper tests")
    }
}

impl LlmClient for MockFilesClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.provider_id.clone())
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["mock-model".to_string()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_file_management()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        Some(self)
    }
}

impl FileUploadProvider for MockFilesClient {
    fn upload_file_provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.provider_id.clone())
    }
}

#[tokio::test]
async fn upload_detects_media_type_and_builds_provider_reference() {
    let client = MockFilesClient::new("openai");
    let png_bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    let result = files::upload(&client, png_bytes, UploadFileOptions::new())
        .await
        .expect("upload result");

    let request = client.take_last_upload();
    assert_eq!(request.filename, None);
    assert_eq!(request.purpose, "assistants");
    assert_eq!(request.mime_type.as_deref(), Some("image/png"));

    assert_eq!(
        result.provider_reference.get("openai"),
        Some("file_generated")
    );
    assert_eq!(result.media_type, None);
    assert_eq!(result.filename, None);
    assert!(
        result
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("openai"))
            .and_then(|metadata| metadata.as_object())
            .is_some(),
        "expected provider metadata to keep the AI SDK provider->object root"
    );
    assert_eq!(
        result
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("openai"))
            .and_then(|metadata| metadata.get("serverChecksum")),
        Some(&serde_json::json!("abc123"))
    );
    assert!(
        result
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("openai"))
            .and_then(|metadata| metadata.get("createdAt"))
            .is_none(),
        "upload helper should not synthesize generic file fields into providerMetadata"
    );
}

#[tokio::test]
async fn upload_base64_text_falls_back_to_text_plain() {
    let client = MockFilesClient::new("openai");
    let base64 = base64::engine::general_purpose::STANDARD.encode("hello from siumai");

    let result = files::upload(
        &client,
        base64,
        UploadFileOptions::new().with_filename("note.txt"),
    )
    .await
    .expect("upload result");

    let request = client.take_last_upload();
    assert_eq!(request.filename.as_deref(), Some("note.txt"));
    assert_eq!(request.mime_type.as_deref(), Some("text/plain"));
    assert_eq!(result.media_type, None);
    assert_eq!(result.filename.as_deref(), Some("note.txt"));
}

#[tokio::test]
async fn upload_accepts_shared_data_content_directly() {
    let client = MockFilesClient::new("openai");

    let result = files::upload(
        &client,
        DataContent::binary(b"hello from shared data".to_vec()),
        UploadFileOptions::new().with_filename("shared.txt"),
    )
    .await
    .expect("upload result");

    let request = client.take_last_upload();
    assert_eq!(request.filename.as_deref(), Some("shared.txt"));
    assert_eq!(request.content, b"hello from shared data");
    assert_eq!(request.mime_type.as_deref(), Some("text/plain"));
    assert_eq!(result.media_type, None);
    assert_eq!(result.filename.as_deref(), Some("shared.txt"));
}

#[tokio::test]
async fn upload_rejects_url_inputs() {
    let client = MockFilesClient::new("openai");
    let error = files::upload(
        &client,
        "https://example.com/file.pdf",
        UploadFileOptions::new(),
    )
    .await
    .expect_err("url upload should fail");

    match error {
        LlmError::InvalidInput(message) => {
            assert!(message.contains("URL data is not supported for file uploads"))
        }
        other => panic!("expected invalid input, got {other:?}"),
    }
}

#[tokio::test]
async fn upload_rejects_invalid_base64_with_shared_data_content_message() {
    let client = MockFilesClient::new("openai");
    let error = files::upload(&client, "***not-base64***", UploadFileOptions::new())
        .await
        .expect_err("invalid base64 upload should fail");

    match error {
        LlmError::InvalidInput(message) => {
            assert!(message.contains("Invalid data content"));
            assert!(message.contains("not a base64-encoded media"));
        }
        other => panic!("expected invalid input, got {other:?}"),
    }
}

#[tokio::test]
async fn upload_requires_explicit_purpose_for_minimaxi() {
    let client = MockFilesClient::new("minimaxi");
    let error = files::upload(
        &client,
        b"hello".to_vec(),
        UploadFileOptions::new().with_filename("sample.txt"),
    )
    .await
    .expect_err("minimaxi upload should require purpose");

    match error {
        LlmError::InvalidInput(message) => {
            assert!(message.contains("MiniMaxi file uploads require UploadFileOptions.purpose"))
        }
        other => panic!("expected invalid input, got {other:?}"),
    }
}

#[tokio::test]
async fn upload_passes_provider_options_to_file_api() {
    let client = MockFilesClient::new("openai");

    files::upload(
        &client,
        b"hello".to_vec(),
        UploadFileOptions::new()
            .with_filename("hello.txt")
            .with_provider_option(
                "openai",
                serde_json::json!({
                    "purpose": "batch",
                    "expiresAfter": 42
                }),
            ),
    )
    .await
    .expect("upload result");

    let request = client.take_last_upload();
    assert_eq!(
        request.provider_options.get("openai"),
        Some(&serde_json::json!({
            "purpose": "batch",
            "expiresAfter": 42
        }))
    );
}

#[tokio::test]
async fn upload_google_family_uses_google_namespace_in_result() {
    let client = MockFilesClient::new("gemini");

    let result = files::upload(&client, b"hello".to_vec(), UploadFileOptions::new())
        .await
        .expect("upload result");

    assert_eq!(
        result.provider_reference.get("google"),
        Some("file_generated")
    );
    assert_eq!(result.provider_reference.get("gemini"), None);
    assert!(
        result
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("google"))
            .and_then(|metadata| metadata.as_object())
            .is_some(),
        "expected google-family file uploads to expose provider metadata under the google namespace"
    );
    assert!(
        result
            .provider_metadata
            .as_ref()
            .is_some_and(|metadata| !metadata.contains_key("gemini"))
    );
}

#[tokio::test]
async fn upload_google_family_warns_when_filename_is_explicit() {
    let client = MockFilesClient::new("gemini");

    let result = files::upload(
        &client,
        b"hello".to_vec(),
        UploadFileOptions::new().with_filename("custom-name.txt"),
    )
    .await
    .expect("upload result");

    assert_eq!(
        result.warnings,
        vec![Warning::unsupported("filename", None::<String>)]
    );
}
