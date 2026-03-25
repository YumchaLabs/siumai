use super::*;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct BridgeFilesClient;

#[async_trait::async_trait]
impl crate::traits::FileManagementCapability for BridgeFilesClient {
    async fn upload_file(
        &self,
        request: crate::types::FileUploadRequest,
    ) -> Result<crate::types::FileObject, LlmError> {
        Ok(crate::types::FileObject {
            id: format!("file:{}", request.filename),
            filename: request.filename,
            bytes: request.content.len() as u64,
            created_at: 1,
            purpose: request.purpose,
            status: "processed".to_string(),
            mime_type: request.mime_type,
            metadata: HashMap::new(),
        })
    }

    async fn list_files(
        &self,
        query: Option<crate::types::FileListQuery>,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        Ok(crate::types::FileListResponse {
            files: vec![crate::types::FileObject {
                id: "file:list".to_string(),
                filename: "listed.txt".to_string(),
                bytes: 5,
                created_at: 2,
                purpose: query
                    .and_then(|item| item.purpose)
                    .unwrap_or_else(|| "assistants".to_string()),
                status: "processed".to_string(),
                mime_type: Some("text/plain".to_string()),
                metadata: HashMap::new(),
            }],
            has_more: false,
            next_cursor: None,
        })
    }

    async fn retrieve_file(&self, file_id: String) -> Result<crate::types::FileObject, LlmError> {
        Ok(crate::types::FileObject {
            id: file_id,
            filename: "retrieved.txt".to_string(),
            bytes: 7,
            created_at: 3,
            purpose: "assistants".to_string(),
            status: "processed".to_string(),
            mime_type: Some("text/plain".to_string()),
            metadata: HashMap::new(),
        })
    }

    async fn delete_file(
        &self,
        file_id: String,
    ) -> Result<crate::types::FileDeleteResponse, LlmError> {
        Ok(crate::types::FileDeleteResponse {
            id: file_id,
            deleted: true,
        })
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        Ok(format!("content:{file_id}").into_bytes())
    }
}

impl LlmClient for BridgeFilesClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_files")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["file-model".into()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_file_management()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_file_management_capability(
        &self,
    ) -> Option<&dyn crate::traits::FileManagementCapability> {
        Some(self)
    }
}

struct BridgeFilesFactory;

#[async_trait::async_trait]
impl ProviderFactory for BridgeFilesFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(BridgeFilesClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_files")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_file_management()
    }
}

#[tokio::test]
async fn language_model_handle_delegates_file_management_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_files".to_string(),
        Arc::new(BridgeFilesFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("testprov_files:file-model").unwrap();

    let uploaded = handle
        .upload_file(crate::types::FileUploadRequest {
            content: b"hello".to_vec(),
            filename: "hello.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            purpose: "assistants".to_string(),
            metadata: HashMap::new(),
            http_config: None,
        })
        .await
        .unwrap();
    let listed = handle
        .list_files(Some(crate::types::FileListQuery {
            purpose: Some("assistants".to_string()),
            ..Default::default()
        }))
        .await
        .unwrap();
    let retrieved = handle.retrieve_file("file-123".to_string()).await.unwrap();
    let content = handle
        .get_file_content("file-123".to_string())
        .await
        .unwrap();
    let deleted = handle.delete_file("file-123".to_string()).await.unwrap();

    assert_eq!(uploaded.id, "file:hello.txt");
    assert_eq!(uploaded.filename, "hello.txt");
    assert_eq!(listed.files.len(), 1);
    assert_eq!(listed.files[0].purpose, "assistants");
    assert_eq!(retrieved.id, "file-123");
    assert_eq!(content, b"content:file-123");
    assert!(deleted.deleted);
    assert!(handle.as_file_management_capability().is_some());
}
