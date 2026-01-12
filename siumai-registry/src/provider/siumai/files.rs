use super::Siumai;
use crate::error::LlmError;
use crate::traits::FileManagementCapability;
use crate::types::*;

#[async_trait::async_trait]
impl FileManagementCapability for Siumai {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.upload_file(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.list_files(query).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.retrieve_file(file_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.delete_file(file_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.get_file_content(file_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }
}
