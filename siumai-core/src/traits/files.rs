//! File management capability trait

use crate::error::LlmError;
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
};
use async_trait::async_trait;

#[async_trait]
pub trait FileManagementCapability: Send + Sync {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError>;
    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError>;
    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError>;
    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError>;
    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError>;
}
