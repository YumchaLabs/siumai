use super::OpenAiClient;
use crate::error::LlmError;
use crate::providers::openai::{OpenAiConfig, OpenAiFiles};
use async_trait::async_trait;

#[async_trait]
impl crate::traits::FileManagementCapability for OpenAiClient {
    async fn upload_file(
        &self,
        request: crate::types::FileUploadRequest,
    ) -> Result<crate::types::FileObject, LlmError> {
        let cfg = OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = OpenAiFiles::new(
            cfg,
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
        );
        files.upload_file(request).await
    }

    async fn list_files(
        &self,
        query: Option<crate::types::FileListQuery>,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        let cfg = OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = OpenAiFiles::new(
            cfg,
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
        );
        files.list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<crate::types::FileObject, LlmError> {
        let cfg = OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = OpenAiFiles::new(
            cfg,
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
        );
        files.retrieve_file(file_id).await
    }

    async fn delete_file(
        &self,
        file_id: String,
    ) -> Result<crate::types::FileDeleteResponse, LlmError> {
        let cfg = OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = OpenAiFiles::new(
            cfg,
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
        );
        files.delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let cfg = OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
        };
        let files = OpenAiFiles::new(
            cfg,
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
        );
        files.get_file_content(file_id).await
    }
}
