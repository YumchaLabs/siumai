//! Unsupported transformers used by ProviderSpec defaults.
//!
//! 该模块提供若干“占位” transformers，在 Provider 未实现某能力时，
//! 通过返回 `UnsupportedOperation` 而不是 panic。

use crate::error::LlmError;

pub(super) struct UnsupportedRequestTx {
    pub(super) provider: &'static str,
    pub(super) capability: &'static str,
}

impl crate::execution::transformers::request::RequestTransformer for UnsupportedRequestTx {
    fn provider_id(&self) -> &str {
        self.provider
    }

    fn transform_chat(
        &self,
        _req: &crate::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support chat ({})",
            self.provider, self.capability
        )))
    }
}

pub(super) struct UnsupportedResponseTx {
    pub(super) provider: &'static str,
}

impl crate::execution::transformers::response::ResponseTransformer for UnsupportedResponseTx {
    fn provider_id(&self) -> &str {
        self.provider
    }
}

pub(super) struct UnsupportedAudioTx {
    pub(super) provider: &'static str,
}

impl crate::execution::transformers::audio::AudioTransformer for UnsupportedAudioTx {
    fn provider_id(&self) -> &str {
        self.provider
    }

    fn build_tts_body(
        &self,
        _req: &crate::types::TtsRequest,
    ) -> Result<crate::execution::transformers::audio::AudioHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support TTS",
            self.provider
        )))
    }

    fn build_stt_body(
        &self,
        _req: &crate::types::SttRequest,
    ) -> Result<crate::execution::transformers::audio::AudioHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support STT",
            self.provider
        )))
    }

    fn tts_endpoint(&self) -> &str {
        ""
    }

    fn stt_endpoint(&self) -> &str {
        ""
    }

    fn parse_stt_response(&self, _json: &serde_json::Value) -> Result<String, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support STT",
            self.provider
        )))
    }
}

pub(super) struct UnsupportedFilesTx {
    pub(super) provider: &'static str,
}

impl crate::execution::transformers::files::FilesTransformer for UnsupportedFilesTx {
    fn provider_id(&self) -> &str {
        self.provider
    }

    fn build_upload_body(
        &self,
        _req: &crate::types::FileUploadRequest,
    ) -> Result<crate::execution::transformers::files::FilesHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support file management",
            self.provider
        )))
    }

    fn list_endpoint(&self, _query: &Option<crate::types::FileListQuery>) -> String {
        "/files".to_string()
    }

    fn retrieve_endpoint(&self, _file_id: &str) -> String {
        "/files".to_string()
    }

    fn delete_endpoint(&self, _file_id: &str) -> String {
        "/files".to_string()
    }

    fn transform_file_object(
        &self,
        _raw: &serde_json::Value,
    ) -> Result<crate::types::FileObject, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support file management",
            self.provider
        )))
    }

    fn transform_list_response(
        &self,
        _raw: &serde_json::Value,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not support file management",
            self.provider
        )))
    }
}

