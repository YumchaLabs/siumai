//! Rust-first xAI language-provider surface.

mod files;
mod language;
mod media;
pub mod models;
mod provider;
mod video;

pub use files::{
    FILES_SOURCE, FILES_VERIFIED_ON, XaiDeletedFile, XaiFile, XaiFileContent, XaiFileId,
    XaiFileList, XaiFileListOptions, XaiFileOrder, XaiFileUpload, XaiFiles,
};
pub use media::{
    IMAGE_API_MODE_ID, IMAGE_PROTOCOL_ID, IMAGE_SOURCE, MEDIA_VERIFIED_ON, SPEECH_API_MODE_ID,
    SPEECH_PROTOCOL_ID, SPEECH_SOURCE, TRANSCRIPTION_API_MODE_ID, TRANSCRIPTION_PROTOCOL_ID,
    TRANSCRIPTION_SOURCE, XaiImageModel, XaiSpeechModel, XaiTranscriptionModel,
};
pub use provider::{
    XaiConfigError, XaiCredential, XaiLanguageApi, XaiLanguageModel, XaiProvider,
    XaiProviderBuilder,
};
pub use video::{
    VIDEO_SOURCE, VIDEO_VERIFIED_ON, XaiVideoArtifact, XaiVideoAspectRatio, XaiVideoCreateRequest,
    XaiVideoImage, XaiVideoJob, XaiVideoJobId, XaiVideoJobState, XaiVideoJobs, XaiVideoResolution,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// xAI error envelope returned by OpenAI-compatible HTTP endpoints.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct XaiErrorData {
    pub error: XaiErrorPayload,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct XaiErrorPayload {
    pub message: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<serde_json::Value>,
}
