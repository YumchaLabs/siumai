use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use http::Method;
use http::header::HeaderValue;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind};
use siumai_transport::{MultipartBody, MultipartPart, ReplaySafety, RequestBody};
use thiserror::Error as ThisError;

use super::common::{
    BaseResponse, NativeResponseEnvelope, NativeRuntime, execute_download, execute_json,
    multipart_body, target,
};

const FILE_UPLOAD_TARGET: &str = "v1/files/upload";
const FILE_LIST_TARGET: &str = "v1/files/list";
const FILE_RETRIEVE_TARGET: &str = "v1/files/retrieve";
const FILE_DOWNLOAD_TARGET: &str = "v1/files/retrieve_content";
const FILE_DELETE_TARGET: &str = "v1/files/delete";

/// A validated MiniMax file identifier.
///
/// MiniMax defines file identifiers as positive signed 64-bit integers. The
/// response decoder also accepts their decimal string representation because
/// older examples represented placeholders as strings.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MinimaxFileId(u64);

impl MinimaxFileId {
    pub fn new(value: u64) -> Result<Self, MinimaxFileIdError> {
        if value == 0 {
            return Err(MinimaxFileIdError::Zero);
        }
        if value > i64::MAX as u64 {
            return Err(MinimaxFileIdError::OutOfRange);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for MinimaxFileId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("MinimaxFileId")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for MinimaxFileId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for MinimaxFileId {
    type Err = MinimaxFileIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(MinimaxFileIdError::InvalidDecimal);
        }
        let value = value
            .parse::<u64>()
            .map_err(|_| MinimaxFileIdError::OutOfRange)?;
        Self::new(value)
    }
}

impl TryFrom<u64> for MinimaxFileId {
    type Error = MinimaxFileIdError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<i64> for MinimaxFileId {
    type Error = MinimaxFileIdError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        let value = u64::try_from(value).map_err(|_| MinimaxFileIdError::OutOfRange)?;
        Self::new(value)
    }
}

impl Serialize for MinimaxFileId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_i64(self.0 as i64)
    }
}

impl<'de> Deserialize<'de> for MinimaxFileId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(MinimaxFileIdVisitor)
    }
}

struct MinimaxFileIdVisitor;

impl Visitor<'_> for MinimaxFileIdVisitor {
    type Value = MinimaxFileId;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a positive signed 64-bit MiniMax file identifier")
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        MinimaxFileId::new(value).map_err(E::custom)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        MinimaxFileId::try_from(value).map_err(E::custom)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        MinimaxFileId::from_str(value).map_err(E::custom)
    }
}

/// Validation failures for [`MinimaxFileId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum MinimaxFileIdError {
    #[error("MiniMax file identifier must be greater than zero")]
    Zero,
    #[error("MiniMax file identifier exceeds the signed 64-bit wire range")]
    OutOfRange,
    #[error("MiniMax file identifier must contain decimal digits only")]
    InvalidDecimal,
}

/// File purposes accepted by the MiniMax upload operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxFileUploadPurpose {
    VoiceClone,
    PromptAudio,
    TextToAudioAsyncInput,
    VideoUnderstanding,
    VideoGenerationInput,
}

impl MinimaxFileUploadPurpose {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VoiceClone => "voice_clone",
            Self::PromptAudio => "prompt_audio",
            Self::TextToAudioAsyncInput => "t2a_async_input",
            Self::VideoUnderstanding => "video_understanding",
            Self::VideoGenerationInput => "video_generation_input",
        }
    }
}

impl fmt::Display for MinimaxFileUploadPurpose {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// File purposes accepted by the MiniMax list operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxFileListPurpose {
    VoiceClone,
    PromptAudio,
    TextToAudioAsyncInput,
    VideoGenerationInput,
}

impl MinimaxFileListPurpose {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VoiceClone => "voice_clone",
            Self::PromptAudio => "prompt_audio",
            Self::TextToAudioAsyncInput => "t2a_async_input",
            Self::VideoGenerationInput => "video_generation_input",
        }
    }
}

impl fmt::Display for MinimaxFileListPurpose {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// File purposes accepted by the MiniMax delete operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxFileDeletePurpose {
    VoiceClone,
    PromptAudio,
    TextToAudioAsync,
    TextToAudioAsyncInput,
    VideoGeneration,
}

impl MinimaxFileDeletePurpose {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VoiceClone => "voice_clone",
            Self::PromptAudio => "prompt_audio",
            Self::TextToAudioAsync => "t2a_async",
            Self::TextToAudioAsyncInput => "t2a_async_input",
            Self::VideoGeneration => "video_generation",
        }
    }
}

impl fmt::Display for MinimaxFileDeletePurpose {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Owned input for one MiniMax file upload.
pub struct MinimaxFileUpload {
    purpose: MinimaxFileUploadPurpose,
    filename: String,
    media_type: String,
    data: Vec<u8>,
}

impl MinimaxFileUpload {
    pub fn new(
        purpose: MinimaxFileUploadPurpose,
        filename: impl Into<String>,
        media_type: impl Into<String>,
        data: impl Into<Vec<u8>>,
    ) -> Result<Self, Error> {
        let filename = filename.into();
        let media_type = media_type.into();
        let data = data.into();
        validate_filename(&filename)?;
        validate_media_type(&media_type)?;
        if data.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax file upload body must not be empty",
            ));
        }
        Ok(Self {
            purpose,
            filename,
            media_type,
            data,
        })
    }

    pub const fn purpose(&self) -> MinimaxFileUploadPurpose {
        self.purpose
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    fn validate_size(&self, maximum: usize) -> Result<(), Error> {
        if self.data.len() > maximum {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "MiniMax file upload exceeds the configured request limit",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for MinimaxFileUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxFileUpload")
            .field("purpose", &self.purpose)
            .field("filename_bytes", &self.filename.len())
            .field("media_type", &self.media_type)
            .field("data_bytes", &self.data.len())
            .finish()
    }
}

/// MiniMax file metadata. Unknown additive response fields are retained.
#[derive(Clone, PartialEq, Deserialize)]
pub struct MinimaxFile {
    #[serde(rename = "file_id")]
    id: MinimaxFileId,
    #[serde(default)]
    bytes: Option<u64>,
    #[serde(default)]
    created_at: Option<i64>,
    #[serde(default)]
    filename: Option<String>,
    #[serde(default)]
    purpose: Option<String>,
    #[serde(default)]
    download_url: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl MinimaxFile {
    pub const fn id(&self) -> MinimaxFileId {
        self.id
    }

    pub const fn bytes(&self) -> Option<u64> {
        self.bytes
    }

    pub const fn created_at_unix_seconds(&self) -> Option<i64> {
        self.created_at
    }

    pub fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }

    pub fn purpose(&self) -> Option<&str> {
        self.purpose.as_deref()
    }

    /// Return the provider-issued download URL without following it.
    ///
    /// The URL may contain transient or signed data and is therefore omitted
    /// from `Debug`. Use [`MinimaxFiles::download`] for the authenticated file
    /// content endpoint.
    pub fn download_url(&self) -> Option<&str> {
        self.download_url.as_deref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxFile")
            .field("id", &self.id)
            .field("bytes", &self.bytes)
            .field("created_at", &self.created_at)
            .field("filename_present", &self.filename.is_some())
            .field("purpose_present", &self.purpose.is_some())
            .field("download_url_present", &self.download_url.is_some())
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Result of one MiniMax list operation.
#[derive(Clone, PartialEq)]
pub struct MinimaxFileList {
    files: Vec<MinimaxFile>,
    extra: BTreeMap<String, Value>,
}

impl MinimaxFileList {
    pub fn files(&self) -> &[MinimaxFile] {
        &self.files
    }

    pub fn into_files(self) -> Vec<MinimaxFile> {
        self.files
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxFileList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxFileList")
            .field("file_count", &self.files.len())
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Result of one successful MiniMax delete operation.
#[derive(Clone, PartialEq)]
pub struct MinimaxFileDeleteResult {
    id: MinimaxFileId,
    extra: BTreeMap<String, Value>,
}

impl MinimaxFileDeleteResult {
    pub const fn id(&self) -> MinimaxFileId {
        self.id
    }

    pub const fn deleted(&self) -> bool {
        true
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxFileDeleteResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxFileDeleteResult")
            .field("id", &self.id)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Shared, lightweight handle for MiniMax's provider-native Files API.
#[derive(Clone)]
pub struct MinimaxFiles {
    runtime: Arc<NativeRuntime>,
}

impl MinimaxFiles {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn upload(&self, upload: MinimaxFileUpload) -> Result<MinimaxFile, Error> {
        self.upload_with_options(upload, CallOptions::default())
            .await
    }

    pub async fn upload_with_options(
        &self,
        upload: MinimaxFileUpload,
        options: CallOptions,
    ) -> Result<MinimaxFile, Error> {
        upload.validate_size(self.runtime.max_request_bytes())?;
        let MinimaxFileUpload {
            purpose,
            filename,
            media_type,
            data,
        } = upload;
        let media_type = HeaderValue::from_str(&media_type).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "MiniMax file media type is invalid",
            )
            .with_source(source)
        })?;
        let purpose =
            MultipartPart::field("purpose", purpose.as_str().to_owned()).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "MiniMax file upload purpose is invalid",
                )
                .with_source(source)
            })?;
        let file = MultipartPart::file("file", filename, media_type, data).map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "MiniMax file upload is invalid")
                .with_source(source)
        })?;
        let response: FileEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(FILE_UPLOAD_TARGET)?,
            multipart_body(MultipartBody::new(vec![purpose, file])),
            ReplaySafety::Never,
            options,
        )
        .await?;
        response.into_file()
    }

    pub async fn list(&self, purpose: MinimaxFileListPurpose) -> Result<MinimaxFileList, Error> {
        self.list_with_options(purpose, CallOptions::default())
            .await
    }

    pub async fn list_with_options(
        &self,
        purpose: MinimaxFileListPurpose,
        options: CallOptions,
    ) -> Result<MinimaxFileList, Error> {
        let response: FileListEnvelope = execute_json(
            &self.runtime,
            Method::GET,
            list_target(purpose)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await?;
        Ok(MinimaxFileList {
            files: response.files,
            extra: response.extra,
        })
    }

    pub async fn retrieve(&self, file_id: MinimaxFileId) -> Result<MinimaxFile, Error> {
        self.retrieve_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        file_id: MinimaxFileId,
        options: CallOptions,
    ) -> Result<MinimaxFile, Error> {
        let response: FileEnvelope = execute_json(
            &self.runtime,
            Method::GET,
            file_target(FILE_RETRIEVE_TARGET, file_id)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await?;
        response.into_file()
    }

    pub async fn download(&self, file_id: MinimaxFileId) -> Result<Vec<u8>, Error> {
        self.download_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn download_with_options(
        &self,
        file_id: MinimaxFileId,
        options: CallOptions,
    ) -> Result<Vec<u8>, Error> {
        execute_download(
            &self.runtime,
            file_target(FILE_DOWNLOAD_TARGET, file_id)?,
            options,
        )
        .await
    }

    /// Delete a file with an explicit operation-specific purpose.
    ///
    /// This method sends exactly one non-replayable delete request. It never
    /// performs a hidden retrieve to infer the purpose.
    pub async fn delete(
        &self,
        file_id: MinimaxFileId,
        purpose: MinimaxFileDeletePurpose,
    ) -> Result<MinimaxFileDeleteResult, Error> {
        self.delete_with_options(file_id, purpose, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        file_id: MinimaxFileId,
        purpose: MinimaxFileDeletePurpose,
        options: CallOptions,
    ) -> Result<MinimaxFileDeleteResult, Error> {
        let body =
            RequestBody::json(&DeleteFileRequest { file_id, purpose }).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "MiniMax file delete request is invalid",
                )
                .with_source(source)
            })?;
        let response: DeleteFileEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(FILE_DELETE_TARGET)?,
            body,
            ReplaySafety::Never,
            options,
        )
        .await?;
        if response.file_id.is_some_and(|returned| returned != file_id) {
            return Err(Error::new(
                ErrorKind::ProtocolViolation,
                "MiniMax file delete response returned a different file identifier",
            ));
        }
        Ok(MinimaxFileDeleteResult {
            id: file_id,
            extra: response.extra,
        })
    }
}

impl fmt::Debug for MinimaxFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxFiles")
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Deserialize)]
struct FileEnvelope {
    #[serde(default)]
    file: Option<MinimaxFile>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
}

impl FileEnvelope {
    fn into_file(self) -> Result<MinimaxFile, Error> {
        self.file.ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax file response omitted file metadata",
            )
        })
    }
}

impl NativeResponseEnvelope for FileEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct FileListEnvelope {
    #[serde(default)]
    files: Vec<MinimaxFile>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl NativeResponseEnvelope for FileListEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct DeleteFileEnvelope {
    #[serde(default)]
    file_id: Option<MinimaxFileId>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl NativeResponseEnvelope for DeleteFileEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Serialize)]
struct DeleteFileRequest {
    file_id: MinimaxFileId,
    #[serde(serialize_with = "serialize_delete_purpose")]
    purpose: MinimaxFileDeletePurpose,
}

fn serialize_delete_purpose<S>(
    purpose: &MinimaxFileDeletePurpose,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(purpose.as_str())
}

fn file_target(
    prefix: &str,
    file_id: MinimaxFileId,
) -> Result<siumai_transport::RequestTarget, Error> {
    target(format!("{prefix}?file_id={file_id}"))
}

fn list_target(purpose: MinimaxFileListPurpose) -> Result<siumai_transport::RequestTarget, Error> {
    target(format!("{FILE_LIST_TARGET}?purpose={}", purpose.as_str()))
}

fn validate_filename(filename: &str) -> Result<(), Error> {
    if filename.trim().is_empty()
        || filename.len() > 1_024
        || filename.chars().any(char::is_control)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "MiniMax file name is invalid",
        ));
    }
    Ok(())
}

fn validate_media_type(media_type: &str) -> Result<(), Error> {
    if media_type.trim().is_empty()
        || media_type.len() > 256
        || media_type.chars().any(char::is_control)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "MiniMax file media type is invalid",
        ));
    }
    HeaderValue::from_str(media_type).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "MiniMax file media type is invalid",
        )
        .with_source(source)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::common::validate_base_response;

    #[test]
    fn file_id_is_a_positive_signed_wire_integer() {
        let id = MinimaxFileId::new(42).expect("valid id");
        assert_eq!(id.get(), 42);
        assert_eq!(
            serde_json::to_value(id).expect("serialize"),
            Value::from(42)
        );
        assert_eq!(
            serde_json::from_value::<MinimaxFileId>(Value::from("42")).expect("string id"),
            id
        );
        assert_eq!(MinimaxFileId::new(0), Err(MinimaxFileIdError::Zero));
        assert_eq!(
            MinimaxFileId::new(i64::MAX as u64 + 1),
            Err(MinimaxFileIdError::OutOfRange)
        );
        assert_eq!(
            " 42".parse::<MinimaxFileId>(),
            Err(MinimaxFileIdError::InvalidDecimal)
        );
    }

    #[test]
    fn purposes_are_operation_specific_and_match_the_wire() {
        assert_eq!(
            MinimaxFileUploadPurpose::VideoUnderstanding.as_str(),
            "video_understanding"
        );
        assert_eq!(
            MinimaxFileUploadPurpose::VideoGenerationInput.as_str(),
            "video_generation_input"
        );
        assert_eq!(
            MinimaxFileListPurpose::VideoGenerationInput.as_str(),
            "video_generation_input"
        );
        assert_eq!(
            MinimaxFileDeletePurpose::TextToAudioAsync.as_str(),
            "t2a_async"
        );
        assert_eq!(
            MinimaxFileDeletePurpose::VideoGeneration.as_str(),
            "video_generation"
        );
    }

    #[test]
    fn upload_validation_is_local_and_bounded() {
        let error = MinimaxFileUpload::new(
            MinimaxFileUploadPurpose::PromptAudio,
            "voice.wav",
            "audio/wav",
            Vec::new(),
        )
        .expect_err("empty body must fail");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);

        let upload = MinimaxFileUpload::new(
            MinimaxFileUploadPurpose::PromptAudio,
            "voice.wav",
            "audio/wav",
            vec![1, 2, 3],
        )
        .expect("valid upload");
        assert!(upload.validate_size(3).is_ok());
        assert_eq!(
            upload
                .validate_size(2)
                .expect_err("configured limit must be enforced")
                .kind(),
            ErrorKind::LimitExceeded
        );
    }

    #[test]
    fn file_response_retains_unknown_fields_and_optional_download_url() {
        let response: FileEnvelope = serde_json::from_value(serde_json::json!({
            "file": {
                "file_id": 42,
                "bytes": 7,
                "created_at": 1_700_000_000,
                "filename": "private.txt",
                "purpose": "future_purpose",
                "download_url": "https://example.invalid/signed?token=secret",
                "future_field": {"nested": true}
            },
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("response should decode");
        validate_base_response(response.base_response()).expect("base response should succeed");
        let file = response.into_file().expect("file should exist");

        assert_eq!(file.id(), MinimaxFileId::new(42).expect("valid id"));
        assert_eq!(file.purpose(), Some("future_purpose"));
        assert_eq!(
            file.download_url(),
            Some("https://example.invalid/signed?token=secret")
        );
        assert!(file.extra().contains_key("future_field"));
        let debug = format!("{file:?}");
        assert!(!debug.contains("private.txt"));
        assert!(!debug.contains("token=secret"));
        assert!(!debug.contains("nested"));
    }

    #[test]
    fn delete_body_always_contains_the_explicit_purpose() {
        let request = DeleteFileRequest {
            file_id: MinimaxFileId::new(42).expect("valid id"),
            purpose: MinimaxFileDeletePurpose::TextToAudioAsyncInput,
        };
        assert_eq!(
            serde_json::to_value(request).expect("delete request should encode"),
            serde_json::json!({
                "file_id": 42,
                "purpose": "t2a_async_input"
            })
        );
    }

    #[test]
    fn targets_use_only_fixed_purpose_values_and_numeric_ids() {
        assert_eq!(
            list_target(MinimaxFileListPurpose::VoiceClone)
                .expect("list target")
                .as_str(),
            "v1/files/list?purpose=voice_clone"
        );
        assert_eq!(
            file_target(
                FILE_RETRIEVE_TARGET,
                MinimaxFileId::new(42).expect("valid id")
            )
            .expect("retrieve target")
            .as_str(),
            "v1/files/retrieve?file_id=42"
        );
    }
}
