//! Provider-owned Kimi Files lifecycle.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use http::Method;
use http::header::{ACCEPT, HeaderValue, RETRY_AFTER};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorKind, PublicDiagnosticText, ResponseDiagnostics, SensitiveResponse,
};
use siumai_protocol_openai::openai_error::{classify_http_error, decode_error_metadata};
use siumai_transport::{
    MultipartBody, MultipartPart, ProviderTransport, ReplaySafety, RequestBody, RequestHeaders,
    RequestPlan, RequestTarget, ResponseHeaders, TransportResponse,
};

const MAX_FILE_BYTES: usize = 100 * 1024 * 1024;
const MAX_FILE_NAME_BYTES: usize = 1_024;
const MAX_FILE_ID_BYTES: usize = 512;

/// Purpose accepted by Kimi's current Files upload endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum KimiFileUploadPurpose {
    FileExtract,
    Image,
    Video,
    Batch,
}

impl KimiFileUploadPurpose {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FileExtract => "file-extract",
            Self::Image => "image",
            Self::Video => "video",
            Self::Batch => "batch",
        }
    }
}

/// One bounded, owned Kimi file upload.
#[derive(Clone)]
pub struct KimiFileUpload {
    filename: String,
    media_type: String,
    data: Vec<u8>,
    purpose: KimiFileUploadPurpose,
}

impl KimiFileUpload {
    pub fn new(
        filename: impl Into<String>,
        media_type: impl Into<String>,
        data: impl Into<Vec<u8>>,
        purpose: KimiFileUploadPurpose,
    ) -> Result<Self, Error> {
        let upload = Self {
            filename: filename.into(),
            media_type: media_type.into(),
            data: data.into(),
            purpose,
        };
        upload.validate()?;
        Ok(upload)
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

    pub const fn purpose(&self) -> KimiFileUploadPurpose {
        self.purpose
    }

    fn validate(&self) -> Result<HeaderValue, Error> {
        if self.filename.trim().is_empty()
            || self.filename != self.filename.trim()
            || self.filename.len() > MAX_FILE_NAME_BYTES
            || self.filename.chars().any(char::is_control)
        {
            return Err(invalid("Kimi upload filename is invalid"));
        }
        if self.data.is_empty() || self.data.len() > MAX_FILE_BYTES {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "Kimi upload must contain between 1 byte and 100 MiB",
            ));
        }
        HeaderValue::from_str(&self.media_type)
            .map_err(|source| invalid("Kimi upload media type is invalid").with_source(source))
    }
}

impl fmt::Debug for KimiFileUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KimiFileUpload")
            .field("filename", &self.filename)
            .field("media_type", &self.media_type)
            .field("data_bytes", &self.data.len())
            .field("purpose", &self.purpose)
            .finish()
    }
}

/// Kimi file metadata. Additive future response fields remain available in `extra`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiFile {
    pub id: String,
    #[serde(rename = "object")]
    pub object_type: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    #[serde(default)]
    pub purpose: Option<String>,
    pub status: String,
    #[serde(default)]
    pub status_details: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Current non-paginated Kimi file list response.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct KimiFileList {
    #[serde(rename = "object")]
    pub object_type: String,
    #[serde(default)]
    pub data: Vec<KimiFile>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Result returned after permanently deleting one Kimi file.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct KimiFileDeleteResult {
    pub id: String,
    #[serde(rename = "object")]
    pub object_type: String,
    pub deleted: bool,
}

#[derive(Clone)]
pub(crate) struct MoonshotNativeRuntime {
    transport: ProviderTransport,
}

impl MoonshotNativeRuntime {
    pub(crate) fn new(transport: ProviderTransport) -> Self {
        Self { transport }
    }
}

/// Shared, lightweight Kimi Files lifecycle client.
#[derive(Clone)]
pub struct KimiFiles {
    runtime: Arc<MoonshotNativeRuntime>,
}

impl KimiFiles {
    pub(crate) fn new(runtime: Arc<MoonshotNativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn upload(&self, upload: KimiFileUpload) -> Result<KimiFile, Error> {
        self.upload_with_options(upload, CallOptions::default())
            .await
    }

    pub async fn upload_with_options(
        &self,
        upload: KimiFileUpload,
        options: CallOptions,
    ) -> Result<KimiFile, Error> {
        let media_type = upload.validate()?;
        let parts = vec![
            MultipartPart::file("file", upload.filename, media_type, upload.data).map_err(
                |source| invalid("Kimi file multipart body is invalid").with_source(source),
            )?,
            MultipartPart::field("purpose", upload.purpose.as_str().as_bytes().to_vec()).map_err(
                |source| invalid("Kimi file multipart body is invalid").with_source(source),
            )?,
        ];
        execute_json(
            &self.runtime.transport,
            Method::POST,
            target("files")?,
            RequestBody::multipart(MultipartBody::new(parts)),
            ReplaySafety::Never,
            options,
        )
        .await
    }

    pub async fn list(&self) -> Result<KimiFileList, Error> {
        self.list_with_options(CallOptions::default()).await
    }

    pub async fn list_with_options(&self, options: CallOptions) -> Result<KimiFileList, Error> {
        execute_json(
            &self.runtime.transport,
            Method::GET,
            target("files")?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await
    }

    pub async fn retrieve(&self, file_id: &str) -> Result<KimiFile, Error> {
        self.retrieve_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        file_id: &str,
        options: CallOptions,
    ) -> Result<KimiFile, Error> {
        validate_file_id(file_id)?;
        execute_json(
            &self.runtime.transport,
            Method::GET,
            target(format!("files/{file_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await
    }

    pub async fn content(&self, file_id: &str) -> Result<String, Error> {
        self.content_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn content_with_options(
        &self,
        file_id: &str,
        options: CallOptions,
    ) -> Result<String, Error> {
        validate_file_id(file_id)?;
        let response = execute(
            &self.runtime.transport,
            Method::GET,
            target(format!("files/{file_id}/content"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            "text/plain",
            options,
        )
        .await?;
        String::from_utf8(response.body().to_vec()).map_err(|source| {
            Error::new(ErrorKind::Protocol, "Kimi file content was not UTF-8 text")
                .with_source(source)
        })
    }

    pub async fn delete(&self, file_id: &str) -> Result<KimiFileDeleteResult, Error> {
        self.delete_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        file_id: &str,
        options: CallOptions,
    ) -> Result<KimiFileDeleteResult, Error> {
        validate_file_id(file_id)?;
        execute_json(
            &self.runtime.transport,
            Method::DELETE,
            target(format!("files/{file_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await
    }
}

impl fmt::Debug for KimiFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KimiFiles")
            .field("runtime", &"shared")
            .finish()
    }
}

async fn execute_json<T: DeserializeOwned>(
    transport: &ProviderTransport,
    method: Method,
    target: RequestTarget,
    body: RequestBody,
    replay_safety: ReplaySafety,
    options: CallOptions,
) -> Result<T, Error> {
    let response = execute(
        transport,
        method,
        target,
        body,
        replay_safety,
        "application/json",
        options,
    )
    .await?;
    serde_json::from_slice(response.body()).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Kimi Files response was not valid JSON",
        )
        .with_source(source)
    })
}

#[allow(clippy::too_many_arguments)]
async fn execute(
    transport: &ProviderTransport,
    method: Method,
    target: RequestTarget,
    body: RequestBody,
    replay_safety: ReplaySafety,
    accept: &'static str,
    options: CallOptions,
) -> Result<TransportResponse, Error> {
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static(accept))
        .map_err(|source| {
            Error::new(
                ErrorKind::Configuration,
                "Kimi Files request headers are invalid",
            )
            .with_source(source)
        })?;
    let plan = RequestPlan::new(method, target)
        .with_headers(headers)
        .with_body(body)
        .with_replay_safety(replay_safety)
        .map_err(|source| {
            Error::new(
                ErrorKind::Configuration,
                "Kimi Files request is not replay-safe",
            )
            .with_source(source)
        })?;
    let response = transport.execute(plan, options).await?;
    if response.status().is_success() {
        Ok(response)
    } else {
        Err(resource_error(response))
    }
}

fn resource_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let metadata = decode_error_metadata(&body);
    let kind = classify_http_error(
        status.as_u16(),
        metadata.as_ref().and_then(|value| value.code()),
        metadata.as_ref().and_then(|value| value.error_type()),
    );
    let sensitive = SensitiveResponse::new(
        headers
            .expose()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.to_string(), value.to_string()))
            })
            .collect(),
        body.to_vec(),
    );
    let mut diagnostics = ResponseDiagnostics::default()
        .with_status(status.as_u16())
        .with_body_truncated(sensitive.was_truncated());
    if let Some(metadata) = metadata {
        if let Some(code) = metadata
            .code()
            .and_then(|value| PublicDiagnosticText::new(value.to_string()).ok())
        {
            diagnostics = diagnostics.with_provider_code(code);
        }
        if let Some(error_type) = metadata
            .error_type()
            .and_then(|value| PublicDiagnosticText::new(value.to_string()).ok())
        {
            diagnostics = diagnostics.with_provider_type(error_type);
        }
        if let Some(param) = metadata
            .param()
            .and_then(|value| PublicDiagnosticText::new(value.to_string()).ok())
        {
            diagnostics = diagnostics.with_provider_param(param);
        }
    }
    if let Some(request_id) = response_header_text(&headers, "x-request-id")
        .or_else(|| response_header_text(&headers, "request-id"))
    {
        diagnostics = diagnostics.with_request_id(request_id);
    }
    if let Some(retry_after) = headers
        .get(&RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
    {
        diagnostics = diagnostics.with_retry_after(retry_after);
    }
    Error::new(kind, "Kimi Files request was rejected")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(sensitive)
}

fn response_header_text(
    headers: &ResponseHeaders,
    name: &'static str,
) -> Option<PublicDiagnosticText> {
    headers
        .expose()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| PublicDiagnosticText::new(value.to_string()).ok())
}

fn target(value: impl Into<String>) -> Result<RequestTarget, Error> {
    RequestTarget::new(value.into())
        .map_err(|source| invalid("Kimi Files target is invalid").with_source(source))
}

fn validate_file_id(file_id: &str) -> Result<(), Error> {
    if file_id.trim().is_empty()
        || file_id != file_id.trim()
        || file_id.len() > MAX_FILE_ID_BYTES
        || file_id.chars().any(char::is_control)
        || file_id.contains('/')
    {
        return Err(invalid("Kimi file identifier is invalid"));
    }
    Ok(())
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_debug_redacts_file_bytes_and_enforces_the_official_limit() {
        let upload = KimiFileUpload::new(
            "notes.txt",
            "text/plain",
            b"sentinel-private-file".to_vec(),
            KimiFileUploadPurpose::FileExtract,
        )
        .expect("upload");
        let debug = format!("{upload:?}");
        assert!(!debug.contains("sentinel-private-file"));
        assert!(debug.contains("21"));

        assert!(
            KimiFileUpload::new(
                "large.bin",
                "application/octet-stream",
                vec![0; MAX_FILE_BYTES + 1],
                KimiFileUploadPurpose::Batch,
            )
            .is_err()
        );
    }
}
