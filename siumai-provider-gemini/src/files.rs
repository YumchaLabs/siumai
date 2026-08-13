use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use siumai_core::{CallOptions, Error, ErrorContext, ErrorKind};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan, RequestTarget,
};
use thiserror::Error as ThisError;

use crate::http::response_error;
use crate::provider::ProviderRuntime;

const FILES_TARGET: &str = "v1/files";
const MAX_FILE_ID_BYTES: usize = 40;
const MAX_PAGE_TOKEN_BYTES: usize = 16 * 1024;
const MAX_DISPLAY_NAME_CHARS: usize = 512;
const MAX_METADATA_TEXT_BYTES: usize = 16 * 1024;

/// A validated Gemini File resource name in the form `files/{id}`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GeminiFileName(String);

impl GeminiFileName {
    pub fn new(value: impl Into<String>) -> Result<Self, GeminiFileNameError> {
        let value = value.into();
        let Some(id) = value.strip_prefix("files/") else {
            return Err(GeminiFileNameError::InvalidPrefix);
        };
        if id.is_empty() || id.len() > MAX_FILE_ID_BYTES {
            return Err(GeminiFileNameError::InvalidLength);
        }
        if id.starts_with('-')
            || id.ends_with('-')
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            return Err(GeminiFileNameError::InvalidId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for GeminiFileName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("GeminiFileName")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for GeminiFileName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum GeminiFileNameError {
    #[error("Gemini file name must start with `files/`")]
    InvalidPrefix,
    #[error("Gemini file ID must contain between 1 and 40 bytes")]
    InvalidLength,
    #[error("Gemini file ID must use lowercase ASCII letters, digits, or interior dashes")]
    InvalidId,
}

/// Processing state returned for a Gemini File resource.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeminiFileState {
    Unspecified,
    Processing,
    Active,
    Failed,
    Unknown(String),
}

/// Origin of a Gemini File resource.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeminiFileSource {
    Unspecified,
    Uploaded,
    Generated,
    Registered,
    Unknown(String),
}

/// Bounded processing failure metadata attached to a File resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiFileProcessingError {
    code: Option<i32>,
    status: Option<String>,
    details_present: bool,
}

impl GeminiFileProcessingError {
    pub const fn code(&self) -> Option<i32> {
        self.code
    }

    pub fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }

    pub const fn details_present(&self) -> bool {
        self.details_present
    }
}

/// A potentially signed provider URL that is redacted from default diagnostics.
#[derive(Clone, PartialEq, Eq)]
pub struct GeminiDownloadUri(String);

impl GeminiDownloadUri {
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for GeminiDownloadUri {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GeminiDownloadUri([REDACTED])")
    }
}

/// Stable-v1 Gemini File metadata.
#[derive(Clone, PartialEq, Eq)]
pub struct GeminiFile {
    pub name: GeminiFileName,
    pub display_name: Option<String>,
    pub mime_type: Option<String>,
    pub size_bytes: Option<u64>,
    pub state: GeminiFileState,
    pub source: GeminiFileSource,
    pub uri: Option<String>,
    pub download_uri: Option<GeminiDownloadUri>,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub sha256_base64: Option<String>,
    pub video_duration: Option<String>,
    pub processing_error: Option<GeminiFileProcessingError>,
}

impl fmt::Debug for GeminiFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiFile")
            .field("name", &self.name)
            .field("display_name", &self.display_name)
            .field("mime_type", &self.mime_type)
            .field("size_bytes", &self.size_bytes)
            .field("state", &self.state)
            .field("source", &self.source)
            .field("uri", &self.uri)
            .field("download_uri", &self.download_uri)
            .field("created_at", &self.created_at)
            .field("updated_at", &self.updated_at)
            .field("expires_at", &self.expires_at)
            .field("sha256_base64", &self.sha256_base64)
            .field("video_duration", &self.video_duration)
            .field("processing_error", &self.processing_error)
            .finish()
    }
}

/// Bounded stable-v1 Files list request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeminiFileListQuery {
    page_size: Option<u16>,
    page_token: Option<String>,
}

impl GeminiFileListQuery {
    pub const fn new() -> Self {
        Self {
            page_size: None,
            page_token: None,
        }
    }

    pub fn with_page_size(mut self, page_size: u16) -> Result<Self, Error> {
        if !(1..=100).contains(&page_size) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Gemini Files page size must be between 1 and 100",
            ));
        }
        self.page_size = Some(page_size);
        Ok(self)
    }

    pub fn with_page_token(mut self, page_token: impl Into<String>) -> Result<Self, Error> {
        let page_token = page_token.into();
        if page_token.is_empty()
            || page_token.len() > MAX_PAGE_TOKEN_BYTES
            || page_token.chars().any(char::is_control)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Gemini Files page token is invalid",
            ));
        }
        self.page_token = Some(page_token);
        Ok(self)
    }
}

/// One stable-v1 Files page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiFilePage {
    pub files: Vec<GeminiFile>,
    pub next_page_token: Option<String>,
}

/// Shared, lightweight handle for stable-v1 Gemini File metadata lifecycle operations.
#[derive(Clone)]
pub struct GeminiFiles {
    runtime: Arc<ProviderRuntime>,
}

impl GeminiFiles {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn get(&self, name: &GeminiFileName) -> Result<GeminiFile, Error> {
        self.get_with_options(name, CallOptions::default()).await
    }

    pub async fn get_with_options(
        &self,
        name: &GeminiFileName,
        options: CallOptions,
    ) -> Result<GeminiFile, Error> {
        let wire = self
            .execute_json::<GeminiFileWire>(
                Method::GET,
                RequestTarget::new(format!("v1/{}", name.as_str())).map_err(request_error)?,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await?;
        decode_file(wire)
    }

    pub async fn list(&self, query: GeminiFileListQuery) -> Result<GeminiFilePage, Error> {
        self.list_with_options(query, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        query: GeminiFileListQuery,
        options: CallOptions,
    ) -> Result<GeminiFilePage, Error> {
        let target = list_target(&query)?;
        let wire = self
            .execute_json::<GeminiFilePageWire>(
                Method::GET,
                target,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await?;
        let files = wire
            .files
            .into_iter()
            .map(decode_file)
            .collect::<Result<Vec<_>, _>>()?;
        let next_page_token = checked_optional_text(
            wire.next_page_token,
            MAX_PAGE_TOKEN_BYTES,
            "Gemini Files next-page token is invalid",
        )?;
        Ok(GeminiFilePage {
            files,
            next_page_token,
        })
    }

    pub async fn delete(&self, name: &GeminiFileName) -> Result<(), Error> {
        self.delete_with_options(name, CallOptions::default()).await
    }

    pub async fn delete_with_options(
        &self,
        name: &GeminiFileName,
        options: CallOptions,
    ) -> Result<(), Error> {
        let _: EmptyWire = self
            .execute_json(
                Method::DELETE,
                RequestTarget::new(format!("v1/{}", name.as_str())).map_err(request_error)?,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await?;
        Ok(())
    }

    async fn execute_json<T>(
        &self,
        method: Method,
        target: RequestTarget,
        replay_safety: ReplaySafety,
        options: CallOptions,
    ) -> Result<T, Error>
    where
        T: DeserializeOwned,
    {
        if options.has_provider_options() {
            return Err(self.contextualize(Error::new(
                ErrorKind::InvalidInput,
                "Gemini Files operations do not accept language-model provider options",
            )));
        }
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_error)?;
        let plan = RequestPlan::new(method, target)
            .with_headers(headers)
            .with_body(RequestBody::Empty)
            .with_replay_safety(replay_safety)
            .map_err(request_error)?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(
                response,
                "Gemini rejected the Files resource request",
            )));
        }
        let (_, _, body) = response.into_parts();
        serde_json::from_slice(&body).map_err(|source| {
            self.contextualize(
                Error::new(
                    ErrorKind::Protocol,
                    "Gemini returned malformed Files resource JSON",
                )
                .with_source(source),
            )
        })
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: None,
            provider: Some(self.runtime.interactions_scope.provider_id().clone()),
            route: None,
            model: None,
        })
    }
}

impl fmt::Debug for GeminiFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiFiles")
            .field("runtime", &"shared")
            .finish()
    }
}

fn list_target(query: &GeminiFileListQuery) -> Result<RequestTarget, Error> {
    let mut pairs = Vec::new();
    if let Some(page_size) = query.page_size {
        pairs.push(format!("pageSize={page_size}"));
    }
    if let Some(page_token) = &query.page_token {
        pairs.push(format!("pageToken={}", urlencoding::encode(page_token)));
    }
    let target = if pairs.is_empty() {
        FILES_TARGET.to_string()
    } else {
        format!("{FILES_TARGET}?{}", pairs.join("&"))
    };
    RequestTarget::new(target).map_err(request_error)
}

fn decode_file(wire: GeminiFileWire) -> Result<GeminiFile, Error> {
    let name = GeminiFileName::new(wire.name).map_err(|source| {
        Error::protocol_violation("Gemini returned an invalid File resource name")
            .with_source(source)
    })?;
    let display_name = checked_optional_chars(
        wire.display_name,
        MAX_DISPLAY_NAME_CHARS,
        "Gemini returned an invalid File display name",
    )?;
    let mime_type = checked_optional_text(
        wire.mime_type,
        256,
        "Gemini returned an invalid File MIME type",
    )?;
    let uri = checked_optional_text(
        wire.uri,
        MAX_METADATA_TEXT_BYTES,
        "Gemini returned an invalid File URI",
    )?;
    let download_uri = checked_optional_text(
        wire.download_uri,
        MAX_METADATA_TEXT_BYTES,
        "Gemini returned an invalid File download URI",
    )?
    .map(GeminiDownloadUri);
    let sha256_base64 = checked_optional_text(
        wire.sha256_hash,
        256,
        "Gemini returned an invalid File SHA-256 hash",
    )?;
    let video_duration = wire
        .video_metadata
        .and_then(|metadata| metadata.video_duration)
        .map(|value| {
            if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
                Err(Error::protocol_violation(
                    "Gemini returned an invalid File video duration",
                ))
            } else {
                Ok(value)
            }
        })
        .transpose()?;
    Ok(GeminiFile {
        name,
        display_name,
        mime_type,
        size_bytes: wire
            .size_bytes
            .map(|value| {
                value.parse::<u64>().map_err(|source| {
                    Error::protocol_violation("Gemini returned an invalid File size")
                        .with_source(source)
                })
            })
            .transpose()?,
        state: decode_state(wire.state),
        source: decode_source(wire.source),
        uri,
        download_uri,
        created_at: decode_time(wire.create_time)?,
        updated_at: decode_time(wire.update_time)?,
        expires_at: decode_time(wire.expiration_time)?,
        sha256_base64,
        video_duration,
        processing_error: wire.error.map(decode_processing_error).transpose()?,
    })
}

fn decode_state(value: Option<String>) -> GeminiFileState {
    match value.as_deref() {
        None | Some("STATE_UNSPECIFIED") => GeminiFileState::Unspecified,
        Some("PROCESSING") => GeminiFileState::Processing,
        Some("ACTIVE") => GeminiFileState::Active,
        Some("FAILED") => GeminiFileState::Failed,
        Some(value) => GeminiFileState::Unknown(value.to_string()),
    }
}

fn decode_source(value: Option<String>) -> GeminiFileSource {
    match value.as_deref() {
        None | Some("SOURCE_UNSPECIFIED") => GeminiFileSource::Unspecified,
        Some("UPLOADED") => GeminiFileSource::Uploaded,
        Some("GENERATED") => GeminiFileSource::Generated,
        Some("REGISTERED") => GeminiFileSource::Registered,
        Some(value) => GeminiFileSource::Unknown(value.to_string()),
    }
}

fn decode_processing_error(
    wire: GeminiFileProcessingErrorWire,
) -> Result<GeminiFileProcessingError, Error> {
    let status = checked_optional_text(
        wire.status,
        256,
        "Gemini returned an invalid File processing status",
    )?;
    Ok(GeminiFileProcessingError {
        code: wire.code,
        status,
        details_present: wire.details.is_some_and(|details| !details.is_empty()),
    })
}

fn decode_time(value: Option<String>) -> Result<Option<DateTime<Utc>>, Error> {
    value
        .map(|value| {
            DateTime::parse_from_rfc3339(&value)
                .map(|value| value.with_timezone(&Utc))
                .map_err(|source| {
                    Error::protocol_violation("Gemini returned an invalid File timestamp")
                        .with_source(source)
                })
        })
        .transpose()
}

fn checked_optional_text(
    value: Option<String>,
    maximum: usize,
    message: &'static str,
) -> Result<Option<String>, Error> {
    value
        .map(|value| {
            if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
                Err(Error::protocol_violation(message))
            } else {
                Ok(value)
            }
        })
        .transpose()
}

fn checked_optional_chars(
    value: Option<String>,
    maximum: usize,
    message: &'static str,
) -> Result<Option<String>, Error> {
    value
        .map(|value| {
            if value.chars().count() > maximum || value.chars().any(char::is_control) {
                Err(Error::protocol_violation(message))
            } else {
                Ok(value)
            }
        })
        .transpose()
}

fn request_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Gemini Files request violates the transport contract",
    )
    .with_source(source)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFilePageWire {
    #[serde(default)]
    files: Vec<GeminiFileWire>,
    #[serde(default)]
    next_page_token: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFileWire {
    name: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    size_bytes: Option<String>,
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    uri: Option<String>,
    #[serde(default)]
    download_uri: Option<String>,
    #[serde(default)]
    create_time: Option<String>,
    #[serde(default)]
    update_time: Option<String>,
    #[serde(default)]
    expiration_time: Option<String>,
    #[serde(default)]
    sha256_hash: Option<String>,
    #[serde(default)]
    video_metadata: Option<GeminiVideoMetadataWire>,
    #[serde(default)]
    error: Option<GeminiFileProcessingErrorWire>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiVideoMetadataWire {
    #[serde(default)]
    video_duration: Option<String>,
}

#[derive(Deserialize)]
struct GeminiFileProcessingErrorWire {
    #[serde(default)]
    code: Option<i32>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    details: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize)]
struct EmptyWire {}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use siumai_core::{ReplayDomain, ReplayDomainId};
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{GeminiCredential, GeminiProvider};

    fn provider(base_url: String) -> GeminiProvider {
        GeminiProvider::builder(GeminiCredential::api_key("test-key"))
            .with_endpoint(EndpointConfig::local_explicit(base_url).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("gemini-files-test").unwrap(),
            ))
            .build()
            .unwrap()
    }

    #[test]
    fn file_name_enforces_the_stable_resource_contract() {
        assert!(GeminiFileName::new("files/abc-123").is_ok());
        assert!(GeminiFileName::new("abc-123").is_err());
        assert!(GeminiFileName::new("files/ABC").is_err());
        assert!(GeminiFileName::new("files/-abc").is_err());
        assert!(GeminiFileName::new("files/abc/def").is_err());
    }

    #[test]
    fn file_decoder_preserves_unknown_lifecycle_values_and_redacts_download_url() {
        let wire = serde_json::from_value::<GeminiFileWire>(json!({
            "name": "files/abc-123",
            "displayName": "fixture",
            "sizeBytes": "42",
            "state": "PAUSED",
            "source": "MIRRORED",
            "downloadUri": "https://example.invalid/file?secret=sentinel",
            "createTime": "2026-08-08T12:00:00Z",
            "error": {
                "code": 13,
                "status": "INTERNAL",
                "message": "sentinel provider detail",
                "details": [{"private": "sentinel"}]
            }
        }))
        .unwrap();

        let file = decode_file(wire).unwrap();
        assert_eq!(file.size_bytes, Some(42));
        assert_eq!(file.state, GeminiFileState::Unknown("PAUSED".to_string()));
        assert_eq!(
            file.source,
            GeminiFileSource::Unknown("MIRRORED".to_string())
        );
        assert!(
            file.download_uri
                .as_ref()
                .unwrap()
                .expose()
                .contains("sentinel")
        );
        let debug = format!("{file:?}");
        assert!(!debug.contains("secret=sentinel"));
        assert!(!debug.contains("provider detail"));
        assert!(!debug.contains("private"));
    }

    #[test]
    fn list_query_is_bounded_and_percent_encoded() {
        let query = GeminiFileListQuery::new()
            .with_page_size(2)
            .unwrap()
            .with_page_token("next token/+")
            .unwrap();
        let target = list_target(&query).unwrap();
        assert_eq!(
            target.as_str(),
            "v1/files?pageSize=2&pageToken=next%20token%2F%2B"
        );
    }

    #[tokio::test]
    async fn metadata_lifecycle_uses_stable_v1_shared_transport() {
        let mut server = mockito::Server::new_async().await;
        let get = server
            .mock("GET", "/v1/files/abc-123")
            .match_header("x-goog-api-key", "test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "name": "files/abc-123",
                    "state": "ACTIVE",
                    "source": "UPLOADED"
                })
                .to_string(),
            )
            .create_async()
            .await;
        let list = server
            .mock("GET", "/v1/files")
            .match_query("pageSize=1")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "files": [{
                        "name": "files/abc-123",
                        "state": "ACTIVE",
                        "source": "UPLOADED"
                    }],
                    "nextPageToken": "next"
                })
                .to_string(),
            )
            .create_async()
            .await;
        let delete = server
            .mock("DELETE", "/v1/files/abc-123")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .create_async()
            .await;
        let files = provider(server.url()).files();
        let name = GeminiFileName::new("files/abc-123").unwrap();

        assert_eq!(files.get(&name).await.unwrap().name, name);
        let page = files
            .list(GeminiFileListQuery::new().with_page_size(1).unwrap())
            .await
            .unwrap();
        assert_eq!(page.files.len(), 1);
        assert_eq!(page.next_page_token.as_deref(), Some("next"));
        files.delete(&name).await.unwrap();

        get.assert_async().await;
        list.assert_async().await;
        delete.assert_async().await;
    }
}
