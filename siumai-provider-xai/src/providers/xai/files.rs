//! Provider-owned xAI Files lifecycle.

use std::fmt;

use bytes::Bytes;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde::{Deserialize, Deserializer, Serialize};
use siumai_core::{
    CallOptions, Error, ErrorKind, PublicDiagnosticText, ResponseDiagnostics, SensitiveResponse,
};
use siumai_transport::{
    MultipartBody, MultipartPart, ProviderTransport, ReplaySafety, RequestBody, RequestBuildError,
    RequestHeaders, RequestPlan, RequestTarget, ResponseHeaders, TransportResponse,
};

pub const FILES_SOURCE: &str = "https://docs.x.ai/developers/files/managing-files";
pub const FILES_VERIFIED_ON: &str = "2026-08-09";

const FILES_TARGET: &str = "files";
const MAX_FILE_BYTES: usize = 48 * 1024 * 1024;
const MIN_EXPIRY_SECONDS: u32 = 3_600;
const MAX_EXPIRY_SECONDS: u32 = 2_592_000;

/// Validated xAI file identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct XaiFileId(String);

impl XaiFileId {
    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 256
            || value.chars().any(|character| {
                !character.is_ascii_alphanumeric() && !matches!(character, '-' | '_')
            })
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "xAI file ID is invalid",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for XaiFileId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Typed xAI file upload with optional time-to-live.
#[derive(Clone)]
pub struct XaiFileUpload {
    data: Bytes,
    media_type: String,
    filename: String,
    purpose: String,
    expires_after_seconds: Option<u32>,
}

impl XaiFileUpload {
    pub fn new(
        data: impl Into<Bytes>,
        media_type: impl Into<String>,
        filename: impl Into<String>,
    ) -> Result<Self, Error> {
        let data = data.into();
        if data.is_empty() || data.len() > MAX_FILE_BYTES {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "xAI file upload must contain at most 48 MiB",
            ));
        }
        let media_type = media_type.into();
        validate_text(&media_type, 256, "xAI file media type is invalid")?;
        let filename = filename.into();
        validate_text(&filename, 512, "xAI file name is invalid")?;
        Ok(Self {
            data,
            media_type,
            filename,
            purpose: "assistants".to_string(),
            expires_after_seconds: None,
        })
    }

    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Result<Self, Error> {
        let purpose = purpose.into();
        validate_text(&purpose, 128, "xAI file purpose is invalid")?;
        self.purpose = purpose;
        Ok(self)
    }

    pub fn with_expires_after_seconds(mut self, seconds: u32) -> Result<Self, Error> {
        if !(MIN_EXPIRY_SECONDS..=MAX_EXPIRY_SECONDS).contains(&seconds) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "xAI file expiry must be between one hour and 30 days",
            ));
        }
        self.expires_after_seconds = Some(seconds);
        Ok(self)
    }
}

impl fmt::Debug for XaiFileUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiFileUpload")
            .field("data_bytes", &self.data.len())
            .field("media_type", &self.media_type)
            .field("filename_bytes", &self.filename.len())
            .field("purpose", &self.purpose)
            .field("expires_after_seconds", &self.expires_after_seconds)
            .finish()
    }
}

/// One xAI file metadata object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XaiFile {
    pub id: XaiFileId,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub bytes: Option<u64>,
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub expires_at: Option<i64>,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub purpose: Option<String>,
}

/// Pagination direction for the Files API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XaiFileOrder {
    Ascending,
    Descending,
}

/// Typed file-list controls.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XaiFileListOptions {
    pub limit: Option<u32>,
    pub order: Option<XaiFileOrder>,
    pub pagination_token: Option<String>,
    pub filter: Option<String>,
}

/// One paginated xAI file listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XaiFileList {
    pub data: Vec<XaiFile>,
    #[serde(default)]
    pub pagination_token: Option<String>,
}

/// Result returned by file deletion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XaiDeletedFile {
    pub id: XaiFileId,
    pub deleted: bool,
    #[serde(default)]
    pub object: Option<String>,
}

/// Downloaded xAI file content.
#[derive(Clone, PartialEq, Eq)]
pub struct XaiFileContent {
    pub media_type: Option<String>,
    pub data: Bytes,
    pub request_id: Option<String>,
}

impl fmt::Debug for XaiFileContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiFileContent")
            .field("media_type", &self.media_type)
            .field("data_bytes", &self.data.len())
            .field("request_id", &self.request_id)
            .finish()
    }
}

/// Provider-owned xAI Files resource.
#[derive(Clone)]
pub struct XaiFiles {
    transport: ProviderTransport,
}

impl XaiFiles {
    pub(crate) fn new(transport: ProviderTransport) -> Self {
        Self { transport }
    }

    pub async fn upload(&self, upload: XaiFileUpload, call: CallOptions) -> Result<XaiFile, Error> {
        let mut parts = vec![text_part("purpose", &upload.purpose)?];
        if let Some(seconds) = upload.expires_after_seconds {
            parts.push(text_part("expires_after", &seconds.to_string())?);
        }
        let content_type = HeaderValue::from_str(&upload.media_type).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "xAI file media type cannot be encoded as multipart",
            )
            .with_source(source)
        })?;
        // xAI requires expiry metadata before the file part.
        parts.push(
            MultipartPart::file("file", upload.filename, content_type, upload.data)
                .map_err(request_build_error)?,
        );
        let plan = RequestPlan::new(
            Method::POST,
            RequestTarget::new(FILES_TARGET).map_err(request_build_error)?,
        )
        .with_body(RequestBody::multipart(MultipartBody::new(parts)))
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_build_error)?;
        decode_json(self.execute(plan, call, "xAI file upload failed").await?)
    }

    pub async fn list(
        &self,
        options: XaiFileListOptions,
        call: CallOptions,
    ) -> Result<XaiFileList, Error> {
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        if let Some(limit) = options.limit {
            if limit == 0 || limit > 1_000 {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "xAI file list limit must be between one and 1000",
                ));
            }
            query.append_pair("limit", &limit.to_string());
        }
        if let Some(order) = options.order {
            query.append_pair(
                "order",
                match order {
                    XaiFileOrder::Ascending => "asc",
                    XaiFileOrder::Descending => "desc",
                },
            );
        }
        if let Some(token) = options.pagination_token {
            validate_text(&token, 512, "xAI pagination token is invalid")?;
            query.append_pair("pagination_token", &token);
        }
        if let Some(filter) = options.filter {
            validate_text(&filter, 4_096, "xAI file filter is invalid")?;
            query.append_pair("filter", &filter);
        }
        let query = query.finish();
        let target = if query.is_empty() {
            FILES_TARGET.to_string()
        } else {
            format!("{FILES_TARGET}?{query}")
        };
        let plan = RequestPlan::new(
            Method::GET,
            RequestTarget::new(target).map_err(request_build_error)?,
        )
        .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
        .map_err(request_build_error)?;
        decode_json(self.execute(plan, call, "xAI file list failed").await?)
    }

    pub async fn retrieve(&self, id: &XaiFileId, call: CallOptions) -> Result<XaiFile, Error> {
        self.get_json(id, Method::GET, call, "xAI file retrieval failed")
            .await
    }

    pub async fn delete(&self, id: &XaiFileId, call: CallOptions) -> Result<XaiDeletedFile, Error> {
        self.get_json(id, Method::DELETE, call, "xAI file deletion failed")
            .await
    }

    pub async fn download(
        &self,
        id: &XaiFileId,
        call: CallOptions,
    ) -> Result<XaiFileContent, Error> {
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("*/*"))
            .map_err(request_build_error)?;
        let plan = RequestPlan::new(
            Method::GET,
            RequestTarget::new(format!("files/{}/content", id.as_str()))
                .map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
        .map_err(request_build_error)?;
        let response = self.execute(plan, call, "xAI file download failed").await?;
        let (_, headers, data) = response.into_parts();
        if data.is_empty() {
            return Err(Error::protocol_violation(
                "xAI file download returned an empty body",
            ));
        }
        Ok(XaiFileContent {
            media_type: response_media_type(&headers).map(str::to_string),
            data,
            request_id: response_request_id(&headers),
        })
    }

    async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        id: &XaiFileId,
        method: Method,
        call: CallOptions,
        message: &'static str,
    ) -> Result<T, Error> {
        let plan = RequestPlan::new(
            method.clone(),
            RequestTarget::new(format!("files/{}", id.as_str())).map_err(request_build_error)?,
        )
        .with_replay_safety(if method == Method::GET {
            ReplaySafety::SemanticallyIdempotent
        } else {
            ReplaySafety::Never
        })
        .map_err(request_build_error)?;
        decode_json(self.execute(plan, call, message).await?)
    }

    async fn execute(
        &self,
        plan: RequestPlan,
        call: CallOptions,
        message: &'static str,
    ) -> Result<TransportResponse, Error> {
        let response = self.transport.execute(plan, call).await?;
        if response.status().is_success() {
            Ok(response)
        } else {
            Err(response_error(message, response))
        }
    }
}

impl fmt::Debug for XaiFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiFiles")
            .field("transport", &"shared")
            .finish()
    }
}

fn decode_json<T: serde::de::DeserializeOwned>(response: TransportResponse) -> Result<T, Error> {
    let (_, _, body) = response.into_parts();
    serde_json::from_slice(&body).map_err(|source| {
        Error::new(ErrorKind::Protocol, "xAI Files returned malformed JSON").with_source(source)
    })
}

fn text_part(name: &str, value: &str) -> Result<MultipartPart, Error> {
    MultipartPart::field(name, value.as_bytes().to_vec()).map_err(request_build_error)
}

fn validate_text(value: &str, maximum: usize, message: &'static str) -> Result<(), Error> {
    if value.trim().is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(Error::new(ErrorKind::InvalidInput, message));
    }
    Ok(())
}

fn response_error(message: &'static str, response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let kind = match status {
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::NOT_FOUND => ErrorKind::Provider,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => ErrorKind::InvalidInput,
        _ => ErrorKind::Provider,
    };
    let mut diagnostics = ResponseDiagnostics::default().with_status(status.as_u16());
    if let Some(request_id) = response_request_id(&headers)
        && let Ok(request_id) = PublicDiagnosticText::new(request_id)
    {
        diagnostics = diagnostics.with_request_id(request_id);
    }
    let raw_headers = headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect();
    Error::new(kind, message)
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body.to_vec()))
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    ["x-request-id", "request-id"].into_iter().find_map(|name| {
        headers
            .get(&http::header::HeaderName::from_static(name))
            .and_then(|value| value.to_str().ok())
            .and_then(|value| PublicDiagnosticText::new(value.to_owned()).ok())
            .map(|value| value.as_str().to_owned())
    })
}

fn response_media_type(headers: &ResponseHeaders) -> Option<&str> {
    headers
        .get(&http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "xAI Files request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use mockito::Matcher;
    use siumai_core::{ReplayDomain, ReplayDomainId};
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{XaiCredential, XaiProvider};

    #[tokio::test]
    async fn upload_orders_ttl_before_file_and_lifecycle_is_typed() {
        let mut server = mockito::Server::new_async().await;
        let upload = server
            .mock("POST", "/v1/files")
            .match_body(Matcher::Regex(
                "name=\"purpose\"[\\s\\S]*assistants[\\s\\S]*name=\"expires_after\"[\\s\\S]*7200[\\s\\S]*name=\"file\"".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "id":"file_123",
                    "object":"file",
                    "bytes":4,
                    "created_at":1,
                    "expires_at":7201,
                    "filename":"doc.txt",
                    "purpose":"assistants"
                })
                .to_string(),
            )
            .create_async()
            .await;
        let retrieve = server
            .mock("GET", "/v1/files/file_123")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "id":"file_123",
                    "bytes":4,
                    "filename":"doc.txt"
                })
                .to_string(),
            )
            .create_async()
            .await;
        let delete = server
            .mock("DELETE", "/v1/files/file_123")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"id":"file_123","deleted":true}).to_string())
            .create_async()
            .await;
        let provider = XaiProvider::builder(XaiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.url())).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-xai-files").unwrap(),
            ))
            .build()
            .unwrap();
        let files = provider.files();
        let file = files
            .upload(
                XaiFileUpload::new(Bytes::from_static(b"test"), "text/plain", "doc.txt")
                    .unwrap()
                    .with_expires_after_seconds(7_200)
                    .unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();
        let retrieved = files
            .retrieve(&file.id, CallOptions::default())
            .await
            .unwrap();
        let deleted = files
            .delete(&file.id, CallOptions::default())
            .await
            .unwrap();

        upload.assert_async().await;
        retrieve.assert_async().await;
        delete.assert_async().await;
        assert_eq!(retrieved.filename.as_deref(), Some("doc.txt"));
        assert!(deleted.deleted);
        assert!(serde_json::from_str::<XaiFileId>(r#""../escape""#).is_err());
        let content = XaiFileContent {
            media_type: Some("text/plain".to_string()),
            data: Bytes::from_static(b"private-file-canary"),
            request_id: Some("request-1".to_string()),
        };
        assert!(!format!("{content:?}").contains("private-file-canary"));
    }
}
