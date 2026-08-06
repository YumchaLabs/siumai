use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use http::Method;
use http::header::HeaderValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind};
use siumai_transport::{MultipartBody, MultipartPart, ReplaySafety, RequestBody};

use super::NativeRuntime;
use super::common::{execute_download, execute_json, multipart_body, target, validate_resource_id};

const FILES_BETA: &str = "files-api-2025-04-14";

/// Owned input for one Anthropic file upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicFileUpload {
    filename: String,
    media_type: String,
    data: Bytes,
    purpose: Option<String>,
}

impl AnthropicFileUpload {
    pub fn new(
        filename: impl Into<String>,
        media_type: impl Into<String>,
        data: impl Into<Bytes>,
    ) -> Result<Self, Error> {
        let filename = filename.into();
        let media_type = media_type.into();
        if filename.trim().is_empty()
            || filename.len() > 1_024
            || filename.chars().any(char::is_control)
            || media_type.trim().is_empty()
            || media_type.len() > 256
            || media_type.chars().any(char::is_control)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic file upload metadata is invalid",
            ));
        }
        Ok(Self {
            filename,
            media_type,
            data: data.into(),
            purpose: None,
        })
    }

    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Result<Self, Error> {
        let purpose = purpose.into();
        if purpose.trim().is_empty() || purpose.len() > 256 || purpose.chars().any(char::is_control)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic file purpose is invalid",
            ));
        }
        self.purpose = Some(purpose);
        Ok(self)
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn data(&self) -> &Bytes {
        &self.data
    }

    pub fn purpose(&self) -> Option<&str> {
        self.purpose.as_deref()
    }
}

/// Anthropic file metadata. Unknown additive response fields are retained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicFile {
    pub id: String,
    #[serde(rename = "type", default)]
    pub object_type: Option<String>,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub downloadable: Option<bool>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnthropicFileListQuery {
    pub before_id: Option<String>,
    pub after_id: Option<String>,
    pub limit: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AnthropicFileList {
    #[serde(default)]
    pub data: Vec<AnthropicFile>,
    #[serde(default)]
    pub first_id: Option<String>,
    #[serde(default)]
    pub last_id: Option<String>,
    #[serde(default)]
    pub has_more: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AnthropicFileDeleteResult {
    pub id: String,
    #[serde(default)]
    pub deleted: bool,
    #[serde(rename = "type", default)]
    pub object_type: Option<String>,
}

/// Shared, lightweight Files API handle.
#[derive(Clone)]
pub struct AnthropicFiles {
    runtime: Arc<NativeRuntime>,
}

impl AnthropicFiles {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn upload(&self, upload: AnthropicFileUpload) -> Result<AnthropicFile, Error> {
        self.upload_with_options(upload, CallOptions::default())
            .await
    }

    pub async fn upload_with_options(
        &self,
        upload: AnthropicFileUpload,
        options: CallOptions,
    ) -> Result<AnthropicFile, Error> {
        let media_type = HeaderValue::from_str(upload.media_type()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic file media type is invalid",
            )
            .with_source(source)
        })?;
        let mut parts = vec![
            MultipartPart::file("file", upload.filename, media_type, upload.data).map_err(
                |source| {
                    Error::new(ErrorKind::InvalidInput, "Anthropic file upload is invalid")
                        .with_source(source)
                },
            )?,
        ];
        if let Some(purpose) = upload.purpose {
            parts.push(MultipartPart::field("purpose", purpose).map_err(|source| {
                Error::new(ErrorKind::InvalidInput, "Anthropic file purpose is invalid")
                    .with_source(source)
            })?);
        }
        execute_json(
            &self.runtime,
            Method::POST,
            target("files")?,
            multipart_body(MultipartBody::new(parts)),
            ReplaySafety::Never,
            &[FILES_BETA],
            options,
        )
        .await
    }

    pub async fn list(&self, query: AnthropicFileListQuery) -> Result<AnthropicFileList, Error> {
        self.list_with_options(query, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        query: AnthropicFileListQuery,
        options: CallOptions,
    ) -> Result<AnthropicFileList, Error> {
        execute_json(
            &self.runtime,
            Method::GET,
            target(list_target("files", &query)?)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[FILES_BETA],
            options,
        )
        .await
    }

    pub async fn retrieve(&self, file_id: &str) -> Result<AnthropicFile, Error> {
        self.retrieve_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        file_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicFile, Error> {
        validate_resource_id(file_id)?;
        execute_json(
            &self.runtime,
            Method::GET,
            target(format!("files/{file_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[FILES_BETA],
            options,
        )
        .await
    }

    pub async fn delete(&self, file_id: &str) -> Result<AnthropicFileDeleteResult, Error> {
        self.delete_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        file_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicFileDeleteResult, Error> {
        validate_resource_id(file_id)?;
        execute_json(
            &self.runtime,
            Method::DELETE,
            target(format!("files/{file_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[FILES_BETA],
            options,
        )
        .await
    }

    pub async fn content(&self, file_id: &str) -> Result<Bytes, Error> {
        self.content_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn content_with_options(
        &self,
        file_id: &str,
        options: CallOptions,
    ) -> Result<Bytes, Error> {
        validate_resource_id(file_id)?;
        execute_download(
            &self.runtime,
            target(format!("files/{file_id}/content"))?,
            &[FILES_BETA],
            "application/octet-stream",
            options,
        )
        .await
    }
}

impl std::fmt::Debug for AnthropicFiles {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicFiles")
            .field("runtime", &"shared")
            .finish()
    }
}

fn list_target(prefix: &str, query: &AnthropicFileListQuery) -> Result<String, Error> {
    let mut pairs = Vec::new();
    if let Some(before) = &query.before_id {
        validate_resource_id(before)?;
        pairs.push(format!("before_id={before}"));
    }
    if let Some(after) = &query.after_id {
        validate_resource_id(after)?;
        pairs.push(format!("after_id={after}"));
    }
    if let Some(limit) = query.limit {
        if limit == 0 || limit > 1_000 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic list limit must be between 1 and 1000",
            ));
        }
        pairs.push(format!("limit={limit}"));
    }
    Ok(if pairs.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}?{}", pairs.join("&"))
    })
}
