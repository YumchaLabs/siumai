use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use http::Method;
use http::header::HeaderValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind};
use siumai_transport::{MultipartBody, MultipartPart, ReplaySafety, RequestBody, RequestTarget};

use super::NativeRuntime;
use super::common::{execute_download, execute_json, multipart_body, target};
use crate::annotations::AnthropicFileReference;

const FILES_BETA: &str = "files-api-2025-04-14";
const MAX_FILE_BYTES: usize = 500_000_000;
const MAX_FILE_ID_BYTES: usize = 512;

/// Owned input for one Anthropic file upload.
#[derive(Clone, PartialEq, Eq)]
pub struct AnthropicFileUpload {
    filename: String,
    media_type: String,
    data: Bytes,
}

impl AnthropicFileUpload {
    pub fn new(
        filename: impl Into<String>,
        media_type: impl Into<String>,
        data: impl Into<Bytes>,
    ) -> Result<Self, Error> {
        let filename = filename.into();
        let media_type = media_type.into();
        let data = data.into();
        let filename_chars = filename.chars().count();
        if filename_chars == 0
            || filename_chars > 255
            || matches!(filename.as_str(), "." | "..")
            || filename.chars().any(|character| {
                character.is_control()
                    || matches!(
                        character,
                        '<' | '>' | ':' | '"' | '|' | '?' | '*' | '\\' | '/'
                    )
            })
            || media_type.trim().is_empty()
            || media_type.len() > 256
            || media_type.chars().any(char::is_control)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic file upload metadata is invalid",
            ));
        }
        validate_file_size(data.len())?;
        Ok(Self {
            filename,
            media_type,
            data,
        })
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
}

impl fmt::Debug for AnthropicFileUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicFileUpload")
            .field("filename_bytes", &self.filename.len())
            .field("media_type_bytes", &self.media_type.len())
            .field("data_bytes", &self.data.len())
            .finish()
    }
}

/// Anthropic file metadata. Unknown additive response fields are retained.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
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

impl fmt::Debug for AnthropicFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicFile")
            .field("id_bytes", &self.id.len())
            .field("object_type_present", &self.object_type.is_some())
            .field("filename_bytes", &self.filename.as_ref().map(String::len))
            .field("mime_type_bytes", &self.mime_type.as_ref().map(String::len))
            .field("size_bytes", &self.size_bytes)
            .field("created_at_present", &self.created_at.is_some())
            .field("downloadable", &self.downloadable)
            .field("extra_fields", &self.extra.len())
            .finish()
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct AnthropicFileListQuery {
    pub before_id: Option<String>,
    pub after_id: Option<String>,
    pub limit: Option<u16>,
}

impl fmt::Debug for AnthropicFileListQuery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicFileListQuery")
            .field("before_id_bytes", &self.before_id.as_ref().map(String::len))
            .field("after_id_bytes", &self.after_id.as_ref().map(String::len))
            .field("limit", &self.limit)
            .finish()
    }
}

#[derive(Clone, PartialEq, Deserialize)]
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

impl fmt::Debug for AnthropicFileList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicFileList")
            .field("file_count", &self.data.len())
            .field("first_id_present", &self.first_id.is_some())
            .field("last_id_present", &self.last_id.is_some())
            .field("has_more", &self.has_more)
            .finish()
    }
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct AnthropicFileDeleteResult {
    pub id: String,
    #[serde(default)]
    pub deleted: bool,
    #[serde(rename = "type", default)]
    pub object_type: Option<String>,
}

impl fmt::Debug for AnthropicFileDeleteResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicFileDeleteResult")
            .field("id_bytes", &self.id.len())
            .field("deleted", &self.deleted)
            .field("object_type_present", &self.object_type.is_some())
            .finish()
    }
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

    /// Bind one opaque file ID to this provider's durable Messages replay scope.
    pub fn reference(&self, file_id: impl Into<String>) -> Result<AnthropicFileReference, Error> {
        AnthropicFileReference::bind(file_id, self.runtime.scope.as_ref().clone()).map_err(
            |source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Anthropic file reference is not valid for durable message replay",
                )
                .with_source(source)
            },
        )
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
        let parts = vec![
            MultipartPart::file("file", upload.filename, media_type, upload.data).map_err(
                |source| {
                    Error::new(ErrorKind::InvalidInput, "Anthropic file upload is invalid")
                        .with_source(source)
                },
            )?,
        ];
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
        execute_json(
            &self.runtime,
            Method::GET,
            file_target(file_id)?,
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
        execute_json(
            &self.runtime,
            Method::DELETE,
            file_target(file_id)?,
            RequestBody::Empty,
            ReplaySafety::Never,
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
        execute_download(
            &self.runtime,
            file_target(file_id)?
                .with_opaque_path_segment("content")
                .map_err(target_error)?,
            &[FILES_BETA],
            "application/octet-stream",
            options,
        )
        .await
    }
}

impl fmt::Debug for AnthropicFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicFiles")
            .field("runtime", &"shared")
            .finish()
    }
}

fn validate_file_size(bytes: usize) -> Result<(), Error> {
    if bytes > MAX_FILE_BYTES {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic file upload exceeds the single-file size limit",
        ));
    }
    Ok(())
}

fn list_target(prefix: &str, query: &AnthropicFileListQuery) -> Result<String, Error> {
    let mut pairs = url::form_urlencoded::Serializer::new(String::new());
    if let Some(before) = &query.before_id {
        validate_file_id(before)?;
        pairs.append_pair("before_id", before);
    }
    if let Some(after) = &query.after_id {
        validate_file_id(after)?;
        pairs.append_pair("after_id", after);
    }
    if let Some(limit) = query.limit {
        if limit == 0 || limit > 1_000 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic list limit must be between 1 and 1000",
            ));
        }
        pairs.append_pair("limit", &limit.to_string());
    }
    let query = pairs.finish();
    Ok(if query.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}?{query}")
    })
}

fn validate_file_id(file_id: &str) -> Result<(), Error> {
    if file_id.is_empty()
        || file_id.len() > MAX_FILE_ID_BYTES
        || file_id.chars().any(char::is_control)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic file identifier is invalid",
        ));
    }
    Ok(())
}

fn file_target(file_id: &str) -> Result<RequestTarget, Error> {
    validate_file_id(file_id)?;
    target("files")?
        .with_opaque_path_segment(file_id)
        .map_err(target_error)
}

fn target_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(ErrorKind::InvalidInput, "Anthropic file target is invalid").with_source(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_contract_matches_documented_filename_and_size_boundaries() {
        assert!(AnthropicFileUpload::new("a", "application/octet-stream", Bytes::new()).is_ok());
        assert!(AnthropicFileUpload::new("文".repeat(255), "text/plain", Bytes::new()).is_ok());
        assert!(AnthropicFileUpload::new("文".repeat(256), "text/plain", Bytes::new()).is_err());
        for filename in ["", ".", "..", "a/b", "a\\b", "a:b", "a?b", "a\0b"] {
            assert!(AnthropicFileUpload::new(filename, "text/plain", Bytes::new()).is_err());
        }
        assert!(validate_file_size(MAX_FILE_BYTES).is_ok());
        assert!(validate_file_size(MAX_FILE_BYTES + 1).is_err());
    }

    #[test]
    fn list_query_encodes_opaque_identifiers_without_disclosing_them_in_debug() {
        let sentinel = "file/secret?#资源";
        let query = AnthropicFileListQuery {
            before_id: Some(sentinel.to_string()),
            after_id: None,
            limit: Some(50),
        };
        assert_eq!(
            list_target("files", &query).expect("list target"),
            "files?before_id=file%2Fsecret%3F%23%E8%B5%84%E6%BA%90&limit=50"
        );
        assert!(!format!("{query:?}").contains(sentinel));
    }

    #[test]
    fn resource_targets_encode_opaque_identifiers_once() {
        for (file_id, expected) in [
            ("file/segment", "files/file%2Fsegment"),
            ("file\\segment", "files/file%5Csegment"),
            ("file?#fragment", "files/file%3F%23fragment"),
            ("file%2Fsegment", "files/file%252Fsegment"),
            ("file%252Fsegment", "files/file%25252Fsegment"),
            ("文件", "files/%E6%96%87%E4%BB%B6"),
        ] {
            let target = file_target(file_id).expect("opaque file target");
            assert_eq!(target.as_str(), expected);
            let content = target
                .with_opaque_path_segment("content")
                .expect("content subresource");
            assert_eq!(content.as_str(), format!("{expected}/content"));
        }
        for invalid in ["", ".", "..", "file\0id"] {
            assert!(file_target(invalid).is_err());
        }
    }

    #[test]
    fn file_diagnostics_are_structural() {
        let sentinel = "private-file-sentinel";
        let file = AnthropicFile {
            id: sentinel.to_string(),
            object_type: Some(sentinel.to_string()),
            filename: Some(sentinel.to_string()),
            mime_type: Some("text/plain".to_string()),
            size_bytes: Some(1),
            created_at: Some(sentinel.to_string()),
            downloadable: Some(true),
            extra: BTreeMap::from([("private".to_string(), Value::String(sentinel.to_string()))]),
        };
        assert!(!format!("{file:?}").contains(sentinel));

        let deleted = AnthropicFileDeleteResult {
            id: sentinel.to_string(),
            deleted: true,
            object_type: Some(sentinel.to_string()),
        };
        assert!(!format!("{deleted:?}").contains(sentinel));
    }
}
