use std::fmt;
use std::sync::Arc;

use http::Method;
use http::header::HeaderValue;
use siumai_core::{CallOptions, Error};
use siumai_protocol_openai::resources::{
    OpenAiCursorPage, OpenAiFile, OpenAiFileDeleted, OpenAiFileExpiresAfter, OpenAiFilePurpose,
    OpenAiListOrder,
};
use siumai_transport::{MultipartBody, MultipartPart, ReplaySafety, RequestBody};

use super::super::provider::OpenAiRuntime;
use super::common::{
    OpenAiBinaryContent, OpenAiNativeRuntime, invalid_input, target, target_with_query,
    validate_bounded_text, validate_file_expiration, validate_resource_id,
};

const MAX_FILE_NAME_BYTES: usize = 1_024;

/// One bounded file upload for the OpenAI Files API.
#[derive(Clone)]
pub struct OpenAiFileUpload {
    pub filename: String,
    pub media_type: String,
    pub data: Vec<u8>,
    pub purpose: OpenAiFilePurpose,
    pub expires_after: Option<OpenAiFileExpiresAfter>,
}

impl OpenAiFileUpload {
    pub fn new(
        filename: impl Into<String>,
        media_type: impl Into<String>,
        data: impl Into<Vec<u8>>,
        purpose: OpenAiFilePurpose,
    ) -> Self {
        Self {
            filename: filename.into(),
            media_type: media_type.into(),
            data: data.into(),
            purpose,
            expires_after: None,
        }
    }

    pub fn with_expiration(mut self, expires_after: OpenAiFileExpiresAfter) -> Self {
        self.expires_after = Some(expires_after);
        self
    }

    fn validate(&self) -> Result<HeaderValue, Error> {
        validate_bounded_text(
            &self.filename,
            MAX_FILE_NAME_BYTES,
            "OpenAI upload filename is invalid",
        )?;
        if self.data.is_empty() {
            return Err(invalid_input("OpenAI upload file cannot be empty"));
        }
        let media_type = HeaderValue::from_str(&self.media_type).map_err(|source| {
            invalid_input("OpenAI upload media type is invalid").with_source(source)
        })?;
        if let Some(expires_after) = self.expires_after.as_ref() {
            validate_file_expiration(expires_after)?;
        }
        Ok(media_type)
    }
}

impl fmt::Debug for OpenAiFileUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiFileUpload")
            .field("filename", &self.filename)
            .field("media_type", &self.media_type)
            .field("data_bytes", &self.data.len())
            .field("purpose", &self.purpose)
            .field("expires_after", &self.expires_after)
            .finish()
    }
}

/// Typed filters for listing OpenAI files.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiFileListOptions {
    pub after: Option<String>,
    pub limit: Option<u16>,
    pub order: Option<OpenAiListOrder>,
    pub purpose: Option<OpenAiFilePurpose>,
}

impl OpenAiFileListOptions {
    fn validate(&self) -> Result<(), Error> {
        if let Some(after) = &self.after {
            validate_resource_id(after)?;
        }
        if self.limit.is_some_and(|limit| limit == 0 || limit > 10_000) {
            return Err(invalid_input(
                "OpenAI file list limit must be between 1 and 10000",
            ));
        }
        Ok(())
    }

    fn query(self) -> Vec<(&'static str, String)> {
        let mut pairs = Vec::new();
        if let Some(after) = self.after {
            pairs.push(("after", after));
        }
        if let Some(limit) = self.limit {
            pairs.push(("limit", limit.to_string()));
        }
        if let Some(order) = self.order {
            pairs.push(("order", order.as_str().to_string()));
        }
        if let Some(purpose) = self.purpose {
            pairs.push(("purpose", purpose.as_str().to_string()));
        }
        pairs
    }
}

/// Provider-owned OpenAI Files lifecycle client.
#[derive(Clone)]
pub struct OpenAiFiles {
    runtime: OpenAiNativeRuntime,
}

impl OpenAiFiles {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>) -> Self {
        Self {
            runtime: OpenAiNativeRuntime::new(runtime),
        }
    }

    pub async fn upload(&self, upload: OpenAiFileUpload) -> Result<OpenAiFile, Error> {
        self.upload_with_options(upload, CallOptions::default())
            .await
    }

    pub async fn upload_with_options(
        &self,
        upload: OpenAiFileUpload,
        options: CallOptions,
    ) -> Result<OpenAiFile, Error> {
        let media_type = upload.validate()?;
        let mut parts = vec![
            MultipartPart::file("file", upload.filename, media_type, upload.data).map_err(
                |source| invalid_input("OpenAI file multipart body is invalid").with_source(source),
            )?,
            MultipartPart::field("purpose", upload.purpose.as_str().as_bytes().to_vec()).map_err(
                |source| invalid_input("OpenAI file multipart body is invalid").with_source(source),
            )?,
        ];
        if let Some(expires_after) = upload.expires_after {
            parts.push(
                MultipartPart::field("expires_after[anchor]", "created_at".as_bytes().to_vec())
                    .map_err(|source| {
                        invalid_input("OpenAI file multipart body is invalid").with_source(source)
                    })?,
            );
            parts.push(
                MultipartPart::field(
                    "expires_after[seconds]",
                    expires_after.seconds.to_string().into_bytes(),
                )
                .map_err(|source| {
                    invalid_input("OpenAI file multipart body is invalid").with_source(source)
                })?,
            );
        }
        self.runtime
            .execute_json(
                Method::POST,
                target("files")?,
                RequestBody::multipart(MultipartBody::new(parts)),
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn list(
        &self,
        list: OpenAiFileListOptions,
    ) -> Result<OpenAiCursorPage<OpenAiFile>, Error> {
        self.list_with_options(list, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        list: OpenAiFileListOptions,
        options: CallOptions,
    ) -> Result<OpenAiCursorPage<OpenAiFile>, Error> {
        list.validate()?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_query("files", list.query())?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn retrieve(&self, file_id: &str) -> Result<OpenAiFile, Error> {
        self.retrieve_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        file_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiFile, Error> {
        validate_resource_id(file_id)?;
        self.runtime
            .execute_json(
                Method::GET,
                target(format!("files/{file_id}"))?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn content(&self, file_id: &str) -> Result<OpenAiBinaryContent, Error> {
        self.content_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn content_with_options(
        &self,
        file_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiBinaryContent, Error> {
        validate_resource_id(file_id)?;
        self.runtime
            .execute_bytes(
                target(format!("files/{file_id}/content"))?,
                "application/octet-stream",
                options,
            )
            .await
    }

    pub async fn delete(&self, file_id: &str) -> Result<OpenAiFileDeleted, Error> {
        self.delete_with_options(file_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        file_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiFileDeleted, Error> {
        validate_resource_id(file_id)?;
        self.runtime
            .execute_json(
                Method::DELETE,
                target(format!("files/{file_id}"))?,
                RequestBody::Empty,
                ReplaySafety::Never,
                options,
            )
            .await
    }
}

impl fmt::Debug for OpenAiFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiFiles")
            .field("runtime", &self.runtime)
            .finish()
    }
}
