use std::fmt;
use std::sync::Arc;

use http::Method;
use http::header::HeaderValue;
use siumai_core::{CallOptions, Error};
use siumai_protocol_openai::experimental::skills::{
    OpenAiDeletedSkill, OpenAiDeletedSkillVersion, OpenAiSkill, OpenAiSkillUpdateRequest,
    OpenAiSkillVersion,
};
use siumai_protocol_openai::resources::{OpenAiCursorPage, OpenAiListOrder};
use siumai_transport::{MultipartBody, MultipartPart, ReplaySafety, RequestBody};

use super::super::provider::{OpenAiProvider, OpenAiRuntime};
use super::common::{
    OpenAiBinaryContent, OpenAiNativeRuntime, invalid_input, json_body, target, target_with_query,
    target_with_segments, target_with_segments_and_query, validate_bounded_text,
    validate_resource_id,
};

const MAX_SKILL_FILE_PATH_BYTES: usize = 1_024;

/// One file included in an OpenAI skill upload.
#[derive(Clone)]
pub struct OpenAiSkillFile {
    pub path: String,
    pub media_type: String,
    pub data: Vec<u8>,
}

impl OpenAiSkillFile {
    pub fn new(
        path: impl Into<String>,
        media_type: impl Into<String>,
        data: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            path: path.into(),
            media_type: media_type.into(),
            data: data.into(),
        }
    }

    fn validate(&self) -> Result<HeaderValue, Error> {
        validate_bounded_text(
            &self.path,
            MAX_SKILL_FILE_PATH_BYTES,
            "OpenAI skill file path is invalid",
        )?;
        if self.data.is_empty() {
            return Err(invalid_input("OpenAI skill file cannot be empty"));
        }
        HeaderValue::from_str(&self.media_type).map_err(|source| {
            invalid_input("OpenAI skill media type is invalid").with_source(source)
        })
    }
}

impl fmt::Debug for OpenAiSkillFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiSkillFile")
            .field("path", &self.path)
            .field("media_type", &self.media_type)
            .field("data_bytes", &self.data.len())
            .finish()
    }
}

/// Multipart files used to create a skill.
#[derive(Clone)]
pub struct OpenAiSkillUpload {
    pub files: Vec<OpenAiSkillFile>,
}

impl OpenAiSkillUpload {
    pub fn new(files: impl IntoIterator<Item = OpenAiSkillFile>) -> Self {
        Self {
            files: files.into_iter().collect(),
        }
    }
}

impl fmt::Debug for OpenAiSkillUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiSkillUpload")
            .field("file_count", &self.files.len())
            .finish()
    }
}

/// Multipart files used to create one immutable skill version.
#[derive(Clone)]
pub struct OpenAiSkillVersionUpload {
    pub files: Vec<OpenAiSkillFile>,
    pub make_default: Option<bool>,
}

impl OpenAiSkillVersionUpload {
    pub fn new(files: impl IntoIterator<Item = OpenAiSkillFile>) -> Self {
        Self {
            files: files.into_iter().collect(),
            make_default: None,
        }
    }

    pub fn make_default(mut self, make_default: bool) -> Self {
        self.make_default = Some(make_default);
        self
    }
}

impl fmt::Debug for OpenAiSkillVersionUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiSkillVersionUpload")
            .field("file_count", &self.files.len())
            .field("make_default", &self.make_default)
            .finish()
    }
}

/// Cursor options shared by skill and skill-version list endpoints.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiSkillListOptions {
    pub after: Option<String>,
    pub limit: Option<u16>,
    pub order: Option<OpenAiListOrder>,
}

impl OpenAiSkillListOptions {
    fn validate(&self) -> Result<(), Error> {
        if let Some(after) = &self.after {
            validate_resource_id(after)?;
        }
        if self.limit == Some(0) {
            return Err(invalid_input(
                "OpenAI skill list limit must be greater than zero",
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
        pairs
    }
}

/// Provider-owned OpenAI Skills and immutable version lifecycle client.
#[derive(Clone)]
pub struct OpenAiSkills {
    runtime: OpenAiNativeRuntime,
}

/// Experimental Skills lifecycle access for [`OpenAiProvider`].
///
/// Import this trait from `siumai_provider_openai::experimental::skills` to opt into the
/// unstable provider-native resource contract without widening the stable provider surface.
pub trait OpenAiSkillsProviderExt {
    fn skills(&self) -> OpenAiSkills;
}

impl OpenAiSkillsProviderExt for OpenAiProvider {
    fn skills(&self) -> OpenAiSkills {
        OpenAiSkills::new(self.runtime.clone())
    }
}

impl OpenAiSkills {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>) -> Self {
        Self {
            runtime: OpenAiNativeRuntime::new(runtime),
        }
    }

    pub async fn create(&self, upload: OpenAiSkillUpload) -> Result<OpenAiSkill, Error> {
        self.create_with_options(upload, CallOptions::default())
            .await
    }

    pub async fn create_with_options(
        &self,
        upload: OpenAiSkillUpload,
        options: CallOptions,
    ) -> Result<OpenAiSkill, Error> {
        self.runtime
            .execute_json(
                Method::POST,
                target("skills")?,
                skill_multipart(upload.files, None)?,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn list(
        &self,
        list: OpenAiSkillListOptions,
    ) -> Result<OpenAiCursorPage<OpenAiSkill>, Error> {
        self.list_with_options(list, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        list: OpenAiSkillListOptions,
        options: CallOptions,
    ) -> Result<OpenAiCursorPage<OpenAiSkill>, Error> {
        list.validate()?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_query("skills", list.query())?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn retrieve(&self, skill_id: &str) -> Result<OpenAiSkill, Error> {
        self.retrieve_with_options(skill_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        skill_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiSkill, Error> {
        validate_resource_id(skill_id)?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_segments("skills", [skill_id])?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn update(
        &self,
        skill_id: &str,
        request: OpenAiSkillUpdateRequest,
    ) -> Result<OpenAiSkill, Error> {
        self.update_with_options(skill_id, request, CallOptions::default())
            .await
    }

    pub async fn update_with_options(
        &self,
        skill_id: &str,
        request: OpenAiSkillUpdateRequest,
        options: CallOptions,
    ) -> Result<OpenAiSkill, Error> {
        validate_resource_id(skill_id)?;
        validate_resource_id(&request.default_version)?;
        self.runtime
            .execute_json(
                Method::POST,
                target_with_segments("skills", [skill_id])?,
                json_body(&request)?,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn content(&self, skill_id: &str) -> Result<OpenAiBinaryContent, Error> {
        self.content_with_options(skill_id, CallOptions::default())
            .await
    }

    pub async fn content_with_options(
        &self,
        skill_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiBinaryContent, Error> {
        validate_resource_id(skill_id)?;
        self.runtime
            .execute_bytes(
                target_with_segments("skills", [skill_id, "content"])?,
                "application/octet-stream",
                options,
            )
            .await
    }

    pub async fn delete(&self, skill_id: &str) -> Result<OpenAiDeletedSkill, Error> {
        self.delete_with_options(skill_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        skill_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiDeletedSkill, Error> {
        validate_resource_id(skill_id)?;
        self.runtime
            .execute_json(
                Method::DELETE,
                target_with_segments("skills", [skill_id])?,
                RequestBody::Empty,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn create_version(
        &self,
        skill_id: &str,
        upload: OpenAiSkillVersionUpload,
    ) -> Result<OpenAiSkillVersion, Error> {
        self.create_version_with_options(skill_id, upload, CallOptions::default())
            .await
    }

    pub async fn create_version_with_options(
        &self,
        skill_id: &str,
        upload: OpenAiSkillVersionUpload,
        options: CallOptions,
    ) -> Result<OpenAiSkillVersion, Error> {
        validate_resource_id(skill_id)?;
        self.runtime
            .execute_json(
                Method::POST,
                target_with_segments("skills", [skill_id, "versions"])?,
                skill_multipart(upload.files, upload.make_default)?,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn list_versions(
        &self,
        skill_id: &str,
        list: OpenAiSkillListOptions,
    ) -> Result<OpenAiCursorPage<OpenAiSkillVersion>, Error> {
        self.list_versions_with_options(skill_id, list, CallOptions::default())
            .await
    }

    pub async fn list_versions_with_options(
        &self,
        skill_id: &str,
        list: OpenAiSkillListOptions,
        options: CallOptions,
    ) -> Result<OpenAiCursorPage<OpenAiSkillVersion>, Error> {
        validate_resource_id(skill_id)?;
        list.validate()?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_segments_and_query("skills", [skill_id, "versions"], list.query())?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn retrieve_version(
        &self,
        skill_id: &str,
        version: &str,
    ) -> Result<OpenAiSkillVersion, Error> {
        self.retrieve_version_with_options(skill_id, version, CallOptions::default())
            .await
    }

    pub async fn retrieve_version_with_options(
        &self,
        skill_id: &str,
        version: &str,
        options: CallOptions,
    ) -> Result<OpenAiSkillVersion, Error> {
        validate_skill_version_ids(skill_id, version)?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_segments("skills", [skill_id, "versions", version])?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn version_content(
        &self,
        skill_id: &str,
        version: &str,
    ) -> Result<OpenAiBinaryContent, Error> {
        self.version_content_with_options(skill_id, version, CallOptions::default())
            .await
    }

    pub async fn version_content_with_options(
        &self,
        skill_id: &str,
        version: &str,
        options: CallOptions,
    ) -> Result<OpenAiBinaryContent, Error> {
        validate_skill_version_ids(skill_id, version)?;
        self.runtime
            .execute_bytes(
                target_with_segments("skills", [skill_id, "versions", version, "content"])?,
                "application/octet-stream",
                options,
            )
            .await
    }

    pub async fn delete_version(
        &self,
        skill_id: &str,
        version: &str,
    ) -> Result<OpenAiDeletedSkillVersion, Error> {
        self.delete_version_with_options(skill_id, version, CallOptions::default())
            .await
    }

    pub async fn delete_version_with_options(
        &self,
        skill_id: &str,
        version: &str,
        options: CallOptions,
    ) -> Result<OpenAiDeletedSkillVersion, Error> {
        validate_skill_version_ids(skill_id, version)?;
        self.runtime
            .execute_json(
                Method::DELETE,
                target_with_segments("skills", [skill_id, "versions", version])?,
                RequestBody::Empty,
                ReplaySafety::Never,
                options,
            )
            .await
    }
}

impl fmt::Debug for OpenAiSkills {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiSkills")
            .field("runtime", &self.runtime)
            .finish()
    }
}

fn skill_multipart(
    files: Vec<OpenAiSkillFile>,
    make_default: Option<bool>,
) -> Result<RequestBody, Error> {
    if files.is_empty() {
        return Err(invalid_input(
            "OpenAI skill upload requires at least one file",
        ));
    }
    let mut parts = Vec::with_capacity(files.len() + usize::from(make_default.is_some()));
    for file in files {
        let media_type = file.validate()?;
        parts.push(
            MultipartPart::file("files[]", file.path, media_type, file.data).map_err(|source| {
                invalid_input("OpenAI skill multipart body is invalid").with_source(source)
            })?,
        );
    }
    if let Some(make_default) = make_default {
        parts.push(
            MultipartPart::field("default", make_default.to_string().into_bytes()).map_err(
                |source| {
                    invalid_input("OpenAI skill multipart body is invalid").with_source(source)
                },
            )?,
        );
    }
    Ok(RequestBody::multipart(MultipartBody::new(parts)))
}

fn validate_skill_version_ids(skill_id: &str, version: &str) -> Result<(), Error> {
    validate_resource_id(skill_id)?;
    validate_resource_id(version)
}
