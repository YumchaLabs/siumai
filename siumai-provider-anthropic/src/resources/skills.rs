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
use super::common::{execute_json, multipart_body, target, validate_resource_id};

const SKILLS_BETA: &str = "skills-2025-10-02";
const MAX_SKILL_FILES: usize = 1_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicSkillFile {
    path: String,
    media_type: String,
    data: Bytes,
}

impl AnthropicSkillFile {
    pub fn new(
        path: impl Into<String>,
        media_type: impl Into<String>,
        data: impl Into<Bytes>,
    ) -> Result<Self, Error> {
        let path = path.into();
        let media_type = media_type.into();
        if path.trim().is_empty()
            || path.len() > 2_048
            || path.chars().any(char::is_control)
            || path.starts_with('/')
            || path.split('/').any(|segment| segment == "..")
            || media_type.trim().is_empty()
            || media_type.len() > 256
            || media_type.chars().any(char::is_control)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic skill file metadata is invalid",
            ));
        }
        Ok(Self {
            path,
            media_type,
            data: data.into(),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn data(&self) -> &Bytes {
        &self.data
    }
}

#[derive(Debug, Clone)]
pub struct AnthropicSkillUpload {
    files: Vec<AnthropicSkillFile>,
    display_title: Option<String>,
}

impl AnthropicSkillUpload {
    pub fn new(files: Vec<AnthropicSkillFile>) -> Result<Self, Error> {
        if files.is_empty() || files.len() > MAX_SKILL_FILES {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic skill upload file count is outside the supported bounds",
            ));
        }
        Ok(Self {
            files,
            display_title: None,
        })
    }

    pub fn with_display_title(mut self, title: impl Into<String>) -> Result<Self, Error> {
        let title = title.into();
        if title.trim().is_empty() || title.len() > 512 || title.chars().any(char::is_control) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic skill display title is invalid",
            ));
        }
        self.display_title = Some(title);
        Ok(self)
    }

    pub fn files(&self) -> &[AnthropicSkillFile] {
        &self.files
    }

    pub fn display_title(&self) -> Option<&str> {
        self.display_title.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicSkill {
    pub id: String,
    #[serde(rename = "type", default)]
    pub object_type: Option<String>,
    #[serde(default)]
    pub display_title: Option<String>,
    #[serde(default)]
    pub latest_version: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicSkillVersion {
    #[serde(default)]
    pub skill_id: Option<String>,
    pub version: String,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AnthropicSkillVersionList {
    #[serde(default)]
    pub data: Vec<AnthropicSkillVersion>,
    #[serde(default)]
    pub has_more: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AnthropicSkillUploadResult {
    #[serde(flatten)]
    pub skill: AnthropicSkill,
}

/// Shared, lightweight Skills API handle.
#[derive(Clone)]
pub struct AnthropicSkills {
    runtime: Arc<NativeRuntime>,
}

impl AnthropicSkills {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn upload(
        &self,
        upload: AnthropicSkillUpload,
    ) -> Result<AnthropicSkillUploadResult, Error> {
        self.upload_with_options(upload, CallOptions::default())
            .await
    }

    pub async fn upload_with_options(
        &self,
        upload: AnthropicSkillUpload,
        options: CallOptions,
    ) -> Result<AnthropicSkillUploadResult, Error> {
        let mut parts = Vec::with_capacity(upload.files.len().saturating_add(1));
        if let Some(title) = upload.display_title {
            parts.push(
                MultipartPart::field("display_title", title).map_err(|source| {
                    Error::new(
                        ErrorKind::InvalidInput,
                        "Anthropic skill display title is invalid",
                    )
                    .with_source(source)
                })?,
            );
        }
        for file in upload.files {
            let media_type = HeaderValue::from_str(&file.media_type).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Anthropic skill file media type is invalid",
                )
                .with_source(source)
            })?;
            parts.push(
                MultipartPart::file("files[]", file.path, media_type, file.data).map_err(
                    |source| {
                        Error::new(
                            ErrorKind::InvalidInput,
                            "Anthropic skill multipart body is invalid",
                        )
                        .with_source(source)
                    },
                )?,
            );
        }
        execute_json(
            &self.runtime,
            Method::POST,
            target("skills")?,
            multipart_body(MultipartBody::new(parts)),
            ReplaySafety::Never,
            &[SKILLS_BETA],
            options,
        )
        .await
    }

    pub async fn retrieve(&self, skill_id: &str) -> Result<AnthropicSkill, Error> {
        self.retrieve_with_options(skill_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        skill_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicSkill, Error> {
        validate_resource_id(skill_id)?;
        execute_json(
            &self.runtime,
            Method::GET,
            target(format!("skills/{skill_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[SKILLS_BETA],
            options,
        )
        .await
    }

    pub async fn versions(&self, skill_id: &str) -> Result<AnthropicSkillVersionList, Error> {
        self.versions_with_options(skill_id, CallOptions::default())
            .await
    }

    pub async fn versions_with_options(
        &self,
        skill_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicSkillVersionList, Error> {
        validate_resource_id(skill_id)?;
        execute_json(
            &self.runtime,
            Method::GET,
            target(format!("skills/{skill_id}/versions"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[SKILLS_BETA],
            options,
        )
        .await
    }

    pub async fn version(
        &self,
        skill_id: &str,
        version: &str,
    ) -> Result<AnthropicSkillVersion, Error> {
        self.version_with_options(skill_id, version, CallOptions::default())
            .await
    }

    pub async fn version_with_options(
        &self,
        skill_id: &str,
        version: &str,
        options: CallOptions,
    ) -> Result<AnthropicSkillVersion, Error> {
        validate_resource_id(skill_id)?;
        validate_resource_id(version)?;
        execute_json(
            &self.runtime,
            Method::GET,
            target(format!("skills/{skill_id}/versions/{version}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[SKILLS_BETA],
            options,
        )
        .await
    }
}

impl std::fmt::Debug for AnthropicSkills {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicSkills")
            .field("runtime", &"shared")
            .finish()
    }
}
