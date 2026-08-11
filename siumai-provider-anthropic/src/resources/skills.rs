use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use http::Method;
use http::header::HeaderValue;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind};
use siumai_transport::{MultipartBody, MultipartPart, ReplaySafety, RequestBody, RequestTarget};
use unicode_normalization::UnicodeNormalization;

use super::NativeRuntime;
use super::common::{execute_json, multipart_body, target};

const SKILLS_BETA: &str = "skills-2025-10-02";
const SKILLS_TARGET: &str = "skills?beta=true";

// These are Siumai-owned bounds for metadata and in-memory multipart assembly.
const MAX_SKILL_FILES: usize = 1_000;
const MAX_SKILL_FILE_PATH_BYTES: usize = 2_048;
const MAX_SKILL_TITLE_BYTES: usize = 512;
const MAX_SKILL_ID_BYTES: usize = 512;
const MAX_SKILL_VERSION_BYTES: usize = 256;
const MAX_SKILL_PAGE_BYTES: usize = 2_048;
const MAX_SKILL_RESPONSE_VALUE_BYTES: usize = 128;

// Anthropic documents a total uncompressed upload size below 30 MB. ZIP content
// is intentionally opaque here, so its encoded bytes receive a separate local
// safety bound before the provider performs archive validation.
const OFFICIAL_MAX_SKILL_UNCOMPRESSED_BYTES: usize = 30_000_000;
const SIUMAI_MAX_SKILL_ARCHIVE_BYTES: usize = 30_000_000;

/// The provider-owned source of a Skill.
///
/// Anthropic currently documents `custom` and `anthropic`. The open wrapper
/// preserves future source values without turning the current list into an
/// allowlist. Its 128-byte, control-free bound is a Siumai response-safety
/// policy rather than an Anthropic source limit.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AnthropicSkillSource(String);

impl AnthropicSkillSource {
    pub const CUSTOM: &'static str = "custom";
    pub const ANTHROPIC: &'static str = "anthropic";

    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_response_value(
            &value,
            "Anthropic skill source is invalid or exceeds Siumai's response-value safety bound",
        )?;
        Ok(Self(value))
    }

    pub fn custom() -> Self {
        Self(Self::CUSTOM.to_owned())
    }

    pub fn anthropic() -> Self {
        Self(Self::ANTHROPIC.to_owned())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_custom(&self) -> bool {
        self.0 == Self::CUSTOM
    }

    pub fn is_anthropic(&self) -> bool {
        self.0 == Self::ANTHROPIC
    }
}

impl Serialize for AnthropicSkillSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AnthropicSkillSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        validate_wire_response_value::<D::Error>(
            &value,
            "Siumai rejected an invalid or overlong Anthropic skill source",
        )?;
        Ok(Self(value))
    }
}

impl fmt::Debug for AnthropicSkillSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillSource")
            .field(
                "known",
                &match self.as_str() {
                    Self::CUSTOM => Some(Self::CUSTOM),
                    Self::ANTHROPIC => Some(Self::ANTHROPIC),
                    _ => None,
                },
            )
            .field("value_bytes", &self.0.len())
            .finish()
    }
}

/// Open response/status type returned by the Skills API.
///
/// Known values include `skill`, `skill_deleted`, `skill_version`, and
/// `skill_version_deleted`; future values remain inspectable through
/// [`Self::as_str`]. Its 128-byte, control-free bound is a Siumai
/// response-safety policy rather than an Anthropic type limit.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AnthropicSkillResponseType(String);

impl AnthropicSkillResponseType {
    pub const SKILL: &'static str = "skill";
    pub const SKILL_DELETED: &'static str = "skill_deleted";
    pub const SKILL_VERSION: &'static str = "skill_version";
    pub const SKILL_VERSION_DELETED: &'static str = "skill_version_deleted";

    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_response_value(
            &value,
            "Anthropic skill response type is invalid or exceeds Siumai's response-value safety bound",
        )?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_skill(&self) -> bool {
        self.0 == Self::SKILL
    }

    pub fn is_deleted(&self) -> bool {
        matches!(
            self.0.as_str(),
            Self::SKILL_DELETED | Self::SKILL_VERSION_DELETED
        )
    }

    pub fn is_skill_version(&self) -> bool {
        self.0 == Self::SKILL_VERSION
    }
}

impl Serialize for AnthropicSkillResponseType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AnthropicSkillResponseType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        validate_wire_response_value::<D::Error>(
            &value,
            "Siumai rejected an invalid or overlong Anthropic skill response type",
        )?;
        Ok(Self(value))
    }
}

impl fmt::Debug for AnthropicSkillResponseType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillResponseType")
            .field(
                "known",
                &match self.as_str() {
                    Self::SKILL
                    | Self::SKILL_DELETED
                    | Self::SKILL_VERSION
                    | Self::SKILL_VERSION_DELETED => Some(self.as_str()),
                    _ => None,
                },
            )
            .field("value_bytes", &self.0.len())
            .finish()
    }
}

/// One file included in a Skills multipart upload.
///
/// The path is an in-memory multipart filename, not a local filesystem path.
/// ZIP archives are accepted as opaque bytes and are validated by Anthropic.
#[derive(Clone, PartialEq, Eq)]
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
        validate_skill_file_path(&path)?;
        validate_media_type(&media_type)?;
        // Keep the potentially large byte buffer untouched until metadata has
        // passed validation.
        let data = data.into();
        Ok(Self {
            path,
            media_type,
            data,
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

    fn is_zip_archive(&self) -> bool {
        self.media_type.eq_ignore_ascii_case("application/zip")
            || self
                .media_type
                .eq_ignore_ascii_case("application/x-zip-compressed")
            || self.path.to_ascii_lowercase().ends_with(".zip")
    }
}

impl fmt::Debug for AnthropicSkillFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillFile")
            .field("path_bytes", &self.path.len())
            .field("media_type_bytes", &self.media_type.len())
            .field("data_bytes", &self.data.len())
            .field("archive", &self.is_zip_archive())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkillUploadFormat {
    Files,
    OpaqueZip,
}

/// Multipart files used to create a Skill.
///
/// Individual files must satisfy Anthropic's common-root and root `SKILL.md`
/// requirements. The documented aggregate limit below 30 MB uncompressed is
/// an official provider constraint. File-count, filename, title, collision,
/// and encoded-ZIP bounds mentioned in this type are explicit Siumai safety
/// bounds.
#[derive(Clone)]
pub struct AnthropicSkillUpload {
    files: Vec<AnthropicSkillFile>,
    display_title: Option<String>,
    format: SkillUploadFormat,
    total_bytes: usize,
}

impl AnthropicSkillUpload {
    pub fn new(files: Vec<AnthropicSkillFile>) -> Result<Self, Error> {
        let (format, total_bytes) = validate_skill_files(&files)?;
        Ok(Self {
            files,
            display_title: None,
            format,
            total_bytes,
        })
    }

    pub fn with_display_title(mut self, title: impl Into<String>) -> Result<Self, Error> {
        let title = title.into();
        if title.trim().is_empty() || title.chars().any(char::is_control) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Siumai requires Anthropic skill display titles to be non-empty and control-free",
            ));
        }
        if title.len() > MAX_SKILL_TITLE_BYTES {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic skill display title exceeds Siumai's 512-byte safety bound",
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

impl fmt::Debug for AnthropicSkillUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillUpload")
            .field("file_count", &self.files.len())
            .field("format", &self.format)
            .field("total_bytes", &self.total_bytes)
            .field("display_title_present", &self.display_title.is_some())
            .finish()
    }
}

/// Multipart files used to create one immutable Skill version.
///
/// The same common-root, `SKILL.md`, aggregate, and opaque-ZIP rules as
/// [`AnthropicSkillUpload`] apply. A version has no display-title field.
#[derive(Clone)]
pub struct AnthropicSkillVersionUpload {
    files: Vec<AnthropicSkillFile>,
    format: SkillUploadFormat,
    total_bytes: usize,
}

impl AnthropicSkillVersionUpload {
    pub fn new(files: Vec<AnthropicSkillFile>) -> Result<Self, Error> {
        let (format, total_bytes) = validate_skill_files(&files)?;
        Ok(Self {
            files,
            format,
            total_bytes,
        })
    }

    pub fn files(&self) -> &[AnthropicSkillFile] {
        &self.files
    }
}

impl fmt::Debug for AnthropicSkillVersionUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillVersionUpload")
            .field("file_count", &self.files.len())
            .field("format", &self.format)
            .field("total_bytes", &self.total_bytes)
            .finish()
    }
}

/// Anthropic Skill metadata. Unknown additive response fields are retained.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicSkill {
    pub id: String,
    #[serde(rename = "type", default)]
    pub object_type: Option<AnthropicSkillResponseType>,
    #[serde(default)]
    pub display_title: Option<String>,
    #[serde(default)]
    pub latest_version: Option<String>,
    #[serde(default)]
    pub source: Option<AnthropicSkillSource>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for AnthropicSkill {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkill")
            .field("id_bytes", &self.id.len())
            .field("object_type", &self.object_type)
            .field(
                "display_title_bytes",
                &self.display_title.as_ref().map(String::len),
            )
            .field(
                "latest_version_bytes",
                &self.latest_version.as_ref().map(String::len),
            )
            .field("source", &self.source)
            .field("created_at_present", &self.created_at.is_some())
            .field("updated_at_present", &self.updated_at.is_some())
            .field("extra_fields", &self.extra.len())
            .finish()
    }
}

/// Bounded list response for the Skills endpoint.
#[derive(Clone, PartialEq, Deserialize)]
pub struct AnthropicSkillList {
    #[serde(default)]
    pub data: Vec<AnthropicSkill>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(default)]
    pub next_page: Option<String>,
}

impl fmt::Debug for AnthropicSkillList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillList")
            .field("skill_count", &self.data.len())
            .field("has_more", &self.has_more)
            .field("next_page_present", &self.next_page.is_some())
            .finish()
    }
}

/// Query parameters for listing Skills.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct AnthropicSkillListQuery {
    pub limit: Option<u16>,
    pub page: Option<String>,
    pub source: Option<AnthropicSkillSource>,
}

impl fmt::Debug for AnthropicSkillListQuery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillListQuery")
            .field("limit", &self.limit)
            .field("page_bytes", &self.page.as_ref().map(String::len))
            .field("source", &self.source)
            .finish()
    }
}

/// Query parameters for listing immutable versions of one Skill.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct AnthropicSkillVersionListQuery {
    pub limit: Option<u16>,
    pub page: Option<String>,
}

impl fmt::Debug for AnthropicSkillVersionListQuery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillVersionListQuery")
            .field("limit", &self.limit)
            .field("page_bytes", &self.page.as_ref().map(String::len))
            .finish()
    }
}

/// Skill version metadata. Unknown additive response fields are retained.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicSkillVersion {
    #[serde(default)]
    pub id: Option<String>,
    pub version: String,
    #[serde(default)]
    pub skill_id: Option<String>,
    #[serde(rename = "type", default)]
    pub object_type: Option<AnthropicSkillResponseType>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub directory: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for AnthropicSkillVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillVersion")
            .field("id_bytes", &self.id.as_ref().map(String::len))
            .field("version_bytes", &self.version.len())
            .field("skill_id_bytes", &self.skill_id.as_ref().map(String::len))
            .field("object_type", &self.object_type)
            .field("created_at_present", &self.created_at.is_some())
            .field(
                "description_bytes",
                &self.description.as_ref().map(String::len),
            )
            .field("directory_bytes", &self.directory.as_ref().map(String::len))
            .field("name_bytes", &self.name.as_ref().map(String::len))
            .field("extra_fields", &self.extra.len())
            .finish()
    }
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct AnthropicSkillVersionList {
    #[serde(default)]
    pub data: Vec<AnthropicSkillVersion>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(default)]
    pub next_page: Option<String>,
}

impl fmt::Debug for AnthropicSkillVersionList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillVersionList")
            .field("version_count", &self.data.len())
            .field("has_more", &self.has_more)
            .field("next_page_present", &self.next_page.is_some())
            .finish()
    }
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct AnthropicSkillDeleteResult {
    pub id: String,
    #[serde(rename = "type", default)]
    pub object_type: Option<AnthropicSkillResponseType>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for AnthropicSkillDeleteResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillDeleteResult")
            .field("id_bytes", &self.id.len())
            .field("object_type", &self.object_type)
            .field("extra_fields", &self.extra.len())
            .finish()
    }
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct AnthropicSkillVersionDeleteResult {
    pub id: String,
    #[serde(rename = "type", default)]
    pub object_type: Option<AnthropicSkillResponseType>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for AnthropicSkillVersionDeleteResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillVersionDeleteResult")
            .field("id_bytes", &self.id.len())
            .field("object_type", &self.object_type)
            .field("extra_fields", &self.extra.len())
            .finish()
    }
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct AnthropicSkillUploadResult {
    #[serde(flatten)]
    pub skill: AnthropicSkill,
}

impl fmt::Debug for AnthropicSkillUploadResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkillUploadResult")
            .field("skill", &self.skill)
            .finish()
    }
}

/// Shared, lightweight Skills API handle.
///
/// Skill and version identifiers remain opaque. Siumai applies only local
/// byte/control safety bounds before encoding each identifier exactly once as
/// one path segment; those bounds are not Anthropic identifier formats.
#[derive(Clone)]
pub struct AnthropicSkills {
    runtime: Arc<NativeRuntime>,
}

impl AnthropicSkills {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    /// Create a Skill from individual files or an opaque ZIP archive.
    pub async fn create(
        &self,
        upload: AnthropicSkillUpload,
    ) -> Result<AnthropicSkillUploadResult, Error> {
        self.create_with_options(upload, CallOptions::default())
            .await
    }

    pub async fn create_with_options(
        &self,
        upload: AnthropicSkillUpload,
        options: CallOptions,
    ) -> Result<AnthropicSkillUploadResult, Error> {
        let AnthropicSkillUpload {
            files,
            display_title,
            ..
        } = upload;
        execute_json(
            &self.runtime,
            Method::POST,
            target(SKILLS_TARGET)?,
            skill_multipart(files, display_title)?,
            ReplaySafety::Never,
            &[SKILLS_BETA],
            options,
        )
        .await
    }

    /// Backwards-compatible alias for [`Self::create`].
    pub async fn upload(
        &self,
        upload: AnthropicSkillUpload,
    ) -> Result<AnthropicSkillUploadResult, Error> {
        self.create(upload).await
    }

    pub async fn upload_with_options(
        &self,
        upload: AnthropicSkillUpload,
        options: CallOptions,
    ) -> Result<AnthropicSkillUploadResult, Error> {
        self.create_with_options(upload, options).await
    }

    pub async fn list(&self, query: AnthropicSkillListQuery) -> Result<AnthropicSkillList, Error> {
        self.list_with_options(query, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        query: AnthropicSkillListQuery,
        options: CallOptions,
    ) -> Result<AnthropicSkillList, Error> {
        execute_json(
            &self.runtime,
            Method::GET,
            skill_list_target(&query)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
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
        execute_json(
            &self.runtime,
            Method::GET,
            skill_target(skill_id)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[SKILLS_BETA],
            options,
        )
        .await
    }

    pub async fn delete(&self, skill_id: &str) -> Result<AnthropicSkillDeleteResult, Error> {
        self.delete_with_options(skill_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        skill_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicSkillDeleteResult, Error> {
        execute_json(
            &self.runtime,
            Method::DELETE,
            skill_target(skill_id)?,
            RequestBody::Empty,
            ReplaySafety::Never,
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
        self.versions_page_with_options(
            skill_id,
            AnthropicSkillVersionListQuery::default(),
            options,
        )
        .await
    }

    pub async fn versions_page(
        &self,
        skill_id: &str,
        query: AnthropicSkillVersionListQuery,
    ) -> Result<AnthropicSkillVersionList, Error> {
        self.versions_page_with_options(skill_id, query, CallOptions::default())
            .await
    }

    pub async fn versions_page_with_options(
        &self,
        skill_id: &str,
        query: AnthropicSkillVersionListQuery,
        options: CallOptions,
    ) -> Result<AnthropicSkillVersionList, Error> {
        execute_json(
            &self.runtime,
            Method::GET,
            skill_versions_list_target(skill_id, &query)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[SKILLS_BETA],
            options,
        )
        .await
    }

    pub async fn create_version(
        &self,
        skill_id: &str,
        upload: AnthropicSkillVersionUpload,
    ) -> Result<AnthropicSkillVersion, Error> {
        self.create_version_with_options(skill_id, upload, CallOptions::default())
            .await
    }

    pub async fn create_version_with_options(
        &self,
        skill_id: &str,
        upload: AnthropicSkillVersionUpload,
        options: CallOptions,
    ) -> Result<AnthropicSkillVersion, Error> {
        let AnthropicSkillVersionUpload { files, .. } = upload;
        execute_json(
            &self.runtime,
            Method::POST,
            skill_versions_target(skill_id)?,
            skill_multipart(files, None)?,
            ReplaySafety::Never,
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
        execute_json(
            &self.runtime,
            Method::GET,
            skill_version_target(skill_id, version)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[SKILLS_BETA],
            options,
        )
        .await
    }

    pub async fn delete_version(
        &self,
        skill_id: &str,
        version: &str,
    ) -> Result<AnthropicSkillVersionDeleteResult, Error> {
        self.delete_version_with_options(skill_id, version, CallOptions::default())
            .await
    }

    pub async fn delete_version_with_options(
        &self,
        skill_id: &str,
        version: &str,
        options: CallOptions,
    ) -> Result<AnthropicSkillVersionDeleteResult, Error> {
        execute_json(
            &self.runtime,
            Method::DELETE,
            skill_version_target(skill_id, version)?,
            RequestBody::Empty,
            ReplaySafety::Never,
            &[SKILLS_BETA],
            options,
        )
        .await
    }
}

impl fmt::Debug for AnthropicSkills {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicSkills")
            .field("runtime", &"shared")
            .finish()
    }
}

fn skill_multipart(
    files: Vec<AnthropicSkillFile>,
    display_title: Option<String>,
) -> Result<RequestBody, Error> {
    // The public constructors validate all files before this function is
    // reached. This function only moves already-bounded bytes into parts.
    let mut parts = Vec::with_capacity(
        files
            .len()
            .saturating_add(usize::from(display_title.is_some())),
    );
    if let Some(title) = display_title {
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
    for file in files {
        let media_type = media_type_header(&file.media_type)?;
        parts.push(
            MultipartPart::file("files[]", file.path, media_type, file.data).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Anthropic skill multipart body is invalid",
                )
                .with_source(source)
            })?,
        );
    }
    Ok(multipart_body(MultipartBody::new(parts)))
}

fn validate_skill_files(files: &[AnthropicSkillFile]) -> Result<(SkillUploadFormat, usize), Error> {
    if files.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic skill upload requires at least one file",
        ));
    }
    if files.len() > MAX_SKILL_FILES {
        return Err(Error::new(
            ErrorKind::LimitExceeded,
            "Anthropic skill upload exceeds Siumai's 1000-file safety bound",
        ));
    }

    let total_bytes = files.iter().try_fold(0_usize, |total, file| {
        total.checked_add(file.data.len()).ok_or_else(|| {
            Error::new(
                ErrorKind::LimitExceeded,
                "Anthropic skill upload exceeds Siumai's aggregate byte bound",
            )
        })
    })?;
    if files.len() == 1 && files[0].is_zip_archive() {
        validate_upload_size(SkillUploadFormat::OpaqueZip, total_bytes)?;
        return Ok((SkillUploadFormat::OpaqueZip, total_bytes));
    }

    validate_upload_size(SkillUploadFormat::Files, total_bytes)?;

    let mut exact_paths = BTreeSet::new();
    let mut normalized_paths = BTreeSet::new();
    let mut root = None;
    let mut has_root_skill = false;
    for file in files {
        let Some((current_root, relative_path)) = file.path.split_once('/') else {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic skill files must include one top-level directory",
            ));
        };
        if root.is_some_and(|expected| expected != current_root) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic skill files must share exactly one top-level directory",
            ));
        }
        root = Some(current_root);
        if !exact_paths.insert(file.path.as_str()) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Siumai rejects duplicate Anthropic skill file paths",
            ));
        }
        if !normalized_paths.insert(normalized_path_key(&file.path)) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Siumai rejects case or Unicode-colliding Anthropic skill file paths",
            ));
        }
        if relative_path == "SKILL.md" {
            has_root_skill = true;
        }
    }
    if !has_root_skill {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic skill upload requires SKILL.md at the top level",
        ));
    }
    Ok((SkillUploadFormat::Files, total_bytes))
}

fn validate_upload_size(format: SkillUploadFormat, total_bytes: usize) -> Result<(), Error> {
    match format {
        SkillUploadFormat::Files if total_bytes >= OFFICIAL_MAX_SKILL_UNCOMPRESSED_BYTES => {
            Err(Error::new(
                ErrorKind::LimitExceeded,
                "Anthropic skill files must remain under the official 30 MB uncompressed aggregate limit",
            ))
        }
        SkillUploadFormat::OpaqueZip if total_bytes >= SIUMAI_MAX_SKILL_ARCHIVE_BYTES => {
            Err(Error::new(
                ErrorKind::LimitExceeded,
                "Anthropic skill ZIP archive exceeds Siumai's encoded-byte safety bound; Anthropic validates the official uncompressed limit",
            ))
        }
        _ => Ok(()),
    }
}

fn validate_skill_file_path(path: &str) -> Result<(), Error> {
    if path.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Siumai rejects empty Anthropic skill file paths",
        ));
    }
    if path.len() > MAX_SKILL_FILE_PATH_BYTES {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic skill file path exceeds Siumai's 2048-byte safety bound",
        ));
    }
    let bytes = path.as_bytes();
    let has_windows_drive = matches!(
        (bytes.first(), bytes.get(1)),
        (Some(first), Some(b':')) if first.is_ascii_alphabetic()
    );
    if path.starts_with('/') || path.contains('\\') || has_windows_drive {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Siumai requires Anthropic skill file paths to be relative and use forward slashes",
        ));
    }
    for segment in path.split('/') {
        if segment.is_empty() || matches!(segment, "." | "..") {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Siumai rejects empty or traversal segments in Anthropic skill file paths",
            ));
        }
        if segment.chars().any(char::is_control) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Siumai rejects control characters in Anthropic skill file paths",
            ));
        }
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
            "Anthropic skill media type is invalid",
        ));
    }
    media_type_header(media_type).map(|_| ())
}

fn media_type_header(media_type: &str) -> Result<HeaderValue, Error> {
    HeaderValue::from_str(media_type).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Anthropic skill media type is invalid",
        )
        .with_source(source)
    })
}

fn validate_skill_id(skill_id: &str) -> Result<(), Error> {
    validate_opaque_id(
        skill_id,
        MAX_SKILL_ID_BYTES,
        "Anthropic skill identifier is empty, contains a control character, or exceeds Siumai's 512-byte safety bound",
    )
}

fn validate_skill_version(version: &str) -> Result<(), Error> {
    validate_opaque_id(
        version,
        MAX_SKILL_VERSION_BYTES,
        "Anthropic skill version identifier is empty, contains a control character, or exceeds Siumai's 256-byte safety bound",
    )
}

fn validate_opaque_id(value: &str, maximum: usize, message: &'static str) -> Result<(), Error> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(Error::new(ErrorKind::InvalidInput, message));
    }
    Ok(())
}

fn validate_page_token(page: &str) -> Result<(), Error> {
    validate_opaque_id(
        page,
        MAX_SKILL_PAGE_BYTES,
        "Anthropic skill page token is empty, contains a control character, or exceeds Siumai's 2048-byte safety bound",
    )
}

fn skill_target(skill_id: &str) -> Result<RequestTarget, Error> {
    validate_skill_id(skill_id)?;
    target(SKILLS_TARGET)?
        .with_opaque_path_segment(skill_id)
        .map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "Anthropic skill target is invalid")
                .with_source(source)
        })
}

fn skill_versions_target(skill_id: &str) -> Result<RequestTarget, Error> {
    skill_target(skill_id)?
        .with_opaque_path_segment("versions")
        .map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic skill versions target is invalid",
            )
            .with_source(source)
        })
}

fn skill_versions_list_target(
    skill_id: &str,
    query: &AnthropicSkillVersionListQuery,
) -> Result<RequestTarget, Error> {
    validate_skill_id(skill_id)?;
    validate_list_limit(query.limit)?;
    let mut pairs = url::form_urlencoded::Serializer::new(String::new());
    pairs.append_pair("beta", "true");
    if let Some(limit) = query.limit {
        pairs.append_pair("limit", &limit.to_string());
    }
    if let Some(page) = &query.page {
        validate_page_token(page)?;
        pairs.append_pair("page", page);
    }
    target(format!("skills?{}", pairs.finish()))?
        .with_opaque_path_segment(skill_id)
        .and_then(|target| target.with_opaque_path_segment("versions"))
        .map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic skill versions list target is invalid",
            )
            .with_source(source)
        })
}

fn skill_version_target(skill_id: &str, version: &str) -> Result<RequestTarget, Error> {
    validate_skill_version(version)?;
    skill_versions_target(skill_id)?
        .with_opaque_path_segment(version)
        .map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic skill version target is invalid",
            )
            .with_source(source)
        })
}

fn skill_list_target(query: &AnthropicSkillListQuery) -> Result<RequestTarget, Error> {
    validate_list_limit(query.limit)?;
    let mut pairs = url::form_urlencoded::Serializer::new(String::new());
    pairs.append_pair("beta", "true");
    if let Some(limit) = query.limit {
        pairs.append_pair("limit", &limit.to_string());
    }
    if let Some(page) = &query.page {
        validate_page_token(page)?;
        pairs.append_pair("page", page);
    }
    if let Some(source) = &query.source {
        pairs.append_pair("source", source.as_str());
    }
    target(format!("skills?{}", pairs.finish()))
}

fn validate_list_limit(limit: Option<u16>) -> Result<(), Error> {
    if limit.is_some_and(|limit| limit == 0 || limit > 100) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic skill list limit must be between 1 and 100 per the official API",
        ));
    }
    Ok(())
}

fn validate_response_value(value: &str, message: &'static str) -> Result<(), Error> {
    validate_opaque_id(value, MAX_SKILL_RESPONSE_VALUE_BYTES, message)
}

fn validate_wire_response_value<E>(value: &str, message: &'static str) -> Result<(), E>
where
    E: serde::de::Error,
{
    validate_opaque_id(value, MAX_SKILL_RESPONSE_VALUE_BYTES, message)
        .map_err(|_| E::custom(message))
}

fn normalized_path_key(path: &str) -> String {
    // This is a conservative Siumai cross-platform collision key, not an
    // Anthropic filename rule. Compatibility composition plus lowercase
    // folding catches common cross-platform Unicode/case aliases without
    // interpreting a filename as a URL host or numeric address.
    let mut key = String::new();
    for (index, segment) in path.split('/').enumerate() {
        if index != 0 {
            key.push('/');
        }
        key.extend(segment.nfkc().flat_map(char::to_lowercase));
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use siumai_core::{ReplayDomain, ReplayDomainId};
    use siumai_transport::EndpointConfig;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::{AnthropicCredential, AnthropicProvider};

    fn file(path: &str) -> AnthropicSkillFile {
        AnthropicSkillFile::new(path, "text/plain", Bytes::from_static(b"x"))
            .expect("valid Skill file")
    }

    fn local_provider(server: &MockServer) -> AnthropicProvider {
        AnthropicProvider::builder(AnthropicCredential::unauthenticated())
            .with_endpoint(
                EndpointConfig::local_explicit(format!("{}/v1/", server.uri()))
                    .expect("local endpoint"),
            )
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("skills-lifecycle-tests").expect("replay domain"),
            ))
            .build()
            .expect("provider")
    }

    #[test]
    fn multipart_files_require_one_root_and_root_skill_markdown() {
        assert!(
            AnthropicSkillUpload::new(vec![file("demo/SKILL.md"), file("demo/scripts/run.py"),])
                .is_ok()
        );

        for files in [
            vec![file("SKILL.md")],
            vec![file("demo/docs/SKILL.md")],
            vec![file("demo/skill.md")],
            vec![file("demo/SKILL.md"), file("other/readme.txt")],
            vec![file("demo/SKILL.md"), file("demo/SKILL.md")],
        ] {
            assert!(AnthropicSkillUpload::new(files).is_err());
        }
    }

    #[test]
    fn multipart_paths_reject_absolute_traversal_separator_and_control_segments() {
        for path in [
            "/demo/SKILL.md",
            "C:/demo/SKILL.md",
            "demo/../SKILL.md",
            "demo\\SKILL.md",
            "demo//SKILL.md",
            "demo/./SKILL.md",
            "demo/line\nfeed/SKILL.md",
        ] {
            assert!(AnthropicSkillFile::new(path, "text/plain", Bytes::new()).is_err());
        }
    }

    #[test]
    fn multipart_paths_reject_case_and_unicode_collisions() {
        let files = vec![
            file("demo/SKILL.md"),
            file("demo/Readme.md"),
            file("demo/README.md"),
        ];
        assert!(AnthropicSkillUpload::new(files).is_err());

        let files = vec![
            file("demo/SKILL.md"),
            file("demo/\u{00e9}.md"),
            file("demo/e\u{301}.md"),
        ];
        assert!(AnthropicSkillUpload::new(files).is_err());

        AnthropicSkillUpload::new(vec![
            file("demo/SKILL.md"),
            file("demo/0177"),
            file("demo/127"),
        ])
        .expect("distinct numeric filenames must not be interpreted as IP addresses");
    }

    #[test]
    fn zip_upload_is_opaque_and_server_validated() {
        let archive = AnthropicSkillFile::new(
            "bundle.zip",
            "application/zip",
            Bytes::from_static(b"not parsed locally; ../escape is server input"),
        )
        .expect("archive metadata");
        assert!(AnthropicSkillUpload::new(vec![archive.clone()]).is_ok());
        assert!(AnthropicSkillUpload::new(vec![archive, file("demo/SKILL.md")]).is_err());
    }

    #[test]
    fn aggregate_bounds_distinguish_official_files_from_local_zip_safety() {
        assert!(
            validate_upload_size(
                SkillUploadFormat::Files,
                OFFICIAL_MAX_SKILL_UNCOMPRESSED_BYTES - 1,
            )
            .is_ok()
        );
        let error = validate_upload_size(
            SkillUploadFormat::Files,
            OFFICIAL_MAX_SKILL_UNCOMPRESSED_BYTES,
        )
        .expect_err("official aggregate bound");
        assert!(error.message().contains("official 30 MB"));

        assert!(
            validate_upload_size(
                SkillUploadFormat::OpaqueZip,
                SIUMAI_MAX_SKILL_ARCHIVE_BYTES - 1,
            )
            .is_ok()
        );
        let error =
            validate_upload_size(SkillUploadFormat::OpaqueZip, SIUMAI_MAX_SKILL_ARCHIVE_BYTES)
                .expect_err("local archive bound");
        assert!(error.message().contains("Siumai's encoded-byte"));
    }

    #[test]
    fn skill_targets_encode_every_dynamic_id_as_one_opaque_segment() {
        let target = skill_target("skill/secret?#资源").expect("skill target");
        assert_eq!(
            target.as_str(),
            "skills/skill%2Fsecret%3F%23%E8%B5%84%E6%BA%90?beta=true"
        );
        let version =
            skill_version_target("skill/secret", "version?#资源").expect("version target");
        assert_eq!(
            version.as_str(),
            "skills/skill%2Fsecret/versions/version%3F%23%E8%B5%84%E6%BA%90?beta=true"
        );
        assert_eq!(
            skill_target("skill%2Fsecret")
                .expect("pre-encoded-looking identifier")
                .as_str(),
            "skills/skill%252Fsecret?beta=true"
        );
        for invalid in ["", ".", "..", "skill\nid"] {
            assert!(skill_target(invalid).is_err());
        }
    }

    #[test]
    fn list_query_encodes_page_and_keeps_debug_structural() {
        let query = AnthropicSkillListQuery {
            limit: Some(100),
            page: Some("page/secret?#资源".to_owned()),
            source: Some(AnthropicSkillSource::custom()),
        };
        let target = skill_list_target(&query).expect("list target");
        assert_eq!(
            target.as_str(),
            "skills?beta=true&limit=100&page=page%2Fsecret%3F%23%E8%B5%84%E6%BA%90&source=custom"
        );
        assert!(!format!("{query:?}").contains("page/secret"));

        let version_query = AnthropicSkillVersionListQuery {
            limit: Some(25),
            page: Some("version/page?#资源".to_owned()),
        };
        let target = skill_versions_list_target("skill/secret", &version_query)
            .expect("version list target");
        assert_eq!(
            target.as_str(),
            "skills/skill%2Fsecret/versions?beta=true&limit=25&page=version%2Fpage%3F%23%E8%B5%84%E6%BA%90"
        );
        assert!(!format!("{version_query:?}").contains("version/page"));
    }

    #[test]
    fn lifecycle_responses_preserve_unknown_open_values_and_redact_debug() {
        let sentinel = "private-skill-sentinel";
        let list: AnthropicSkillList = serde_json::from_value(json!({
            "data": [{
                "id": sentinel,
                "type": "future_skill_type",
                "display_title": sentinel,
                "latest_version": sentinel,
                "source": "future_source",
                "created_at": sentinel,
                "updated_at": sentinel,
                "private": sentinel
            }],
            "has_more": true,
            "next_page": sentinel
        }))
        .expect("skill list");
        assert_eq!(
            list.data[0].source.as_ref().expect("source").as_str(),
            "future_source"
        );
        assert_eq!(
            list.data[0].object_type.as_ref().expect("type").as_str(),
            "future_skill_type"
        );
        assert!(!format!("{list:?}").contains(sentinel));

        let deleted: AnthropicSkillDeleteResult = serde_json::from_value(json!({
            "id": sentinel,
            "type": "skill_deleted"
        }))
        .expect("delete response");
        assert!(deleted.object_type.as_ref().expect("status").is_deleted());
        assert!(!format!("{deleted:?}").contains(sentinel));

        let version: AnthropicSkillVersion = serde_json::from_value(json!({
            "id": sentinel,
            "version": sentinel,
            "skill_id": sentinel,
            "type": "skill_version",
            "description": sentinel,
            "directory": sentinel,
            "name": sentinel
        }))
        .expect("version response");
        assert!(!format!("{version:?}").contains(sentinel));

        let version_deleted: AnthropicSkillVersionDeleteResult = serde_json::from_value(json!({
            "id": sentinel,
            "type": "skill_version_deleted"
        }))
        .expect("version delete response");
        assert!(
            version_deleted
                .object_type
                .as_ref()
                .expect("status")
                .is_deleted()
        );
        assert!(!format!("{version_deleted:?}").contains(sentinel));
    }

    #[test]
    fn open_response_values_reject_control_and_unbounded_data() {
        assert!(AnthropicSkillSource::new("future").is_ok());
        assert!(AnthropicSkillSource::new("bad\nvalue").is_err());
        assert!(
            AnthropicSkillResponseType::new("x".repeat(MAX_SKILL_RESPONSE_VALUE_BYTES + 1))
                .is_err()
        );
    }

    #[test]
    fn upload_debug_does_not_expose_paths_titles_or_bytes() {
        let sentinel = "private-skill-sentinel";
        let upload = AnthropicSkillUpload::new(vec![
            AnthropicSkillFile::new(
                "demo/SKILL.md",
                "text/plain",
                Bytes::from_static(b"private-skill-sentinel"),
            )
            .expect("file"),
        ])
        .expect("upload")
        .with_display_title(sentinel)
        .expect("title");
        let debug = format!("{upload:?}");
        assert!(!debug.contains(sentinel));
    }

    #[tokio::test]
    async fn lifecycle_routes_preserve_beta_query_and_opaque_identifiers() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/skills"))
            .and(query_param("beta", "true"))
            .and(query_param("limit", "25"))
            .and(query_param("page", "page/one"))
            .and(header("anthropic-beta", SKILLS_BETA))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "skill-list", "type": "skill"}],
                "has_more": true,
                "next_page": "page-two"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/skills/skill%2Fretrieve"))
            .and(query_param("beta", "true"))
            .and(header("anthropic-beta", SKILLS_BETA))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"id": "skill-retrieve", "type": "skill"})),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path("/v1/skills/skill%2Fdelete"))
            .and(query_param("beta", "true"))
            .and(header("anthropic-beta", SKILLS_BETA))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "skill-delete",
                "type": "skill_deleted"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/skills/skill%2Fversions/versions"))
            .and(query_param("beta", "true"))
            .and(query_param("limit", "10"))
            .and(query_param("page", "version/page"))
            .and(header("anthropic-beta", SKILLS_BETA))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{
                    "version": "2",
                    "skill_id": "skill-versions",
                    "type": "skill_version"
                }],
                "has_more": true,
                "next_page": "version-page-two"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/skills/skill%2Fcreate/versions"))
            .and(query_param("beta", "true"))
            .and(header("anthropic-beta", SKILLS_BETA))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "version": "3",
                "skill_id": "skill-create",
                "type": "skill_version"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/skills/skill%2Fversion/versions/v%2F2"))
            .and(query_param("beta", "true"))
            .and(header("anthropic-beta", SKILLS_BETA))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "version": "v/2",
                "skill_id": "skill-version",
                "type": "skill_version"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path("/v1/skills/skill%2Fdelete-version/versions/v%2F3"))
            .and(query_param("beta", "true"))
            .and(header("anthropic-beta", SKILLS_BETA))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "v/3",
                "type": "skill_version_deleted"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let skills = local_provider(&server).skills();
        let list = skills
            .list(AnthropicSkillListQuery {
                limit: Some(25),
                page: Some("page/one".to_owned()),
                source: None,
            })
            .await
            .expect("list Skills");
        assert_eq!(list.next_page.as_deref(), Some("page-two"));
        skills
            .retrieve("skill/retrieve")
            .await
            .expect("retrieve Skill");
        skills.delete("skill/delete").await.expect("delete Skill");
        let versions = skills
            .versions_page(
                "skill/versions",
                AnthropicSkillVersionListQuery {
                    limit: Some(10),
                    page: Some("version/page".to_owned()),
                },
            )
            .await
            .expect("list Skill versions");
        assert_eq!(versions.next_page.as_deref(), Some("version-page-two"));
        skills
            .create_version(
                "skill/create",
                AnthropicSkillVersionUpload::new(vec![file("fixture/SKILL.md")])
                    .expect("version upload"),
            )
            .await
            .expect("create Skill version");
        skills
            .version("skill/version", "v/2")
            .await
            .expect("retrieve Skill version");
        skills
            .delete_version("skill/delete-version", "v/3")
            .await
            .expect("delete Skill version");
    }

    #[tokio::test]
    async fn destructive_skill_operations_are_never_retried() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/v1/skills/skill%2Ffailure"))
            .and(query_param("beta", "true"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "type": "error",
                "error": {"type": "overloaded_error"}
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path("/v1/skills/skill%2Ffailure/versions/v%2Ffailure"))
            .and(query_param("beta", "true"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "type": "error",
                "error": {"type": "overloaded_error"}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let skills = local_provider(&server).skills();
        assert!(skills.delete("skill/failure").await.is_err());
        assert!(
            skills
                .delete_version("skill/failure", "v/failure")
                .await
                .is_err()
        );
    }
}
