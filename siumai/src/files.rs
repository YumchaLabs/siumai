//! High-level file upload helpers aligned with AI SDK `uploadFile`.

use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use siumai_core::client::LlmClient;
use siumai_core::error::LlmError;
use siumai_core::traits::FileManagementCapability;
use siumai_core::types::{
    FileObject, FileUploadRequest, HttpConfig, ProviderMetadataMap, ProviderOptionsMap,
    ProviderReference, Warning,
};
use siumai_core::utils::mime::guess_mime_from_bytes;
use std::borrow::Cow;
use std::collections::HashMap;

/// Provider-id keyed metadata map used by `UploadFileResult`.
pub type UploadFileProviderMetadata = ProviderMetadataMap;

/// Upload input data for `files::upload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadFileData {
    /// Raw file bytes.
    Bytes(Vec<u8>),
    /// Base64-encoded file bytes.
    Base64(String),
    /// URL input is intentionally rejected at runtime to match AI SDK `uploadFile`.
    Url(String),
}

impl UploadFileData {
    /// Create upload data from raw bytes.
    pub fn bytes(data: Vec<u8>) -> Self {
        Self::Bytes(data)
    }

    /// Create upload data from a base64 string.
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64(data.into())
    }

    /// Create upload data from a URL string.
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }
}

impl From<Vec<u8>> for UploadFileData {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for UploadFileData {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

/// Options for `files::upload`.
#[derive(Debug, Clone, Default)]
pub struct UploadFileOptions {
    /// Optional media type. When absent, media type is inferred from bytes and text heuristics.
    pub media_type: Option<String>,
    /// Optional filename. When absent, `blob` is used.
    pub filename: Option<String>,
    /// Optional provider upload purpose.
    ///
    /// This is required for providers such as MiniMaxi that do not have a stable default purpose.
    pub purpose: Option<String>,
    /// Optional provider upload metadata forwarded through low-level file APIs when supported.
    pub metadata: HashMap<String, String>,
    /// Optional provider-specific options (`providerOptions`).
    pub provider_options: ProviderOptionsMap,
    /// Optional per-request HTTP overrides.
    pub http_config: Option<HttpConfig>,
}

impl UploadFileOptions {
    /// Create empty upload options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the media type.
    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Self {
        self.media_type = Some(media_type.into());
        self
    }

    /// Set the filename.
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// Set the provider upload purpose.
    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = Some(purpose.into());
        self
    }

    /// Add one metadata item.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Replace the provider options map.
    pub fn with_provider_options(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Insert one provider option entry.
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        value: serde_json::Value,
    ) -> Self {
        self.provider_options.insert(provider_id, value);
        self
    }

    /// Set the per-request HTTP config.
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }

    /// Add one per-request header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut http_config = self.http_config.take().unwrap_or_else(HttpConfig::empty);
        http_config.headers.insert(key.into(), value.into());
        self.http_config = Some(http_config);
        self
    }
}

/// Result returned by `files::upload`.
#[derive(Debug, Clone, PartialEq)]
pub struct UploadFileResult {
    /// Provider-owned file reference in stable AI SDK-style shape.
    pub provider_reference: ProviderReference,
    /// Resolved media type.
    pub media_type: Option<String>,
    /// Resolved filename.
    pub filename: Option<String>,
    /// Provider-owned metadata under the provider id root.
    pub provider_metadata: Option<UploadFileProviderMetadata>,
    /// Non-fatal warnings emitted while uploading.
    pub warnings: Vec<Warning>,
}

/// Resolved upload payload passed to `UploadFileApi`.
#[derive(Debug, Clone)]
pub struct UploadFilePayload {
    /// Decoded file bytes.
    pub data: Vec<u8>,
    /// Resolved media type.
    pub media_type: String,
    /// Resolved filename.
    pub filename: String,
    /// Whether the caller explicitly supplied a filename.
    pub filename_was_explicit: bool,
    /// Optional provider upload purpose.
    pub purpose: Option<String>,
    /// Optional provider upload metadata.
    pub metadata: HashMap<String, String>,
    /// Optional provider-specific options.
    pub provider_options: ProviderOptionsMap,
    /// Optional per-request HTTP overrides.
    pub http_config: Option<HttpConfig>,
}

/// Advanced upload hook implemented by provider clients/resources.
#[async_trait]
pub trait UploadFileApi: Send + Sync {
    /// Canonical provider id.
    fn provider_id(&self) -> Cow<'static, str>;

    /// Upload a resolved file payload.
    async fn upload_prepared_file(
        &self,
        payload: UploadFilePayload,
    ) -> Result<UploadFileResult, LlmError>;
}

/// Provider-id hook for the generic `FileManagementCapability` upload adapter.
pub trait FileUploadProvider: Send + Sync {
    /// Canonical provider id used for `providerReference` and default-purpose resolution.
    fn upload_file_provider_id(&self) -> Cow<'static, str>;
}

/// Upload a file through a high-level files API surface.
pub async fn upload<A: UploadFileApi + ?Sized>(
    api: &A,
    data: UploadFileData,
    options: UploadFileOptions,
) -> Result<UploadFileResult, LlmError> {
    let bytes = decode_upload_data(data)?;
    let filename_was_explicit = options.filename.is_some();
    let media_type = options
        .media_type
        .unwrap_or_else(|| detect_media_type(&bytes));
    let filename = options.filename.unwrap_or_else(default_upload_filename);

    api.upload_prepared_file(UploadFilePayload {
        data: bytes,
        media_type,
        filename,
        filename_was_explicit,
        purpose: options.purpose,
        metadata: options.metadata,
        provider_options: options.provider_options,
        http_config: options.http_config,
    })
    .await
}

#[async_trait]
impl<T> UploadFileApi for T
where
    T: FileManagementCapability + FileUploadProvider + Send + Sync,
{
    fn provider_id(&self) -> Cow<'static, str> {
        self.upload_file_provider_id()
    }

    async fn upload_prepared_file(
        &self,
        payload: UploadFilePayload,
    ) -> Result<UploadFileResult, LlmError> {
        let provider_id = self.upload_file_provider_id().into_owned();
        upload_via_file_management(&provider_id, self, payload).await
    }
}

impl FileUploadProvider for crate::provider::Siumai {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

impl FileUploadProvider for crate::registry::LanguageModelHandle {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

#[cfg(feature = "openai")]
impl FileUploadProvider for siumai_provider_openai::providers::openai::OpenAiClient {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

#[cfg(feature = "openai")]
impl FileUploadProvider for siumai_provider_openai::providers::openai::OpenAiFiles {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("openai")
    }
}

#[cfg(feature = "azure")]
impl FileUploadProvider for siumai_provider_azure::providers::azure_openai::AzureOpenAiClient {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

#[cfg(feature = "google")]
impl FileUploadProvider for siumai_provider_gemini::providers::gemini::GeminiClient {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

#[cfg(feature = "google")]
impl FileUploadProvider for siumai_provider_gemini::providers::gemini::GeminiFiles {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("gemini")
    }
}

#[cfg(feature = "minimaxi")]
impl FileUploadProvider for siumai_provider_minimaxi::providers::minimaxi::MinimaxiClient {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

#[cfg(feature = "minimaxi")]
impl FileUploadProvider for siumai_provider_minimaxi::providers::minimaxi::MinimaxiFiles {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("minimaxi")
    }
}

#[cfg(feature = "xai")]
impl FileUploadProvider for siumai_provider_xai::providers::xai::XaiClient {
    fn upload_file_provider_id(&self) -> Cow<'static, str> {
        LlmClient::provider_id(self)
    }
}

#[cfg(feature = "anthropic")]
#[async_trait]
impl UploadFileApi for siumai_provider_anthropic::providers::anthropic::files::AnthropicFiles {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("anthropic")
    }

    async fn upload_prepared_file(
        &self,
        payload: UploadFilePayload,
    ) -> Result<UploadFileResult, LlmError> {
        let UploadFilePayload {
            data,
            media_type,
            filename,
            filename_was_explicit: _filename_was_explicit,
            purpose,
            metadata,
            provider_options,
            http_config,
        } = payload;

        let headers = http_config
            .as_ref()
            .map(|config| config.headers.clone())
            .filter(|headers| !headers.is_empty());

        let response = self
            .upload(
                filename.clone(),
                data,
                Some(media_type.clone()),
                purpose,
                headers,
            )
            .await?;

        let mut result = upload_result_from_anthropic_file(response, filename, media_type)?;

        if !metadata.is_empty() {
            result.warnings.push(Warning::compatibility(
                "metadata",
                Some("Anthropic file uploads currently ignore UploadFileOptions.metadata."),
            ));
        }

        if !provider_options.is_empty() {
            result.warnings.push(Warning::compatibility(
                "providerOptions",
                Some("Anthropic file uploads currently ignore UploadFileOptions.provider_options."),
            ));
        }

        if http_config
            .as_ref()
            .is_some_and(has_non_header_http_overrides)
        {
            result.warnings.push(Warning::compatibility(
                "httpConfig",
                Some("Anthropic file uploads currently forward only per-request headers."),
            ));
        }

        Ok(result)
    }
}

#[cfg(feature = "anthropic")]
#[async_trait]
impl UploadFileApi for siumai_provider_anthropic::providers::anthropic::AnthropicClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("anthropic")
    }

    async fn upload_prepared_file(
        &self,
        payload: UploadFilePayload,
    ) -> Result<UploadFileResult, LlmError> {
        self.files().upload_prepared_file(payload).await
    }
}

fn decode_upload_data(data: UploadFileData) -> Result<Vec<u8>, LlmError> {
    match data {
        UploadFileData::Bytes(data) => Ok(data),
        UploadFileData::Base64(data) => STANDARD
            .decode(data)
            .map_err(|error| LlmError::InvalidInput(format!("Invalid base64 file data: {error}"))),
        UploadFileData::Url(_) => Err(LlmError::InvalidInput(
            "URL data is not supported for file uploads. Fetch the URL content first and pass the bytes.".to_string(),
        )),
    }
}

fn detect_media_type(bytes: &[u8]) -> String {
    guess_mime_from_bytes(bytes).unwrap_or_else(|| {
        if is_likely_text(bytes) {
            "text/plain".to_string()
        } else {
            "application/octet-stream".to_string()
        }
    })
}

fn is_likely_text(bytes: &[u8]) -> bool {
    const CHECK_LENGTH: usize = 512;

    let check_length = bytes.len().min(CHECK_LENGTH);
    if check_length == 0 {
        return false;
    }

    bytes
        .iter()
        .take(check_length)
        .all(|byte| *byte != 0x00 && (*byte >= 0x20 || matches!(*byte, 0x09 | 0x0A | 0x0D)))
}

fn default_upload_filename() -> String {
    "blob".to_string()
}

async fn upload_via_file_management<F>(
    provider_id: &str,
    api: &F,
    payload: UploadFilePayload,
) -> Result<UploadFileResult, LlmError>
where
    F: FileManagementCapability + ?Sized,
{
    let UploadFilePayload {
        data,
        media_type,
        filename,
        filename_was_explicit,
        purpose,
        metadata,
        provider_options,
        http_config,
    } = payload;

    let purpose = resolve_upload_purpose(provider_id, purpose.as_deref())?;

    let file = api
        .upload_file(FileUploadRequest {
            content: data,
            filename: filename.clone(),
            mime_type: Some(media_type.clone()),
            purpose,
            metadata,
            provider_options,
            http_config,
        })
        .await?;

    let mut result = upload_result_from_file_object(provider_id, file, filename, media_type);

    if matches!(provider_id, "gemini" | "google") && filename_was_explicit {
        result
            .warnings
            .push(Warning::unsupported("filename", None::<String>));
    }

    Ok(result)
}

fn upload_provider_namespace(provider_id: &str) -> &str {
    match provider_id {
        "gemini" | "google" => "google",
        other => other,
    }
}

fn resolve_upload_purpose(provider_id: &str, purpose: Option<&str>) -> Result<String, LlmError> {
    if let Some(purpose) = purpose.map(str::trim).filter(|purpose| !purpose.is_empty()) {
        return Ok(purpose.to_string());
    }

    match provider_id {
        "openai" | "azure" | "deepinfra" | "deepseek" | "fireworks" | "groq" | "mistral"
        | "openai-compatible" | "openaicompatible" | "perplexity" | "togetherai" | "xai" => {
            Ok("assistants".to_string())
        }
        "gemini" | "google" => Ok(String::new()),
        "minimaxi" => Err(LlmError::InvalidInput(
            "MiniMaxi file uploads require UploadFileOptions.purpose.".to_string(),
        )),
        _ => Err(LlmError::InvalidInput(format!(
            "File uploads for provider '{provider_id}' require UploadFileOptions.purpose."
        ))),
    }
}

fn upload_result_from_file_object(
    provider_id: &str,
    file: FileObject,
    fallback_filename: String,
    fallback_media_type: String,
) -> UploadFileResult {
    let provider_namespace = upload_provider_namespace(provider_id);
    let mut provider_metadata = file.metadata;
    provider_metadata.insert(
        "filename".to_string(),
        serde_json::Value::String(file.filename.clone()),
    );
    provider_metadata.insert(
        "purpose".to_string(),
        serde_json::Value::String(file.purpose.clone()),
    );
    provider_metadata.insert("bytes".to_string(), serde_json::json!(file.bytes));
    provider_metadata.insert("createdAt".to_string(), serde_json::json!(file.created_at));
    provider_metadata.insert(
        "status".to_string(),
        serde_json::Value::String(file.status.clone()),
    );
    if let Some(mime_type) = file.mime_type.as_ref() {
        provider_metadata
            .entry("mimeType".to_string())
            .or_insert_with(|| serde_json::Value::String(mime_type.clone()));
    }

    let filename = non_empty_string(file.filename).or_else(|| non_empty_string(fallback_filename));
    let media_type = file
        .mime_type
        .and_then(non_empty_string)
        .or_else(|| non_empty_string(fallback_media_type));

    UploadFileResult {
        provider_reference: ProviderReference::single(provider_namespace, file.id),
        media_type,
        filename,
        provider_metadata: Some(wrap_provider_metadata(
            provider_namespace,
            provider_metadata,
        )),
        warnings: Vec::new(),
    }
}

#[cfg(feature = "anthropic")]
fn upload_result_from_anthropic_file(
    file: siumai_provider_anthropic::providers::anthropic::files::AnthropicFile,
    fallback_filename: String,
    fallback_media_type: String,
) -> Result<UploadFileResult, LlmError> {
    let file_id = file
        .id
        .clone()
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| {
            LlmError::ParseError("Anthropic file upload response missing file id".to_string())
        })?;

    let mut provider_metadata = HashMap::new();

    if let Some(file_type) = file.r#type.clone().filter(|value| !value.trim().is_empty()) {
        provider_metadata.insert("type".to_string(), serde_json::Value::String(file_type));
    }

    if let Some(filename) = file
        .extra
        .get("filename")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        provider_metadata.insert(
            "filename".to_string(),
            serde_json::Value::String(filename.to_string()),
        );
    }

    if let Some(mime_type) = file
        .extra
        .get("mime_type")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        provider_metadata.insert(
            "mimeType".to_string(),
            serde_json::Value::String(mime_type.to_string()),
        );
    }

    if let Some(size_bytes) = file.extra.get("size_bytes") {
        provider_metadata.insert("sizeBytes".to_string(), size_bytes.clone());
    }

    if let Some(created_at) = file.extra.get("created_at") {
        provider_metadata.insert("createdAt".to_string(), created_at.clone());
    }

    if let Some(downloadable) = file.extra.get("downloadable") {
        provider_metadata.insert("downloadable".to_string(), downloadable.clone());
    }

    for (key, value) in file.extra {
        provider_metadata.entry(key).or_insert(value);
    }

    let filename = provider_metadata
        .get("filename")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| non_empty_string(fallback_filename));

    let media_type = provider_metadata
        .get("mimeType")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| non_empty_string(fallback_media_type));

    Ok(UploadFileResult {
        provider_reference: ProviderReference::single("anthropic", file_id),
        media_type,
        filename,
        provider_metadata: Some(wrap_provider_metadata("anthropic", provider_metadata)),
        warnings: Vec::new(),
    })
}

fn wrap_provider_metadata(
    provider_id: &str,
    provider_metadata: HashMap<String, serde_json::Value>,
) -> UploadFileProviderMetadata {
    let mut metadata = HashMap::new();
    metadata.insert(
        provider_id.to_string(),
        serde_json::Value::Object(provider_metadata.into_iter().collect()),
    );
    metadata
}

fn non_empty_string(value: impl Into<String>) -> Option<String> {
    let value = value.into();
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

#[cfg(feature = "anthropic")]
fn has_non_header_http_overrides(http_config: &HttpConfig) -> bool {
    http_config.timeout.is_some()
        || http_config.connect_timeout.is_some()
        || http_config.proxy.is_some()
        || http_config.user_agent.is_some()
}
