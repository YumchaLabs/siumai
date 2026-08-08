use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use http::Method;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorContext, ErrorKind};
use siumai_protocol_openai::resources::{
    OpenAiChunkingStrategy, OpenAiFileExpiresAfter, OpenAiMetadata, OpenAiVectorStoreExpiration,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget, TransportResponse,
};

use super::super::http_error;
use super::super::mode::OpenAiApiMode;
use super::super::provider::OpenAiRuntime;

pub(crate) const MAX_RESOURCE_ID_BYTES: usize = 512;
pub(crate) const MAX_METADATA_ENTRIES: usize = 16;
pub(crate) const MAX_METADATA_KEY_BYTES: usize = 64;
pub(crate) const MAX_METADATA_VALUE_BYTES: usize = 512;
pub(crate) const MAX_FILE_EXPIRATION_SECONDS: u32 = 2_592_000;
pub(crate) const MIN_FILE_EXPIRATION_SECONDS: u32 = 3_600;
const MAX_VECTOR_STORE_EXPIRATION_DAYS: u32 = 365;

#[derive(Clone, Copy)]
enum OpenAiResourceApiHeaders {
    Default,
    VectorStoresV2,
}

/// Bounded provider-native binary content with payload-redacted diagnostics.
#[derive(Clone, PartialEq, Eq)]
pub struct OpenAiBinaryContent {
    bytes: Vec<u8>,
}

impl OpenAiBinaryContent {
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl fmt::Debug for OpenAiBinaryContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiBinaryContent")
            .field("bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(crate) struct OpenAiNativeRuntime {
    pub(crate) inner: Arc<OpenAiRuntime>,
}

impl OpenAiNativeRuntime {
    pub(crate) fn new(inner: Arc<OpenAiRuntime>) -> Self {
        Self { inner }
    }

    pub(crate) async fn execute_json<T: DeserializeOwned>(
        &self,
        method: Method,
        target: RequestTarget,
        body: RequestBody,
        replay_safety: ReplaySafety,
        options: CallOptions,
    ) -> Result<T, Error> {
        self.execute_json_with_api_headers(
            method,
            target,
            body,
            replay_safety,
            OpenAiResourceApiHeaders::Default,
            options,
        )
        .await
    }

    pub(crate) async fn execute_vector_store_json<T: DeserializeOwned>(
        &self,
        method: Method,
        target: RequestTarget,
        body: RequestBody,
        replay_safety: ReplaySafety,
        options: CallOptions,
    ) -> Result<T, Error> {
        self.execute_json_with_api_headers(
            method,
            target,
            body,
            replay_safety,
            OpenAiResourceApiHeaders::VectorStoresV2,
            options,
        )
        .await
    }

    async fn execute_json_with_api_headers<T: DeserializeOwned>(
        &self,
        method: Method,
        target: RequestTarget,
        body: RequestBody,
        replay_safety: ReplaySafety,
        api_headers: OpenAiResourceApiHeaders,
        options: CallOptions,
    ) -> Result<T, Error> {
        let response = self
            .execute(
                method,
                target,
                body,
                replay_safety,
                "application/json",
                api_headers,
                options,
            )
            .await?;
        serde_json::from_slice(response.body()).map_err(|source| {
            self.contextualize(
                Error::new(
                    ErrorKind::Protocol,
                    "OpenAI resource response was not valid JSON",
                )
                .with_source(source),
            )
        })
    }

    pub(crate) async fn execute_bytes(
        &self,
        target: RequestTarget,
        accept: &'static str,
        options: CallOptions,
    ) -> Result<OpenAiBinaryContent, Error> {
        let response = self
            .execute(
                Method::GET,
                target,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                accept,
                OpenAiResourceApiHeaders::Default,
                options,
            )
            .await?;
        Ok(OpenAiBinaryContent::new(response.body().to_vec()))
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute(
        &self,
        method: Method,
        target: RequestTarget,
        body: RequestBody,
        replay_safety: ReplaySafety,
        accept: &'static str,
        api_headers: OpenAiResourceApiHeaders,
        options: CallOptions,
    ) -> Result<TransportResponse, Error> {
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static(accept))
            .and_then(|headers| match api_headers {
                OpenAiResourceApiHeaders::Default => Ok(headers),
                OpenAiResourceApiHeaders::VectorStoresV2 => headers.try_insert(
                    HeaderName::from_static("openai-beta"),
                    HeaderValue::from_static("assistants=v2"),
                ),
            })
            .map_err(|source| {
                http_error::request_build_error(
                    "OpenAI resource request violates the transport contract",
                    source,
                )
            })?;
        let plan = RequestPlan::new(method, target)
            .with_headers(headers)
            .with_body(body)
            .with_replay_safety(replay_safety)
            .map_err(|source| {
                http_error::request_build_error(
                    "OpenAI resource request violates the transport contract",
                    source,
                )
            })?;
        let response = self
            .inner
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if response.status().is_success() {
            Ok(response)
        } else {
            Err(self.contextualize(http_error::response_error(
                "OpenAI rejected the resource request",
                response,
            )))
        }
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: None,
            provider: Some(
                self.inner
                    .scope(OpenAiApiMode::Responses)
                    .provider_id()
                    .clone(),
            ),
            route: None,
            model: None,
        })
    }
}

impl fmt::Debug for OpenAiNativeRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiNativeRuntime")
            .field("transport", &"shared")
            .finish()
    }
}

pub(crate) fn json_body(value: &impl Serialize) -> Result<RequestBody, Error> {
    RequestBody::json(value).map_err(|source| {
        http_error::request_build_error(
            "OpenAI resource JSON body violates the transport contract",
            source,
        )
    })
}

pub(crate) fn target(path: impl Into<String>) -> Result<RequestTarget, Error> {
    RequestTarget::new(path.into()).map_err(|source| {
        http_error::request_build_error(
            "OpenAI resource target violates the transport contract",
            source,
        )
    })
}

pub(crate) fn target_with_query(
    path: &str,
    pairs: impl IntoIterator<Item = (&'static str, String)>,
) -> Result<RequestTarget, Error> {
    let pairs = pairs.into_iter().collect::<Vec<_>>();
    if pairs.is_empty() {
        return target(path);
    }
    let query = pairs
        .into_iter()
        .map(|(name, value)| format!("{name}={}", urlencoding::encode(&value)))
        .collect::<Vec<_>>()
        .join("&");
    target(format!("{path}?{query}"))
}

pub(crate) fn validate_resource_id(value: &str) -> Result<(), Error> {
    if value.is_empty()
        || value.len() > MAX_RESOURCE_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(invalid_input("OpenAI resource identifier is invalid"));
    }
    Ok(())
}

pub(crate) fn validate_metadata(metadata: &OpenAiMetadata) -> Result<(), Error> {
    if metadata.len() > MAX_METADATA_ENTRIES
        || metadata.iter().any(|(key, value)| {
            key.is_empty()
                || key.len() > MAX_METADATA_KEY_BYTES
                || value.len() > MAX_METADATA_VALUE_BYTES
        })
    {
        return Err(invalid_input("OpenAI resource metadata is invalid"));
    }
    Ok(())
}

pub(crate) fn validate_attributes(attributes: &BTreeMap<String, Value>) -> Result<(), Error> {
    if attributes.len() > MAX_METADATA_ENTRIES
        || attributes.iter().any(|(key, value)| {
            key.is_empty()
                || key.len() > MAX_METADATA_KEY_BYTES
                || match value {
                    Value::String(value) => value.len() > MAX_METADATA_VALUE_BYTES,
                    Value::Bool(_) | Value::Number(_) => false,
                    _ => true,
                }
        })
    {
        return Err(invalid_input(
            "OpenAI vector-store file attributes are invalid",
        ));
    }
    Ok(())
}

pub(crate) fn validate_chunking(strategy: OpenAiChunkingStrategy) -> Result<(), Error> {
    if let OpenAiChunkingStrategy::Static { settings } = strategy
        && (!(100..=4096).contains(&settings.max_chunk_size_tokens)
            || settings.chunk_overlap_tokens > settings.max_chunk_size_tokens / 2)
    {
        return Err(invalid_input(
            "OpenAI vector-store chunking strategy is invalid",
        ));
    }
    Ok(())
}

pub(crate) fn validate_file_expiration(expires: &OpenAiFileExpiresAfter) -> Result<(), Error> {
    if !(MIN_FILE_EXPIRATION_SECONDS..=MAX_FILE_EXPIRATION_SECONDS).contains(&expires.seconds) {
        return Err(invalid_input(
            "OpenAI file expiration is outside the supported range",
        ));
    }
    Ok(())
}

pub(crate) fn validate_vector_store_expiration(
    expires: OpenAiVectorStoreExpiration,
) -> Result<(), Error> {
    if !(1..=MAX_VECTOR_STORE_EXPIRATION_DAYS).contains(&expires.days) {
        return Err(invalid_input(
            "OpenAI vector-store expiration must be between 1 and 365 days",
        ));
    }
    Ok(())
}

pub(crate) fn validate_bounded_text(
    value: &str,
    maximum: usize,
    message: &'static str,
) -> Result<(), Error> {
    if value.is_empty()
        || value.len() > maximum
        || value != value.trim()
        || value.chars().any(char::is_control)
    {
        return Err(invalid_input(message));
    }
    Ok(())
}

pub(crate) fn invalid_input(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
