//! Shared transport and bounded diagnostics for provider-owned ARK resources.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use http::Method;
use http::header::{ACCEPT, HeaderValue, RETRY_AFTER};
use serde::de::DeserializeOwned;
use siumai_core::{
    CallOptions, Error, ErrorKind, ProviderInstanceId, PublicDiagnosticText, ResponseDiagnostics,
    SensitiveResponse,
};
use siumai_protocol_openai::openai_error::{classify_http_error, decode_error_metadata};
use siumai_transport::{
    ProviderTransport, ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget,
    ResponseHeaders, TransportResponse,
};

#[derive(Clone)]
pub(crate) struct ArkNativeRuntime {
    pub(crate) instance_id: ProviderInstanceId,
    transport: ProviderTransport,
}

impl ArkNativeRuntime {
    pub(crate) fn new(instance_id: ProviderInstanceId, transport: ProviderTransport) -> Self {
        Self {
            instance_id,
            transport,
        }
    }
}

impl fmt::Debug for ArkNativeRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkNativeRuntime")
            .field("transport", &"shared")
            .finish()
    }
}

pub(crate) type SharedArkNativeRuntime = Arc<ArkNativeRuntime>;

pub(crate) async fn execute_json<T: DeserializeOwned>(
    runtime: &ArkNativeRuntime,
    method: Method,
    target: RequestTarget,
    body: RequestBody,
    replay_safety: ReplaySafety,
    options: CallOptions,
) -> Result<T, Error> {
    let response = execute(runtime, method, target, body, replay_safety, options).await?;
    let (status, headers, body) = response.into_parts();
    serde_json::from_slice(&body).map_err(|source| {
        attach_response(
            Error::new(
                ErrorKind::Protocol,
                "ARK resource response was not valid JSON",
            )
            .with_source(source),
            status.as_u16(),
            &headers,
            &body,
        )
    })
}

async fn execute(
    runtime: &ArkNativeRuntime,
    method: Method,
    target: RequestTarget,
    body: RequestBody,
    replay_safety: ReplaySafety,
    options: CallOptions,
) -> Result<TransportResponse, Error> {
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
        .map_err(|source| {
            Error::new(ErrorKind::Configuration, "ARK resource headers are invalid")
                .with_source(source)
        })?;
    let plan = RequestPlan::new(method, target)
        .with_headers(headers)
        .with_body(body)
        .with_replay_safety(replay_safety)
        .map_err(|source| {
            Error::new(
                ErrorKind::Configuration,
                "ARK resource request is not replay-safe",
            )
            .with_source(source)
        })?;
    let response = runtime.transport.execute(plan, options).await?;
    if response.status().is_success() {
        Ok(response)
    } else {
        Err(resource_error(response))
    }
}

fn resource_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let metadata = decode_error_metadata(&body);
    let kind = classify_http_error(
        status.as_u16(),
        metadata.as_ref().and_then(|value| value.code()),
        metadata.as_ref().and_then(|value| value.error_type()),
    );
    let mut error = Error::new(kind, "ARK resource request was rejected");
    let mut diagnostics = ResponseDiagnostics::default().with_status(status.as_u16());
    if let Some(metadata) = metadata {
        if let Some(code) = metadata
            .code()
            .and_then(|value| PublicDiagnosticText::new(value.to_string()).ok())
        {
            diagnostics = diagnostics.with_provider_code(code);
        }
        if let Some(error_type) = metadata
            .error_type()
            .and_then(|value| PublicDiagnosticText::new(value.to_string()).ok())
        {
            diagnostics = diagnostics.with_provider_type(error_type);
        }
        if let Some(param) = metadata
            .param()
            .and_then(|value| PublicDiagnosticText::new(value.to_string()).ok())
        {
            diagnostics = diagnostics.with_provider_param(param);
        }
    }
    if let Some(request_id) = response_header_text(&headers, "x-request-id")
        .or_else(|| response_header_text(&headers, "request-id"))
    {
        diagnostics = diagnostics.with_request_id(request_id);
    }
    if let Some(retry_after) = headers
        .get(&RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
    {
        diagnostics = diagnostics.with_retry_after(retry_after);
    }
    let sensitive = sensitive_response(&headers, &body);
    diagnostics = diagnostics.with_body_truncated(sensitive.was_truncated());
    error = error.with_diagnostics(diagnostics);
    error.with_sensitive_response(sensitive)
}

fn attach_response(error: Error, status: u16, headers: &ResponseHeaders, body: &[u8]) -> Error {
    let sensitive = sensitive_response(headers, body);
    error
        .with_diagnostics(
            ResponseDiagnostics::default()
                .with_status(status)
                .with_body_truncated(sensitive.was_truncated()),
        )
        .with_sensitive_response(sensitive)
}

fn sensitive_response(headers: &ResponseHeaders, body: &[u8]) -> SensitiveResponse {
    SensitiveResponse::new(
        headers
            .expose()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.to_string(), value.to_string()))
            })
            .collect(),
        body.to_vec(),
    )
}

fn response_header_text(
    headers: &ResponseHeaders,
    name: &'static str,
) -> Option<PublicDiagnosticText> {
    headers
        .expose()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| PublicDiagnosticText::new(value.to_string()).ok())
}

pub(crate) fn target(value: impl Into<String>) -> Result<RequestTarget, Error> {
    RequestTarget::new(value.into()).map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "ARK resource target is invalid").with_source(source)
    })
}

pub(crate) fn validate_identifier(value: &str, kind: &'static str) -> Result<(), Error> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 512
        || value.chars().any(char::is_control)
        || value.contains('/')
    {
        return Err(Error::new(ErrorKind::InvalidInput, kind));
    }
    Ok(())
}
