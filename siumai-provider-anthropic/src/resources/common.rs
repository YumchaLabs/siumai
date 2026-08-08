use std::collections::BTreeSet;

use bytes::Bytes;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use http::{Method, StatusCode};
use serde::{Deserialize, de::DeserializeOwned};
use siumai_core::{CallOptions, Error, ErrorKind, PublicDiagnosticText, SensitiveResponse};
use siumai_transport::{
    MultipartBody, ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget,
    ResponseHeaders, TransportResponse,
};

use super::NativeRuntime;

pub(crate) const ANTHROPIC_VERSION_HEADER: HeaderName =
    HeaderName::from_static("anthropic-version");
pub(crate) const ANTHROPIC_BETA_HEADER: HeaderName = HeaderName::from_static("anthropic-beta");

pub(crate) fn validate_resource_id(id: &str) -> Result<(), Error> {
    if id.is_empty()
        || id.len() > 256
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic resource identifier is invalid",
        ));
    }
    Ok(())
}

pub(crate) fn target(value: impl Into<String>) -> Result<RequestTarget, Error> {
    RequestTarget::new(value.into()).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Anthropic resource target is invalid",
        )
        .with_source(source)
    })
}

pub(crate) fn headers(
    runtime: &NativeRuntime,
    additional_beta: &[&str],
    accept: &'static str,
) -> Result<RequestHeaders, Error> {
    let mut headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static(accept))
        .map_err(|source| request_error("failed to build Anthropic resource headers", source))?
        .try_insert(
            ANTHROPIC_VERSION_HEADER,
            HeaderValue::from_str(runtime.api_version.as_ref()).map_err(|source| {
                request_error("Anthropic API version header is invalid", source)
            })?,
        )
        .map_err(|source| request_error("failed to build Anthropic version header", source))?;

    let mut beta = BTreeSet::new();
    beta.extend(runtime.beta_features.iter().map(String::as_str));
    beta.extend(additional_beta.iter().copied());
    if !beta.is_empty() {
        let value = beta.into_iter().collect::<Vec<_>>().join(",");
        headers = headers
            .try_insert(
                ANTHROPIC_BETA_HEADER,
                HeaderValue::from_str(&value).map_err(|source| {
                    request_error("Anthropic beta feature header is invalid", source)
                })?,
            )
            .map_err(|source| request_error("failed to build Anthropic beta header", source))?;
    }
    Ok(headers)
}

pub(crate) async fn execute_json<T: DeserializeOwned>(
    runtime: &NativeRuntime,
    method: Method,
    target: RequestTarget,
    body: RequestBody,
    replay: ReplaySafety,
    additional_beta: &[&str],
    options: CallOptions,
) -> Result<T, Error> {
    let plan = RequestPlan::new(method, target)
        .with_headers(headers(runtime, additional_beta, "application/json")?)
        .with_body(body)
        .with_replay_safety(replay)
        .map_err(|source| request_error("Anthropic resource request is not replay-safe", source))?;
    let response = runtime.transport.execute(plan, options).await?;
    let (_status, _headers, body) = ensure_success(response)?;
    serde_json::from_slice(&body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Anthropic resource response was not valid JSON",
        )
        .with_source(source)
    })
}

pub(crate) async fn execute_download(
    runtime: &NativeRuntime,
    target: RequestTarget,
    additional_beta: &[&str],
    accept: &'static str,
    options: CallOptions,
) -> Result<Bytes, Error> {
    let plan = RequestPlan::new(Method::GET, target)
        .with_headers(headers(runtime, additional_beta, accept)?)
        .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
        .map_err(|source| request_error("Anthropic resource request is not replay-safe", source))?;
    let response = runtime.transport.execute(plan, options).await?;
    let (_status, _headers, body) = ensure_success(response)?;
    Ok(body)
}

pub(crate) fn multipart_body(body: MultipartBody) -> RequestBody {
    RequestBody::multipart(body)
}

fn ensure_success(
    response: TransportResponse,
) -> Result<(StatusCode, siumai_transport::ResponseHeaders, Bytes), Error> {
    let (status, headers, body) = response.into_parts();
    if status.is_success() {
        return Ok((status, headers, body));
    }
    Err(resource_status_error(status, headers, body))
}

fn request_error<E>(message: &'static str, source: E) -> Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    Error::new(ErrorKind::Configuration, message).with_source(source)
}

#[derive(Deserialize)]
struct ErrorEnvelope {
    #[serde(default)]
    error: Option<ErrorBody>,
    #[serde(default)]
    request_id: Option<String>,
}

#[derive(Deserialize)]
struct ErrorBody {
    #[serde(rename = "type")]
    kind: Option<String>,
}

fn resource_status_error(status: StatusCode, headers: ResponseHeaders, body: Bytes) -> Error {
    let envelope = serde_json::from_slice::<ErrorEnvelope>(&body).ok();
    let provider_type = envelope
        .as_ref()
        .and_then(|envelope| envelope.error.as_ref())
        .and_then(|error| error.kind.as_deref())
        .and_then(public_provider_identifier);
    let kind = classify_http_error(
        status,
        provider_type.as_ref().map(PublicDiagnosticText::as_str),
    );
    let body_request_id = envelope
        .as_ref()
        .and_then(|envelope| envelope.request_id.as_deref())
        .and_then(public_provider_identifier);
    let raw_headers = headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect();
    let sensitive = SensitiveResponse::new(raw_headers, body.to_vec());
    let mut diagnostics = headers
        .diagnostics()
        .with_status(status.as_u16())
        .with_body_truncated(sensitive.was_truncated());
    if let Some(provider_type) = provider_type {
        diagnostics = diagnostics.with_provider_type(provider_type);
    }
    if diagnostics.request_id().is_none()
        && let Some(request_id) = body_request_id
    {
        diagnostics = diagnostics.with_request_id(request_id);
    }
    Error::new(kind, "Anthropic resource request was rejected")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(sensitive)
}

fn classify_http_error(status: StatusCode, provider_type: Option<&str>) -> ErrorKind {
    match provider_type {
        Some("authentication_error") => return ErrorKind::Authentication,
        Some("billing_error") => return ErrorKind::QuotaExceeded,
        Some("permission_error") => return ErrorKind::Authorization,
        Some("rate_limit_error") => return ErrorKind::RateLimited,
        Some("invalid_request_error" | "not_found_error") => return ErrorKind::InvalidInput,
        Some("request_too_large") => return ErrorKind::LimitExceeded,
        Some("timeout_error") => return ErrorKind::Timeout,
        Some("api_error" | "overloaded_error") => return ErrorKind::Unavailable,
        _ => {}
    }
    match status.as_u16() {
        400 | 404 | 405 | 409 | 422 => ErrorKind::InvalidInput,
        401 => ErrorKind::Authentication,
        402 => ErrorKind::QuotaExceeded,
        403 => ErrorKind::Authorization,
        408 | 504 => ErrorKind::Timeout,
        413 => ErrorKind::LimitExceeded,
        429 => ErrorKind::RateLimited,
        500..=599 => ErrorKind::Unavailable,
        _ => ErrorKind::Provider,
    }
}

fn public_provider_identifier(value: &str) -> Option<PublicDiagnosticText> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return None;
    }
    PublicDiagnosticText::new(value.to_string()).ok()
}
