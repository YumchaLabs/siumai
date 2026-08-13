use std::fmt;
use std::time::Duration;

use http::header::{ACCEPT, HeaderValue, RETRY_AFTER};
use http::{Method, StatusCode};
use serde::{Deserialize, de::DeserializeOwned};
use siumai_core::{
    CallOptions, Error, ErrorKind, ProviderInstanceId, PublicDiagnosticText, ResponseDiagnostics,
    SensitiveResponse,
};
use siumai_transport::{
    MultipartBody, ProviderTransport, ReplaySafety, RequestBody, RequestHeaders, RequestPlan,
    RequestTarget, ResponseHeaders, TransportResponse,
};

/// Shared runtime for MiniMax provider-native resources.
///
/// Authentication, endpoint policy, retries, and request/response bounds are
/// configured once on the transport by the provider builder.
pub(crate) struct NativeRuntime {
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) transport: ProviderTransport,
}

impl NativeRuntime {
    pub(crate) fn new(instance_id: ProviderInstanceId, transport: ProviderTransport) -> Self {
        Self {
            instance_id,
            transport,
        }
    }

    pub(crate) fn max_request_bytes(&self) -> usize {
        self.transport.limits().max_request_bytes
    }

    pub(crate) fn max_response_bytes(&self) -> usize {
        self.transport.limits().max_response_bytes
    }
}

impl fmt::Debug for NativeRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeRuntime")
            .field("transport", &"shared")
            .field("max_request_bytes", &self.max_request_bytes())
            .field("max_response_bytes", &self.max_response_bytes())
            .finish()
    }
}

#[derive(Deserialize)]
pub(crate) struct BaseResponse {
    pub(crate) status_code: i64,
    #[serde(default, rename = "status_msg")]
    _status_message: Option<String>,
}

pub(crate) trait NativeResponseEnvelope {
    const BASE_RESPONSE_REQUIRED: bool = true;

    fn base_response(&self) -> Option<&BaseResponse>;
}

pub(crate) fn target(value: impl Into<String>) -> Result<RequestTarget, Error> {
    RequestTarget::new(value.into()).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "MiniMax resource target is invalid",
        )
        .with_source(source)
    })
}

pub(crate) async fn execute_json<T>(
    runtime: &NativeRuntime,
    method: Method,
    target: RequestTarget,
    body: RequestBody,
    replay_safety: ReplaySafety,
    options: CallOptions,
) -> Result<T, Error>
where
    T: DeserializeOwned + NativeResponseEnvelope,
{
    let plan = RequestPlan::new(method, target)
        .with_headers(accept_header("application/json")?)
        .with_body(body)
        .with_replay_safety(replay_safety)
        .map_err(|source| request_error("MiniMax resource request is not replay-safe", source))?;
    let response = runtime.transport.execute(plan, options).await?;
    let (status, headers, body) = ensure_http_success(response)?;
    let decoded = serde_json::from_slice::<T>(&body).map_err(|source| {
        attach_response(
            Error::new(
                ErrorKind::Protocol,
                "MiniMax resource response was not valid JSON",
            )
            .with_source(source),
            status,
            &headers,
            &body,
        )
    })?;
    match decoded.base_response() {
        Some(base) => {
            if let Err(error) = validate_base_response(Some(base)) {
                return Err(attach_response(error, status, &headers, &body));
            }
        }
        None if T::BASE_RESPONSE_REQUIRED => {
            return Err(attach_response(
                Error::new(
                    ErrorKind::Protocol,
                    "MiniMax resource response omitted base_resp",
                ),
                status,
                &headers,
                &body,
            ));
        }
        None => {}
    }
    Ok(decoded)
}

pub(crate) async fn execute_download(
    runtime: &NativeRuntime,
    target: RequestTarget,
    options: CallOptions,
) -> Result<Vec<u8>, Error> {
    let plan = RequestPlan::new(Method::GET, target)
        .with_headers(accept_header("application/octet-stream")?)
        .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
        .map_err(|source| request_error("MiniMax resource request is not replay-safe", source))?;
    let response = runtime.transport.execute(plan, options).await?;
    let (_status, _headers, body) = ensure_http_success(response)?;
    validate_body_size(body.len(), runtime.max_response_bytes())?;
    Ok(body)
}

pub(crate) fn multipart_body(body: MultipartBody) -> RequestBody {
    RequestBody::multipart(body)
}

pub(crate) fn validate_body_size(actual: usize, maximum: usize) -> Result<(), Error> {
    if actual > maximum {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "MiniMax resource body exceeds the configured limit",
        ));
    }
    Ok(())
}

pub(crate) fn validate_base_response(base: Option<&BaseResponse>) -> Result<(), Error> {
    let Some(base) = base else {
        return Err(Error::new(
            ErrorKind::Protocol,
            "MiniMax resource response omitted base_resp",
        ));
    };
    if base.status_code == 0 {
        return Ok(());
    }
    Err(base_response_error(base.status_code))
}

fn accept_header(accept: &'static str) -> Result<RequestHeaders, Error> {
    RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static(accept))
        .map_err(|source| request_error("failed to build MiniMax resource headers", source))
}

fn ensure_http_success(
    response: TransportResponse,
) -> Result<(StatusCode, ResponseHeaders, Vec<u8>), Error> {
    let (status, headers, body) = response.into_parts();
    if status.is_success() {
        return Ok((status, headers, body.to_vec()));
    }

    let error = serde_json::from_slice::<BaseResponseErrorEnvelope>(&body)
        .ok()
        .and_then(|envelope| envelope.base_resp)
        .filter(|base| base.status_code != 0)
        .map_or_else(
            || {
                Error::new(
                    classify_http_error(status),
                    "MiniMax resource request was rejected",
                )
            },
            |base| base_response_error(base.status_code),
        );
    Err(attach_response(error, status, &headers, &body))
}

#[derive(Deserialize)]
struct BaseResponseErrorEnvelope {
    #[serde(default)]
    base_resp: Option<BaseResponse>,
}

fn base_response_error(code: i64) -> Error {
    let mut diagnostics =
        ResponseDiagnostics::default().with_provider_type(PublicDiagnosticText::from("base_resp"));
    if let Ok(code) = PublicDiagnosticText::new(code.to_string()) {
        diagnostics = diagnostics.with_provider_code(code);
    }
    Error::new(
        classify_base_response(code),
        "MiniMax resource request was rejected",
    )
    .with_diagnostics(diagnostics)
}

fn classify_base_response(code: i64) -> ErrorKind {
    match code {
        1001 => ErrorKind::Timeout,
        1002 | 1039 => ErrorKind::RateLimited,
        1004 | 2049 => ErrorKind::Authentication,
        1008 => ErrorKind::QuotaExceeded,
        1026 | 2013 => ErrorKind::InvalidInput,
        1000 | 1013 | 1027 => ErrorKind::Provider,
        _ => ErrorKind::Provider,
    }
}

fn classify_http_error(status: StatusCode) -> ErrorKind {
    match status.as_u16() {
        400 | 404 | 405 | 409 | 422 => ErrorKind::InvalidInput,
        401 => ErrorKind::Authentication,
        403 => ErrorKind::Authorization,
        408 | 504 => ErrorKind::Timeout,
        413 => ErrorKind::LimitExceeded,
        429 => ErrorKind::RateLimited,
        500..=599 => ErrorKind::Provider,
        _ => ErrorKind::Provider,
    }
}

fn attach_response(
    error: Error,
    status: StatusCode,
    headers: &ResponseHeaders,
    body: &[u8],
) -> Error {
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
    let mut diagnostics = error
        .diagnostics()
        .cloned()
        .unwrap_or_default()
        .with_status(status.as_u16())
        .with_body_truncated(sensitive.was_truncated());
    if let Some(request_id) = response_header_text(headers, "request-id")
        .or_else(|| response_header_text(headers, "x-request-id"))
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
    error
        .with_diagnostics(diagnostics)
        .with_sensitive_response(sensitive)
}

fn request_error<E>(message: &'static str, source: E) -> Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    Error::new(ErrorKind::Configuration, message).with_source(source)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_response_codes_are_classified_without_exposing_status_message() {
        let base: BaseResponse = serde_json::from_value(serde_json::json!({
            "status_code": 1004,
            "status_msg": "secret upstream detail"
        }))
        .expect("base response should decode");

        let error = validate_base_response(Some(&base)).expect_err("code must fail");
        assert_eq!(error.kind(), ErrorKind::Authentication);
        assert_eq!(
            error
                .diagnostics()
                .and_then(ResponseDiagnostics::provider_code),
            Some("1004")
        );
        assert!(!error.to_string().contains("secret upstream detail"));
        assert!(!format!("{error:?}").contains("secret upstream detail"));
    }

    #[test]
    fn missing_base_response_is_a_protocol_error() {
        let error = validate_base_response(None).expect_err("missing base_resp must fail");
        assert_eq!(error.kind(), ErrorKind::Protocol);
    }

    #[test]
    fn resource_body_limit_is_inclusive() {
        assert!(validate_body_size(16, 16).is_ok());
        let error = validate_body_size(17, 16).expect_err("oversized body must fail");
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);
    }
}
