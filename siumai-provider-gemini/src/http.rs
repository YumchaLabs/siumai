use std::collections::BTreeMap;

use futures::StreamExt;
use http::StatusCode;
use http::header::HeaderName;
use siumai_core::{Error, ErrorKind, ResponseDiagnostics, SensitiveResponse};
use siumai_transport::framing::SseFrameError;
use siumai_transport::{ResponseHeaders, TransportResponse, TransportStreamResponse};

const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

pub(crate) fn response_error(response: TransportResponse, message: &'static str) -> Error {
    let (status, headers, body) = response.into_parts();
    build_response_error(
        status,
        headers,
        body[..body.len().min(ERROR_CAPTURE_BYTES)].to_vec(),
        body.len() > ERROR_CAPTURE_BYTES,
        message,
    )
}

pub(crate) async fn stream_response_error(
    response: TransportStreamResponse,
    message: &'static str,
) -> Error {
    let (status, headers, mut body) = response.into_parts();
    let mut captured = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = body.next().await {
        let Ok(chunk) = chunk else {
            truncated = true;
            break;
        };
        let remaining = ERROR_CAPTURE_BYTES.saturating_sub(captured.len());
        if chunk.len() > remaining {
            captured.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        captured.extend_from_slice(&chunk);
    }
    build_response_error(status, headers, captured, truncated, message)
}

pub(crate) fn response_diagnostics(
    status: StatusCode,
    _headers: &ResponseHeaders,
) -> ResponseDiagnostics {
    ResponseDiagnostics::default().with_status(status.as_u16())
}

pub(crate) fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    ["x-request-id", "x-goog-request-id"]
        .into_iter()
        .find_map(|name| {
            headers
                .get(&HeaderName::from_static(name))
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty() && value.len() <= 4 * 1024)
                .map(ToOwned::to_owned)
        })
}

pub(crate) fn sse_error(source: SseFrameError) -> Error {
    let kind = match source {
        SseFrameError::FrameTooLarge
        | SseFrameError::EventTooLarge
        | SseFrameError::TooManyEvents => ErrorKind::ResponseLimit,
        SseFrameError::InvalidUtf8 | SseFrameError::UnexpectedEof => ErrorKind::Protocol,
        _ => ErrorKind::Protocol,
    };
    Error::new(kind, "provider returned an invalid Gemini SSE stream").with_source(source)
}

fn build_response_error(
    status: StatusCode,
    headers: ResponseHeaders,
    body: Vec<u8>,
    truncated: bool,
    message: &'static str,
) -> Error {
    let kind = match status {
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => ErrorKind::InvalidInput,
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => ErrorKind::Timeout,
        StatusCode::SERVICE_UNAVAILABLE | StatusCode::BAD_GATEWAY => ErrorKind::Unavailable,
        _ => ErrorKind::Provider,
    };
    let diagnostics = response_diagnostics(status, &headers).with_body_truncated(truncated);
    Error::new(kind, message)
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(response_headers(&headers), body))
}

fn response_headers(headers: &ResponseHeaders) -> BTreeMap<String, String> {
    headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect()
}
