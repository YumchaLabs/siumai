use std::collections::BTreeMap;

use futures_util::StreamExt;
use http::StatusCode;
use siumai_core::{Error, ErrorKind, PublicDiagnosticText, SensitiveResponse};
use siumai_protocol_openai::openai_error::{classify_http_error, decode_error_metadata};
use siumai_transport::{
    RequestBuildError, ResponseHeaders, TransportResponse, TransportStreamResponse,
};

const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

pub(crate) fn request_build_error(message: &'static str, source: RequestBuildError) -> Error {
    Error::new(ErrorKind::InvalidInput, message).with_source(source)
}

pub(crate) fn response_error(message: &'static str, response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let truncated = body.len() > ERROR_CAPTURE_BYTES;
    let captured = body[..body.len().min(ERROR_CAPTURE_BYTES)].to_vec();
    provider_status_error(message, status, headers, captured, truncated)
}

pub(crate) async fn stream_response_error(
    message: &'static str,
    response: TransportStreamResponse,
) -> Error {
    let (status, headers, mut body) = response.into_parts();
    let mut bytes = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(chunk) => {
                let remaining = ERROR_CAPTURE_BYTES.saturating_sub(bytes.len());
                if remaining == 0 {
                    truncated = true;
                    break;
                }
                bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                if chunk.len() > remaining {
                    truncated = true;
                    break;
                }
            }
            Err(error) => return error,
        }
    }
    provider_status_error(message, status, headers, bytes, truncated)
}

fn provider_status_error(
    message: &'static str,
    status: StatusCode,
    headers: ResponseHeaders,
    body: Vec<u8>,
    body_truncated: bool,
) -> Error {
    let metadata = decode_error_metadata(&body);
    let provider_code = metadata
        .as_ref()
        .and_then(|metadata| metadata.code())
        .and_then(public_provider_identifier);
    let provider_type = metadata
        .as_ref()
        .and_then(|metadata| metadata.error_type())
        .and_then(public_provider_identifier);
    let provider_param = metadata
        .as_ref()
        .and_then(|metadata| metadata.param())
        .and_then(public_provider_identifier);
    let kind = classify_http_error(
        status.as_u16(),
        provider_code.as_ref().map(PublicDiagnosticText::as_str),
        provider_type.as_ref().map(PublicDiagnosticText::as_str),
    );
    let mut diagnostics = headers
        .diagnostics()
        .with_status(status.as_u16())
        .with_body_truncated(body_truncated);
    if let Some(code) = provider_code {
        diagnostics = diagnostics.with_provider_code(code);
    }
    if let Some(kind) = provider_type {
        diagnostics = diagnostics.with_provider_type(kind);
    }
    if let Some(param) = provider_param {
        diagnostics = diagnostics.with_provider_param(param);
    }
    let raw_headers = headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect::<BTreeMap<_, _>>();
    Error::new(kind, message)
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body))
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
