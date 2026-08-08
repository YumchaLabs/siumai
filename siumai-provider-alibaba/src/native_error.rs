use std::collections::BTreeMap;

use http::StatusCode;
use http::header::HeaderName;
use serde::Deserialize;
use siumai_core::{Error, ErrorKind, PublicDiagnosticText, SensitiveResponse};
use siumai_transport::{ResponseHeaders, TransportResponse};

const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

#[derive(Debug, Deserialize)]
struct AlibabaErrorWire {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    request_id: Option<String>,
}

pub(crate) fn provider_status_error(
    response: TransportResponse,
    message: &'static str,
    classify_status: impl FnOnce(StatusCode) -> ErrorKind,
) -> Error {
    let (status, headers, body) = response.into_parts();
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
    let error_body = serde_json::from_slice::<AlibabaErrorWire>(&body).ok();
    let mut diagnostics = headers
        .diagnostics()
        .with_status(status.as_u16())
        .with_body_truncated(body.len() > ERROR_CAPTURE_BYTES);
    if let Some(code) = error_body.as_ref().and_then(|error| error.code.as_deref())
        && let Ok(code) = PublicDiagnosticText::new(code)
    {
        diagnostics = diagnostics.with_provider_code(code);
    }
    let request_id = error_body
        .and_then(|error| error.request_id)
        .or_else(|| response_request_id(&headers));
    if let Some(request_id) = request_id
        && let Ok(request_id) = PublicDiagnosticText::new(request_id)
    {
        diagnostics = diagnostics.with_request_id(request_id);
    }
    Error::new(classify_status(status), message)
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::with_limit(
            raw_headers,
            body.to_vec(),
            ERROR_CAPTURE_BYTES,
        ))
}

pub(crate) fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    headers
        .get(&HeaderName::from_static("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}
