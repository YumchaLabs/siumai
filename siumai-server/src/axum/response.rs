use axum::body::Body;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE, HeaderName, HeaderValue};
use axum::http::{StatusCode, header};
use axum::response::Response;
use serde_json::json;
use siumai_core::LanguageResponse;
use siumai_runtime::RunTerminal;

use crate::{
    GatewayEvent, GatewayPolicy, GatewayProjectionError, ServerGatewayError, ServerTrustContext,
};

const ROUTE_HEADER: &str = "x-siumai-route";
const PROJECTION_POLICY_HEADER: &str = "x-siumai-projection-policy";

/// Project a terminal language response into the canonical server JSON shape.
pub fn language_response(response: LanguageResponse, policy: &GatewayPolicy) -> Response<Body> {
    match GatewayEvent::from_language_response(response, policy) {
        Ok(event) => event_response(StatusCode::OK, event, None, policy),
        Err(error) => projection_error_response(&error, None, policy),
    }
}

/// Project a trusted tool-run terminal into the canonical server JSON shape.
pub fn run_response(
    terminal: RunTerminal,
    trust: &ServerTrustContext,
    policy: &GatewayPolicy,
) -> Response<Body> {
    if terminal
        .report()
        .and_then(|report| report.initial_target().route())
        .is_some_and(|route| route != trust.route())
    {
        return error_response(
            StatusCode::FORBIDDEN,
            "trust_route_mismatch",
            "authenticated request is not authorized for this route",
            None,
            policy,
        );
    }
    let status = run_terminal_status(&terminal);
    match GatewayEvent::from_run_terminal(terminal, policy) {
        Ok(event) => event_response(status, event, Some(trust), policy),
        Err(error) => projection_error_response(&error, Some(trust), policy),
    }
}

/// Convert a gateway/runtime failure into a sanitized JSON response.
pub fn gateway_error_response(
    error: &ServerGatewayError,
    policy: &GatewayPolicy,
) -> Response<Body> {
    let (status, code, public_message) = match error {
        ServerGatewayError::LocalToolsDisabled => (
            StatusCode::FORBIDDEN,
            "local_tools_disabled",
            "local tool execution is disabled for this route",
        ),
        ServerGatewayError::TrustRouteMismatch => (
            StatusCode::FORBIDDEN,
            "trust_route_mismatch",
            "authenticated request is not authorized for this route",
        ),
        ServerGatewayError::ModelRouteMismatch => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "model_route_mismatch",
            "server route configuration is inconsistent",
        ),
        ServerGatewayError::LanguageCall(error) => {
            return language_call_error_response(error, policy);
        }
        ServerGatewayError::Runtime(_) => (
            StatusCode::BAD_GATEWAY,
            "runtime_failed",
            "model runtime request failed",
        ),
    };
    error_response(status, code, public_message, None, policy)
}

fn language_call_error_response(
    error: &siumai_core::LanguageCallError,
    policy: &GatewayPolicy,
) -> Response<Body> {
    let body = serde_json::to_vec(&json!({
        "error": {
            "code": "language_call_failed",
            "message": "model language request failed",
        },
        "partial": error.partial(),
    }));
    match body {
        Ok(body) if body.len() <= policy.limits().json_response_bytes() => {
            build_json_response(StatusCode::BAD_GATEWAY, body, None, policy)
        }
        _ => error_response(
            StatusCode::BAD_GATEWAY,
            "language_call_failed",
            "model language request failed",
            None,
            policy,
        ),
    }
}

pub(crate) fn error_response(
    status: StatusCode,
    code: &'static str,
    message: &str,
    trust: Option<&ServerTrustContext>,
    policy: &GatewayPolicy,
) -> Response<Body> {
    let body = serde_json::to_vec(&json!({
        "error": {
            "code": code,
            "message": message,
        }
    }))
    .unwrap_or_else(|_| {
        b"{\"error\":{\"code\":\"serialization_failed\",\"message\":\"server response serialization failed\"}}"
            .to_vec()
    });
    build_json_response(status, body, trust, policy)
}

pub(crate) fn apply_gateway_headers(
    response: &mut Response<Body>,
    trust: Option<&ServerTrustContext>,
    policy: &GatewayPolicy,
) {
    response.headers_mut().insert(
        CACHE_CONTROL,
        HeaderValue::from_static("no-store, no-cache, must-revalidate"),
    );
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );

    let headers = policy.headers();
    if headers.emits_route()
        && headers.allows(ROUTE_HEADER)
        && let Some(trust) = trust
    {
        insert_header(response, ROUTE_HEADER, trust.route().as_str());
    }
    if headers.emits_projection_policy() && headers.allows(PROJECTION_POLICY_HEADER) {
        insert_header(
            response,
            PROJECTION_POLICY_HEADER,
            policy.loss_policy().as_str(),
        );
    }
}

fn event_response(
    status: StatusCode,
    event: GatewayEvent,
    trust: Option<&ServerTrustContext>,
    policy: &GatewayPolicy,
) -> Response<Body> {
    match serde_json::to_vec(&event) {
        Ok(body) if body.len() <= policy.limits().json_response_bytes() => {
            build_json_response(status, body, trust, policy)
        }
        Ok(_) => error_response(
            StatusCode::BAD_GATEWAY,
            "response_body_too_large",
            "server response exceeded its configured byte limit",
            trust,
            policy,
        ),
        Err(_) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "response_serialization_failed",
            "server response serialization failed",
            trust,
            policy,
        ),
    }
}

fn projection_error_response(
    error: &GatewayProjectionError,
    trust: Option<&ServerTrustContext>,
    policy: &GatewayPolicy,
) -> Response<Body> {
    let (status, message) = match error {
        GatewayProjectionError::TrustRouteMismatch => (
            StatusCode::FORBIDDEN,
            "authenticated request is not authorized for this route",
        ),
        _ => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "server projection rejected non-portable response data",
        ),
    };
    error_response(status, error.code(), message, trust, policy)
}

fn build_json_response(
    status: StatusCode,
    body: Vec<u8>,
    trust: Option<&ServerTrustContext>,
    policy: &GatewayPolicy,
) -> Response<Body> {
    let mut response = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json; charset=utf-8")
        .body(Body::from(body))
        .unwrap_or_else(|_| Response::new(Body::from("internal server error")));
    apply_gateway_headers(&mut response, trust, policy);
    response
}

fn insert_header(response: &mut Response<Body>, name: &'static str, value: &str) {
    let name = HeaderName::from_static(name);
    let Ok(value) = HeaderValue::from_str(value) else {
        return;
    };
    response.headers_mut().insert(name, value);
}

fn run_terminal_status(terminal: &RunTerminal) -> StatusCode {
    match terminal {
        RunTerminal::Completed { .. } | RunTerminal::Stopped { .. } => StatusCode::OK,
        RunTerminal::Suspended { .. } => StatusCode::ACCEPTED,
        RunTerminal::BudgetExceeded { .. } => StatusCode::TOO_MANY_REQUESTS,
        RunTerminal::TimedOut { .. } => StatusCode::GATEWAY_TIMEOUT,
        RunTerminal::Indeterminate { .. } | RunTerminal::ResumeConflict { .. } => {
            StatusCode::CONFLICT
        }
        RunTerminal::HistoryProjectionRejected { .. } => StatusCode::UNPROCESSABLE_ENTITY,
        RunTerminal::Failed { .. } => StatusCode::BAD_GATEWAY,
        RunTerminal::Cancelled { .. } => StatusCode::REQUEST_TIMEOUT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use siumai_core::{LanguageCompletionReason, LanguageTermination, RouteId, Usage};
    use siumai_runtime::approval::TrustIdentity;

    use super::*;
    use crate::{GatewayHeaderPolicy, GatewayLimits};

    #[test]
    fn language_response_sets_safe_headers() {
        let response = LanguageResponse::new(
            LanguageTermination::Completed(LanguageCompletionReason::Stop),
            Vec::new(),
            Usage::default(),
        )
        .unwrap();
        let policy = GatewayPolicy::default()
            .with_header_policy(GatewayHeaderPolicy::default().with_projection_policy_header(true));

        let response = language_response(response, &policy);

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[PROJECTION_POLICY_HEADER],
            policy.loss_policy().as_str()
        );
        assert_eq!(
            response.headers()[CACHE_CONTROL],
            "no-store, no-cache, must-revalidate"
        );
    }

    #[test]
    fn oversized_json_projection_is_replaced_by_bounded_error() {
        let policy = GatewayPolicy::default().with_limits(
            GatewayLimits::new(
                64,
                64,
                crate::MIN_SERVER_RESPONSE_LIMIT_BYTES,
                crate::MIN_SERVER_RESPONSE_LIMIT_BYTES,
            )
            .unwrap(),
        );
        let response = LanguageResponse::new(
            LanguageTermination::Completed(LanguageCompletionReason::Stop),
            vec![siumai_core::ContentPart::Text {
                text: "x".repeat(crate::MIN_SERVER_RESPONSE_LIMIT_BYTES * 2),
            }],
            Usage::default(),
        )
        .unwrap();

        let response = language_response(response, &policy);

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn route_header_comes_only_from_host_trust_context() {
        let trust = ServerTrustContext::new(
            TrustIdentity::new("issuer", "audience", "subject", "tenant").unwrap(),
            RouteId::new("trusted-route").unwrap(),
        );
        let policy = GatewayPolicy::default()
            .with_header_policy(GatewayHeaderPolicy::default().with_route_header(true));
        let response = error_response(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "bad request",
            Some(&trust),
            &policy,
        );

        assert_eq!(response.headers()[ROUTE_HEADER], "trusted-route");
    }
}
