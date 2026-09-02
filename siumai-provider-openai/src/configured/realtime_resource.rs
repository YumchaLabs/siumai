//! Provider-native OpenAI Realtime client-secret lifecycle operations.

use std::fmt;
use std::sync::Arc;

use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorContext, ErrorKind};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget, TransportResponse,
};

use super::http_error;
use super::mode::OpenAiApiMode;
use super::provider::OpenAiRuntime;
use super::realtime::{
    OpenAiRealtimeClientSecretRequest, OpenAiRealtimeClientSecretResource, OpenAiRealtimeRoute,
};

const MIN_CLIENT_SECRET_EXPIRY_SECONDS: u64 = 10;
const MAX_CLIENT_SECRET_EXPIRY_SECONDS: u64 = 7_200;

/// Provider-native creator for short-lived OpenAI Realtime client secrets.
#[derive(Clone)]
pub struct OpenAiRealtimeResource {
    runtime: Arc<OpenAiRuntime>,
}

impl OpenAiRealtimeResource {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>) -> Self {
        Self { runtime }
    }

    /// Create one short-lived credential for a browser or mobile Realtime session.
    ///
    /// This mutation is never replayed after dispatch. Authentication, bounded
    /// response buffering, cancellation, and deadlines are all owned by the
    /// configured provider's shared transport.
    pub async fn create_client_secret(
        &self,
        request: OpenAiRealtimeClientSecretRequest,
        call_options: CallOptions,
    ) -> Result<OpenAiRealtimeClientSecretResource, Error> {
        let route = request.route();
        let plan = client_secret_request_plan(route, &request)?;
        let response = self.execute(plan, call_options).await?;
        let body = std::str::from_utf8(response.body()).map_err(|source| {
            self.contextualize(
                Error::new(
                    ErrorKind::Protocol,
                    "OpenAI returned a non-UTF-8 Realtime client-secret resource",
                )
                .with_source(source),
            )
        })?;
        OpenAiRealtimeClientSecretResource::decode(route, body).map_err(|source| {
            self.contextualize(
                Error::new(
                    ErrorKind::Protocol,
                    "OpenAI returned a malformed Realtime client-secret resource",
                )
                .with_source(source),
            )
        })
    }

    async fn execute(
        &self,
        plan: RequestPlan,
        call_options: CallOptions,
    ) -> Result<TransportResponse, Error> {
        let response = self
            .runtime
            .transport
            .execute(plan, call_options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if response.status().is_success() {
            Ok(response)
        } else {
            Err(self.contextualize(http_error::response_error(
                "OpenAI rejected the Realtime client-secret request",
                response,
            )))
        }
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: None,
            provider: Some(
                self.runtime
                    .scope(OpenAiApiMode::Responses)
                    .provider_id()
                    .clone(),
            ),
            route: None,
            model: None,
        })
    }
}

impl fmt::Debug for OpenAiRealtimeResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRealtimeResource")
            .field(
                "provider",
                self.runtime.scope(OpenAiApiMode::Responses).provider_id(),
            )
            .field("transport", &"shared")
            .finish()
    }
}

fn client_secret_request_plan(
    route: OpenAiRealtimeRoute,
    request: &OpenAiRealtimeClientSecretRequest,
) -> Result<RequestPlan, Error> {
    let body = serde_json::to_value(request).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "OpenAI Realtime client-secret request could not be serialized",
        )
        .with_source(source)
    })?;
    validate_client_secret_expiry(&body)?;
    let target =
        RequestTarget::new(client_secret_target(route)).map_err(realtime_request_build_error)?;
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
        .map_err(realtime_request_build_error)?;
    RequestPlan::new(Method::POST, target)
        .with_headers(headers)
        .with_body(RequestBody::json(&body).map_err(realtime_request_build_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(realtime_request_build_error)
}

const fn client_secret_target(route: OpenAiRealtimeRoute) -> &'static str {
    match route {
        OpenAiRealtimeRoute::Conversation => "realtime/client_secrets",
        OpenAiRealtimeRoute::Translation => "realtime/translations/client_secrets",
    }
}

fn validate_client_secret_expiry(body: &Value) -> Result<(), Error> {
    let Some(expires_after) = body.get("expires_after") else {
        return Ok(());
    };
    let seconds = expires_after
        .get("seconds")
        .and_then(Value::as_u64)
        .ok_or_else(invalid_client_secret_expiry)?;
    if !(MIN_CLIENT_SECRET_EXPIRY_SECONDS..=MAX_CLIENT_SECRET_EXPIRY_SECONDS).contains(&seconds) {
        return Err(invalid_client_secret_expiry());
    }
    Ok(())
}

fn invalid_client_secret_expiry() -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "OpenAI Realtime client-secret expiry must be between 10 and 7200 seconds",
    )
}

fn realtime_request_build_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "OpenAI Realtime client-secret request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use http::StatusCode;
    use serde_json::json;
    use siumai_core::{ReplayDomain, ReplayDomainId};
    use siumai_transport::{EndpointConfig, RetryPolicy};
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::configured::{OpenAiCredential, OpenAiProvider};

    fn retry_policy(maximum_attempts: u8) -> RetryPolicy {
        RetryPolicy::new(maximum_attempts)
            .unwrap()
            .with_backoff(Duration::ZERO, Duration::ZERO)
            .with_jitter(false)
    }

    fn resource(
        server: &MockServer,
        credential: OpenAiCredential,
        maximum_attempts: u8,
    ) -> OpenAiRealtimeResource {
        let provider = OpenAiProvider::builder(credential)
            .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("realtime-resource-test").unwrap(),
            ))
            .with_http_transport_settings(
                siumai_transport::ProviderHttpTransportSettings::default()
                    .with_retry_policy(retry_policy(maximum_attempts)),
            )
            .build()
            .unwrap();
        OpenAiRealtimeResource::new(provider.runtime.clone())
    }

    #[tokio::test]
    async fn conversation_secret_uses_relative_route_and_shared_transport_auth_once() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/realtime/client_secrets"))
            .and(header("accept", "application/json"))
            .and(header("authorization", "Bearer canary-api-key"))
            .and(body_json(json!({
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime-2.1"
                }
            })))
            .respond_with(
                ResponseTemplate::new(StatusCode::OK.as_u16()).set_body_json(json!({
                    "value": "ek_conversation",
                    "expires_at": 1_800_000_000_i64,
                    "session": {
                        "type": "realtime",
                        "model": "gpt-realtime-2.1"
                    }
                })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let secret = resource(&server, OpenAiCredential::api_key("canary-api-key"), 3)
            .create_client_secret(
                OpenAiRealtimeClientSecretRequest::conversation("gpt-realtime-2.1").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(secret.route(), OpenAiRealtimeRoute::Conversation);
        assert_eq!(secret.expose_value(), "ek_conversation");
        assert_eq!(secret.model(), Some("gpt-realtime-2.1"));
        assert_eq!(secret.expires_at_unix(), Some(1_800_000_000));
    }

    #[tokio::test]
    async fn translation_secret_uses_translation_route_and_typed_decoder() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/realtime/translations/client_secrets"))
            .and(body_json(json!({
                "session": {
                    "model": "gpt-realtime-translate"
                }
            })))
            .respond_with(
                ResponseTemplate::new(StatusCode::OK.as_u16()).set_body_json(json!({
                    "value": "ek_translation",
                    "expires_at": 1_800_000_001_i64,
                    "session": {
                        "type": "translation",
                        "model": "gpt-realtime-translate"
                    }
                })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let secret = resource(&server, OpenAiCredential::unauthenticated(), 3)
            .create_client_secret(
                OpenAiRealtimeClientSecretRequest::translation("gpt-realtime-translate").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(secret.route(), OpenAiRealtimeRoute::Translation);
        assert_eq!(secret.expose_value(), "ek_translation");
        assert_eq!(secret.model(), Some("gpt-realtime-translate"));
    }

    #[tokio::test]
    async fn server_error_is_contextualized_and_never_replayed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/realtime/client_secrets"))
            .respond_with(
                ResponseTemplate::new(StatusCode::INTERNAL_SERVER_ERROR.as_u16()).set_body_json(
                    json!({
                        "error": {
                            "type": "server_error",
                            "code": "realtime_unavailable"
                        }
                    }),
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let error = resource(&server, OpenAiCredential::unauthenticated(), 5)
            .create_client_secret(
                OpenAiRealtimeClientSecretRequest::conversation("gpt-realtime-2.1").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Unavailable);
        assert_eq!(
            error
                .context()
                .provider
                .as_ref()
                .map(|provider| provider.as_str()),
            Some("openai")
        );
        assert_eq!(
            error
                .diagnostics()
                .and_then(|diagnostics| diagnostics.status()),
            Some(StatusCode::INTERNAL_SERVER_ERROR.as_u16())
        );
    }

    #[tokio::test]
    async fn invalid_secret_and_non_utf8_response_are_protocol_errors() {
        let invalid_secret_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/realtime/client_secrets"))
            .respond_with(
                ResponseTemplate::new(StatusCode::OK.as_u16()).set_body_json(json!({
                    "value": "",
                    "session": {
                        "type": "realtime",
                        "model": "gpt-realtime-2.1"
                    }
                })),
            )
            .expect(1)
            .mount(&invalid_secret_server)
            .await;

        let invalid_secret = resource(
            &invalid_secret_server,
            OpenAiCredential::unauthenticated(),
            3,
        )
        .create_client_secret(
            OpenAiRealtimeClientSecretRequest::conversation("gpt-realtime-2.1").unwrap(),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
        assert_eq!(invalid_secret.kind(), ErrorKind::Protocol);

        let invalid_utf8_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/realtime/translations/client_secrets"))
            .respond_with(
                ResponseTemplate::new(StatusCode::OK.as_u16()).set_body_bytes([0xff, 0xfe]),
            )
            .expect(1)
            .mount(&invalid_utf8_server)
            .await;

        let invalid_utf8 = resource(&invalid_utf8_server, OpenAiCredential::unauthenticated(), 3)
            .create_client_secret(
                OpenAiRealtimeClientSecretRequest::translation("gpt-realtime-translate").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(invalid_utf8.kind(), ErrorKind::Protocol);
    }

    #[test]
    fn client_secret_expiry_is_limited_to_the_official_range() {
        for seconds in [10, 60, 7_200] {
            assert!(
                validate_client_secret_expiry(&json!({
                    "expires_after": {
                        "anchor": "created_at",
                        "seconds": seconds
                    }
                }))
                .is_ok()
            );
        }
        for seconds in [0, 9, 7_201, u64::MAX] {
            let error = validate_client_secret_expiry(&json!({
                "expires_after": {
                    "anchor": "created_at",
                    "seconds": seconds
                }
            }))
            .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
        }
        assert!(
            validate_client_secret_expiry(&json!({
                "expires_after": {
                    "anchor": "created_at"
                }
            }))
            .is_err()
        );
    }
}
