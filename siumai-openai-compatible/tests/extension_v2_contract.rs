use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use http::header::{HeaderName, HeaderValue};
use serde_json::{Value, json};
use siumai_core::{
    CallOptions, Cancellation, Error, ErrorContext, ErrorKind, LanguageCallError,
    LanguageStreamEvent, PartialLanguageOutput, PartialLanguageOutputPart, Usage, Warning,
    WarningKind,
};
use siumai_openai_compatible::extension::v2::{
    DirectDecoder, DirectResponse, ExecutionContext, PreparedCall, PreparedJsonBody, SseStream,
    SseStreamDecoder, StreamResponseContext, execute_direct, execute_sse,
};
use siumai_transport::{
    EndpointConfig, ProviderTransport, ReplaySafety, RequestBuildError, RequestHeaders,
    RequestTarget, TransportLimits,
};

#[derive(Debug, PartialEq)]
struct NativeDirectOutput {
    body: Value,
    response_id: String,
    attempts: u8,
    warning_count: usize,
}

struct NativeDirectDecoder;

impl DirectDecoder for NativeDirectDecoder {
    type Output = NativeDirectOutput;

    fn decode(self, response: DirectResponse<'_>) -> Result<Self::Output, Error> {
        let response_id = response
            .headers()
            .get(&HeaderName::from_static("x-native-response-id"))
            .and_then(|value| value.to_str().ok())
            .unwrap()
            .to_string();
        Ok(NativeDirectOutput {
            body: serde_json::from_slice(response.body()).unwrap(),
            response_id,
            attempts: response.attempts(),
            warning_count: response.warnings().len(),
        })
    }
}

struct PartialFailureDecoder;

impl DirectDecoder for PartialFailureDecoder {
    type Output = Result<(), LanguageCallError>;

    fn decode(self, _response: DirectResponse<'_>) -> Result<Self::Output, Error> {
        let partial = PartialLanguageOutput::new(
            vec![PartialLanguageOutputPart::Text {
                text: "bounded partial".to_string(),
            }],
            Usage::default(),
        )
        .unwrap();
        Ok(Err(LanguageCallError::new(
            Error::new(ErrorKind::Provider, "provider resource failed"),
            Some(partial),
        )))
    }
}

#[derive(Debug, PartialEq)]
struct NativeEvent {
    raw: Value,
    terminal: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StartObservation {
    status: u16,
    attempts: u8,
    warning_count: usize,
    response_id: Option<String>,
}

struct NativeSseDecoder {
    observation: Arc<Mutex<Option<StartObservation>>>,
}

impl SseStreamDecoder for NativeSseDecoder {
    type Event = NativeEvent;

    fn start(&mut self, context: StreamResponseContext) -> Result<(), Error> {
        let response_id = context
            .headers()
            .get(&HeaderName::from_static("x-native-response-id"))
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        *self.observation.lock().unwrap() = Some(StartObservation {
            status: context.status().as_u16(),
            attempts: context.attempts(),
            warning_count: context.warnings().len(),
            response_id,
        });
        Ok(())
    }

    fn decode(&mut self, data: &str) -> Result<Vec<Self::Event>, Error> {
        let raw = serde_json::from_str::<Value>(data).map_err(|source| {
            Error::new(ErrorKind::Protocol, "native fixture received invalid JSON")
                .with_source(source)
        })?;
        let terminal = raw.get("terminal").and_then(Value::as_bool) == Some(true);
        Ok(vec![NativeEvent { raw, terminal }])
    }

    fn is_terminal(&self, event: &Self::Event) -> bool {
        event.terminal
    }

    fn finish(&mut self) -> Result<Vec<Self::Event>, Error> {
        Ok(Vec::new())
    }
}

struct PortableDecoder;

impl SseStreamDecoder for PortableDecoder {
    type Event = LanguageStreamEvent;

    fn decode(&mut self, _data: &str) -> Result<Vec<Self::Event>, Error> {
        Ok(Vec::new())
    }

    fn is_terminal(&self, event: &Self::Event) -> bool {
        event.terminal().is_some()
    }

    fn finish(&mut self) -> Result<Vec<Self::Event>, Error> {
        Err(Error::unexpected_eof())
    }
}

fn transport(server: &mockito::Server) -> ProviderTransport {
    transport_with_limits(server, TransportLimits::default())
}

fn transport_with_limits(server: &mockito::Server, limits: TransportLimits) -> ProviderTransport {
    ProviderTransport::builder(
        EndpointConfig::local_explicit(format!("{}/v1", server.url())).unwrap(),
    )
    .with_http_transport_settings(
        siumai_transport::ProviderHttpTransportSettings::default()
            .with_limits(limits)
            .unwrap(),
    )
    .build()
    .unwrap()
}

fn context() -> ExecutionContext {
    ExecutionContext::new(
        ErrorContext::default(),
        "fixture request violates the transport contract",
        "fixture provider rejected the request",
        "fixture provider returned invalid SSE",
    )
}

fn prepared_call<D>(
    transport: &ProviderTransport,
    target: &str,
    body: Value,
    decoder: D,
) -> PreparedCall<D> {
    let body = PreparedJsonBody::new(transport, &body).unwrap();
    PreparedCall::new(
        RequestTarget::new(target).unwrap(),
        body,
        ReplaySafety::Never,
        decoder,
        context(),
    )
}

#[tokio::test]
async fn external_direct_decoder_preserves_native_output_and_partial_failure_carrier() {
    let mut server = mockito::Server::new_async().await;
    let native = server
        .mock("POST", "/v1/native")
        .match_header("accept", "application/json")
        .match_header("x-provider-beta", "enabled")
        .match_body(mockito::Matcher::Regex(
            r#"\"future_request_field\":true"#.to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_header("x-native-response-id", "native-1")
        .with_body(r#"{"known":"ok","future":{"nested":true}}"#)
        .expect(1)
        .create_async()
        .await;
    let headers = RequestHeaders::new()
        .try_insert(
            HeaderName::from_static("x-provider-beta"),
            HeaderValue::from_static("enabled"),
        )
        .unwrap();
    let transport = transport(&server);
    let call = prepared_call(
        &transport,
        "native",
        json!({"future_request_field": true}),
        NativeDirectDecoder,
    )
    .with_headers(headers)
    .with_warnings(vec![Warning::new(
        WarningKind::IgnoredOption,
        "fixture warning",
    )]);

    let output = execute_direct(&transport, call, CallOptions::default())
        .await
        .unwrap();
    assert_eq!(output.response_id, "native-1");
    assert_eq!(output.attempts, 1);
    assert_eq!(output.warning_count, 1);
    assert_eq!(output.body["future"]["nested"], true);
    native.assert_async().await;

    let partial = server
        .mock("POST", "/v1/partial")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect(1)
        .create_async()
        .await;
    let outcome = execute_direct(
        &transport,
        prepared_call(&transport, "partial", json!({}), PartialFailureDecoder),
        CallOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(outcome.unwrap_err().partial().unwrap().content().len(), 1);
    partial.assert_async().await;
}

#[tokio::test]
async fn external_sse_decoder_preserves_custom_native_events_and_child_cancellation() {
    let mut server = mockito::Server::new_async().await;
    let stream_mock = server
        .mock("POST", "/v1/native-stream")
        .match_header("accept", "text/event-stream")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_header("x-native-response-id", "stream-1")
        .with_body(concat!(
            "data: {\"kind\":\"delta\",\"future\":{\"nested\":true}}\n\n",
            "data: {\"kind\":\"done\",\"terminal\":true,\"unknown\":7}\n\n",
            "data: [DONE]\n\n",
        ))
        .expect(1)
        .create_async()
        .await;
    let observation = Arc::new(Mutex::new(None));
    let parent = Cancellation::new();
    let transport = transport(&server);
    let call = prepared_call(
        &transport,
        "native-stream",
        json!({"stream": true}),
        NativeSseDecoder {
            observation: observation.clone(),
        },
    )
    .with_warnings(vec![Warning::new(
        WarningKind::IgnoredOption,
        "fixture warning",
    )]);
    let events = execute_sse(
        &transport,
        call,
        CallOptions::default().with_cancellation(parent.clone()),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(events.len(), 2);
    assert_eq!(events[0].as_ref().unwrap().raw["future"]["nested"], true);
    assert_eq!(events[1].as_ref().unwrap().raw["unknown"], 7);
    assert!(events[1].as_ref().unwrap().terminal);
    assert_eq!(
        observation.lock().unwrap().clone().unwrap(),
        StartObservation {
            status: 200,
            attempts: 1,
            warning_count: 1,
            response_id: Some("stream-1".to_string()),
        }
    );
    assert!(!parent.is_cancelled());
    stream_mock.assert_async().await;
}

#[tokio::test]
async fn cancelled_parent_fails_stream_setup_before_network_submission() {
    let mut server = mockito::Server::new_async().await;
    let untouched = server
        .mock("POST", "/v1/cancelled-stream")
        .with_status(200)
        .expect(0)
        .create_async()
        .await;
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let transport = transport(&server);
    let error = execute_sse(
        &transport,
        prepared_call(
            &transport,
            "cancelled-stream",
            json!({}),
            NativeSseDecoder {
                observation: Arc::new(Mutex::new(None)),
            },
        ),
        CallOptions::default().with_cancellation(cancellation),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Cancelled);
    untouched.assert_async().await;
}

#[tokio::test]
async fn non_success_response_is_classified_bounded_and_redacted() {
    let mut server = mockito::Server::new_async().await;
    let body = format!(
        "{{\"error\":{{\"code\":\"rate_limit_exceeded\",\"type\":\"rate_limit_error\",\"param\":\"input\"}},\"secret\":\"{}\"}}",
        "body-secret-canary".repeat(8_000),
    );
    let rejected = server
        .mock("POST", "/v1/rejected")
        .with_status(429)
        .with_header("x-private-canary", "header-secret-canary")
        .with_body(body)
        .expect(1)
        .create_async()
        .await;
    let transport = transport(&server);
    let error = execute_direct(
        &transport,
        prepared_call(&transport, "rejected", json!({}), NativeDirectDecoder),
        CallOptions::default(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert_eq!(error.diagnostics().unwrap().status(), Some(429));
    assert!(error.diagnostics().unwrap().body_truncated());
    let debug = format!("{error:?}");
    let display = error.to_string();
    for secret in ["body-secret-canary", "header-secret-canary"] {
        assert!(!debug.contains(secret));
        assert!(!display.contains(secret));
    }
    let (headers, body) = error.sensitive_response().unwrap().expose();
    assert_eq!(headers["x-private-canary"], "header-secret-canary");
    assert_eq!(body.len(), 64 * 1024);
    rejected.assert_async().await;
}

#[tokio::test]
async fn stream_rejection_is_bounded_and_decoder_is_not_started() {
    let mut server = mockito::Server::new_async().await;
    let body = format!(
        "{{\"error\":{{\"code\":\"service_unavailable\"}},\"secret\":\"{}\"}}",
        "stream-body-secret-canary".repeat(8_000),
    );
    let rejected = server
        .mock("POST", "/v1/rejected-stream")
        .with_status(503)
        .with_header("x-private-canary", "stream-header-secret-canary")
        .with_chunked_body(move |writer| writer.write_all(body.as_bytes()))
        .expect(1)
        .create_async()
        .await;
    let observation = Arc::new(Mutex::new(None));
    let transport = transport(&server);
    let error = execute_sse(
        &transport,
        prepared_call(
            &transport,
            "rejected-stream",
            json!({}),
            NativeSseDecoder {
                observation: observation.clone(),
            },
        ),
        CallOptions::default(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Unavailable);
    assert!(error.diagnostics().unwrap().body_truncated());
    assert!(observation.lock().unwrap().is_none());
    let (_, body) = error.sensitive_response().unwrap().expose();
    assert_eq!(body.len(), 64 * 1024);
    let debug = format!("{error:?}");
    assert!(!debug.contains("stream-body-secret-canary"));
    assert!(!debug.contains("stream-header-secret-canary"));
    rejected.assert_async().await;
}

#[tokio::test]
async fn oversized_json_fails_before_network_submission() {
    let mut server = mockito::Server::new_async().await;
    let untouched = server
        .mock("POST", "/v1/oversized")
        .with_status(200)
        .expect(0)
        .create_async()
        .await;
    let limits = TransportLimits {
        max_request_bytes: 32,
        ..TransportLimits::default()
    };
    let transport = transport_with_limits(&server, limits);
    let error = PreparedJsonBody::new(&transport, &json!({"input": "x".repeat(128)})).unwrap_err();

    assert!(matches!(
        error,
        RequestBuildError::BodyTooLarge { maximum: 32 }
    ));
    untouched.assert_async().await;
}

#[test]
fn public_contract_accepts_portable_and_named_native_stream_carriers() {
    fn assert_decoder<D: SseStreamDecoder<Event = LanguageStreamEvent>>() {}
    fn assert_stream_type(_: Option<SseStream<NativeEvent>>) {}

    assert_decoder::<PortableDecoder>();
    assert_stream_type(None);
}
