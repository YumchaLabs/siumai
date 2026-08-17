use std::collections::VecDeque;
use std::net::SocketAddr;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use http::header::{AUTHORIZATION, HeaderName, HeaderValue};
use http::{Method, StatusCode};
use siumai_core::{CallOptions, Cancellation, Error, ErrorKind};
use siumai_transport::{
    AttemptLoopOutcome, AuthApplier, AuthContext, AuthRefresh, CredentialPatch, CredentialRevision,
    EndpointConfig, EndpointError, EndpointPolicy, HttpTransportRoute, IdempotencyHeader,
    MultipartBody, MultipartPart, ProviderHttpTransportSettings, ProviderTransport,
    ProxyBasicCredential, ProxyEndpoint, ReplaySafety, RequestBody, RequestBuildError,
    RequestHeaders, RequestPlan, RequestTarget, Resolver, ResourceDownloadOptions,
    ResourceDownloader, ResourceUrl, RetryClassifier, RetryLimit, RetryPolicy, RetryReason,
    TransportConfigError, TransportEvent, TransportLimits, TransportObserver, WebSocketEndpoint,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Clone)]
enum ServerAction {
    DropAfterRead,
    HoldAfterRead,
    Respond {
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    },
    RespondThenHold {
        status: u16,
        prefix: Vec<u8>,
        content_length: usize,
    },
}

#[derive(Debug, Clone)]
struct RecordedRequest {
    head: String,
    body: Vec<u8>,
}

impl RecordedRequest {
    fn header(&self, expected: &str) -> Option<&str> {
        self.head.lines().skip(1).find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case(expected).then(|| value.trim())
        })
    }
}

struct TestServer {
    address: SocketAddr,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    task: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn spawn(actions: Vec<ServerAction>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let task = tokio::spawn(async move {
            let mut actions = VecDeque::from(actions);
            let mut handlers = Vec::new();
            while let Some(action) = actions.pop_front() {
                let Ok(Ok((stream, _))) =
                    tokio::time::timeout(Duration::from_secs(3), listener.accept()).await
                else {
                    break;
                };
                let recorded = recorded.clone();
                handlers.push(tokio::spawn(async move {
                    handle_connection(stream, action, recorded).await;
                }));
            }
            for handler in handlers {
                let _ = handler.await;
            }
        });
        Self {
            address,
            requests,
            task,
        }
    }

    fn endpoint(&self) -> EndpointConfig {
        EndpointConfig::local_explicit(format!("http://{}/v1", self.address)).unwrap()
    }

    fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    action: ServerAction,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
) {
    let request = read_request(&mut stream).await;
    requests.lock().unwrap().push(request);
    match action {
        ServerAction::DropAfterRead => {}
        ServerAction::HoldAfterRead => {
            let mut byte = [0_u8; 1];
            while stream.read(&mut byte).await.unwrap_or(0) != 0 {}
        }
        ServerAction::Respond {
            status,
            headers,
            body,
        } => {
            let mut response = format!(
                "HTTP/1.1 {status} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
                reason(status),
                body.len()
            );
            for (name, value) in headers {
                response.push_str(&format!("{name}: {value}\r\n"));
            }
            response.push_str("\r\n");
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.write_all(&body).await.unwrap();
        }
        ServerAction::RespondThenHold {
            status,
            prefix,
            content_length,
        } => {
            let response = format!(
                "HTTP/1.1 {status} {}\r\nContent-Length: {content_length}\r\n\r\n",
                reason(status)
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.write_all(&prefix).await.unwrap();
            let mut byte = [0_u8; 1];
            while stream.read(&mut byte).await.unwrap_or(0) != 0 {}
        }
    }
}

async fn read_request(stream: &mut TcpStream) -> RecordedRequest {
    let mut bytes = Vec::new();
    let header_end = loop {
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await.unwrap();
        assert!(read > 0, "connection ended before request headers");
        bytes.extend_from_slice(&chunk[..read]);
    };
    let head = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
    let content_length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    while bytes.len() - header_end < content_length {
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await.unwrap();
        assert!(read > 0, "connection ended before request body");
        bytes.extend_from_slice(&chunk[..read]);
    }
    RecordedRequest {
        head,
        body: bytes[header_end..header_end + content_length].to_vec(),
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        302 => "Found",
        401 => "Unauthorized",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "Test",
    }
}

#[derive(Default)]
struct RecordingObserver(Mutex<Vec<TransportEvent>>);

impl RecordingObserver {
    fn attempts(&self) -> usize {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(event, TransportEvent::AttemptStarted { .. }))
            .count()
    }

    fn events(&self) -> Vec<TransportEvent> {
        self.0.lock().unwrap().clone()
    }
}

impl TransportObserver for RecordingObserver {
    fn observe(&self, event: &TransportEvent) {
        self.0.lock().unwrap().push(event.clone());
    }
}

#[derive(Default)]
struct ScopedEventSink(Mutex<Vec<(&'static str, TransportEvent)>>);

struct ScopedObserver {
    scope: &'static str,
    sink: Arc<ScopedEventSink>,
}

impl TransportObserver for ScopedObserver {
    fn observe(&self, event: &TransportEvent) {
        self.sink
            .0
            .lock()
            .unwrap()
            .push((self.scope, event.clone()));
    }
}

fn retry_policy(maximum_attempts: u8) -> RetryPolicy {
    RetryPolicy::new(maximum_attempts)
        .unwrap()
        .with_backoff(Duration::ZERO, Duration::ZERO)
}

fn transport(endpoint: EndpointConfig, observer: Arc<dyn TransportObserver>) -> ProviderTransport {
    ProviderTransport::builder(endpoint)
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default()
                .with_retry_policy(retry_policy(3))
                .with_observer(observer),
        )
        .build()
        .unwrap()
}

fn json_post(replay: ReplaySafety) -> RequestPlan {
    RequestPlan::new(Method::POST, RequestTarget::new("responses").unwrap())
        .with_body(RequestBody::json(&serde_json::json!({ "input": "hello" })).unwrap())
        .with_replay_safety(replay)
        .unwrap()
}

#[test]
fn provider_http_settings_preserve_direct_defaults_and_validate_before_build() {
    let defaults = ProviderHttpTransportSettings::default();
    assert_eq!(defaults.limits(), &TransportLimits::default());
    assert_eq!(defaults.retry_policy(), RetryPolicy::default());
    assert_eq!(defaults.connect_timeout(), Duration::from_secs(10));
    assert_eq!(defaults.call_timeout(), Duration::from_secs(15 * 60));
    assert_eq!(defaults.read_timeout(), Duration::from_secs(5 * 60));
    let debug = format!("{defaults:?}");
    assert!(debug.contains("Direct"));
    assert!(debug.contains("observer_configured: false"));

    let zero_limit = defaults
        .clone()
        .with_limits(TransportLimits {
            max_request_bytes: 0,
            ..TransportLimits::default()
        })
        .unwrap_err();
    assert_eq!(
        zero_limit,
        TransportConfigError::ZeroLimit {
            name: "max_request_bytes"
        }
    );
    assert_eq!(
        defaults
            .clone()
            .with_connect_timeout(Duration::ZERO)
            .unwrap_err(),
        TransportConfigError::ZeroTimeout {
            name: "connect_timeout"
        }
    );
    assert_eq!(
        defaults.with_call_timeout(Duration::MAX).unwrap_err(),
        TransportConfigError::TimeoutTooLarge {
            name: "call_timeout"
        }
    );
}

#[test]
fn trusted_connect_route_rejects_unsafe_proxy_shapes_and_credentials() {
    assert_eq!(
        ProxyEndpoint::new(
            "https://proxy.example.test/tenant",
            EndpointPolicy::PublicCustom,
        )
        .unwrap_err(),
        EndpointError::ProxyOriginMustBeRoot
    );
    assert_eq!(
        ProxyEndpoint::new("http://proxy.example.test", EndpointPolicy::PublicCustom,).unwrap_err(),
        EndpointError::SchemeNotAllowed
    );
    for candidate in [
        "https://proxy.example.test?tenant=secret",
        "https://proxy.example.test#fragment",
        "https://user:secret@proxy.example.test",
    ] {
        assert!(ProxyEndpoint::https(candidate).is_err());
    }

    let local_proxy = ProxyEndpoint::local_explicit("http://127.0.0.1:3128").unwrap();
    let credential = ProxyBasicCredential::new("proxy-user", "proxy-secret").unwrap();
    assert_eq!(
        HttpTransportRoute::trusted_connect(local_proxy)
            .with_basic_auth(credential)
            .unwrap_err(),
        TransportConfigError::ProxyCredentialsRequireTls
    );
    assert!(ProxyBasicCredential::new("", "proxy-secret").is_err());
    assert!(ProxyBasicCredential::new("proxy-user", "").is_err());
    assert!(ProxyBasicCredential::new("proxy-user\n", "proxy-secret").is_err());
    assert!(ProxyBasicCredential::new("proxy:user", "proxy-secret").is_err());
    assert!(ProxyBasicCredential::new("proxy-user", "proxy-secret\0").is_err());
    assert!(ProxyBasicCredential::new("u".repeat(257), "proxy-secret").is_err());
    assert!(ProxyBasicCredential::new("proxy-user", "p".repeat(257)).is_err());
    let debug = format!(
        "{:?}",
        ProxyBasicCredential::new("proxy-user", "proxy-secret").unwrap()
    );
    assert!(!debug.contains("proxy-user"));
    assert!(!debug.contains("proxy-secret"));

    let secure_route = HttpTransportRoute::trusted_connect(
        ProxyEndpoint::https("https://proxy.example.test").unwrap(),
    )
    .with_basic_auth(ProxyBasicCredential::new("proxy-user", "proxy-secret").unwrap())
    .unwrap();
    let secure_debug = format!("{secure_route:?}");
    assert!(!secure_debug.contains("proxy.example.test"));
    assert!(!secure_debug.contains("proxy-user"));
    assert!(!secure_debug.contains("proxy-secret"));
    ProviderTransport::builder(
        EndpointConfig::public_custom("https://provider.example.test/v1").unwrap(),
    )
    .with_http_transport_settings(
        ProviderHttpTransportSettings::default()
            .with_route(secure_route)
            .unwrap(),
    )
    .build()
    .unwrap();

    let route = HttpTransportRoute::trusted_connect(
        ProxyEndpoint::local_explicit("http://127.0.0.1:3128").unwrap(),
    );
    let settings = ProviderHttpTransportSettings::default()
        .with_route(route)
        .unwrap();
    for provider in [
        EndpointConfig::local_explicit("http://127.0.0.1:8080/v1").unwrap(),
        EndpointConfig::local_explicit("https://127.0.0.1:8443/v1").unwrap(),
    ] {
        assert_eq!(
            ProviderTransport::builder(provider)
                .with_http_transport_settings(settings.clone())
                .build()
                .unwrap_err(),
            TransportConfigError::ProxyDestinationMustBePublicHttps
        );
    }
}

#[tokio::test]
async fn trusted_connect_route_resolves_and_validates_the_proxy_peer() {
    let proxy = TestServer::spawn(vec![ServerAction::DropAfterRead]).await;
    let calls = Arc::new(Mutex::new(Vec::new()));
    let resolver = FixedResolver {
        address: proxy.address,
        calls: calls.clone(),
    };
    let provider = EndpointConfig::public_custom("https://provider.example.test/v1").unwrap();
    let proxy_endpoint = ProxyEndpoint::local_explicit(format!(
        "http://proxy.example.test:{}",
        proxy.address.port()
    ))
    .unwrap();
    let settings = ProviderHttpTransportSettings::default()
        .with_route(HttpTransportRoute::trusted_connect(proxy_endpoint))
        .unwrap();
    let transport = ProviderTransport::builder(provider)
        .with_resolver(Arc::new(resolver))
        .with_auth(Arc::new(SecretAuth))
        .with_http_transport_settings(settings)
        .build()
        .unwrap();
    let error = transport
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap()),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        &[("proxy.example.test".to_owned(), 0)]
    );
    let requests = proxy.requests();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0]
            .head
            .starts_with("CONNECT provider.example.test:443 HTTP/1.1\r\n")
    );
    assert!(requests[0].header("authorization").is_none());
    assert!(!requests[0].head.contains("canary-header-secret"));
    assert!(!requests[0].head.contains("canary-query-secret"));
    for surface in [format!("{error:?}"), error.to_string()] {
        assert!(!surface.contains("proxy.example.test"));
        assert!(!surface.contains("provider.example.test"));
        assert!(!surface.contains("canary-header-secret"));
        assert!(!surface.contains("canary-query-secret"));
    }
}

#[tokio::test]
async fn proxy_connect_failure_cancellation_and_deadline_use_one_attempt() {
    let failed_proxy = TestServer::spawn(vec![ServerAction::Respond {
        status: 407,
        headers: Vec::new(),
        body: Vec::new(),
    }])
    .await;
    let failed_transport = ProviderTransport::builder(
        EndpointConfig::public_custom("https://provider.example.test/v1").unwrap(),
    )
    .with_resolver(Arc::new(FixedResolver {
        address: failed_proxy.address,
        calls: Arc::new(Mutex::new(Vec::new())),
    }))
    .with_http_transport_settings(
        ProviderHttpTransportSettings::default()
            .with_route(HttpTransportRoute::trusted_connect(
                ProxyEndpoint::local_explicit(format!(
                    "http://proxy.example.test:{}",
                    failed_proxy.address.port()
                ))
                .unwrap(),
            ))
            .unwrap(),
    )
    .build()
    .unwrap();
    let error = failed_transport
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap()),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(failed_proxy.requests().len(), 1);

    let holding_proxy = TestServer::spawn(vec![ServerAction::HoldAfterRead]).await;
    let timed_transport = ProviderTransport::builder(
        EndpointConfig::public_custom("https://provider.example.test/v1").unwrap(),
    )
    .with_resolver(Arc::new(FixedResolver {
        address: holding_proxy.address,
        calls: Arc::new(Mutex::new(Vec::new())),
    }))
    .with_http_transport_settings(
        ProviderHttpTransportSettings::default()
            .with_route(HttpTransportRoute::trusted_connect(
                ProxyEndpoint::local_explicit(format!(
                    "http://proxy.example.test:{}",
                    holding_proxy.address.port()
                ))
                .unwrap(),
            ))
            .unwrap(),
    )
    .build()
    .unwrap();
    let error = timed_transport
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap()),
            CallOptions::default()
                .with_deadline(std::time::Instant::now() + Duration::from_millis(50)),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert_eq!(holding_proxy.requests().len(), 1);

    let cancelling_proxy = TestServer::spawn(vec![ServerAction::HoldAfterRead]).await;
    let cancelling_transport = ProviderTransport::builder(
        EndpointConfig::public_custom("https://provider.example.test/v1").unwrap(),
    )
    .with_resolver(Arc::new(FixedResolver {
        address: cancelling_proxy.address,
        calls: Arc::new(Mutex::new(Vec::new())),
    }))
    .with_http_transport_settings(
        ProviderHttpTransportSettings::default()
            .with_route(HttpTransportRoute::trusted_connect(
                ProxyEndpoint::local_explicit(format!(
                    "http://proxy.example.test:{}",
                    cancelling_proxy.address.port()
                ))
                .unwrap(),
            ))
            .unwrap(),
    )
    .build()
    .unwrap();
    let cancellation = Cancellation::new();
    let pending = {
        let cancellation = cancellation.clone();
        tokio::spawn(async move {
            cancelling_transport
                .execute(
                    RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap()),
                    CallOptions::default().with_cancellation(cancellation),
                )
                .await
        })
    };
    tokio::time::timeout(Duration::from_secs(1), async {
        while cancelling_proxy.requests().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the proxy must receive CONNECT before cancellation");
    cancellation.cancel();
    let error = pending.await.unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!(cancelling_proxy.requests().len(), 1);
}

#[test]
fn direct_mode_ignores_common_proxy_environment_variables() {
    const HELPER_ENV: &str = "SIUMAI_DIRECT_PROXY_ENV_HELPER";
    if std::env::var_os(HELPER_ENV).is_some() {
        return;
    }

    let proxy = "http://127.0.0.1:9";
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "direct_mode_proxy_environment_helper",
            "--nocapture",
        ])
        .env(HELPER_ENV, "1")
        .env("HTTP_PROXY", proxy)
        .env("HTTPS_PROXY", proxy)
        .env("ALL_PROXY", proxy)
        .env("http_proxy", proxy)
        .env("https_proxy", proxy)
        .env("all_proxy", proxy)
        .env("NO_PROXY", "")
        .env("no_proxy", "")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child output:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn direct_mode_proxy_environment_helper() {
    if std::env::var_os("SIUMAI_DIRECT_PROXY_ENV_HELPER").is_none() {
        return;
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let server = TestServer::spawn(vec![ServerAction::Respond {
                status: 200,
                headers: Vec::new(),
                body: b"direct".to_vec(),
            }])
            .await;
            let resolver = FixedResolver {
                address: server.address,
                calls: Arc::new(Mutex::new(Vec::new())),
            };
            let endpoint = EndpointConfig::new(
                format!("http://provider.example.test:{}/v1", server.address.port()),
                EndpointPolicy::LocalExplicit(siumai_transport::LocalNetworkGrant::Loopback),
            )
            .unwrap();
            let transport = ProviderTransport::builder(endpoint)
                .with_resolver(Arc::new(resolver))
                .build()
                .unwrap();
            let response = transport
                .execute(
                    RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap()),
                    CallOptions::default(),
                )
                .await
                .unwrap();
            assert_eq!(response.body(), &bytes::Bytes::from_static(b"direct"));
        });
}

#[tokio::test]
async fn buffered_and_streaming_calls_have_precise_attempt_loop_boundaries() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"buffered".to_vec(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"streamed".to_vec(),
        },
    ])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let settings = ProviderHttpTransportSettings::default().with_observer(observer.clone());
    let transport = ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(settings)
        .build()
        .unwrap();
    let plan = RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap());

    transport
        .execute(plan.clone(), CallOptions::default())
        .await
        .unwrap();
    let buffered_events = observer.events();
    assert_eq!(buffered_events.len(), 4);
    let buffered_call_id = buffered_events[0].call_id();
    assert!(
        buffered_events
            .iter()
            .all(|event| event.call_id() == buffered_call_id)
    );
    assert!(matches!(
        buffered_events.as_slice(),
        [
            TransportEvent::AttemptBudgetResolved { .. },
            TransportEvent::AttemptStarted { attempt: 1, .. },
            TransportEvent::ResponseHeadReceived {
                status: StatusCode::OK,
                attempt: 1,
                retry_reason: None,
                server_retry_after: None,
                ..
            },
            TransportEvent::AttemptLoopFinished {
                attempts: 1,
                outcome: AttemptLoopOutcome::ResponseReturned {
                    status: StatusCode::OK
                },
                ..
            },
        ]
    ));

    let response = transport
        .execute_stream(plan, CallOptions::default())
        .await
        .unwrap();
    let stream_boundary_events = observer.events();
    let stream_events = &stream_boundary_events[buffered_events.len()..];
    assert_eq!(stream_events.len(), 4);
    let stream_call_id = stream_events[0].call_id();
    assert_ne!(buffered_call_id, stream_call_id);
    assert!(
        stream_events
            .iter()
            .all(|event| event.call_id() == stream_call_id)
    );
    assert!(matches!(
        stream_events.last(),
        Some(TransportEvent::AttemptLoopFinished {
            attempts: 1,
            outcome: AttemptLoopOutcome::StreamEstablished {
                status: StatusCode::OK
            },
            ..
        })
    ));

    let mut body = response.into_body();
    while let Some(chunk) = body.next().await {
        chunk.unwrap();
    }
    assert_eq!(observer.events(), stream_boundary_events);
}

#[tokio::test]
async fn final_attempt_outcomes_distinguish_failure_cancellation_and_timeout() {
    let failed_server = TestServer::spawn(vec![ServerAction::DropAfterRead]).await;
    let failed_observer = Arc::new(RecordingObserver::default());
    let failed_transport = ProviderTransport::builder(failed_server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_observer(failed_observer.clone()),
        )
        .build()
        .unwrap();
    let failed = failed_transport
        .execute(json_post(ReplaySafety::Never), CallOptions::default())
        .await
        .unwrap_err();
    assert_eq!(failed.kind(), ErrorKind::Transport);
    assert!(matches!(
        failed_observer.events().last(),
        Some(TransportEvent::AttemptLoopFinished {
            attempts: 1,
            outcome: AttemptLoopOutcome::Failed {
                kind: ErrorKind::Transport
            },
            ..
        })
    ));

    let cancelled_server = TestServer::spawn(Vec::new()).await;
    let cancelled_observer = Arc::new(RecordingObserver::default());
    let cancelled_transport = ProviderTransport::builder(cancelled_server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_observer(cancelled_observer.clone()),
        )
        .build()
        .unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let cancelled = cancelled_transport
        .execute(
            json_post(ReplaySafety::Never),
            CallOptions::default().with_cancellation(cancellation),
        )
        .await
        .unwrap_err();
    assert_eq!(cancelled.kind(), ErrorKind::Cancelled);
    assert!(matches!(
        cancelled_observer.events().last(),
        Some(TransportEvent::AttemptLoopFinished {
            attempts: 0,
            outcome: AttemptLoopOutcome::Cancelled,
            ..
        })
    ));

    let timed_out_server = TestServer::spawn(Vec::new()).await;
    let timed_out_observer = Arc::new(RecordingObserver::default());
    let timed_out_transport = ProviderTransport::builder(timed_out_server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_observer(timed_out_observer.clone()),
        )
        .build()
        .unwrap();
    let timed_out = timed_out_transport
        .execute(
            json_post(ReplaySafety::Never),
            CallOptions::default().with_deadline(std::time::Instant::now()),
        )
        .await
        .unwrap_err();
    assert_eq!(timed_out.kind(), ErrorKind::Timeout);
    assert!(matches!(
        timed_out_observer.events().last(),
        Some(TransportEvent::AttemptLoopFinished {
            attempts: 0,
            outcome: AttemptLoopOutcome::TimedOut,
            ..
        })
    ));
}

#[tokio::test]
async fn retry_events_keep_one_call_id_and_report_structural_response_advice() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 500,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"recovered".to_vec(),
        },
    ])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let settings = ProviderHttpTransportSettings::default()
        .with_retry_policy(retry_policy(2))
        .with_observer(observer.clone());
    ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(settings)
        .build()
        .unwrap()
        .execute(
            json_post(ReplaySafety::SemanticallyIdempotent),
            CallOptions::default(),
        )
        .await
        .unwrap();

    let events = observer.events();
    assert_eq!(events.len(), 7);
    let call_id = events[0].call_id();
    assert!(events.iter().all(|event| event.call_id() == call_id));
    assert!(matches!(
        events.as_slice(),
        [
            TransportEvent::AttemptBudgetResolved { .. },
            TransportEvent::AttemptStarted { attempt: 1, .. },
            TransportEvent::ResponseHeadReceived {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                attempt: 1,
                retry_reason: Some(RetryReason::ServerUnavailable),
                server_retry_after: None,
                ..
            },
            TransportEvent::RetryScheduled {
                reason: RetryReason::ServerUnavailable,
                completed_attempts: 1,
                delay,
                ..
            },
            TransportEvent::AttemptStarted { attempt: 2, .. },
            TransportEvent::ResponseHeadReceived {
                status: StatusCode::OK,
                attempt: 2,
                retry_reason: None,
                server_retry_after: None,
                ..
            },
            TransportEvent::AttemptLoopFinished {
                attempts: 2,
                outcome: AttemptLoopOutcome::ResponseReturned {
                    status: StatusCode::OK
                },
                ..
            },
        ] if delay.is_zero()
    ));
}

#[tokio::test]
async fn concurrent_executions_use_distinct_call_ids() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"first".to_vec(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"second".to_vec(),
        },
    ])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let transport = ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_observer(observer.clone()),
        )
        .build()
        .unwrap();
    let first = transport.execute(
        RequestPlan::new(Method::GET, RequestTarget::new("first").unwrap()),
        CallOptions::default(),
    );
    let second = transport.execute(
        RequestPlan::new(Method::GET, RequestTarget::new("second").unwrap()),
        CallOptions::default(),
    );
    let (first, second) = tokio::join!(first, second);
    first.unwrap();
    second.unwrap();

    let events = observer.events();
    let call_ids = events
        .iter()
        .filter_map(|event| match event {
            TransportEvent::AttemptBudgetResolved { call_id, .. } => Some(*call_id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(call_ids.len(), 2);
    assert_ne!(call_ids[0], call_ids[1]);
    assert!(
        events
            .iter()
            .all(|event| call_ids.contains(&event.call_id()))
    );
}

#[tokio::test]
async fn response_body_timeout_finishes_the_buffered_loop_once() {
    let server = TestServer::spawn(vec![ServerAction::RespondThenHold {
        status: 200,
        prefix: b"partial".to_vec(),
        content_length: 100,
    }])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let transport = ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_observer(observer.clone()),
        )
        .build()
        .unwrap();
    let error = transport
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("slow").unwrap()),
            CallOptions::default()
                .with_deadline(std::time::Instant::now() + Duration::from_millis(50)),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);

    let events = observer.events();
    assert!(matches!(
        events.as_slice(),
        [
            TransportEvent::AttemptBudgetResolved { .. },
            TransportEvent::AttemptStarted { attempt: 1, .. },
            TransportEvent::ResponseHeadReceived {
                status: StatusCode::OK,
                attempt: 1,
                ..
            },
            TransportEvent::AttemptLoopFinished {
                attempts: 1,
                outcome: AttemptLoopOutcome::TimedOut,
                ..
            },
        ]
    ));
}

#[tokio::test]
async fn observer_wrappers_add_host_attribution_without_expanding_transport_events() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"first".to_vec(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"second".to_vec(),
        },
    ])
    .await;
    let sink = Arc::new(ScopedEventSink::default());
    let first_scope = "canary-provider-first";
    let second_scope = "canary-provider-second";
    for scope in [first_scope, second_scope] {
        let settings =
            ProviderHttpTransportSettings::default().with_observer(Arc::new(ScopedObserver {
                scope,
                sink: sink.clone(),
            }));
        let debug = format!("{settings:?}");
        assert!(debug.contains("observer_configured: true"));
        assert!(!debug.contains(scope));
        ProviderTransport::builder(server.endpoint())
            .with_http_transport_settings(settings)
            .build()
            .unwrap()
            .execute(
                RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap()),
                CallOptions::default(),
            )
            .await
            .unwrap();
    }

    let events = sink.0.lock().unwrap();
    assert!(events.iter().any(|(scope, _)| *scope == first_scope));
    assert!(events.iter().any(|(scope, _)| *scope == second_scope));
    for (_, event) in events.iter() {
        let debug = format!("{event:?}");
        assert!(!debug.contains(first_scope));
        assert!(!debug.contains(second_scope));
    }
}

#[tokio::test]
async fn transport_events_redact_credentials_endpoint_query_response_prompt_and_tool_payloads() {
    let server = TestServer::spawn(vec![ServerAction::Respond {
        status: 200,
        headers: vec![(
            "X-Canary-Response-Header".to_owned(),
            "canary-response-header-value".to_owned(),
        )],
        body: b"canary-response-body".to_vec(),
    }])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let transport = ProviderTransport::builder(server.endpoint())
        .with_auth(Arc::new(SecretAuth))
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_observer(observer.clone()),
        )
        .build()
        .unwrap();
    let plan = RequestPlan::new(
        Method::POST,
        RequestTarget::new("canary-request-target?hint=canary-endpoint-query").unwrap(),
    )
    .with_headers(
        RequestHeaders::new()
            .try_insert(
                HeaderName::from_static("x-canary-request-header"),
                HeaderValue::from_static("canary-request-header-value"),
            )
            .unwrap(),
    )
    .with_body(
        RequestBody::json(&serde_json::json!({
            "prompt": "canary-prompt-payload",
            "tool": { "input": "canary-tool-payload" }
        }))
        .unwrap(),
    );
    transport
        .execute(plan, CallOptions::default())
        .await
        .unwrap();

    let debug = format!("{:?}", observer.events());
    for sentinel in [
        "canary-request-target",
        "canary-endpoint-query",
        "x-canary-request-header",
        "canary-request-header-value",
        "canary-prompt-payload",
        "canary-tool-payload",
        "x-canary-response-header",
        "canary-response-header-value",
        "canary-response-body",
        "canary-header-secret",
        "canary-query-secret",
    ] {
        assert!(!debug.contains(sentinel), "event leaked {sentinel}");
    }
}

#[tokio::test]
async fn cloned_settings_share_observation_but_built_transports_isolate_admission() {
    let server = TestServer::spawn(vec![
        ServerAction::RespondThenHold {
            status: 200,
            prefix: b"held".to_vec(),
            content_length: 100,
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"independent".to_vec(),
        },
    ])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let settings = ProviderHttpTransportSettings::default()
        .with_limits(TransportLimits {
            max_connections: 1,
            max_in_flight_requests: 1,
            max_queued_requests: 1,
            ..TransportLimits::default()
        })
        .unwrap()
        .with_observer(observer.clone());
    let first = ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(settings.clone())
        .build()
        .unwrap();
    let second = ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(settings)
        .build()
        .unwrap();

    let held = first
        .execute_stream(
            RequestPlan::new(Method::GET, RequestTarget::new("held").unwrap()),
            CallOptions::default(),
        )
        .await
        .unwrap();
    let independent = tokio::time::timeout(
        Duration::from_secs(1),
        second.execute(
            RequestPlan::new(Method::GET, RequestTarget::new("independent").unwrap()),
            CallOptions::default(),
        ),
    )
    .await
    .expect("independently built transports must not share admission")
    .unwrap();
    assert_eq!(
        independent.body(),
        &bytes::Bytes::from_static(b"independent")
    );
    drop(held);

    let call_ids = observer
        .events()
        .iter()
        .filter_map(|event| match event {
            TransportEvent::AttemptBudgetResolved { call_id, .. } => Some(*call_id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(call_ids.len(), 2);
    assert_ne!(call_ids[0], call_ids[1]);
}

#[test]
fn common_credential_and_transport_headers_are_protected_exactly() {
    for name in ["Authorization", "API-Key", "X-API-Key", "Host"] {
        let error = RequestHeaders::new()
            .try_insert(
                HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_static("caller-value"),
            )
            .unwrap_err();
        assert_eq!(error, RequestBuildError::ProtectedHeader, "header: {name}");
    }

    for name in [
        "x-token-count-mode",
        "x-secret-sampling-mode",
        "x-api-key-count",
        "api_key",
    ] {
        RequestHeaders::new()
            .try_insert(
                HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_static("ordinary-provider-value"),
            )
            .unwrap_or_else(|error| panic!("ordinary header {name} was rejected: {error}"));
    }
}

#[derive(Default)]
struct ExactCredentialHeaderAuth;

#[async_trait]
impl AuthApplier for ExactCredentialHeaderAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        CredentialPatch::new()
            .try_insert(
                AUTHORIZATION,
                HeaderValue::from_static("Bearer transport-owned"),
            )
            .and_then(|patch| {
                patch.try_insert(
                    HeaderName::from_static("x-provider-credential"),
                    HeaderValue::from_static("transport-owned"),
                )
            })
            .map_err(|_| Error::new(ErrorKind::Authentication, "credential construction failed"))
    }
}

#[tokio::test]
async fn selected_credential_header_collision_fails_before_submission() {
    let server = TestServer::spawn(vec![ServerAction::Respond {
        status: 200,
        headers: Vec::new(),
        body: Vec::new(),
    }])
    .await;
    let plan = RequestPlan::new(Method::POST, RequestTarget::new("responses").unwrap())
        .with_headers(
            RequestHeaders::new()
                .try_insert(
                    HeaderName::from_static("x-provider-credential"),
                    HeaderValue::from_static("caller-canary-credential"),
                )
                .unwrap(),
        );
    let error = ProviderTransport::builder(server.endpoint())
        .with_auth(Arc::new(ExactCredentialHeaderAuth))
        .build()
        .unwrap()
        .execute(plan, CallOptions::default())
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(server.requests().is_empty());
    for surface in [format!("{error:?}"), error.to_string()] {
        assert!(!surface.contains("caller-canary-credential"));
        assert!(!surface.contains("transport-owned"));
    }
}

#[tokio::test]
async fn provider_body_authority_names_are_inert_transport_data() {
    let server = TestServer::spawn(vec![
        ServerAction::DropAfterRead,
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"ok".to_vec(),
        },
    ])
    .await;
    let body = serde_json::json!({
        "endpoint": "http://attacker.invalid/admin",
        "method": "DELETE",
        "retry": 0,
        "timeout": 0,
        "authorization": "Bearer body-owned",
        "api_key": "body-owned",
        "headers": {
            "Authorization": "Bearer nested-body-owned",
            "X-Provider-Credential": "nested-body-owned"
        }
    });
    let plan = RequestPlan::new(Method::POST, RequestTarget::new("authority-check").unwrap())
        .with_headers(
            RequestHeaders::new()
                .try_insert(
                    HeaderName::from_static("x-token-count-mode"),
                    HeaderValue::from_static("enabled"),
                )
                .unwrap(),
        )
        .with_body(RequestBody::json(&body).unwrap())
        .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
        .unwrap();
    let response = ProviderTransport::builder(server.endpoint())
        .with_auth(Arc::new(ExactCredentialHeaderAuth))
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_retry_policy(retry_policy(2)),
        )
        .build()
        .unwrap()
        .execute(plan, CallOptions::default())
        .await
        .unwrap();

    assert_eq!(response.attempts(), 2);
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert!(
            request
                .head
                .starts_with("POST /v1/authority-check HTTP/1.1")
        );
        assert_eq!(
            request.header("authorization"),
            Some("Bearer transport-owned")
        );
        assert_eq!(
            request.header("x-provider-credential"),
            Some("transport-owned")
        );
        assert_eq!(request.header("x-token-count-mode"), Some("enabled"));
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
            body
        );
    }
}

#[tokio::test]
async fn post_is_not_replayed_after_the_server_reads_its_body() {
    let server = TestServer::spawn(vec![ServerAction::DropAfterRead]).await;
    let observer = Arc::new(RecordingObserver::default());
    let result = transport(server.endpoint(), observer.clone())
        .execute(json_post(ReplaySafety::Never), CallOptions::default())
        .await;
    assert!(result.is_err());
    assert_eq!(observer.attempts(), 1);
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn keyed_post_rebuilds_the_same_body_and_key_but_not_across_calls() {
    let server = TestServer::spawn(vec![
        ServerAction::DropAfterRead,
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"first".to_vec(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"second".to_vec(),
        },
    ])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let transport = transport(server.endpoint(), observer);
    let plan = json_post(ReplaySafety::IdempotencyKey(
        IdempotencyHeader::new(HeaderName::from_static("idempotency-key")).unwrap(),
    ));
    let first = transport
        .execute(plan.clone(), CallOptions::default())
        .await
        .unwrap();
    let second = transport
        .execute(plan, CallOptions::default())
        .await
        .unwrap();
    assert_eq!(first.attempts(), 2);
    assert_eq!(second.attempts(), 1);

    let requests = server.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].body, requests[1].body);
    let first_key = requests[0].header("idempotency-key").unwrap();
    assert_eq!(first_key, requests[1].header("idempotency-key").unwrap());
    assert_ne!(first_key, requests[2].header("idempotency-key").unwrap());
}

#[tokio::test]
async fn multipart_retry_is_byte_for_byte_rebuildable() {
    let server = TestServer::spawn(vec![
        ServerAction::DropAfterRead,
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: Vec::new(),
        },
    ])
    .await;
    let body = MultipartBody::new(vec![
        MultipartPart::field("purpose", "batch").unwrap(),
        MultipartPart::file(
            "file",
            "input.jsonl",
            HeaderValue::from_static("application/jsonl"),
            "{\"input\":\"hello\"}\n",
        )
        .unwrap(),
    ]);
    let plan = RequestPlan::new(Method::POST, RequestTarget::new("files").unwrap())
        .with_body(RequestBody::multipart(body))
        .with_replay_safety(ReplaySafety::IdempotencyKey(
            IdempotencyHeader::new(HeaderName::from_static("idempotency-key")).unwrap(),
        ))
        .unwrap();
    transport(server.endpoint(), Arc::new(RecordingObserver::default()))
        .execute(plan, CallOptions::default())
        .await
        .unwrap();
    let requests = server.requests();
    assert_eq!(requests[0].body, requests[1].body);
    assert!(
        requests[0]
            .header("content-type")
            .unwrap()
            .starts_with("multipart/form-data; boundary=")
    );
}

#[derive(Default)]
struct RefreshingAuth(Mutex<Vec<AuthRefresh>>);

#[async_trait]
impl AuthApplier for RefreshingAuth {
    fn supports_refresh(&self) -> bool {
        true
    }

    async fn apply(
        &self,
        _context: AuthContext<'_>,
        refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        self.0.lock().unwrap().push(refresh);
        CredentialPatch::new()
            .try_insert(AUTHORIZATION, HeaderValue::from_static("Bearer secret"))
            .map(|patch| {
                patch.with_revision(match refresh {
                    AuthRefresh::Current => CredentialRevision::new(1),
                    AuthRefresh::AfterUnauthorized { .. } => CredentialRevision::new(2),
                })
            })
            .map_err(|_| Error::new(ErrorKind::Authentication, "credential construction failed"))
    }
}

#[tokio::test]
async fn unauthorized_rate_limit_and_server_error_share_one_budget() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 401,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 429,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 500,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: Vec::new(),
        },
    ])
    .await;
    let auth = Arc::new(RefreshingAuth::default());
    let transport = ProviderTransport::builder(server.endpoint())
        .with_auth(auth.clone())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_retry_policy(retry_policy(3)),
        )
        .build()
        .unwrap();
    let response = transport
        .execute(
            json_post(ReplaySafety::IdempotencyKey(
                IdempotencyHeader::new(HeaderName::from_static("idempotency-key")).unwrap(),
            )),
            CallOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(response.attempts(), 3);
    assert_eq!(server.requests().len(), 3);
    assert_eq!(
        *auth.0.lock().unwrap(),
        vec![
            AuthRefresh::Current,
            AuthRefresh::AfterUnauthorized {
                rejected_revision: CredentialRevision::new(1),
            },
            AuthRefresh::Current
        ]
    );
}

#[tokio::test]
async fn caller_attempt_cap_also_bounds_authentication_refresh() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 401,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 500,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"must-not-reach".to_vec(),
        },
    ])
    .await;
    let auth = Arc::new(RefreshingAuth::default());
    let transport = ProviderTransport::builder(server.endpoint())
        .with_auth(auth.clone())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_retry_policy(retry_policy(5)),
        )
        .build()
        .unwrap();

    let response = transport
        .execute(
            json_post(ReplaySafety::IdempotencyKey(
                IdempotencyHeader::new(HeaderName::from_static("idempotency-key")).unwrap(),
            )),
            CallOptions::default().with_max_attempts(2).unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(response.attempts(), 2);
    assert_eq!(server.requests().len(), 2);
    assert_eq!(
        *auth.0.lock().unwrap(),
        vec![
            AuthRefresh::Current,
            AuthRefresh::AfterUnauthorized {
                rejected_revision: CredentialRevision::new(1),
            },
        ]
    );
}

#[derive(Debug)]
struct OverloadClassifier;

impl RetryClassifier for OverloadClassifier {
    fn classify_status(&self, status: StatusCode) -> Option<RetryReason> {
        (status.as_u16() == 529).then_some(RetryReason::ServerUnavailable)
    }
}

#[tokio::test]
async fn provider_classifier_extends_statuses_without_owning_the_retry_budget() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 529,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"recovered".to_vec(),
        },
    ])
    .await;
    let transport = ProviderTransport::builder(server.endpoint())
        .with_retry_classifier(Arc::new(OverloadClassifier))
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_retry_policy(retry_policy(2)),
        )
        .build()
        .unwrap();
    let response = transport
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap())
                .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
                .unwrap(),
            CallOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(response.attempts(), 2);
    assert_eq!(response.body(), &bytes::Bytes::from_static(b"recovered"));
    assert_eq!(server.requests().len(), 2);
}

#[tokio::test]
async fn caller_attempt_cap_narrows_policy_without_expanding_replay_authority() {
    let safe_server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 500,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 500,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"must-not-reach".to_vec(),
        },
    ])
    .await;
    let safe_observer = Arc::new(RecordingObserver::default());
    let safe_transport = ProviderTransport::builder(safe_server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default()
                .with_retry_policy(retry_policy(5))
                .with_observer(safe_observer.clone()),
        )
        .build()
        .unwrap();
    let safe_response = safe_transport
        .execute(
            json_post(ReplaySafety::SemanticallyIdempotent),
            CallOptions::default().with_max_attempts(2).unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(safe_response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(safe_response.attempts(), 2);
    assert_eq!(safe_server.requests().len(), 2);
    assert!(safe_observer.events().iter().any(|event| matches!(
        event,
        TransportEvent::AttemptBudgetResolved {
            provider_maximum_attempts: 5,
            caller_maximum_attempts: Some(2),
            effective_maximum_attempts: 2,
            limiting_authority: RetryLimit::CallerCap,
            ..
        }
    )));
    assert!(safe_observer.events().iter().any(|event| matches!(
        event,
        TransportEvent::AttemptLoopFinished {
            attempts: 2,
            outcome: AttemptLoopOutcome::ResponseReturned { status },
            ..
        } if *status == StatusCode::INTERNAL_SERVER_ERROR
    )));

    let unsafe_server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 500,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"must-not-replay".to_vec(),
        },
    ])
    .await;
    let unsafe_observer = Arc::new(RecordingObserver::default());
    let unsafe_transport = ProviderTransport::builder(unsafe_server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default()
                .with_retry_policy(retry_policy(5))
                .with_observer(unsafe_observer.clone()),
        )
        .build()
        .unwrap();
    let unsafe_response = unsafe_transport
        .execute(
            json_post(ReplaySafety::Never),
            CallOptions::default().with_max_attempts(2).unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(unsafe_response.attempts(), 1);
    assert_eq!(unsafe_server.requests().len(), 1);
    assert!(unsafe_observer.events().iter().any(|event| matches!(
        event,
        TransportEvent::AttemptBudgetResolved {
            provider_maximum_attempts: 5,
            caller_maximum_attempts: Some(2),
            effective_maximum_attempts: 1,
            limiting_authority: RetryLimit::ReplaySafety,
            ..
        }
    )));
}

#[tokio::test]
async fn server_retry_after_beyond_the_policy_is_declined_without_sleeping() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 429,
            headers: vec![("Retry-After".to_owned(), "1".to_owned())],
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"must-not-retry".to_vec(),
        },
    ])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let transport = ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default()
                .with_retry_policy(retry_policy(2).with_max_server_delay(Duration::from_millis(50)))
                .with_observer(observer.clone()),
        )
        .build()
        .unwrap();
    let started = std::time::Instant::now();
    let response = transport
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap())
                .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
                .unwrap(),
            CallOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(started.elapsed() < Duration::from_millis(500));
    assert_eq!(server.requests().len(), 1);
    assert!(observer.events().iter().any(|event| matches!(
        event,
        TransportEvent::RetryDeclined {
            reason: RetryReason::RateLimited,
            completed_attempts: 1,
            delay: Some(delay),
            limiting_authority: RetryLimit::ServerDelayPolicy,
            ..
        } if *delay == Duration::from_secs(1)
    )));
}

#[tokio::test]
async fn server_retry_after_cannot_outlive_the_deadline() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 429,
            headers: vec![("Retry-After".to_owned(), "1".to_owned())],
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"must-not-retry".to_vec(),
        },
    ])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let transport = ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default()
                .with_retry_policy(retry_policy(2))
                .with_observer(observer.clone()),
        )
        .build()
        .unwrap();
    let response = transport
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap())
                .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
                .unwrap(),
            CallOptions::default()
                .with_deadline(std::time::Instant::now() + Duration::from_millis(100)),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(server.requests().len(), 1);
    assert!(observer.events().iter().any(|event| matches!(
        event,
        TransportEvent::RetryDeclined {
            reason: RetryReason::RateLimited,
            completed_attempts: 1,
            delay: Some(delay),
            limiting_authority: RetryLimit::CallDeadline,
            ..
        } if *delay == Duration::from_secs(1)
    )));
}

#[tokio::test]
async fn local_backoff_deadline_reports_retry_declined_before_timeout() {
    let scenarios = [
        (ServerAction::DropAfterRead, RetryReason::Transport),
        (
            ServerAction::Respond {
                status: 503,
                headers: Vec::new(),
                body: Vec::new(),
            },
            RetryReason::ServerUnavailable,
        ),
    ];

    for (first_action, expected_reason) in scenarios {
        let server = TestServer::spawn(vec![
            first_action,
            ServerAction::Respond {
                status: 200,
                headers: Vec::new(),
                body: b"must-not-retry".to_vec(),
            },
        ])
        .await;
        let observer = Arc::new(RecordingObserver::default());
        let transport = ProviderTransport::builder(server.endpoint())
            .with_http_transport_settings(
                ProviderHttpTransportSettings::default()
                    .with_retry_policy(
                        RetryPolicy::new(2)
                            .unwrap()
                            .with_backoff(Duration::from_secs(1), Duration::from_secs(1))
                            .with_jitter(false),
                    )
                    .with_observer(observer.clone()),
            )
            .build()
            .unwrap();
        let error = transport
            .execute(
                RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap())
                    .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
                    .unwrap(),
                CallOptions::default()
                    .with_deadline(std::time::Instant::now() + Duration::from_millis(100)),
            )
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert_eq!(server.requests().len(), 1);
        assert!(observer.events().iter().any(|event| matches!(
            event,
            TransportEvent::RetryDeclined {
                reason,
                completed_attempts: 1,
                delay: None,
                limiting_authority: RetryLimit::CallDeadline,
                ..
            } if *reason == expected_reason
        )));
        assert!(matches!(
            observer.events().last(),
            Some(TransportEvent::AttemptLoopFinished {
                attempts: 1,
                outcome: AttemptLoopOutcome::TimedOut,
                ..
            })
        ));
    }
}

#[tokio::test]
async fn api_redirect_is_returned_without_following_it() {
    let destination = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let destination_address = destination.local_addr().unwrap();
    let server = TestServer::spawn(vec![ServerAction::Respond {
        status: 302,
        headers: vec![(
            "Location".to_owned(),
            format!("http://{destination_address}/stolen"),
        )],
        body: Vec::new(),
    }])
    .await;
    let response = transport(server.endpoint(), Arc::new(RecordingObserver::default()))
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap())
                .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
                .unwrap(),
            CallOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FOUND);
    assert!(
        tokio::time::timeout(Duration::from_millis(100), destination.accept())
            .await
            .is_err()
    );
}

#[derive(Clone)]
struct FixedResolver {
    address: SocketAddr,
    calls: Arc<Mutex<Vec<(String, u16)>>>,
}

#[async_trait]
impl Resolver for FixedResolver {
    async fn resolve(&self, host: &str, port: u16) -> Result<Vec<SocketAddr>, EndpointError> {
        self.calls.lock().unwrap().push((host.to_owned(), port));
        Ok(vec![self.address])
    }
}

#[tokio::test]
async fn connector_dns_guard_resolves_the_actual_host_and_rechecks_the_peer() {
    let server = TestServer::spawn(vec![ServerAction::Respond {
        status: 200,
        headers: Vec::new(),
        body: b"ok".to_vec(),
    }])
    .await;
    let calls = Arc::new(Mutex::new(Vec::new()));
    let resolver = FixedResolver {
        address: server.address,
        calls: calls.clone(),
    };
    let endpoint = EndpointConfig::new(
        format!("http://model.local:{}/v1", server.address.port()),
        EndpointPolicy::LocalExplicit(siumai_transport::LocalNetworkGrant::Loopback),
    )
    .unwrap();
    let transport = ProviderTransport::builder(endpoint)
        .with_resolver(Arc::new(resolver))
        .build()
        .unwrap();
    let response = transport
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("models").unwrap()),
            CallOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(response.body(), &bytes::Bytes::from_static(b"ok"));
    assert_eq!(*calls.lock().unwrap(), vec![("model.local".to_owned(), 0)]);
}

#[test]
fn shared_address_grant_is_exact_across_http_resource_and_websocket_surfaces() {
    for address in ["100.64.0.1", "100.127.255.254"] {
        EndpointConfig::shared_address_space_explicit(format!("http://{address}:8080/v1")).unwrap();
        ResourceUrl::shared_address_space_explicit(format!("http://{address}:8080/file")).unwrap();
        WebSocketEndpoint::shared_address_space_explicit(format!("ws://{address}:8080/live"))
            .unwrap();
    }

    for address in ["100.63.255.255", "100.128.0.0"] {
        assert!(
            EndpointConfig::shared_address_space_explicit(format!("http://{address}:8080/v1"))
                .is_err()
        );
        assert!(
            ResourceUrl::shared_address_space_explicit(format!("http://{address}:8080/file"))
                .is_err()
        );
        assert!(
            WebSocketEndpoint::shared_address_space_explicit(format!("ws://{address}:8080/live"))
                .is_err()
        );
    }

    assert!(EndpointConfig::public_custom("https://100.64.0.1/v1").is_err());
    assert!(ResourceUrl::public("https://100.64.0.1/file").is_err());
    assert!(WebSocketEndpoint::public_custom("wss://100.64.0.1/live").is_err());
}

#[tokio::test]
async fn queued_cancellation_and_stream_drop_release_all_permits() {
    let server = TestServer::spawn(vec![
        ServerAction::RespondThenHold {
            status: 200,
            prefix: b"first".to_vec(),
            content_length: 100,
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"after-drop".to_vec(),
        },
    ])
    .await;
    let limits = TransportLimits {
        max_connections: 1,
        max_in_flight_requests: 1,
        max_queued_requests: 1,
        ..TransportLimits::default()
    };
    let transport = ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default()
                .with_limits(limits)
                .unwrap(),
        )
        .build()
        .unwrap();
    let first = transport
        .execute_stream(
            RequestPlan::new(Method::GET, RequestTarget::new("stream").unwrap()),
            CallOptions::default(),
        )
        .await
        .unwrap();

    let cancellation = Cancellation::new();
    let queued = {
        let transport = transport.clone();
        let cancellation = cancellation.clone();
        tokio::spawn(async move {
            transport
                .execute(
                    RequestPlan::new(Method::GET, RequestTarget::new("queued").unwrap()),
                    CallOptions::default().with_cancellation(cancellation),
                )
                .await
        })
    };
    tokio::task::yield_now().await;
    cancellation.cancel();
    let error = queued.await.unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);

    drop(first);
    let response = tokio::time::timeout(
        Duration::from_secs(1),
        transport.execute(
            RequestPlan::new(Method::GET, RequestTarget::new("after-drop").unwrap()),
            CallOptions::default(),
        ),
    )
    .await
    .expect("stream drop must release the permit")
    .unwrap();
    assert_eq!(response.body(), &bytes::Bytes::from_static(b"after-drop"));
}

#[tokio::test]
async fn established_stream_failure_never_replays() {
    let server = TestServer::spawn(vec![
        ServerAction::RespondThenHold {
            status: 200,
            prefix: b"partial".to_vec(),
            content_length: 100,
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: b"unexpected-replay".to_vec(),
        },
    ])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let transport = transport(server.endpoint(), observer.clone());
    let response = transport
        .execute_stream(
            RequestPlan::new(Method::GET, RequestTarget::new("stream").unwrap())
                .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
                .unwrap(),
            CallOptions::default()
                .with_deadline(std::time::Instant::now() + Duration::from_millis(50)),
        )
        .await
        .unwrap();
    let mut body = response.into_body();
    assert_eq!(
        body.next().await.unwrap().unwrap(),
        bytes::Bytes::from_static(b"partial")
    );
    let error = body.next().await.unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert_eq!(observer.attempts(), 1);
    assert_eq!(server.requests().len(), 1);
}

#[derive(Default)]
struct SecretAuth;

#[async_trait]
impl AuthApplier for SecretAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        CredentialPatch::new()
            .try_insert(
                AUTHORIZATION,
                HeaderValue::from_static("Bearer canary-header-secret"),
            )
            .and_then(|patch| patch.try_insert_query("key", "canary-query-secret"))
            .map_err(|_| Error::new(ErrorKind::Authentication, "credential construction failed"))
    }
}

#[tokio::test]
async fn default_error_surfaces_redact_credentials_endpoint_query_response_prompt_and_tool_payloads()
 {
    let server = TestServer::spawn(vec![ServerAction::Respond {
        status: 200,
        headers: vec![(
            "X-Request-Id".to_owned(),
            "canary-response-header".to_owned(),
        )],
        body: b"canary-response-body".to_vec(),
    }])
    .await;
    let limits = TransportLimits {
        max_response_bytes: 4,
        ..TransportLimits::default()
    };
    let transport = ProviderTransport::builder(server.endpoint())
        .with_auth(Arc::new(SecretAuth))
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default()
                .with_limits(limits)
                .unwrap(),
        )
        .build()
        .unwrap();
    let error = transport
        .execute(
            RequestPlan::new(
                Method::POST,
                RequestTarget::new("secret?hint=canary-endpoint-query").unwrap(),
            )
            .with_body(
                RequestBody::json(&serde_json::json!({
                    "prompt": "canary-prompt-payload",
                    "tool": { "input": "canary-tool-payload" }
                }))
                .unwrap(),
            ),
            CallOptions::default(),
        )
        .await
        .unwrap_err();
    for surface in [
        format!("{error:?}"),
        error.to_string(),
        serde_json::to_string(&error).unwrap(),
    ] {
        assert!(!surface.contains("canary-header-secret"));
        assert!(!surface.contains("canary-query-secret"));
        assert!(!surface.contains("canary-endpoint-query"));
        assert!(!surface.contains("canary-response-header"));
        assert!(!surface.contains("canary-response-body"));
        assert!(!surface.contains("canary-prompt-payload"));
        assert!(!surface.contains("canary-tool-payload"));
    }
    assert!(error.sensitive_response().unwrap().expose().1.len() <= 4);
}

#[tokio::test]
async fn cancellation_interrupts_backoff_before_another_attempt() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 500,
            headers: Vec::new(),
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: Vec::new(),
            body: Vec::new(),
        },
    ])
    .await;
    let observer = Arc::new(RecordingObserver::default());
    let transport = ProviderTransport::builder(server.endpoint())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default()
                .with_retry_policy(
                    RetryPolicy::new(3)
                        .unwrap()
                        .with_backoff(Duration::from_secs(10), Duration::from_secs(10))
                        .with_jitter(false),
                )
                .with_observer(observer.clone()),
        )
        .build()
        .unwrap();
    let cancellation = Cancellation::new();
    let call = {
        let cancellation = cancellation.clone();
        tokio::spawn(async move {
            transport
                .execute(
                    RequestPlan::new(Method::GET, RequestTarget::new("retry").unwrap())
                        .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
                        .unwrap(),
                    CallOptions::default().with_cancellation(cancellation),
                )
                .await
        })
    };
    while observer.attempts() == 0 {
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    cancellation.cancel();
    let error = call.await.unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!(observer.attempts(), 1);
    assert!(observer.events().iter().any(|event| matches!(
        event,
        TransportEvent::AttemptLoopFinished {
            attempts: 1,
            outcome: AttemptLoopOutcome::Cancelled,
            ..
        }
    )));
}

#[tokio::test]
async fn resource_redirects_are_manual_bounded_and_never_authenticated() {
    let server = TestServer::spawn(vec![
        ServerAction::Respond {
            status: 302,
            headers: vec![(
                "Location".to_owned(),
                "/final.png?sig=canary-signature".to_owned(),
            )],
            body: Vec::new(),
        },
        ServerAction::Respond {
            status: 200,
            headers: vec![("Content-Type".to_owned(), "image/png".to_owned())],
            body: b"\x89PNG\r\n\x1a\ncontent".to_vec(),
        },
    ])
    .await;
    let downloader = ResourceDownloader::builder().build().unwrap();
    let resource = downloader
        .download(
            ResourceUrl::local_explicit(format!("http://{}/start", server.address)).unwrap(),
            ResourceDownloadOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(resource.declared_media_type(), Some("image/png"));
    assert_eq!(resource.detected_media_type(), Some("image/png"));
    assert!(!format!("{resource:?}").contains("canary-signature"));
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|request| {
        request.header("authorization").is_none() && request.header("proxy-authorization").is_none()
    }));
}

#[tokio::test]
async fn local_resource_redirect_cannot_change_origin() {
    let destination = TestServer::spawn(vec![ServerAction::Respond {
        status: 200,
        headers: Vec::new(),
        body: b"must-not-arrive".to_vec(),
    }])
    .await;
    let source = TestServer::spawn(vec![ServerAction::Respond {
        status: 302,
        headers: vec![(
            "Location".to_owned(),
            format!("http://{}/redirected", destination.address),
        )],
        body: Vec::new(),
    }])
    .await;

    let error = ResourceDownloader::builder()
        .build()
        .unwrap()
        .download(
            ResourceUrl::local_explicit(format!("http://{}/start", source.address)).unwrap(),
            ResourceDownloadOptions::default(),
        )
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(source.requests().len(), 1);
    assert!(destination.requests().is_empty());
}

#[tokio::test]
async fn loopback_resource_redirect_cannot_pivot_to_link_local_metadata() {
    let server = TestServer::spawn(vec![ServerAction::Respond {
        status: 302,
        headers: vec![(
            "Location".to_owned(),
            "http://169.254.169.254/latest/meta-data".to_owned(),
        )],
        body: Vec::new(),
    }])
    .await;
    let error = ResourceDownloader::builder()
        .build()
        .unwrap()
        .download(
            ResourceUrl::local_explicit(format!("http://{}/start", server.address)).unwrap(),
            ResourceDownloadOptions::default(),
        )
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn resource_body_limit_and_deadline_apply_after_headers() {
    let oversized = TestServer::spawn(vec![ServerAction::Respond {
        status: 200,
        headers: Vec::new(),
        body: b"too-large".to_vec(),
    }])
    .await;
    let downloader = ResourceDownloader::builder()
        .with_limits(TransportLimits {
            max_response_bytes: 4,
            ..TransportLimits::default()
        })
        .build()
        .unwrap();
    let error = downloader
        .download(
            ResourceUrl::local_explicit(format!("http://{}/large", oversized.address)).unwrap(),
            ResourceDownloadOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResponseLimit);

    let slow = TestServer::spawn(vec![ServerAction::RespondThenHold {
        status: 200,
        prefix: b"part".to_vec(),
        content_length: 100,
    }])
    .await;
    let error = ResourceDownloader::builder()
        .build()
        .unwrap()
        .download(
            ResourceUrl::local_explicit(format!("http://{}/slow", slow.address)).unwrap(),
            ResourceDownloadOptions::default()
                .with_deadline(std::time::Instant::now() + Duration::from_millis(50)),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
}

#[tokio::test]
async fn resource_default_total_timeout_and_error_capture_are_bounded() {
    let slow = TestServer::spawn(vec![ServerAction::RespondThenHold {
        status: 200,
        prefix: b"part".to_vec(),
        content_length: 100,
    }])
    .await;
    let timeout_error = ResourceDownloader::builder()
        .with_download_timeout(Duration::from_millis(50))
        .build()
        .unwrap()
        .download(
            ResourceUrl::local_explicit(format!("http://{}/slow", slow.address)).unwrap(),
            ResourceDownloadOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(timeout_error.kind(), ErrorKind::Timeout);

    let failed = TestServer::spawn(vec![ServerAction::Respond {
        status: 500,
        headers: vec![(
            "X-Request-Id".to_owned(),
            "canary-resource-request-id".to_owned(),
        )],
        body: b"canary-resource-body".to_vec(),
    }])
    .await;
    let status_error = ResourceDownloader::builder()
        .with_limits(TransportLimits {
            max_response_bytes: 4,
            ..TransportLimits::default()
        })
        .build()
        .unwrap()
        .download(
            ResourceUrl::local_explicit(format!("http://{}/failed", failed.address)).unwrap(),
            ResourceDownloadOptions::default(),
        )
        .await
        .unwrap_err();
    for surface in [
        format!("{status_error:?}"),
        status_error.to_string(),
        serde_json::to_string(&status_error).unwrap(),
    ] {
        assert!(!surface.contains("canary-resource-request-id"));
        assert!(!surface.contains("canary-resource-body"));
    }
    let diagnostics = status_error.diagnostics().unwrap();
    assert!(diagnostics.body_truncated());
    assert_eq!(
        status_error.sensitive_response().unwrap().expose().1.len(),
        4
    );
}
