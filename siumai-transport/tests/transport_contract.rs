use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use http::header::{AUTHORIZATION, HeaderName, HeaderValue};
use http::{Method, StatusCode};
use siumai_core::{CallOptions, Cancellation, Error, ErrorKind};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, CredentialRevision, EndpointConfig,
    EndpointError, EndpointPolicy, IdempotencyHeader, MultipartBody, MultipartPart,
    ProviderTransport, ReplaySafety, RequestBody, RequestPlan, RequestTarget, Resolver,
    ResourceDownloadOptions, ResourceDownloader, ResourceUrl, RetryClassifier, RetryPolicy,
    RetryReason, TransportEvent, TransportLimits, TransportObserver,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Clone)]
enum ServerAction {
    DropAfterRead,
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
}

impl TransportObserver for RecordingObserver {
    fn observe(&self, event: &TransportEvent) {
        self.0.lock().unwrap().push(event.clone());
    }
}

fn retry_policy(maximum_attempts: u8) -> RetryPolicy {
    RetryPolicy::new(maximum_attempts)
        .unwrap()
        .with_backoff(Duration::ZERO, Duration::ZERO)
}

fn transport(endpoint: EndpointConfig, observer: Arc<dyn TransportObserver>) -> ProviderTransport {
    ProviderTransport::builder(endpoint)
        .with_retry_policy(retry_policy(3))
        .with_observer(observer)
        .build()
        .unwrap()
}

fn json_post(replay: ReplaySafety) -> RequestPlan {
    RequestPlan::new(Method::POST, RequestTarget::new("responses").unwrap())
        .with_body(RequestBody::json(&serde_json::json!({ "input": "hello" })).unwrap())
        .with_replay_safety(replay)
        .unwrap()
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
        .with_retry_policy(retry_policy(3))
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
        .with_retry_policy(retry_policy(2))
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
async fn server_retry_after_is_not_capped_and_cannot_outlive_the_deadline() {
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
    let transport = ProviderTransport::builder(server.endpoint())
        .with_retry_policy(retry_policy(2))
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
        .with_limits(limits)
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
async fn default_error_surfaces_redact_credentials_response_headers_and_body() {
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
        .with_limits(limits)
        .build()
        .unwrap();
    let error = transport
        .execute(
            RequestPlan::new(Method::GET, RequestTarget::new("secret").unwrap()),
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
        assert!(!surface.contains("canary-response-header"));
        assert!(!surface.contains("canary-response-body"));
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
        .with_observer(observer.clone())
        .with_retry_policy(
            RetryPolicy::new(3)
                .unwrap()
                .with_backoff(Duration::from_secs(10), Duration::from_secs(10))
                .with_jitter(false),
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
