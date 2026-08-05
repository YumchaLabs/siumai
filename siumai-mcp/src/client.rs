use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use rmcp::ClientHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientInfo, JsonObject, Meta, NumberOrString,
    PaginatedRequestParams, ProgressNotificationParam, ProgressToken, Tool,
};
use rmcp::service::{Peer, RoleClient, RunningService, ServiceExt};
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use serde::Serialize;
use serde_json::Value;
use siumai_core::ToolOutcome;
use siumai_runtime::tool::{EffectCertainty, ToolExecutionError};
use tokio::sync::{Mutex, broadcast};

use crate::catalog::{McpCatalogFingerprint, McpToolCatalog};
use crate::{McpClientConfig, McpError, McpHttpEndpointPolicy};

#[derive(Debug, Clone, Serialize)]
pub struct McpProgress {
    pub progress_token: Value,
    pub progress: f64,
    pub total: Option<f64>,
    pub message: Option<String>,
}

#[derive(Default)]
struct SessionState {
    closed: AtomicBool,
    stale: AtomicBool,
    notification_overflowed: AtomicBool,
    epoch: AtomicU64,
    notification_count: AtomicUsize,
    progress_count: AtomicUsize,
    active_catalog: RwLock<Option<McpCatalogFingerprint>>,
    progress_tx: RwLock<Option<broadcast::Sender<McpProgress>>>,
    max_notifications: AtomicUsize,
}

impl SessionState {
    fn configure(&self, config: &McpClientConfig) {
        self.max_notifications
            .store(config.limits().max_notifications(), Ordering::Release);
        let (sender, _) = broadcast::channel(config.limits().progress_queue_capacity());
        *self
            .progress_tx
            .write()
            .expect("state lock is not poisoned") = Some(sender);
    }

    fn record_notification(&self) -> bool {
        let count = self.notification_count.fetch_add(1, Ordering::AcqRel) + 1;
        if count > self.max_notifications.load(Ordering::Acquire) {
            self.notification_overflowed.store(true, Ordering::Release);
            return false;
        }
        true
    }

    fn mark_catalog_changed(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.stale.store(true, Ordering::Release);
    }

    fn mark_progress(&self, params: ProgressNotificationParam) {
        if !self.record_notification() {
            return;
        }
        let count = self.progress_count.fetch_add(1, Ordering::AcqRel) + 1;
        if count > self.max_notifications.load(Ordering::Acquire) {
            self.notification_overflowed.store(true, Ordering::Release);
            return;
        }
        let progress_token = serde_json::to_value(params.progress_token).unwrap_or(Value::Null);
        let event = McpProgress {
            progress_token,
            progress: params.progress,
            total: params.total,
            message: params.message,
        };
        if let Some(sender) = self
            .progress_tx
            .read()
            .expect("state lock is not poisoned")
            .as_ref()
        {
            let _ = sender.send(event);
        }
    }

    fn install_catalog(
        &self,
        epoch: u64,
        fingerprint: McpCatalogFingerprint,
    ) -> Result<(), McpError> {
        if self.epoch.load(Ordering::Acquire) != epoch {
            return Err(McpError::CatalogStale);
        }
        *self
            .active_catalog
            .write()
            .expect("state lock is not poisoned") = Some(fingerprint);
        self.stale.store(false, Ordering::Release);
        Ok(())
    }

    fn catalog_matches(&self, expected: &McpCatalogFingerprint) -> bool {
        !self.stale.load(Ordering::Acquire)
            && self
                .active_catalog
                .read()
                .expect("state lock is not poisoned")
                .as_ref()
                .is_some_and(|actual| actual == expected)
    }
}

#[derive(Clone)]
struct McpClientHandler {
    state: Arc<SessionState>,
}

impl ClientHandler for McpClientHandler {
    async fn on_tool_list_changed(&self, _context: rmcp::service::NotificationContext<RoleClient>) {
        if self.state.record_notification() {
            self.state.mark_catalog_changed();
        }
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) {
        self.state.mark_progress(params);
    }

    fn get_info(&self) -> ClientInfo {
        ClientInfo::default()
    }
}

#[async_trait]
trait McpBackend: Send + Sync {
    async fn list_tools_page(&self, cursor: Option<String>) -> Result<McpToolsPage, String>;
    async fn call_tool(
        &self,
        name: &str,
        arguments: JsonObject,
        call_id: &str,
    ) -> Result<CallToolResult, String>;
    async fn close(&self, timeout: Duration) -> Result<bool, String>;
    fn is_closed(&self) -> bool;
}

struct RmcpBackend {
    peer: Peer<RoleClient>,
    service: Mutex<Option<RunningService<RoleClient, McpClientHandler>>>,
    state: Arc<SessionState>,
}

#[async_trait]
impl McpBackend for RmcpBackend {
    async fn list_tools_page(&self, cursor: Option<String>) -> Result<McpToolsPage, String> {
        if self.is_closed() {
            return Err("service is closed".to_string());
        }
        let result = self
            .peer
            .list_tools(Some(PaginatedRequestParams::default().with_cursor(cursor)))
            .await
            .map_err(|error| error.to_string())?;
        Ok(McpToolsPage {
            tools: result.tools,
            next_cursor: result.next_cursor,
        })
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: JsonObject,
        call_id: &str,
    ) -> Result<CallToolResult, String> {
        if self.is_closed() {
            return Err("service is closed".to_string());
        }
        let token = ProgressToken(NumberOrString::String(call_id.to_string().into()));
        let mut params = CallToolRequestParams::new(name.to_string()).with_arguments(arguments);
        params.meta = Some(Meta::with_progress_token(token));
        self.peer
            .call_tool(params)
            .await
            .map_err(|error| error.to_string())
    }

    async fn close(&self, timeout: Duration) -> Result<bool, String> {
        let mut guard = self.service.lock().await;
        let Some(service) = guard.as_mut() else {
            return Ok(true);
        };
        let result = service
            .close_with_timeout(timeout)
            .await
            .map_err(|error| error.to_string())?;
        if result.is_some() {
            self.state.closed.store(true, Ordering::Release);
            *guard = None;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn is_closed(&self) -> bool {
        self.state.closed.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
struct McpToolsPage {
    tools: Vec<Tool>,
    next_cursor: Option<String>,
}

#[derive(Clone)]
pub struct McpClient {
    inner: Arc<McpClientInner>,
}

struct McpClientInner {
    config: McpClientConfig,
    state: Arc<SessionState>,
    backend: Arc<dyn McpBackend>,
    discovery: Mutex<()>,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpClient")
            .field("namespace", &self.inner.config.namespace())
            .field("closed", &self.inner.backend.is_closed())
            .finish_non_exhaustive()
    }
}

impl McpClient {
    /// Connect to a trusted host-configured stdio command without shell parsing.
    pub async fn from_stdio<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: McpClientConfig,
    ) -> Result<Self, McpError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut command = tokio::process::Command::new(program);
        command.args(args);
        let state = Arc::new(SessionState::default());
        state.configure(&config);
        let handler = McpClientHandler {
            state: state.clone(),
        };
        let transport = TokioChildProcess::new(command)
            .map_err(|error| McpError::Connect(error.to_string()))?;
        let service = handler
            .serve(transport)
            .await
            .map_err(|error| McpError::Connect(error.to_string()))?;
        let peer = service.peer().clone();
        let backend = Arc::new(RmcpBackend {
            peer,
            service: Mutex::new(Some(service)),
            state: state.clone(),
        });
        Ok(Self::from_parts(config, state, backend))
    }

    /// Connect to a streamable HTTP MCP endpoint after applying endpoint policy.
    pub async fn from_http(url: &str, config: McpClientConfig) -> Result<Self, McpError> {
        validate_http_endpoint(url, config.endpoint_policy())?;
        let state = Arc::new(SessionState::default());
        state.configure(&config);
        let handler = McpClientHandler {
            state: state.clone(),
        };
        let transport = StreamableHttpClientTransport::from_uri(url);
        let service = handler
            .serve(transport)
            .await
            .map_err(|error| McpError::Connect(error.to_string()))?;
        let peer = service.peer().clone();
        let backend = Arc::new(RmcpBackend {
            peer,
            service: Mutex::new(Some(service)),
            state: state.clone(),
        });
        Ok(Self::from_parts(config, state, backend))
    }

    /// Discover a bounded, immutable tool catalog. A list-changed notification
    /// invalidates all previously returned bindings until this method succeeds.
    pub async fn discover_tools(&self) -> Result<McpToolCatalog, McpError> {
        let _discovery = self.inner.discovery.lock().await;
        self.ensure_usable()?;
        self.inner.state.stale.store(true, Ordering::Release);
        let start_epoch = self.inner.state.epoch.load(Ordering::Acquire);
        let mut cursor = None;
        let mut seen_cursors = BTreeSet::new();
        let mut tools = Vec::new();
        for page in 0..self.inner.config.limits().max_pages() {
            let page_result = self
                .inner
                .backend
                .list_tools_page(cursor.clone())
                .await
                .map_err(McpError::ListTools)?;
            tools.extend(page_result.tools);
            if tools.len() > self.inner.config.limits().max_tools() {
                return Err(McpError::ToolLimitExceeded {
                    maximum: self.inner.config.limits().max_tools(),
                });
            }
            let Some(next_cursor) = page_result.next_cursor else {
                break;
            };
            if !seen_cursors.insert(next_cursor.clone()) {
                return Err(McpError::RepeatedCursor {
                    cursor: next_cursor,
                });
            }
            cursor = Some(next_cursor);
            if page + 1 == self.inner.config.limits().max_pages() {
                return Err(McpError::PageLimitExceeded {
                    maximum: self.inner.config.limits().max_pages(),
                });
            }
        }
        if self.inner.state.epoch.load(Ordering::Acquire) != start_epoch {
            return Err(McpError::CatalogStale);
        }
        let catalog = McpToolCatalog::from_remote(self, tools)?;
        self.inner
            .state
            .install_catalog(start_epoch, catalog.fingerprint().clone())?;
        Ok(catalog)
    }

    /// Subscribe to bounded progress notifications emitted by the MCP peer.
    pub fn subscribe_progress(&self) -> Result<broadcast::Receiver<McpProgress>, McpError> {
        self.ensure_usable()?;
        self.inner
            .state
            .progress_tx
            .read()
            .expect("state lock is not poisoned")
            .as_ref()
            .map(broadcast::Sender::subscribe)
            .ok_or(McpError::Closed)
    }

    pub fn catalog_is_stale(&self) -> bool {
        self.inner.state.stale.load(Ordering::Acquire)
    }

    /// Close the owned MCP service and wait at most the configured deadline.
    pub async fn close(&self) -> Result<(), McpError> {
        if self.inner.backend.is_closed() {
            return Ok(());
        }
        if !self
            .inner
            .backend
            .close(self.inner.config.limits().close_timeout())
            .await
            .map_err(McpError::Close)?
        {
            return Err(McpError::CloseTimeout);
        }
        Ok(())
    }

    pub(crate) fn config(&self) -> &McpClientConfig {
        &self.inner.config
    }

    pub(crate) fn exposed_name(&self, remote_name: &str) -> Result<String, McpError> {
        let exposed = match self.inner.config.namespace() {
            Some(namespace) => format!("{namespace}__{remote_name}"),
            None => remote_name.to_string(),
        };
        if exposed.is_empty()
            || exposed.len() > 128
            || !exposed
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(McpError::InvalidToolDefinition {
                tool: remote_name.to_string(),
                message: "name cannot be represented as a Siumai ToolSpec".to_string(),
            });
        }
        Ok(exposed)
    }

    pub(crate) async fn execute_bound_tool(
        &self,
        remote_name: &str,
        arguments: &Value,
        expected_catalog: &McpCatalogFingerprint,
        call_id: &str,
    ) -> Result<ToolOutcome, ToolExecutionError> {
        self.ensure_usable().map_err(|error| {
            ToolExecutionError::executor_failed(
                remote_name,
                error.to_string(),
                false,
                EffectCertainty::KnownNotApplied,
            )
        })?;
        if !self.inner.state.catalog_matches(expected_catalog) {
            return Err(ToolExecutionError::executor_failed(
                remote_name,
                McpError::CatalogFingerprintMismatch.to_string(),
                false,
                EffectCertainty::KnownNotApplied,
            ));
        }
        let arguments = arguments.as_object().cloned().ok_or_else(|| {
            ToolExecutionError::executor_failed(
                remote_name,
                "MCP tool arguments must be an object",
                false,
                EffectCertainty::KnownNotApplied,
            )
        })?;
        let result = self
            .inner
            .backend
            .call_tool(remote_name, arguments, call_id)
            .await
            .map_err(|_error| {
                ToolExecutionError::executor_failed(
                    remote_name,
                    "MCP transport outcome is unknown",
                    true,
                    EffectCertainty::Indeterminate,
                )
            })?;
        let value = serde_json::to_value(&result).map_err(|error| {
            ToolExecutionError::executor_failed(
                remote_name,
                error.to_string(),
                false,
                EffectCertainty::Applied,
            )
        })?;
        let encoded = serde_json::to_vec(&value).map_err(|error| {
            ToolExecutionError::executor_failed(
                remote_name,
                error.to_string(),
                false,
                EffectCertainty::Applied,
            )
        })?;
        if encoded.len() > self.inner.config.limits().max_result_bytes() {
            return Err(ToolExecutionError::executor_failed(
                remote_name,
                McpError::ResultLimitExceeded {
                    maximum: self.inner.config.limits().max_result_bytes(),
                }
                .to_string(),
                false,
                EffectCertainty::Applied,
            ));
        }
        if result.is_error.unwrap_or(false) {
            Ok(ToolOutcome::ExecutionFailed {
                message: "MCP server reported a tool error".to_string(),
                retryable: false,
                details: Some(value),
            })
        } else {
            Ok(ToolOutcome::Success { value })
        }
    }

    fn ensure_usable(&self) -> Result<(), McpError> {
        if self.inner.backend.is_closed() || self.inner.state.closed.load(Ordering::Acquire) {
            return Err(McpError::Closed);
        }
        if self
            .inner
            .state
            .notification_overflowed
            .load(Ordering::Acquire)
        {
            return Err(McpError::NotificationLimitExceeded);
        }
        Ok(())
    }

    fn from_parts(
        config: McpClientConfig,
        state: Arc<SessionState>,
        backend: Arc<dyn McpBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(McpClientInner {
                config,
                state,
                backend,
                discovery: Mutex::new(()),
            }),
        }
    }
}

fn validate_http_endpoint(endpoint: &str, policy: McpHttpEndpointPolicy) -> Result<(), McpError> {
    let parsed = url::Url::parse(endpoint).map_err(|_| McpError::InvalidEndpoint)?;
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(McpError::EndpointNotAllowed);
    }
    match parsed.scheme() {
        "https" => Ok(()),
        "http" if policy == McpHttpEndpointPolicy::AllowHttpLoopback => {
            let loopback = parsed.host_str().is_some_and(|host| {
                let literal = host
                    .strip_prefix('[')
                    .and_then(|value| value.strip_suffix(']'))
                    .unwrap_or(host);
                host.eq_ignore_ascii_case("localhost")
                    || literal
                        .parse::<std::net::IpAddr>()
                        .is_ok_and(|address| address.is_loopback())
            });
            if loopback {
                Ok(())
            } else {
                Err(McpError::EndpointNotAllowed)
            }
        }
        _ => Err(McpError::EndpointNotAllowed),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;

    use rmcp::model::{Content, ToolAnnotations};
    use serde_json::json;
    use siumai_runtime::tool::{ApprovalPolicy, EffectCertainty, ToolEffect};

    use super::*;

    struct MockBackend {
        pages: StdMutex<VecDeque<Result<McpToolsPage, String>>>,
        results: StdMutex<VecDeque<Result<CallToolResult, String>>>,
        calls: StdMutex<Vec<String>>,
        closed: AtomicBool,
    }

    impl MockBackend {
        fn new(pages: impl IntoIterator<Item = McpToolsPage>) -> Self {
            Self {
                pages: StdMutex::new(pages.into_iter().map(Ok).collect()),
                results: StdMutex::new(VecDeque::new()),
                calls: StdMutex::new(Vec::new()),
                closed: AtomicBool::new(false),
            }
        }

        fn push_result(&self, result: CallToolResult) {
            self.results
                .lock()
                .expect("test lock is not poisoned")
                .push_back(Ok(result));
        }
    }

    #[async_trait]
    impl McpBackend for MockBackend {
        async fn list_tools_page(&self, _cursor: Option<String>) -> Result<McpToolsPage, String> {
            self.pages
                .lock()
                .expect("test lock is not poisoned")
                .pop_front()
                .unwrap_or_else(|| {
                    Ok(McpToolsPage {
                        tools: Vec::new(),
                        next_cursor: None,
                    })
                })
        }

        async fn call_tool(
            &self,
            name: &str,
            _arguments: JsonObject,
            _call_id: &str,
        ) -> Result<CallToolResult, String> {
            self.calls
                .lock()
                .expect("test lock is not poisoned")
                .push(name.to_string());
            self.results
                .lock()
                .expect("test lock is not poisoned")
                .pop_front()
                .unwrap_or_else(|| Ok(CallToolResult::success(Vec::new())))
        }

        async fn close(&self, _timeout: Duration) -> Result<bool, String> {
            self.closed.store(true, Ordering::Release);
            Ok(true)
        }

        fn is_closed(&self) -> bool {
            self.closed.load(Ordering::Acquire)
        }
    }

    fn tool(name: &str) -> Tool {
        Tool::new(name.to_string(), "test tool", JsonObject::new())
    }

    fn page(tools: Vec<Tool>, next_cursor: Option<&str>) -> McpToolsPage {
        McpToolsPage {
            tools,
            next_cursor: next_cursor.map(str::to_string),
        }
    }

    fn client(config: McpClientConfig, backend: Arc<MockBackend>) -> McpClient {
        let state = Arc::new(SessionState::default());
        state.configure(&config);
        McpClient::from_parts(config, state, backend)
    }

    #[tokio::test]
    async fn remote_annotations_cannot_relax_host_policy() {
        let mut remote = tool("delete_record");
        let mut annotations = ToolAnnotations::default();
        annotations.read_only_hint = Some(true);
        remote.annotations = Some(annotations);
        let backend = Arc::new(MockBackend::new([page(vec![remote], None)]));
        let client = client(McpClientConfig::default(), backend);

        let catalog = client.discover_tools().await.unwrap();
        let binding = catalog.tool_set().get("delete_record").unwrap().clone();

        assert_eq!(binding.effect(), ToolEffect::SideEffecting);
        assert_eq!(binding.approval_policy(), ApprovalPolicy::Required);
        assert!(
            catalog.definitions()[0]
                .native_definition()
                .get("annotations")
                .is_some()
        );
    }

    #[tokio::test]
    async fn error_result_retains_rich_protocol_details_as_failure() {
        let backend = Arc::new(MockBackend::new([page(vec![tool("lookup")], None)]));
        backend.push_result(CallToolResult::error(vec![Content::text("not found")]));
        let client = client(McpClientConfig::default(), backend);
        let catalog = client.discover_tools().await.unwrap();

        let outcome = client
            .execute_bound_tool("lookup", &json!({}), catalog.fingerprint(), "call-1")
            .await
            .unwrap();

        match outcome {
            ToolOutcome::ExecutionFailed {
                retryable,
                details: Some(details),
                ..
            } => {
                assert!(!retryable);
                assert_eq!(details["isError"], true);
                assert!(details["content"].is_array());
            }
            other => panic!("expected typed MCP failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn list_changed_invalidates_old_bindings_before_dispatch() {
        let backend = Arc::new(MockBackend::new([page(vec![tool("lookup")], None)]));
        let client = client(McpClientConfig::default(), backend.clone());
        let catalog = client.discover_tools().await.unwrap();
        client.inner.state.mark_catalog_changed();

        let error = client
            .execute_bound_tool("lookup", &json!({}), catalog.fingerprint(), "call-1")
            .await
            .unwrap_err();

        assert_eq!(error.effect_certainty(), EffectCertainty::KnownNotApplied);
        assert!(backend.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn repeated_pagination_cursor_is_rejected() {
        let backend = Arc::new(MockBackend::new([
            page(vec![tool("one")], Some("same")),
            page(vec![tool("two")], Some("same")),
        ]));
        let client = client(McpClientConfig::default(), backend);

        let error = client.discover_tools().await.unwrap_err();
        assert!(matches!(error, McpError::RepeatedCursor { .. }));
    }

    #[tokio::test]
    async fn explicit_close_is_idempotent() {
        let backend = Arc::new(MockBackend::new([]));
        let client = client(McpClientConfig::default(), backend);

        client.close().await.unwrap();
        client.close().await.unwrap();
    }

    #[test]
    fn http_policy_allows_only_https_or_literal_loopback() {
        assert!(
            validate_http_endpoint("https://example.com/mcp", McpHttpEndpointPolicy::HttpsOnly)
                .is_ok()
        );
        assert!(
            validate_http_endpoint(
                "http://127.0.0.1:3000/mcp",
                McpHttpEndpointPolicy::HttpsOnly
            )
            .is_err()
        );
        assert!(
            validate_http_endpoint(
                "http://[::1]:3000/mcp",
                McpHttpEndpointPolicy::AllowHttpLoopback,
            )
            .is_ok()
        );
        assert!(
            validate_http_endpoint(
                "http://example.com/mcp",
                McpHttpEndpointPolicy::AllowHttpLoopback,
            )
            .is_err()
        );
        assert!(
            validate_http_endpoint(
                "https://user:secret@example.com/mcp",
                McpHttpEndpointPolicy::HttpsOnly,
            )
            .is_err()
        );
    }
}
