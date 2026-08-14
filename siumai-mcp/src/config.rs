use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::time::Duration;

use siumai_runtime::tool::{ApprovalPolicy, RecoveryPolicy, ToolConcurrency, ToolEffect};

use crate::McpError;

/// Resource limits enforced by one MCP client session.
#[derive(Debug, Clone)]
pub struct McpLimits {
    max_pages: NonZeroUsize,
    max_tools: NonZeroUsize,
    max_message_bytes: NonZeroUsize,
    max_schema_bytes: NonZeroUsize,
    max_result_bytes: NonZeroUsize,
    progress_queue_capacity: NonZeroUsize,
    close_timeout: Duration,
}

impl Default for McpLimits {
    fn default() -> Self {
        Self {
            max_pages: NonZeroUsize::new(32).expect("constant is non-zero"),
            max_tools: NonZeroUsize::new(256).expect("constant is non-zero"),
            max_message_bytes: NonZeroUsize::new(8 * 1024 * 1024).expect("constant is non-zero"),
            max_schema_bytes: NonZeroUsize::new(256 * 1024).expect("constant is non-zero"),
            max_result_bytes: NonZeroUsize::new(2 * 1024 * 1024).expect("constant is non-zero"),
            progress_queue_capacity: NonZeroUsize::new(128).expect("constant is non-zero"),
            close_timeout: Duration::from_secs(5),
        }
    }
}

impl McpLimits {
    pub fn with_max_pages(mut self, value: usize) -> Result<Self, McpError> {
        self.max_pages = non_zero("max_pages", value)?;
        Ok(self)
    }

    pub fn with_max_tools(mut self, value: usize) -> Result<Self, McpError> {
        self.max_tools = non_zero("max_tools", value)?;
        Ok(self)
    }

    /// Set the maximum raw size of one JSON-RPC message, HTTP body, or SSE event.
    pub fn with_max_message_bytes(mut self, value: usize) -> Result<Self, McpError> {
        self.max_message_bytes = non_zero("max_message_bytes", value)?;
        Ok(self)
    }

    pub fn with_max_schema_bytes(mut self, value: usize) -> Result<Self, McpError> {
        self.max_schema_bytes = non_zero("max_schema_bytes", value)?;
        Ok(self)
    }

    pub fn with_max_result_bytes(mut self, value: usize) -> Result<Self, McpError> {
        self.max_result_bytes = non_zero("max_result_bytes", value)?;
        Ok(self)
    }

    pub fn with_progress_queue_capacity(mut self, value: usize) -> Result<Self, McpError> {
        self.progress_queue_capacity = non_zero("progress_queue_capacity", value)?;
        Ok(self)
    }

    pub fn with_close_timeout(mut self, value: Duration) -> Result<Self, McpError> {
        if value.is_zero() {
            return Err(McpError::InvalidLimit {
                name: "close_timeout",
            });
        }
        self.close_timeout = value;
        Ok(self)
    }

    pub(crate) fn max_pages(&self) -> usize {
        self.max_pages.get()
    }

    pub(crate) fn max_tools(&self) -> usize {
        self.max_tools.get()
    }

    pub(crate) fn max_message_bytes(&self) -> usize {
        self.max_message_bytes.get()
    }

    pub(crate) fn max_schema_bytes(&self) -> usize {
        self.max_schema_bytes.get()
    }

    pub(crate) fn max_result_bytes(&self) -> usize {
        self.max_result_bytes.get()
    }

    pub(crate) fn progress_queue_capacity(&self) -> usize {
        self.progress_queue_capacity.get()
    }

    pub(crate) fn close_timeout(&self) -> Duration {
        self.close_timeout
    }
}

fn non_zero(name: &'static str, value: usize) -> Result<NonZeroUsize, McpError> {
    NonZeroUsize::new(value).ok_or(McpError::InvalidLimit { name })
}

/// Policy for streamable HTTP MCP endpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum McpHttpEndpointPolicy {
    /// Require HTTPS for every remote endpoint.
    #[default]
    HttpsOnly,
    /// Additionally allow plain HTTP for literal loopback hosts.
    AllowHttpLoopback,
}

/// Trusted host policy for one remote MCP tool.
///
/// MCP annotations are descriptive hints and never override these values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McpToolPolicy {
    effect: ToolEffect,
    concurrency: ToolConcurrency,
    approval: ApprovalPolicy,
    recovery: RecoveryPolicy,
}

impl Default for McpToolPolicy {
    fn default() -> Self {
        Self {
            effect: ToolEffect::SideEffecting,
            concurrency: ToolConcurrency::Sequential,
            approval: ApprovalPolicy::Required,
            recovery: RecoveryPolicy::NeverReplay,
        }
    }
}

impl McpToolPolicy {
    pub fn with_effect(mut self, value: ToolEffect) -> Self {
        self.effect = value;
        self
    }

    pub fn with_concurrency(mut self, value: ToolConcurrency) -> Self {
        self.concurrency = value;
        self
    }

    pub fn with_approval(mut self, value: ApprovalPolicy) -> Self {
        self.approval = value;
        self
    }

    pub fn with_recovery(mut self, value: RecoveryPolicy) -> Self {
        self.recovery = value;
        self
    }

    pub(crate) fn effect(self) -> ToolEffect {
        self.effect
    }

    pub(crate) fn concurrency(self) -> ToolConcurrency {
        self.concurrency
    }

    pub(crate) fn approval(self) -> ApprovalPolicy {
        self.approval
    }

    pub(crate) fn recovery(self) -> RecoveryPolicy {
        self.recovery
    }
}

/// Configuration for one connected MCP session.
#[derive(Debug, Clone, Default)]
pub struct McpClientConfig {
    namespace: Option<String>,
    limits: McpLimits,
    endpoint_policy: McpHttpEndpointPolicy,
    default_tool_policy: McpToolPolicy,
    tool_policies: BTreeMap<String, McpToolPolicy>,
}

impl McpClientConfig {
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Result<Self, McpError> {
        let namespace = namespace.into();
        if namespace.is_empty()
            || namespace.len() > 48
            || !namespace
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(McpError::InvalidNamespace);
        }
        self.namespace = Some(namespace);
        Ok(self)
    }

    pub fn with_limits(mut self, limits: McpLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_http_endpoint_policy(mut self, policy: McpHttpEndpointPolicy) -> Self {
        self.endpoint_policy = policy;
        self
    }

    pub fn with_default_tool_policy(mut self, policy: McpToolPolicy) -> Self {
        self.default_tool_policy = policy;
        self
    }

    pub fn with_tool_policy(
        mut self,
        remote_name: impl Into<String>,
        policy: McpToolPolicy,
    ) -> Result<Self, McpError> {
        let remote_name = remote_name.into();
        if remote_name.is_empty() {
            return Err(McpError::InvalidRemoteToolName);
        }
        self.tool_policies.insert(remote_name, policy);
        Ok(self)
    }

    pub(crate) fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    pub(crate) fn limits(&self) -> &McpLimits {
        &self.limits
    }

    pub(crate) fn endpoint_policy(&self) -> McpHttpEndpointPolicy {
        self.endpoint_policy
    }

    pub(crate) fn tool_policy(&self, remote_name: &str) -> McpToolPolicy {
        self.tool_policies
            .get(remote_name)
            .copied()
            .unwrap_or(self.default_tool_policy)
    }
}
