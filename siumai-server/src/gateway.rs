use std::sync::Arc;

use siumai_core::{
    CallOptions, Error, LanguageCallError, LanguageModel, LanguageRequest, LanguageResponse,
    LanguageStream, RouteId,
};
use siumai_runtime::tool::{
    ApprovalDecider, ApprovalDecisionFuture, ApprovalPolicyFingerprint, ApprovalRequest, ToolSet,
};
use siumai_runtime::{RunStream, RunTerminal, Runtime, StepOptions, ToolLoop, ToolOutcomePolicy};
use thiserror::Error;

use crate::ServerTrustContext;

/// Host policy for a required local tool call under an authenticated request.
pub trait TrustedApprovalDecider: Send + Sync + 'static {
    fn decide<'a>(
        &'a self,
        trust: &'a ServerTrustContext,
        request: &'a ApprovalRequest,
    ) -> ApprovalDecisionFuture<'a>;

    fn fingerprint(&self) -> &ApprovalPolicyFingerprint;
}

/// Server gateway configuration or runtime failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ServerGatewayError {
    #[error("local tool execution is disabled for this route")]
    LocalToolsDisabled,
    #[error("authenticated request route does not match the trusted tool route")]
    TrustRouteMismatch,
    #[error("configured tool route does not match the model's Registry route")]
    ModelRouteMismatch,
    #[error(transparent)]
    LanguageCall(#[from] LanguageCallError),
    #[error(transparent)]
    Runtime(#[from] Error),
}

#[derive(Clone)]
struct LocalToolRoute {
    route: RouteId,
    tools: ToolSet,
    approvals: Arc<dyn TrustedApprovalDecider>,
}

struct BoundApprovalDecider {
    trust: ServerTrustContext,
    inner: Arc<dyn TrustedApprovalDecider>,
}

impl ApprovalDecider for BoundApprovalDecider {
    fn decide<'a>(&'a self, request: &'a ApprovalRequest) -> ApprovalDecisionFuture<'a> {
        self.inner.decide(&self.trust, request)
    }

    fn fingerprint(&self) -> &ApprovalPolicyFingerprint {
        self.inner.fingerprint()
    }
}

/// Clone-cheap server route over one model and one shared runtime.
///
/// `generate` and `stream` are always single-call routes. The host must call
/// [`ServerGateway::enable_local_tools`] and then use `run`/`run_stream` to
/// enter the explicit tool-loop surface.
#[derive(Clone)]
pub struct ServerGateway {
    runtime: Runtime,
    model: Arc<dyn LanguageModel>,
    step_options: StepOptions,
    local_tools: Option<LocalToolRoute>,
    tool_outcome_policy: ToolOutcomePolicy,
}

impl std::fmt::Debug for ServerGateway {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ServerGateway")
            .field(
                "target",
                &siumai_runtime::ModelTarget::from_model(&self.model),
            )
            .field("local_tools_enabled", &self.local_tools.is_some())
            .field(
                "local_tool_route",
                &self.local_tools.as_ref().map(|route| &route.route),
            )
            .finish_non_exhaustive()
    }
}

impl ServerGateway {
    pub fn new(model: Arc<dyn LanguageModel>) -> Self {
        Self {
            runtime: Runtime::default(),
            model,
            step_options: StepOptions::default(),
            local_tools: None,
            tool_outcome_policy: ToolOutcomePolicy::default(),
        }
    }

    pub fn with_runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = runtime;
        self
    }

    pub fn with_step_options(mut self, options: StepOptions) -> Self {
        self.step_options = options;
        self
    }

    /// Install trusted host bindings for this route.
    ///
    /// Client-supplied tool definitions remain untrusted request data. The
    /// runtime rejects collisions with this frozen catalog before dispatch.
    pub fn enable_local_tools(
        mut self,
        route: RouteId,
        tools: ToolSet,
        approvals: Arc<dyn TrustedApprovalDecider>,
    ) -> Self {
        self.local_tools = Some(LocalToolRoute {
            route,
            tools,
            approvals,
        });
        self
    }

    pub fn with_tool_outcome_policy(mut self, policy: ToolOutcomePolicy) -> Self {
        self.tool_outcome_policy = policy;
        self
    }

    pub fn local_tools_enabled(&self) -> bool {
        self.local_tools.is_some()
    }

    pub fn local_tool_route(&self) -> Option<&RouteId> {
        self.local_tools.as_ref().map(|route| &route.route)
    }

    /// Execute exactly one model call. Local bindings are never consulted.
    pub async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, ServerGatewayError> {
        let options = resolve_options(options)?;
        self.runtime
            .generate(
                self.model.as_ref(),
                request,
                self.step_options.clone(),
                options,
            )
            .await
            .map_err(Into::into)
    }

    /// Establish exactly one provider stream. Local bindings are never consulted.
    pub async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, ServerGatewayError> {
        let options = resolve_options(options)?;
        self.runtime
            .stream(
                self.model.as_ref(),
                request,
                self.step_options.clone(),
                options,
            )
            .await
            .map_err(Into::into)
    }

    /// Run the shared runtime's explicit tool loop for an enabled route.
    pub async fn run(
        &self,
        trust: &ServerTrustContext,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<RunTerminal, ServerGatewayError> {
        let options = resolve_options(options)?;
        self.tool_loop(trust)?
            .run(request, options)
            .await
            .map_err(Into::into)
    }

    /// Stream the shared runtime's explicit tool loop for an enabled route.
    pub async fn run_stream(
        &self,
        trust: &ServerTrustContext,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<RunStream, ServerGatewayError> {
        let options = resolve_options(options)?;
        self.tool_loop(trust)?
            .stream(request, options)
            .await
            .map_err(Into::into)
    }

    fn tool_loop(&self, trust: &ServerTrustContext) -> Result<ToolLoop, ServerGatewayError> {
        let route = self
            .local_tools
            .as_ref()
            .ok_or(ServerGatewayError::LocalToolsDisabled)?;
        if trust.route() != &route.route {
            return Err(ServerGatewayError::TrustRouteMismatch);
        }
        if siumai_runtime::ModelTarget::from_model(&self.model)
            .route()
            .is_some_and(|model_route| model_route != &route.route)
        {
            return Err(ServerGatewayError::ModelRouteMismatch);
        }

        let approvals: Arc<dyn ApprovalDecider> = Arc::new(BoundApprovalDecider {
            trust: trust.clone(),
            inner: Arc::clone(&route.approvals),
        });
        Ok(self
            .runtime
            .tool_loop(self.model.clone(), route.tools.clone())
            .with_step_options(self.step_options.clone())
            .with_approval_decider(approvals)
            .with_outcome_policy(self.tool_outcome_policy))
    }
}

fn resolve_options(options: CallOptions) -> Result<CallOptions, ServerGatewayError> {
    options
        .resolve_deadline()
        .map_err(Error::from)
        .map_err(ServerGatewayError::from)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use siumai_core::stream::established_stream;
    use siumai_core::{
        ContentPart, LanguageCompletionReason, LanguageStream, LanguageStreamEvent, Message,
        MessageRole, Model, ModelDescriptor, ModelFamily, ModelId, ProviderId, StreamTerminal,
        Usage,
    };
    use siumai_runtime::approval::TrustIdentity;
    use siumai_runtime::tool::{ApprovalDecision, ApprovalDecisionError};

    use super::*;

    struct NeverCalledModel {
        descriptor: ModelDescriptor,
    }

    struct RecordingModel {
        descriptor: ModelDescriptor,
        generate_options: Mutex<Vec<CallOptions>>,
        stream_options: Mutex<Vec<CallOptions>>,
    }

    struct AwaitExternal {
        fingerprint: ApprovalPolicyFingerprint,
    }

    impl AwaitExternal {
        fn new() -> Self {
            Self {
                fingerprint: ApprovalPolicyFingerprint::new("test.server-approval.v1").unwrap(),
            }
        }
    }

    impl TrustedApprovalDecider for AwaitExternal {
        fn decide<'a>(
            &'a self,
            _trust: &'a ServerTrustContext,
            _request: &'a ApprovalRequest,
        ) -> ApprovalDecisionFuture<'a> {
            Box::pin(async {
                Ok::<ApprovalDecision, ApprovalDecisionError>(ApprovalDecision::AwaitExternal)
            })
        }

        fn fingerprint(&self) -> &ApprovalPolicyFingerprint {
            &self.fingerprint
        }
    }

    impl NeverCalledModel {
        fn new() -> Self {
            Self {
                descriptor: ModelDescriptor::new(
                    ProviderId::new("test").unwrap(),
                    ModelId::new("never-called").unwrap(),
                    ModelFamily::Language,
                ),
            }
        }
    }

    impl RecordingModel {
        fn new() -> Self {
            Self {
                descriptor: ModelDescriptor::new(
                    ProviderId::new("test").unwrap(),
                    ModelId::new("recording").unwrap(),
                    ModelFamily::Language,
                ),
                generate_options: Mutex::new(Vec::new()),
                stream_options: Mutex::new(Vec::new()),
            }
        }
    }

    impl Model for NeverCalledModel {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }
    }

    impl Model for RecordingModel {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }
    }

    #[async_trait]
    impl LanguageModel for RecordingModel {
        async fn generate(
            &self,
            _request: LanguageRequest,
            options: CallOptions,
        ) -> Result<LanguageResponse, LanguageCallError> {
            self.generate_options.lock().unwrap().push(options);
            Ok(LanguageResponse::completed(
                vec![ContentPart::Text {
                    text: "ok".to_string(),
                }],
                LanguageCompletionReason::Stop,
                Usage::default(),
            )
            .unwrap())
        }

        async fn stream(
            &self,
            _request: LanguageRequest,
            options: CallOptions,
        ) -> Result<LanguageStream, Error> {
            self.stream_options.lock().unwrap().push(options.clone());
            let cancellation = options.cancellation().clone();
            let response = LanguageResponse::completed(
                vec![ContentPart::Text {
                    text: "done".to_string(),
                }],
                LanguageCompletionReason::Stop,
                Usage::default(),
            )
            .unwrap();
            Ok(established_stream(cancellation, |_| {
                futures::stream::iter([Ok(LanguageStreamEvent::Terminal(
                    StreamTerminal::Completed {
                        response: Box::new(response),
                    },
                ))])
            }))
        }
    }

    #[async_trait]
    impl LanguageModel for NeverCalledModel {
        async fn generate(
            &self,
            _request: LanguageRequest,
            _options: CallOptions,
        ) -> Result<LanguageResponse, LanguageCallError> {
            panic!("disabled route must not invoke the model")
        }

        async fn stream(
            &self,
            _request: LanguageRequest,
            _options: CallOptions,
        ) -> Result<LanguageStream, Error> {
            panic!("disabled route must not invoke the model")
        }
    }

    #[tokio::test]
    async fn local_tool_route_is_default_deny() {
        let gateway = ServerGateway::new(Arc::new(NeverCalledModel::new()));
        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);
        let trust = ServerTrustContext::new(
            TrustIdentity::new("issuer", "audience", "subject", "tenant").unwrap(),
            RouteId::new("tools").unwrap(),
        );

        let error = gateway
            .run(&trust, request, CallOptions::default())
            .await
            .unwrap_err();

        assert!(matches!(error, ServerGatewayError::LocalToolsDisabled));
    }

    #[tokio::test]
    async fn gateway_preserves_resolved_call_intent_for_single_and_runtime_routes() {
        let model = Arc::new(RecordingModel::new());
        let route = RouteId::new("tools").unwrap();
        let gateway = ServerGateway::new(model.clone()).enable_local_tools(
            route.clone(),
            ToolSet::default(),
            Arc::new(AwaitExternal::new()),
        );
        let trust = ServerTrustContext::new(
            TrustIdentity::new("issuer", "audience", "subject", "tenant").unwrap(),
            route,
        );
        let options = || {
            CallOptions::default()
                .with_timeout(Duration::from_secs(1))
                .unwrap()
                .with_max_attempts(2)
                .unwrap()
        };
        let generate_options = options();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let generate_invoked_at = Instant::now();

        gateway
            .generate(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                generate_options,
            )
            .await
            .unwrap();

        {
            let generate_options = model.generate_options.lock().unwrap();
            assert_eq!(generate_options.len(), 1);
            assert_eq!(generate_options[0].timeout(), None);
            assert_eq!(generate_options[0].retry().maximum_attempts(), Some(2));
            assert!(
                generate_options[0].deadline().unwrap()
                    >= generate_invoked_at + Duration::from_millis(925)
            );
        }

        let run_options = options();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let run_invoked_at = Instant::now();
        let terminal = gateway
            .run(
                &trust,
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                run_options,
            )
            .await
            .unwrap();
        assert!(matches!(terminal, RunTerminal::Completed { .. }));
        let stream_options = model.stream_options.lock().unwrap();
        assert_eq!(stream_options.len(), 1);
        assert_eq!(stream_options[0].timeout(), None);
        assert_eq!(stream_options[0].retry().maximum_attempts(), Some(2));
        assert!(
            stream_options[0].deadline().unwrap() >= run_invoked_at + Duration::from_millis(925)
        );
    }

    #[tokio::test]
    async fn gateway_resolves_call_options_before_route_policy_work() {
        let gateway = ServerGateway::new(Arc::new(NeverCalledModel::new())).enable_local_tools(
            RouteId::new("tools").unwrap(),
            ToolSet::default(),
            Arc::new(AwaitExternal::new()),
        );
        let wrong_trust = ServerTrustContext::new(
            TrustIdentity::new("issuer", "audience", "subject", "tenant").unwrap(),
            RouteId::new("other-route").unwrap(),
        );

        let error = gateway
            .run(
                &wrong_trust,
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                CallOptions::default().with_timeout(Duration::MAX).unwrap(),
            )
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            ServerGatewayError::Runtime(error) if error.kind() == siumai_core::ErrorKind::InvalidInput
        ));
    }

    #[tokio::test]
    async fn trusted_tool_route_rejects_cross_route_context() {
        let gateway = ServerGateway::new(Arc::new(NeverCalledModel::new())).enable_local_tools(
            RouteId::new("tools").unwrap(),
            ToolSet::default(),
            Arc::new(AwaitExternal::new()),
        );
        let trust = ServerTrustContext::new(
            TrustIdentity::new("issuer", "audience", "subject", "tenant").unwrap(),
            RouteId::new("other-route").unwrap(),
        );
        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);

        let error = gateway
            .run(&trust, request, CallOptions::default())
            .await
            .unwrap_err();

        assert!(matches!(error, ServerGatewayError::TrustRouteMismatch));
    }
}
