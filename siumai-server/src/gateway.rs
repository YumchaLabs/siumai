use std::sync::Arc;

use siumai_core::{
    CallOptions, Error, LanguageModel, LanguageRequest, LanguageResponse, LanguageStream,
};
use siumai_runtime::tool::{ApprovalDecider, ToolSet};
use siumai_runtime::{RunStream, RunTerminal, Runtime, StepOptions, ToolLoop, ToolOutcomePolicy};
use thiserror::Error;

/// Server gateway configuration or runtime failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ServerGatewayError {
    #[error("local tool execution is disabled for this route")]
    LocalToolsDisabled,
    #[error(transparent)]
    Runtime(#[from] Error),
}

#[derive(Clone)]
struct LocalToolRoute {
    tools: ToolSet,
    approvals: Arc<dyn ApprovalDecider>,
    outcome_policy: ToolOutcomePolicy,
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
        tools: ToolSet,
        approvals: Arc<dyn ApprovalDecider>,
    ) -> Self {
        self.local_tools = Some(LocalToolRoute {
            tools,
            approvals,
            outcome_policy: ToolOutcomePolicy::default(),
        });
        self
    }

    pub fn with_tool_outcome_policy(mut self, policy: ToolOutcomePolicy) -> Self {
        if let Some(route) = self.local_tools.as_mut() {
            route.outcome_policy = policy;
        }
        self
    }

    pub fn local_tools_enabled(&self) -> bool {
        self.local_tools.is_some()
    }

    /// Execute exactly one model call. Local bindings are never consulted.
    pub async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, ServerGatewayError> {
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
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<RunTerminal, ServerGatewayError> {
        self.tool_loop()?
            .run(request, options)
            .await
            .map_err(Into::into)
    }

    /// Stream the shared runtime's explicit tool loop for an enabled route.
    pub async fn run_stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<RunStream, ServerGatewayError> {
        self.tool_loop()?
            .stream(request, options)
            .await
            .map_err(Into::into)
    }

    fn tool_loop(&self) -> Result<ToolLoop, ServerGatewayError> {
        let route = self
            .local_tools
            .as_ref()
            .ok_or(ServerGatewayError::LocalToolsDisabled)?;
        Ok(self
            .runtime
            .tool_loop(self.model.clone(), route.tools.clone())
            .with_step_options(self.step_options.clone())
            .with_approval_decider(route.approvals.clone())
            .with_outcome_policy(route.outcome_policy))
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use siumai_core::{
        LanguageStream, Message, MessageRole, Model, ModelDescriptor, ModelFamily, ModelId,
        ProviderId,
    };

    use super::*;

    struct NeverCalledModel {
        descriptor: ModelDescriptor,
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

    impl Model for NeverCalledModel {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }
    }

    #[async_trait]
    impl LanguageModel for NeverCalledModel {
        async fn generate(
            &self,
            _request: LanguageRequest,
            _options: CallOptions,
        ) -> Result<LanguageResponse, Error> {
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

        let error = gateway
            .run(request, CallOptions::default())
            .await
            .unwrap_err();

        assert!(matches!(error, ServerGatewayError::LocalToolsDisabled));
    }
}
