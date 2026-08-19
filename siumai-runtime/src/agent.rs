//! Reusable, clone-cheap facade over the shared tool-loop runtime.

use std::sync::Arc;

use siumai_core::{
    CallOptions, Error, LanguageInput, LanguageModel, LanguageRequest, Message, MessageRole,
};
use thiserror::Error;

use crate::tool::{ToolBinding, ToolSet, ToolSetBuildError};
use crate::{ProjectionPolicy, RunStream, RunTerminal, StepModelSelector, ToolLoop};

/// Stateless facade for repeatedly running one configured [`ToolLoop`].
#[derive(Clone)]
pub struct Agent {
    tool_loop: ToolLoop,
    instructions: Arc<[Message]>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Agent")
            .field("tool_loop", &self.tool_loop)
            .field("instructions", &self.instructions.len())
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Construct an agent from one concrete language model and no local tools.
    pub fn new<M>(model: M) -> Self
    where
        M: LanguageModel + 'static,
    {
        Self::from_shared_model(Arc::new(model))
    }

    /// Construct an agent from an already erased or Registry-resolved model.
    pub fn from_shared_model(model: Arc<dyn LanguageModel>) -> Self {
        Self::from_tool_loop(ToolLoop::new(model, ToolSet::default()))
    }

    /// Use an advanced ToolLoop configuration without mirroring its interface.
    pub fn from_tool_loop(tool_loop: ToolLoop) -> Self {
        Self {
            tool_loop,
            instructions: Arc::from([]),
        }
    }

    /// Prepend one reusable system instruction to every independent run.
    pub fn with_instructions(self, instructions: impl Into<String>) -> Self {
        Self {
            instructions: Arc::from([Message::text(MessageRole::System, instructions.into())]),
            ..self
        }
    }

    /// Install reusable system/developer messages for every independent run.
    pub fn with_instruction_messages<I>(self, instructions: I) -> Result<Self, AgentConfigError>
    where
        I: IntoIterator<Item = Message>,
    {
        let instructions = instructions.into_iter().collect::<Vec<_>>();
        if let Some((index, message)) = instructions.iter().enumerate().find(|(_, message)| {
            !matches!(message.role(), MessageRole::System | MessageRole::Developer)
        }) {
            return Err(AgentConfigError::InvalidInstructionRole {
                index,
                role: message.role(),
            });
        }
        Ok(Self {
            instructions: Arc::from(instructions),
            ..self
        })
    }

    /// Replace the trusted local tool set used by this agent.
    pub fn with_tool_set(self, tools: ToolSet) -> Self {
        Self {
            tool_loop: self.tool_loop.with_tools(tools),
            ..self
        }
    }

    /// Build and install a trusted local tool set.
    pub fn with_tools<I>(self, bindings: I) -> Result<Self, ToolSetBuildError>
    where
        I: IntoIterator<Item = ToolBinding>,
    {
        Ok(self.with_tool_set(ToolSet::from_bindings(bindings)?))
    }

    pub fn with_model_selector<S>(self, selector: S) -> Self
    where
        S: StepModelSelector,
    {
        Self {
            tool_loop: self.tool_loop.with_model_selector(selector),
            ..self
        }
    }

    pub fn with_projection_policy(self, policy: ProjectionPolicy) -> Self {
        Self {
            tool_loop: self.tool_loop.with_projection_policy(policy),
            ..self
        }
    }

    /// Stream one independent run with default call options.
    pub async fn stream<I>(&self, input: I) -> Result<RunStream, Error>
    where
        I: Into<LanguageInput>,
    {
        self.stream_with(input, CallOptions::default()).await
    }

    /// Stream one independent run with explicit call options.
    pub async fn stream_with<I>(&self, input: I, options: CallOptions) -> Result<RunStream, Error>
    where
        I: Into<LanguageInput>,
    {
        self.tool_loop
            .stream(self.prepare_request(input.into()), options)
            .await
    }

    /// Collect one independent run through the same streaming execution path.
    pub async fn run<I>(&self, input: I) -> Result<RunTerminal, Error>
    where
        I: Into<LanguageInput>,
    {
        self.run_with(input, CallOptions::default()).await
    }

    /// Collect one independent run with explicit call options.
    pub async fn run_with<I>(&self, input: I, options: CallOptions) -> Result<RunTerminal, Error>
    where
        I: Into<LanguageInput>,
    {
        self.tool_loop
            .run(self.prepare_request(input.into()), options)
            .await
    }

    fn prepare_request(&self, input: LanguageInput) -> LanguageRequest {
        let mut request = input.into_request();
        if !self.instructions.is_empty() {
            let mut messages = Vec::with_capacity(self.instructions.len() + request.messages.len());
            messages.extend(self.instructions.iter().cloned());
            messages.append(&mut request.messages);
            request.messages = messages;
        }
        request
    }
}

/// Invalid reusable Agent configuration.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AgentConfigError {
    #[error("agent instruction {index} uses unsupported role {role:?}")]
    InvalidInstructionRole { index: usize, role: MessageRole },
}
