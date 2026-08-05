//! Canonical one-model-step execution.
//!
//! This module deliberately has no tool catalog or run report. It is the
//! smallest execution boundary shared by direct calls and [`StepEngine`].

use siumai_core::{
    CallOptions, Error, LanguageModel, LanguageRequest, LanguageResponse, LanguageStream,
};

use crate::call::validate_request;
use crate::{Runtime, StepOptions};

/// One validated call against one already-selected language model.
///
/// Provider-returned tool calls remain response/stream data. Local tool
/// execution belongs exclusively to the higher-level `StepEngine`.
pub(crate) struct SingleStep<'a, M: LanguageModel + ?Sized> {
    runtime: &'a Runtime,
    model: &'a M,
    step_options: &'a StepOptions,
}

impl<'a, M> SingleStep<'a, M>
where
    M: LanguageModel + ?Sized,
{
    pub(crate) fn new(runtime: &'a Runtime, model: &'a M, step_options: &'a StepOptions) -> Self {
        Self {
            runtime,
            model,
            step_options,
        }
    }

    /// Perform exactly one non-streaming provider call.
    pub(crate) async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        validate_request(&request)?;
        self.model
            .generate(
                request,
                self.runtime
                    .prepare_options(self.model, self.step_options, options),
            )
            .await
    }

    /// Establish exactly one provider stream.
    pub(crate) async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        validate_request(&request)?;
        self.model
            .stream(
                request,
                self.runtime
                    .prepare_options(self.model, self.step_options, options),
            )
            .await
    }
}
