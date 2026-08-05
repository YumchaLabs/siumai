//! Structured-output execution projected from the shared step engine.

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use serde::de::DeserializeOwned;
use siumai_core::{CallOptions, Error, ErrorKind, LanguageModel, LanguageRequest};

use crate::engine::{StepEngine, ToolHandling};
use crate::run::established_run_stream;
use crate::tool::{ExternalApprovalDecider, ToolSet};
use crate::{
    OutputDescriptor, RunReport, RunTerminal, Runtime, StepOptions, StructuredOutputAttemptKind,
    StructuredOutputError, StructuredOutputResult, ToolOutcomePolicy,
};

/// One validated structured value plus the complete shared-engine report.
#[derive(Debug)]
pub struct StructuredOutputRunResult<T> {
    output: StructuredOutputResult<T>,
    report: RunReport,
    initial_failure: Option<StructuredOutputError>,
}

impl<T> StructuredOutputRunResult<T> {
    pub fn output(&self) -> &StructuredOutputResult<T> {
        &self.output
    }

    pub fn report(&self) -> &RunReport {
        &self.report
    }

    pub fn initial_failure(&self) -> Option<&StructuredOutputError> {
        self.initial_failure.as_ref()
    }

    pub fn was_repaired(&self) -> bool {
        self.initial_failure.is_some()
    }

    pub fn into_parts(
        self,
    ) -> (
        StructuredOutputResult<T>,
        RunReport,
        Option<StructuredOutputError>,
    ) {
        (self.output, self.report, self.initial_failure)
    }
}

/// Failure while executing or validating a structured-output run.
#[derive(Debug)]
#[non_exhaustive]
pub enum StructuredOutputRunError {
    /// Validation or the first model handshake failed before a run stream existed.
    Establishment(Error),
    /// An established shared-engine run reached a non-completed terminal.
    Runtime(Box<RunTerminal>),
    /// The model returned a final response that failed strict output validation.
    Validation {
        error: StructuredOutputError,
        report: Box<RunReport>,
    },
}

impl StructuredOutputRunError {
    pub fn report(&self) -> Option<&RunReport> {
        match self {
            Self::Establishment(_) => None,
            Self::Runtime(terminal) => terminal.report(),
            Self::Validation { report, .. } => Some(report),
        }
    }

    pub fn output_error(&self) -> Option<&StructuredOutputError> {
        match self {
            Self::Validation { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for StructuredOutputRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Establishment(error) => {
                write!(formatter, "structured-output run failed: {error}")
            }
            Self::Runtime(_) => formatter
                .write_str("structured-output run reached a non-completed runtime terminal"),
            Self::Validation { error, .. } => error.fmt(formatter),
        }
    }
}

impl StdError for StructuredOutputRunError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Establishment(error) => Some(error),
            Self::Validation { error, .. } => Some(error),
            Self::Runtime(_) => None,
        }
    }
}

/// Clone-cheap structured-output execution configured for one language model.
#[derive(Clone)]
pub struct StructuredOutputRunner<T> {
    runtime: crate::Runtime,
    model: Arc<dyn LanguageModel>,
    descriptor: OutputDescriptor<T>,
    step_options: StepOptions,
}

impl<T> fmt::Debug for StructuredOutputRunner<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StructuredOutputRunner")
            .field(
                "target",
                &crate::ModelTarget::from_model(self.model.as_ref()),
            )
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl<T> StructuredOutputRunner<T> {
    pub fn new(model: Arc<dyn LanguageModel>, descriptor: OutputDescriptor<T>) -> Self {
        Self {
            runtime: crate::Runtime::default(),
            model,
            descriptor,
            step_options: StepOptions::default(),
        }
    }

    pub fn with_runtime(mut self, runtime: crate::Runtime) -> Self {
        self.runtime = runtime;
        self
    }

    pub fn with_step_options(mut self, options: StepOptions) -> Self {
        self.step_options = options;
        self
    }

    pub fn descriptor(&self) -> &OutputDescriptor<T> {
        &self.descriptor
    }

    /// Generate one strict structured value, optionally spending one bounded repair step.
    pub async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<StructuredOutputRunResult<T>, StructuredOutputRunError>
    where
        T: DeserializeOwned,
    {
        if !request.tools.is_empty() || request.tool_choice.is_some() {
            return Err(StructuredOutputRunError::Establishment(Error::new(
                ErrorKind::InvalidInput,
                "structured-output runner is tool-free; use ToolLoop for executable tools",
            )));
        }

        let original_request = request.clone();
        let shared_deadline = self
            .runtime
            .run_budget()
            .deadline_from(Instant::now(), options.deadline());
        let shared_options = options.with_deadline(shared_deadline);
        let shaped_request = self.descriptor.shape_request(request);
        let engine = StepEngine::establish(
            self.runtime.clone(),
            Arc::clone(&self.model),
            ToolSet::default(),
            shaped_request,
            self.step_options.clone(),
            shared_options.clone(),
            ToolOutcomePolicy::default(),
            Arc::new(ExternalApprovalDecider::default()),
            ToolHandling::ObserveOnly,
        )
        .await
        .map_err(StructuredOutputRunError::Establishment)?;
        let report = completed_report(
            drive(engine).await,
            &self.descriptor,
            StructuredOutputAttemptKind::Initial,
            0,
        )?;
        let (report, response) = take_final_response(report)?;

        match self.descriptor.consume_response(response) {
            Ok(output) => Ok(StructuredOutputRunResult {
                output,
                report,
                initial_failure: None,
            }),
            Err(initial_failure) => {
                let repair = match self.descriptor.plan_repair(
                    original_request,
                    initial_failure,
                    &shared_options,
                ) {
                    Ok(repair) => repair,
                    Err(error) => {
                        return Err(StructuredOutputRunError::Validation {
                            error,
                            report: Box::new(report),
                        });
                    }
                };
                let (repair_request, repair_options, initial_failure) = repair.into_parts();
                let Some(next_step) = report
                    .steps()
                    .last()
                    .and_then(|record| record.index().checked_add(1))
                else {
                    return Err(StructuredOutputRunError::Runtime(Box::new(
                        RunTerminal::Failed {
                            error: Error::new(
                                ErrorKind::LimitExceeded,
                                "structured-output step index overflowed",
                            ),
                            report: Box::new(report),
                        },
                    )));
                };
                let mut seeded_report = report;
                seeded_report.replace_messages(repair_request.messages.clone());
                let fallback_report = seeded_report.clone();
                let engine = StepEngine::establish_seeded(
                    self.runtime.clone(),
                    Arc::clone(&self.model),
                    ToolSet::default(),
                    repair_request,
                    self.step_options.clone(),
                    repair_options,
                    ToolOutcomePolicy::default(),
                    Arc::new(ExternalApprovalDecider::default()),
                    ToolHandling::ObserveOnly,
                    seeded_report,
                    next_step,
                )
                .await
                .map_err(|error| StructuredOutputRunError::Validation {
                    error: StructuredOutputError::from_transport(
                        error,
                        StructuredOutputAttemptKind::Repair,
                    ),
                    report: Box::new(fallback_report),
                })?;
                let repaired_report = completed_report(
                    drive(engine).await,
                    &self.descriptor,
                    StructuredOutputAttemptKind::Repair,
                    next_step,
                )?;
                let (repaired_report, repaired_response) = take_final_response(repaired_report)?;
                let output = self
                    .descriptor
                    .consume_repair_response(repaired_response)
                    .map_err(|error| StructuredOutputRunError::Validation {
                        error,
                        report: Box::new(repaired_report.clone()),
                    })?;

                Ok(StructuredOutputRunResult {
                    output,
                    report: repaired_report,
                    initial_failure: Some(initial_failure),
                })
            }
        }
    }
}

impl Runtime {
    /// Configure strict structured-output generation on this runtime.
    pub fn structured_output<T>(
        &self,
        model: Arc<dyn LanguageModel>,
        descriptor: OutputDescriptor<T>,
    ) -> StructuredOutputRunner<T> {
        StructuredOutputRunner::new(model, descriptor).with_runtime(self.clone())
    }
}

async fn drive(engine: StepEngine) -> RunTerminal {
    let cancellation = engine.cancellation().clone();
    let fallback_report = engine.report().clone();
    let initial_report = fallback_report.clone();
    let mut stream = established_run_stream(cancellation, engine.into_stream(), initial_report);
    while let Some(event) = stream.next().await {
        if let crate::RunEvent::Terminal(terminal) = event {
            return terminal;
        }
    }
    RunTerminal::Failed {
        error: Error::unexpected_eof(),
        report: Box::new(fallback_report),
    }
}

fn completed_report<T>(
    terminal: RunTerminal,
    descriptor: &OutputDescriptor<T>,
    attempt: StructuredOutputAttemptKind,
    expected_step: u32,
) -> Result<RunReport, StructuredOutputRunError>
where
    T: DeserializeOwned,
{
    match terminal {
        RunTerminal::Completed { report } => Ok(*report),
        RunTerminal::Failed { error, report } => {
            let response_failure = report
                .final_response()
                .filter(|_| {
                    report
                        .steps()
                        .last()
                        .is_some_and(|step| step.index() == expected_step)
                })
                .cloned()
                .and_then(|response| match attempt {
                    StructuredOutputAttemptKind::Initial => {
                        descriptor.consume_response(response).err()
                    }
                    StructuredOutputAttemptKind::Repair => {
                        descriptor.consume_repair_response(response).err()
                    }
                });
            let output_error = match response_failure {
                Some(failure) => failure.with_model_error_source(error),
                None => StructuredOutputError::from_transport(error, attempt),
            };
            Err(StructuredOutputRunError::Validation {
                error: output_error,
                report,
            })
        }
        terminal => Err(StructuredOutputRunError::Runtime(Box::new(terminal))),
    }
}

fn take_final_response(
    report: RunReport,
) -> Result<(RunReport, siumai_core::LanguageResponse), StructuredOutputRunError> {
    match report.final_response().cloned() {
        Some(response) => Ok((report, response)),
        None => Err(StructuredOutputRunError::Runtime(Box::new(
            RunTerminal::Failed {
                error: Error::new(
                    ErrorKind::Internal,
                    "completed structured-output run has no final response",
                ),
                report: Box::new(report),
            },
        ))),
    }
}
