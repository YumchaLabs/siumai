use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt, stream};
use siumai_core::{
    CallOptions, Cancellation, ContentPart, Error, ErrorKind, ExecutionOwner, LanguageModel,
    LanguageRequest, LanguageResponse, LanguageStream, LanguageStreamEvent, Message, MessageRole,
    StreamTerminal, ToolCall, ToolOutcome, ToolResult,
};

use crate::snapshot::{IndeterminateReason, PreparedToolSnapshot, ToolExecutionEvent};
use crate::tool::{
    ApprovalDecider, ApprovalDecision, ApprovalPolicy, ApprovalRequest, AuthorizedToolCall,
    EffectCertainty, ToolConcurrency, ToolEffect, ToolExecutionError, ToolExecutionRequest,
    ToolSet,
};
use crate::tool_loop::{ToolOutcomeAction, ToolOutcomePolicy};
use crate::{
    IndeterminateEffect, ModelTarget, RunBudget, RunEvent, RunReport, RunStopReason, RunTerminal,
    RunTimeoutKind, Runtime, StepOptions, StepRecord, SuspensionReason,
};

type BoxToolFuture = Pin<Box<dyn Future<Output = ToolAttempt> + Send + 'static>>;

pub(crate) struct StepEngine {
    runtime: Runtime,
    model: Arc<dyn LanguageModel>,
    tools: ToolSet,
    request: LanguageRequest,
    step_options: StepOptions,
    call_options: CallOptions,
    outcome_policy: ToolOutcomePolicy,
    approval_decider: Arc<dyn ApprovalDecider>,
    tool_handling: ToolHandling,
    budget: RunBudget,
    cancellation: Cancellation,
    target: ModelTarget,
    total_deadline: Instant,
    report: RunReport,
    step: u32,
    current_stream: Option<StepStream>,
    prepared_step: Option<PreparedStep>,
    needs_next_step: bool,
    pending: VecDeque<RunEvent>,
    done: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolHandling {
    Execute,
    ObserveOnly,
}

impl StepEngine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn establish(
        runtime: Runtime,
        model: Arc<dyn LanguageModel>,
        tools: ToolSet,
        request: LanguageRequest,
        step_options: StepOptions,
        options: CallOptions,
        outcome_policy: ToolOutcomePolicy,
        approval_decider: Arc<dyn ApprovalDecider>,
        tool_handling: ToolHandling,
    ) -> Result<Self, Error> {
        let target = ModelTarget::from_model(model.as_ref());
        let report = RunReport::new(target, request.messages.clone());
        Self::establish_seeded_inner(
            runtime,
            model,
            tools,
            request,
            step_options,
            options,
            outcome_policy,
            approval_decider,
            tool_handling,
            report,
            0,
            false,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn establish_seeded(
        runtime: Runtime,
        model: Arc<dyn LanguageModel>,
        tools: ToolSet,
        request: LanguageRequest,
        step_options: StepOptions,
        options: CallOptions,
        outcome_policy: ToolOutcomePolicy,
        approval_decider: Arc<dyn ApprovalDecider>,
        tool_handling: ToolHandling,
        report: RunReport,
        step: u32,
    ) -> Result<Self, Error> {
        Self::establish_seeded_inner(
            runtime,
            model,
            tools,
            request,
            step_options,
            options,
            outcome_policy,
            approval_decider,
            tool_handling,
            report,
            step,
            true,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn establish_seeded_inner(
        runtime: Runtime,
        model: Arc<dyn LanguageModel>,
        tools: ToolSet,
        mut request: LanguageRequest,
        step_options: StepOptions,
        options: CallOptions,
        outcome_policy: ToolOutcomePolicy,
        approval_decider: Arc<dyn ApprovalDecider>,
        tool_handling: ToolHandling,
        report: RunReport,
        step: u32,
        establishment_failure_as_terminal: bool,
    ) -> Result<Self, Error> {
        reject_untrusted_collisions(&request, &tools)?;
        request.tools = tools.specs().to_vec();

        let budget = runtime.run_budget().clone();
        let cancellation = options.cancellation().child();
        let total_deadline = budget.deadline_from(Instant::now(), options.deadline());
        let call_options = options
            .with_cancellation(cancellation.clone())
            .with_deadline(total_deadline);
        let target = ModelTarget::from_model(model.as_ref());
        if report.initial_target() != &target {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "seeded run report targets a different model",
            ));
        }
        request.messages = report.messages().to_vec();
        let mut pending = VecDeque::new();
        if step == 0 {
            pending.push_back(RunEvent::Started {
                target: target.clone(),
            });
        }
        pending.push_back(RunEvent::StepStarted {
            index: step,
            target: target.clone(),
        });
        let mut engine = Self {
            runtime,
            model,
            tools,
            request,
            step_options,
            call_options,
            outcome_policy,
            approval_decider,
            tool_handling,
            budget,
            cancellation,
            target: target.clone(),
            total_deadline,
            report,
            step,
            current_stream: None,
            prepared_step: None,
            needs_next_step: false,
            pending,
            done: false,
        };

        if let Err(error) = engine.report.budget_mut().charge_model_step(&engine.budget) {
            if establishment_failure_as_terminal {
                engine.queue_terminal(RunTerminal::BudgetExceeded {
                    error,
                    report: Box::new(engine.report.clone()),
                });
                return Ok(engine);
            }
            return Err(budget_start_error(error));
        }
        match engine.establish_model_stream(step == 0).await {
            Ok(stream) => {
                engine.current_stream = Some(stream);
                Ok(engine)
            }
            Err(error) if establishment_failure_as_terminal => {
                engine.queue_handshake_failure(error);
                Ok(engine)
            }
            Err(error) => Err(error),
        }
    }

    pub(crate) fn cancellation(&self) -> &Cancellation {
        &self.cancellation
    }

    pub(crate) fn report(&self) -> &RunReport {
        &self.report
    }

    pub(crate) fn into_stream(
        self,
    ) -> impl Stream<Item = Result<RunEvent, Error>> + Send + 'static {
        stream::try_unfold(self, |mut engine| async move {
            Ok::<_, Error>(engine.next_event().await.map(|event| (event, engine)))
        })
    }

    async fn next_event(&mut self) -> Option<RunEvent> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            if self.done {
                return None;
            }
            if self.cancellation.is_cancelled() {
                self.queue_terminal(RunTerminal::Cancelled {
                    reason: "tool loop cancelled".to_string(),
                    report: Box::new(self.report.clone()),
                });
                continue;
            }
            if Instant::now() >= self.total_deadline {
                self.queue_terminal(RunTerminal::TimedOut {
                    kind: RunTimeoutKind::Total,
                    report: Box::new(self.report.clone()),
                });
                continue;
            }
            if let Some(prepared) = self.prepared_step.take() {
                self.execute_prepared_step(prepared).await;
                continue;
            }
            if let Some(step_stream) = self.current_stream.take() {
                self.poll_model_stream(step_stream).await;
                continue;
            }
            if self.needs_next_step {
                self.needs_next_step = false;
                self.step = self.step.saturating_add(1);
                self.establish_later_step().await;
                continue;
            }

            self.queue_terminal(RunTerminal::Failed {
                error: Error::new(
                    ErrorKind::Internal,
                    "tool loop reached an invalid engine state",
                ),
                report: Box::new(self.report.clone()),
            });
        }
    }

    async fn establish_model_stream(&mut self, first: bool) -> Result<StepStream, Error> {
        let now = Instant::now();
        let step_deadline = self
            .total_deadline
            .min(checked_deadline(now, self.budget.timeouts().model_step()));
        let (deadline, timeout_kind) = classify_deadline(
            self.total_deadline,
            step_deadline,
            RunTimeoutKind::ModelStep,
        );
        let mut request = self.request.clone();
        request.messages = self.report.messages().to_vec();
        request.tools = self.tools.specs().to_vec();
        let future = self.runtime.stream(
            self.model.as_ref(),
            request,
            self.step_options.clone(),
            self.call_options.clone(),
        );

        match wait_for(future, &self.cancellation, deadline, timeout_kind).await {
            WaitResult::Ready(result) => result.map(|stream| StepStream {
                stream,
                deadline: step_deadline,
                first_event: true,
                provider_state_ids: Vec::new(),
                provider_result_ids: BTreeSet::new(),
            }),
            WaitResult::Cancelled => Err(Error::cancelled(if first {
                "tool loop cancelled during first model handshake"
            } else {
                "tool loop cancelled during model handshake"
            })),
            WaitResult::TimedOut(_) => Err(Error::new(
                ErrorKind::Timeout,
                if first {
                    "tool loop first model handshake timed out"
                } else {
                    "tool loop model handshake timed out"
                },
            )),
        }
    }

    async fn establish_later_step(&mut self) {
        if let Err(error) = self.report.budget_mut().charge_model_step(&self.budget) {
            self.queue_terminal(RunTerminal::BudgetExceeded {
                error,
                report: Box::new(self.report.clone()),
            });
            return;
        }

        match self.establish_model_stream(false).await {
            Ok(stream) => {
                self.pending.push_back(RunEvent::StepStarted {
                    index: self.step,
                    target: self.target.clone(),
                });
                self.current_stream = Some(stream);
            }
            Err(error) => self.queue_handshake_failure(error),
        }
    }

    fn queue_handshake_failure(&mut self, error: Error) {
        match error.kind() {
            ErrorKind::Cancelled => self.queue_terminal(RunTerminal::Cancelled {
                reason: "tool loop cancelled during model handshake".to_string(),
                report: Box::new(self.report.clone()),
            }),
            ErrorKind::Timeout => {
                let kind = if Instant::now() >= self.total_deadline {
                    RunTimeoutKind::Total
                } else {
                    RunTimeoutKind::ModelStep
                };
                self.queue_terminal(RunTerminal::TimedOut {
                    kind,
                    report: Box::new(self.report.clone()),
                });
            }
            _ => self.queue_terminal(RunTerminal::Failed {
                error,
                report: Box::new(self.report.clone()),
            }),
        }
    }

    async fn poll_model_stream(&mut self, mut state: StepStream) {
        let chunk_deadline = checked_deadline(
            Instant::now(),
            if state.first_event {
                self.budget.timeouts().first_chunk()
            } else {
                self.budget.timeouts().inter_chunk()
            },
        );
        let (deadline, kind) = earliest_timeout([
            (self.total_deadline, RunTimeoutKind::Total),
            (state.deadline, RunTimeoutKind::ModelStep),
            (
                chunk_deadline,
                if state.first_event {
                    RunTimeoutKind::FirstChunk
                } else {
                    RunTimeoutKind::InterChunk
                },
            ),
        ]);

        match wait_for(state.stream.next(), &self.cancellation, deadline, kind).await {
            WaitResult::Cancelled => self.queue_terminal(RunTerminal::Cancelled {
                reason: "tool loop cancelled while reading model stream".to_string(),
                report: Box::new(self.report.clone()),
            }),
            WaitResult::TimedOut(kind) => self.queue_terminal(RunTerminal::TimedOut {
                kind,
                report: Box::new(self.report.clone()),
            }),
            WaitResult::Ready(None) => self.queue_terminal(RunTerminal::Failed {
                error: Error::unexpected_eof(),
                report: Box::new(self.report.clone()),
            }),
            WaitResult::Ready(Some(LanguageStreamEvent::Terminal(terminal))) => {
                self.consume_model_terminal(terminal, state).await;
            }
            WaitResult::Ready(Some(event)) => {
                state.first_event = false;
                match &event {
                    LanguageStreamEvent::ProviderDeferred { id, state: item } => {
                        state.provider_state_ids.push(id.clone());
                        self.report.provider_deferred_mut().push(item.clone());
                    }
                    LanguageStreamEvent::ToolResult(result) => {
                        state.provider_result_ids.insert(result.call_id.clone());
                    }
                    _ => {}
                }
                self.current_stream = Some(state);
                self.pending.push_back(RunEvent::Model {
                    step: self.step,
                    event,
                });
            }
        }
    }

    async fn consume_model_terminal(&mut self, terminal: StreamTerminal, state: StepStream) {
        match terminal {
            StreamTerminal::Completed { response } => {
                self.prepare_completed_response(*response, state).await;
            }
            StreamTerminal::Failed { error, response } => {
                if let Some(response) = response
                    && let Err(error) = self.record_terminal_response(*response)
                {
                    self.queue_terminal(RunTerminal::BudgetExceeded {
                        error,
                        report: Box::new(self.report.clone()),
                    });
                    return;
                }
                self.queue_terminal(RunTerminal::Failed {
                    error,
                    report: Box::new(self.report.clone()),
                });
            }
            StreamTerminal::Cancelled { reason, response } => {
                if let Some(response) = response
                    && let Err(error) = self.record_terminal_response(*response)
                {
                    self.queue_terminal(RunTerminal::BudgetExceeded {
                        error,
                        report: Box::new(self.report.clone()),
                    });
                    return;
                }
                self.queue_terminal(RunTerminal::Cancelled {
                    reason,
                    report: Box::new(self.report.clone()),
                });
            }
            _ => self.queue_terminal(RunTerminal::Failed {
                error: Error::protocol_violation("unsupported model stream terminal"),
                report: Box::new(self.report.clone()),
            }),
        }
    }

    fn record_terminal_response(
        &mut self,
        response: LanguageResponse,
    ) -> Result<(), crate::BudgetError> {
        self.report.accumulate_usage(response.usage());
        let budget_result = self
            .report
            .budget_mut()
            .charge_usage(response.usage(), &self.budget);
        self.finish_step(response, Vec::new());
        budget_result
    }

    async fn prepare_completed_response(&mut self, response: LanguageResponse, state: StepStream) {
        self.report.accumulate_usage(response.usage());
        if let Err(error) = self
            .report
            .budget_mut()
            .charge_usage(response.usage(), &self.budget)
        {
            self.append_assistant_message(&response);
            self.finish_step(response, Vec::new());
            self.queue_terminal(RunTerminal::BudgetExceeded {
                error,
                report: Box::new(self.report.clone()),
            });
            return;
        }
        self.append_assistant_message(&response);

        if self.tool_handling == ToolHandling::ObserveOnly {
            self.finish_step(response, Vec::new());
            self.queue_terminal(RunTerminal::Completed {
                report: Box::new(self.report.clone()),
            });
            return;
        }

        let mut provider_result_ids = state.provider_result_ids;
        provider_result_ids.extend(response.content().iter().filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(result.call_id.clone()),
            _ => None,
        }));
        let calls = response
            .content()
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        // Provider-owned work is an orchestration boundary. Inspect the whole
        // step before resolving any local name so a provider/local collision or
        // an unrelated invalid local call cannot cross that boundary.
        let mut provider_states = state.provider_state_ids;
        provider_states.extend(calls.iter().filter_map(|call| match &call.owner {
            ExecutionOwner::Provider { .. } if !provider_result_ids.contains(&call.id) => {
                Some(call.id.clone())
            }
            ExecutionOwner::Local => None,
            _ => Some(call.id.clone()),
        }));
        provider_states.sort();
        provider_states.dedup();
        if !provider_states.is_empty() {
            self.finish_step(response, Vec::new());
            self.queue_terminal(RunTerminal::Suspended {
                reason: SuspensionReason::AwaitingProvider {
                    state_ids: provider_states,
                },
                report: Box::new(self.report.clone()),
            });
            return;
        }

        let mut requests = Vec::new();
        let mut approvals = Vec::new();
        let mut immediate_results = Vec::new();

        for (ordinal, call) in calls.into_iter().enumerate() {
            match &call.owner {
                ExecutionOwner::Provider { .. } => {}
                ExecutionOwner::Local => {
                    let request = match self.tools.resolve(call.clone()) {
                        Ok(request) => request,
                        Err(error) => {
                            self.finish_preparation_failure(response, ordinal, call, error);
                            return;
                        }
                    };
                    if let Err(error) = request.validate() {
                        self.finish_preparation_failure(response, ordinal, call, error);
                        return;
                    }
                    let argument_bytes = match serde_json::to_vec(request.arguments()) {
                        Ok(arguments) => arguments.len(),
                        Err(error) => {
                            self.finish_step(response, Vec::new());
                            self.queue_terminal(RunTerminal::Failed {
                                error: Error::new(
                                    ErrorKind::Internal,
                                    "tool arguments could not be serialized for budget accounting",
                                )
                                .with_source(error),
                                report: Box::new(self.report.clone()),
                            });
                            return;
                        }
                    };
                    if let Err(error) = self
                        .report
                        .budget_mut()
                        .charge_tool_call(argument_bytes, &self.budget)
                    {
                        self.finish_step(response, Vec::new());
                        self.queue_terminal(RunTerminal::BudgetExceeded {
                            error,
                            report: Box::new(self.report.clone()),
                        });
                        return;
                    }
                    if let Err(error) = self.log_prepared(ordinal, &request) {
                        self.finish_step(response, Vec::new());
                        self.queue_terminal(RunTerminal::Failed {
                            error,
                            report: Box::new(self.report.clone()),
                        });
                        return;
                    }
                    self.pending.push_back(RunEvent::ToolPrepared {
                        step: self.step,
                        ordinal,
                        call: call.clone(),
                    });
                    let authorized = match request.approval_policy() {
                        ApprovalPolicy::NotRequired => match request.authorize_not_required() {
                            Ok(authorized) => Some(authorized),
                            Err(error) => {
                                self.finish_step(
                                    response,
                                    immediate_results
                                        .into_iter()
                                        .map(|result: IndexedResult| result.result)
                                        .collect(),
                                );
                                self.queue_terminal(RunTerminal::Failed {
                                    error: execution_authorization_error(error),
                                    report: Box::new(self.report.clone()),
                                });
                                return;
                            }
                        },
                        ApprovalPolicy::Required => {
                            let approval_request = ApprovalRequest::from_frozen(
                                self.step,
                                ordinal,
                                self.target.clone(),
                                &request,
                            );
                            let decision = wait_for(
                                self.approval_decider.decide(&approval_request),
                                &self.cancellation,
                                self.total_deadline,
                                RunTimeoutKind::Total,
                            )
                            .await;
                            match decision {
                                WaitResult::Ready(Ok(ApprovalDecision::Approve)) => {
                                    Some(request.authorize_host_auto_approved())
                                }
                                WaitResult::Ready(Ok(ApprovalDecision::Deny(denial))) => {
                                    let result = ToolResult {
                                        call_id: request.call_id().to_string(),
                                        name: request.name().to_string(),
                                        outcome: ToolOutcome::Denied {
                                            reason: denial.reason().to_owned(),
                                        },
                                    };
                                    if let Err(error) =
                                        self.record_non_dispatch_result(&request, &result)
                                    {
                                        self.finish_step(
                                            response,
                                            immediate_results
                                                .into_iter()
                                                .map(|result: IndexedResult| result.result)
                                                .collect(),
                                        );
                                        match error {
                                            NonDispatchResultError::Budget(error) => {
                                                self.queue_terminal(RunTerminal::BudgetExceeded {
                                                    error,
                                                    report: Box::new(self.report.clone()),
                                                });
                                            }
                                            NonDispatchResultError::Failed(error) => {
                                                self.queue_terminal(RunTerminal::Failed {
                                                    error,
                                                    report: Box::new(self.report.clone()),
                                                });
                                            }
                                        }
                                        return;
                                    }
                                    self.pending.push_back(RunEvent::ToolCompleted {
                                        step: self.step,
                                        ordinal,
                                        result: result.clone(),
                                    });
                                    let stop = self.outcome_policy.action(&result.outcome)
                                        == ToolOutcomeAction::Stop;
                                    immediate_results.push(IndexedResult {
                                        ordinal,
                                        result: result.clone(),
                                    });
                                    if stop {
                                        let tool_results = immediate_results
                                            .into_iter()
                                            .map(|result| result.result)
                                            .collect();
                                        self.finish_step(response, tool_results);
                                        self.queue_terminal(RunTerminal::Stopped {
                                            reason: RunStopReason::ToolOutcome {
                                                call_id: result.call_id,
                                                outcome: result.outcome,
                                            },
                                            report: Box::new(self.report.clone()),
                                        });
                                        return;
                                    }
                                    None
                                }
                                WaitResult::Ready(Ok(ApprovalDecision::AwaitExternal)) => {
                                    if let Err(error) = self
                                        .report
                                        .budget_mut()
                                        .reserve_pending_approval(&self.budget)
                                    {
                                        self.finish_step(
                                            response,
                                            immediate_results
                                                .into_iter()
                                                .map(|result: IndexedResult| result.result)
                                                .collect(),
                                        );
                                        self.queue_terminal(RunTerminal::BudgetExceeded {
                                            error,
                                            report: Box::new(self.report.clone()),
                                        });
                                        return;
                                    }
                                    approvals.push(call.id.clone());
                                    None
                                }
                                WaitResult::Ready(Err(error)) => {
                                    self.finish_step(
                                        response,
                                        immediate_results
                                            .into_iter()
                                            .map(|result: IndexedResult| result.result)
                                            .collect(),
                                    );
                                    self.queue_terminal(RunTerminal::Failed {
                                        error: execution_approval_decision_error(error),
                                        report: Box::new(self.report.clone()),
                                    });
                                    return;
                                }
                                WaitResult::TimedOut(kind) => {
                                    self.finish_step(
                                        response,
                                        immediate_results
                                            .into_iter()
                                            .map(|result: IndexedResult| result.result)
                                            .collect(),
                                    );
                                    self.queue_terminal(RunTerminal::TimedOut {
                                        kind,
                                        report: Box::new(self.report.clone()),
                                    });
                                    return;
                                }
                                WaitResult::Cancelled => {
                                    self.finish_step(
                                        response,
                                        immediate_results
                                            .into_iter()
                                            .map(|result: IndexedResult| result.result)
                                            .collect(),
                                    );
                                    self.queue_terminal(RunTerminal::Cancelled {
                                        reason: "tool loop cancelled during approval decision"
                                            .to_string(),
                                        report: Box::new(self.report.clone()),
                                    });
                                    return;
                                }
                            }
                        }
                    };
                    if let Some(call) = authorized {
                        requests.push(IndexedAuthorizedCall { ordinal, call });
                    }
                }
                _ => {}
            }
        }

        if !approvals.is_empty() {
            self.finish_step(
                response,
                immediate_results
                    .into_iter()
                    .map(|result| result.result)
                    .collect(),
            );
            self.queue_terminal(RunTerminal::Suspended {
                reason: SuspensionReason::AwaitingApproval {
                    call_ids: approvals,
                },
                report: Box::new(self.report.clone()),
            });
        } else if requests.is_empty() {
            let continued = !immediate_results.is_empty();
            self.finish_step(
                response,
                immediate_results
                    .into_iter()
                    .map(|result| result.result)
                    .collect(),
            );
            if continued {
                self.needs_next_step = true;
            } else {
                self.queue_terminal(RunTerminal::Completed {
                    report: Box::new(self.report.clone()),
                });
            }
        } else {
            self.prepared_step = Some(PreparedStep {
                response,
                requests,
                results: immediate_results,
            });
        }
    }

    fn finish_preparation_failure(
        &mut self,
        response: LanguageResponse,
        ordinal: usize,
        call: ToolCall,
        error: ToolExecutionError,
    ) {
        let outcome = execution_error_outcome(error);
        let result = ToolResult {
            call_id: call.id.clone(),
            name: call.name,
            outcome: outcome.clone(),
        };
        self.pending.push_back(RunEvent::ToolCompleted {
            step: self.step,
            ordinal,
            result: result.clone(),
        });
        self.finish_step(response, vec![result]);
        if self.outcome_policy.action(&outcome) == ToolOutcomeAction::Continue {
            self.needs_next_step = true;
        } else {
            self.queue_terminal(RunTerminal::Stopped {
                reason: RunStopReason::ToolOutcome {
                    call_id: call.id,
                    outcome,
                },
                report: Box::new(self.report.clone()),
            });
        }
    }

    async fn execute_prepared_step(&mut self, prepared: PreparedStep) {
        let PreparedStep {
            response,
            requests,
            mut results,
        } = prepared;
        let progress = self.execute_requests(requests).await;
        results.extend(progress.results);
        results.sort_unstable_by_key(|result| result.ordinal);
        for result in &results {
            self.pending.push_back(RunEvent::ToolCompleted {
                step: self.step,
                ordinal: result.ordinal,
                result: result.result.clone(),
            });
        }
        let tool_results = results
            .iter()
            .map(|result| result.result.clone())
            .collect::<Vec<_>>();
        self.finish_step(response, tool_results);

        match progress.stop {
            None => self.needs_next_step = true,
            Some(ExecutionStop::Policy { call_id, outcome }) => {
                self.queue_terminal(RunTerminal::Stopped {
                    reason: RunStopReason::ToolOutcome { call_id, outcome },
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::Budget(error)) => {
                self.queue_terminal(RunTerminal::BudgetExceeded {
                    error,
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::TimedOut(kind)) => {
                self.queue_terminal(RunTerminal::TimedOut {
                    kind,
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::Cancelled) => {
                self.queue_terminal(RunTerminal::Cancelled {
                    reason: "tool loop cancelled during tool execution".to_string(),
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::Indeterminate(effect)) => {
                self.queue_terminal(RunTerminal::Indeterminate {
                    effect,
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::Failed(error)) => {
                self.queue_terminal(RunTerminal::Failed {
                    error,
                    report: Box::new(self.report.clone()),
                });
            }
        }
    }

    async fn execute_requests(
        &mut self,
        requests: Vec<IndexedAuthorizedCall>,
    ) -> ExecutionProgress {
        let mut queue = VecDeque::from(requests);
        let mut results = Vec::new();

        while let Some(front) = queue.front() {
            if is_parallel_safe(front.call.request()) {
                let mut batch = Vec::new();
                while queue
                    .front()
                    .is_some_and(|request| is_parallel_safe(request.call.request()))
                {
                    if let Some(request) = queue.pop_front() {
                        batch.push(request);
                    }
                }
                let progress = self.execute_parallel_batch(batch).await;
                results.extend(progress.results);
                if progress.stop.is_some() {
                    return ExecutionProgress {
                        results,
                        stop: progress.stop,
                    };
                }
            } else if let Some(request) = queue.pop_front() {
                let progress = self.execute_sequential(request).await;
                results.extend(progress.results);
                if progress.stop.is_some() {
                    return ExecutionProgress {
                        results,
                        stop: progress.stop,
                    };
                }
            }
        }

        ExecutionProgress {
            results,
            stop: None,
        }
    }

    async fn execute_sequential(&mut self, request: IndexedAuthorizedCall) -> ExecutionProgress {
        if let Some(stop) = self.pre_dispatch_stop() {
            return ExecutionProgress::stopped(stop);
        }
        if let Err(error) = self.log_dispatched(request.call.request()) {
            return ExecutionProgress::stopped(ExecutionStop::Failed(error));
        }
        let attempt = self.await_tool(request).await;
        self.process_attempt(attempt)
    }

    async fn execute_parallel_batch(
        &mut self,
        requests: Vec<IndexedAuthorizedCall>,
    ) -> ExecutionProgress {
        let mut pending = VecDeque::from(requests);
        let mut active = FuturesUnordered::<BoxToolFuture>::new();
        let mut active_by_binding = BTreeMap::<String, usize>::new();
        let mut active_calls = BTreeMap::new();
        let mut results = Vec::new();
        let mut policy_stop: Option<(usize, String, ToolOutcome)> = None;

        loop {
            while policy_stop.is_none() && active.len() < self.budget.max_concurrent_tools() {
                if let Some(stop) = self.pre_dispatch_stop() {
                    self.mark_active_indeterminate(&active_calls);
                    return ExecutionProgress {
                        results,
                        stop: Some(stop),
                    };
                }
                let eligible = pending.iter().position(|candidate| {
                    let key = binding_key(candidate.call.request());
                    let active_count = active_by_binding.get(&key).copied().unwrap_or(0);
                    active_count < binding_parallel_limit(candidate.call.request())
                });
                let Some(position) = eligible else {
                    break;
                };
                let Some(request) = pending.remove(position) else {
                    break;
                };
                if let Err(error) = self.log_dispatched(request.call.request()) {
                    self.mark_active_indeterminate(&active_calls);
                    return ExecutionProgress {
                        results,
                        stop: Some(ExecutionStop::Failed(error)),
                    };
                }
                let key = binding_key(request.call.request());
                *active_by_binding.entry(key).or_default() += 1;
                active_calls.insert(
                    request.ordinal,
                    (
                        request.call.request().call_id().to_string(),
                        request.call.request().attempt(),
                    ),
                );
                let cancellation = self.cancellation.clone();
                let total_deadline = self.total_deadline;
                let tool_timeout = self.budget.timeouts().tool();
                active.push(Box::pin(async move {
                    await_tool_attempt(request, cancellation, total_deadline, tool_timeout).await
                }));
            }

            if active.is_empty() {
                break;
            }
            let Some(attempt) = active.next().await else {
                break;
            };
            let key = binding_key(&attempt.request);
            if let Some(count) = active_by_binding.get_mut(&key) {
                *count = count.saturating_sub(1);
            }
            active_calls.remove(&attempt.ordinal);
            let progress = self.process_attempt(attempt);
            if let Some(stop) = progress.stop {
                match stop {
                    ExecutionStop::Policy { call_id, outcome } => {
                        let ordinal = progress
                            .results
                            .first()
                            .map_or(usize::MAX, |result| result.ordinal);
                        let replace = policy_stop
                            .as_ref()
                            .is_none_or(|(current, _, _)| ordinal < *current);
                        if replace {
                            policy_stop = Some((ordinal, call_id, outcome));
                        }
                    }
                    terminal => {
                        self.mark_active_indeterminate(&active_calls);
                        results.extend(progress.results);
                        return ExecutionProgress {
                            results,
                            stop: Some(terminal),
                        };
                    }
                }
            }
            results.extend(progress.results);
        }

        let stop =
            policy_stop.map(|(_, call_id, outcome)| ExecutionStop::Policy { call_id, outcome });
        ExecutionProgress { results, stop }
    }

    async fn await_tool(&mut self, request: IndexedAuthorizedCall) -> ToolAttempt {
        await_tool_attempt(
            request,
            self.cancellation.clone(),
            self.total_deadline,
            self.budget.timeouts().tool(),
        )
        .await
    }

    fn process_attempt(&mut self, attempt: ToolAttempt) -> ExecutionProgress {
        let effect = attempt.request.effect();
        let call_id = attempt.request.call_id().to_string();
        let execution_attempt = attempt.request.attempt();
        let binding_fingerprint = attempt.request.binding_identity().fingerprint.clone();
        let ordinal = attempt.ordinal;

        let result = match attempt.outcome {
            ToolAttemptOutcome::Completed(Ok(result)) => result,
            ToolAttemptOutcome::Completed(Err(error)) => {
                if error.effect_certainty() == EffectCertainty::Indeterminate
                    && effect == ToolEffect::SideEffecting
                {
                    self.log_indeterminate(
                        &call_id,
                        execution_attempt,
                        IndeterminateReason::DispatchOutcomeUnknown,
                    );
                    return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool effect is indeterminate after executor failure"
                                .to_string(),
                        },
                    ));
                }
                ToolResult {
                    call_id: call_id.clone(),
                    name: attempt.request.name().to_string(),
                    outcome: execution_error_outcome(error),
                }
            }
            ToolAttemptOutcome::TimedOut(kind) => {
                self.log_indeterminate(
                    &call_id,
                    execution_attempt,
                    IndeterminateReason::DispatchOutcomeUnknown,
                );
                if effect == ToolEffect::SideEffecting {
                    return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool execution timed out after dispatch".to_string(),
                        },
                    ));
                }
                return ExecutionProgress::stopped(ExecutionStop::TimedOut(kind));
            }
            ToolAttemptOutcome::Cancelled => {
                self.log_indeterminate(
                    &call_id,
                    execution_attempt,
                    IndeterminateReason::CancellationAfterDispatch,
                );
                if effect == ToolEffect::SideEffecting {
                    return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool execution was cancelled after dispatch".to_string(),
                        },
                    ));
                }
                return ExecutionProgress::stopped(ExecutionStop::Cancelled);
            }
        };

        let result_bytes = match serde_json::to_vec(&result) {
            Ok(value) => value.len(),
            Err(error) => {
                self.log_indeterminate(
                    &call_id,
                    execution_attempt,
                    IndeterminateReason::CheckpointFailure,
                );
                if effect == ToolEffect::SideEffecting {
                    return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool result could not be checkpointed".to_string(),
                        },
                    ));
                }
                return ExecutionProgress::stopped(ExecutionStop::Failed(
                    Error::new(
                        ErrorKind::Internal,
                        "tool result could not be serialized for budget accounting",
                    )
                    .with_source(error),
                ));
            }
        };
        if let Err(error) = self
            .report
            .budget_mut()
            .charge_tool_result(result_bytes, &self.budget)
        {
            self.log_indeterminate(
                &call_id,
                execution_attempt,
                IndeterminateReason::CheckpointFailure,
            );
            if effect == ToolEffect::SideEffecting {
                return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                    IndeterminateEffect {
                        call_id,
                        binding_fingerprint,
                        reason: "tool result exceeded checkpoint budget".to_string(),
                    },
                ));
            }
            return ExecutionProgress::stopped(ExecutionStop::Budget(error));
        }
        if let Err(error) = self.log_completed(&attempt.request, &result) {
            self.log_indeterminate(
                &call_id,
                execution_attempt,
                IndeterminateReason::CheckpointFailure,
            );
            if effect == ToolEffect::SideEffecting {
                return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                    IndeterminateEffect {
                        call_id,
                        binding_fingerprint,
                        reason: "tool completion could not be checkpointed".to_string(),
                    },
                ));
            }
            return ExecutionProgress::stopped(ExecutionStop::Failed(error));
        }

        let stop =
            (self.outcome_policy.action(&result.outcome) == ToolOutcomeAction::Stop).then(|| {
                ExecutionStop::Policy {
                    call_id: result.call_id.clone(),
                    outcome: result.outcome.clone(),
                }
            });
        ExecutionProgress {
            results: vec![IndexedResult { ordinal, result }],
            stop,
        }
    }

    fn append_assistant_message(&mut self, response: &LanguageResponse) {
        self.report.messages_mut().push(Message {
            role: MessageRole::Assistant,
            content: response.content().to_vec(),
        });
    }

    fn finish_step(&mut self, response: LanguageResponse, results: Vec<ToolResult>) {
        if !results.is_empty() {
            self.report.messages_mut().push(Message {
                role: MessageRole::Tool,
                content: results
                    .iter()
                    .cloned()
                    .map(ContentPart::ToolResult)
                    .collect(),
            });
        }
        let record = StepRecord::new(self.step, self.target.clone(), response, results);
        self.report.steps_mut().push(record.clone());
        self.pending.push_back(RunEvent::StepFinished {
            record: Box::new(record),
        });
    }

    fn log_prepared(
        &mut self,
        ordinal: usize,
        request: &ToolExecutionRequest,
    ) -> Result<(), Error> {
        let ordinal = u32::try_from(ordinal).map_err(|error| {
            Error::new(
                ErrorKind::LimitExceeded,
                "tool ordinal cannot be represented in a durable execution log",
            )
            .with_source(error)
        })?;
        let sequence = self.report.execution_log().next_sequence();
        self.report
            .execution_log_mut()
            .append(ToolExecutionEvent::prepared(
                sequence,
                unix_millis(),
                self.step,
                PreparedToolSnapshot::new(
                    ordinal,
                    request.call().clone(),
                    request.binding_identity().clone(),
                    request.recovery_policy(),
                    request.idempotency_key().cloned(),
                    request.attempt(),
                ),
            ))
            .map_err(execution_log_error)
    }

    fn log_dispatched(&mut self, request: &ToolExecutionRequest) -> Result<(), Error> {
        let sequence = self.report.execution_log().next_sequence();
        self.report
            .execution_log_mut()
            .append(ToolExecutionEvent::dispatched(
                sequence,
                unix_millis(),
                request.call_id(),
                request.attempt(),
                None,
            ))
            .map_err(execution_log_error)
    }

    fn log_completed(
        &mut self,
        request: &ToolExecutionRequest,
        result: &ToolResult,
    ) -> Result<(), Error> {
        let sequence = self.report.execution_log().next_sequence();
        self.report
            .execution_log_mut()
            .append(ToolExecutionEvent::completed(
                sequence,
                unix_millis(),
                &result.call_id,
                request.attempt(),
                result.outcome.clone(),
            ))
            .map_err(execution_log_error)
    }

    fn record_non_dispatch_result(
        &mut self,
        request: &ToolExecutionRequest,
        result: &ToolResult,
    ) -> Result<(), NonDispatchResultError> {
        let result_bytes = serde_json::to_vec(result)
            .map_err(|error| {
                NonDispatchResultError::Failed(
                    Error::new(
                        ErrorKind::Internal,
                        "tool result could not be serialized for budget accounting",
                    )
                    .with_source(error),
                )
            })?
            .len();
        self.report
            .budget_mut()
            .charge_tool_result(result_bytes, &self.budget)
            .map_err(NonDispatchResultError::Budget)?;
        self.log_completed(request, result)
            .map_err(NonDispatchResultError::Failed)
    }

    fn log_indeterminate(
        &mut self,
        call_id: &str,
        attempt: crate::tool::ToolExecutionAttempt,
        reason: IndeterminateReason,
    ) {
        let sequence = self.report.execution_log().next_sequence();
        let _ = self
            .report
            .execution_log_mut()
            .append(ToolExecutionEvent::indeterminate(
                sequence,
                unix_millis(),
                call_id,
                attempt,
                reason,
            ));
    }

    fn mark_active_indeterminate(
        &mut self,
        active: &BTreeMap<usize, (String, crate::tool::ToolExecutionAttempt)>,
    ) {
        for (call_id, attempt) in active.values() {
            self.log_indeterminate(
                call_id,
                *attempt,
                IndeterminateReason::DispatchOutcomeUnknown,
            );
        }
    }

    fn pre_dispatch_stop(&self) -> Option<ExecutionStop> {
        if self.cancellation.is_cancelled() {
            Some(ExecutionStop::Cancelled)
        } else if Instant::now() >= self.total_deadline {
            Some(ExecutionStop::TimedOut(RunTimeoutKind::Total))
        } else {
            None
        }
    }

    fn queue_terminal(&mut self, terminal: RunTerminal) {
        if self.done {
            return;
        }
        self.current_stream = None;
        self.prepared_step = None;
        self.needs_next_step = false;
        self.done = true;
        self.pending.push_back(RunEvent::Terminal(terminal));
    }
}

struct StepStream {
    stream: LanguageStream,
    deadline: Instant,
    first_event: bool,
    provider_state_ids: Vec<String>,
    provider_result_ids: BTreeSet<String>,
}

struct PreparedStep {
    response: LanguageResponse,
    requests: Vec<IndexedAuthorizedCall>,
    results: Vec<IndexedResult>,
}

struct IndexedAuthorizedCall {
    ordinal: usize,
    call: AuthorizedToolCall,
}

struct IndexedResult {
    ordinal: usize,
    result: ToolResult,
}

struct ToolAttempt {
    ordinal: usize,
    request: ToolExecutionRequest,
    outcome: ToolAttemptOutcome,
}

enum ToolAttemptOutcome {
    Completed(Result<ToolResult, ToolExecutionError>),
    TimedOut(RunTimeoutKind),
    Cancelled,
}

struct ExecutionProgress {
    results: Vec<IndexedResult>,
    stop: Option<ExecutionStop>,
}

impl ExecutionProgress {
    fn stopped(stop: ExecutionStop) -> Self {
        Self {
            results: Vec::new(),
            stop: Some(stop),
        }
    }
}

enum ExecutionStop {
    Policy {
        call_id: String,
        outcome: ToolOutcome,
    },
    Budget(crate::BudgetError),
    TimedOut(RunTimeoutKind),
    Cancelled,
    Indeterminate(IndeterminateEffect),
    Failed(Error),
}

enum NonDispatchResultError {
    Budget(crate::BudgetError),
    Failed(Error),
}

enum WaitResult<T> {
    Ready(T),
    TimedOut(RunTimeoutKind),
    Cancelled,
}

async fn wait_for<F, T>(
    future: F,
    cancellation: &Cancellation,
    deadline: Instant,
    timeout_kind: RunTimeoutKind,
) -> WaitResult<T>
where
    F: Future<Output = T>,
{
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => WaitResult::Cancelled,
        _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
            WaitResult::TimedOut(timeout_kind)
        }
        output = future => WaitResult::Ready(output),
    }
}

async fn await_tool_attempt(
    request: IndexedAuthorizedCall,
    cancellation: Cancellation,
    total_deadline: Instant,
    tool_timeout: std::time::Duration,
) -> ToolAttempt {
    let tool_deadline = checked_deadline(Instant::now(), tool_timeout);
    let (deadline, kind) = classify_deadline(total_deadline, tool_deadline, RunTimeoutKind::Tool);
    let IndexedAuthorizedCall { ordinal, call } = request;
    let frozen_request = call.request().clone();
    let outcome = match wait_for(call.dispatch(), &cancellation, deadline, kind).await {
        WaitResult::Ready(result) => ToolAttemptOutcome::Completed(result),
        WaitResult::TimedOut(kind) => ToolAttemptOutcome::TimedOut(kind),
        WaitResult::Cancelled => ToolAttemptOutcome::Cancelled,
    };
    ToolAttempt {
        ordinal,
        request: frozen_request,
        outcome,
    }
}

fn reject_untrusted_collisions(request: &LanguageRequest, tools: &ToolSet) -> Result<(), Error> {
    if request
        .tools
        .iter()
        .any(|spec| tools.get(spec.name()).is_some())
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "request tool definition conflicts with a trusted local binding",
        ));
    }
    Ok(())
}

fn is_parallel_safe(request: &ToolExecutionRequest) -> bool {
    request.effect() == ToolEffect::ReadOnly
        && matches!(request.concurrency(), ToolConcurrency::SafeParallel { .. })
}

fn binding_key(request: &ToolExecutionRequest) -> String {
    request.binding_identity().fingerprint.clone()
}

fn binding_parallel_limit(request: &ToolExecutionRequest) -> usize {
    match request.concurrency() {
        ToolConcurrency::Sequential => 1,
        ToolConcurrency::SafeParallel { max_in_flight } => max_in_flight.get(),
    }
}

fn execution_error_outcome(error: ToolExecutionError) -> ToolOutcome {
    match error {
        ToolExecutionError::ExecutorFailed {
            message, retryable, ..
        } => ToolOutcome::ExecutionFailed { message, retryable },
        ToolExecutionError::InvalidArguments { message, .. } => ToolOutcome::ExecutionFailed {
            message,
            retryable: false,
        },
        other => ToolOutcome::ExecutionFailed {
            message: other.to_string(),
            retryable: false,
        },
    }
}

fn execution_log_error(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::new(ErrorKind::Internal, "tool execution log transition failed").with_source(error)
}

fn execution_authorization_error(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::new(
        ErrorKind::Internal,
        "tool execution authorization invariant failed",
    )
    .with_source(error)
}

fn execution_approval_decision_error(
    error: impl std::error::Error + Send + Sync + 'static,
) -> Error {
    Error::new(ErrorKind::Internal, "host approval decision failed").with_source(error)
}

fn budget_start_error(error: crate::BudgetError) -> Error {
    Error::new(
        ErrorKind::LimitExceeded,
        "tool loop could not reserve its first model step",
    )
    .with_source(error)
}

fn checked_deadline(now: Instant, duration: std::time::Duration) -> Instant {
    now.checked_add(duration).unwrap_or(now)
}

fn classify_deadline(
    total_deadline: Instant,
    local_deadline: Instant,
    local_kind: RunTimeoutKind,
) -> (Instant, RunTimeoutKind) {
    if total_deadline <= local_deadline {
        (total_deadline, RunTimeoutKind::Total)
    } else {
        (local_deadline, local_kind)
    }
}

fn earliest_timeout<const N: usize>(
    deadlines: [(Instant, RunTimeoutKind); N],
) -> (Instant, RunTimeoutKind) {
    deadlines
        .into_iter()
        .min_by_key(|(deadline, _)| *deadline)
        .unwrap_or((Instant::now(), RunTimeoutKind::Total))
}

fn unix_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}
