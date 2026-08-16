//! Atomic planning for one authoritative completed model step.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use siumai_core::{
    Cancellation, Error, ErrorKind, LanguageResponse, Message, ToolOutcome, ToolResult,
};

use crate::provider_deferred::{ProviderDeferredCommit, ProviderDeferredLedger};
use crate::snapshot::{PreparedToolSnapshot, ProviderStateSnapshot};
use crate::tool::{
    ApprovalDecider, ApprovalDecision, ApprovalPolicy, ApprovalRequest, ToolExecutionError,
    ToolJournal, ToolJournalError, ToolSet,
};
use crate::tool_loop::{ToolOutcomeAction, ToolOutcomePolicy};
use crate::{BudgetError, BudgetLedger, ModelTarget, RunBudget, RunEvent, RunTimeoutKind};

use super::{
    IndexedAuthorizedCall, IndexedResult, PendingApprovalCall, ToolHandling, WaitResult,
    execution_error_outcome, wait_for,
};

pub(super) struct CompletedStepPlan {
    pub(super) response: LanguageResponse,
    pub(super) assistant_message: Option<Message>,
    pub(super) budget: BudgetLedger,
    pub(super) journal: Option<ToolJournal>,
    pub(super) provider_ledger: ProviderDeferredLedger,
    pub(super) provider_pending: Vec<ProviderStateSnapshot>,
    pub(super) prepared: Vec<PreparedToolSnapshot>,
    pub(super) requests: Vec<IndexedAuthorizedCall>,
    pub(super) immediate_results: Vec<IndexedResult>,
    pub(super) pending_approvals: Vec<PendingApprovalCall>,
    pub(super) events: Vec<RunEvent>,
    pub(super) stop: Option<(String, ToolOutcome)>,
}

pub(super) struct CompletedStepContext<'a> {
    pub(super) step: u32,
    pub(super) target: &'a ModelTarget,
    pub(super) tool_handling: ToolHandling,
    pub(super) tools: &'a ToolSet,
    pub(super) approval_decider: &'a Arc<dyn ApprovalDecider>,
    pub(super) cancellation: &'a Cancellation,
    pub(super) deadline: Instant,
    pub(super) budget_limits: &'a RunBudget,
    pub(super) current_budget: &'a BudgetLedger,
    pub(super) current_journal: &'a ToolJournal,
    pub(super) outcome_policy: ToolOutcomePolicy,
}

pub(super) enum CompletedStepPlanningError {
    Budget(BudgetError),
    Failed(Error),
    TimedOut(RunTimeoutKind),
    Cancelled,
}

pub(super) async fn plan_completed_step(
    context: CompletedStepContext<'_>,
    response: LanguageResponse,
    provider_commit: ProviderDeferredCommit,
) -> Result<CompletedStepPlan, CompletedStepPlanningError> {
    let CompletedStepContext {
        step,
        target,
        tool_handling,
        tools,
        approval_decider,
        cancellation,
        deadline,
        budget_limits,
        current_budget,
        current_journal,
        outcome_policy,
    } = context;
    let (provider_ledger, provider_step, provider_scope, provider_pending) =
        provider_commit.into_parts();
    debug_assert_eq!(provider_step, step);
    debug_assert_eq!(&provider_scope, target.scope());

    let assistant_message = response.project_assistant_history().into_message();
    let mut budget = current_budget.clone();
    let mut journal = None;
    let mut prepared = Vec::new();
    let mut requests = Vec::new();
    let mut immediate_results = Vec::new();
    let mut pending_approvals = Vec::new();
    let mut events = Vec::new();
    let mut stop = None;

    if tool_handling == ToolHandling::ObserveOnly {
        return Ok(CompletedStepPlan {
            response,
            assistant_message,
            budget,
            journal,
            provider_ledger,
            provider_pending,
            prepared,
            requests,
            immediate_results,
            pending_approvals,
            events,
            stop,
        });
    }

    let calls = response
        .content()
        .iter()
        .filter_map(|part| match part {
            siumai_core::ContentPart::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut call_ids = BTreeSet::new();
    let mut resolved = Vec::with_capacity(calls.len());

    // Resolve and validate the whole local-call batch before requesting any
    // approval or writing a durable journal event.
    for (ordinal, call) in calls.into_iter().enumerate() {
        if !call_ids.insert(call.id().to_string()) {
            return Err(CompletedStepPlanningError::Failed(Error::new(
                ErrorKind::InvalidInput,
                "completed model step contains duplicate local tool call identifiers",
            )));
        }
        match tools.resolve(call.clone()) {
            Ok(request) => {
                request.validate().map_err(planning_tool_error)?;
                resolved.push((ordinal, call, Some(request)));
            }
            Err(error @ ToolExecutionError::UnknownLocalTool { .. }) => {
                let outcome = execution_error_outcome(error);
                let result = ToolResult {
                    call_id: call.id().to_string(),
                    name: call.name().to_string(),
                    outcome: outcome.clone(),
                };
                resolved.push((ordinal, call, None));
                if stop.is_none() {
                    charge_result(&mut budget, budget_limits, &result)?;
                    events.push(RunEvent::ToolCompleted {
                        step,
                        ordinal,
                        result: result.clone(),
                    });
                    immediate_results.push(IndexedResult {
                        ordinal,
                        result: result.clone(),
                    });
                    if outcome_policy.action(&outcome) == ToolOutcomeAction::Stop {
                        stop = Some((result.call_id, outcome));
                    }
                }
            }
            Err(error) => return Err(planning_tool_error(error)),
        }
    }

    if stop.is_some() {
        return Ok(CompletedStepPlan {
            response,
            assistant_message,
            budget,
            journal,
            provider_ledger,
            provider_pending,
            prepared,
            requests,
            immediate_results,
            pending_approvals,
            events,
            stop,
        });
    }

    for (ordinal, call, request) in resolved {
        let Some(request) = request else {
            continue;
        };
        let argument_bytes = request.call().input().encoded_json_bytes();
        budget
            .charge_tool_call(argument_bytes, budget_limits)
            .map_err(CompletedStepPlanningError::Budget)?;
        let prepared_tool = staged_journal(&mut journal, current_journal)
            .prepare(step, ordinal, &request)
            .map_err(planning_journal_error)?;
        prepared.push(prepared_tool);
        events.push(RunEvent::ToolPrepared {
            step,
            ordinal,
            call,
        });

        match request.approval_policy() {
            ApprovalPolicy::NotRequired => {
                let call = request
                    .authorize_not_required()
                    .map_err(planning_authorization_error)?;
                requests.push(IndexedAuthorizedCall { ordinal, call });
            }
            ApprovalPolicy::Required => {
                let approval_request =
                    ApprovalRequest::from_frozen(step, ordinal, target.clone(), &request);
                match wait_for(
                    approval_decider.decide(&approval_request),
                    cancellation,
                    deadline,
                    RunTimeoutKind::Total,
                )
                .await
                {
                    WaitResult::Ready(Ok(ApprovalDecision::Approve)) => {
                        requests.push(IndexedAuthorizedCall {
                            ordinal,
                            call: request.authorize_host_auto_approved(),
                        });
                    }
                    WaitResult::Ready(Ok(ApprovalDecision::Deny(denial))) => {
                        let result = ToolResult {
                            call_id: request.call_id().to_string(),
                            name: request.name().to_string(),
                            outcome: ToolOutcome::Denied {
                                reason: denial.reason().to_owned(),
                            },
                        };
                        charge_result(&mut budget, budget_limits, &result)?;
                        staged_journal(&mut journal, current_journal)
                            .complete(&request, &result)
                            .map_err(planning_journal_error)?;
                        events.push(RunEvent::ToolCompleted {
                            step,
                            ordinal,
                            result: result.clone(),
                        });
                        immediate_results.push(IndexedResult {
                            ordinal,
                            result: result.clone(),
                        });
                        if outcome_policy.action(&result.outcome) == ToolOutcomeAction::Stop {
                            stop = Some((result.call_id.clone(), result.outcome.clone()));
                            break;
                        }
                    }
                    WaitResult::Ready(Ok(ApprovalDecision::AwaitExternal)) => {
                        budget
                            .reserve_pending_approval(budget_limits)
                            .map_err(CompletedStepPlanningError::Budget)?;
                        pending_approvals.push(PendingApprovalCall { request });
                    }
                    WaitResult::Ready(Err(error)) => {
                        return Err(CompletedStepPlanningError::Failed(
                            Error::new(ErrorKind::Internal, "host approval decision failed")
                                .with_source(error),
                        ));
                    }
                    WaitResult::TimedOut(kind) => {
                        return Err(CompletedStepPlanningError::TimedOut(kind));
                    }
                    WaitResult::Cancelled => {
                        return Err(CompletedStepPlanningError::Cancelled);
                    }
                }
            }
        }
    }

    Ok(CompletedStepPlan {
        response,
        assistant_message,
        budget,
        journal,
        provider_ledger,
        provider_pending,
        prepared,
        requests,
        immediate_results,
        pending_approvals,
        events,
        stop,
    })
}

fn staged_journal<'a>(
    staged: &'a mut Option<ToolJournal>,
    current: &ToolJournal,
) -> &'a mut ToolJournal {
    staged.get_or_insert_with(|| current.clone())
}

fn planning_tool_error(error: ToolExecutionError) -> CompletedStepPlanningError {
    CompletedStepPlanningError::Failed(
        Error::new(
            ErrorKind::InvalidInput,
            "completed model step contains invalid local tool work",
        )
        .with_source(error),
    )
}

fn planning_journal_error(error: ToolJournalError) -> CompletedStepPlanningError {
    CompletedStepPlanningError::Failed(
        Error::new(
            ErrorKind::Internal,
            "completed model step could not be journaled",
        )
        .with_source(error),
    )
}

fn planning_authorization_error(
    error: impl std::error::Error + Send + Sync + 'static,
) -> CompletedStepPlanningError {
    CompletedStepPlanningError::Failed(
        Error::new(
            ErrorKind::Authorization,
            "completed model step could not authorize local tool work",
        )
        .with_source(error),
    )
}

fn charge_result(
    budget: &mut BudgetLedger,
    limits: &RunBudget,
    result: &ToolResult,
) -> Result<(), CompletedStepPlanningError> {
    budget
        .charge_tool_result(encoded_json_bytes(result)?, limits)
        .map_err(CompletedStepPlanningError::Budget)
}

fn encoded_json_bytes(value: &impl serde::Serialize) -> Result<usize, CompletedStepPlanningError> {
    serde_json::to_vec(value)
        .map(|value| value.len())
        .map_err(|error| {
            CompletedStepPlanningError::Failed(
                Error::new(
                    ErrorKind::Internal,
                    "completed model step could not be measured for budget accounting",
                )
                .with_source(error),
            )
        })
}
