//! Validated limits and accounting for one high-level run.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use siumai_core::{Usage, UsageValue};
use thiserror::Error;

/// The bounded resource that stopped a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BudgetKind {
    ModelSteps,
    ToolCalls,
    ArgumentBytesPerCall,
    ArgumentBytesTotal,
    ResultBytesPerCall,
    ResultBytesTotal,
    SnapshotBytes,
    PendingApprovals,
    KnownTokens,
    KnownCostMicrounits,
}

/// Invalid budget configuration or an exhausted run limit.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum BudgetError {
    #[error("run budget field `{field}` must be greater than zero")]
    InvalidZero { field: &'static str },
    #[error("run budget exceeded {kind:?}: actual {actual}, maximum {maximum}")]
    Exceeded {
        kind: BudgetKind,
        actual: u64,
        maximum: u64,
    },
    #[error("run budget accounting overflowed for {kind:?}")]
    Overflow { kind: BudgetKind },
}

/// Timeout classes owned by the high-level runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunTimeouts {
    total: Duration,
    model_step: Duration,
    first_chunk: Duration,
    inter_chunk: Duration,
    tool: Duration,
}

impl RunTimeouts {
    pub fn new(
        total: Duration,
        model_step: Duration,
        first_chunk: Duration,
        inter_chunk: Duration,
        tool: Duration,
    ) -> Result<Self, BudgetError> {
        validate_duration("total_timeout", total)?;
        validate_duration("model_step_timeout", model_step)?;
        validate_duration("first_chunk_timeout", first_chunk)?;
        validate_duration("inter_chunk_timeout", inter_chunk)?;
        validate_duration("tool_timeout", tool)?;
        Ok(Self {
            total,
            model_step,
            first_chunk,
            inter_chunk,
            tool,
        })
    }

    pub fn total(self) -> Duration {
        self.total
    }

    pub fn model_step(self) -> Duration {
        self.model_step
    }

    pub fn first_chunk(self) -> Duration {
        self.first_chunk
    }

    pub fn inter_chunk(self) -> Duration {
        self.inter_chunk
    }

    pub fn tool(self) -> Duration {
        self.tool
    }
}

impl Default for RunTimeouts {
    fn default() -> Self {
        Self {
            total: Duration::from_secs(10 * 60),
            model_step: Duration::from_secs(2 * 60),
            first_chunk: Duration::from_secs(30),
            inter_chunk: Duration::from_secs(60),
            tool: Duration::from_secs(2 * 60),
        }
    }
}

/// Immutable limits for one tool-loop or structured-output run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunBudget {
    max_model_steps: u32,
    max_tool_calls: u32,
    max_concurrent_tools: usize,
    max_argument_bytes_per_call: usize,
    max_argument_bytes_total: usize,
    max_result_bytes_per_call: usize,
    max_result_bytes_total: usize,
    max_snapshot_bytes: usize,
    max_pending_approvals: u32,
    max_known_tokens: Option<u64>,
    max_known_cost_microunits: Option<u64>,
    timeouts: RunTimeouts,
}

impl RunBudget {
    pub fn builder() -> RunBudgetBuilder {
        RunBudgetBuilder::default()
    }

    pub fn max_model_steps(&self) -> u32 {
        self.max_model_steps
    }

    pub fn max_tool_calls(&self) -> u32 {
        self.max_tool_calls
    }

    pub fn max_concurrent_tools(&self) -> usize {
        self.max_concurrent_tools
    }

    pub fn max_argument_bytes_per_call(&self) -> usize {
        self.max_argument_bytes_per_call
    }

    pub fn max_argument_bytes_total(&self) -> usize {
        self.max_argument_bytes_total
    }

    pub fn max_result_bytes_per_call(&self) -> usize {
        self.max_result_bytes_per_call
    }

    pub fn max_result_bytes_total(&self) -> usize {
        self.max_result_bytes_total
    }

    pub fn max_snapshot_bytes(&self) -> usize {
        self.max_snapshot_bytes
    }

    pub fn max_pending_approvals(&self) -> u32 {
        self.max_pending_approvals
    }

    pub fn max_known_tokens(&self) -> Option<u64> {
        self.max_known_tokens
    }

    pub fn max_known_cost_microunits(&self) -> Option<u64> {
        self.max_known_cost_microunits
    }

    pub fn timeouts(&self) -> RunTimeouts {
        self.timeouts
    }

    pub fn ledger(&self) -> BudgetLedger {
        BudgetLedger::default()
    }

    /// Select the earliest total deadline without extending a caller deadline.
    pub fn deadline_from(&self, now: Instant, caller_deadline: Option<Instant>) -> Instant {
        let budget_deadline = now.checked_add(self.timeouts.total).unwrap_or(now);
        caller_deadline.map_or(budget_deadline, |deadline| deadline.min(budget_deadline))
    }
}

impl Default for RunBudget {
    fn default() -> Self {
        RunBudgetBuilder::default()
            .build()
            .expect("default run budget must remain valid")
    }
}

/// Mutable construction for [`RunBudget`].
#[derive(Debug, Clone)]
pub struct RunBudgetBuilder {
    budget: RunBudget,
}

impl Default for RunBudgetBuilder {
    fn default() -> Self {
        Self {
            budget: RunBudget {
                max_model_steps: 16,
                max_tool_calls: 64,
                max_concurrent_tools: 4,
                max_argument_bytes_per_call: 256 * 1024,
                max_argument_bytes_total: 2 * 1024 * 1024,
                max_result_bytes_per_call: 1024 * 1024,
                max_result_bytes_total: 8 * 1024 * 1024,
                max_snapshot_bytes: 16 * 1024 * 1024,
                max_pending_approvals: 32,
                max_known_tokens: None,
                max_known_cost_microunits: None,
                timeouts: RunTimeouts::default(),
            },
        }
    }
}

impl RunBudgetBuilder {
    pub fn max_model_steps(mut self, value: u32) -> Self {
        self.budget.max_model_steps = value;
        self
    }

    pub fn max_tool_calls(mut self, value: u32) -> Self {
        self.budget.max_tool_calls = value;
        self
    }

    pub fn max_concurrent_tools(mut self, value: usize) -> Self {
        self.budget.max_concurrent_tools = value;
        self
    }

    pub fn max_argument_bytes(mut self, per_call: usize, total: usize) -> Self {
        self.budget.max_argument_bytes_per_call = per_call;
        self.budget.max_argument_bytes_total = total;
        self
    }

    pub fn max_result_bytes(mut self, per_call: usize, total: usize) -> Self {
        self.budget.max_result_bytes_per_call = per_call;
        self.budget.max_result_bytes_total = total;
        self
    }

    pub fn max_snapshot_bytes(mut self, value: usize) -> Self {
        self.budget.max_snapshot_bytes = value;
        self
    }

    pub fn max_pending_approvals(mut self, value: u32) -> Self {
        self.budget.max_pending_approvals = value;
        self
    }

    pub fn max_known_tokens(mut self, value: Option<u64>) -> Self {
        self.budget.max_known_tokens = value;
        self
    }

    pub fn max_known_cost_microunits(mut self, value: Option<u64>) -> Self {
        self.budget.max_known_cost_microunits = value;
        self
    }

    pub fn timeouts(mut self, value: RunTimeouts) -> Self {
        self.budget.timeouts = value;
        self
    }

    pub fn build(self) -> Result<RunBudget, BudgetError> {
        validate_nonzero("max_model_steps", self.budget.max_model_steps as u64)?;
        validate_nonzero("max_tool_calls", self.budget.max_tool_calls as u64)?;
        validate_nonzero(
            "max_concurrent_tools",
            self.budget.max_concurrent_tools as u64,
        )?;
        validate_nonzero(
            "max_argument_bytes_per_call",
            self.budget.max_argument_bytes_per_call as u64,
        )?;
        validate_nonzero(
            "max_argument_bytes_total",
            self.budget.max_argument_bytes_total as u64,
        )?;
        validate_nonzero(
            "max_result_bytes_per_call",
            self.budget.max_result_bytes_per_call as u64,
        )?;
        validate_nonzero(
            "max_result_bytes_total",
            self.budget.max_result_bytes_total as u64,
        )?;
        validate_nonzero("max_snapshot_bytes", self.budget.max_snapshot_bytes as u64)?;
        validate_nonzero(
            "max_pending_approvals",
            self.budget.max_pending_approvals as u64,
        )?;
        validate_duration("total_timeout", self.budget.timeouts.total)?;
        validate_duration("model_step_timeout", self.budget.timeouts.model_step)?;
        validate_duration("first_chunk_timeout", self.budget.timeouts.first_chunk)?;
        validate_duration("inter_chunk_timeout", self.budget.timeouts.inter_chunk)?;
        validate_duration("tool_timeout", self.budget.timeouts.tool)?;
        Ok(self.budget)
    }
}

/// Serializable resource consumption recorded in a run snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetLedger {
    model_steps: u32,
    tool_calls: u32,
    argument_bytes: u64,
    result_bytes: u64,
    pending_approvals: u32,
    known_tokens: u64,
    usage_steps_with_unknown_tokens: u32,
    known_cost_microunits: u64,
}

impl BudgetLedger {
    pub fn model_steps(&self) -> u32 {
        self.model_steps
    }

    pub fn tool_calls(&self) -> u32 {
        self.tool_calls
    }

    pub fn argument_bytes(&self) -> u64 {
        self.argument_bytes
    }

    pub fn result_bytes(&self) -> u64 {
        self.result_bytes
    }

    pub fn pending_approvals(&self) -> u32 {
        self.pending_approvals
    }

    pub fn known_tokens(&self) -> u64 {
        self.known_tokens
    }

    pub fn usage_steps_with_unknown_tokens(&self) -> u32 {
        self.usage_steps_with_unknown_tokens
    }

    pub fn known_cost_microunits(&self) -> u64 {
        self.known_cost_microunits
    }

    pub fn charge_model_step(&mut self, budget: &RunBudget) -> Result<(), BudgetError> {
        let next = checked_increment(self.model_steps, BudgetKind::ModelSteps)?;
        enforce(
            BudgetKind::ModelSteps,
            next as u64,
            budget.max_model_steps as u64,
        )?;
        self.model_steps = next;
        Ok(())
    }

    pub fn charge_tool_call(
        &mut self,
        argument_bytes: usize,
        budget: &RunBudget,
    ) -> Result<(), BudgetError> {
        enforce(
            BudgetKind::ArgumentBytesPerCall,
            usize_to_u64(argument_bytes, BudgetKind::ArgumentBytesPerCall)?,
            usize_to_u64(
                budget.max_argument_bytes_per_call,
                BudgetKind::ArgumentBytesPerCall,
            )?,
        )?;
        let next_calls = checked_increment(self.tool_calls, BudgetKind::ToolCalls)?;
        enforce(
            BudgetKind::ToolCalls,
            next_calls as u64,
            budget.max_tool_calls as u64,
        )?;
        let next_argument_bytes = checked_add(
            self.argument_bytes,
            usize_to_u64(argument_bytes, BudgetKind::ArgumentBytesTotal)?,
            BudgetKind::ArgumentBytesTotal,
        )?;
        enforce(
            BudgetKind::ArgumentBytesTotal,
            next_argument_bytes,
            usize_to_u64(
                budget.max_argument_bytes_total,
                BudgetKind::ArgumentBytesTotal,
            )?,
        )?;
        self.tool_calls = next_calls;
        self.argument_bytes = next_argument_bytes;
        Ok(())
    }

    pub fn charge_tool_result(
        &mut self,
        result_bytes: usize,
        budget: &RunBudget,
    ) -> Result<(), BudgetError> {
        enforce(
            BudgetKind::ResultBytesPerCall,
            usize_to_u64(result_bytes, BudgetKind::ResultBytesPerCall)?,
            usize_to_u64(
                budget.max_result_bytes_per_call,
                BudgetKind::ResultBytesPerCall,
            )?,
        )?;
        let next = checked_add(
            self.result_bytes,
            usize_to_u64(result_bytes, BudgetKind::ResultBytesTotal)?,
            BudgetKind::ResultBytesTotal,
        )?;
        enforce(
            BudgetKind::ResultBytesTotal,
            next,
            usize_to_u64(budget.max_result_bytes_total, BudgetKind::ResultBytesTotal)?,
        )?;
        self.result_bytes = next;
        Ok(())
    }

    pub fn reserve_pending_approval(&mut self, budget: &RunBudget) -> Result<(), BudgetError> {
        let next = checked_increment(self.pending_approvals, BudgetKind::PendingApprovals)?;
        enforce(
            BudgetKind::PendingApprovals,
            next as u64,
            budget.max_pending_approvals as u64,
        )?;
        self.pending_approvals = next;
        Ok(())
    }

    pub fn release_pending_approval(&mut self) {
        self.pending_approvals = self.pending_approvals.saturating_sub(1);
    }

    pub fn charge_usage(&mut self, usage: &Usage, budget: &RunBudget) -> Result<(), BudgetError> {
        match usage.total_tokens {
            UsageValue::Known(tokens) => {
                let next = checked_add(self.known_tokens, tokens, BudgetKind::KnownTokens)?;
                if let Some(maximum) = budget.max_known_tokens {
                    enforce(BudgetKind::KnownTokens, next, maximum)?;
                }
                self.known_tokens = next;
            }
            UsageValue::Unknown => {
                let next = checked_increment(
                    self.usage_steps_with_unknown_tokens,
                    BudgetKind::KnownTokens,
                )?;
                self.usage_steps_with_unknown_tokens = next;
            }
        }
        Ok(())
    }

    pub fn charge_known_cost(
        &mut self,
        microunits: u64,
        budget: &RunBudget,
    ) -> Result<(), BudgetError> {
        let next = checked_add(
            self.known_cost_microunits,
            microunits,
            BudgetKind::KnownCostMicrounits,
        )?;
        if let Some(maximum) = budget.max_known_cost_microunits {
            enforce(BudgetKind::KnownCostMicrounits, next, maximum)?;
        }
        self.known_cost_microunits = next;
        Ok(())
    }
}

fn validate_nonzero(field: &'static str, value: u64) -> Result<(), BudgetError> {
    if value == 0 {
        Err(BudgetError::InvalidZero { field })
    } else {
        Ok(())
    }
}

fn validate_duration(field: &'static str, value: Duration) -> Result<(), BudgetError> {
    if value.is_zero() {
        Err(BudgetError::InvalidZero { field })
    } else {
        Ok(())
    }
}

fn enforce(kind: BudgetKind, actual: u64, maximum: u64) -> Result<(), BudgetError> {
    if actual > maximum {
        Err(BudgetError::Exceeded {
            kind,
            actual,
            maximum,
        })
    } else {
        Ok(())
    }
}

fn checked_increment(value: u32, kind: BudgetKind) -> Result<u32, BudgetError> {
    value.checked_add(1).ok_or(BudgetError::Overflow { kind })
}

fn checked_add(left: u64, right: u64, kind: BudgetKind) -> Result<u64, BudgetError> {
    left.checked_add(right)
        .ok_or(BudgetError::Overflow { kind })
}

fn usize_to_u64(value: usize, kind: BudgetKind) -> Result<u64, BudgetError> {
    u64::try_from(value).map_err(|_| BudgetError::Overflow { kind })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_are_checked_before_accounting_can_continue() {
        let budget = RunBudget::builder()
            .max_model_steps(1)
            .max_tool_calls(1)
            .max_argument_bytes(4, 4)
            .max_result_bytes(4, 4)
            .build()
            .unwrap();
        let mut ledger = budget.ledger();

        ledger.charge_model_step(&budget).unwrap();
        assert!(matches!(
            ledger.charge_model_step(&budget),
            Err(BudgetError::Exceeded {
                kind: BudgetKind::ModelSteps,
                ..
            })
        ));
        ledger.charge_tool_call(4, &budget).unwrap();
        assert!(matches!(
            ledger.charge_tool_result(5, &budget),
            Err(BudgetError::Exceeded {
                kind: BudgetKind::ResultBytesPerCall,
                ..
            })
        ));
    }

    #[test]
    fn unknown_usage_is_not_invented_as_zero() {
        let budget = RunBudget::builder()
            .max_known_tokens(Some(4))
            .build()
            .unwrap();
        let mut ledger = budget.ledger();

        ledger.charge_usage(&Usage::default(), &budget).unwrap();
        assert_eq!(ledger.known_tokens(), 0);
        assert_eq!(ledger.usage_steps_with_unknown_tokens(), 1);

        let usage = Usage::default().with_total_tokens(5_u64);
        assert!(matches!(
            ledger.charge_usage(&usage, &budget),
            Err(BudgetError::Exceeded {
                kind: BudgetKind::KnownTokens,
                ..
            })
        ));
    }

    #[test]
    fn caller_deadline_is_never_extended() {
        let budget = RunBudget::default();
        let now = Instant::now();
        let caller = now + Duration::from_secs(1);
        assert_eq!(budget.deadline_from(now, Some(caller)), caller);
    }

    #[test]
    fn timeout_matrix_rejects_zero_durations_at_construction() {
        let error = RunTimeouts::new(
            Duration::from_secs(1),
            Duration::from_secs(1),
            Duration::ZERO,
            Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .unwrap_err();

        assert_eq!(
            error,
            BudgetError::InvalidZero {
                field: "first_chunk_timeout"
            }
        );
    }
}
