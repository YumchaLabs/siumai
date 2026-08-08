use serde::Serialize;
use serde_json::{Map, Value, json};
use siumai_core::{
    ContentPart, LanguageResponse, LanguageStreamEvent, ModelId, StreamTerminal, Usage,
};
use siumai_runtime::{RunEvent, RunReport, RunTerminal, StepRecord};
use thiserror::Error;

use crate::{GatewayLossPolicy, GatewayPolicy};

/// Bounded diagnostic describing one intentionally omitted value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GatewayLoss {
    code: &'static str,
    path: String,
}

impl GatewayLoss {
    fn new(code: &'static str, path: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
        }
    }

    pub const fn code(&self) -> &'static str {
        self.code
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Failure to project a canonical value into the public gateway vocabulary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GatewayProjectionError {
    #[error("gateway run target does not match the authenticated route")]
    TrustRouteMismatch,
    #[error("gateway projection rejected non-portable data at `{path}`")]
    LossRejected { code: &'static str, path: String },
    #[error("gateway projection serialization failed for {context}")]
    Serialization {
        context: &'static str,
        #[source]
        source: serde_json::Error,
    },
}

impl GatewayProjectionError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::TrustRouteMismatch => "trust_route_mismatch",
            Self::LossRejected { .. } => "projection_loss_rejected",
            Self::Serialization { .. } => "projection_serialization_failed",
        }
    }
}

/// Stable JSON envelope used by server adapters.
#[derive(Debug, Clone, Serialize)]
pub struct GatewayEvent {
    #[serde(rename = "type")]
    kind: &'static str,
    data: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    losses: Vec<GatewayLoss>,
    #[serde(skip)]
    terminal: bool,
}

impl GatewayEvent {
    pub fn kind(&self) -> &'static str {
        self.kind
    }

    pub fn data(&self) -> &Value {
        &self.data
    }

    pub fn losses(&self) -> &[GatewayLoss] {
        &self.losses
    }

    pub(crate) const fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub fn from_language_response(
        response: LanguageResponse,
        policy: &GatewayPolicy,
    ) -> Result<Self, GatewayProjectionError> {
        let (response, losses) = project_language_response(&response, policy.loss_policy())?;
        Ok(Self::terminal(
            "model.response",
            json!({ "response": response }),
            losses,
        ))
    }

    pub fn from_language(
        event: LanguageStreamEvent,
        policy: &GatewayPolicy,
    ) -> Result<Self, GatewayProjectionError> {
        match event {
            LanguageStreamEvent::Started { id, model } => Ok(Self::event(
                "model.started",
                json!({ "id": id, "model": model }),
            )),
            LanguageStreamEvent::TextStart { id } => {
                Ok(Self::event("model.text.started", json!({ "id": id })))
            }
            LanguageStreamEvent::TextDelta { id, delta } => Ok(Self::event(
                "model.text.delta",
                json!({ "id": id, "delta": delta }),
            )),
            LanguageStreamEvent::TextEnd { id } => {
                Ok(Self::event("model.text.finished", json!({ "id": id })))
            }
            LanguageStreamEvent::ReasoningStart { id } => {
                Ok(Self::event("model.reasoning.started", json!({ "id": id })))
            }
            LanguageStreamEvent::ReasoningDelta { id, delta } => Ok(Self::event(
                "model.reasoning.delta",
                json!({ "id": id, "delta": delta }),
            )),
            LanguageStreamEvent::ReasoningEnd { id } => {
                Ok(Self::event("model.reasoning.finished", json!({ "id": id })))
            }
            LanguageStreamEvent::ToolInputStart { id, name, owner } => Ok(Self::event(
                "model.tool-input.started",
                json!({ "id": id, "name": name, "owner": owner }),
            )),
            LanguageStreamEvent::ToolInputDelta { id, delta } => Ok(Self::event(
                "model.tool-input.delta",
                json!({ "id": id, "delta": delta }),
            )),
            LanguageStreamEvent::ToolCall(call) => {
                Ok(Self::event("model.tool-call", json!({ "call": call })))
            }
            LanguageStreamEvent::ToolResult(result) => Ok(Self::event(
                "model.tool-result",
                json!({ "result": result }),
            )),
            LanguageStreamEvent::Citation(citation) => Ok(Self::event(
                "model.citation",
                json!({ "citation": citation }),
            )),
            LanguageStreamEvent::Refusal { reason } => {
                Ok(Self::event("model.refusal", json!({ "reason": reason })))
            }
            LanguageStreamEvent::ProviderDeferred { id, .. } => loss_event(
                policy.loss_policy(),
                "provider_deferred_omitted",
                "event.state",
                "model.projection.loss",
                json!({ "id": id }),
                false,
            ),
            LanguageStreamEvent::ProviderOpaque(_) => loss_event(
                policy.loss_policy(),
                "provider_opaque_omitted",
                "event.item",
                "model.projection.loss",
                Value::Null,
                false,
            ),
            LanguageStreamEvent::Usage(usage) => {
                Ok(Self::event("model.usage", json!({ "usage": usage })))
            }
            LanguageStreamEvent::Terminal(terminal) => {
                Self::from_language_terminal(terminal, policy.loss_policy())
            }
            _ => loss_event(
                policy.loss_policy(),
                "unknown_language_event_omitted",
                "event",
                "model.projection.loss",
                Value::Null,
                false,
            ),
        }
    }

    pub fn from_run_terminal(
        terminal: RunTerminal,
        policy: &GatewayPolicy,
    ) -> Result<Self, GatewayProjectionError> {
        Self::project_run_terminal(terminal, policy.loss_policy())
    }

    pub fn from_run(
        event: RunEvent,
        policy: &GatewayPolicy,
    ) -> Result<Self, GatewayProjectionError> {
        match event {
            RunEvent::Started { target } => {
                Ok(Self::event("run.started", json!({ "target": target })))
            }
            RunEvent::StepStarted { index, target } => Ok(Self::event(
                "run.step.started",
                json!({ "index": index, "target": target }),
            )),
            RunEvent::Model { step, event } => {
                let projected = Self::from_language(event, policy)?;
                let losses = projected.losses.clone();
                let event = to_value(&projected, "nested language event")?;
                Ok(Self::with_losses(
                    "run.model",
                    json!({ "step": step, "event": event }),
                    losses,
                    false,
                ))
            }
            RunEvent::ToolPrepared {
                step,
                ordinal,
                call,
            } => Ok(Self::event(
                "run.tool.prepared",
                json!({ "step": step, "ordinal": ordinal, "call": call }),
            )),
            RunEvent::ToolCompleted {
                step,
                ordinal,
                result,
            } => Ok(Self::event(
                "run.tool.completed",
                json!({ "step": step, "ordinal": ordinal, "result": result }),
            )),
            RunEvent::StepFinished { record } => {
                let (record, losses) = project_step_record(&record, policy.loss_policy())?;
                Ok(Self::with_losses(
                    "run.step.finished",
                    json!({ "record": record }),
                    losses,
                    false,
                ))
            }
            RunEvent::ModelTransition { transition } => Ok(Self::event(
                "run.model.transition",
                json!({ "transition": transition }),
            )),
            RunEvent::Terminal(terminal) => {
                Self::project_run_terminal(terminal, policy.loss_policy())
            }
            _ => loss_event(
                policy.loss_policy(),
                "unknown_run_event_omitted",
                "event",
                "run.projection.loss",
                Value::Null,
                false,
            ),
        }
    }

    pub(crate) fn projection_failed(error: &GatewayProjectionError) -> Self {
        Self::terminal(
            "server.projection.failed",
            json!({
                "error": {
                    "code": error.code(),
                    "message": "server projection failed",
                }
            }),
            Vec::new(),
        )
    }

    pub(crate) fn stream_failed(code: &'static str, message: &'static str) -> Self {
        Self::terminal(
            "server.stream.failed",
            json!({ "error": { "code": code, "message": message } }),
            Vec::new(),
        )
    }

    fn from_language_terminal(
        terminal: StreamTerminal,
        loss_policy: GatewayLossPolicy,
    ) -> Result<Self, GatewayProjectionError> {
        match terminal {
            StreamTerminal::Completed { response } => {
                let (response, losses) = project_language_response(&response, loss_policy)?;
                Ok(Self::terminal(
                    "model.completed",
                    json!({ "status": "completed", "response": response }),
                    losses,
                ))
            }
            StreamTerminal::Failed { error, response } => {
                let (response, losses) =
                    project_optional_response(response.as_deref(), loss_policy)?;
                Ok(Self::terminal(
                    "model.failed",
                    json!({
                        "status": "failed",
                        "error": public_error(&error),
                        "response": response,
                    }),
                    losses,
                ))
            }
            StreamTerminal::Cancelled { reason, response } => {
                let (response, losses) =
                    project_optional_response(response.as_deref(), loss_policy)?;
                Ok(Self::terminal(
                    "model.cancelled",
                    json!({
                        "status": "cancelled",
                        "reason": reason,
                        "response": response,
                    }),
                    losses,
                ))
            }
            _ => loss_event(
                loss_policy,
                "unknown_language_terminal",
                "terminal",
                "model.terminal.unavailable",
                Value::Null,
                true,
            ),
        }
    }

    fn project_run_terminal(
        terminal: RunTerminal,
        loss_policy: GatewayLossPolicy,
    ) -> Result<Self, GatewayProjectionError> {
        match terminal {
            RunTerminal::Completed { report } => {
                let (report, losses) = project_run_report(&report, loss_policy)?;
                Ok(Self::terminal(
                    "run.completed",
                    json!({ "status": "completed", "report": report }),
                    losses,
                ))
            }
            RunTerminal::Stopped { reason, report } => {
                let (report, losses) = project_run_report(&report, loss_policy)?;
                Ok(Self::terminal(
                    "run.stopped",
                    json!({ "status": "stopped", "reason": reason, "report": report }),
                    losses,
                ))
            }
            RunTerminal::Suspended { reason, report } => {
                let (report, losses) = project_run_report(&report, loss_policy)?;
                Ok(Self::terminal(
                    "run.suspended",
                    json!({ "status": "suspended", "reason": reason, "report": report }),
                    losses,
                ))
            }
            RunTerminal::BudgetExceeded { error, report } => {
                let (report, losses) = project_run_report(&report, loss_policy)?;
                Ok(Self::terminal(
                    "run.budget-exceeded",
                    json!({
                        "status": "budget_exceeded",
                        "error": error.to_string(),
                        "report": report,
                    }),
                    losses,
                ))
            }
            RunTerminal::TimedOut { kind, report } => {
                let (report, losses) = project_run_report(&report, loss_policy)?;
                Ok(Self::terminal(
                    "run.timed-out",
                    json!({ "status": "timed_out", "kind": kind, "report": report }),
                    losses,
                ))
            }
            RunTerminal::Indeterminate { effect, report } => {
                let (report, losses) = project_run_report(&report, loss_policy)?;
                Ok(Self::terminal(
                    "run.indeterminate",
                    json!({
                        "status": "indeterminate",
                        "effect": effect,
                        "report": report,
                    }),
                    losses,
                ))
            }
            RunTerminal::HistoryProjectionRejected { transition, report } => {
                let (report, losses) = project_run_report(&report, loss_policy)?;
                Ok(Self::terminal(
                    "run.history-projection-rejected",
                    json!({
                        "status": "history_projection_rejected",
                        "transition": transition,
                        "report": report,
                    }),
                    losses,
                ))
            }
            RunTerminal::ResumeConflict { reason } => Ok(Self::terminal(
                "run.resume-conflict",
                json!({ "status": "resume_conflict", "reason": reason }),
                Vec::new(),
            )),
            RunTerminal::Failed { error, report } => {
                let (report, losses) = project_run_report(&report, loss_policy)?;
                Ok(Self::terminal(
                    "run.failed",
                    json!({
                        "status": "failed",
                        "error": public_error(&error),
                        "report": report,
                    }),
                    losses,
                ))
            }
            RunTerminal::Cancelled { reason, report } => {
                let (report, losses) = project_run_report(&report, loss_policy)?;
                Ok(Self::terminal(
                    "run.cancelled",
                    json!({ "status": "cancelled", "reason": reason, "report": report }),
                    losses,
                ))
            }
            _ => loss_event(
                loss_policy,
                "unknown_run_terminal",
                "terminal",
                "run.terminal.unavailable",
                Value::Null,
                true,
            ),
        }
    }

    fn event(kind: &'static str, data: Value) -> Self {
        Self::with_losses(kind, data, Vec::new(), false)
    }

    fn terminal(kind: &'static str, data: Value, losses: Vec<GatewayLoss>) -> Self {
        Self::with_losses(kind, data, losses, true)
    }

    fn with_losses(
        kind: &'static str,
        data: Value,
        losses: Vec<GatewayLoss>,
        terminal: bool,
    ) -> Self {
        Self {
            kind,
            data,
            losses,
            terminal,
        }
    }
}

#[derive(Serialize)]
struct LanguageResponseProjection<'a> {
    id: Option<&'a str>,
    model: Option<&'a ModelId>,
    status: &'a siumai_core::LanguageResponseStatus,
    content: Vec<Value>,
    finish_reason: &'a siumai_core::FinishReason,
    usage: &'a Usage,
    warnings: &'a [siumai_core::Warning],
}

fn project_language_response(
    response: &LanguageResponse,
    loss_policy: GatewayLossPolicy,
) -> Result<(Value, Vec<GatewayLoss>), GatewayProjectionError> {
    let mut losses = Vec::new();
    let mut content = Vec::with_capacity(response.content().len());
    for (index, part) in response.content().iter().enumerate() {
        match part {
            ContentPart::ProviderOpaque(_) => record_loss(
                loss_policy,
                &mut losses,
                "provider_opaque_omitted",
                format!("response.content[{index}]"),
            )?,
            _ => content.push(to_value(part, "language response content")?),
        }
    }
    if !response.provider_metadata().is_empty() {
        record_loss(
            loss_policy,
            &mut losses,
            "provider_metadata_omitted",
            "response.provider",
        )?;
    }

    let projection = LanguageResponseProjection {
        id: response.id(),
        model: response.model(),
        status: response.status(),
        content,
        finish_reason: response.finish_reason(),
        usage: response.usage(),
        warnings: response.warnings(),
    };
    Ok((to_value(&projection, "language response")?, losses))
}

fn project_optional_response(
    response: Option<&LanguageResponse>,
    loss_policy: GatewayLossPolicy,
) -> Result<(Option<Value>, Vec<GatewayLoss>), GatewayProjectionError> {
    match response {
        Some(response) => {
            let (response, losses) = project_language_response(response, loss_policy)?;
            Ok((Some(response), losses))
        }
        None => Ok((None, Vec::new())),
    }
}

fn project_step_record(
    record: &StepRecord,
    loss_policy: GatewayLossPolicy,
) -> Result<(Value, Vec<GatewayLoss>), GatewayProjectionError> {
    let (response, losses) = project_language_response(record.response(), loss_policy)?;
    Ok((
        json!({
            "index": record.index(),
            "target": record.target(),
            "response": response,
            "tool_results": record.tool_results(),
            "assistant_history_omissions": record.assistant_history_omissions(),
        }),
        losses,
    ))
}

fn project_run_report(
    report: &RunReport,
    loss_policy: GatewayLossPolicy,
) -> Result<(Value, Vec<GatewayLoss>), GatewayProjectionError> {
    let mut losses = Vec::new();
    let mut steps = Vec::with_capacity(report.steps().len());
    for step in report.steps() {
        let (step, step_losses) = project_step_record(step, loss_policy)?;
        steps.push(step);
        losses.extend(step_losses);
    }
    if !report.provider_deferred().is_empty() {
        record_loss(
            loss_policy,
            &mut losses,
            "provider_deferred_omitted",
            "report.provider_deferred",
        )?;
    }

    let mut projection = Map::new();
    projection.insert(
        "initial_target".to_string(),
        to_value(report.initial_target(), "run initial target")?,
    );
    projection.insert(
        "current_target".to_string(),
        to_value(report.current_target(), "run current target")?,
    );
    projection.insert("steps".to_string(), Value::Array(steps));
    projection.insert(
        "model_transitions".to_string(),
        to_value(report.model_transitions(), "model transitions")?,
    );
    projection.insert("usage".to_string(), to_value(report.usage(), "run usage")?);
    projection.insert(
        "budget".to_string(),
        to_value(report.budget(), "run budget")?,
    );
    Ok((Value::Object(projection), losses))
}

fn loss_event(
    policy: GatewayLossPolicy,
    code: &'static str,
    path: impl Into<String>,
    kind: &'static str,
    data: Value,
    terminal: bool,
) -> Result<GatewayEvent, GatewayProjectionError> {
    let path = path.into();
    match policy {
        GatewayLossPolicy::Reject => Err(GatewayProjectionError::LossRejected { code, path }),
        GatewayLossPolicy::Report => Ok(GatewayEvent::with_losses(
            kind,
            data,
            vec![GatewayLoss::new(code, path)],
            terminal,
        )),
    }
}

fn record_loss(
    policy: GatewayLossPolicy,
    losses: &mut Vec<GatewayLoss>,
    code: &'static str,
    path: impl Into<String>,
) -> Result<(), GatewayProjectionError> {
    let path = path.into();
    match policy {
        GatewayLossPolicy::Reject => Err(GatewayProjectionError::LossRejected { code, path }),
        GatewayLossPolicy::Report => {
            losses.push(GatewayLoss::new(code, path));
            Ok(())
        }
    }
}

fn to_value<T>(value: &T, context: &'static str) -> Result<Value, GatewayProjectionError>
where
    T: Serialize + ?Sized,
{
    serde_json::to_value(value)
        .map_err(|source| GatewayProjectionError::Serialization { context, source })
}

fn public_error(error: &siumai_core::Error) -> Value {
    json!({
        "kind": format!("{:?}", error.kind()),
        "message": error.message(),
    })
}

#[cfg(test)]
mod tests {
    use siumai_core::{
        AssistantHistoryOmissionKind, FinishReason, LanguageResponseStatus, ModelId,
        OpaqueProviderItem, ProtocolId, ProviderId, ProviderProvenance, ProviderScope,
        ReplayDomain, ReplayDomainId,
    };
    use siumai_runtime::ModelTarget;

    use super::*;

    fn test_provenance() -> ProviderProvenance {
        let scope = ProviderScope::new(ProviderId::new("test").unwrap())
            .with_protocol(ProtocolId::new("test").unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("gateway-test").unwrap(),
            ));
        ProviderProvenance::from_scope(&scope, ModelId::new("test-model").unwrap()).unwrap()
    }

    #[test]
    fn text_delta_projects_to_stable_typed_json() {
        let event = GatewayEvent::from_language(
            LanguageStreamEvent::TextDelta {
                id: "text-1".to_string(),
                delta: "hello".to_string(),
            },
            &GatewayPolicy::default(),
        )
        .unwrap();

        assert_eq!(event.kind(), "model.text.delta");
        assert_eq!(event.data()["id"], "text-1");
        assert_eq!(event.data()["delta"], "hello");
        assert!(!event.is_terminal());
    }

    #[test]
    fn provider_opaque_response_data_is_reported_and_omitted() {
        let item = OpaqueProviderItem::new(
            test_provenance(),
            "encrypted_state",
            json!({ "secret": "never-export" }),
        )
        .unwrap();
        let response = LanguageResponse::new(
            LanguageResponseStatus::Completed,
            vec![ContentPart::ProviderOpaque(item)],
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap();

        let event = GatewayEvent::from_language_response(response, &GatewayPolicy::default())
            .expect("report policy should preserve the portable response");
        let serialized = serde_json::to_string(&event).unwrap();

        assert_eq!(event.losses()[0].code(), "provider_opaque_omitted");
        assert!(!serialized.contains("never-export"));
    }

    #[test]
    fn strict_loss_policy_rejects_provider_opaque_response_data() {
        let item = OpaqueProviderItem::new(
            test_provenance(),
            "encrypted_state",
            json!({ "secret": true }),
        )
        .unwrap();
        let response = LanguageResponse::new(
            LanguageResponseStatus::Completed,
            vec![ContentPart::ProviderOpaque(item)],
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap();
        let policy = GatewayPolicy::default().with_loss_policy(GatewayLossPolicy::Reject);

        assert!(matches!(
            GatewayEvent::from_language_response(response, &policy),
            Err(GatewayProjectionError::LossRejected { .. })
        ));
    }

    #[test]
    fn step_projection_preserves_assistant_history_omissions() {
        let response = LanguageResponse::new(
            LanguageResponseStatus::Completed,
            vec![ContentPart::Refusal {
                reason: Some("policy".to_string()),
            }],
            FinishReason::Refusal,
            Usage::default(),
        )
        .unwrap();
        let record = StepRecord::new(
            0,
            ModelTarget::new(
                ProviderId::new("test").unwrap(),
                ModelId::new("test-model").unwrap(),
            ),
            response,
            Vec::new(),
        );

        let (projection, losses) = project_step_record(&record, GatewayLossPolicy::Report).unwrap();
        assert!(losses.is_empty());
        assert_eq!(
            projection["assistant_history_omissions"][0]["kind"],
            serde_json::to_value(AssistantHistoryOmissionKind::Refusal).unwrap()
        );
        assert_eq!(
            projection["assistant_history_omissions"][0]["content_index"],
            0
        );
    }
}
