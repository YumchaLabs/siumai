use serde::Serialize;
use serde_json::{Value, json};
use siumai_core::{LanguageStreamEvent, StreamTerminal};
use siumai_runtime::{RunEvent, RunTerminal};

/// Stable JSON envelope used by server adapters.
#[derive(Debug, Clone, Serialize)]
pub struct GatewayEvent {
    #[serde(rename = "type")]
    kind: &'static str,
    data: Value,
}

impl GatewayEvent {
    pub fn kind(&self) -> &'static str {
        self.kind
    }

    pub fn data(&self) -> &Value {
        &self.data
    }

    pub fn from_run(event: RunEvent) -> Self {
        match event {
            RunEvent::Started { target } => Self::new("run.started", json!({ "target": target })),
            RunEvent::StepStarted { index, target } => Self::new(
                "run.step.started",
                json!({ "index": index, "target": target }),
            ),
            RunEvent::Model { step, event } => {
                let projected = Self::from_language(event);
                Self::new(
                    "run.model",
                    json!({
                        "step": step,
                        "event": projected,
                    }),
                )
            }
            RunEvent::ToolPrepared {
                step,
                ordinal,
                call,
            } => Self::new(
                "run.tool.prepared",
                json!({ "step": step, "ordinal": ordinal, "call": call }),
            ),
            RunEvent::ToolCompleted {
                step,
                ordinal,
                result,
            } => Self::new(
                "run.tool.completed",
                json!({ "step": step, "ordinal": ordinal, "result": result }),
            ),
            RunEvent::StepFinished { record } => {
                Self::new("run.step.finished", json!({ "record": record }))
            }
            RunEvent::ModelTransition { transition } => {
                Self::new("run.model.transition", json!({ "transition": transition }))
            }
            RunEvent::Terminal(terminal) => Self::from_run_terminal(terminal),
            _ => Self::new("run.unknown", Value::Null),
        }
    }

    pub fn from_language(event: LanguageStreamEvent) -> Self {
        match event {
            LanguageStreamEvent::Started { id, model } => {
                Self::new("model.started", json!({ "id": id, "model": model }))
            }
            LanguageStreamEvent::TextStart { id } => {
                Self::new("model.text.started", json!({ "id": id }))
            }
            LanguageStreamEvent::TextDelta { id, delta } => {
                Self::new("model.text.delta", json!({ "id": id, "delta": delta }))
            }
            LanguageStreamEvent::TextEnd { id } => {
                Self::new("model.text.finished", json!({ "id": id }))
            }
            LanguageStreamEvent::ReasoningStart { id } => {
                Self::new("model.reasoning.started", json!({ "id": id }))
            }
            LanguageStreamEvent::ReasoningDelta { id, delta } => {
                Self::new("model.reasoning.delta", json!({ "id": id, "delta": delta }))
            }
            LanguageStreamEvent::ReasoningEnd { id } => {
                Self::new("model.reasoning.finished", json!({ "id": id }))
            }
            LanguageStreamEvent::ToolInputStart { id, name, owner } => Self::new(
                "model.tool-input.started",
                json!({ "id": id, "name": name, "owner": owner }),
            ),
            LanguageStreamEvent::ToolInputDelta { id, delta } => Self::new(
                "model.tool-input.delta",
                json!({ "id": id, "delta": delta }),
            ),
            LanguageStreamEvent::ToolCall(call) => {
                Self::new("model.tool-call", json!({ "call": call }))
            }
            LanguageStreamEvent::ToolResult(result) => {
                Self::new("model.tool-result", json!({ "result": result }))
            }
            LanguageStreamEvent::Citation(citation) => {
                Self::new("model.citation", json!({ "citation": citation }))
            }
            LanguageStreamEvent::Refusal { reason } => {
                Self::new("model.refusal", json!({ "reason": reason }))
            }
            LanguageStreamEvent::ProviderDeferred { id, state } => Self::new(
                "model.provider-deferred",
                json!({ "id": id, "state": state }),
            ),
            LanguageStreamEvent::ProviderOpaque(item) => {
                Self::new("model.provider-opaque", json!({ "item": item }))
            }
            LanguageStreamEvent::Usage(usage) => {
                Self::new("model.usage", json!({ "usage": usage }))
            }
            LanguageStreamEvent::Terminal(terminal) => Self::from_language_terminal(terminal),
            _ => Self::new("model.unknown", Value::Null),
        }
    }

    fn from_language_terminal(terminal: StreamTerminal) -> Self {
        match terminal {
            StreamTerminal::Completed { response } => Self::new(
                "model.completed",
                json!({ "status": "completed", "response": response }),
            ),
            StreamTerminal::Failed { error, response } => Self::new(
                "model.failed",
                json!({
                    "status": "failed",
                    "error": public_error(&error),
                    "response": response,
                }),
            ),
            StreamTerminal::Cancelled { reason, response } => Self::new(
                "model.cancelled",
                json!({
                    "status": "cancelled",
                    "reason": reason,
                    "response": response,
                }),
            ),
            _ => Self::new("model.terminal.unknown", Value::Null),
        }
    }

    fn from_run_terminal(terminal: RunTerminal) -> Self {
        match terminal {
            RunTerminal::Completed { report } => {
                Self::new("run.completed", json!({ "report": report }))
            }
            RunTerminal::Stopped { reason, report } => {
                Self::new("run.stopped", json!({ "reason": reason, "report": report }))
            }
            RunTerminal::Suspended { reason, report } => Self::new(
                "run.suspended",
                json!({ "reason": reason, "report": report }),
            ),
            RunTerminal::BudgetExceeded { error, report } => Self::new(
                "run.budget-exceeded",
                json!({ "error": error.to_string(), "report": report }),
            ),
            RunTerminal::TimedOut { kind, report } => {
                Self::new("run.timed-out", json!({ "kind": kind, "report": report }))
            }
            RunTerminal::Indeterminate { effect, report } => Self::new(
                "run.indeterminate",
                json!({ "effect": effect, "report": report }),
            ),
            RunTerminal::HistoryProjectionRejected { transition, report } => Self::new(
                "run.history-projection-rejected",
                json!({ "transition": transition, "report": report }),
            ),
            RunTerminal::ResumeConflict { reason } => {
                Self::new("run.resume-conflict", json!({ "reason": reason }))
            }
            RunTerminal::Failed { error, report } => Self::new(
                "run.failed",
                json!({ "error": public_error(&error), "report": report }),
            ),
            RunTerminal::Cancelled { reason, report } => Self::new(
                "run.cancelled",
                json!({ "reason": reason, "report": report }),
            ),
            _ => Self::new("run.terminal.unknown", Value::Null),
        }
    }

    fn new(kind: &'static str, data: Value) -> Self {
        Self { kind, data }
    }
}

fn public_error(error: &siumai_core::Error) -> Value {
    json!({
        "kind": format!("{:?}", error.kind()),
        "message": error.message(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_delta_projects_to_stable_typed_json() {
        let event = GatewayEvent::from_language(LanguageStreamEvent::TextDelta {
            id: "text-1".to_string(),
            delta: "hello".to_string(),
        });

        assert_eq!(event.kind(), "model.text.delta");
        assert_eq!(event.data()["id"], "text-1");
        assert_eq!(event.data()["delta"], "hello");
    }
}
