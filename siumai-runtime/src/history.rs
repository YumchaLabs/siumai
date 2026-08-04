//! Cross-provider projection for provider-neutral conversation history.
//!
//! A [`LanguageRequest`](siumai_core::LanguageRequest) is intentionally more
//! expressive than the wire format accepted by any one provider.  In
//! particular, it may contain provider-native opaque items and provider-owned
//! tool executions.  This module makes the boundary explicit: a continuation
//! in the same target/protocol is lossless, while a transition to another
//! protocol either succeeds with an audit trail or is rejected before any
//! provider call is made.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use siumai_core::{ContentPart, ExecutionOwner, LanguageRequest, Message, OpaqueProviderItem};
use thiserror::Error;

use crate::options::ModelTarget;

/// How a history projection handles representational loss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProjectionPolicy {
    /// Reject required loss.  Blocking state is rejected under either policy.
    #[default]
    Strict,
    /// Return the portable request together with diagnostics for dropped data.
    BestEffort,
}

/// How serious one projection diagnostic is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProjectionSeverity {
    /// Provider metadata was removed, but the provider-neutral meaning remains.
    Advisory,
    /// Source data has no provider-neutral representation and was removed.
    Required,
    /// Continuing would leave an unresolved execution or approval state.
    Blocking,
}

/// Why a history item could not be carried to the target protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProjectionLossReason {
    /// An opaque item belongs to another provider, platform, or protocol.
    ForeignProviderOpaque,
    /// Response-side provider metadata on a citation was intentionally removed.
    ProviderMetadataRemoved,
    /// A provider-owned tool call cannot be replayed by the target protocol.
    ProviderOwnedToolState,
    /// A provider-owned tool call has no terminal result yet.
    UnresolvedProviderToolState,
    /// An approval request is still waiting for an explicit decision.
    PendingApproval,
    /// The provider has deferred work that has not reached a terminal state.
    ProviderDeferred,
    /// A local tool call has not received a terminal result.
    UnresolvedLocalToolState,
    /// The runtime does not understand the tool execution owner yet.
    UnknownToolExecutionOwner,
    /// More than one tool call uses the same call identity.
    AmbiguousToolCallIdentity,
    /// A tool result has no call whose execution ownership can be verified.
    OrphanedToolResult,
    /// The runtime does not understand a newly added content part yet.
    UnsupportedContentPart,
}

/// Location of a projection diagnostic in the input request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProjectionLocation {
    /// A content part inside a request message.
    MessageContent {
        /// Zero-based message index.
        message_index: usize,
        /// Zero-based content-part index within the message.
        content_index: usize,
    },
}

impl ProjectionLocation {
    /// Construct a message-content location.
    pub const fn message_content(message_index: usize, content_index: usize) -> Self {
        Self::MessageContent {
            message_index,
            content_index,
        }
    }

    /// Return the zero-based message index.
    pub const fn message_index(self) -> usize {
        match self {
            Self::MessageContent { message_index, .. } => message_index,
        }
    }

    /// Return the zero-based content-part index.
    pub const fn content_index(self) -> usize {
        match self {
            Self::MessageContent { content_index, .. } => content_index,
        }
    }
}

/// One structured explanation of data changed or removed during projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionLoss {
    /// The category of state that was changed or removed.
    pub reason: ProjectionLossReason,
    /// Whether the diagnostic is advisory, required, or blocking.
    pub severity: ProjectionSeverity,
    /// Where the affected item appeared in the source request.
    pub location: ProjectionLocation,
}

impl ProjectionLoss {
    fn new(
        reason: ProjectionLossReason,
        severity: ProjectionSeverity,
        location: ProjectionLocation,
    ) -> Self {
        Self {
            reason,
            severity,
            location,
        }
    }
}

/// A projected request and the diagnostics produced while creating it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedHistory {
    request: LanguageRequest,
    losses: Vec<ProjectionLoss>,
}

impl ProjectedHistory {
    /// Return the projected request by reference.
    pub fn request(&self) -> &LanguageRequest {
        &self.request
    }

    /// Return all diagnostics in source order.
    pub fn losses(&self) -> &[ProjectionLoss] {
        &self.losses
    }

    /// Alias for callers that use diagnostics terminology.
    pub fn diagnostics(&self) -> &[ProjectionLoss] {
        self.losses()
    }

    /// Consume the projection and return the request and diagnostics.
    pub fn into_parts(self) -> (LanguageRequest, Vec<ProjectionLoss>) {
        (self.request, self.losses)
    }

    /// Consume the projection and return only the request.
    pub fn into_request(self) -> LanguageRequest {
        self.request
    }
}

/// Error returned when the selected policy cannot safely project history.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("history projection under {policy:?} policy contains incompatible provider state")]
pub struct HistoryProjectionError {
    policy: ProjectionPolicy,
    losses: Vec<ProjectionLoss>,
}

impl HistoryProjectionError {
    fn new(policy: ProjectionPolicy, losses: Vec<ProjectionLoss>) -> Self {
        Self { policy, losses }
    }

    /// Return the policy that rejected the projection.
    pub const fn policy(&self) -> ProjectionPolicy {
        self.policy
    }

    /// Return all diagnostics that caused rejection.
    pub fn losses(&self) -> &[ProjectionLoss] {
        &self.losses
    }

    /// Consume the error and return its diagnostics.
    pub fn into_losses(self) -> Vec<ProjectionLoss> {
        self.losses
    }
}

/// Project a request from one model target to another.
///
/// Exact target equality and a matching provider/platform/protocol (including
/// API mode) are treated as one replay domain and are returned byte-for-byte
/// unchanged.  Other transitions keep portable content, preserve opaque items
/// already native to the target protocol, and remove source-native state with
/// explicit diagnostics.  No response metadata is converted into request
/// options by this function.
pub fn project_history(
    request: LanguageRequest,
    source: &ModelTarget,
    target: &ModelTarget,
    policy: ProjectionPolicy,
) -> Result<ProjectedHistory, HistoryProjectionError> {
    let same_domain = same_replay_domain(source, target);

    let LanguageRequest {
        messages,
        generation,
        tools,
        tool_choice,
        structured_output,
    } = request;

    let result_ids = collect_result_ids(&messages);
    let call_dispositions = collect_call_dispositions(&messages, &result_ids, same_domain, target);
    let mut losses = Vec::new();
    let mut projected_messages = Vec::with_capacity(messages.len());

    for (message_index, message) in messages.into_iter().enumerate() {
        let had_content = !message.content.is_empty();
        let mut projected_content = Vec::with_capacity(message.content.len());

        for (content_index, part) in message.content.into_iter().enumerate() {
            let location = ProjectionLocation::message_content(message_index, content_index);
            match part {
                ContentPart::Text { .. }
                | ContentPart::Reasoning { .. }
                | ContentPart::Media(_)
                | ContentPart::Refusal { .. } => projected_content.push(part),
                ContentPart::Citation(mut citation) => {
                    if !same_domain && !citation.provider.is_empty() {
                        citation.provider.clear();
                        losses.push(ProjectionLoss::new(
                            ProjectionLossReason::ProviderMetadataRemoved,
                            ProjectionSeverity::Advisory,
                            location,
                        ));
                    }
                    projected_content.push(ContentPart::Citation(citation));
                }
                ContentPart::ProviderOpaque(item) => {
                    if opaque_matches_target(&item, source, target) {
                        projected_content.push(ContentPart::ProviderOpaque(item));
                    } else {
                        let (reason, severity) = classify_opaque_loss(&item);
                        losses.push(ProjectionLoss::new(reason, severity, location));
                    }
                }
                ContentPart::ToolCall(call) => match call_dispositions.get(&call.id) {
                    Some(ToolCallDisposition::Portable) => {
                        projected_content.push(ContentPart::ToolCall(call));
                    }
                    Some(ToolCallDisposition::Loss { reason, severity }) => {
                        losses.push(ProjectionLoss::new(reason.clone(), *severity, location));
                    }
                    None => losses.push(ProjectionLoss::new(
                        ProjectionLossReason::UnknownToolExecutionOwner,
                        ProjectionSeverity::Blocking,
                        location,
                    )),
                },
                ContentPart::ToolResult(result) => match call_dispositions.get(&result.call_id) {
                    Some(ToolCallDisposition::Loss { reason, severity }) => {
                        losses.push(ProjectionLoss::new(reason.clone(), *severity, location));
                    }
                    Some(ToolCallDisposition::Portable) => {
                        projected_content.push(ContentPart::ToolResult(result));
                    }
                    None if same_domain => projected_content.push(ContentPart::ToolResult(result)),
                    None => losses.push(ProjectionLoss::new(
                        ProjectionLossReason::OrphanedToolResult,
                        ProjectionSeverity::Blocking,
                        location,
                    )),
                },
                other if same_domain => projected_content.push(other),
                _ => losses.push(ProjectionLoss::new(
                    ProjectionLossReason::UnsupportedContentPart,
                    ProjectionSeverity::Required,
                    location,
                )),
            }
        }

        if !projected_content.is_empty() || !had_content {
            projected_messages.push(Message {
                role: message.role,
                content: projected_content,
            });
        }
    }

    let projected = ProjectedHistory {
        request: LanguageRequest {
            messages: projected_messages,
            generation,
            tools,
            tool_choice,
            structured_output,
        },
        losses,
    };

    if should_reject(policy, projected.losses()) {
        return Err(HistoryProjectionError::new(
            policy,
            projected.losses.clone(),
        ));
    }

    Ok(projected)
}

fn same_replay_domain(source: &ModelTarget, target: &ModelTarget) -> bool {
    if source == target {
        return true;
    }

    source.provider() == target.provider()
        && source.platform() == target.platform()
        && source.protocol().is_some()
        && source.protocol() == target.protocol()
        && source.api_mode() == target.api_mode()
}

fn opaque_matches_target(
    item: &OpaqueProviderItem,
    source: &ModelTarget,
    target: &ModelTarget,
) -> bool {
    let provenance = item.provenance();
    let target_protocol_matches = provenance.provider == *target.provider()
        && provenance.platform.as_deref() == target.platform().map(|platform| platform.as_str())
        && target
            .protocol()
            .is_some_and(|protocol| provenance.protocol == protocol.as_str());
    let exact_unspecified_target_matches = source == target
        && target.protocol().is_none()
        && provenance.provider == *target.provider()
        && provenance.model == *target.model()
        && target
            .platform()
            .is_none_or(|platform| provenance.platform.as_deref() == Some(platform.as_str()));

    target_protocol_matches || exact_unspecified_target_matches
}

fn collect_result_ids(messages: &[Message]) -> BTreeSet<String> {
    messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(result.call_id.clone()),
            _ => None,
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolCallDisposition {
    Portable,
    Loss {
        reason: ProjectionLossReason,
        severity: ProjectionSeverity,
    },
}

fn collect_call_dispositions(
    messages: &[Message],
    result_ids: &BTreeSet<String>,
    same_domain: bool,
    target: &ModelTarget,
) -> BTreeMap<String, ToolCallDisposition> {
    let mut dispositions = BTreeMap::new();
    for part in messages.iter().flat_map(|message| message.content.iter()) {
        let ContentPart::ToolCall(call) = part else {
            continue;
        };
        if dispositions.contains_key(&call.id) {
            dispositions.insert(
                call.id.clone(),
                ToolCallDisposition::Loss {
                    reason: ProjectionLossReason::AmbiguousToolCallIdentity,
                    severity: ProjectionSeverity::Blocking,
                },
            );
            continue;
        }

        let disposition = match &call.owner {
            ExecutionOwner::Local if same_domain => ToolCallDisposition::Portable,
            ExecutionOwner::Local if result_ids.contains(&call.id) => ToolCallDisposition::Portable,
            ExecutionOwner::Local => ToolCallDisposition::Loss {
                reason: ProjectionLossReason::UnresolvedLocalToolState,
                severity: ProjectionSeverity::Blocking,
            },
            ExecutionOwner::Provider { provider }
                if same_domain && provider == target.provider() =>
            {
                ToolCallDisposition::Portable
            }
            ExecutionOwner::Provider { .. } if result_ids.contains(&call.id) => {
                ToolCallDisposition::Loss {
                    reason: ProjectionLossReason::ProviderOwnedToolState,
                    severity: ProjectionSeverity::Required,
                }
            }
            ExecutionOwner::Provider { .. } => ToolCallDisposition::Loss {
                reason: ProjectionLossReason::UnresolvedProviderToolState,
                severity: ProjectionSeverity::Blocking,
            },
            _ => ToolCallDisposition::Loss {
                reason: ProjectionLossReason::UnknownToolExecutionOwner,
                severity: ProjectionSeverity::Blocking,
            },
        };
        dispositions.entry(call.id.clone()).or_insert(disposition);
    }
    dispositions
}

fn should_reject(policy: ProjectionPolicy, losses: &[ProjectionLoss]) -> bool {
    losses.iter().any(|loss| {
        loss.severity == ProjectionSeverity::Blocking
            || (policy == ProjectionPolicy::Strict && loss.severity == ProjectionSeverity::Required)
    })
}

fn classify_opaque_loss(item: &OpaqueProviderItem) -> (ProjectionLossReason, ProjectionSeverity) {
    let kind = normalize_kind(item.kind());
    let object = item.data().as_object();
    let top_level =
        |field: &str| object.and_then(|data| data.get(field).and_then(|value| value.as_str()));
    let nested = |parent: &str, field: &str| {
        object.and_then(|data| {
            data.get(parent)
                .and_then(|value| value.as_object())
                .and_then(|nested| nested.get(field))
                .and_then(|value| value.as_str())
        })
    };
    let has_lifecycle_value = |states: &[&str]| {
        if object
            .and_then(|data| data.get("pending"))
            .and_then(|value| value.as_bool())
            .is_some_and(|pending| pending)
            && states.contains(&"pending")
        {
            return true;
        }
        if object
            .and_then(|data| data.get("requires_action"))
            .and_then(|value| value.as_bool())
            .is_some_and(|required| required)
            && states.contains(&"requires_action")
        {
            return true;
        }

        [
            top_level("state"),
            top_level("status"),
            top_level("phase"),
            nested("response", "state"),
            nested("response", "status"),
            nested("item", "state"),
            nested("item", "status"),
        ]
        .into_iter()
        .flatten()
        .map(normalize_kind)
        .any(|state| states.iter().any(|candidate| *candidate == state))
    };
    let discriminators = [
        top_level("type"),
        top_level("event_type"),
        top_level("kind"),
        nested("event", "type"),
        nested("item", "type"),
        nested("response", "type"),
    ];

    if is_pending_approval_kind(&kind)
        || discriminators
            .iter()
            .flatten()
            .map(|value| normalize_kind(value))
            .any(|value| is_pending_approval_kind(&value))
        || has_lifecycle_value(APPROVAL_STATES)
    {
        return (
            ProjectionLossReason::PendingApproval,
            ProjectionSeverity::Blocking,
        );
    }
    if is_provider_deferred_kind(&kind)
        || discriminators
            .iter()
            .flatten()
            .map(|value| normalize_kind(value))
            .any(|value| is_provider_deferred_kind(&value))
        || has_lifecycle_value(DEFERRED_STATES)
    {
        return (
            ProjectionLossReason::ProviderDeferred,
            ProjectionSeverity::Blocking,
        );
    }

    (
        ProjectionLossReason::ForeignProviderOpaque,
        ProjectionSeverity::Required,
    )
}

const APPROVAL_STATES: &[&str] = &[
    "approval_required",
    "awaiting_approval",
    "pending_approval",
    "requires_approval",
];

const DEFERRED_STATES: &[&str] = &[
    "deferred",
    "in_progress",
    "pending",
    "queued",
    "requires_action",
];

fn normalize_kind(kind: &str) -> String {
    kind.trim()
        .to_ascii_lowercase()
        .chars()
        .map(|character| match character {
            '-' | '.' | ':' | '/' | ' ' => '_',
            other => other,
        })
        .collect()
}

fn is_pending_approval_kind(kind: &str) -> bool {
    kind.contains("approval_request")
        || kind.contains("pending_approval")
        || kind.contains("approval_required")
        || kind.contains("requires_approval")
}

fn is_provider_deferred_kind(kind: &str) -> bool {
    kind.contains("deferred")
        || kind.contains("requires_action")
        || kind.contains("provider_pending")
        || kind.contains("pending_provider")
        || kind.ends_with("_in_progress")
        || kind.ends_with("_queued")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use siumai_core::{
        Citation, ContentPart, ExecutionOwner, LanguageRequest, Message, MessageRole, ModelId,
        OpaqueProviderItem, ProtocolId, ProviderId, ProviderProvenance, ToolCall, ToolOutcome,
        ToolResult,
    };

    use super::*;

    fn target(provider: &str, protocol: &str, model: &str) -> ModelTarget {
        ModelTarget::new(
            ProviderId::new(provider).expect("test provider id"),
            ModelId::new(model).expect("test model id"),
        )
        .with_protocol(ProtocolId::new(protocol).expect("test protocol id"))
    }

    fn opaque(
        provider: &str,
        protocol: &str,
        model: &str,
        kind: &str,
        data: serde_json::Value,
    ) -> OpaqueProviderItem {
        OpaqueProviderItem::new(
            ProviderProvenance {
                provider: ProviderId::new(provider).expect("test provider id"),
                platform: None,
                protocol: protocol.to_string(),
                model: ModelId::new(model).expect("test model id"),
            },
            kind,
            data,
        )
        .expect("test opaque item")
    }

    fn request(content: Vec<ContentPart>) -> LanguageRequest {
        LanguageRequest::new(vec![Message {
            role: MessageRole::User,
            content,
        }])
    }

    #[test]
    fn strict_is_the_default_policy() {
        assert_eq!(ProjectionPolicy::default(), ProjectionPolicy::Strict);
    }

    #[test]
    fn same_target_is_lossless() {
        let source = target("openai", "responses", "gpt-5.6");
        let item = opaque(
            "openai",
            "responses",
            "gpt-5.6",
            "response.output",
            json!({"id": "x"}),
        );
        let input = request(vec![ContentPart::ProviderOpaque(item)]);

        let projected = project_history(input.clone(), &source, &source, ProjectionPolicy::Strict)
            .expect("same target must be lossless");
        assert_eq!(projected.request(), &input);
        assert!(projected.losses().is_empty());
    }

    #[test]
    fn same_protocol_different_model_preserves_native_item() {
        let source = target("openai", "responses", "gpt-5.6");
        let target = target("openai", "responses", "gpt-5.7");
        let item = opaque(
            "openai",
            "responses",
            "gpt-5.6",
            "response.output",
            json!({"id": "x"}),
        );
        let input = request(vec![ContentPart::ProviderOpaque(item.clone())]);

        let projected = project_history(input, &source, &target, ProjectionPolicy::Strict)
            .expect("same protocol must preserve native item");
        assert_eq!(
            projected.request().messages[0].content,
            vec![ContentPart::ProviderOpaque(item)]
        );
        assert!(projected.losses().is_empty());
    }

    #[test]
    fn same_target_still_rejects_foreign_opaque_state() {
        let source = target("openai", "openai.responses", "gpt-5.6");
        let input = request(vec![ContentPart::ProviderOpaque(opaque(
            "anthropic",
            "anthropic.messages",
            "opus-5",
            "message.content_block",
            json!({"type": "thinking"}),
        ))]);

        let error = project_history(input, &source, &source, ProjectionPolicy::Strict)
            .expect_err("foreign opaque state must be checked inside same-target history");
        assert_eq!(
            error.losses()[0].reason,
            ProjectionLossReason::ForeignProviderOpaque
        );
    }

    #[test]
    fn strict_rejects_foreign_opaque_state() {
        let source = target("openai", "responses", "gpt-5.6");
        let target = target("anthropic", "messages", "opus-5");
        let input = request(vec![ContentPart::ProviderOpaque(opaque(
            "openai",
            "responses",
            "gpt-5.6",
            "response.output",
            json!({"id": "x"}),
        ))]);

        let error = project_history(input, &source, &target, ProjectionPolicy::Strict)
            .expect_err("strict must reject foreign opaque state");
        assert_eq!(
            error.losses()[0].reason,
            ProjectionLossReason::ForeignProviderOpaque
        );
        assert_eq!(error.losses()[0].severity, ProjectionSeverity::Required);
    }

    #[test]
    fn best_effort_keeps_portable_content_and_strips_provider_metadata() {
        let source = target("openai", "responses", "gpt-5.6");
        let target = target("anthropic", "messages", "opus-5");
        let mut citation = Citation {
            source_id: "source-1".to_string(),
            title: None,
            url: None,
            start: None,
            end: None,
            provider: BTreeMap::new(),
        };
        citation
            .provider
            .insert("cache_hit".to_string(), json!(true));
        let input = request(vec![
            ContentPart::Text {
                text: "hello".to_string(),
            },
            ContentPart::Citation(citation),
            ContentPart::ProviderOpaque(opaque(
                "openai",
                "responses",
                "gpt-5.6",
                "response.output",
                json!({"id": "x"}),
            )),
        ]);

        let projected = project_history(input, &source, &target, ProjectionPolicy::BestEffort)
            .expect("best effort may remove representational loss");
        assert_eq!(projected.request().messages[0].content.len(), 2);
        assert!(matches!(
            &projected.request().messages[0].content[0],
            ContentPart::Text { .. }
        ));
        let ContentPart::Citation(citation) = &projected.request().messages[0].content[1] else {
            panic!("citation should remain portable");
        };
        assert!(citation.provider.is_empty());
        assert!(projected.losses().iter().any(|loss| {
            loss.reason == ProjectionLossReason::ProviderMetadataRemoved
                && loss.severity == ProjectionSeverity::Advisory
        }));
        assert!(
            projected
                .losses()
                .iter()
                .any(|loss| loss.reason == ProjectionLossReason::ForeignProviderOpaque)
        );
    }

    #[test]
    fn blocking_opaque_state_rejects_best_effort() {
        let source = target("openai", "responses", "gpt-5.6");
        let target = target("anthropic", "messages", "opus-5");
        let input = request(vec![ContentPart::ProviderOpaque(opaque(
            "openai",
            "openai.responses",
            "gpt-5.6",
            "response.output_item",
            json!({
                "id": "approval-1",
                "type": "mcp_approval_request"
            }),
        ))]);

        let error = project_history(input, &source, &target, ProjectionPolicy::BestEffort)
            .expect_err("pending approval is blocking even in best effort");
        assert_eq!(
            error.losses()[0].reason,
            ProjectionLossReason::PendingApproval
        );
        assert_eq!(error.losses()[0].severity, ProjectionSeverity::Blocking);
    }

    #[test]
    fn provider_deferred_state_rejects_best_effort() {
        let source = target("openai", "responses", "gpt-5.6");
        let target = target("anthropic", "messages", "opus-5");
        let input = request(vec![ContentPart::ProviderOpaque(opaque(
            "openai",
            "openai.responses",
            "gpt-5.6",
            "response.output_item.stream_event",
            json!({
                "type": "response.updated",
                "response": {"status": "in_progress"}
            }),
        ))]);

        let error = project_history(input, &source, &target, ProjectionPolicy::BestEffort)
            .expect_err("provider-deferred work is blocking even in best effort");
        assert_eq!(
            error.losses()[0].reason,
            ProjectionLossReason::ProviderDeferred
        );
        assert_eq!(error.losses()[0].severity, ProjectionSeverity::Blocking);
    }

    #[test]
    fn unresolved_local_tool_state_is_blocking() {
        let source = target("openai", "responses", "gpt-5.6");
        let target = target("anthropic", "messages", "opus-5");
        let call = ToolCall {
            id: "call-1".to_string(),
            name: "lookup".to_string(),
            arguments: json!({"q": "rust"}),
            owner: ExecutionOwner::Local,
        };

        let error = project_history(
            request(vec![ContentPart::ToolCall(call)]),
            &source,
            &target,
            ProjectionPolicy::BestEffort,
        )
        .expect_err("unresolved local tool state is blocking");
        assert_eq!(
            error.losses()[0].reason,
            ProjectionLossReason::UnresolvedLocalToolState
        );
        assert_eq!(error.losses()[0].severity, ProjectionSeverity::Blocking);
    }

    #[test]
    fn completed_local_tool_call_remains_portable() {
        let source = target("openai", "responses", "gpt-5.6");
        let target = target("anthropic", "messages", "opus-5");
        let call = ToolCall {
            id: "call-1".to_string(),
            name: "lookup".to_string(),
            arguments: json!({"q": "rust"}),
            owner: ExecutionOwner::Local,
        };
        let result = ToolResult {
            call_id: "call-1".to_string(),
            name: "lookup".to_string(),
            outcome: ToolOutcome::Success { value: json!("ok") },
        };

        let projected = project_history(
            request(vec![
                ContentPart::ToolCall(call),
                ContentPart::ToolResult(result),
            ]),
            &source,
            &target,
            ProjectionPolicy::Strict,
        )
        .expect("completed local tool state is portable");
        assert_eq!(projected.request().messages[0].content.len(), 2);
        assert!(projected.losses().is_empty());
    }

    #[test]
    fn provider_owned_tool_state_is_removed_with_diagnostics() {
        let source = target("openai", "responses", "gpt-5.6");
        let target = target("anthropic", "messages", "opus-5");
        let call = ToolCall {
            id: "call-1".to_string(),
            name: "web_search".to_string(),
            arguments: json!({"q": "rust"}),
            owner: ExecutionOwner::Provider {
                provider: ProviderId::new("openai").expect("test provider id"),
            },
        };
        let result = ToolResult {
            call_id: "call-1".to_string(),
            name: "web_search".to_string(),
            outcome: ToolOutcome::Success { value: json!("ok") },
        };

        let projected = project_history(
            request(vec![
                ContentPart::ToolCall(call),
                ContentPart::ToolResult(result),
            ]),
            &source,
            &target,
            ProjectionPolicy::BestEffort,
        )
        .expect("best effort removes completed provider-owned state");
        assert!(projected.request().messages.is_empty());
        assert_eq!(projected.losses().len(), 2);
        assert!(
            projected
                .losses()
                .iter()
                .all(|loss| loss.reason == ProjectionLossReason::ProviderOwnedToolState)
        );
    }
}
