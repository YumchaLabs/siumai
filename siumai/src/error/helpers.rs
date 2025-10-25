//! High-level error helper utilities for user-facing error handling.
//!
//! This module provides structured error summaries, friendly suggestions,
//! and provider/model hints suitable for CLI/UI rendering, inspired by
//! Cherry Studio's UX while keeping the logic library-first.

use super::types::{ErrorCategory, LlmError};
// Note: Do not import ProviderType here; helpers are provider-agnostic

/// Error kind for presentation (coarse-grained)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Auth,
    RateLimit,
    Quota,
    Client,
    Server,
    Network,
    Parsing,
    Validation,
    Unsupported,
    Unknown,
}

/// Additional details for verbose rendering
#[derive(Debug, Clone, Default)]
pub struct ErrorDetails {
    /// Optional opaque JSON details when available (e.g., provider error body)
    pub details: Option<serde_json::Value>,
}

/// Provider hint based on model prefixes and registry aliases
#[derive(Debug, Clone, Default)]
pub struct ProviderHint {
    /// Suggested provider id (best-effort)
    pub suggested_provider_id: Option<String>,
    /// Known aliases for this provider (if any)
    pub aliases: Vec<String>,
}

/// Optional raw info extracted from provider/API error
#[derive(Debug, Clone, Default)]
pub struct RawInfo {
    /// Original provider error message (unmodified when available)
    pub message: Option<String>,
    /// Optional raw response body
    pub body: Option<serde_json::Value>,
    /// Optional raw response headers (not always available)
    pub headers: Option<serde_json::Value>,
}

/// Optional diagnosis information (kept separate from raw information)
#[derive(Debug, Clone, Default)]
pub struct Diagnosis {
    /// Short note that explains our classification or likely cause
    pub note: Option<String>,
}

/// Structured error summary for CLI/UI consumption (raw-first design)
#[derive(Debug, Clone)]
pub struct ErrorSummary {
    pub kind: ErrorKind,
    pub status: Option<u16>,
    /// Original provider message (raw) when available; otherwise best-effort
    pub message: String,
    /// Raw provider info for verbose display
    pub raw: RawInfo,
    /// Optional diagnosis (our interpretation), separate from raw info
    pub diagnosis: Diagnosis,
    pub suggestions: Vec<String>,
    pub details: ErrorDetails,
    pub provider_hint: Option<ProviderHint>,
}

impl Default for ErrorSummary {
    fn default() -> Self {
        Self {
            kind: ErrorKind::Unknown,
            status: None,
            message: String::new(),
            raw: RawInfo::default(),
            diagnosis: Diagnosis::default(),
            suggestions: Vec::new(),
            details: ErrorDetails::default(),
            provider_hint: None,
        }
    }
}

/// Summarize an LlmError with friendly suggestions and optional provider hint.
///
/// - `model` and `provider_name` are optional and used for better suggestions.
pub fn summarize_error(
    err: &LlmError,
    model: Option<&str>,
    provider_name: Option<&str>,
) -> ErrorSummary {
    let status = err.status_code();
    let kind = map_error_kind(err);
    let message = extract_raw_message(err);
    let mut suggestions = suggest_fixes(err, provider_name);

    // Add Vertex-specific header hint when auth/category implies Vertex usage
    if matches!(
        kind,
        ErrorKind::Auth | ErrorKind::RateLimit | ErrorKind::Quota
    ) && provider_name.map(|p| p.contains("vertex")).unwrap_or(false)
    {
        suggestions.push(
            "If using Google Vertex AI, consider setting x-goog-user-project for billing/quota"
                .to_string(),
        );
    }

    let raw = RawInfo {
        message: Some(message.clone()),
        body: extract_details(err),
        headers: None,
    };
    let diagnosis = Diagnosis {
        note: diagnosis_note(err),
    };
    let details = ErrorDetails {
        details: extract_details(err),
    };
    let provider_hint = provider_hint_for_model(model);

    ErrorSummary {
        kind,
        status,
        message,
        raw,
        diagnosis,
        suggestions,
        details,
        provider_hint,
    }
}

/// Map LlmError to presentation ErrorKind.
pub fn map_error_kind(err: &LlmError) -> ErrorKind {
    match err.category() {
        ErrorCategory::Authentication => ErrorKind::Auth,
        ErrorCategory::RateLimit => ErrorKind::RateLimit,
        ErrorCategory::Client => {
            if err.status_code() == Some(403) {
                ErrorKind::Quota
            } else {
                ErrorKind::Client
            }
        }
        ErrorCategory::Server => ErrorKind::Server,
        ErrorCategory::Network => ErrorKind::Network,
        ErrorCategory::Parsing => ErrorKind::Parsing,
        ErrorCategory::Validation => ErrorKind::Validation,
        ErrorCategory::Unsupported => ErrorKind::Unsupported,
        _ => ErrorKind::Unknown,
    }
}

/// Format a concise user-facing message from LlmError.
fn extract_raw_message(err: &LlmError) -> String {
    match err {
        LlmError::ApiError { message, .. } => message.clone(),
        LlmError::AuthenticationError(msg)
        | LlmError::RateLimitError(msg)
        | LlmError::QuotaExceededError(msg)
        | LlmError::TimeoutError(msg)
        | LlmError::ConnectionError(msg)
        | LlmError::ParseError(msg)
        | LlmError::InvalidParameter(msg)
        | LlmError::InvalidInput(msg) => msg.clone(),
        _ => err.to_string(),
    }
}

/// Suggest fixes based on error type and optional provider name.
pub fn suggest_fixes(err: &LlmError, provider_name: Option<&str>) -> Vec<String> {
    let mut tips = Vec::new();
    match err.category() {
        ErrorCategory::Authentication => {
            tips.push("Verify API key or Bearer token".to_string());
            tips.push("If token is short-lived, retry after refreshing".to_string());
        }
        ErrorCategory::RateLimit => {
            tips.push("Reduce request rate or wait and retry (Retry-After if present)".to_string());
            tips.push("Consider batching requests or optimizing payload size".to_string());
        }
        ErrorCategory::Client => {
            tips.push("Check request parameters and required fields".to_string());
        }
        ErrorCategory::Server => {
            tips.push("Retry with backoff; check provider status page".to_string());
        }
        ErrorCategory::Parsing => {
            tips.push("Verify response format or disable strict parsing".to_string());
        }
        ErrorCategory::Validation => {
            tips.push("Validate input ranges and formats".to_string());
        }
        ErrorCategory::Configuration => {
            tips.push("Check environment variables and client configuration".to_string());
        }
        _ => {}
    }
    if let Some(name) = provider_name {
        tips.push(format!(
            "Open provider settings to review {} configuration",
            name
        ));
    }
    tips
}

fn extract_details(err: &LlmError) -> Option<serde_json::Value> {
    match err {
        LlmError::ApiError { details, .. } => details.clone(),
        _ => None,
    }
}

/// Best-effort provider hint by model id (using registry prefixes/aliases).
pub fn provider_hint_for_model(model: Option<&str>) -> Option<ProviderHint> {
    let model = model?;
    if model.is_empty() {
        return None;
    }
    let registry = crate::registry::global_registry();
    let guard = registry.read().ok()?;
    let rec = guard.resolve_for_model(model)?;
    let mut aliases = rec.aliases.clone();
    aliases.sort();
    aliases.dedup();
    Some(ProviderHint {
        suggested_provider_id: Some(rec.id.clone()),
        aliases,
    })
}

/// Render a CLI-friendly string for ErrorSummary.
pub fn format_summary(summary: &ErrorSummary, verbose: bool) -> String {
    let mut out = String::new();
    if let Some(code) = summary.status {
        let _ = std::fmt::Write::write_str(&mut out, &format!("Status: {}\n", code));
    }
    let _ = std::fmt::Write::write_str(&mut out, &format!("Message: {}\n", summary.message));
    if let Some(note) = &summary.diagnosis.note {
        let _ = std::fmt::Write::write_str(&mut out, &format!("Diagnosis: {}\n", note));
    }
    if let Some(code) = summary.status {
        let _ = code; // keep for potential future extensions
    }
    if let Some(hint) = &summary.provider_hint {
        if let Some(id) = &hint.suggested_provider_id {
            let _ = std::fmt::Write::write_str(&mut out, &format!("Suggested provider: {}\n", id));
        }
        if !hint.aliases.is_empty() {
            let _ = std::fmt::Write::write_str(
                &mut out,
                &format!("Aliases: {}\n", hint.aliases.join(", ")),
            );
        }
    }
    if !summary.suggestions.is_empty() {
        let _ = std::fmt::Write::write_str(&mut out, "Suggestions:\n");
        for s in &summary.suggestions {
            let _ = std::fmt::Write::write_str(&mut out, &format!("  - {}\n", s));
        }
    }
    if verbose {
        if let Some(body) = &summary.raw.body {
            let _ = std::fmt::Write::write_str(&mut out, &format!("Raw body: {body}\n"));
        }
        if let Some(d) = &summary.details.details {
            let _ = std::fmt::Write::write_str(&mut out, &format!("Details: {d}\n"));
        }
    }
    out
}

fn diagnosis_note(err: &LlmError) -> Option<String> {
    match map_error_kind(err) {
        ErrorKind::Auth => Some("Authentication failed; check key/token".to_string()),
        ErrorKind::RateLimit => Some("Rate limited; respect Retry-After or backoff".to_string()),
        ErrorKind::Quota => Some("Quota/permission issue (403); check project/billing".to_string()),
        ErrorKind::Server => Some("Provider server error; retry with backoff".to_string()),
        ErrorKind::Client => Some("Client-side request error; verify parameters".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_mapping_basic() {
        let e = LlmError::api_error(429, "too many requests");
        assert_eq!(map_error_kind(&e), ErrorKind::RateLimit);
        let e = LlmError::AuthenticationError("missing".into());
        assert_eq!(map_error_kind(&e), ErrorKind::Auth);
    }

    #[test]
    fn summary_includes_suggestions() {
        let e = LlmError::api_error(401, "unauthorized");
        let s = summarize_error(&e, Some("claude-3-7-sonnet-latest"), Some("anthropic"));
        assert!(!s.suggestions.is_empty());
        // Provider hint is best-effort; it may be None depending on registry state in unit tests.
    }
}
