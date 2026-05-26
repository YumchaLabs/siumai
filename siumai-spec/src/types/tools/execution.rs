//! Tool execution ownership contract.
//!
//! AI SDK exposes execution ownership as optional boolean wire fields:
//! `providerExecuted` on generated tool calls and `isProviderExecuted` on provider tools.
//! Siumai keeps those fields for compatibility, but internal routing should use this enum so
//! prompt validation, UI projection, stream assembly, and provider adapters all share one rule.

/// Semantic owner of a tool invocation's execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolExecutionOwner {
    /// The caller/Siumai runtime owns execution and must provide a tool result when required.
    Caller,
    /// The provider/model service owns execution and may return provider-executed results.
    Provider,
}

impl ToolExecutionOwner {
    /// AI SDK default: absent or `false` `providerExecuted` means caller-owned execution.
    pub const DEFAULT: Self = Self::Caller;

    /// Build an execution owner from an AI SDK `providerExecuted` flag.
    pub const fn from_provider_executed(provider_executed: Option<bool>) -> Self {
        match provider_executed {
            Some(true) => Self::Provider,
            Some(false) | None => Self::Caller,
        }
    }

    /// Build an execution owner from an AI SDK `isProviderExecuted` flag.
    pub const fn from_is_provider_executed(is_provider_executed: bool) -> Self {
        if is_provider_executed {
            Self::Provider
        } else {
            Self::Caller
        }
    }

    /// Return whether the provider/model service owns execution.
    pub const fn is_provider(self) -> bool {
        matches!(self, Self::Provider)
    }

    /// Return whether the caller/Siumai runtime owns execution.
    pub const fn is_caller(self) -> bool {
        matches!(self, Self::Caller)
    }

    /// Convert to the compact AI SDK `providerExecuted` wire flag.
    ///
    /// Caller-owned execution is represented by omission, matching AI SDK's default.
    pub const fn to_provider_executed_flag(self) -> Option<bool> {
        match self {
            Self::Provider => Some(true),
            Self::Caller => None,
        }
    }

    /// Convert to the AI SDK provider-tool `isProviderExecuted` wire flag.
    pub const fn to_is_provider_executed(self) -> bool {
        matches!(self, Self::Provider)
    }

    /// Prefer an explicit part-level flag and otherwise inherit a fallback flag.
    ///
    /// `Some(false)` is intentionally explicit and must not fall through to a provider-owned
    /// fallback. This is the rule used when stream terminal replay and incremental stream state
    /// both carry ownership.
    pub const fn merge_provider_executed_flags(
        explicit: Option<bool>,
        fallback: Option<bool>,
    ) -> Option<bool> {
        match explicit {
            Some(value) => Some(value),
            None => fallback,
        }
    }
}
