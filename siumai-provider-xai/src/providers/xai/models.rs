//! Dated xAI model-ID completion hints.
//!
//! These constants are intentionally non-exhaustive. The provider accepts every syntactically
//! valid future model ID and applies protocol-baseline behavior when no dated advisory exists.

/// Official model catalog used for the current completion hints.
pub const OFFICIAL_SOURCE: &str = "https://docs.x.ai/developers/models";
/// Date on which the named hints below were checked against the official catalog.
pub const VERIFIED_ON: &str = "2026-08-06";

pub mod language {
    /// Current flagship Grok language model.
    pub const GROK_4_5: &str = "grok-4.5";
    /// Current lower-cost general model.
    pub const GROK_4_3: &str = "grok-4.3";
    /// Provider-maintained alias for the current general Grok model.
    pub const GROK_LATEST: &str = "grok-latest";
    /// Current Grok 4.20 reasoning route.
    pub const GROK_4_20_REASONING: &str = "grok-4.20-reasoning";
    /// Current Grok 4.20 non-reasoning route.
    pub const GROK_4_20_NON_REASONING: &str = "grok-4.20-non-reasoning";
    /// Multi-agent research model with provider-controlled agent orchestration.
    pub const GROK_4_20_MULTI_AGENT: &str = "grok-4.20-multi-agent";

    pub const HINTS: &[&str] = &[
        GROK_4_5,
        GROK_4_3,
        GROK_LATEST,
        GROK_4_20_REASONING,
        GROK_4_20_NON_REASONING,
        GROK_4_20_MULTI_AGENT,
    ];
}

pub mod code {
    pub const GROK_BUILD_0_1: &str = "grok-build-0.1";
    pub const HINTS: &[&str] = &[GROK_BUILD_0_1];
}

pub(crate) mod catalog {
    use super::{code, language};

    pub const CHAT_EXACT_IDS: &[&str] =
        &[language::GROK_4_5, language::GROK_4_3, code::GROK_BUILD_0_1];
    pub const CHAT_ROLLING_ALIASES: &[&str] = &[
        language::GROK_LATEST,
        language::GROK_4_20_REASONING,
        language::GROK_4_20_NON_REASONING,
    ];
    pub const RESPONSES_EXACT_IDS: &[&str] = CHAT_EXACT_IDS;
    pub const RESPONSES_ROLLING_ALIASES: &[&str] = &[
        language::GROK_LATEST,
        language::GROK_4_20_REASONING,
        language::GROK_4_20_NON_REASONING,
        language::GROK_4_20_MULTI_AGENT,
    ];
}

pub mod recommended {
    pub const LANGUAGE: &str = super::language::GROK_4_5;
    pub const REASONING: &str = super::language::GROK_4_5;
    pub const CODE: &str = super::code::GROK_BUILD_0_1;
}

/// Return all dated completion hints without implying a closed allowlist.
pub fn hints() -> impl Iterator<Item = &'static str> {
    language::HINTS.iter().chain(code::HINTS).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_are_non_empty_and_unique() {
        let unique = hints().collect::<std::collections::BTreeSet<_>>();
        assert!(unique.contains(language::GROK_4_5));
        assert!(unique.contains(code::GROK_BUILD_0_1));
        assert_eq!(unique.len(), hints().count());
    }
}
