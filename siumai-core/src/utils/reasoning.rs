//! AI SDK-style reasoning mapping helpers.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::types::{LanguageModelReasoning, Warning};

/// AI SDK reasoning levels that can be lowered to provider effort/budget values.
///
/// This intentionally excludes `provider-default` and `none`, matching the AI SDK
/// `ReasoningLevel` helper type used by provider-utils mapping functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReasoningLevel {
    /// Minimal reasoning.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort.
    #[serde(rename = "xhigh")]
    XHigh,
}

impl ReasoningLevel {
    /// All supported provider-utils reasoning levels.
    pub const ALL: [Self; 5] = [
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::XHigh,
    ];

    /// Return the AI SDK wire string for this reasoning level.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }
}

impl fmt::Display for ReasoningLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when a top-level reasoning value is not a provider-utils level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningLevelConversionError {
    /// `provider-default` should be omitted rather than mapped.
    ProviderDefault,
    /// `none` disables reasoning and is not an effort/budget level.
    Disabled,
}

impl fmt::Display for ReasoningLevelConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderDefault => f.write_str("provider-default is not a reasoning level"),
            Self::Disabled => f.write_str("none disables reasoning and cannot be mapped"),
        }
    }
}

impl std::error::Error for ReasoningLevelConversionError {}

impl TryFrom<LanguageModelReasoning> for ReasoningLevel {
    type Error = ReasoningLevelConversionError;

    fn try_from(value: LanguageModelReasoning) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&LanguageModelReasoning> for ReasoningLevel {
    type Error = ReasoningLevelConversionError;

    fn try_from(value: &LanguageModelReasoning) -> Result<Self, Self::Error> {
        match value {
            LanguageModelReasoning::ProviderDefault => {
                Err(ReasoningLevelConversionError::ProviderDefault)
            }
            LanguageModelReasoning::None => Err(ReasoningLevelConversionError::Disabled),
            LanguageModelReasoning::Minimal => Ok(Self::Minimal),
            LanguageModelReasoning::Low => Ok(Self::Low),
            LanguageModelReasoning::Medium => Ok(Self::Medium),
            LanguageModelReasoning::High => Ok(Self::High),
            LanguageModelReasoning::XHigh => Ok(Self::XHigh),
        }
    }
}

/// Return whether a top-level reasoning value should be treated as custom.
///
/// This mirrors AI SDK `isCustomReasoning`: only missing and `provider-default`
/// are non-custom; `none` is custom because it explicitly disables reasoning.
pub fn is_custom_reasoning(reasoning: Option<&LanguageModelReasoning>) -> bool {
    !matches!(
        reasoning,
        None | Some(LanguageModelReasoning::ProviderDefault)
    )
}

/// Map a cross-provider reasoning level to a provider-specific effort value.
///
/// A missing level records an unsupported warning. A mapped effort that differs
/// from the source level records a compatibility warning.
pub fn map_reasoning_to_provider_effort<T>(
    reasoning: ReasoningLevel,
    effort_map: &[(ReasoningLevel, T)],
    warnings: &mut Vec<Warning>,
) -> Option<T>
where
    T: Clone + PartialEq + fmt::Display,
{
    let mapped = effort_map
        .iter()
        .find_map(|(level, effort)| (*level == reasoning).then(|| effort.clone()));

    let Some(mapped) = mapped else {
        warnings.push(unsupported_reasoning_warning(reasoning));
        return None;
    };

    if mapped.to_string() != reasoning.as_str() {
        warnings.push(Warning::compatibility(
            "reasoning",
            Some(format!(
                "reasoning \"{reasoning}\" is not directly supported by this model. mapped to effort \"{mapped}\"."
            )),
        ));
    }

    Some(mapped)
}

/// AI SDK provider-utils default reasoning-budget percentages.
pub const DEFAULT_REASONING_BUDGET_PERCENTAGES: [(ReasoningLevel, f64); 5] = [
    (ReasoningLevel::Minimal, 0.02),
    (ReasoningLevel::Low, 0.1),
    (ReasoningLevel::Medium, 0.3),
    (ReasoningLevel::High, 0.6),
    (ReasoningLevel::XHigh, 0.9),
];

/// Options for mapping a reasoning level to an absolute token budget.
#[derive(Debug, Clone, Copy)]
pub struct ReasoningBudgetOptions<'a> {
    /// Cross-provider reasoning level.
    pub reasoning: ReasoningLevel,
    /// Maximum output tokens for the model call.
    pub max_output_tokens: u32,
    /// Provider/model maximum reasoning budget.
    pub max_reasoning_budget: u32,
    /// Provider/model minimum reasoning budget. Defaults to 1024.
    pub min_reasoning_budget: u32,
    /// Percentage table used to map levels to budgets.
    pub budget_percentages: &'a [(ReasoningLevel, f64)],
}

impl ReasoningBudgetOptions<'static> {
    /// Create reasoning-budget mapping options with AI SDK defaults.
    pub const fn new(
        reasoning: ReasoningLevel,
        max_output_tokens: u32,
        max_reasoning_budget: u32,
    ) -> Self {
        Self {
            reasoning,
            max_output_tokens,
            max_reasoning_budget,
            min_reasoning_budget: 1024,
            budget_percentages: &DEFAULT_REASONING_BUDGET_PERCENTAGES,
        }
    }
}

impl<'a> ReasoningBudgetOptions<'a> {
    /// Set the minimum reasoning budget.
    pub const fn with_min_reasoning_budget(mut self, min_reasoning_budget: u32) -> Self {
        self.min_reasoning_budget = min_reasoning_budget;
        self
    }

    /// Replace the reasoning-level percentage table.
    pub const fn with_budget_percentages<'b>(
        self,
        budget_percentages: &'b [(ReasoningLevel, f64)],
    ) -> ReasoningBudgetOptions<'b> {
        ReasoningBudgetOptions {
            reasoning: self.reasoning,
            max_output_tokens: self.max_output_tokens,
            max_reasoning_budget: self.max_reasoning_budget,
            min_reasoning_budget: self.min_reasoning_budget,
            budget_percentages,
        }
    }
}

/// Map a cross-provider reasoning level to an absolute provider token budget.
///
/// The computed budget is `round(max_output_tokens * percentage)`, bounded by
/// `min_reasoning_budget` and `max_reasoning_budget`, matching AI SDK's helper.
pub fn map_reasoning_to_provider_budget(
    options: ReasoningBudgetOptions<'_>,
    warnings: &mut Vec<Warning>,
) -> Option<u32> {
    let pct = options
        .budget_percentages
        .iter()
        .find_map(|(level, pct)| (*level == options.reasoning).then_some(*pct));

    let Some(pct) = pct else {
        warnings.push(unsupported_reasoning_warning(options.reasoning));
        return None;
    };

    let rounded = (f64::from(options.max_output_tokens) * pct).round();
    let rounded = if rounded.is_finite() && rounded > 0.0 {
        rounded as u64
    } else {
        0
    };

    let lower_bounded = rounded.max(u64::from(options.min_reasoning_budget));
    let upper_bounded = lower_bounded.min(u64::from(options.max_reasoning_budget));
    Some(upper_bounded as u32)
}

fn unsupported_reasoning_warning(reasoning: ReasoningLevel) -> Warning {
    Warning::unsupported(
        "reasoning",
        Some(format!(
            "reasoning \"{reasoning}\" is not supported by this model."
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_custom_reasoning_like_ai_sdk() {
        assert!(!is_custom_reasoning(None));
        assert!(!is_custom_reasoning(Some(
            &LanguageModelReasoning::ProviderDefault
        )));
        assert!(is_custom_reasoning(Some(&LanguageModelReasoning::None)));
        assert!(is_custom_reasoning(Some(&LanguageModelReasoning::Medium)));
    }

    #[test]
    fn converts_only_provider_reasoning_levels() {
        assert_eq!(
            ReasoningLevel::try_from(LanguageModelReasoning::XHigh).expect("xhigh"),
            ReasoningLevel::XHigh
        );
        assert!(matches!(
            ReasoningLevel::try_from(LanguageModelReasoning::None),
            Err(ReasoningLevelConversionError::Disabled)
        ));
        assert!(matches!(
            ReasoningLevel::try_from(LanguageModelReasoning::ProviderDefault),
            Err(ReasoningLevelConversionError::ProviderDefault)
        ));
    }

    #[test]
    fn maps_effort_and_records_warnings() {
        let mut warnings = Vec::new();
        let effort = map_reasoning_to_provider_effort(
            ReasoningLevel::Minimal,
            &[(ReasoningLevel::Minimal, "low")],
            &mut warnings,
        );

        assert_eq!(effort, Some("low"));
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0], Warning::Compatibility { .. }));

        let missing = map_reasoning_to_provider_effort::<&str>(
            ReasoningLevel::High,
            &[(ReasoningLevel::Minimal, "low")],
            &mut warnings,
        );

        assert_eq!(missing, None);
        assert_eq!(warnings.len(), 2);
        assert!(matches!(warnings[1], Warning::Unsupported { .. }));
    }

    #[test]
    fn maps_budget_with_default_percentages_and_bounds() {
        let mut warnings = Vec::new();
        let budget = map_reasoning_to_provider_budget(
            ReasoningBudgetOptions::new(ReasoningLevel::Medium, 10_000, 8_000),
            &mut warnings,
        );

        assert_eq!(budget, Some(3_000));
        assert!(warnings.is_empty());

        let budget = map_reasoning_to_provider_budget(
            ReasoningBudgetOptions::new(ReasoningLevel::Minimal, 10_000, 8_000),
            &mut warnings,
        );

        assert_eq!(budget, Some(1_024));
    }

    #[test]
    fn budget_missing_percentage_records_warning() {
        let mut warnings = Vec::new();
        let budget = map_reasoning_to_provider_budget(
            ReasoningBudgetOptions::new(ReasoningLevel::High, 10_000, 8_000)
                .with_budget_percentages(&[(ReasoningLevel::Low, 0.1)]),
            &mut warnings,
        );

        assert_eq!(budget, None);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0], Warning::Unsupported { .. }));
    }
}
