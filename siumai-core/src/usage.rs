//! Provider-neutral usage accounting that preserves unknown values.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A usage count that distinguishes an absent value from a known zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UsageValue {
    #[default]
    Unknown,
    Known(u64),
}

impl UsageValue {
    pub const fn known(value: u64) -> Self {
        Self::Known(value)
    }

    pub const fn value(self) -> Option<u64> {
        match self {
            Self::Unknown => None,
            Self::Known(value) => Some(value),
        }
    }

    /// Add two values without inventing a count for an unknown operand.
    pub const fn checked_add(self, other: Self) -> Self {
        match (self, other) {
            (Self::Known(left), Self::Known(right)) => match left.checked_add(right) {
                Some(value) => Self::Known(value),
                None => Self::Unknown,
            },
            _ => Self::Unknown,
        }
    }
}

impl From<u64> for UsageValue {
    fn from(value: u64) -> Self {
        Self::Known(value)
    }
}

impl From<Option<u64>> for UsageValue {
    fn from(value: Option<u64>) -> Self {
        value.map_or(Self::Unknown, Self::Known)
    }
}

/// Stable usage dimensions plus provider-owned details.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    pub input_tokens: UsageValue,
    pub output_tokens: UsageValue,
    pub total_tokens: UsageValue,
    pub reasoning_tokens: UsageValue,
    pub cache_read_tokens: UsageValue,
    pub cache_write_tokens: UsageValue,
    pub audio_input_tokens: UsageValue,
    pub audio_output_tokens: UsageValue,
    pub orchestration_tokens: UsageValue,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

impl Usage {
    pub fn with_input_tokens(mut self, value: impl Into<UsageValue>) -> Self {
        self.input_tokens = value.into();
        self
    }

    pub fn with_output_tokens(mut self, value: impl Into<UsageValue>) -> Self {
        self.output_tokens = value.into();
        self
    }

    pub fn with_total_tokens(mut self, value: impl Into<UsageValue>) -> Self {
        self.total_tokens = value.into();
        self
    }

    pub fn with_reasoning_tokens(mut self, value: impl Into<UsageValue>) -> Self {
        self.reasoning_tokens = value.into();
        self
    }

    pub fn with_cache_read_tokens(mut self, value: impl Into<UsageValue>) -> Self {
        self.cache_read_tokens = value.into();
        self
    }

    pub fn with_cache_write_tokens(mut self, value: impl Into<UsageValue>) -> Self {
        self.cache_write_tokens = value.into();
        self
    }

    pub fn with_audio_input_tokens(mut self, value: impl Into<UsageValue>) -> Self {
        self.audio_input_tokens = value.into();
        self
    }

    pub fn with_audio_output_tokens(mut self, value: impl Into<UsageValue>) -> Self {
        self.audio_output_tokens = value.into();
        self
    }

    pub fn with_orchestration_tokens(mut self, value: impl Into<UsageValue>) -> Self {
        self.orchestration_tokens = value.into();
        self
    }

    pub fn with_provider_value(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.provider.insert(key.into(), value.into());
        self
    }

    /// Combine usage from two steps while preserving unknown dimensions.
    pub fn checked_add(&self, other: &Self) -> Self {
        let mut provider = self.provider.clone();
        for (key, value) in &other.provider {
            provider.insert(key.clone(), value.clone());
        }
        Self {
            input_tokens: self.input_tokens.checked_add(other.input_tokens),
            output_tokens: self.output_tokens.checked_add(other.output_tokens),
            total_tokens: self.total_tokens.checked_add(other.total_tokens),
            reasoning_tokens: self.reasoning_tokens.checked_add(other.reasoning_tokens),
            cache_read_tokens: self.cache_read_tokens.checked_add(other.cache_read_tokens),
            cache_write_tokens: self
                .cache_write_tokens
                .checked_add(other.cache_write_tokens),
            audio_input_tokens: self
                .audio_input_tokens
                .checked_add(other.audio_input_tokens),
            audio_output_tokens: self
                .audio_output_tokens
                .checked_add(other.audio_output_tokens),
            orchestration_tokens: self
                .orchestration_tokens
                .checked_add(other.orchestration_tokens),
            provider,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_distinguishes_unknown_from_known_zero() {
        let unknown = serde_json::to_value(UsageValue::Unknown).unwrap();
        let zero = serde_json::to_value(UsageValue::Known(0)).unwrap();

        assert_eq!(unknown, Value::Null);
        assert_eq!(zero, Value::from(0));
        assert_ne!(unknown, zero);
    }

    #[test]
    fn aggregation_does_not_turn_unknown_into_zero() {
        assert_eq!(
            UsageValue::Unknown.checked_add(UsageValue::Known(4)),
            UsageValue::Unknown
        );
        assert_eq!(
            UsageValue::Known(0).checked_add(UsageValue::Known(4)),
            UsageValue::Known(4)
        );
    }

    #[test]
    fn fluent_construction_preserves_known_zero_and_unknown() {
        let usage = Usage::default()
            .with_input_tokens(Some(0))
            .with_output_tokens(None)
            .with_provider_value("search_units", 1_u64);

        assert_eq!(usage.input_tokens, UsageValue::Known(0));
        assert_eq!(usage.output_tokens, UsageValue::Unknown);
        assert_eq!(usage.provider.get("search_units"), Some(&Value::from(1)));
    }
}
