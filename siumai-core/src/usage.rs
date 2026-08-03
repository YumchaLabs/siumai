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
}
