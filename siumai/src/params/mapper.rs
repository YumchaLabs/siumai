//! Parameter constraints (kept for validator/compat checks)
//!
//! Historical `ParameterMapper` has been fully removed; parameter mapping/validation
//! is handled by provider-local Transformers. This module only keeps
//! provider-agnostic `ParameterConstraints` for generic validator usage.

/// Parameter constraints for validation
#[derive(Debug, Clone)]
pub struct ParameterConstraints {
    pub temperature_min: f64,
    pub temperature_max: f64,
    pub max_tokens_min: u64,
    pub max_tokens_max: u64,
    pub top_p_min: f64,
    pub top_p_max: f64,
}

impl Default for ParameterConstraints {
    fn default() -> Self {
        Self {
            temperature_min: 0.0,
            temperature_max: 2.0,
            max_tokens_min: 1,
            max_tokens_max: 100_000,
            top_p_min: 0.0,
            top_p_max: 1.0,
        }
    }
}

// ParameterMapperFactory and ParameterMappingUtils removed in favor of Transformers.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_constraints_defaults() {
        let constraints = ParameterConstraints::default();
        assert_eq!(constraints.temperature_min, 0.0);
        assert_eq!(constraints.temperature_max, 2.0);
        assert_eq!(constraints.max_tokens_min, 1);
        assert_eq!(constraints.top_p_min, 0.0);
        assert_eq!(constraints.top_p_max, 1.0);
    }
}
