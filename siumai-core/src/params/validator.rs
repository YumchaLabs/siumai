//! Minimal Parameter Validation System
//!
//! This module provides provider-agnostic parameter validation.
//!
//! ## Validation Philosophy
//!
//! This validator uses a **minimal validation approach** to avoid maintenance overhead
//! as LLM models evolve. Instead of tracking provider-specific limits:
//!
//! - **Basic validation only**: Only validates fundamental constraints (e.g., non-negative values)
//! - **No warnings**: No provider-specific suggestions that require maintenance
//! - **Provider delegation**: Lets providers handle all their own specific limits
//! - **Zero maintenance**: Works with any new models without code changes

use crate::error::LlmError;
use crate::types::{CommonParams, ProviderType};

/// Enhanced parameter validator with provider-agnostic checks.
pub struct EnhancedParameterValidator;

impl EnhancedParameterValidator {
    /// Validates parameters with provider-agnostic sanity checks.
    pub fn validate_for_provider(
        params: &CommonParams,
        provider_type: &ProviderType,
    ) -> Result<ValidationReport, LlmError> {
        let mut report = ValidationReport::new(provider_type.clone());
        let mut has_errors = false;

        // Fast validation with early returns for better performance

        // Validate temperature with minimal validation (only basic constraints)
        if let Some(temp) = params.temperature {
            // Only validate that temperature is non-negative
            if temp < 0.0 {
                report.add_error(ValidationError::OutOfRange {
                    parameter: "temperature".to_string(),
                    value: temp.to_string(),
                    min: 0.0,
                    max: f64::INFINITY,
                    provider: format!("{provider_type:?}"),
                });
                has_errors = true;
            } else {
                // All non-negative values are accepted - let the provider handle limits
                report.add_valid_param("temperature".to_string());
            }
        }

        // Validate max_tokens with minimal validation (only basic constraints)
        if let Some(max_tokens) = params.max_tokens {
            // Only validate that max_tokens is positive
            if max_tokens == 0 {
                report.add_error(ValidationError::OutOfRange {
                    parameter: "max_tokens".to_string(),
                    value: max_tokens.to_string(),
                    min: 1.0,
                    max: f64::INFINITY,
                    provider: format!("{provider_type:?}"),
                });
                has_errors = true;
            } else {
                // All positive values are accepted - let the provider handle limits
                report.add_valid_param("max_tokens".to_string());
            }
        }

        // Validate top_p with simplified range checks
        if let Some(top_p) = params.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                report.add_error(ValidationError::OutOfRange {
                    parameter: "top_p".to_string(),
                    value: top_p.to_string(),
                    min: 0.0,
                    max: 1.0,
                    provider: format!("{provider_type:?}"),
                });
                has_errors = true;
            } else {
                report.add_valid_param("top_p".to_string());
            }
        }

        // Stop sequences are provider-owned semantically, but empty sequences are never useful.
        if let Some(stop_sequences) = &params.stop_sequences
            && let Some(index) = stop_sequences.iter().position(|value| value.is_empty())
        {
            report.add_error(ValidationError::InvalidFormat {
                parameter: format!("stop_sequences[{index}]"),
                value: String::new(),
                expected_format: "non-empty string".to_string(),
            });
            has_errors = true;
        }

        if has_errors {
            Err(LlmError::InvalidParameter(format!(
                "Parameter validation failed for {:?}: {}",
                provider_type,
                report.error_summary()
            )))
        } else {
            Ok(report)
        }
    }

    /// Cross-provider parameter compatibility check.
    ///
    /// This check is intentionally provider-agnostic. Provider/model support is validated by
    /// provider-owned config and request layers, not by `siumai-core`.
    pub fn check_cross_provider_compatibility(
        params: &CommonParams,
        source_provider: &ProviderType,
        target_provider: &ProviderType,
    ) -> CompatibilityReport {
        let mut report = CompatibilityReport::new(source_provider.clone(), target_provider.clone());

        // Use simplified, provider-agnostic constraints for compatibility checking
        let target_constraints = super::mapper::ParameterConstraints::default();

        // Check temperature compatibility
        if let Some(temp) = params.temperature
            && (temp < target_constraints.temperature_min
                || temp > target_constraints.temperature_max)
        {
            report.add_incompatibility(ParameterIncompatibility {
                parameter: "temperature".to_string(),
                issue: format!(
                    "Value {} is outside target provider range [{}, {}]",
                    temp, target_constraints.temperature_min, target_constraints.temperature_max
                ),
                suggestion: Some(format!(
                    "Clamp to range [{}, {}]",
                    target_constraints.temperature_min, target_constraints.temperature_max
                )),
            });
        }

        report
    }

    /// Optimize parameters with provider-agnostic constraints.
    pub fn optimize_for_provider(
        params: &mut CommonParams,
        provider_type: &ProviderType,
    ) -> OptimizationReport {
        let mut report = OptimizationReport::new(provider_type.clone());
        // Use default constraints for optimization (provider-agnostic)
        let constraints = super::mapper::ParameterConstraints::default();

        // Optimize temperature (only clamp negative values)
        if let Some(temp) = params.temperature
            && temp < 0.0
        {
            let optimal_temp = 0.0;
            report.add_optimization(ParameterOptimization {
                parameter: "temperature".to_string(),
                original_value: temp.to_string(),
                optimized_value: optimal_temp.to_string(),
                reason: "Clamped negative temperature to 0.0".to_string(),
            });
            params.temperature = Some(optimal_temp);
        }
        // Note: We don't clamp high temperatures anymore, let the provider handle it

        // Optimize max_tokens (only fix zero/invalid values)
        if let Some(max_tokens) = params.max_tokens
            && max_tokens == 0
        {
            let optimal_tokens = 1;
            report.add_optimization(ParameterOptimization {
                parameter: "max_tokens".to_string(),
                original_value: max_tokens.to_string(),
                optimized_value: optimal_tokens.to_string(),
                reason: "Changed zero max_tokens to 1 (minimum valid value)".to_string(),
            });
            params.max_tokens = Some(optimal_tokens);
        }
        // Note: We don't clamp large max_tokens anymore, let the provider handle it

        // Optimize top_p
        if let Some(top_p) = params.top_p {
            let optimal_top_p = top_p.clamp(constraints.top_p_min, constraints.top_p_max);
            if optimal_top_p != top_p {
                report.add_optimization(ParameterOptimization {
                    parameter: "top_p".to_string(),
                    original_value: top_p.to_string(),
                    optimized_value: optimal_top_p.to_string(),
                    reason: "Clamped to provider constraints".to_string(),
                });
                params.top_p = Some(optimal_top_p);
            }
        }

        report
    }
}

/// Validation report containing errors, warnings, and valid parameters
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub provider: ProviderType,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub valid_params: Vec<String>,
}

impl ValidationReport {
    pub const fn new(provider: ProviderType) -> Self {
        Self {
            provider,
            errors: Vec::new(),
            warnings: Vec::new(),
            valid_params: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    pub fn add_valid_param(&mut self, param: String) {
        self.valid_params.push(param);
    }

    pub const fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn error_summary(&self) -> String {
        self.errors
            .iter()
            .map(|e| format!("{e:?}"))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Validation error types
#[derive(Debug, Clone)]
pub enum ValidationError {
    OutOfRange {
        parameter: String,
        value: String,
        min: f64,
        max: f64,
        provider: String,
    },
    InvalidFormat {
        parameter: String,
        value: String,
        expected_format: String,
    },
}

/// Validation warning types
#[derive(Debug, Clone)]
pub enum ValidationWarning {
    UnsupportedParameter {
        parameter: String,
        provider: String,
    },
    SuboptimalValue {
        parameter: String,
        value: String,
        suggestion: String,
    },
}

/// Cross-provider compatibility report
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    pub source_provider: ProviderType,
    pub target_provider: ProviderType,
    pub incompatibilities: Vec<ParameterIncompatibility>,
}

impl CompatibilityReport {
    pub const fn new(source: ProviderType, target: ProviderType) -> Self {
        Self {
            source_provider: source,
            target_provider: target,
            incompatibilities: Vec::new(),
        }
    }

    pub fn add_incompatibility(&mut self, incompatibility: ParameterIncompatibility) {
        self.incompatibilities.push(incompatibility);
    }

    pub const fn is_compatible(&self) -> bool {
        self.incompatibilities.is_empty()
    }
}

/// Parameter incompatibility description
#[derive(Debug, Clone)]
pub struct ParameterIncompatibility {
    pub parameter: String,
    pub issue: String,
    pub suggestion: Option<String>,
}

/// Parameter optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub provider: ProviderType,
    pub optimizations: Vec<ParameterOptimization>,
}

impl OptimizationReport {
    pub const fn new(provider: ProviderType) -> Self {
        Self {
            provider,
            optimizations: Vec::new(),
        }
    }

    pub fn add_optimization(&mut self, optimization: ParameterOptimization) {
        self.optimizations.push(optimization);
    }

    pub const fn has_optimizations(&self) -> bool {
        !self.optimizations.is_empty()
    }
}

/// Parameter optimization description
#[derive(Debug, Clone)]
pub struct ParameterOptimization {
    pub parameter: String,
    pub original_value: String,
    pub optimized_value: String,
    pub reason: String,
}
