//! Minimal Parameter Validation System
//!
//! This module provides minimal parameter validation for all providers,
//! including cross-provider compatibility checks and parameter optimization.
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

/// Enhanced parameter validator with cross-provider support
pub struct EnhancedParameterValidator;

impl EnhancedParameterValidator {
    /// Validates parameters for a specific provider with simplified logic for better performance
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

        // Simplified model validation
        if !params.model.is_empty() && !Self::is_model_supported(&params.model, provider_type) {
            report.add_warning(ValidationWarning::UnsupportedModel {
                model: params.model.clone(),
                provider: format!("{provider_type:?}"),
                suggestion: Self::suggest_alternative_model(&params.model, provider_type),
            });
        }

        // Simplified stop sequences validation
        if let Some(stop_sequences) = &params.stop_sequences {
            let max_sequences = Self::max_stop_sequences(provider_type);
            if stop_sequences.len() > max_sequences {
                report.add_error(ValidationError::TooManyStopSequences {
                    count: stop_sequences.len(),
                    max: max_sequences,
                    provider: format!("{provider_type:?}"),
                });
                has_errors = true;
            }
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

    /// Cross-provider parameter compatibility check
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

        // Check model compatibility
        if !params.model.is_empty() && !Self::is_model_supported(&params.model, target_provider) {
            let suggested_model = Self::suggest_alternative_model(&params.model, target_provider);
            report.add_incompatibility(ParameterIncompatibility {
                parameter: "model".to_string(),
                issue: format!("Model '{}' not supported by target provider", params.model),
                suggestion: suggested_model.map(|m| format!("Use '{m}' instead")),
            });
        }

        report
    }

    /// Optimize parameters for a specific provider
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

    fn is_model_supported(model: &str, provider_type: &ProviderType) -> bool {
        match provider_type {
            ProviderType::OpenAi => model.starts_with("gpt-") || model.starts_with("o1-"),
            ProviderType::Azure => true,
            ProviderType::Anthropic => model.starts_with("claude-"),
            ProviderType::Gemini => model.starts_with("gemini-"),
            ProviderType::Vertex => {
                model.starts_with("gemini-")
                    || model.starts_with("imagen-")
                    || model.starts_with("text-embedding-")
                    || model.starts_with("multimodalembedding")
            }
            ProviderType::AnthropicVertex => model.starts_with("claude-"),
            ProviderType::VertexMaas => {
                model.ends_with("-maas")
                    || matches!(
                        model,
                        "deepseek-ai/deepseek-r1-0528-maas"
                            | "deepseek-ai/deepseek-v3.1-maas"
                            | "deepseek-ai/deepseek-v3.2-maas"
                            | "openai/gpt-oss-120b-maas"
                            | "openai/gpt-oss-20b-maas"
                            | "meta/llama-4-maverick-17b-128e-instruct-maas"
                            | "meta/llama-4-scout-17b-16e-instruct-maas"
                            | "minimax/minimax-m2-maas"
                            | "qwen/qwen3-coder-480b-a35b-instruct-maas"
                            | "qwen/qwen3-next-80b-a3b-instruct-maas"
                            | "qwen/qwen3-next-80b-a3b-thinking-maas"
                            | "moonshotai/kimi-k2-thinking-maas"
                    )
            }
            ProviderType::DeepSeek => {
                matches!(model, "chat" | "reasoner") || model.starts_with("deepseek-")
            }
            ProviderType::DeepInfra => true,
            ProviderType::Cohere => {
                model.starts_with("command")
                    || model.starts_with("embed-")
                    || model.starts_with("rerank-")
            }
            ProviderType::TogetherAi => true,
            ProviderType::Bedrock => true,
            ProviderType::Mistral => true,
            ProviderType::Fireworks => true,
            ProviderType::Perplexity => true,
            ProviderType::XAI => model.starts_with("grok-"),
            ProviderType::Ollama => true, // Ollama supports various models
            ProviderType::Custom(_) => true,
            ProviderType::Groq => {
                model.contains("llama")
                    || model.contains("mixtral")
                    || model.contains("gemma")
                    || model.contains("whisper")
            }
            ProviderType::MiniMaxi => {
                model.starts_with("MiniMax-")
                    || model.starts_with("speech-")
                    || model.starts_with("hailuo-")
                    || model.starts_with("music-")
            }
        }
    }

    fn suggest_alternative_model(model: &str, provider_type: &ProviderType) -> Option<String> {
        match provider_type {
            ProviderType::OpenAi => {
                if model.contains("4") {
                    Some("gpt-4".to_string())
                } else if model.contains("3.5") {
                    Some("gpt-3.5-turbo".to_string())
                } else {
                    Some("gpt-4".to_string())
                }
            }
            ProviderType::Azure => None,
            ProviderType::Anthropic => Some("claude-3-5-sonnet-20241022".to_string()),
            ProviderType::Gemini => Some("gemini-1.5-pro".to_string()),
            ProviderType::Vertex => Some("gemini-2.5-flash".to_string()),
            ProviderType::AnthropicVertex => Some("claude-sonnet-4-5-latest".to_string()),
            ProviderType::VertexMaas => Some("deepseek-ai/deepseek-v3.2-maas".to_string()),
            ProviderType::DeepSeek => Some("deepseek-chat".to_string()),
            ProviderType::DeepInfra => Some("meta-llama/Llama-3.3-70B-Instruct".to_string()),
            ProviderType::Cohere => Some("command-a-03-2025".to_string()),
            ProviderType::TogetherAi => {
                Some("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string())
            }
            ProviderType::Bedrock => None,
            ProviderType::Mistral => Some("mistral-large-latest".to_string()),
            ProviderType::Fireworks => {
                Some("accounts/fireworks/models/llama-v3p1-8b-instruct".to_string())
            }
            ProviderType::Perplexity => Some("llama-3.1-sonar-small-128k-online".to_string()),
            ProviderType::XAI => Some("grok-beta".to_string()),
            ProviderType::Ollama => Some("llama3.2:latest".to_string()),
            ProviderType::Custom(_) => None,
            ProviderType::Groq => Some("llama-3.3-70b-versatile".to_string()),
            ProviderType::MiniMaxi => Some("MiniMax-M2".to_string()),
        }
    }

    const fn max_stop_sequences(provider_type: &ProviderType) -> usize {
        match provider_type {
            ProviderType::OpenAi => 4,
            ProviderType::Azure => 10,
            ProviderType::Anthropic => 5,
            ProviderType::Gemini => 5,
            ProviderType::Vertex => 5,
            ProviderType::AnthropicVertex => 5,
            ProviderType::VertexMaas => 4,
            ProviderType::DeepSeek => 4,
            ProviderType::DeepInfra => 4,
            ProviderType::Cohere => 5,
            ProviderType::TogetherAi => 4,
            ProviderType::Bedrock => 10,
            ProviderType::Mistral => 4,
            ProviderType::Fireworks => 4,
            ProviderType::Perplexity => 4,
            ProviderType::XAI => 4,
            ProviderType::Ollama => 10,
            ProviderType::Custom(_) => 10,
            ProviderType::Groq => 4,
            ProviderType::MiniMaxi => 4, // OpenAI-compatible
        }
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
    TooManyStopSequences {
        count: usize,
        max: usize,
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
    UnsupportedModel {
        model: String,
        provider: String,
        suggestion: Option<String>,
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
