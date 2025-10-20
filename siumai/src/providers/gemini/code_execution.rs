//! Gemini Code Execution Support
//!
//! This module implements Google Gemini's code execution feature which allows
//! the model to write and execute Python code to solve problems.
//!
//! API Reference: <https://ai.google.dev/gemini-api/docs/code-execution>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::types::ChatResponse;

/// Code execution configuration for Gemini
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionConfig {
    /// Whether code execution is enabled
    pub enabled: bool,
    /// Maximum execution time in seconds
    pub timeout: Option<u32>,
    /// Allowed libraries/packages
    pub allowed_packages: Option<Vec<String>>,
    /// Code execution environment
    pub environment: CodeExecutionEnvironment,
    /// Whether to include execution output in response
    pub include_output: bool,
}

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            timeout: Some(30),
            allowed_packages: None,
            environment: CodeExecutionEnvironment::Python,
            include_output: true,
        }
    }
}

impl CodeExecutionConfig {
    /// Create a new code execution configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable code execution
    pub const fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Disable code execution
    pub const fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Set execution timeout
    pub const fn with_timeout(mut self, timeout_seconds: u32) -> Self {
        self.timeout = Some(timeout_seconds);
        self
    }

    /// Set allowed packages
    pub fn with_allowed_packages(mut self, packages: Vec<String>) -> Self {
        self.allowed_packages = Some(packages);
        self
    }

    /// Set execution environment
    pub const fn with_environment(mut self, environment: CodeExecutionEnvironment) -> Self {
        self.environment = environment;
        self
    }

    /// Set whether to include output
    pub const fn include_output(mut self, include: bool) -> Self {
        self.include_output = include;
        self
    }

    /// Convert to request parameters
    pub fn to_request_params(&self) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        if self.enabled {
            let mut tools = Vec::new();

            let mut code_execution_tool = serde_json::json!({
                "code_execution": {}
            });

            // Add timeout if specified
            if let Some(timeout) = self.timeout {
                code_execution_tool["code_execution"]["timeout"] =
                    serde_json::Value::Number(timeout.into());
            }

            // Add allowed packages if specified
            if let Some(ref packages) = self.allowed_packages {
                code_execution_tool["code_execution"]["allowed_packages"] =
                    serde_json::Value::Array(
                        packages
                            .iter()
                            .map(|p| serde_json::Value::String(p.clone()))
                            .collect(),
                    );
            }

            tools.push(code_execution_tool);
            params.insert("tools".to_string(), serde_json::Value::Array(tools));
        }

        params
    }
}

/// Code execution environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeExecutionEnvironment {
    /// Python environment
    Python,
    /// JavaScript environment (if supported)
    JavaScript,
}

/// Code execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    /// Executed code
    pub code: String,
    /// Execution output
    pub output: Option<String>,
    /// Execution error (if any)
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time: Option<u64>,
    /// Generated files or artifacts
    pub artifacts: Vec<CodeArtifact>,
}

/// Code artifact (generated files, plots, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeArtifact {
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// Artifact name/filename
    pub name: String,
    /// Artifact content (base64 encoded for binary)
    pub content: String,
    /// MIME type
    pub mime_type: String,
    /// Artifact metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Artifact type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Text file
    Text,
    /// Image file
    Image,
    /// Data file (CSV, JSON, etc.)
    Data,
    /// Plot/visualization
    Plot,
    /// Other binary file
    Binary,
}

/// Code execution parser
pub struct CodeExecutionParser;

impl CodeExecutionParser {
    /// Extract code execution results from Gemini response
    pub fn extract_execution_results(response: &ChatResponse) -> Vec<CodeExecutionResult> {
        let mut results = Vec::new();

        // Check provider data for code execution results
        if let Some(execution_data) = response.metadata.get("code_execution") {
            if let Some(executions) = execution_data.as_array() {
                for execution in executions {
                    if let Ok(result) = Self::parse_execution_result(execution) {
                        results.push(result);
                    }
                }
            } else if let Ok(result) = Self::parse_execution_result(execution_data) {
                results.push(result);
            }
        }

        results
    }

    /// Parse a single execution result
    fn parse_execution_result(data: &serde_json::Value) -> Result<CodeExecutionResult, LlmError> {
        let code = data
            .get("code")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let output = data
            .get("output")
            .and_then(|o| o.as_str())
            .map(std::string::ToString::to_string);

        let error = data
            .get("error")
            .and_then(|e| e.as_str())
            .map(std::string::ToString::to_string);

        let execution_time = data
            .get("execution_time")
            .and_then(serde_json::Value::as_u64);

        let artifacts = data
            .get("artifacts")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| Self::parse_artifact(item).ok())
                    .collect()
            })
            .unwrap_or_default();

        Ok(CodeExecutionResult {
            code,
            output,
            error,
            execution_time,
            artifacts,
        })
    }

    /// Parse a code artifact
    fn parse_artifact(data: &serde_json::Value) -> Result<CodeArtifact, LlmError> {
        let artifact_type = data
            .get("type")
            .and_then(|t| t.as_str())
            .map(|s| match s {
                "image" => ArtifactType::Image,
                "data" => ArtifactType::Data,
                "plot" => ArtifactType::Plot,
                "binary" => ArtifactType::Binary,
                _ => ArtifactType::Text,
            })
            .unwrap_or(ArtifactType::Text);

        let name = data
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("untitled")
            .to_string();

        let content = data
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let mime_type = data
            .get("mime_type")
            .and_then(|m| m.as_str())
            .unwrap_or("text/plain")
            .to_string();

        let metadata = data
            .get("metadata")
            .and_then(|m| m.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        Ok(CodeArtifact {
            artifact_type,
            name,
            content,
            mime_type,
            metadata,
        })
    }

    /// Format execution results for display
    pub fn format_execution_results(results: &[CodeExecutionResult]) -> String {
        let mut formatted = String::new();

        for (i, result) in results.iter().enumerate() {
            formatted.push_str(&format!("🔧 **Code Execution {}:**\n\n", i + 1));
            formatted.push_str(&format!("```python\n{}\n```\n\n", result.code));

            if let Some(ref output) = result.output {
                formatted.push_str(&format!("📤 **Output:**\n```\n{output}\n```\n\n"));
            }

            if let Some(ref error) = result.error {
                formatted.push_str(&format!("❌ **Error:**\n```\n{error}\n```\n\n"));
            }

            if let Some(time) = result.execution_time {
                formatted.push_str(&format!("⏱️ **Execution Time:** {time}ms\n\n"));
            }

            if !result.artifacts.is_empty() {
                formatted.push_str("📁 **Generated Artifacts:**\n");
                for artifact in &result.artifacts {
                    formatted.push_str(&format!("- {} ({})\n", artifact.name, artifact.mime_type));
                }
                formatted.push('\n');
            }
        }

        formatted
    }
}

/// Helper functions for common code execution patterns
pub mod patterns {
    use super::*;

    /// Create a data analysis configuration
    pub fn data_analysis_config() -> CodeExecutionConfig {
        CodeExecutionConfig::new()
            .enable()
            .with_timeout(60)
            .with_allowed_packages(vec![
                "pandas".to_string(),
                "numpy".to_string(),
                "matplotlib".to_string(),
                "seaborn".to_string(),
                "scipy".to_string(),
            ])
            .include_output(true)
    }

    /// Create a machine learning configuration
    pub fn machine_learning_config() -> CodeExecutionConfig {
        CodeExecutionConfig::new()
            .enable()
            .with_timeout(120)
            .with_allowed_packages(vec![
                "scikit-learn".to_string(),
                "pandas".to_string(),
                "numpy".to_string(),
                "matplotlib".to_string(),
                "seaborn".to_string(),
            ])
            .include_output(true)
    }

    /// Create a visualization configuration
    pub fn visualization_config() -> CodeExecutionConfig {
        CodeExecutionConfig::new()
            .enable()
            .with_timeout(45)
            .with_allowed_packages(vec![
                "matplotlib".to_string(),
                "seaborn".to_string(),
                "plotly".to_string(),
                "pandas".to_string(),
                "numpy".to_string(),
            ])
            .include_output(true)
    }

    /// Create a basic math configuration
    pub fn math_config() -> CodeExecutionConfig {
        CodeExecutionConfig::new()
            .enable()
            .with_timeout(30)
            .with_allowed_packages(vec![
                "numpy".to_string(),
                "scipy".to_string(),
                "sympy".to_string(),
                "math".to_string(),
            ])
            .include_output(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_execution_config() {
        let config = CodeExecutionConfig::new()
            .enable()
            .with_timeout(60)
            .with_allowed_packages(vec!["pandas".to_string(), "numpy".to_string()])
            .include_output(true);

        assert!(config.enabled);
        assert_eq!(config.timeout, Some(60));
        assert_eq!(
            config.allowed_packages,
            Some(vec!["pandas".to_string(), "numpy".to_string()])
        );
        assert!(config.include_output);
    }

    #[test]
    fn test_request_params() {
        let config = CodeExecutionConfig::new().enable().with_timeout(30);

        let params = config.to_request_params();
        assert!(params.contains_key("tools"));

        let tools = params.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0]["code_execution"].is_object());
    }

    #[test]
    fn test_data_analysis_pattern() {
        let config = patterns::data_analysis_config();
        assert!(config.enabled);
        assert_eq!(config.timeout, Some(60));
        assert!(config.allowed_packages.is_some());
        assert!(
            config
                .allowed_packages
                .as_ref()
                .unwrap()
                .contains(&"pandas".to_string())
        );
    }

    #[test]
    fn test_artifact_parsing() {
        let artifact_data = serde_json::json!({
            "type": "image",
            "name": "plot.png",
            "content": "base64encodeddata",
            "mime_type": "image/png",
            "metadata": {
                "width": 800,
                "height": 600
            }
        });

        let artifact = CodeExecutionParser::parse_artifact(&artifact_data).unwrap();
        assert!(matches!(artifact.artifact_type, ArtifactType::Image));
        assert_eq!(artifact.name, "plot.png");
        assert_eq!(artifact.mime_type, "image/png");
    }
}
