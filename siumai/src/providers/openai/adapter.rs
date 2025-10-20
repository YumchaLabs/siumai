//! OpenAI standard adapter for OpenAI-compatible transformer path
//!
//! This adapter provides a thin wrapper implementing the ProviderAdapter
//! trait so OpenAI native code can reuse the OpenAI-compatible transformers
//! and streaming converters without redefining local adapters.

use crate::error::LlmError;
use crate::providers::openai_compatible::adapter::{ProviderAdapter, ProviderCompatibility};
use crate::providers::openai_compatible::types::{FieldMappings, ModelConfig, RequestType};
use crate::traits::ProviderCapabilities;

#[derive(Debug, Clone)]
pub struct OpenAiStandardAdapter {
    pub base_url: String,
}

impl ProviderAdapter for OpenAiStandardAdapter {
    fn provider_id(&self) -> &'static str {
        "openai"
    }

    fn transform_request_params(
        &self,
        _params: &mut serde_json::Value,
        _model: &str,
        _ty: RequestType,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn get_field_mappings(&self, _model: &str) -> FieldMappings {
        FieldMappings::standard()
    }

    fn get_model_config(&self, _model: &str) -> ModelConfig {
        ModelConfig::default()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn compatibility(&self) -> ProviderCompatibility {
        ProviderCompatibility::openai_standard()
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
        Box::new(self.clone())
    }
}
