//! OpenAI provider adapter (externalized)
//!
//! 提供与聚合层一致的 `OpenAiStandardAdapter`，实现自
//! `siumai-provider-openai-compatible` 内定义的 `ProviderAdapter` 接口，
//! 以便共享 OpenAI-compatible 的转换与能力检测逻辑。

use siumai_core::error::LlmError;
use siumai_core::traits::ProviderCapabilities;
use siumai_provider_openai_compatible::adapter::{ProviderAdapter, ProviderCompatibility};
use siumai_provider_openai_compatible::types::{
    FieldAccessor, FieldMappings, JsonFieldAccessor, ModelConfig, RequestType,
};

#[derive(Debug, Clone)]
pub struct OpenAiStandardAdapter {
    pub base_url: String,
}

impl Default for OpenAiStandardAdapter {
    fn default() -> Self {
        Self {
            base_url: crate::constants::OPENAI_V1_ENDPOINT.to_string(),
        }
    }
}

impl ProviderAdapter for OpenAiStandardAdapter {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("openai")
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
        FieldMappings::default()
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

    fn get_field_accessor(&self) -> Box<dyn FieldAccessor> {
        Box::new(JsonFieldAccessor)
    }

    fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
        Box::new(self.clone())
    }
}
