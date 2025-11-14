//! Provider adapter trait and compatibility flags (extracted)

use crate::types::{FieldAccessor, FieldMappings, ModelConfig, RequestType};
use reqwest::header::HeaderMap;
use siumai_core::error::LlmError;
use siumai_core::traits::ProviderCapabilities;
use siumai_core::types::common::HttpConfig;
use std::collections::HashMap;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProviderCompatibility {
    pub supports_array_content: bool,
    pub supports_stream_options: bool,
    pub supports_developer_role: bool,
    pub supports_enable_thinking: bool,
    pub supports_service_tier: bool,
    pub force_streaming_models: Vec<String>,
    pub custom_flags: HashMap<String, bool>,
}

impl ProviderCompatibility {
    pub fn openai_standard() -> Self {
        Self {
            supports_array_content: true,
            supports_stream_options: true,
            supports_developer_role: true,
            supports_enable_thinking: true,
            supports_service_tier: true,
            force_streaming_models: vec![],
            custom_flags: HashMap::new(),
        }
    }
    pub fn deepseek() -> Self {
        Self {
            supports_array_content: false,
            supports_stream_options: true,
            supports_developer_role: true,
            supports_enable_thinking: false,
            supports_service_tier: false,
            force_streaming_models: vec!["deepseek-reasoner".into()],
            custom_flags: HashMap::new(),
        }
    }
    pub fn limited_compatibility() -> Self {
        Self::default()
    }
}

pub trait ProviderAdapter: Send + Sync + std::fmt::Debug {
    fn provider_id(&self) -> std::borrow::Cow<'static, str>;
    fn transform_request_params(
        &self,
        params: &mut serde_json::Value,
        model: &str,
        request_type: RequestType,
    ) -> Result<(), LlmError>;
    fn get_field_mappings(&self, model: &str) -> FieldMappings;
    fn get_model_config(&self, model: &str) -> ModelConfig;
    fn get_field_accessor(&self) -> Box<dyn FieldAccessor>;
    fn capabilities(&self) -> ProviderCapabilities;
    fn base_url(&self) -> &str;
    fn clone_adapter(&self) -> Box<dyn ProviderAdapter>;
    fn supports_image_generation(&self) -> bool {
        false
    }
    fn transform_image_request(
        &self,
        _request: &mut siumai_core::types::image::ImageGenerationRequest,
    ) -> Result<(), LlmError> {
        Ok(())
    }
    fn get_supported_image_sizes(&self) -> Vec<String> {
        vec!["256x256".into(), "512x512".into(), "1024x1024".into()]
    }
    fn get_supported_image_formats(&self) -> Vec<String> {
        vec!["url".into(), "b64_json".into()]
    }
    fn supports_image_editing(&self) -> bool {
        self.supports_image_generation()
    }
    fn supports_image_variations(&self) -> bool {
        self.supports_image_generation()
    }

    // Optional: HTTP headers override hooks
    fn custom_headers(&self) -> HeaderMap {
        HeaderMap::new()
    }
    fn http_config_override(&self) -> Option<HttpConfig> {
        None
    }

    fn compatibility(&self) -> ProviderCompatibility {
        ProviderCompatibility::openai_standard()
    }
    fn apply_http_config(&self, http: HttpConfig) -> HttpConfig {
        http
    }
    fn validate_model(&self, _model: &str) -> Result<(), LlmError> {
        Ok(())
    }
    fn route_for(&self, req: RequestType) -> &'static str {
        match req {
            RequestType::Chat => "chat/completions",
            RequestType::Embedding => "embeddings",
            RequestType::ImageGeneration => "images/generations",
            RequestType::Rerank => "rerank",
        }
    }
}
