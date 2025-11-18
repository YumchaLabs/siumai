//! Minimal registry types for OpenAI-compatible providers (extracted)

use crate::types::{FieldMappings, JsonFieldAccessor};
use siumai_core::traits::ProviderCapabilities;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub field_mappings: ProviderFieldMappings,
    pub capabilities: Vec<String>,
    pub default_model: Option<String>,
    pub supports_reasoning: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ProviderFieldMappings {
    pub thinking_fields: Vec<String>,
    pub content_field: String,
    pub tool_calls_field: String,
    pub role_field: String,
}

impl From<&ProviderFieldMappings> for FieldMappings {
    fn from(p: &ProviderFieldMappings) -> Self {
        FieldMappings {
            thinking_fields: p
                .thinking_fields
                .iter()
                .map(|s| std::borrow::Cow::Owned(s.clone()))
                .collect(),
            content_field: std::borrow::Cow::Owned(p.content_field.clone()),
            tool_calls_field: std::borrow::Cow::Owned(p.tool_calls_field.clone()),
            role_field: std::borrow::Cow::Owned(p.role_field.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConfigurableAdapter {
    pub(crate) config: ProviderConfig,
}

impl ConfigurableAdapter {
    pub fn new(config: ProviderConfig) -> Self {
        Self { config }
    }
}

impl crate::adapter::ProviderAdapter for ConfigurableAdapter {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.config.id.clone())
    }
    fn transform_request_params(
        &self,
        _params: &mut serde_json::Value,
        _model: &str,
        _request_type: crate::types::RequestType,
    ) -> Result<(), siumai_core::error::LlmError> {
        Ok(())
    }
    fn get_field_mappings(&self, _model: &str) -> FieldMappings {
        FieldMappings::from(&self.config.field_mappings)
    }
    fn get_model_config(&self, _model: &str) -> crate::types::ModelConfig {
        crate::types::ModelConfig {
            supports_thinking: self.config.supports_reasoning,
            ..Default::default()
        }
    }
    fn get_field_accessor(&self) -> Box<dyn crate::types::FieldAccessor> {
        Box::new(JsonFieldAccessor)
    }
    fn capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::new().with_chat().with_streaming();
        if self.config.capabilities.contains(&"tools".to_string()) {
            caps = caps.with_tools();
        }
        if self.config.capabilities.contains(&"vision".to_string()) {
            caps = caps.with_vision();
        }
        if self.config.capabilities.contains(&"embedding".to_string()) {
            caps = caps.with_embedding();
        }
        if self.config.capabilities.contains(&"rerank".to_string()) {
            caps = caps.with_custom_feature("rerank", true);
        }
        if self.config.supports_reasoning {
            caps = caps.with_custom_feature("reasoning", true);
        }
        caps
    }
    fn base_url(&self) -> &str {
        &self.config.base_url
    }
    fn clone_adapter(&self) -> Box<dyn crate::adapter::ProviderAdapter> {
        Box::new(self.clone())
    }
    fn custom_headers(&self) -> reqwest::header::HeaderMap {
        reqwest::header::HeaderMap::new()
    }
    fn http_config_override(&self) -> Option<siumai_core::types::common::HttpConfig> {
        None
    }
    fn compatibility(&self) -> crate::adapter::ProviderCompatibility {
        crate::adapter::ProviderCompatibility::openai_standard()
    }
    fn apply_http_config(
        &self,
        http: siumai_core::types::common::HttpConfig,
    ) -> siumai_core::types::common::HttpConfig {
        http
    }
    fn validate_model(&self, _model: &str) -> Result<(), siumai_core::error::LlmError> {
        Ok(())
    }
    fn route_for(&self, req: crate::types::RequestType) -> &'static str {
        match req {
            crate::types::RequestType::Chat => "chat/completions",
            crate::types::RequestType::Embedding => "embeddings",
            crate::types::RequestType::ImageGeneration => "images/generations",
            crate::types::RequestType::Rerank => "rerank",
        }
    }
    fn supports_image_generation(&self) -> bool {
        self.config
            .capabilities
            .contains(&"image_generation".to_string())
    }
    fn transform_image_request(
        &self,
        _request: &mut siumai_core::types::image::ImageGenerationRequest,
    ) -> Result<(), siumai_core::error::LlmError> {
        Ok(())
    }
    fn get_supported_image_sizes(&self) -> Vec<String> {
        vec![
            "256x256".into(),
            "512x512".into(),
            "1024x1024".into(),
            "1024x1792".into(),
            "1792x1024".into(),
        ]
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
}

/// Minimal external adapter resolver (scaffolding)
///
/// Returns a simple ConfigurableAdapter for common provider IDs.
/// This is a placeholder used during extraction; aggregator may override base_url later.
pub fn get_provider_adapter(
    provider_id: &str,
) -> Result<Arc<dyn crate::adapter::ProviderAdapter>, siumai_core::error::LlmError> {
    let (name, base_url, capabilities, supports_reasoning) = match provider_id {
        "deepseek" => (
            "DeepSeek",
            "https://api.deepseek.com/v1",
            vec![
                "chat".to_string(),
                "streaming".to_string(),
                "reasoning".to_string(),
            ],
            true,
        ),
        "openrouter" => (
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            false,
        ),
        _ => (
            provider_id,
            "https://api.openai.com/v1",
            vec!["chat".to_string(), "streaming".to_string()],
            false,
        ),
    };

    let cfg = ProviderConfig {
        id: provider_id.to_string(),
        name: name.to_string(),
        base_url: base_url.to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities,
        default_model: None,
        supports_reasoning,
    };
    Ok(Arc::new(ConfigurableAdapter::new(cfg)))
}

/// Minimal provider config by id (external)
pub fn get_provider_config(provider_id: &str) -> Option<ProviderConfig> {
    match provider_id {
        "deepseek" => Some(ProviderConfig {
            id: "deepseek".into(),
            name: "DeepSeek".into(),
            base_url: "https://api.deepseek.com/v1".into(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".into(), "streaming".into(), "reasoning".into()],
            default_model: Some("deepseek-reasoner".into()),
            supports_reasoning: true,
        }),
        "openrouter" => Some(ProviderConfig {
            id: "openrouter".into(),
            name: "OpenRouter".into(),
            base_url: "https://openrouter.ai/api/v1".into(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".into(), "streaming".into(), "tools".into()],
            default_model: None,
            supports_reasoning: false,
        }),
        "together" => Some(ProviderConfig {
            id: "together".into(),
            name: "Together".into(),
            base_url: "https://api.together.xyz/v1".into(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".into(), "streaming".into()],
            default_model: None,
            supports_reasoning: false,
        }),
        "fireworks" => Some(ProviderConfig {
            id: "fireworks".into(),
            name: "Fireworks".into(),
            base_url: "https://api.fireworks.ai/inference/v1".into(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".into(), "streaming".into(), "tools".into()],
            default_model: None,
            supports_reasoning: false,
        }),
        "siliconflow" => Some(ProviderConfig {
            id: "siliconflow".into(),
            name: "SiliconFlow".into(),
            base_url: "https://api.siliconflow.cn/v1".into(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".into(),
                "streaming".into(),
                "tools".into(),
                "image_generation".into(),
            ],
            default_model: None,
            supports_reasoning: false,
        }),
        "perplexity" => Some(ProviderConfig {
            id: "perplexity".into(),
            name: "Perplexity".into(),
            base_url: "https://api.perplexity.ai".into(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".into(), "streaming".into(), "search".into()],
            default_model: None,
            supports_reasoning: false,
        }),
        "groq" => Some(ProviderConfig {
            id: "groq".into(),
            name: "Groq".into(),
            base_url: "https://api.groq.com/openai/v1".into(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".into(), "streaming".into()],
            default_model: None,
            supports_reasoning: false,
        }),
        "ppio" => Some(ProviderConfig {
            id: "ppio".into(),
            name: "PPIO".into(),
            base_url: "https://api.ppinfra.com/v3/openai/v1".into(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".into(),
                "streaming".into(),
                "tools".into(),
                "vision".into(),
            ],
            default_model: Some("gpt-4o-mini".into()),
            supports_reasoning: false,
        }),
        "ocoolai" => Some(ProviderConfig {
            id: "ocoolai".into(),
            name: "OcoolAI".into(),
            base_url: "https://api.ocoolai.com/v1".into(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".into(),
                "streaming".into(),
                "tools".into(),
                "vision".into(),
            ],
            default_model: Some("gpt-4o-mini".into()),
            supports_reasoning: false,
        }),
        _ => None,
    }
}

/// Minimal list of known provider IDs (external)
pub fn list_provider_ids() -> Vec<String> {
    vec![
        "deepseek".into(),
        "openrouter".into(),
        "together".into(),
        "fireworks".into(),
        "siliconflow".into(),
        "perplexity".into(),
        "groq".into(),
        "ppio".into(),
        "ocoolai".into(),
    ]
}

/// Minimal capability check (external)
pub fn provider_supports_capability(provider_id: &str, capability: &str) -> bool {
    get_provider_config(provider_id)
        .map(|c| c.capabilities.contains(&capability.to_string()))
        .unwrap_or(false)
}
