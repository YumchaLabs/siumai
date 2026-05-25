use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Dynamic provider sorting strategy for Vercel AI Gateway routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GatewaySort {
    Cost,
    Ttft,
    Tps,
}

/// Gateway-level service tier intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GatewayServiceTier {
    Flex,
    Priority,
}

/// Request-scoped BYOK provider timeouts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayProviderTimeouts {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byok: Option<BTreeMap<String, u64>>,
}

/// Typed Gateway provider options stored under `provider_options_map["gateway"]`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<GatewaySort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byok: Option<BTreeMap<String, Vec<serde_json::Value>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_data_retention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disallow_prompt_training: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hipaa_compliant: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quota_entity_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_timeouts: Option<GatewayProviderTimeouts>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<GatewayServiceTier>,

    /// Forward-compatible raw options merged at serialization boundaries by callers.
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl GatewayOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_order(mut self, order: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.order = Some(order.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_only(mut self, only: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.only = Some(only.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_sort(mut self, sort: GatewaySort) -> Self {
        self.sort = Some(sort);
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub const fn with_zero_data_retention(mut self, enabled: bool) -> Self {
        self.zero_data_retention = Some(enabled);
        self
    }

    pub const fn with_service_tier(mut self, service_tier: GatewayServiceTier) -> Self {
        self.service_tier = Some(service_tier);
        self
    }
}

pub type GatewayLanguageModelOptions = GatewayOptions;
pub type GatewayEmbeddingModelOptions = GatewayOptions;
