use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai_core::{
    LanguageRequest, Message, MessageRole, ProviderOptionError, ProviderOptionLayers,
    ProviderOptionMerger, ProviderOptionOrigin, ProviderOptionSelection, ProviderOptions,
};
use siumai_protocol_anthropic::messages::{
    CacheControl, ContextManagement, InferenceGeo, InferenceSpeed, McpServer, MessagesContainer,
    MessagesMetadata, MessagesRequestOptions, MessagesServiceTierPreference, OutputEffort,
    ServerFallbacks, ThinkingConfig, TokenTaskBudget, is_protected_option_field,
};

/// Provider-independent Messages call shaping understood by the compatible engine.
///
/// This type intentionally does not implement `TypedProviderOptions`: branded providers
/// own their namespaces and may expose their own typed option structs. After erasure, the
/// engine accepts the same typed Messages schema from provider-owned options and applies the
/// canonical precedence stack. Checked raw layers are retained as a bounded final body overlay;
/// canonical request fields and provider or transport security fields remain protected.
#[derive(Debug, Clone, Default)]
pub struct MessagesCallOptions {
    metadata: Option<MessagesMetadata>,
    thinking: Option<ThinkingConfig>,
    output_effort: Option<OutputEffort>,
    task_budget: Option<TokenTaskBudget>,
    fallbacks: Option<ServerFallbacks>,
    top_k: Option<u64>,
    service_tier: Option<MessagesServiceTierPreference>,
    cache_control: Option<CacheControl>,
    speed: Option<InferenceSpeed>,
    inference_geo: Option<InferenceGeo>,
    container: Option<MessagesContainer>,
    context_management: Option<ContextManagement>,
    mcp_servers: Option<Vec<McpServer>>,
    extra: BTreeMap<String, Value>,
    raw_body_overlay: BTreeMap<String, Value>,
}

impl MessagesCallOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn metadata(&self) -> Option<&MessagesMetadata> {
        self.metadata.as_ref()
    }

    pub const fn thinking(&self) -> Option<ThinkingConfig> {
        self.thinking
    }

    pub const fn output_effort(&self) -> Option<OutputEffort> {
        self.output_effort
    }

    pub const fn task_budget(&self) -> Option<TokenTaskBudget> {
        self.task_budget
    }

    pub fn fallbacks(&self) -> Option<&ServerFallbacks> {
        self.fallbacks.as_ref()
    }

    pub const fn top_k(&self) -> Option<u64> {
        self.top_k
    }

    pub const fn service_tier(&self) -> Option<MessagesServiceTierPreference> {
        self.service_tier
    }

    pub const fn cache_control(&self) -> Option<CacheControl> {
        self.cache_control
    }

    pub const fn speed(&self) -> Option<InferenceSpeed> {
        self.speed
    }

    pub const fn inference_geo(&self) -> Option<InferenceGeo> {
        self.inference_geo
    }

    pub fn container(&self) -> Option<&MessagesContainer> {
        self.container.as_ref()
    }

    pub fn context_management(&self) -> Option<&ContextManagement> {
        self.context_management.as_ref()
    }

    pub fn mcp_servers(&self) -> Option<&[McpServer]> {
        self.mcp_servers.as_deref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }

    pub fn with_metadata(mut self, metadata: MessagesMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn with_output_effort(mut self, effort: OutputEffort) -> Self {
        self.output_effort = Some(effort);
        self
    }

    pub const fn with_task_budget(mut self, task_budget: TokenTaskBudget) -> Self {
        self.task_budget = Some(task_budget);
        self
    }

    pub fn with_fallbacks(mut self, fallbacks: ServerFallbacks) -> Self {
        self.fallbacks = Some(fallbacks);
        self
    }

    pub const fn with_top_k(mut self, top_k: u64) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub const fn with_service_tier(mut self, service_tier: MessagesServiceTierPreference) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    pub const fn with_speed(mut self, speed: InferenceSpeed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub const fn with_inference_geo(mut self, inference_geo: InferenceGeo) -> Self {
        self.inference_geo = Some(inference_geo);
        self
    }

    pub fn with_container(mut self, container: MessagesContainer) -> Self {
        self.container = Some(container);
        self
    }

    pub fn with_context_management(mut self, context_management: ContextManagement) -> Self {
        self.context_management = Some(context_management);
        self
    }

    pub fn with_mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = Some(mcp_servers);
        self
    }

    pub fn with_extra(mut self, extra: BTreeMap<String, Value>) -> Self {
        self.extra = extra;
        self
    }

    pub(crate) fn into_protocol(self, stream: bool) -> MessagesRequestOptions {
        let mut options = MessagesRequestOptions::new(stream).with_extra(self.extra);
        if let Some(metadata) = self.metadata {
            options = options.with_metadata(metadata);
        }
        if let Some(thinking) = self.thinking {
            options = options.with_thinking(thinking);
        }
        if let Some(effort) = self.output_effort {
            options = options.with_output_effort(effort);
        }
        if let Some(task_budget) = self.task_budget {
            options = options.with_task_budget(task_budget);
        }
        if let Some(fallbacks) = self.fallbacks {
            options = options.with_fallbacks(fallbacks);
        }
        if let Some(top_k) = self.top_k {
            options = options.with_top_k(top_k);
        }
        if let Some(service_tier) = self.service_tier {
            options = options.with_service_tier(service_tier);
        }
        if let Some(cache_control) = self.cache_control {
            options = options.with_cache_control(cache_control);
        }
        if let Some(speed) = self.speed {
            options = options.with_speed(speed);
        }
        if let Some(inference_geo) = self.inference_geo {
            options = options.with_inference_geo(inference_geo);
        }
        if let Some(container) = self.container {
            options = options.with_container(container);
        }
        if let Some(context_management) = self.context_management {
            options = options.with_context_management(context_management);
        }
        if let Some(mcp_servers) = self.mcp_servers {
            options = options.with_mcp_servers(mcp_servers);
        }
        options
    }

    pub(crate) fn apply_raw_body_overlay(
        &self,
        body: &mut Value,
    ) -> Result<(), ProviderOptionError> {
        if self.raw_body_overlay.is_empty() {
            return Ok(());
        }
        let object = body.as_object_mut().ok_or_else(|| {
            rejected(
                "request",
                "Anthropic Messages encoding must produce a JSON object",
            )
        })?;
        object.extend(self.raw_body_overlay.clone());
        Ok(())
    }

    fn apply(&mut self, patch: OptionsPatch) {
        if let Some(metadata) = patch.metadata {
            self.metadata = metadata;
        }
        if let Some(thinking) = patch.thinking {
            self.thinking = thinking;
        }
        if let Some(output_effort) = patch.output_effort {
            self.output_effort = output_effort;
        }
        if let Some(task_budget) = patch.task_budget {
            self.task_budget = task_budget;
        }
        if let Some(fallbacks) = patch.fallbacks {
            self.fallbacks = fallbacks;
        }
        if let Some(top_k) = patch.top_k {
            self.top_k = top_k;
        }
        if let Some(service_tier) = patch.service_tier {
            self.service_tier = service_tier;
        }
        if let Some(cache_control) = patch.cache_control {
            self.cache_control = cache_control;
        }
        if let Some(speed) = patch.speed {
            self.speed = speed;
        }
        if let Some(inference_geo) = patch.inference_geo {
            self.inference_geo = inference_geo;
        }
        if let Some(container) = patch.container {
            self.container = container;
        }
        if let Some(context_management) = patch.context_management {
            self.context_management = context_management;
        }
        if let Some(mcp_servers) = patch.mcp_servers {
            self.mcp_servers = mcp_servers;
        }
        self.extra.extend(patch.extra);
        self.raw_body_overlay.extend(patch.raw_body_overlay);
    }

    fn validate_static(&self) -> Result<(), ProviderOptionError> {
        let mut request =
            LanguageRequest::new(vec![Message::text(MessageRole::User, "option validation")]);
        request.generation.max_output_tokens = Some(u64::MAX);
        self.clone()
            .into_protocol(false)
            .validate(&request)
            .map_err(codec_option_error)
    }
}

pub(crate) struct MessagesOptionMerger {
    defaults: MessagesCallOptions,
}

impl MessagesOptionMerger {
    pub(crate) fn new(defaults: MessagesCallOptions) -> Result<Self, ProviderOptionError> {
        defaults.validate_static()?;
        Ok(Self { defaults })
    }

    pub(crate) fn merge_selected(
        &self,
        layers: &ProviderOptionLayers,
        selection: &ProviderOptionSelection<'_>,
    ) -> Result<MessagesCallOptions, ProviderOptionError> {
        let mut merged = self.defaults.clone();
        let mut has_raw = false;
        for (origin, options) in layers.in_precedence_order() {
            self.validate_layer(origin, options)?;
            has_raw |= origin == ProviderOptionOrigin::RawOverride;
            merged.apply(parse_patch(options)?);
        }
        for options in selection.typed() {
            self.validate_layer(ProviderOptionOrigin::Call, options)?;
            merged.apply(parse_patch(options)?);
        }
        if let Some(options) = selection.raw_override() {
            if has_raw {
                return Err(ProviderOptionError::DuplicateRawTarget);
            }
            self.validate_layer(ProviderOptionOrigin::RawOverride, options)?;
            merged.apply(parse_patch(options)?);
        }
        Ok(merged)
    }
}

impl ProviderOptionMerger for MessagesOptionMerger {
    type Output = MessagesCallOptions;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        let patch = parse_patch(options)?;
        let mut layer = MessagesCallOptions::default();
        layer.apply(patch);
        layer.validate_static()
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = self.defaults.clone();
        for (_, options) in layers.in_precedence_order() {
            merged.apply(parse_patch(options)?);
        }
        Ok(merged)
    }
}

#[derive(Default)]
struct OptionsPatch {
    metadata: Option<Option<MessagesMetadata>>,
    thinking: Option<Option<ThinkingConfig>>,
    output_effort: Option<Option<OutputEffort>>,
    task_budget: Option<Option<TokenTaskBudget>>,
    fallbacks: Option<Option<ServerFallbacks>>,
    top_k: Option<Option<u64>>,
    service_tier: Option<Option<MessagesServiceTierPreference>>,
    cache_control: Option<Option<CacheControl>>,
    speed: Option<Option<InferenceSpeed>>,
    inference_geo: Option<Option<InferenceGeo>>,
    container: Option<Option<MessagesContainer>>,
    context_management: Option<Option<ContextManagement>>,
    mcp_servers: Option<Option<Vec<McpServer>>>,
    extra: BTreeMap<String, Value>,
    raw_body_overlay: BTreeMap<String, Value>,
}

fn parse_patch(options: &ProviderOptions) -> Result<OptionsPatch, ProviderOptionError> {
    let mut patch = OptionsPatch::default();
    for (name, value) in options.value() {
        if options.is_raw() {
            validate_raw_body_field(name, value, name)?;
            insert_extra(&mut patch.raw_body_overlay, name, value)?;
            continue;
        }
        match compact_name(name).as_str() {
            "metadata" => {
                patch.metadata = Some(parse_metadata(value)?);
            }
            "thinking" => {
                patch.thinking = Some(parse_thinking(value)?);
            }
            "outputeffort" | "effort" => {
                patch.output_effort = Some(parse_output_effort(value)?);
            }
            "taskbudget" => {
                patch.task_budget = Some(parse_typed(value, "task_budget")?);
            }
            "fallbacks" => {
                patch.fallbacks = Some(parse_fallbacks(value)?);
            }
            "topk" => {
                patch.top_k = Some(parse_top_k(value)?);
            }
            "servicetier" => {
                patch.service_tier = Some(parse_service_tier(value)?);
            }
            "cachecontrol" => {
                patch.cache_control = Some(parse_typed(value, "cache_control")?);
            }
            "speed" => {
                patch.speed = Some(parse_typed(value, "speed")?);
            }
            "inferencegeo" => {
                patch.inference_geo = Some(parse_typed(value, "inference_geo")?);
            }
            "container" => {
                patch.container = Some(parse_typed(value, "container")?);
            }
            "contextmanagement" => {
                patch.context_management = Some(parse_typed(value, "context_management")?);
            }
            "mcpservers" => {
                patch.mcp_servers = Some(parse_typed(value, "mcp_servers")?);
            }
            "extra" => {
                let object = value.as_object().ok_or_else(|| {
                    rejected(
                        "extra",
                        "must be an object containing additional request-body fields",
                    )
                })?;
                for (extra_name, extra_value) in object {
                    validate_extra_field(extra_name, extra_value, extra_name)?;
                    insert_extra(&mut patch.extra, extra_name, extra_value)?;
                }
            }
            "anthropictool" | "anthropictools" => {
                return Err(rejected(
                    name,
                    "Anthropic-defined tool wire semantics are not implemented by this engine",
                ));
            }
            _ => {
                validate_extra_field(name, value, name)?;
                insert_extra(&mut patch.extra, name, value)?;
            }
        }
    }
    Ok(patch)
}

fn parse_metadata(value: &Value) -> Result<Option<MessagesMetadata>, ProviderOptionError> {
    if value.is_null() {
        return Ok(None);
    }
    let decoded = serde_json::from_value::<MessagesMetadata>(value.clone())
        .map_err(|_| rejected("metadata", "must contain exactly one string user_id field"))?;
    MessagesMetadata::new(decoded.user_id().to_string())
        .map(Some)
        .map_err(codec_option_error)
}

fn parse_thinking(value: &Value) -> Result<Option<ThinkingConfig>, ProviderOptionError> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value::<ThinkingConfig>(value.clone())
        .map(Some)
        .map_err(|_| {
            rejected(
                "thinking",
                "must be a typed disabled, enabled, or adaptive thinking object",
            )
        })
}

fn parse_output_effort(value: &Value) -> Result<Option<OutputEffort>, ProviderOptionError> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value::<OutputEffort>(value.clone())
        .map(Some)
        .map_err(|_| rejected("output_effort", "must be low, medium, high, xhigh, or max"))
}

fn parse_fallbacks(value: &Value) -> Result<Option<ServerFallbacks>, ProviderOptionError> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value::<ServerFallbacks>(value.clone())
        .map(Some)
        .map_err(|_| rejected("fallbacks", "must be default or a typed fallback array"))
}

fn parse_top_k(value: &Value) -> Result<Option<u64>, ProviderOptionError> {
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_u64()
        .filter(|top_k| *top_k > 0)
        .map(Some)
        .ok_or_else(|| rejected("top_k", "must be a positive integer"))
}

fn parse_service_tier(
    value: &Value,
) -> Result<Option<MessagesServiceTierPreference>, ProviderOptionError> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value::<MessagesServiceTierPreference>(value.clone())
        .map(Some)
        .map_err(|_| rejected("service_tier", "must be auto or standard_only"))
}

fn parse_typed<T: DeserializeOwned>(
    value: &Value,
    field: &'static str,
) -> Result<Option<T>, ProviderOptionError> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|_| rejected(field, "must use the canonical typed option shape"))
}

fn validate_extra_field(name: &str, value: &Value, path: &str) -> Result<(), ProviderOptionError> {
    if is_engine_protected(name) || is_protected_option_field(name) || is_security_sensitive(name) {
        return Err(rejected(
            path,
            "field is owned by the engine or canonical codec",
        ));
    }
    validate_nested_extra(value, path)
}

fn validate_raw_body_field(
    name: &str,
    value: &Value,
    path: &str,
) -> Result<(), ProviderOptionError> {
    if is_canonical_body_field(name) || is_security_sensitive(name) {
        return Err(rejected(
            path,
            "field is owned by the canonical request, provider, or transport",
        ));
    }
    validate_nested_extra(value, path)
}

fn is_canonical_body_field(name: &str) -> bool {
    matches!(
        compact_name(name).as_str(),
        "model"
            | "messages"
            | "system"
            | "maxtokens"
            | "maxoutputtokens"
            | "stream"
            | "tools"
            | "toolchoice"
            | "temperature"
            | "topp"
            | "stopsequence"
            | "stopsequences"
            | "diagnostics"
    )
}

fn validate_nested_extra(value: &Value, path: &str) -> Result<(), ProviderOptionError> {
    match value {
        Value::Object(object) => {
            for (name, child) in object {
                let child_path = format!("{path}.{name}");
                if is_security_sensitive(name) {
                    return Err(rejected(
                        &child_path,
                        "credentials, endpoints, versions, and headers are protected",
                    ));
                }
                validate_nested_extra(child, &child_path)?;
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                validate_nested_extra(child, &format!("{path}[{index}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_engine_protected(name: &str) -> bool {
    matches!(
        compact_name(name).as_str(),
        "model"
            | "messages"
            | "system"
            | "maxtokens"
            | "maxoutputtokens"
            | "stream"
            | "tools"
            | "toolchoice"
            | "temperature"
            | "topp"
            | "topk"
            | "stopsequence"
            | "stopsequences"
            | "outputconfig"
            | "taskbudget"
            | "servicetier"
            | "cachecontrol"
            | "speed"
            | "inferencegeo"
            | "container"
            | "contextmanagement"
            | "mcpservers"
            | "mcptoolset"
            | "apikey"
            | "xapikey"
            | "authorization"
            | "auth"
            | "token"
            | "bearer"
            | "endpoint"
            | "baseurl"
            | "url"
            | "host"
            | "headers"
            | "header"
            | "anthropicversion"
            | "anthropicbeta"
            | "proxy"
            | "tls"
            | "audience"
    )
}

fn is_security_sensitive(name: &str) -> bool {
    let name = compact_name(name);
    matches!(
        name.as_str(),
        "apikey"
            | "xapikey"
            | "authorization"
            | "auth"
            | "token"
            | "bearer"
            | "endpoint"
            | "baseurl"
            | "url"
            | "host"
            | "headers"
            | "header"
            | "anthropicversion"
            | "anthropicbeta"
            | "proxy"
            | "tls"
            | "audience"
    ) || name.ends_with("apikey")
        || name.ends_with("token")
        || name.ends_with("credential")
        || name.ends_with("credentials")
        || name.ends_with("authorization")
        || name.ends_with("endpoint")
        || name.ends_with("baseurl")
        || name.ends_with("headers")
}

fn compact_name(name: &str) -> String {
    name.trim()
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn codec_option_error(error: impl std::fmt::Display) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: "messages".to_string(),
        reason: error.to_string(),
    }
}

fn rejected(path: &str, reason: &str) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: safe_path(path),
        reason: reason.to_string(),
    }
}

fn insert_extra(
    extra: &mut BTreeMap<String, Value>,
    name: &str,
    value: &Value,
) -> Result<(), ProviderOptionError> {
    if extra.insert(name.to_string(), value.clone()).is_some() {
        return Err(rejected(
            name,
            "field is defined both directly and inside the extra object",
        ));
    }
    Ok(())
}

fn safe_path(path: &str) -> String {
    let mut value = path
        .chars()
        .take(256)
        .map(|character| {
            if character.is_control() {
                '?'
            } else {
                character
            }
        })
        .collect::<String>();
    if path.chars().count() > 256 {
        value.push_str("...");
    }
    value
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use siumai_core::ProviderId;

    use super::*;

    #[test]
    fn raw_provider_values_bypass_closed_typed_decoding() {
        let raw = ProviderOptions::checked_raw(
            ProviderId::new("anthropic").expect("provider"),
            json!({
                "service_tier": "priority_v2",
                "output_config": {"effort": "ultra"}
            }),
        )
        .expect("raw options");
        let mut options = MessagesCallOptions::new();
        options.apply(parse_patch(&raw).expect("raw patch"));
        let mut body = json!({"model": "future-model", "messages": []});

        options
            .apply_raw_body_overlay(&mut body)
            .expect("raw overlay");

        assert_eq!(body["service_tier"], "priority_v2");
        assert_eq!(body["output_config"]["effort"], "ultra");
    }

    #[test]
    fn raw_provider_values_cannot_override_canonical_or_security_fields() {
        let canonical = ProviderOptions::checked_raw(
            ProviderId::new("anthropic").expect("provider"),
            json!({"model": "override"}),
        )
        .expect("bounded raw options");
        assert!(parse_patch(&canonical).is_err());

        let nested_secret = ProviderOptions::checked_raw(
            ProviderId::new("anthropic").expect("provider"),
            json!({"future_feature": {"authorization_token": "secret"}}),
        )
        .expect("bounded raw options");
        assert!(parse_patch(&nested_secret).is_err());
    }
}
