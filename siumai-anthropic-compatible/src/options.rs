use std::collections::BTreeMap;

use serde_json::Value;
use siumai_core::{
    LanguageRequest, Message, MessageRole, ProviderOptionError, ProviderOptionLayers,
    ProviderOptionMerger, ProviderOptionOrigin, ProviderOptions,
};
use siumai_protocol_anthropic::messages::{
    MessagesMetadata, MessagesRequestOptions, MessagesServiceTier, OutputEffort, ServerFallbacks,
    ThinkingConfig, is_protected_option_field,
};

/// Provider-independent Messages call shaping understood by the compatible engine.
///
/// This type intentionally does not implement `TypedProviderOptions`: branded providers
/// own their namespaces and may expose their own typed option structs. After erasure, the
/// engine accepts the same typed Messages schema from provider-owned options and applies the
/// canonical precedence stack. Checked raw layers may only supply bounded, unprotected extras.
#[derive(Debug, Clone, Default)]
pub struct MessagesCallOptions {
    metadata: Option<MessagesMetadata>,
    thinking: Option<ThinkingConfig>,
    output_effort: Option<OutputEffort>,
    fallbacks: Option<ServerFallbacks>,
    top_k: Option<u64>,
    service_tier: Option<MessagesServiceTier>,
    extra: BTreeMap<String, Value>,
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

    pub fn fallbacks(&self) -> Option<&ServerFallbacks> {
        self.fallbacks.as_ref()
    }

    pub const fn top_k(&self) -> Option<u64> {
        self.top_k
    }

    pub const fn service_tier(&self) -> Option<MessagesServiceTier> {
        self.service_tier
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

    pub fn with_fallbacks(mut self, fallbacks: ServerFallbacks) -> Self {
        self.fallbacks = Some(fallbacks);
        self
    }

    pub const fn with_top_k(mut self, top_k: u64) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub const fn with_service_tier(mut self, service_tier: MessagesServiceTier) -> Self {
        self.service_tier = Some(service_tier);
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
        if let Some(fallbacks) = self.fallbacks {
            options = options.with_fallbacks(fallbacks);
        }
        if let Some(top_k) = self.top_k {
            options = options.with_top_k(top_k);
        }
        if let Some(service_tier) = self.service_tier {
            options = options.with_service_tier(service_tier);
        }
        options
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
        if let Some(fallbacks) = patch.fallbacks {
            self.fallbacks = fallbacks;
        }
        if let Some(top_k) = patch.top_k {
            self.top_k = top_k;
        }
        if let Some(service_tier) = patch.service_tier {
            self.service_tier = service_tier;
        }
        self.extra.extend(patch.extra);
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
    fallbacks: Option<Option<ServerFallbacks>>,
    top_k: Option<Option<u64>>,
    service_tier: Option<Option<MessagesServiceTier>>,
    extra: BTreeMap<String, Value>,
}

fn parse_patch(options: &ProviderOptions) -> Result<OptionsPatch, ProviderOptionError> {
    let mut patch = OptionsPatch::default();
    for (name, value) in options.value() {
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
            "fallbacks" => {
                patch.fallbacks = Some(parse_fallbacks(value)?);
            }
            "topk" => {
                patch.top_k = Some(parse_top_k(value)?);
            }
            "servicetier" => {
                if options.is_raw() {
                    return Err(ProviderOptionError::ProtectedField { path: name.clone() });
                }
                patch.service_tier = Some(parse_service_tier(value)?);
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

fn parse_service_tier(value: &Value) -> Result<Option<MessagesServiceTier>, ProviderOptionError> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value::<MessagesServiceTier>(value.clone())
        .map(Some)
        .map_err(|_| {
            rejected(
                "service_tier",
                "must be standard, priority, auto, or standard_only",
            )
        })
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
            | "servicetier"
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
