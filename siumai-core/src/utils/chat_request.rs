use crate::types::{ChatRequest, CommonParams, HttpConfig, ProviderOptionsMap};

/// Shared defaults used to normalize chat requests before dispatch.
#[derive(Debug, Clone, Copy)]
pub struct ChatRequestDefaults<'a> {
    pub common_params: &'a CommonParams,
    pub provider_options_map: Option<&'a ProviderOptionsMap>,
    pub http_config: Option<&'a HttpConfig>,
}

impl<'a> ChatRequestDefaults<'a> {
    pub const fn new(common_params: &'a CommonParams) -> Self {
        Self {
            common_params,
            provider_options_map: None,
            http_config: None,
        }
    }

    pub const fn with_provider_options_map(
        mut self,
        provider_options_map: &'a ProviderOptionsMap,
    ) -> Self {
        self.provider_options_map = Some(provider_options_map);
        self
    }

    pub const fn with_http_config(mut self, http_config: &'a HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }
}

/// Merge request common params with provider/client defaults.
///
/// Request values win. Empty model strings are treated as missing.
pub fn merge_common_params(defaults: &CommonParams, request_params: CommonParams) -> CommonParams {
    CommonParams {
        model: if request_params.model.trim().is_empty() {
            defaults.model.clone()
        } else {
            request_params.model
        },
        temperature: request_params.temperature.or(defaults.temperature),
        max_tokens: request_params.max_tokens.or(defaults.max_tokens),
        max_completion_tokens: request_params
            .max_completion_tokens
            .or(defaults.max_completion_tokens),
        top_p: request_params.top_p.or(defaults.top_p),
        top_k: request_params.top_k.or(defaults.top_k),
        stop_sequences: request_params
            .stop_sequences
            .or_else(|| defaults.stop_sequences.clone()),
        seed: request_params.seed.or(defaults.seed),
        frequency_penalty: request_params
            .frequency_penalty
            .or(defaults.frequency_penalty),
        presence_penalty: request_params
            .presence_penalty
            .or(defaults.presence_penalty),
    }
}

/// Normalize a chat request against provider/client defaults.
///
/// This keeps the precedence consistent across providers:
/// explicit request values > provider/client defaults.
pub fn normalize_chat_request(
    mut request: ChatRequest,
    defaults: ChatRequestDefaults<'_>,
    stream: bool,
) -> ChatRequest {
    request.common_params = merge_common_params(defaults.common_params, request.common_params);

    if let Some(default_provider_options_map) = defaults.provider_options_map
        && !default_provider_options_map.is_empty()
    {
        let mut merged = default_provider_options_map.clone();
        merged.merge_overrides(std::mem::take(&mut request.provider_options_map));
        request.provider_options_map = merged;
    }

    if request.http_config.is_none()
        && let Some(http_config) = defaults.http_config
    {
        request.http_config = Some(http_config.clone());
    }

    request.stream = stream;
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_common_params_fills_missing_fields_from_defaults() {
        let defaults = CommonParams {
            model: "default-model".to_string(),
            temperature: Some(0.2),
            max_tokens: Some(256),
            max_completion_tokens: Some(512),
            top_p: Some(0.9),
            top_k: Some(40.0),
            stop_sequences: Some(vec!["END".to_string()]),
            seed: Some(7),
            frequency_penalty: Some(0.3),
            presence_penalty: Some(0.4),
        };

        let merged = merge_common_params(
            &defaults,
            CommonParams {
                model: String::new(),
                temperature: Some(0.7),
                max_tokens: None,
                max_completion_tokens: None,
                top_p: None,
                top_k: Some(10.0),
                stop_sequences: None,
                seed: None,
                frequency_penalty: None,
                presence_penalty: Some(0.1),
            },
        );

        assert_eq!(merged.model, "default-model");
        assert_eq!(merged.temperature, Some(0.7));
        assert_eq!(merged.max_tokens, Some(256));
        assert_eq!(merged.max_completion_tokens, Some(512));
        assert_eq!(merged.top_p, Some(0.9));
        assert_eq!(merged.top_k, Some(10.0));
        assert_eq!(merged.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(merged.seed, Some(7));
        assert_eq!(merged.frequency_penalty, Some(0.3));
        assert_eq!(merged.presence_penalty, Some(0.1));
    }

    #[test]
    fn normalize_chat_request_merges_defaults_without_overriding_request_values() {
        let defaults = CommonParams {
            model: "default-model".to_string(),
            temperature: Some(0.2),
            max_tokens: Some(256),
            top_p: Some(0.9),
            ..Default::default()
        };
        let mut default_provider_options_map = ProviderOptionsMap::new();
        default_provider_options_map.insert(
            "anthropic",
            serde_json::json!({
                "thinking_mode": { "enabled": true },
                "structured_output_mode": "jsonTool"
            }),
        );
        let default_http_config = HttpConfig::default();

        let request = ChatRequest::builder()
            .messages(vec![])
            .common_params(CommonParams {
                model: "request-model".to_string(),
                temperature: Some(0.7),
                max_tokens: None,
                top_p: None,
                ..Default::default()
            })
            .provider_option(
                "anthropic",
                serde_json::json!({
                    "structured_output_mode": "outputFormat",
                    "disable_parallel_tool_use": true
                }),
            )
            .build();

        let normalized = normalize_chat_request(
            request,
            ChatRequestDefaults::new(&defaults)
                .with_provider_options_map(&default_provider_options_map)
                .with_http_config(&default_http_config),
            true,
        );

        assert!(normalized.stream);
        assert_eq!(normalized.common_params.model, "request-model");
        assert_eq!(normalized.common_params.temperature, Some(0.7));
        assert_eq!(normalized.common_params.max_tokens, Some(256));
        assert_eq!(normalized.common_params.top_p, Some(0.9));
        assert!(normalized.http_config.is_some());
        let anthropic = normalized
            .provider_options_map
            .get("anthropic")
            .expect("anthropic options");
        assert_eq!(
            anthropic.get("thinking_mode"),
            Some(&serde_json::json!({ "enabled": true }))
        );
        assert_eq!(
            anthropic.get("structured_output_mode"),
            Some(&serde_json::json!("outputFormat"))
        );
        assert_eq!(
            anthropic.get("disable_parallel_tool_use"),
            Some(&serde_json::json!(true))
        );
    }
}
