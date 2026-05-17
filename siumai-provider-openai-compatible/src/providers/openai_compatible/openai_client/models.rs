use super::OpenAiCompatibleClient;
use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::ModelInfo;
use async_trait::async_trait;
use std::sync::Arc;

impl OpenAiCompatibleClient {
    /// List available models from the provider.
    async fn list_models_internal(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let spec = Arc::new(self.compat_spec());
        let ctx = self.build_context().await?;
        let url = spec.try_models_url(&ctx)?;
        let config = self.http_wiring(ctx).config(spec);

        let result =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await?;

        parse_models_response(&result.json, &self.config.provider_id, |model_id| {
            self.determine_model_capabilities(model_id)
        })
    }

    /// Determine model capabilities based on model ID.
    fn determine_model_capabilities(&self, model_id: &str) -> Vec<String> {
        let mut capabilities = vec!["chat".to_string()];

        if model_id.contains("embed") || model_id.contains("embedding") {
            capabilities.push("embedding".to_string());
        }

        if model_id.contains("rerank") || model_id.contains("bge-reranker") {
            capabilities.push("rerank".to_string());
        }

        if model_id.contains("flux")
            || model_id.contains("stable-diffusion")
            || model_id.contains("kolors")
        {
            capabilities.push("image_generation".to_string());
        }

        if self
            .config
            .adapter
            .get_model_config(model_id)
            .supports_thinking
        {
            capabilities.push("thinking".to_string());
        }

        capabilities
    }

    /// Get detailed information about a specific model.
    async fn get_model_internal(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        // Prefer the dedicated retrieve endpoint, then fall back to list and synthetic metadata.
        let spec = Arc::new(self.compat_spec());
        let ctx = self.build_context().await?;
        let url = spec.try_model_url(&model_id, &ctx)?;
        let config = self.http_wiring(ctx).config(spec);

        let direct =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await;

        match direct {
            Ok(result) => {
                if let Some(model_info) =
                    parse_model_like_response(&result.json, &self.config.provider_id, |model_id| {
                        self.determine_model_capabilities(model_id)
                    })
                {
                    return Ok(model_info);
                }
            }
            Err(LlmError::ApiError { code: 404, .. } | LlmError::NotFound(_)) => {
                // Fall through to list+basic.
            }
            Err(e) => {
                // If the provider advertises ModelListingCapability but doesn't support
                // the retrieve endpoint, it may still support listing. For other errors
                // (auth/rate limit/etc.), do not mask the failure.
                return Err(e);
            }
        }

        let models = self.list_models_internal().await?;
        if let Some(model) = models.into_iter().find(|m| m.id == model_id) {
            return Ok(model);
        }

        Ok(model_info_from_json(
            &serde_json::json!({ "id": model_id }),
            &self.config.provider_id,
            |model_id| self.determine_model_capabilities(model_id),
            Some(&self.config.provider_id),
        )
        .expect("synthetic model id should produce ModelInfo"))
    }
}

fn parse_models_response(
    json: &serde_json::Value,
    provider_id: &str,
    capabilities_for: impl Fn(&str) -> Vec<String>,
) -> Result<Vec<ModelInfo>, LlmError> {
    let models = json
        .get("data")
        .and_then(|data| data.as_array())
        .ok_or_else(|| LlmError::ParseError("Invalid models response format".to_string()))?;

    Ok(models
        .iter()
        .filter_map(|model| model_info_from_json(model, provider_id, &capabilities_for, None))
        .collect())
}

fn parse_model_like_response(
    json: &serde_json::Value,
    provider_id: &str,
    capabilities_for: impl Fn(&str) -> Vec<String>,
) -> Option<ModelInfo> {
    model_info_from_json(json, provider_id, &capabilities_for, None).or_else(|| {
        json.get("data")
            .and_then(|data| data.as_array())
            .and_then(|items| items.first())
            .and_then(|model| model_info_from_json(model, provider_id, &capabilities_for, None))
    })
}

fn model_info_from_json(
    json: &serde_json::Value,
    provider_id: &str,
    capabilities_for: impl Fn(&str) -> Vec<String>,
    fallback_description_provider: Option<&str>,
) -> Option<ModelInfo> {
    let model_id = json.get("id").and_then(|id| id.as_str())?;
    let description = json
        .get("description")
        .and_then(|description| description.as_str())
        .map(|description| description.to_string())
        .or_else(|| {
            fallback_description_provider.map(|provider| format!("{provider} model: {model_id}"))
        });

    Some(ModelInfo {
        id: model_id.to_string(),
        name: Some(model_id.to_string()),
        description,
        owned_by: json
            .get("owned_by")
            .and_then(|owned_by| owned_by.as_str())
            .unwrap_or(provider_id)
            .to_string(),
        created: json.get("created").and_then(|created| created.as_u64()),
        capabilities: capabilities_for(model_id),
        context_window: None,
        max_output_tokens: None,
        input_cost_per_token: None,
        output_cost_per_token: None,
    })
}

#[async_trait]
impl ModelListingCapability for OpenAiCompatibleClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.list_models_internal().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.get_model_internal(model_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::LlmClient;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportGetRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use crate::providers::openai_compatible::OpenAiCompatibleConfig;
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct GetResponse {
        status: u16,
        body: serde_json::Value,
    }

    #[derive(Clone, Default)]
    struct GetTransport {
        responses: Arc<Mutex<Vec<GetResponse>>>,
        requests: Arc<Mutex<Vec<HttpTransportGetRequest>>>,
    }

    impl GetTransport {
        fn new(responses: Vec<GetResponse>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses)),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn take_requests(&self) -> Vec<HttpTransportGetRequest> {
            std::mem::take(&mut *self.requests.lock().unwrap())
        }
    }

    #[async_trait]
    impl HttpTransport for GetTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "json unsupported in model listing tests".to_string(),
            ))
        }

        async fn execute_get(
            &self,
            request: HttpTransportGetRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.requests.lock().unwrap().push(request);
            let response = self.responses.lock().unwrap().remove(0);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: response.status,
                headers,
                body: serde_json::to_vec(&response.body).expect("response body"),
            })
        }
    }

    fn get_response(status: u16, body: serde_json::Value) -> GetResponse {
        GetResponse { status, body }
    }

    fn make_model_listing_adapter(
        id: &str,
        capabilities: Vec<&str>,
        supports_reasoning: bool,
    ) -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: id.to_string(),
            name: format!("{id} provider"),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: capabilities.into_iter().map(str::to_string).collect(),
            default_model: Some("chat-default".to_string()),
            supports_reasoning,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    #[tokio::test]
    async fn list_models_uses_provider_spec_url_and_parses_capabilities() {
        let transport = GetTransport::new(vec![get_response(
            200,
            serde_json::json!({
                "object": "list",
                "data": [
                    {
                        "id": "text-embedding-3-small",
                        "owned_by": "provider-owner",
                        "created": 123,
                        "description": "embedding model"
                    },
                    { "id": "jina-reranker-m0" },
                    { "id": "flux-1-dev" },
                    { "object": "model-without-id" }
                ]
            }),
        )]);

        let cfg = OpenAiCompatibleConfig::new(
            "compat-models",
            "test-key",
            "https://api.test.com/v1",
            make_model_listing_adapter("compat-models", vec!["chat"], false),
        )
        .with_model("chat-default")
        .with_query_param("api-version", "2026-01-01")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let models = ModelListingCapability::list_models(&client)
            .await
            .expect("models");

        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "text-embedding-3-small");
        assert_eq!(models[0].owned_by, "provider-owner");
        assert_eq!(models[0].created, Some(123));
        assert_eq!(models[0].description.as_deref(), Some("embedding model"));
        assert!(models[0].capabilities.contains(&"chat".to_string()));
        assert!(models[0].capabilities.contains(&"embedding".to_string()));
        assert!(models[1].capabilities.contains(&"rerank".to_string()));
        assert!(
            models[2]
                .capabilities
                .contains(&"image_generation".to_string())
        );

        let requests = transport.take_requests();
        assert_eq!(
            requests[0].url,
            "https://api.test.com/v1/models?api-version=2026-01-01"
        );
    }

    #[tokio::test]
    async fn get_model_prefers_direct_model_endpoint() {
        let transport = GetTransport::new(vec![get_response(
            200,
            serde_json::json!({
                "id": "deepseek-reasoner",
                "owned_by": "compat-models",
                "description": "direct model"
            }),
        )]);

        let cfg = OpenAiCompatibleConfig::new(
            "compat-models",
            "test-key",
            "https://api.test.com/v1",
            make_model_listing_adapter("compat-models", vec!["chat"], true),
        )
        .with_model("chat-default")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let model = ModelListingCapability::get_model(&client, "deepseek-reasoner".to_string())
            .await
            .expect("model");

        assert_eq!(model.id, "deepseek-reasoner");
        assert_eq!(model.description.as_deref(), Some("direct model"));
        assert!(model.capabilities.contains(&"thinking".to_string()));

        let requests = transport.take_requests();
        assert_eq!(
            requests[0].url,
            "https://api.test.com/v1/models/deepseek-reasoner"
        );
    }

    #[tokio::test]
    async fn get_model_falls_back_to_list_then_synthetic_model_info() {
        let transport = GetTransport::new(vec![
            get_response(
                404,
                serde_json::json!({
                    "error": {
                        "message": "missing",
                        "type": "not_found_error",
                        "code": "model_not_found"
                    }
                }),
            ),
            get_response(
                200,
                serde_json::json!({
                    "data": [
                        { "id": "another-model" }
                    ]
                }),
            ),
        ]);

        let cfg = OpenAiCompatibleConfig::new(
            "compat-models",
            "test-key",
            "https://api.test.com/v1",
            make_model_listing_adapter("compat-models", vec!["chat"], false),
        )
        .with_model("chat-default")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let model = ModelListingCapability::get_model(&client, "missing-model".to_string())
            .await
            .expect("synthetic model");

        assert_eq!(model.id, "missing-model");
        assert_eq!(
            model.description.as_deref(),
            Some("compat-models model: missing-model")
        );
        assert_eq!(model.owned_by, "compat-models");

        let requests = transport.take_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].url,
            "https://api.test.com/v1/models/missing-model"
        );
        assert_eq!(requests[1].url, "https://api.test.com/v1/models");
    }

    #[tokio::test]
    async fn openai_compatible_client_exposes_model_listing_capability_view() {
        let cfg = OpenAiCompatibleConfig::new(
            "compat-models",
            "test-key",
            "https://api.test.com/v1",
            make_model_listing_adapter("compat-models", vec!["chat"], false),
        )
        .with_model("chat-default");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        assert!(client.as_model_listing_capability().is_some());
    }

    #[test]
    fn model_listing_logic_stays_out_of_monolithic_client_module() {
        let source = include_str!("../openai_client.rs");
        for forbidden in [
            "fn list_models_internal(",
            "fn get_model_internal(",
            "fn determine_model_capabilities(",
            "impl ModelListingCapability for OpenAiCompatibleClient",
            "try_models_url(",
            "try_model_url(",
            "execute_get_request(&config",
        ] {
            assert!(
                !source.contains(forbidden),
                "OpenAI-compatible model listing logic should live in openai_client/models.rs"
            );
        }
    }
}
