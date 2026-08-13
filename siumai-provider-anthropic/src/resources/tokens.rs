use std::sync::Arc;

use http::Method;
use serde::{Deserialize, Serialize};
use siumai_anthropic_compatible::MessagesRequestPolicy;
use siumai_core::{CallOptions, Error, ErrorKind, LanguageRequest, ModelId};
use siumai_protocol_anthropic::messages::{
    MESSAGES_COUNT_TOKENS_TARGET, MessagesEncodingRules,
    encode_count_tokens_request_for_scope_with_resolver_and_rules,
};
use siumai_transport::{ReplaySafety, RequestBody};

use crate::AnthropicTokenCountOptions;
use crate::request_policy::AnthropicRequestPolicy;

use super::NativeRuntime;
use super::common::{execute_json, target};

/// Anthropic token-counting response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicTokenCount {
    pub input_tokens: u64,
}

/// Shared, lightweight Token Counting API handle.
#[derive(Clone)]
pub struct AnthropicTokens {
    runtime: Arc<NativeRuntime>,
}

impl AnthropicTokens {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn count(
        &self,
        model: impl Into<String>,
        request: LanguageRequest,
        options: AnthropicTokenCountOptions,
    ) -> Result<AnthropicTokenCount, Error> {
        self.count_with_options(model, request, options, CallOptions::default())
            .await
    }

    pub async fn count_with_options(
        &self,
        model: impl Into<String>,
        request: LanguageRequest,
        options: AnthropicTokenCountOptions,
        call_options: CallOptions,
    ) -> Result<AnthropicTokenCount, Error> {
        let model = ModelId::new(model.into()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic model identifier is invalid",
            )
            .with_source(source)
        })?;
        let mut engine_options = options.to_engine();
        let requirements = AnthropicRequestPolicy.prepare(&model, &request, &mut engine_options)?;
        let protocol_options = options.to_protocol();
        let body = encode_count_tokens_request_for_scope_with_resolver_and_rules(
            &self.runtime.scope,
            &model,
            &request,
            &protocol_options,
            self.runtime.annotation_resolver.as_ref(),
            &MessagesEncodingRules::native(),
        )
        .map_err(Error::from)?;
        let beta_features = requirements.beta_features().collect::<Vec<_>>();
        execute_json(
            &self.runtime,
            Method::POST,
            target(MESSAGES_COUNT_TOKENS_TARGET)?,
            RequestBody::json(&body).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Anthropic token-count request could not be encoded",
                )
                .with_source(source)
            })?,
            ReplaySafety::SemanticallyIdempotent,
            &beta_features,
            call_options,
        )
        .await
    }
}

impl std::fmt::Debug for AnthropicTokens {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicTokens")
            .field("runtime", &"shared")
            .finish()
    }
}
