use std::sync::Arc;

use http::Method;
use serde::{Deserialize, Serialize};
use siumai_core::{CallOptions, Error, ErrorKind, LanguageRequest, ModelId};
use siumai_protocol_anthropic::messages::encode_request_with_resolver;
use siumai_transport::{ReplaySafety, RequestBody};

use crate::AnthropicMessagesOptions;

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
        options: AnthropicMessagesOptions,
    ) -> Result<AnthropicTokenCount, Error> {
        self.count_with_options(model, request, options, CallOptions::default())
            .await
    }

    pub async fn count_with_options(
        &self,
        model: impl Into<String>,
        request: LanguageRequest,
        options: AnthropicMessagesOptions,
        call_options: CallOptions,
    ) -> Result<AnthropicTokenCount, Error> {
        let model = ModelId::new(model.into()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic model identifier is invalid",
            )
            .with_source(source)
        })?;
        let protocol_options = options.to_protocol(false);
        let mut body = encode_request_with_resolver(
            &model,
            &request,
            &protocol_options,
            self.runtime.annotation_resolver.as_ref(),
        )
        .map_err(Error::from)?;
        if let Some(body) = body.as_object_mut() {
            body.remove("stream");
        }
        execute_json(
            &self.runtime,
            Method::POST,
            target("messages/count_tokens")?,
            RequestBody::json(&body).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Anthropic token-count request could not be encoded",
                )
                .with_source(source)
            })?,
            ReplaySafety::SemanticallyIdempotent,
            &[],
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
