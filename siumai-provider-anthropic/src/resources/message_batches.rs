use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytes::Bytes;
use http::Method;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind, LanguageRequest, ModelId};
use siumai_protocol_anthropic::messages::encode_request_with_resolver;
use siumai_transport::{ReplaySafety, RequestBody};

use crate::AnthropicMessagesOptions;

use super::NativeRuntime;
use super::common::{execute_download, execute_json, target, validate_resource_id};

const MAX_BATCH_REQUESTS: usize = 100_000;

/// One language request submitted under a caller-stable batch identifier.
#[derive(Debug, Clone)]
pub struct AnthropicBatchItem {
    custom_id: String,
    model: ModelId,
    request: LanguageRequest,
    options: AnthropicMessagesOptions,
}

impl AnthropicBatchItem {
    pub fn new(
        custom_id: impl Into<String>,
        model: impl Into<String>,
        request: LanguageRequest,
    ) -> Result<Self, Error> {
        let custom_id = custom_id.into();
        validate_custom_id(&custom_id)?;
        let model = ModelId::new(model.into()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Anthropic model identifier is invalid",
            )
            .with_source(source)
        })?;
        Ok(Self {
            custom_id,
            model,
            request,
            options: AnthropicMessagesOptions::default(),
        })
    }

    pub fn with_options(mut self, options: AnthropicMessagesOptions) -> Self {
        self.options = options;
        self
    }

    pub fn custom_id(&self) -> &str {
        &self.custom_id
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn request(&self) -> &LanguageRequest {
        &self.request
    }

    pub fn options(&self) -> &AnthropicMessagesOptions {
        &self.options
    }

    fn validate_for_batch(&self) -> Result<(), Error> {
        if self.options.fallbacks().is_some() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic Message Batches do not support server-side fallbacks",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct AnthropicBatchRequest {
    requests: Vec<AnthropicBatchItem>,
}

impl AnthropicBatchRequest {
    pub fn new(requests: Vec<AnthropicBatchItem>) -> Result<Self, Error> {
        if requests.is_empty() || requests.len() > MAX_BATCH_REQUESTS {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic batch request count is outside the supported bounds",
            ));
        }
        let unique = requests
            .iter()
            .map(|request| request.custom_id.as_str())
            .collect::<BTreeSet<_>>();
        if unique.len() != requests.len() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic batch custom identifiers must be unique",
            ));
        }
        Ok(Self { requests })
    }

    pub fn requests(&self) -> &[AnthropicBatchItem] {
        &self.requests
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnthropicBatchListQuery {
    pub before_id: Option<String>,
    pub after_id: Option<String>,
    pub limit: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessageBatch {
    pub id: String,
    #[serde(rename = "type", default)]
    pub object_type: Option<String>,
    #[serde(default)]
    pub processing_status: Option<String>,
    #[serde(default)]
    pub request_counts: Option<AnthropicBatchRequestCounts>,
    #[serde(default)]
    pub ended_at: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub cancel_initiated_at: Option<String>,
    #[serde(default)]
    pub results_url: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicBatchRequestCounts {
    #[serde(default)]
    pub processing: u64,
    #[serde(default)]
    pub succeeded: u64,
    #[serde(default)]
    pub errored: u64,
    #[serde(default)]
    pub canceled: u64,
    #[serde(default)]
    pub expired: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AnthropicBatchList {
    #[serde(default)]
    pub data: Vec<AnthropicMessageBatch>,
    #[serde(default)]
    pub first_id: Option<String>,
    #[serde(default)]
    pub last_id: Option<String>,
    #[serde(default)]
    pub has_more: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AnthropicBatchDeleteResult {
    pub id: String,
    #[serde(default)]
    pub deleted: bool,
    #[serde(rename = "type", default)]
    pub object_type: Option<String>,
}

/// Shared, lightweight Message Batches API handle.
#[derive(Clone)]
pub struct AnthropicMessageBatches {
    runtime: Arc<NativeRuntime>,
}

impl AnthropicMessageBatches {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn create(
        &self,
        request: AnthropicBatchRequest,
    ) -> Result<AnthropicMessageBatch, Error> {
        self.create_with_options(request, CallOptions::default())
            .await
    }

    pub async fn create_with_options(
        &self,
        request: AnthropicBatchRequest,
        call_options: CallOptions,
    ) -> Result<AnthropicMessageBatch, Error> {
        let requests = request
            .requests
            .into_iter()
            .map(|item| self.encode_item(item))
            .collect::<Result<Vec<_>, _>>()?;
        execute_json(
            &self.runtime,
            Method::POST,
            target("messages/batches")?,
            RequestBody::json(&BatchCreateWire { requests }).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Anthropic message batch could not be encoded",
                )
                .with_source(source)
            })?,
            ReplaySafety::Never,
            &[],
            call_options,
        )
        .await
    }

    pub async fn retrieve(&self, batch_id: &str) -> Result<AnthropicMessageBatch, Error> {
        self.retrieve_with_options(batch_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        batch_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicMessageBatch, Error> {
        validate_resource_id(batch_id)?;
        execute_json(
            &self.runtime,
            Method::GET,
            target(format!("messages/batches/{batch_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[],
            options,
        )
        .await
    }

    pub async fn list(&self, query: AnthropicBatchListQuery) -> Result<AnthropicBatchList, Error> {
        self.list_with_options(query, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        query: AnthropicBatchListQuery,
        options: CallOptions,
    ) -> Result<AnthropicBatchList, Error> {
        execute_json(
            &self.runtime,
            Method::GET,
            target(list_target(&query)?)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[],
            options,
        )
        .await
    }

    pub async fn cancel(&self, batch_id: &str) -> Result<AnthropicMessageBatch, Error> {
        self.cancel_with_options(batch_id, CallOptions::default())
            .await
    }

    pub async fn cancel_with_options(
        &self,
        batch_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicMessageBatch, Error> {
        validate_resource_id(batch_id)?;
        execute_json(
            &self.runtime,
            Method::POST,
            target(format!("messages/batches/{batch_id}/cancel"))?,
            RequestBody::json(&serde_json::json!({})).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Anthropic batch cancel body is invalid",
                )
                .with_source(source)
            })?,
            ReplaySafety::SemanticallyIdempotent,
            &[],
            options,
        )
        .await
    }

    pub async fn delete(&self, batch_id: &str) -> Result<AnthropicBatchDeleteResult, Error> {
        self.delete_with_options(batch_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        batch_id: &str,
        options: CallOptions,
    ) -> Result<AnthropicBatchDeleteResult, Error> {
        validate_resource_id(batch_id)?;
        execute_json(
            &self.runtime,
            Method::DELETE,
            target(format!("messages/batches/{batch_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            &[],
            options,
        )
        .await
    }

    pub async fn results(&self, batch_id: &str) -> Result<Bytes, Error> {
        self.results_with_options(batch_id, CallOptions::default())
            .await
    }

    pub async fn results_with_options(
        &self,
        batch_id: &str,
        options: CallOptions,
    ) -> Result<Bytes, Error> {
        validate_resource_id(batch_id)?;
        execute_download(
            &self.runtime,
            target(format!("messages/batches/{batch_id}/results"))?,
            &[],
            "application/x-jsonlines",
            options,
        )
        .await
    }

    fn encode_item(&self, item: AnthropicBatchItem) -> Result<BatchItemWire, Error> {
        item.validate_for_batch()?;
        let protocol_options = item.options.to_protocol(false);
        let mut params = encode_request_with_resolver(
            &item.model,
            &item.request,
            &protocol_options,
            self.runtime.annotation_resolver.as_ref(),
        )
        .map_err(Error::from)?;
        if let Some(params) = params.as_object_mut() {
            params.remove("stream");
        }
        Ok(BatchItemWire {
            custom_id: item.custom_id,
            params,
        })
    }
}

impl std::fmt::Debug for AnthropicMessageBatches {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicMessageBatches")
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Serialize)]
struct BatchCreateWire {
    requests: Vec<BatchItemWire>,
}

#[derive(Serialize)]
struct BatchItemWire {
    custom_id: String,
    params: Value,
}

fn validate_custom_id(id: &str) -> Result<(), Error> {
    if id.is_empty()
        || id.len() > 256
        || id.chars().any(char::is_control)
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Anthropic batch custom identifier is invalid",
        ));
    }
    Ok(())
}

fn list_target(query: &AnthropicBatchListQuery) -> Result<String, Error> {
    let mut pairs = Vec::new();
    if let Some(before) = &query.before_id {
        validate_resource_id(before)?;
        pairs.push(format!("before_id={before}"));
    }
    if let Some(after) = &query.after_id {
        validate_resource_id(after)?;
        pairs.push(format!("after_id={after}"));
    }
    if let Some(limit) = query.limit {
        if limit == 0 || limit > 1_000 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Anthropic list limit must be between 1 and 1000",
            ));
        }
        pairs.push(format!("limit={limit}"));
    }
    Ok(if pairs.is_empty() {
        "messages/batches".to_string()
    } else {
        format!("messages/batches?{}", pairs.join("&"))
    })
}

#[cfg(test)]
mod tests {
    use siumai_protocol_anthropic::messages::ServerFallbacks;

    use super::*;

    #[test]
    fn batch_item_rejects_server_side_fallbacks_before_encoding() {
        let item = AnthropicBatchItem::new(
            "request-with-fallbacks",
            "claude-sonnet-5",
            LanguageRequest::new(Vec::new()),
        )
        .expect("valid batch item")
        .with_options(AnthropicMessagesOptions::new().with_fallbacks(ServerFallbacks::Default));

        let error = item
            .validate_for_batch()
            .expect_err("message batches must reject server-side fallbacks");

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(
            error.message(),
            "Anthropic Message Batches do not support server-side fallbacks"
        );
    }
}
