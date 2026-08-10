use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::NaiveDate;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use siumai_core::{
    ApiStability, CallOptions, CatalogError, EmbeddingLimits, EmbeddingModel, EmbeddingRequest,
    EmbeddingResponse, Error, ErrorContext, ErrorKind, GenericSupportClaim, InvalidId, Model,
    ModelCatalog, ModelDescriptor, ModelFamily, ModelId, ModelLifecycle, ModelOperation,
    ModelProfile, OfficialSource, ProfileError, ProfileId, ProtocolContractId, ProviderInstanceId,
    ProviderOptionContext, ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger,
    ProviderOptionOrigin, ProviderOptions, ProviderProfile, ProviderScope, ResponseMetadata,
    SupportScope, TypedProviderOptions, Usage, UsageValue, VerificationDate, VerificationEvidence,
    VerifiedFidelity, VerifiedSupportClaim,
};
use siumai_transport::{
    ProviderTransport, ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan,
    RequestTarget, TransportResponse,
};

use crate::native_error::{self, response_request_id};

pub const LEGACY_SINGAPORE_EMBEDDING_BASE_URL: &str = "https://dashscope-intl.aliyuncs.com/api/v1";
pub const EMBEDDING_SOURCE: &str =
    "https://www.alibabacloud.com/help/en/model-studio/text-embedding-synchronous-api";
pub const EMBEDDING_VERIFIED_ON: &str = "2026-08-06";
pub const EMBEDDING_API_MODE_ID: &str = "text-embedding";
pub const EMBEDDING_PROTOCOL_ID: &str = "alibaba-native";
pub const TEXT_EMBEDDING_V4: &str = "text-embedding-v4";
pub const TEXT_EMBEDDING_V3: &str = "text-embedding-v3";

const EMBEDDING_TARGET: &str = "services/embeddings/text-embedding/text-embedding";

pub(crate) fn support_profile(
    scope: &ProviderScope,
    verified_endpoint: bool,
) -> Result<ProviderProfile, AlibabaEmbeddingProfileError> {
    let support_scope = SupportScope::new(
        scope.provider_id().clone(),
        scope
            .platform()
            .cloned()
            .ok_or(AlibabaEmbeddingProfileError::MissingPlatform)?,
        ModelFamily::Embedding,
        scope
            .protocol()
            .cloned()
            .ok_or(AlibabaEmbeddingProfileError::MissingProtocol)?,
        scope
            .api_mode()
            .cloned()
            .ok_or(AlibabaEmbeddingProfileError::MissingApiMode)?,
    );
    let profile_id = ProfileId::new("alibaba-embedding")?;

    if !verified_endpoint {
        return Ok(ProviderProfile::generic(
            profile_id,
            GenericSupportClaim::new(support_scope, ApiStability::Experimental),
        ));
    }

    let verified_at = VerificationDate::new(
        NaiveDate::parse_from_str(EMBEDDING_VERIFIED_ON, "%Y-%m-%d")
            .map_err(|_| AlibabaEmbeddingProfileError::InvalidVerificationDate)?,
    );
    let evidence = VerificationEvidence::new(
        OfficialSource::new(EMBEDDING_SOURCE)?,
        verified_at,
        ProtocolContractId::new("alibaba-native-embedding-2026-08")?,
    );
    let claim = VerifiedSupportClaim::new(
        support_scope.clone(),
        VerifiedFidelity::Native,
        ApiStability::Stable,
        evidence.clone(),
    );
    let catalog_entries = [TEXT_EMBEDDING_V4, TEXT_EMBEDDING_V3]
        .into_iter()
        .map(|model| {
            Ok(ModelProfile::new(
                ModelId::new(model)?,
                support_scope.clone(),
                [ModelOperation::Embed],
                ModelLifecycle::Active,
                evidence.clone(),
            )?)
        })
        .collect::<Result<Vec<_>, AlibabaEmbeddingProfileError>>()?;
    let catalog = ModelCatalog::new(catalog_entries)?;

    Ok(ProviderProfile::verified(profile_id, vec![claim], catalog)?)
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AlibabaEmbeddingProfileError {
    #[error("Alibaba embedding scope is missing a platform")]
    MissingPlatform,
    #[error("Alibaba embedding scope is missing a protocol")]
    MissingProtocol,
    #[error("Alibaba embedding scope is missing an API mode")]
    MissingApiMode,
    #[error("invalid Alibaba embedding identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Alibaba embedding support evidence: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid Alibaba embedding model catalog: {0}")]
    Catalog(#[from] CatalogError),
    #[error("Alibaba embedding verification date is invalid")]
    InvalidVerificationDate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlibabaEmbeddingTextType {
    Query,
    Document,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlibabaEmbeddingOutputType {
    #[serde(rename = "dense")]
    Dense,
    #[serde(rename = "dense&sparse")]
    DenseAndSparse,
    /// Provider-native sparse-only output.
    ///
    /// The provider-neutral [`EmbeddingModel`] contract requires a dense vector for every input,
    /// so this mode is represented for explicit validation and rejected before submission.
    #[serde(rename = "sparse")]
    SparseOnly,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AlibabaEmbeddingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_type: Option<AlibabaEmbeddingTextType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_type: Option<AlibabaEmbeddingOutputType>,
    /// Task description used to specialize the embedding for a downstream scenario.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruct: Option<String>,
}

impl AlibabaEmbeddingOptions {
    pub const fn new() -> Self {
        Self {
            text_type: None,
            output_type: None,
            instruct: None,
        }
    }

    pub const fn with_text_type(mut self, text_type: AlibabaEmbeddingTextType) -> Self {
        self.text_type = Some(text_type);
        self
    }

    pub const fn with_output_type(mut self, output_type: AlibabaEmbeddingOutputType) -> Self {
        self.output_type = Some(output_type);
        self
    }

    pub fn with_instruct(mut self, instruct: impl Into<String>) -> Self {
        self.instruct = Some(instruct.into());
        self
    }
}

impl TypedProviderOptions for AlibabaEmbeddingOptions {
    const NAMESPACE: &'static str = "alibaba";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Embedding;
    const API_MODE: Option<&'static str> = Some(EMBEDDING_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.output_type == Some(AlibabaEmbeddingOutputType::SparseOnly) {
            return Err(ProviderOptionError::Rejected {
                path: "output_type".to_string(),
                reason: "sparse-only output cannot be represented by the provider-neutral embedding response"
                    .to_string(),
            });
        }
        Ok(())
    }
}

pub(crate) struct AlibabaEmbeddingRuntime {
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) transport: ProviderTransport,
    pub(crate) defaults: AlibabaEmbeddingOptions,
    pub(crate) replay_safety: ReplaySafety,
}

impl fmt::Debug for AlibabaEmbeddingRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaEmbeddingRuntime")
            .field("scope", &self.scope)
            .field("transport", &"shared")
            .field("replay_safety", &self.replay_safety)
            .finish()
    }
}

#[derive(Clone)]
pub struct AlibabaEmbeddingModel {
    pub(crate) runtime: Arc<AlibabaEmbeddingRuntime>,
    descriptor: ModelDescriptor,
}

impl AlibabaEmbeddingModel {
    pub(crate) fn new(runtime: Arc<AlibabaEmbeddingRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.scope.clone(),
            model,
            ModelFamily::Embedding,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::Embed),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl fmt::Debug for AlibabaEmbeddingModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaEmbeddingModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for AlibabaEmbeddingModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl EmbeddingModel for AlibabaEmbeddingModel {
    fn limits(&self) -> EmbeddingLimits {
        EmbeddingLimits {
            max_inputs: Some(max_inputs(self.model_id())),
            max_input_tokens: None,
        }
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        options: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let provider_options =
            embedding_options(&options, self.descriptor.scope(), &self.runtime.defaults)
                .map_err(option_error)
                .map_err(|error| self.contextualize(error))?;
        let dimensions = request.dimensions().map(|value| value.get());
        validate_output_type(self.model_id(), provider_options.output_type)
            .map_err(|error| self.contextualize(error))?;

        let wire = EmbeddingWireRequest {
            model: self.model_id().as_str(),
            input: EmbeddingWireInput {
                texts: request.inputs(),
            },
            parameters: EmbeddingWireParameters {
                text_type: provider_options.text_type,
                dimension: dimensions,
                output_type: provider_options.output_type,
                instruct: provider_options.instruct.as_deref(),
            },
        };
        let response = self
            .runtime
            .transport
            .execute(
                json_plan(&wire, self.runtime.replay_safety.clone())
                    .map_err(|error| self.contextualize(error))?,
                options,
            )
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(provider_status_error(response)));
        }
        let header_request_id = response_request_id(response.headers());
        let decoded =
            decode_response(response.body()).map_err(|error| self.contextualize(error))?;
        let ordered = ordered_embeddings(decoded.output.embeddings, request.inputs().len())
            .map_err(|error| self.contextualize(error))?;
        let result = EmbeddingResponse {
            embeddings: ordered.values,
            metadata: ResponseMetadata {
                response_id: None,
                request_id: decoded.request_id.or(header_request_id),
                model: Some(self.model_id().clone()),
            },
            usage: embedding_usage(decoded.usage.total_tokens),
            warnings: Vec::new(),
            provider: ordered.provider_metadata,
        };
        result
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        Ok(result)
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingWireRequest<'a> {
    model: &'a str,
    input: EmbeddingWireInput<'a>,
    #[serde(skip_serializing_if = "EmbeddingWireParameters::is_empty")]
    parameters: EmbeddingWireParameters<'a>,
}

#[derive(Debug, Serialize)]
struct EmbeddingWireInput<'a> {
    texts: &'a [String],
}

#[derive(Debug, Serialize)]
struct EmbeddingWireParameters<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    text_type: Option<AlibabaEmbeddingTextType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimension: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_type: Option<AlibabaEmbeddingOutputType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instruct: Option<&'a str>,
}

impl EmbeddingWireParameters<'_> {
    fn is_empty(&self) -> bool {
        self.text_type.is_none()
            && self.dimension.is_none()
            && self.output_type.is_none()
            && self.instruct.is_none()
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingWireResponse {
    #[serde(default)]
    request_id: Option<String>,
    output: EmbeddingWireOutput,
    #[serde(default)]
    usage: EmbeddingWireUsage,
}

#[derive(Debug, Deserialize)]
struct EmbeddingWireOutput {
    embeddings: Vec<EmbeddingWireItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingWireItem {
    embedding: Vec<f32>,
    #[serde(default)]
    sparse_embedding: Vec<SparseEmbeddingWireValue>,
    text_index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct SparseEmbeddingWireValue {
    index: usize,
    value: f32,
}

#[derive(Debug, Default, Deserialize)]
struct EmbeddingWireUsage {
    total_tokens: Option<u64>,
}

struct OrderedEmbeddings {
    values: Vec<Vec<f32>>,
    provider_metadata: BTreeMap<String, Value>,
}

fn ordered_embeddings(
    mut items: Vec<EmbeddingWireItem>,
    expected: usize,
) -> Result<OrderedEmbeddings, Error> {
    if items.len() != expected {
        return Err(Error::protocol_violation(
            "Alibaba embedding response count does not match the request",
        ));
    }
    items.sort_by_key(|item| item.text_index);
    if items
        .iter()
        .enumerate()
        .any(|(index, item)| item.text_index != index)
    {
        return Err(Error::protocol_violation(
            "Alibaba embedding response contains duplicate or missing text indexes",
        ));
    }
    let sparse = items
        .iter()
        .map(|item| serde_json::to_value(&item.sparse_embedding))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "Alibaba sparse embedding metadata could not be represented",
            )
            .with_source(source)
        })?;
    let embeddings = items.into_iter().map(|item| item.embedding).collect();
    let provider = if sparse.iter().any(|embedding| {
        embedding
            .as_array()
            .is_some_and(|values| !values.is_empty())
    }) {
        BTreeMap::from([(
            "alibaba".to_string(),
            serde_json::json!({"sparse_embeddings": sparse}),
        )])
    } else {
        BTreeMap::new()
    };
    Ok(OrderedEmbeddings {
        values: embeddings,
        provider_metadata: provider,
    })
}

fn embedding_usage(total_tokens: Option<u64>) -> Usage {
    let value = total_tokens.map_or(UsageValue::Unknown, UsageValue::Known);
    Usage::default()
        .with_input_tokens(value)
        .with_total_tokens(value)
}

fn json_plan(body: &impl Serialize, replay_safety: ReplaySafety) -> Result<RequestPlan, Error> {
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
        .map_err(request_build_error)?;
    RequestPlan::new(
        Method::POST,
        RequestTarget::new(EMBEDDING_TARGET).map_err(request_build_error)?,
    )
    .with_headers(headers)
    .with_body(RequestBody::json(body).map_err(request_build_error)?)
    .with_replay_safety(replay_safety)
    .map_err(request_build_error)
}

fn decode_response(body: &[u8]) -> Result<EmbeddingWireResponse, Error> {
    serde_json::from_slice(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Alibaba returned an invalid native embedding response",
        )
        .with_source(source)
    })
}

fn provider_status_error(response: TransportResponse) -> Error {
    native_error::provider_status_error(
        response,
        "Alibaba rejected the native embedding request",
        |status| match status {
            StatusCode::BAD_REQUEST => ErrorKind::InvalidInput,
            StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
            StatusCode::FORBIDDEN => ErrorKind::Authorization,
            StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
            _ => ErrorKind::Provider,
        },
    )
}

fn validate_output_type(
    _model: &ModelId,
    output_type: Option<AlibabaEmbeddingOutputType>,
) -> Result<(), Error> {
    if output_type == Some(AlibabaEmbeddingOutputType::SparseOnly) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Alibaba sparse-only output cannot be represented by the provider-neutral embedding response",
        ));
    }
    Ok(())
}

fn max_inputs(_model: &ModelId) -> usize {
    10
}

fn embedding_options(
    call: &CallOptions,
    scope: &ProviderScope,
    defaults: &AlibabaEmbeddingOptions,
) -> Result<AlibabaEmbeddingOptions, ProviderOptionError> {
    let defaults = ProviderOptions::typed(defaults)?;
    let layers = call.apply_provider_options(
        scope.provider_id(),
        ProviderOptionLayers::default().with_provider_default(defaults)?,
    )?;
    layers.merge_for(
        ProviderOptionContext::new(
            scope.provider_id(),
            ModelFamily::Embedding,
            scope.api_mode(),
        ),
        &EmbeddingOptionMerger,
    )
}

struct EmbeddingOptionMerger;

impl ProviderOptionMerger for EmbeddingOptionMerger {
    type Output = AlibabaEmbeddingOptions;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        let allowed = BTreeSet::from(["text_type", "output_type", "instruct"]);
        if let Some(field) = options
            .value()
            .keys()
            .find(|field| !allowed.contains(field.as_str()))
        {
            return Err(ProviderOptionError::Rejected {
                path: field.clone(),
                reason: "field is not valid for Alibaba embedding".to_string(),
            });
        }
        Ok(())
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = Map::new();
        for (_, options) in layers.in_precedence_order() {
            merged.extend(options.value().clone());
        }
        let options = serde_json::from_value::<AlibabaEmbeddingOptions>(Value::Object(merged))
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?;
        options.validate()?;
        Ok(options)
    }
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Alibaba embedding",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Alibaba embedding request violates the transport contract",
    )
    .with_source(source)
}
