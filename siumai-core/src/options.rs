//! Request-scoped controls and provider-owned option serialization.

use std::fmt;
use std::time::Instant;

use serde::Serialize;
use serde_json::{Map, Value};
use thiserror::Error;
use tokio_util::sync::{CancellationToken, WaitForCancellationFuture};

use crate::provider::ProviderId;

/// Cloneable request cancellation shared by model, transport, and runtime layers.
#[derive(Clone, Default)]
pub struct Cancellation {
    token: CancellationToken,
}

impl fmt::Debug for Cancellation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Cancellation")
            .field("is_cancelled", &self.is_cancelled())
            .finish()
    }
}

impl Cancellation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.token.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    pub fn cancelled(&self) -> WaitForCancellationFuture<'_> {
        self.token.cancelled()
    }

    pub fn child(&self) -> Self {
        Self {
            token: self.token.child_token(),
        }
    }

    pub(crate) fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

/// Caller intent; transport still requires operation-specific replay proof.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryIntent {
    /// Apply the configured provider policy only when replay safety is proven.
    #[default]
    ProviderPolicy,
    /// Do not retry this logical call.
    Never,
}

/// The source of one provider-option layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderOptionOrigin {
    ProviderDefault,
    RouteDefault,
    ModelDefault,
    RuntimeStep,
    Call,
    RawOverride,
}

/// Provider option validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ProviderOptionError {
    #[error("provider options namespace `{0}` is invalid")]
    InvalidNamespace(String),
    #[error("provider options for `{namespace}` must serialize to an object")]
    ExpectedObject { namespace: String },
    #[error("provider option `{path}` is protected and cannot be set per call")]
    ProtectedField { path: String },
    #[error("provider options namespace `{actual}` does not match `{expected}`")]
    NamespaceMismatch { expected: String, actual: String },
    #[error("raw provider options may only occupy the explicit raw-override layer")]
    RawLayerMismatch,
    #[error("typed provider options are required for this precedence layer")]
    TypedLayerRequired,
    #[error("provider option layer {origin:?} was configured more than once")]
    DuplicateLayer { origin: ProviderOptionOrigin },
    #[error("provider options exceed the {maximum}-byte limit")]
    TooLarge { maximum: usize },
    #[error("provider options exceed the maximum JSON nesting depth of {maximum}")]
    TooDeep { maximum: usize },
    #[error("provider options exceed the {maximum}-field limit")]
    TooManyFields { maximum: usize },
    #[error("provider rejected option `{path}`: {reason}")]
    Rejected { path: String, reason: String },
    #[error("failed to serialize provider options: {0}")]
    Serialization(String),
}

/// Implemented by typed provider option structs in provider crates.
///
/// Implementations are ergonomic codecs, not a trust boundary. After erasure,
/// every layer is still checked by [`ProviderOptionMerger::validate_layer`].
pub trait TypedProviderOptions: Serialize {
    const NAMESPACE: &'static str;

    /// Validate provider-specific relationships before serialization.
    fn validate(&self) -> Result<(), ProviderOptionError> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderOptionKind {
    Typed,
    Raw,
}

/// One opaque but validated provider-option layer.
#[derive(Clone, PartialEq)]
pub struct ProviderOptions {
    namespace: ProviderId,
    value: Map<String, Value>,
    kind: ProviderOptionKind,
}

impl fmt::Debug for ProviderOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderOptions")
            .field("namespace", &self.namespace)
            .field("kind", &self.kind)
            .field("fields", &self.value.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ProviderOptions {
    /// Serialize a provider-owned typed option struct.
    pub fn typed<T: TypedProviderOptions>(value: &T) -> Result<Self, ProviderOptionError> {
        value.validate()?;
        let namespace = ProviderId::new(T::NAMESPACE)
            .map_err(|_| ProviderOptionError::InvalidNamespace(T::NAMESPACE.to_string()))?;
        let value = serde_json::to_value(value)
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?;
        Self::from_value(namespace, value, ProviderOptionKind::Typed)
    }

    /// Build the explicit checked raw escape hatch.
    pub fn checked_raw(namespace: ProviderId, value: Value) -> Result<Self, ProviderOptionError> {
        Self::from_value(namespace, value, ProviderOptionKind::Raw)
    }

    fn from_value(
        namespace: ProviderId,
        value: Value,
        kind: ProviderOptionKind,
    ) -> Result<Self, ProviderOptionError> {
        let Value::Object(value) = value else {
            return Err(ProviderOptionError::ExpectedObject {
                namespace: namespace.to_string(),
            });
        };
        validate_option_shape(&value)?;
        reject_protected_fields(&value, "")?;
        Ok(Self {
            namespace,
            value,
            kind,
        })
    }

    pub fn namespace(&self) -> &ProviderId {
        &self.namespace
    }

    pub fn value(&self) -> &Map<String, Value> {
        &self.value
    }

    pub fn is_raw(&self) -> bool {
        self.kind == ProviderOptionKind::Raw
    }
}

/// Explicit precedence stack. The provider owns the merge algorithm.
#[derive(Debug, Clone, Default)]
pub struct ProviderOptionLayers {
    provider_default: Option<ProviderOptions>,
    route_default: Option<ProviderOptions>,
    model_default: Option<ProviderOptions>,
    runtime_step: Option<ProviderOptions>,
    call: Option<ProviderOptions>,
    raw_override: Option<ProviderOptions>,
}

impl ProviderOptionLayers {
    pub fn with_provider_default(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(
            &self.provider_default,
            ProviderOptionOrigin::ProviderDefault,
        )?;
        self.provider_default = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_route_default(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.route_default, ProviderOptionOrigin::RouteDefault)?;
        self.route_default = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_model_default(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.model_default, ProviderOptionOrigin::ModelDefault)?;
        self.model_default = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_runtime_step(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.runtime_step, ProviderOptionOrigin::RuntimeStep)?;
        self.runtime_step = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_call(mut self, options: ProviderOptions) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.call, ProviderOptionOrigin::Call)?;
        self.call = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_raw_override(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.raw_override, ProviderOptionOrigin::RawOverride)?;
        if !options.is_raw() {
            return Err(ProviderOptionError::RawLayerMismatch);
        }
        self.raw_override = Some(options);
        Ok(self)
    }

    pub fn in_precedence_order(
        &self,
    ) -> impl Iterator<Item = (ProviderOptionOrigin, &ProviderOptions)> {
        [
            (
                ProviderOptionOrigin::ProviderDefault,
                self.provider_default.as_ref(),
            ),
            (
                ProviderOptionOrigin::RouteDefault,
                self.route_default.as_ref(),
            ),
            (
                ProviderOptionOrigin::ModelDefault,
                self.model_default.as_ref(),
            ),
            (
                ProviderOptionOrigin::RuntimeStep,
                self.runtime_step.as_ref(),
            ),
            (ProviderOptionOrigin::Call, self.call.as_ref()),
            (
                ProviderOptionOrigin::RawOverride,
                self.raw_override.as_ref(),
            ),
        ]
        .into_iter()
        .filter_map(|(origin, options)| options.map(|options| (origin, options)))
    }

    /// Validate namespace ownership and invoke the provider-owned raw schema
    /// and merge policy. Raw fields never reach transport or authentication
    /// configuration through this contract.
    pub fn merge_for<M: ProviderOptionMerger>(
        &self,
        expected: &ProviderId,
        merger: &M,
    ) -> Result<M::Output, ProviderOptionError> {
        for (origin, options) in self.in_precedence_order() {
            if options.namespace() != expected {
                return Err(ProviderOptionError::NamespaceMismatch {
                    expected: expected.to_string(),
                    actual: options.namespace().to_string(),
                });
            }
            merger.validate_layer(origin, options)?;
        }
        merger.merge(self)
    }
}

fn require_typed(options: ProviderOptions) -> Result<ProviderOptions, ProviderOptionError> {
    if options.is_raw() {
        Err(ProviderOptionError::TypedLayerRequired)
    } else {
        Ok(options)
    }
}

fn ensure_empty(
    slot: &Option<ProviderOptions>,
    origin: ProviderOptionOrigin,
) -> Result<(), ProviderOptionError> {
    if slot.is_some() {
        Err(ProviderOptionError::DuplicateLayer { origin })
    } else {
        Ok(())
    }
}

/// Provider-owned merge and rejection policy.
pub trait ProviderOptionMerger: Send + Sync {
    type Output;

    /// Validate every erased typed or raw layer against the selected provider's
    /// request-body schema. A public Rust trait implementation is not treated as
    /// proof that typed fields are provider-owned.
    fn validate_layer(
        &self,
        origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError>;

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError>;
}

#[derive(Debug, Clone)]
struct ProviderOptionEntry {
    origin: ProviderOptionOrigin,
    options: ProviderOptions,
}

/// Controls shared by all six stable model families.
#[derive(Clone, Default)]
pub struct CallOptions {
    deadline: Option<Instant>,
    cancellation: Cancellation,
    retry: RetryIntent,
    provider_options: Vec<ProviderOptionEntry>,
}

impl fmt::Debug for CallOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CallOptions")
            .field("deadline", &self.deadline)
            .field("cancellation", &self.cancellation)
            .field("retry", &self.retry)
            .field(
                "provider_option_namespaces",
                &self
                    .provider_options
                    .iter()
                    .map(|entry| entry.options.namespace().as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl CallOptions {
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn cancellation(&self) -> &Cancellation {
        &self.cancellation
    }

    pub fn retry(&self) -> RetryIntent {
        self.retry
    }

    pub fn provider_options(&self) -> impl ExactSizeIterator<Item = &ProviderOptions> {
        self.provider_options.iter().map(|entry| &entry.options)
    }

    pub fn has_provider_options(&self) -> bool {
        !self.provider_options.is_empty()
    }

    /// Add call-scoped typed and raw options to an existing precedence stack.
    /// Foreign namespaces and duplicate call/raw layers are rejected.
    pub fn apply_provider_options(
        &self,
        expected: &ProviderId,
        mut layers: ProviderOptionLayers,
    ) -> Result<ProviderOptionLayers, ProviderOptionError> {
        for entry in &self.provider_options {
            let options = &entry.options;
            if options.namespace() != expected {
                return Err(ProviderOptionError::NamespaceMismatch {
                    expected: expected.to_string(),
                    actual: options.namespace().to_string(),
                });
            }
            layers = match entry.origin {
                ProviderOptionOrigin::ProviderDefault => {
                    layers.with_provider_default(options.clone())?
                }
                ProviderOptionOrigin::RouteDefault => layers.with_route_default(options.clone())?,
                ProviderOptionOrigin::ModelDefault => layers.with_model_default(options.clone())?,
                ProviderOptionOrigin::RuntimeStep => layers.with_runtime_step(options.clone())?,
                ProviderOptionOrigin::Call => layers.with_call(options.clone())?,
                ProviderOptionOrigin::RawOverride => layers.with_raw_override(options.clone())?,
            };
        }
        Ok(layers)
    }

    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_cancellation(mut self, cancellation: Cancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    pub fn without_retry(mut self) -> Self {
        self.retry = RetryIntent::Never;
        self
    }

    pub fn with_provider_options(mut self, options: ProviderOptions) -> Self {
        let origin = if options.is_raw() {
            ProviderOptionOrigin::RawOverride
        } else {
            ProviderOptionOrigin::Call
        };
        self.provider_options
            .push(ProviderOptionEntry { origin, options });
        self
    }

    /// Attach typed defaults selected by a configured Registry route.
    pub fn with_route_default_provider_options(mut self, options: ProviderOptions) -> Self {
        self.provider_options.push(ProviderOptionEntry {
            origin: ProviderOptionOrigin::RouteDefault,
            options,
        });
        self
    }

    /// Attach typed defaults selected for one concrete model target.
    pub fn with_model_default_provider_options(mut self, options: ProviderOptions) -> Self {
        self.provider_options.push(ProviderOptionEntry {
            origin: ProviderOptionOrigin::ModelDefault,
            options,
        });
        self
    }

    /// Attach typed options selected for one runtime model step.
    pub fn with_runtime_step_provider_options(mut self, options: ProviderOptions) -> Self {
        self.provider_options.push(ProviderOptionEntry {
            origin: ProviderOptionOrigin::RuntimeStep,
            options,
        });
        self
    }
}

const PROTECTED_FIELDS: &[&str] = &[
    "apikey",
    "apitoken",
    "accesstoken",
    "authorization",
    "auth",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "credentials",
    "credential",
    "secretkey",
    "baseurl",
    "baseuri",
    "endpoint",
    "audience",
    "proxy",
    "proxyurl",
    "tls",
    "host",
    "redirect",
    "redirectpolicy",
    "headers",
    "defaultheaders",
];

const MAX_PROVIDER_OPTION_BYTES: usize = 64 * 1024;
const MAX_PROVIDER_OPTION_DEPTH: usize = 32;
const MAX_PROVIDER_OPTION_FIELDS: usize = 1024;

fn validate_option_shape(object: &Map<String, Value>) -> Result<(), ProviderOptionError> {
    let mut fields = object.len();
    if fields > MAX_PROVIDER_OPTION_FIELDS {
        return Err(ProviderOptionError::TooManyFields {
            maximum: MAX_PROVIDER_OPTION_FIELDS,
        });
    }
    for value in object.values() {
        validate_option_value(value, 1, &mut fields)?;
    }
    let encoded = serde_json::to_vec(object)
        .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?;
    if encoded.len() > MAX_PROVIDER_OPTION_BYTES {
        return Err(ProviderOptionError::TooLarge {
            maximum: MAX_PROVIDER_OPTION_BYTES,
        });
    }
    Ok(())
}

fn validate_option_value(
    value: &Value,
    depth: usize,
    fields: &mut usize,
) -> Result<(), ProviderOptionError> {
    if depth > MAX_PROVIDER_OPTION_DEPTH {
        return Err(ProviderOptionError::TooDeep {
            maximum: MAX_PROVIDER_OPTION_DEPTH,
        });
    }
    match value {
        Value::Object(object) => {
            *fields = fields.saturating_add(object.len());
            if *fields > MAX_PROVIDER_OPTION_FIELDS {
                return Err(ProviderOptionError::TooManyFields {
                    maximum: MAX_PROVIDER_OPTION_FIELDS,
                });
            }
            for value in object.values() {
                validate_option_value(value, depth + 1, fields)?;
            }
        }
        Value::Array(items) => {
            for value in items {
                validate_option_value(value, depth + 1, fields)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn reject_protected_fields(
    object: &Map<String, Value>,
    parent: &str,
) -> Result<(), ProviderOptionError> {
    for (key, value) in object {
        let path = if parent.is_empty() {
            key.clone()
        } else {
            format!("{parent}.{key}")
        };
        let normalized = key
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        if PROTECTED_FIELDS.contains(&normalized.as_str()) {
            return Err(ProviderOptionError::ProtectedField { path });
        }
        reject_protected_value(value, &path)?;
    }
    Ok(())
}

fn reject_protected_value(value: &Value, path: &str) -> Result<(), ProviderOptionError> {
    match value {
        Value::Object(object) => reject_protected_fields(object, path),
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                reject_protected_value(item, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use serde::Serialize;
    use serde_json::json;

    use super::*;

    #[derive(Serialize)]
    struct OpenAiOptions {
        reasoning_effort: &'static str,
    }

    impl TypedProviderOptions for OpenAiOptions {
        const NAMESPACE: &'static str = "openai";
    }

    #[test]
    fn typed_options_keep_namespace_and_hide_values_from_debug() {
        let options = ProviderOptions::typed(&OpenAiOptions {
            reasoning_effort: "high",
        })
        .unwrap();

        assert_eq!(options.namespace().as_str(), "openai");
        assert_eq!(options.value()["reasoning_effort"], "high");
        assert!(!format!("{options:?}").contains("high"));
    }

    #[test]
    fn raw_options_reject_protected_fields_recursively() {
        let error = ProviderOptions::checked_raw(
            ProviderId::new("openai").unwrap(),
            json!({"transport": {"api_key": "secret"}}),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ProviderOptionError::ProtectedField { ref path }
                if path == "transport.api_key"
        ));
    }

    #[test]
    fn raw_options_reject_protected_fields_inside_arrays() {
        let error = ProviderOptions::checked_raw(
            ProviderId::new("openai").unwrap(),
            json!({"items": [{"headers": {"x-secret": "secret"}}]}),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ProviderOptionError::ProtectedField { ref path }
                if path == "items[0].headers"
        ));
    }

    #[test]
    fn foreign_namespace_is_rejected_instead_of_ignored() {
        struct Merger;
        impl ProviderOptionMerger for Merger {
            type Output = ();

            fn validate_layer(
                &self,
                _origin: ProviderOptionOrigin,
                _options: &ProviderOptions,
            ) -> Result<(), ProviderOptionError> {
                Ok(())
            }

            fn merge(
                &self,
                _layers: &ProviderOptionLayers,
            ) -> Result<Self::Output, ProviderOptionError> {
                Ok(())
            }
        }

        let options = ProviderOptions::checked_raw(
            ProviderId::new("anthropic").unwrap(),
            json!({"thinking": {"type": "adaptive"}}),
        )
        .unwrap();
        let layers = ProviderOptionLayers::default()
            .with_raw_override(options)
            .unwrap();

        let error = layers
            .merge_for(&ProviderId::new("openai").unwrap(), &Merger)
            .unwrap_err();
        assert!(matches!(
            error,
            ProviderOptionError::NamespaceMismatch { .. }
        ));
    }

    #[test]
    fn layers_expose_the_fixed_precedence_without_generic_merging() {
        fn layer() -> ProviderOptions {
            ProviderOptions::typed(&OpenAiOptions {
                reasoning_effort: "high",
            })
            .unwrap()
        }

        let layers = ProviderOptionLayers::default()
            .with_provider_default(layer())
            .unwrap()
            .with_route_default(layer())
            .unwrap()
            .with_model_default(layer())
            .unwrap()
            .with_runtime_step(layer())
            .unwrap()
            .with_call(layer())
            .unwrap()
            .with_raw_override(
                ProviderOptions::checked_raw(
                    ProviderId::new("openai").unwrap(),
                    json!({"reasoning_effort": "xhigh"}),
                )
                .unwrap(),
            )
            .unwrap();

        assert_eq!(
            layers
                .in_precedence_order()
                .map(|(origin, _)| origin)
                .collect::<Vec<_>>(),
            vec![
                ProviderOptionOrigin::ProviderDefault,
                ProviderOptionOrigin::RouteDefault,
                ProviderOptionOrigin::ModelDefault,
                ProviderOptionOrigin::RuntimeStep,
                ProviderOptionOrigin::Call,
                ProviderOptionOrigin::RawOverride,
            ]
        );
    }

    #[test]
    fn raw_options_cannot_occupy_a_typed_precedence_layer() {
        let raw = ProviderOptions::checked_raw(
            ProviderId::new("openai").unwrap(),
            json!({"reasoning_effort": "high"}),
        )
        .unwrap();
        assert!(matches!(
            ProviderOptionLayers::default().with_call(raw),
            Err(ProviderOptionError::TypedLayerRequired)
        ));
    }

    #[test]
    fn call_options_reject_foreign_namespaces_and_duplicate_layers() {
        let openai = ProviderOptions::typed(&OpenAiOptions {
            reasoning_effort: "high",
        })
        .unwrap();
        let anthropic = ProviderOptions::checked_raw(
            ProviderId::new("anthropic").unwrap(),
            json!({"thinking": {"type": "adaptive"}}),
        )
        .unwrap();
        let foreign = CallOptions::default()
            .with_provider_options(openai.clone())
            .with_provider_options(anthropic)
            .apply_provider_options(
                &ProviderId::new("openai").unwrap(),
                ProviderOptionLayers::default(),
            )
            .unwrap_err();
        assert!(matches!(
            foreign,
            ProviderOptionError::NamespaceMismatch { .. }
        ));

        let duplicate = CallOptions::default()
            .with_provider_options(openai.clone())
            .with_provider_options(openai)
            .apply_provider_options(
                &ProviderId::new("openai").unwrap(),
                ProviderOptionLayers::default(),
            )
            .unwrap_err();
        assert!(matches!(
            duplicate,
            ProviderOptionError::DuplicateLayer {
                origin: ProviderOptionOrigin::Call
            }
        ));
    }

    #[test]
    fn runtime_option_origins_join_the_provider_owned_precedence_stack() {
        fn layer(value: &'static str) -> ProviderOptions {
            #[derive(Serialize)]
            struct Layer {
                value: &'static str,
            }

            impl TypedProviderOptions for Layer {
                const NAMESPACE: &'static str = "openai";
            }

            ProviderOptions::typed(&Layer { value }).unwrap()
        }

        let layers = CallOptions::default()
            .with_route_default_provider_options(layer("route"))
            .with_model_default_provider_options(layer("model"))
            .with_runtime_step_provider_options(layer("step"))
            .with_provider_options(layer("call"))
            .apply_provider_options(
                &ProviderId::new("openai").unwrap(),
                ProviderOptionLayers::default()
                    .with_provider_default(layer("provider"))
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(
            layers
                .in_precedence_order()
                .map(|(origin, options)| (origin, options.value()["value"].as_str().unwrap()))
                .collect::<Vec<_>>(),
            vec![
                (ProviderOptionOrigin::ProviderDefault, "provider"),
                (ProviderOptionOrigin::RouteDefault, "route"),
                (ProviderOptionOrigin::ModelDefault, "model"),
                (ProviderOptionOrigin::RuntimeStep, "step"),
                (ProviderOptionOrigin::Call, "call"),
            ]
        );
    }

    #[test]
    fn provider_schema_validates_erased_typed_layers_too() {
        #[derive(Serialize)]
        struct ForgedOpenAiOptions {
            unrecognized: bool,
        }

        impl TypedProviderOptions for ForgedOpenAiOptions {
            const NAMESPACE: &'static str = "openai";
        }

        struct StrictMerger;
        impl ProviderOptionMerger for StrictMerger {
            type Output = ();

            fn validate_layer(
                &self,
                _origin: ProviderOptionOrigin,
                options: &ProviderOptions,
            ) -> Result<(), ProviderOptionError> {
                if options.value().contains_key("unrecognized") {
                    return Err(ProviderOptionError::Rejected {
                        path: "unrecognized".to_string(),
                        reason: "unknown request-body field".to_string(),
                    });
                }
                Ok(())
            }

            fn merge(
                &self,
                _layers: &ProviderOptionLayers,
            ) -> Result<Self::Output, ProviderOptionError> {
                Ok(())
            }
        }

        let forged = ProviderOptions::typed(&ForgedOpenAiOptions { unrecognized: true }).unwrap();
        let layers = ProviderOptionLayers::default().with_call(forged).unwrap();
        assert!(matches!(
            layers.merge_for(&ProviderId::new("openai").unwrap(), &StrictMerger),
            Err(ProviderOptionError::Rejected { .. })
        ));
    }

    #[test]
    fn raw_options_reject_common_transport_and_auth_aliases() {
        for key in [
            "base_uri",
            "proxy_url",
            "client_secret",
            "access-token",
            "default_headers",
            "redirect_policy",
        ] {
            let error = ProviderOptions::checked_raw(
                ProviderId::new("openai").unwrap(),
                Value::Object(Map::from_iter([(key.to_string(), json!("secret"))])),
            )
            .unwrap_err();
            assert!(matches!(error, ProviderOptionError::ProtectedField { .. }));
        }
    }
}
