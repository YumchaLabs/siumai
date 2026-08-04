use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger,
    ProviderOptionOrigin, ProviderOptions, TypedProviderOptions,
};

use crate::provider_options::{CohereEmbeddingOptions, CohereRerankOptions};

pub(crate) fn embedding_options(
    call: &CallOptions,
) -> Result<CohereEmbeddingOptions, ProviderOptionError> {
    merge_options(
        call,
        &["inputType", "truncate", "outputDimension"],
        canonical_embedding_field,
    )
}

pub(crate) fn rerank_options(
    call: &CallOptions,
) -> Result<CohereRerankOptions, ProviderOptionError> {
    merge_options(
        call,
        &["maxTokensPerDoc", "priority"],
        canonical_rerank_field,
    )
}

fn merge_options<T>(
    call: &CallOptions,
    allowed: &'static [&'static str],
    canonicalize: fn(&str) -> Option<&'static str>,
) -> Result<T, ProviderOptionError>
where
    T: DeserializeOwned + TypedProviderOptions,
{
    let provider = siumai_core::ProviderId::new(T::NAMESPACE)
        .map_err(|_| ProviderOptionError::InvalidNamespace(T::NAMESPACE.to_string()))?;
    let layers = call.apply_provider_options(&provider, ProviderOptionLayers::default())?;
    let value = layers.merge_for(
        &provider,
        &CohereOptionMerger {
            allowed,
            canonicalize,
        },
    )?;
    decode_and_validate(value)
}

struct CohereOptionMerger {
    allowed: &'static [&'static str],
    canonicalize: fn(&str) -> Option<&'static str>,
}

impl ProviderOptionMerger for CohereOptionMerger {
    type Output = Value;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        for field in options.value().keys() {
            let Some(canonical) = (self.canonicalize)(field) else {
                return Err(ProviderOptionError::Rejected {
                    path: field.clone(),
                    reason: "field is not valid for this Cohere model family".to_string(),
                });
            };
            if !self.allowed.contains(&canonical) {
                return Err(ProviderOptionError::Rejected {
                    path: field.clone(),
                    reason: "field is not valid for this Cohere model family".to_string(),
                });
            }
        }
        Ok(())
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = Map::new();
        for (_, options) in layers.in_precedence_order() {
            for (field, value) in options.value() {
                let canonical =
                    (self.canonicalize)(field).ok_or_else(|| ProviderOptionError::Rejected {
                        path: field.clone(),
                        reason: "field is not valid for this Cohere model family".to_string(),
                    })?;
                merged.insert(canonical.to_string(), value.clone());
            }
        }
        Ok(Value::Object(merged))
    }
}

fn decode_and_validate<T>(value: Value) -> Result<T, ProviderOptionError>
where
    T: DeserializeOwned + TypedProviderOptions,
{
    let options = serde_json::from_value::<T>(value)
        .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?;
    options.validate()?;
    Ok(options)
}

fn canonical_embedding_field(field: &str) -> Option<&'static str> {
    match field {
        "inputType" | "input_type" => Some("inputType"),
        "truncate" => Some("truncate"),
        "outputDimension" | "output_dimension" => Some("outputDimension"),
        _ => None,
    }
}

fn canonical_rerank_field(field: &str) -> Option<&'static str> {
    match field {
        "maxTokensPerDoc" | "max_tokens_per_doc" => Some("maxTokensPerDoc"),
        "priority" => Some("priority"),
        _ => None,
    }
}
