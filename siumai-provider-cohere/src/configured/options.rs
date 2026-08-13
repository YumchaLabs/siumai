use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, Model, ProviderOptionError, ProviderOptionSelection, ProviderOptions,
    TypedProviderOptions,
};

use crate::provider_options::{CohereEmbeddingOptions, CohereRerankOptions};

pub(crate) fn embedding_options<M: Model + ?Sized>(
    call: &CallOptions,
    model: &M,
) -> Result<CohereEmbeddingOptions, ProviderOptionError> {
    merge_options(
        call,
        model,
        &["inputType", "truncate", "outputDimension"],
        canonical_embedding_field,
        "Cohere embedding",
    )
}

pub(crate) fn rerank_options<M: Model + ?Sized>(
    call: &CallOptions,
    model: &M,
) -> Result<CohereRerankOptions, ProviderOptionError> {
    merge_options(
        call,
        model,
        &["maxTokensPerDoc", "priority"],
        canonical_rerank_field,
        "Cohere rerank",
    )
}

fn merge_options<T, M>(
    call: &CallOptions,
    model: &M,
    allowed: &'static [&'static str],
    canonicalize: fn(&str) -> Option<&'static str>,
    mode: &'static str,
) -> Result<T, ProviderOptionError>
where
    T: DeserializeOwned + TypedProviderOptions,
    M: Model + ?Sized,
{
    let selection = call.provider_options_for(model)?;
    merge_selected(&selection, allowed, canonicalize, mode)
}

fn merge_selected<T>(
    selection: &ProviderOptionSelection<'_>,
    allowed: &'static [&'static str],
    canonicalize: fn(&str) -> Option<&'static str>,
    mode: &'static str,
) -> Result<T, ProviderOptionError>
where
    T: DeserializeOwned + TypedProviderOptions,
{
    let mut merged = Map::new();
    for options in selection.typed() {
        let patch = canonicalize_patch(options, allowed, canonicalize)?;
        decode_and_validate::<T>(Value::Object(patch.clone()))?;
        merged.extend(patch);
    }
    if selection.raw_override().is_some() {
        return Err(ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: format!("{mode} only accepts typed provider options"),
        });
    }
    decode_and_validate(Value::Object(merged))
}

fn canonicalize_patch(
    options: &ProviderOptions,
    allowed: &'static [&'static str],
    canonicalize: fn(&str) -> Option<&'static str>,
) -> Result<Map<String, Value>, ProviderOptionError> {
    let mut patch = Map::new();
    for (field, value) in options.value() {
        let canonical = canonicalize(field).ok_or_else(|| ProviderOptionError::Rejected {
            path: field.clone(),
            reason: "field is not valid for this Cohere model family".to_string(),
        })?;
        if !allowed.contains(&canonical) {
            return Err(ProviderOptionError::Rejected {
                path: field.clone(),
                reason: "field is not valid for this Cohere model family".to_string(),
            });
        }
        patch.insert(canonical.to_string(), value.clone());
    }
    Ok(patch)
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
