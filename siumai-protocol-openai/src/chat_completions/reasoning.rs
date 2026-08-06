use serde_json::Value;
use siumai_core::{
    Error, ErrorKind, ModelId, OpaqueProviderItem, ProviderProvenance, ProviderScope,
};

/// Opaque kind used for structured Chat Completions reasoning history.
pub const REASONING_DETAILS_OPAQUE_KIND: &str = "chat.message.reasoning_details";

const MAX_REASONING_DETAILS: usize = 64;
const MAX_REASONING_DETAILS_BYTES: usize = 1024 * 1024;

/// One bounded structured-reasoning snapshot.
#[derive(Clone)]
pub(crate) struct ReasoningDetailsSnapshot {
    value: Value,
}

impl ReasoningDetailsSnapshot {
    pub(crate) fn response(value: &Value) -> Result<Self, Error> {
        Self::parse(
            value,
            ErrorKind::Protocol,
            "Chat Completions reasoning_details had an invalid response shape",
        )
    }

    pub(crate) fn history(value: &Value) -> Result<Self, Error> {
        Self::parse(
            value,
            ErrorKind::InvalidInput,
            "Chat Completions reasoning_details history had an invalid shape",
        )
    }

    fn parse(value: &Value, kind: ErrorKind, message: &'static str) -> Result<Self, Error> {
        let array = value.as_array().ok_or_else(|| Error::new(kind, message))?;
        if array.len() > MAX_REASONING_DETAILS {
            return Err(Error::new(kind, message));
        }
        if array.iter().any(|detail| !detail.is_object()) {
            return Err(Error::new(kind, message));
        }

        let encoded_bytes = serde_json::to_vec(value)
            .map_err(|source| Error::new(kind, message).with_source(source))?
            .len();
        if encoded_bytes > MAX_REASONING_DETAILS_BYTES {
            return Err(Error::new(kind, message));
        }

        Ok(Self {
            value: value.clone(),
        })
    }

    pub(crate) fn into_opaque(
        self,
        scope: &ProviderScope,
        model: &ModelId,
    ) -> Result<OpaqueProviderItem, Error> {
        OpaqueProviderItem::with_limit(
            provenance(scope, model),
            REASONING_DETAILS_OPAQUE_KIND,
            self.value,
            MAX_REASONING_DETAILS_BYTES,
        )
        .map_err(|source| {
            Error::new(
                ErrorKind::ResponseLimit,
                "Chat Completions reasoning_details exceeded its preservation limit",
            )
            .with_source(source)
        })
    }
}

pub(crate) fn preserve_reasoning_details(
    scope: &ProviderScope,
    model: &ModelId,
    value: &Value,
) -> Result<OpaqueProviderItem, Error> {
    ReasoningDetailsSnapshot::response(value)?.into_opaque(scope, model)
}

pub(crate) fn replay_reasoning_details(
    scope: &ProviderScope,
    model: &ModelId,
    item: &OpaqueProviderItem,
) -> Result<Value, Error> {
    if item.kind() != REASONING_DETAILS_OPAQUE_KIND
        || !provenance_matches(item.provenance(), scope, model)
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "provider-native Chat Completions history cannot be replayed in this scope",
        ));
    }
    ReasoningDetailsSnapshot::history(item.data())?;
    Ok(item.data().clone())
}

fn provenance(scope: &ProviderScope, model: &ModelId) -> ProviderProvenance {
    ProviderProvenance {
        provider: scope.provider_id().clone(),
        platform: scope.platform().map(ToString::to_string),
        protocol: scope
            .protocol()
            .map(ToString::to_string)
            .unwrap_or_else(|| super::PROTOCOL_ID.to_string()),
        model: model.clone(),
    }
}

fn provenance_matches(
    provenance: &ProviderProvenance,
    scope: &ProviderScope,
    model: &ModelId,
) -> bool {
    provenance.provider == *scope.provider_id()
        && provenance.platform.as_deref() == scope.platform().map(|value| value.as_str())
        && provenance.protocol
            == scope
                .protocol()
                .map_or(super::PROTOCOL_ID, |value| value.as_str())
        && provenance.model == *model
}
