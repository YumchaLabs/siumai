use std::collections::BTreeMap;

use serde_json::Value;
use siumai_core::{Error, LanguageRequest, ModelId, Warning};
use siumai_protocol_openai::chat_completions::ChatCompletionsDialect;

/// Provider-owned call preparation and bounded request/response codec selection.
///
/// A policy cannot alter provider identity, endpoint, authentication, transport,
/// retry behavior, operation type, or stream lifecycle.
pub(crate) trait ChatCodecPolicy: Send + Sync {
    fn name(&self) -> &'static str;

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        dialect: ChatCompletionsDialect,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error>;
}

pub(crate) struct PreparedChatCall {
    pub(crate) request: LanguageRequest,
    pub(crate) dialect: ChatCompletionsDialect,
    pub(crate) extra: BTreeMap<String, Value>,
    pub(crate) warnings: Vec<Warning>,
}

#[derive(Debug, Default)]
pub(crate) struct IdentityChatCodecPolicy;

impl ChatCodecPolicy for IdentityChatCodecPolicy {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn prepare(
        &self,
        _model: &ModelId,
        request: LanguageRequest,
        dialect: ChatCompletionsDialect,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error> {
        Ok(PreparedChatCall {
            request,
            dialect,
            extra,
            warnings: Vec::new(),
        })
    }
}
