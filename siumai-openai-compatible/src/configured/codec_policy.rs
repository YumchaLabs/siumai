use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde_json::{Map, Value};
use siumai_core::{
    Error, LanguageRequest, LanguageResponse, LanguageStreamDecoder, ModelId, ProviderOptionError,
    ProviderScope, Warning,
};
use siumai_protocol_openai::PromptCacheAnnotationResolver;
use siumai_protocol_openai::chat_completions::{
    ChatCompletionsDialect, ChatCompletionsStreamDecoder, ChatRequestEncodingOptions,
    decode_response as decode_chat_response, encode_request_with_options as encode_chat_request,
    is_protected_option_field as is_chat_protected_field,
};
use siumai_protocol_openai::responses::{
    FunctionToolEncodingOptions, RequestEncodingOptions, ResponsesStreamDecoder,
    ResponsesWireDialect, decode_response as decode_responses_response,
    encode_request_with_options as encode_responses_request,
    is_protected_option_field as is_responses_protected_field,
};
use siumai_transport::{RequestHeaders, ResponseHeaders};

pub type CompatibleStreamDecoder = Box<dyn LanguageStreamDecoder<ProtocolFrame = str> + Send>;

/// Provider-owned call preparation and bounded request/response codec selection.
///
/// A policy cannot alter provider identity, endpoint, authentication, transport,
/// retry behavior, operation type, or stream lifecycle.
pub trait ChatCodecPolicy: Send + Sync {
    fn name(&self) -> &'static str;

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        dialect: ChatCompletionsDialect,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error>;

    fn encode_request(
        &self,
        scope: &ProviderScope,
        model: &ModelId,
        prepared: &PreparedChatCall,
        stream: bool,
    ) -> Result<Value, Error> {
        prepared.encode(scope, model, stream)
    }

    /// Apply a reviewed raw provider-body overlay after all typed codec behavior.
    ///
    /// The fail-closed default prevents a branded codec from accidentally
    /// reinterpreting raw body data as headers or other request authority.
    fn apply_raw_body_overlay(
        &self,
        _body: &mut Value,
        raw: Option<&Map<String, Value>>,
    ) -> Result<(), ProviderOptionError> {
        reject_unreviewed_raw(self.name(), raw)
    }

    fn decode_response(
        &self,
        scope: &ProviderScope,
        model: &ModelId,
        _headers: &ResponseHeaders,
        body: &[u8],
        dialect: &ChatCompletionsDialect,
    ) -> Result<LanguageResponse, Error> {
        decode_chat_response(scope, model, body, dialect)
    }

    fn stream_decoder(
        &self,
        scope: ProviderScope,
        model: ModelId,
        dialect: ChatCompletionsDialect,
    ) -> CompatibleStreamDecoder {
        Box::new(ChatCompletionsStreamDecoder::new(scope, model, dialect))
    }
}

pub struct PreparedChatCall {
    pub request: LanguageRequest,
    pub dialect: ChatCompletionsDialect,
    pub extra: BTreeMap<String, Value>,
    /// Non-credential protocol headers selected by the provider codec.
    ///
    /// `RequestHeaders` rejects authentication and transport-controlled headers,
    /// so codec policies cannot use this hook to change endpoint or credential policy.
    pub headers: RequestHeaders,
    /// Provider-owned projection from content annotations to Chat wire cache markers.
    ///
    /// Compatibility policies without a typed cache contract leave this unset.
    pub prompt_cache_resolver: Option<Arc<dyn PromptCacheAnnotationResolver>>,
    pub warnings: Vec<Warning>,
}

impl fmt::Debug for PreparedChatCall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedChatCall")
            .field("request", &self.request)
            .field("dialect", &self.dialect)
            .field("extra", &self.extra)
            .field("headers", &self.headers)
            .field(
                "has_prompt_cache_resolver",
                &self.prompt_cache_resolver.is_some(),
            )
            .field("warnings", &self.warnings)
            .finish()
    }
}

impl PreparedChatCall {
    pub fn encode(
        &self,
        scope: &ProviderScope,
        model: &ModelId,
        stream: bool,
    ) -> Result<Value, Error> {
        let options = ChatRequestEncodingOptions::new(stream).with_extra(self.extra.clone());
        if let Some(resolver) = &self.prompt_cache_resolver {
            siumai_protocol_openai::chat_completions::encode_request_with_options_and_resolver(
                scope,
                model,
                &self.request,
                &self.dialect,
                &options,
                resolver.as_ref(),
            )
        } else {
            encode_chat_request(scope, model, &self.request, &self.dialect, &options)
        }
    }
}

/// Provider-owned Responses call preparation with no transport authority.
pub trait ResponsesCodecPolicy: Send + Sync {
    fn name(&self) -> &'static str;

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedResponsesCall, Error>;

    fn encode_request(
        &self,
        scope: &ProviderScope,
        model: &ModelId,
        prepared: &PreparedResponsesCall,
        stream: bool,
    ) -> Result<Value, Error> {
        prepared.encode(scope, model, stream)
    }

    /// Apply a reviewed raw provider-body overlay after all typed codec behavior.
    fn apply_raw_body_overlay(
        &self,
        _body: &mut Value,
        raw: Option<&Map<String, Value>>,
    ) -> Result<(), ProviderOptionError> {
        reject_unreviewed_raw(self.name(), raw)
    }

    fn decode_response(
        &self,
        scope: &ProviderScope,
        model: &ModelId,
        _headers: &ResponseHeaders,
        body: &[u8],
    ) -> Result<LanguageResponse, Error> {
        let decoded = decode_responses_response(body, scope, model)?;
        let (_, response) = decoded.into_parts();
        Ok(response)
    }

    fn stream_decoder(
        &self,
        scope: ProviderScope,
        model: ModelId,
        wire_dialect: ResponsesWireDialect,
    ) -> CompatibleStreamDecoder {
        Box::new(ResponsesStreamDecoder::new(scope, model).with_wire_dialect(wire_dialect))
    }
}

#[derive(Debug)]
pub struct PreparedResponsesCall {
    pub request: LanguageRequest,
    pub extra: BTreeMap<String, Value>,
    pub headers: RequestHeaders,
    pub native_tools: Vec<Value>,
    pub function_tools: BTreeMap<String, FunctionToolEncodingOptions>,
    pub warnings: Vec<Warning>,
}

impl PreparedResponsesCall {
    pub fn encode(
        &self,
        scope: &ProviderScope,
        model: &ModelId,
        stream: bool,
    ) -> Result<Value, Error> {
        let mut encoding = RequestEncodingOptions::new(stream).with_extra(self.extra.clone());
        for tool in &self.native_tools {
            encoding = encoding.with_native_tool(tool.clone());
        }
        for (name, options) in &self.function_tools {
            encoding = encoding.with_function_tool_options(name, options.clone());
        }
        encode_responses_request(scope, model, &self.request, &encoding)
    }
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
            headers: RequestHeaders::new(),
            prompt_cache_resolver: None,
            warnings: Vec::new(),
        })
    }

    fn apply_raw_body_overlay(
        &self,
        body: &mut Value,
        raw: Option<&Map<String, Value>>,
    ) -> Result<(), ProviderOptionError> {
        apply_reviewed_raw_body_overlay(body, raw, is_chat_protected_field, "Chat Completions")
    }
}

#[derive(Debug, Default)]
pub(crate) struct IdentityResponsesCodecPolicy;

impl ResponsesCodecPolicy for IdentityResponsesCodecPolicy {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn prepare(
        &self,
        _model: &ModelId,
        request: LanguageRequest,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedResponsesCall, Error> {
        Ok(PreparedResponsesCall {
            request,
            extra,
            headers: RequestHeaders::new(),
            native_tools: Vec::new(),
            function_tools: BTreeMap::new(),
            warnings: Vec::new(),
        })
    }

    fn apply_raw_body_overlay(
        &self,
        body: &mut Value,
        raw: Option<&Map<String, Value>>,
    ) -> Result<(), ProviderOptionError> {
        apply_reviewed_raw_body_overlay(body, raw, is_responses_protected_field, "Responses")
    }
}

fn reject_unreviewed_raw(
    policy_name: &str,
    raw: Option<&Map<String, Value>>,
) -> Result<(), ProviderOptionError> {
    if raw.is_none() {
        return Ok(());
    }
    Err(ProviderOptionError::Rejected {
        path: "$".to_string(),
        reason: format!(
            "raw provider options are not supported by the `{policy_name}` codec; use typed provider options"
        ),
    })
}

fn apply_reviewed_raw_body_overlay(
    body: &mut Value,
    raw: Option<&Map<String, Value>>,
    is_protected: fn(&str) -> bool,
    mode_name: &str,
) -> Result<(), ProviderOptionError> {
    let Some(raw) = raw else {
        return Ok(());
    };
    if let Some(name) = raw.keys().find(|name| is_protected(name)) {
        return Err(ProviderOptionError::Rejected {
            path: name.clone(),
            reason: format!("field is owned by the canonical {mode_name} request"),
        });
    }
    let Value::Object(body) = body else {
        return Err(ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: format!("the encoded {mode_name} request body is not a JSON object"),
        });
    };
    body.extend(raw.clone());
    Ok(())
}
