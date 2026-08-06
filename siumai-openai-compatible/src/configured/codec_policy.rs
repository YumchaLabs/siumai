use std::collections::BTreeMap;

use serde_json::Value;
use siumai_core::{
    Error, LanguageRequest, LanguageResponse, LanguageStreamDecoder, ModelId, ProviderScope,
    Warning,
};
use siumai_protocol_openai::chat_completions::{
    ChatCompletionsDialect, ChatCompletionsStreamDecoder, ChatPromptCacheBlock,
    ChatRequestEncodingOptions, decode_response as decode_chat_response,
    encode_request_with_options as encode_chat_request,
};
use siumai_protocol_openai::responses_next::{
    FunctionToolEncodingOptions, PromptCacheBlock, RequestEncodingOptions, ResponsesStreamDecoder,
    decode_response as decode_responses_response,
    encode_request_with_options as encode_responses_request,
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

#[derive(Debug)]
pub struct PreparedChatCall {
    pub request: LanguageRequest,
    pub dialect: ChatCompletionsDialect,
    pub extra: BTreeMap<String, Value>,
    /// Non-credential protocol headers selected by the provider codec.
    ///
    /// `RequestHeaders` rejects authentication and transport-controlled headers,
    /// so codec policies cannot use this hook to change endpoint or credential policy.
    pub headers: RequestHeaders,
    pub prompt_cache_breakpoints: Vec<ChatPromptCacheBlock>,
    pub warnings: Vec<Warning>,
}

impl PreparedChatCall {
    pub fn encode(
        &self,
        scope: &ProviderScope,
        model: &ModelId,
        stream: bool,
    ) -> Result<Value, Error> {
        let mut options = ChatRequestEncodingOptions::new(stream).with_extra(self.extra.clone());
        for breakpoint in &self.prompt_cache_breakpoints {
            options = options.with_prompt_cache_breakpoint(*breakpoint);
        }
        encode_chat_request(scope, model, &self.request, &self.dialect, &options)
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

    fn stream_decoder(&self, scope: ProviderScope, model: ModelId) -> CompatibleStreamDecoder {
        Box::new(ResponsesStreamDecoder::new(scope, model))
    }
}

#[derive(Debug)]
pub struct PreparedResponsesCall {
    pub request: LanguageRequest,
    pub extra: BTreeMap<String, Value>,
    pub headers: RequestHeaders,
    pub prompt_cache_breakpoints: Vec<PromptCacheBlock>,
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
        for breakpoint in &self.prompt_cache_breakpoints {
            encoding = encoding.with_prompt_cache_breakpoint(*breakpoint);
        }
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
            prompt_cache_breakpoints: Vec::new(),
            warnings: Vec::new(),
        })
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
            prompt_cache_breakpoints: Vec::new(),
            native_tools: Vec::new(),
            function_tools: BTreeMap::new(),
            warnings: Vec::new(),
        })
    }
}
