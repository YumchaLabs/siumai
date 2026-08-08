use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde_json::Value;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, LanguageModel, LanguageRequest, LanguageResponse,
    LanguageStream, LanguageStreamDecoder, LanguageStreamEvent, Model, ModelAdvisory,
    ModelDescriptor, ModelFamily, ModelId, ModelOperation, ModelPolicy, ModelPolicyDecision,
    ProviderOptionError, PublicDiagnosticText, SensitiveResponse, StreamTerminal, SupportState,
    Warning, WarningKind,
};
use siumai_protocol_openai::chat_completions::{
    CHAT_COMPLETIONS_TARGET, ChatCompletionsDialect, ChatCompletionsStreamDecoder,
    ChatPromptCacheBlock, ChatRequestEncodingOptions, MaxOutputTokensField,
    decode_response as decode_chat_response,
    encode_request_with_options as encode_chat_request_with_options,
};
use siumai_protocol_openai::openai_error::{classify_http_error, decode_error_metadata};
use siumai_protocol_openai::responses::{
    PromptCacheBlock, RequestEncodingOptions, ResponsesStreamDecoder,
    decode_response as decode_responses_response, encode_request_with_options,
};
use siumai_transport::framing::{SseDecoder, SseFrameError};
use siumai_transport::{
    RequestBody, RequestHeaders, RequestPlan, RequestTarget, ResponseHeaders, TransportByteStream,
    TransportLimits, TransportResponse, TransportStreamResponse,
};

use super::catalog::classify_model;
use super::mode::OpenAiApiMode;
use super::provider::{OpenAiMergedOptions, OpenAiRuntime};
use super::responses_resource::OpenAiBackgroundResponse;

const RESPONSES_TARGET: &str = "responses";
const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

/// Lightweight model handle for the recommended OpenAI Responses API.
#[derive(Clone)]
pub struct OpenAiResponsesModel {
    pub(crate) runtime: Arc<OpenAiRuntime>,
    descriptor: ModelDescriptor,
}

impl OpenAiResponsesModel {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.scope_arc(OpenAiApiMode::Responses),
            model,
            ModelFamily::Language,
        );
        Self {
            runtime,
            descriptor,
        }
    }

    pub const fn api_mode(&self) -> OpenAiApiMode {
        OpenAiApiMode::Responses
    }

    fn plan(
        &self,
        request: &LanguageRequest,
        stream: bool,
        merged: OpenAiMergedOptions,
    ) -> Result<RequestPlan, Error> {
        self.plan_internal(request, stream, false, merged)
    }

    fn background_plan(
        &self,
        request: &LanguageRequest,
        merged: OpenAiMergedOptions,
    ) -> Result<RequestPlan, Error> {
        self.plan_internal(request, false, true, merged)
    }

    fn plan_internal(
        &self,
        request: &LanguageRequest,
        stream: bool,
        background: bool,
        merged: OpenAiMergedOptions,
    ) -> Result<RequestPlan, Error> {
        let mut encoding = RequestEncodingOptions::new(stream).with_extra(merged.wire);
        for breakpoint in merged.prompt_cache_breakpoints {
            encoding = encoding.with_prompt_cache_breakpoint(PromptCacheBlock::new(
                breakpoint.message_index,
                breakpoint.content_index,
            ));
        }
        for tool in merged.native_tools {
            encoding = encoding.with_native_tool(tool);
        }
        for (name, options) in merged.function_tools {
            encoding = encoding.with_function_tool_options(name, options);
        }
        let mut body = encode_request_with_options(
            self.runtime.scope(OpenAiApiMode::Responses),
            self.model_id(),
            request,
            &encoding,
        )?;
        if background {
            body.as_object_mut()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Protocol,
                        "OpenAI Responses encoder produced a non-object request",
                    )
                })?
                .insert("background".to_string(), Value::Bool(true));
        }
        request_plan(
            RESPONSES_TARGET,
            body,
            stream,
            self.runtime.replay_safety.clone(),
            OpenAiApiMode::Responses,
        )
    }

    fn contextualize(&self, operation: ModelOperation, error: Error) -> Error {
        contextualize(self, operation, error)
    }

    /// Create a provider-native background response without forcing a terminal projection.
    pub async fn create_background(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<OpenAiBackgroundResponse, Error> {
        let operation = ModelOperation::Generate;
        let mut warnings = policy_warnings(
            &self.runtime,
            OpenAiApiMode::Responses,
            self.model_id(),
            operation,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        let mut merged = self
            .runtime
            .merge_options(OpenAiApiMode::Responses, &options)
            .map_err(|source| {
                self.contextualize(operation, option_error(OpenAiApiMode::Responses, source))
            })?;
        let (request, compatibility_warnings) = normalize_request(
            OpenAiApiMode::Responses,
            self.model_id(),
            request,
            &mut merged,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        warnings.extend(compatibility_warnings);
        let plan = self
            .background_plan(&request, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(
                operation,
                response_error(OpenAiApiMode::Responses, response),
            ));
        }
        let resource = siumai_protocol_openai::responses::decode_response_resource(response.body())
            .map_err(|error| self.contextualize(operation, error))?;
        Ok(OpenAiBackgroundResponse::new(resource, warnings))
    }
}

impl std::fmt::Debug for OpenAiResponsesModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for OpenAiResponsesModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for OpenAiResponsesModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        let operation = ModelOperation::Generate;
        let mut warnings = policy_warnings(
            &self.runtime,
            OpenAiApiMode::Responses,
            self.model_id(),
            operation,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        let mut merged = self
            .runtime
            .merge_options(OpenAiApiMode::Responses, &options)
            .map_err(|source| {
                self.contextualize(operation, option_error(OpenAiApiMode::Responses, source))
            })?;
        let (request, compatibility_warnings) = normalize_request(
            OpenAiApiMode::Responses,
            self.model_id(),
            request,
            &mut merged,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        warnings.extend(compatibility_warnings);
        let plan = self
            .plan(&request, false, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(
                operation,
                response_error(OpenAiApiMode::Responses, response),
            ));
        }
        let decoded = decode_responses_response(
            response.body(),
            self.runtime.scope(OpenAiApiMode::Responses),
            self.model_id(),
        )
        .map_err(|error| self.contextualize(operation, error))?;
        let (_, response) = decoded.into_parts();
        Ok(with_policy_warnings(response, &warnings))
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let operation = ModelOperation::Stream;
        let mut warnings = policy_warnings(
            &self.runtime,
            OpenAiApiMode::Responses,
            self.model_id(),
            operation,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        let mut merged = self
            .runtime
            .merge_options(OpenAiApiMode::Responses, &options)
            .map_err(|source| {
                self.contextualize(operation, option_error(OpenAiApiMode::Responses, source))
            })?;
        let (request, compatibility_warnings) = normalize_request(
            OpenAiApiMode::Responses,
            self.model_id(),
            request,
            &mut merged,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        warnings.extend(compatibility_warnings);
        let plan = self
            .plan(&request, true, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let cancellation = options.cancellation().clone();
        let response = self
            .runtime
            .transport
            .execute_stream(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            let error = stream_response_error(OpenAiApiMode::Responses, response).await;
            return Err(self.contextualize(operation, error));
        }
        let (status, headers, body) = response.into_parts();
        let diagnostics = headers.diagnostics().with_status(status.as_u16());

        Ok(decode_sse_stream(
            cancellation,
            body,
            self.runtime.transport.limits().clone(),
            ResponsesStreamDecoder::new(
                self.runtime.scope(OpenAiApiMode::Responses).clone(),
                self.model_id().clone(),
            )
            .with_response_diagnostics(diagnostics),
            warnings,
            OpenAiApiMode::Responses,
            model_error_context(self, operation),
        ))
    }
}

/// Lightweight model handle for explicit OpenAI Chat Completions compatibility.
#[derive(Clone)]
pub struct OpenAiChatCompletionsModel {
    pub(crate) runtime: Arc<OpenAiRuntime>,
    descriptor: ModelDescriptor,
}

impl OpenAiChatCompletionsModel {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.scope_arc(OpenAiApiMode::ChatCompletions),
            model,
            ModelFamily::Language,
        );
        Self {
            runtime,
            descriptor,
        }
    }

    pub const fn api_mode(&self) -> OpenAiApiMode {
        OpenAiApiMode::ChatCompletions
    }

    fn plan(
        &self,
        request: &LanguageRequest,
        stream: bool,
        merged: OpenAiMergedOptions,
    ) -> Result<RequestPlan, Error> {
        let mut encoding = ChatRequestEncodingOptions::new(stream).with_extra(merged.wire);
        for breakpoint in merged.prompt_cache_breakpoints {
            encoding = encoding.with_prompt_cache_breakpoint(ChatPromptCacheBlock::new(
                breakpoint.message_index,
                breakpoint.content_index,
            ));
        }
        let body = encode_chat_request_with_options(
            self.runtime.scope(OpenAiApiMode::ChatCompletions),
            self.model_id(),
            request,
            &official_chat_dialect(),
            &encoding,
        )?;
        request_plan(
            CHAT_COMPLETIONS_TARGET,
            body,
            stream,
            self.runtime.replay_safety.clone(),
            OpenAiApiMode::ChatCompletions,
        )
    }

    fn contextualize(&self, operation: ModelOperation, error: Error) -> Error {
        contextualize(self, operation, error)
    }
}

impl std::fmt::Debug for OpenAiChatCompletionsModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiChatCompletionsModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for OpenAiChatCompletionsModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for OpenAiChatCompletionsModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        let operation = ModelOperation::Generate;
        let mut warnings = policy_warnings(
            &self.runtime,
            OpenAiApiMode::ChatCompletions,
            self.model_id(),
            operation,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        let mut merged = self
            .runtime
            .merge_options(OpenAiApiMode::ChatCompletions, &options)
            .map_err(|source| {
                self.contextualize(
                    operation,
                    option_error(OpenAiApiMode::ChatCompletions, source),
                )
            })?;
        let (request, compatibility_warnings) = normalize_request(
            OpenAiApiMode::ChatCompletions,
            self.model_id(),
            request,
            &mut merged,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        warnings.extend(compatibility_warnings);
        let plan = self
            .plan(&request, false, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(
                operation,
                response_error(OpenAiApiMode::ChatCompletions, response),
            ));
        }
        let response = decode_chat_response(
            self.runtime.scope(OpenAiApiMode::ChatCompletions),
            self.model_id(),
            response.body(),
            &official_chat_dialect(),
        )
        .map_err(|error| self.contextualize(operation, error))?;
        Ok(with_policy_warnings(response, &warnings))
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let operation = ModelOperation::Stream;
        let mut warnings = policy_warnings(
            &self.runtime,
            OpenAiApiMode::ChatCompletions,
            self.model_id(),
            operation,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        let mut merged = self
            .runtime
            .merge_options(OpenAiApiMode::ChatCompletions, &options)
            .map_err(|source| {
                self.contextualize(
                    operation,
                    option_error(OpenAiApiMode::ChatCompletions, source),
                )
            })?;
        let (request, compatibility_warnings) = normalize_request(
            OpenAiApiMode::ChatCompletions,
            self.model_id(),
            request,
            &mut merged,
        )
        .map_err(|error| self.contextualize(operation, error))?;
        warnings.extend(compatibility_warnings);
        let plan = self
            .plan(&request, true, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let cancellation = options.cancellation().clone();
        let response = self
            .runtime
            .transport
            .execute_stream(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            let error = stream_response_error(OpenAiApiMode::ChatCompletions, response).await;
            return Err(self.contextualize(operation, error));
        }
        let (status, headers, body) = response.into_parts();
        let diagnostics = headers.diagnostics().with_status(status.as_u16());

        Ok(decode_sse_stream(
            cancellation,
            body,
            self.runtime.transport.limits().clone(),
            ChatCompletionsStreamDecoder::new(
                self.runtime.scope(OpenAiApiMode::ChatCompletions).clone(),
                self.model_id().clone(),
                official_chat_dialect(),
            )
            .with_response_diagnostics(diagnostics),
            warnings,
            OpenAiApiMode::ChatCompletions,
            model_error_context(self, operation),
        ))
    }
}

fn official_chat_dialect() -> ChatCompletionsDialect {
    ChatCompletionsDialect::generic()
        .with_developer_role(true)
        .with_max_output_tokens_field(MaxOutputTokensField::MaxCompletionTokens)
        .with_stream_usage(true)
}

fn normalize_request(
    mode: OpenAiApiMode,
    model: &ModelId,
    mut request: LanguageRequest,
    merged: &mut OpenAiMergedOptions,
) -> Result<(LanguageRequest, Vec<Warning>), Error> {
    let mut warnings = Vec::new();
    if mode == OpenAiApiMode::Responses {
        if request.generation.seed.take().is_some() {
            warnings.push(Warning::new(
                WarningKind::UnsupportedOption,
                "OpenAI Responses does not support the neutral seed control; the field was omitted",
            ));
        }
        if !request.generation.stop_sequences.is_empty() {
            request.generation.stop_sequences.clear();
            warnings.push(Warning::new(
                WarningKind::UnsupportedOption,
                "OpenAI Responses does not support neutral stop sequences; the field was omitted",
            ));
        }
    }

    let model_class = classify_model(model.as_str());
    validate_prompt_cache_model_policy(model_class, merged)?;

    if !model_class.is_gpt_5_6() {
        return Ok((request, warnings));
    }

    let effort = selected_reasoning_effort(mode, &merged.wire);
    if effort == Some("minimal") {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "GPT-5.6 does not support the minimal reasoning effort",
        ));
    }
    if effort == Some("none") {
        return Ok((request, warnings));
    }

    if request.generation.temperature.take().is_some() {
        warnings.push(Warning::new(
            WarningKind::UnsupportedOption,
            "GPT-5.6 temperature is only supported with reasoning effort none; the field was omitted",
        ));
    }
    if request.generation.top_p.take().is_some() {
        warnings.push(Warning::new(
            WarningKind::UnsupportedOption,
            "GPT-5.6 top_p is only supported with reasoning effort none; the field was omitted",
        ));
    }

    match mode {
        OpenAiApiMode::Responses => {
            let removed_top_logprobs = merged.wire.remove("top_logprobs").is_some();
            let mut removed_logprobs_include = false;
            let mut remove_empty_include = false;
            if let Some(Value::Array(include)) = merged.wire.get_mut("include") {
                let original_len = include.len();
                include.retain(|value| value.as_str() != Some("message.output_text.logprobs"));
                removed_logprobs_include = include.len() != original_len;
                remove_empty_include = include.is_empty();
            }
            if remove_empty_include {
                merged.wire.remove("include");
            }
            if removed_top_logprobs || removed_logprobs_include {
                warnings.push(Warning::new(
                    WarningKind::UnsupportedOption,
                    "GPT-5.6 log probabilities are only supported with reasoning effort none; the field was omitted",
                ));
            }
        }
        OpenAiApiMode::ChatCompletions => {
            let removed_logprobs = merged.wire.remove("logprobs").is_some();
            let removed_top_logprobs = merged.wire.remove("top_logprobs").is_some();
            if removed_logprobs || removed_top_logprobs {
                warnings.push(Warning::new(
                    WarningKind::UnsupportedOption,
                    "GPT-5.6 log probabilities are only supported with reasoning effort none; the fields were omitted",
                ));
            }
            if merged.wire.remove("logit_bias").is_some() {
                warnings.push(Warning::new(
                    WarningKind::UnsupportedOption,
                    "GPT-5.6 logit bias is unavailable while reasoning is enabled; the field was omitted",
                ));
            }
        }
    }
    Ok((request, warnings))
}

fn validate_prompt_cache_model_policy(
    model_class: super::catalog::OpenAiModelClass,
    merged: &OpenAiMergedOptions,
) -> Result<(), Error> {
    let retention = merged
        .wire
        .get("prompt_cache_retention")
        .and_then(Value::as_str);
    if (model_class.is_gpt_5_5() || model_class.is_gpt_5_6()) && retention == Some("in_memory") {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "GPT-5.5 and later models only support 24h prompt-cache retention",
        ));
    }
    if model_class.is_gpt_5_5()
        && (merged.wire.contains_key("prompt_cache_options")
            || !merged.prompt_cache_breakpoints.is_empty())
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "GPT-5.5 does not support GPT-5.6 prompt-cache TTL or explicit breakpoints",
        ));
    }
    Ok(())
}

fn selected_reasoning_effort(mode: OpenAiApiMode, wire: &BTreeMap<String, Value>) -> Option<&str> {
    match mode {
        OpenAiApiMode::Responses => wire
            .get("reasoning")
            .and_then(Value::as_object)
            .and_then(|reasoning| reasoning.get("effort"))
            .and_then(Value::as_str),
        OpenAiApiMode::ChatCompletions => wire.get("reasoning_effort").and_then(Value::as_str),
    }
}

fn request_plan(
    target: &'static str,
    body: Value,
    stream: bool,
    replay_safety: siumai_transport::ReplaySafety,
    mode: OpenAiApiMode,
) -> Result<RequestPlan, Error> {
    let accept = if stream {
        HeaderValue::from_static("text/event-stream")
    } else {
        HeaderValue::from_static("application/json")
    };
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, accept)
        .map_err(|source| request_build_error(mode, source))?;
    RequestPlan::new(
        Method::POST,
        RequestTarget::new(target).map_err(|source| request_build_error(mode, source))?,
    )
    .with_headers(headers)
    .with_body(RequestBody::json(&body).map_err(|source| request_build_error(mode, source))?)
    .with_replay_safety(replay_safety)
    .map_err(|source| request_build_error(mode, source))
}

fn contextualize(model: &impl Model, operation: ModelOperation, error: Error) -> Error {
    error.with_context(model_error_context(model, operation))
}

fn model_error_context(model: &impl Model, operation: ModelOperation) -> ErrorContext {
    ErrorContext {
        operation: Some(operation),
        provider: Some(model.provider_id().clone()),
        route: None,
        model: Some(model.model_id().clone()),
    }
}

fn policy_warnings(
    runtime: &OpenAiRuntime,
    mode: OpenAiApiMode,
    model: &ModelId,
    operation: ModelOperation,
) -> Result<Vec<Warning>, Error> {
    let decision = runtime
        .policy
        .evaluate(&siumai_core::ModelPolicyContext::new(
            runtime.scope_arc(mode),
            model.clone(),
            operation,
        ));
    reject_unsupported(mode, &decision)?;
    Ok(decision.advisories().iter().map(advisory_warning).collect())
}

fn reject_unsupported(mode: OpenAiApiMode, decision: &ModelPolicyDecision) -> Result<(), Error> {
    if matches!(decision.state(), SupportState::Unsupported { .. }) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            match mode {
                OpenAiApiMode::Responses => {
                    "model policy rejected the requested OpenAI Responses operation"
                }
                OpenAiApiMode::ChatCompletions => {
                    "model policy rejected the requested OpenAI Chat Completions operation"
                }
            },
        ));
    }
    Ok(())
}

fn advisory_warning(advisory: &ModelAdvisory) -> Warning {
    match advisory {
        ModelAdvisory::UnknownModel => Warning::new(
            WarningKind::UnknownModel,
            "model is absent from the verified OpenAI advisory catalog",
        ),
        ModelAdvisory::Deprecated { .. } => Warning::new(
            WarningKind::DeprecatedModel,
            "OpenAI marks this model as deprecated; inspect the provider profile for its replacement",
        ),
        ModelAdvisory::Retired { .. } => Warning::new(
            WarningKind::RetiredModel,
            "OpenAI marks this model as retired",
        ),
        ModelAdvisory::RollingAlias => Warning::new(
            WarningKind::RollingModelAlias,
            "the OpenAI model ID is a rolling alias whose routed snapshot may change",
        ),
        _ => Warning::provider(
            "model_advisory",
            "the OpenAI provider profile returned an additional model advisory",
        ),
    }
}

fn with_policy_warnings(mut response: LanguageResponse, warnings: &[Warning]) -> LanguageResponse {
    if warnings.is_empty() {
        return response;
    }
    let mut combined = response.warnings().to_vec();
    combined.extend_from_slice(warnings);
    response = response.with_warnings(combined);
    response
}

fn attach_policy_warnings(event: &mut LanguageStreamEvent, warnings: &[Warning]) {
    if warnings.is_empty() {
        return;
    }
    match event {
        LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => {
            replace_response_warnings(response, warnings);
        }
        LanguageStreamEvent::Terminal(StreamTerminal::Failed {
            response: Some(response),
            ..
        })
        | LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
            response: Some(response),
            ..
        }) => {
            replace_response_warnings(response, warnings);
        }
        _ => {}
    }
}

fn replace_response_warnings(response: &mut Box<LanguageResponse>, warnings: &[Warning]) {
    let updated = with_policy_warnings(response.as_ref().clone(), warnings);
    **response = updated;
}

fn decode_sse_stream<D>(
    cancellation: siumai_core::Cancellation,
    body: TransportByteStream,
    limits: TransportLimits,
    mut protocol: D,
    warnings: Vec<Warning>,
    mode: OpenAiApiMode,
    context: ErrorContext,
) -> LanguageStream
where
    D: LanguageStreamDecoder<ProtocolFrame = str> + Send + 'static,
{
    established_stream(cancellation, move |_| {
        async_stream::try_stream! {
            let mut body = body;
            let mut framing = SseDecoder::new(&limits);
            while let Some(chunk) = body.next().await {
                let chunk = chunk.map_err(|error| error.with_context(context.clone()))?;
                let frames = framing
                    .push(&chunk)
                    .map_err(|source| sse_error(mode, source).with_context(context.clone()))?;
                for frame in frames {
                    let events = protocol
                        .decode(frame.data())
                        .map_err(|error| error.with_context(context.clone()))?;
                    for mut event in events {
                        contextualize_terminal_error(&mut event, &context);
                        attach_policy_warnings(&mut event, &warnings);
                        let terminal = event.terminal().is_some();
                        yield event;
                        if terminal {
                            return;
                        }
                    }
                }
            }
            framing
                .finish()
                .map_err(|source| sse_error(mode, source).with_context(context.clone()))?;
            let events = protocol
                .finish()
                .map_err(|error| error.with_context(context.clone()))?;
            for mut event in events {
                contextualize_terminal_error(&mut event, &context);
                attach_policy_warnings(&mut event, &warnings);
                let terminal = event.terminal().is_some();
                yield event;
                if terminal {
                    return;
                }
            }
        }
    })
}

fn contextualize_terminal_error(event: &mut LanguageStreamEvent, context: &ErrorContext) {
    let LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, .. }) = event else {
        return;
    };
    let original = std::mem::replace(
        error,
        Error::new(
            ErrorKind::Internal,
            "stream error context replacement failed",
        ),
    );
    *error = original.with_context(context.clone());
}

fn option_error(mode: OpenAiApiMode, source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        match mode {
            OpenAiApiMode::Responses => "provider options are invalid for the OpenAI Responses API",
            OpenAiApiMode::ChatCompletions => {
                "provider options are invalid for OpenAI Chat Completions"
            }
        },
    )
    .with_source(source)
}

pub(crate) fn request_build_error(
    mode: OpenAiApiMode,
    source: siumai_transport::RequestBuildError,
) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        match mode {
            OpenAiApiMode::Responses => "OpenAI Responses request violates the transport contract",
            OpenAiApiMode::ChatCompletions => {
                "OpenAI Chat Completions request violates the transport contract"
            }
        },
    )
    .with_source(source)
}

fn sse_error(mode: OpenAiApiMode, source: SseFrameError) -> Error {
    let kind = match source {
        SseFrameError::FrameTooLarge
        | SseFrameError::EventTooLarge
        | SseFrameError::TooManyEvents => ErrorKind::ResponseLimit,
        SseFrameError::InvalidUtf8 | SseFrameError::UnexpectedEof => ErrorKind::Protocol,
        _ => ErrorKind::Protocol,
    };
    Error::new(
        kind,
        match mode {
            OpenAiApiMode::Responses => "provider returned an invalid OpenAI Responses SSE stream",
            OpenAiApiMode::ChatCompletions => {
                "provider returned an invalid OpenAI Chat Completions SSE stream"
            }
        },
    )
    .with_source(source)
}

pub(crate) fn response_error(mode: OpenAiApiMode, response: TransportResponse) -> Error {
    response_error_with_message(
        match mode {
            OpenAiApiMode::Responses => "OpenAI rejected the Responses request",
            OpenAiApiMode::ChatCompletions => "OpenAI rejected the Chat Completions request",
        },
        response,
    )
}

pub(crate) fn response_error_with_message(
    message: &'static str,
    response: TransportResponse,
) -> Error {
    let (status, headers, body) = response.into_parts();
    let truncated = body.len() > ERROR_CAPTURE_BYTES;
    let captured = body[..body.len().min(ERROR_CAPTURE_BYTES)].to_vec();
    provider_status_error(message, status, headers, captured, truncated)
}

async fn stream_response_error(mode: OpenAiApiMode, response: TransportStreamResponse) -> Error {
    let (status, headers, mut body) = response.into_parts();
    let mut bytes = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(chunk) => {
                let remaining = ERROR_CAPTURE_BYTES.saturating_sub(bytes.len());
                if remaining == 0 {
                    truncated = true;
                    break;
                }
                bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                if chunk.len() > remaining {
                    truncated = true;
                    break;
                }
            }
            Err(error) => return error,
        }
    }
    provider_status_error(
        match mode {
            OpenAiApiMode::Responses => "OpenAI rejected the Responses request",
            OpenAiApiMode::ChatCompletions => "OpenAI rejected the Chat Completions request",
        },
        status,
        headers,
        bytes,
        truncated,
    )
}

fn provider_status_error(
    message: &'static str,
    status: StatusCode,
    headers: ResponseHeaders,
    body: Vec<u8>,
    body_truncated: bool,
) -> Error {
    let metadata = decode_error_metadata(&body);
    let provider_code = metadata
        .as_ref()
        .and_then(|metadata| metadata.code())
        .and_then(public_provider_identifier);
    let provider_type = metadata
        .as_ref()
        .and_then(|metadata| metadata.error_type())
        .and_then(public_provider_identifier);
    let provider_param = metadata
        .as_ref()
        .and_then(|metadata| metadata.param())
        .and_then(public_provider_identifier);
    let kind = classify_http_error(
        status.as_u16(),
        provider_code.as_ref().map(PublicDiagnosticText::as_str),
        provider_type.as_ref().map(PublicDiagnosticText::as_str),
    );
    let mut diagnostics = headers
        .diagnostics()
        .with_status(status.as_u16())
        .with_body_truncated(body_truncated);
    if let Some(code) = provider_code {
        diagnostics = diagnostics.with_provider_code(code);
    }
    if let Some(kind) = provider_type {
        diagnostics = diagnostics.with_provider_type(kind);
    }
    if let Some(param) = provider_param {
        diagnostics = diagnostics.with_provider_param(param);
    }
    let raw_headers = headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect::<BTreeMap<_, _>>();
    Error::new(kind, message)
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body))
}

fn public_provider_identifier(value: &str) -> Option<PublicDiagnosticText> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return None;
    }
    PublicDiagnosticText::new(value.to_string()).ok()
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use siumai_core::{
        Message, MessageRole, ProviderOptions, ReplayDomain, ReplayDomainId, ToolSpec,
    };
    use siumai_transport::EndpointConfig;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::configured::{
        GPT_5_5, GPT_5_6_SOL, OpenAiChatCompletionsOptions, OpenAiCredential,
        OpenAiFunctionToolOptions, OpenAiPromptCacheBreakpoint, OpenAiPromptCacheMode,
        OpenAiPromptCacheOptions, OpenAiPromptCacheRetention, OpenAiProvider, OpenAiReasoning,
        OpenAiReasoningEffort, OpenAiResponsesOptions, OpenAiResponsesTool, OpenAiTextVerbosity,
    };

    fn provider() -> OpenAiProvider {
        OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:43191/v1").unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("openai-model-test").unwrap(),
            ))
            .build()
            .unwrap()
    }

    fn request() -> LanguageRequest {
        LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
    }

    async fn provider_for(server: &MockServer) -> OpenAiProvider {
        OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("openai-model-test").unwrap(),
            ))
            .build()
            .unwrap()
    }

    fn body_json(plan: &RequestPlan) -> Value {
        match plan.body() {
            RequestBody::Bytes { data, .. } => serde_json::from_slice(data).unwrap(),
            _ => panic!("expected a JSON request body"),
        }
    }

    #[test]
    fn neutral_prompt_uses_distinct_native_routes_and_wire_shapes() {
        let provider = provider();
        let responses = provider.responses(GPT_5_6_SOL).unwrap();
        let chat = provider.chat_completions(GPT_5_6_SOL).unwrap();
        let responses_options = responses
            .runtime
            .merge_options(OpenAiApiMode::Responses, &CallOptions::default())
            .unwrap();
        let chat_options = chat
            .runtime
            .merge_options(OpenAiApiMode::ChatCompletions, &CallOptions::default())
            .unwrap();
        let responses_plan = responses
            .plan(&request(), false, responses_options)
            .unwrap();
        let chat_plan = chat.plan(&request(), false, chat_options).unwrap();
        let responses_body = body_json(&responses_plan);
        let chat_body = body_json(&chat_plan);

        assert_eq!(responses_plan.target().as_str(), "responses");
        assert_eq!(chat_plan.target().as_str(), "chat/completions");
        assert!(responses_body.get("input").is_some());
        assert!(responses_body.get("messages").is_none());
        assert!(chat_body.get("messages").is_some());
        assert!(chat_body.get("input").is_none());
    }

    #[test]
    fn typed_responses_options_shape_cache_breakpoints_and_native_tools() {
        let provider = provider();
        let model = provider.responses(GPT_5_6_SOL).unwrap();
        let typed = OpenAiResponsesOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions {
                mode: Some(OpenAiPromptCacheMode::Explicit),
                ttl: None,
            }),
            prompt_cache_write_candidates: vec![OpenAiPromptCacheBreakpoint::new(0, 0)],
            top_logprobs: Some(5),
            reasoning: Some(OpenAiReasoning::default().with_effort(OpenAiReasoningEffort::None)),
            text_verbosity: Some(OpenAiTextVerbosity::High),
            tools: vec![
                OpenAiResponsesTool::web_search(),
                OpenAiResponsesTool::programmatic_tool_calling(),
            ],
            ..OpenAiResponsesOptions::default()
        };
        let call_options =
            CallOptions::default().with_provider_options(ProviderOptions::typed(&typed).unwrap());
        let merged = model
            .runtime
            .merge_options(OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let plan = model.plan(&request(), false, merged).unwrap();
        let body = body_json(&plan);

        assert_eq!(body["text"]["verbosity"], "high");
        assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
        assert_eq!(body["top_logprobs"], 5);
        assert!(
            body["include"]
                .as_array()
                .unwrap()
                .contains(&json!("message.output_text.logprobs"))
        );
        assert_eq!(
            body["input"][0]["content"][0]["prompt_cache_breakpoint"],
            json!({"mode": "explicit"})
        );
        assert_eq!(body["tools"][0]["type"], "web_search");
        assert_eq!(body["tools"][1]["type"], "programmatic_tool_calling");
        assert!(body.get("native_tools").is_none());
        assert!(body.get("prompt_cache_breakpoints").is_none());
    }

    #[test]
    fn typed_chat_options_shape_explicit_cache_breakpoints() {
        let provider = provider();
        let model = provider.chat_completions(GPT_5_6_SOL).unwrap();
        let typed = OpenAiChatCompletionsOptions::default()
            .with_prompt_cache(OpenAiPromptCacheOptions::explicit_30_minutes())
            .with_prompt_cache_write_candidate(OpenAiPromptCacheBreakpoint::new(0, 0));
        let call_options =
            CallOptions::default().with_provider_options(ProviderOptions::typed(&typed).unwrap());
        let merged = model
            .runtime
            .merge_options(OpenAiApiMode::ChatCompletions, &call_options)
            .unwrap();
        let plan = model.plan(&request(), false, merged).unwrap();
        let body = body_json(&plan);

        assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
        assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
        assert_eq!(
            body["messages"][0]["content"][0]["prompt_cache_breakpoint"],
            json!({"mode": "explicit"})
        );
        assert!(body.get("prompt_cache_breakpoints").is_none());
    }

    #[test]
    fn checked_raw_options_forward_unknown_wire_fields_but_not_model_identity() {
        let provider = provider();
        let model = provider.responses(GPT_5_6_SOL).unwrap();
        let future = ProviderOptions::checked_raw(
            model.provider_id().clone(),
            json!({"future_feature": {"mode": "next"}}),
        )
        .unwrap();
        let call_options = CallOptions::default().with_provider_options(future);
        let merged = model
            .runtime
            .merge_options(OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let plan = model.plan(&request(), false, merged).unwrap();
        let body = body_json(&plan);
        assert_eq!(body["future_feature"], json!({"mode": "next"}));

        let identity_override = ProviderOptions::checked_raw(
            model.provider_id().clone(),
            json!({"model": "gpt-5.6-luna"}),
        )
        .unwrap();
        let result = model.runtime.merge_options(
            OpenAiApiMode::Responses,
            &CallOptions::default().with_provider_options(identity_override),
        );
        assert!(
            matches!(result, Err(ProviderOptionError::Rejected { path, .. }) if path == "model")
        );

        let background =
            ProviderOptions::checked_raw(model.provider_id().clone(), json!({"background": true}))
                .unwrap();
        let result = model.runtime.merge_options(
            OpenAiApiMode::Responses,
            &CallOptions::default().with_provider_options(background),
        );
        assert!(
            matches!(result, Err(ProviderOptionError::Rejected { path, .. }) if path == "background")
        );
    }

    #[test]
    fn explicit_max_reasoning_survives_future_model_ids_without_implicit_summary() {
        let provider = provider();
        let responses = provider.responses("private-reasoning-model").unwrap();
        let responses_options = OpenAiResponsesOptions::default()
            .with_reasoning(OpenAiReasoning::default().with_effort(OpenAiReasoningEffort::Max));
        let responses_call = CallOptions::default()
            .with_provider_options(ProviderOptions::typed(&responses_options).unwrap());
        let mut merged = responses
            .runtime
            .merge_options(OpenAiApiMode::Responses, &responses_call)
            .unwrap();
        let (normalized, _) = normalize_request(
            OpenAiApiMode::Responses,
            responses.model_id(),
            request(),
            &mut merged,
        )
        .unwrap();
        let responses_body = body_json(&responses.plan(&normalized, false, merged).unwrap());
        assert_eq!(responses_body["reasoning"]["effort"], "max");
        assert!(responses_body["reasoning"].get("summary").is_none());

        let chat = provider
            .chat_completions("private-reasoning-model")
            .unwrap();
        let chat_options = OpenAiChatCompletionsOptions {
            reasoning_effort: Some(OpenAiReasoningEffort::Max),
            ..OpenAiChatCompletionsOptions::default()
        };
        let chat_call = CallOptions::default()
            .with_provider_options(ProviderOptions::typed(&chat_options).unwrap());
        let mut merged = chat
            .runtime
            .merge_options(OpenAiApiMode::ChatCompletions, &chat_call)
            .unwrap();
        let (normalized, _) = normalize_request(
            OpenAiApiMode::ChatCompletions,
            chat.model_id(),
            request(),
            &mut merged,
        )
        .unwrap();
        let chat_body = body_json(&chat.plan(&normalized, false, merged).unwrap());
        assert_eq!(chat_body["reasoning_effort"], "max");
    }

    #[test]
    fn known_models_enforce_prompt_cache_generation_rules() {
        let provider = provider();
        let model = provider.responses(GPT_5_6_SOL).unwrap();
        let typed = OpenAiResponsesOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions::explicit_30_minutes()),
            prompt_cache_retention: Some(OpenAiPromptCacheRetention::TwentyFourHours),
            ..OpenAiResponsesOptions::default()
        };
        let call_options =
            CallOptions::default().with_provider_options(ProviderOptions::typed(&typed).unwrap());
        let mut merged = model
            .runtime
            .merge_options(OpenAiApiMode::Responses, &call_options)
            .unwrap();

        normalize_request(
            OpenAiApiMode::Responses,
            model.model_id(),
            request(),
            &mut merged,
        )
        .unwrap();

        let invalid_retention = OpenAiResponsesOptions {
            prompt_cache_retention: Some(OpenAiPromptCacheRetention::InMemory),
            ..OpenAiResponsesOptions::default()
        };
        let call_options = CallOptions::default()
            .with_provider_options(ProviderOptions::typed(&invalid_retention).unwrap());
        let mut merged = model
            .runtime
            .merge_options(OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let error = normalize_request(
            OpenAiApiMode::Responses,
            model.model_id(),
            request(),
            &mut merged,
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);

        let gpt_5_5 = provider.responses(GPT_5_5).unwrap();
        let unsupported = OpenAiResponsesOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions::explicit()),
            prompt_cache_write_candidates: vec![OpenAiPromptCacheBreakpoint::new(0, 0)],
            ..OpenAiResponsesOptions::default()
        };
        let call_options = CallOptions::default()
            .with_provider_options(ProviderOptions::typed(&unsupported).unwrap());
        let mut merged = gpt_5_5
            .runtime
            .merge_options(OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let error = normalize_request(
            OpenAiApiMode::Responses,
            gpt_5_5.model_id(),
            request(),
            &mut merged,
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn programmatic_tool_calling_can_invoke_typed_function_tools() {
        let provider = provider();
        let model = provider.responses(GPT_5_6_SOL).unwrap();
        let mut request = request();
        request.tools.push(
            ToolSpec::new(
                "get_inventory",
                Some("Read inventory".to_string()),
                json!({"type": "object"}),
            )
            .unwrap(),
        );
        let typed = OpenAiResponsesOptions::default()
            .with_tool(OpenAiResponsesTool::programmatic_tool_calling())
            .with_function_tool_options(
                "get_inventory",
                OpenAiFunctionToolOptions::programmatic()
                    .with_strict(true)
                    .with_output_schema(json!({"type": "object"})),
            );
        let call_options =
            CallOptions::default().with_provider_options(ProviderOptions::typed(&typed).unwrap());
        let merged = model
            .runtime
            .merge_options(OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let plan = model.plan(&request, false, merged).unwrap();
        let body = body_json(&plan);

        assert_eq!(body["tools"][0]["name"], "get_inventory");
        assert_eq!(body["tools"][0]["allowed_callers"], json!(["programmatic"]));
        assert_eq!(body["tools"][0]["strict"], true);
        assert_eq!(body["tools"][0]["output_schema"], json!({"type": "object"}));
        assert_eq!(body["tools"][1]["type"], "programmatic_tool_calling");
        assert!(body.get("function_tool_options").is_none());
    }

    #[test]
    fn mode_descriptors_use_protocol_specific_provenance() {
        let provider = provider();
        let responses = provider.responses(GPT_5_6_SOL).unwrap();
        let chat = provider.chat_completions(GPT_5_6_SOL).unwrap();

        assert_eq!(responses.descriptor().protocol(), Some("openai.responses"));
        assert_eq!(chat.descriptor().protocol(), Some("openai"));
    }

    #[tokio::test]
    async fn background_creation_preserves_non_terminal_native_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_background",
                "created_at": 1,
                "model": "gpt-5.6-sol",
                "status": "queued",
                "output": []
            })))
            .expect(1)
            .mount(&server)
            .await;

        let model = provider_for(&server).await.responses(GPT_5_6_SOL).unwrap();
        let background = model
            .create_background(request(), CallOptions::default())
            .await
            .unwrap();
        assert_eq!(background.resource().status.as_str(), "queued");

        let requests = server.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["background"], true);
        assert_eq!(body["model"], GPT_5_6_SOL);
    }

    #[test]
    fn gpt_5_6_omits_non_reasoning_controls_unless_effort_is_none() {
        let provider = provider();
        let model = provider.responses(GPT_5_6_SOL).unwrap();
        let mut request = request();
        request.generation.temperature = Some(0.7);
        request.generation.top_p = Some(0.9);
        request.generation.seed = Some(42);
        request.generation.stop_sequences = vec!["stop".to_string()];
        let typed = OpenAiResponsesOptions {
            top_logprobs: Some(5),
            ..OpenAiResponsesOptions::default()
        };
        let call_options =
            CallOptions::default().with_provider_options(ProviderOptions::typed(&typed).unwrap());
        let mut merged = model
            .runtime
            .merge_options(OpenAiApiMode::Responses, &call_options)
            .unwrap();

        let (request, warnings) = normalize_request(
            OpenAiApiMode::Responses,
            model.model_id(),
            request,
            &mut merged,
        )
        .unwrap();

        assert_eq!(request.generation.temperature, None);
        assert_eq!(request.generation.top_p, None);
        assert_eq!(request.generation.seed, None);
        assert!(request.generation.stop_sequences.is_empty());
        assert!(!merged.wire.contains_key("top_logprobs"));
        assert!(warnings.len() >= 5);
    }
}
