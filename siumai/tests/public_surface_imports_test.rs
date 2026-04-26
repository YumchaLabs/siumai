use std::mem::size_of;

#[test]
#[allow(deprecated)]
fn public_surface_unified_imports_compile() {
    use siumai::prelude::unified::*;

    let _ = size_of::<AISDKError>();
    let _ = size_of::<APICallError>();
    let _ = size_of::<ChatRequest>();
    let _ = size_of::<ChatInit>();
    let _ = size_of::<ChatRequestOptions>();
    let _ = size_of::<ChatResponse>();
    let _ = size_of::<ChatState>();
    let _ = size_of::<ChatStatus>();
    let _ = size_of::<ChatTransportReconnectToStreamOptions>();
    let _ = size_of::<ChatTransportSendMessagesOptions>();
    let _ = size_of::<ChatTransportTrigger>();
    let _ = size_of::<ResponseFormat>();
    let _ = size_of::<CallSettings>();
    let _ = size_of::<JSONSchema7>();
    let _ = size_of::<JSONValue>();
    let _ = size_of::<Schema>();
    let _ = size_of::<LazySchema>();
    let _ = size_of::<FlexibleSchema>();
    let _ = size_of::<Arrayable<String>>();
    let _ = size_of::<SerialJobExecutor>();
    let _ = size_of::<HeaderRecord>();
    let _ = size_of::<JsonInstructionOptions>();
    let _ = size_of::<JsonInstructionMessageOptions>();
    let _ = size_of::<JsonParseResult>();
    let _ = size_of::<LoadApiKeyOptions>();
    let _ = size_of::<LoadOptionalSettingOptions>();
    let _ = size_of::<LoadSettingOptions>();
    let _ = size_of::<ReasoningBudgetOptions<'static>>();
    let _ = size_of::<ReasoningLevel>();
    let _ = size_of::<ReasoningLevelConversionError>();
    let _ = size_of::<StreamingToolCallDelta>();
    let _ = size_of::<StreamingToolCallFunctionDelta>();
    let _ = size_of::<StreamingToolCallTracker>();
    let _ = size_of::<StreamingToolCallTrackerOptions>();
    let _ = size_of::<StreamingToolCallTypeValidation>();
    let _ = size_of::<ToolNameMapping>();
    let _ = size_of::<TypeValidationResult>();
    let _ = size_of::<ValidationResult>();
    let _ = size_of::<ModelCallResponseData>();
    let _ = size_of::<GenerateObjectOptions>();
    let _ = size_of::<GenerateObjectResult<JSONValue>>();
    let _ = size_of::<GenerateObjectSchema<JSONValue>>();
    let _ = size_of::<GenerateObjectEndEvent<JSONValue>>();
    let _ = size_of::<GenerateObjectOutputStrategy>();
    let _ = size_of::<GenerateObjectResponseMetadata>();
    let _ = size_of::<GenerateObjectStartEvent>();
    let _ = size_of::<GenerateObjectStepEndEvent>();
    let _ = size_of::<GenerateObjectStepStartEvent>();
    let _ = size_of::<ObjectStreamErrorPart>();
    let _ = size_of::<ObjectStreamFinishPart>();
    let _ = size_of::<ObjectStreamObjectPart<JSONValue>>();
    let _ = size_of::<ObjectStreamPart<JSONValue>>();
    let _ = size_of::<ObjectStreamTextDeltaPart>();
    let _ = size_of::<PartialJsonParseResult>();
    let _ = size_of::<PartialJsonParseState>();
    let _ = size_of::<PartialJsonValueStream>();
    let _ = size_of::<PartialJsonValueStreamEvent>();
    let _ = size_of::<RepairTextContext>();
    let _ = size_of::<RepairTextFunction>();
    let _ = size_of::<RepairTextFuture>();
    let _ = size_of::<IdGenerator>();
    let _ = size_of::<IdGeneratorOptions>();
    let _ = size_of::<ExecutableTool>();
    let _ = size_of::<ExecutableTools>();
    let _ = size_of::<ToolExecuteFunction>();
    let _ = size_of::<ToolExecutionOptions>();
    let _ = size_of::<ToolExecutionResult>();
    let _ = size_of::<ToolExecutionStream>();
    let _ = size_of::<ToolModelOutputContext>();
    let _ = size_of::<ToolSet>();
    let _ = size_of::<UITool>();
    let _ = size_of::<UIToolInvocation>();
    let _ = size_of::<UITools>();
    let _ = size_of::<CallWarning>();
    let _ = size_of::<CancelHandle>();
    let _ = size_of::<CallbackModelInfo>();
    let _ = size_of::<Context>();
    let _ = size_of::<CreateUIMessage>();
    let _ = size_of::<DataContent>();
    let _ = size_of::<DefaultGeneratedAudioFile>();
    let _ = size_of::<DefaultGeneratedAudioFileWithType>();
    let _ = size_of::<DefaultGeneratedFile>();
    let _ = size_of::<DefaultGeneratedFileWithType>();
    let _ = size_of::<DefaultStepResult>();
    let _ = size_of::<CompletionRequest>();
    let _ = size_of::<CompletionRequestOptions>();
    let _ = size_of::<CompletionResponse>();
    let _ = size_of::<CompletionStreamProtocol>();
    let _ = size_of::<Download>();
    let _ = size_of::<DownloadError>();
    let _ = size_of::<DownloadedFile>();
    let _ = size_of::<DownloadOptions>();
    let _ = size_of::<DynamicToolCall>();
    let _ = size_of::<DynamicToolError>();
    let _ = size_of::<DynamicToolResult>();
    let _ = size_of::<EmbedEndEvent>();
    let _ = size_of::<EmbedManyResult>();
    let _ = size_of::<EmbedOutput>();
    let _ = size_of::<EmbedResponseData>();
    let _ = size_of::<EmbedResult>();
    let _ = size_of::<EmbedStartEvent>();
    let _ = size_of::<EmbedValue>();
    let _ = size_of::<Embedding>();
    let _ = size_of::<EmbeddingModelCallEndEvent>();
    let _ = size_of::<EmbeddingModelCallStartEvent>();
    let _ = size_of::<EmbeddingModelUsage>();
    let _ = size_of::<EmptyResponseBodyError>();
    let _ = size_of::<Experimental_GenerateImageResult>();
    let _ = size_of::<Experimental_GeneratedImage>();
    let _ = size_of::<Experimental_SpeechResult>();
    let _ = size_of::<Experimental_TranscriptionResult>();
    let _ = size_of::<TextPart>();
    let _ = size_of::<TextUIPart>();
    let _ = size_of::<ImagePart>();
    let _ = size_of::<FilePart>();
    let _ = size_of::<FileUIPart>();
    let _ = size_of::<ReasoningPart>();
    let _ = size_of::<ReasoningUIPart>();
    let _ = size_of::<CustomPart>();
    let _ = size_of::<CustomContentUIPart>();
    let _ = size_of::<ReasoningFilePart>();
    let _ = size_of::<ReasoningFileUIPart>();
    let _ = size_of::<ToolCallPart>();
    let _ = size_of::<ToolUIPart>();
    let _ = size_of::<DynamicToolUIPart>();
    let _ = size_of::<ToolResultPart>();
    let _ = size_of::<ToolApprovalRequest>();
    let _ = size_of::<ToolApprovalRequestOutput<String, JSONValue>>();
    let _ = size_of::<ToolApprovalResponse>();
    let _ = size_of::<ToolApprovalResponseOutput<String, JSONValue>>();
    let _ = size_of::<ToolError<String, JSONValue>>();
    let _ = size_of::<ToolOutput<String, JSONValue, ToolResultOutput>>();
    let _ = size_of::<ToolOutputDenied<String>>();
    let _ = size_of::<StaticToolOutputDenied<String>>();
    let _ = size_of::<TypedToolOutputDenied<String>>();
    let _ = size_of::<UserContentPart>();
    let _ = size_of::<AssistantContentPart>();
    let _ = size_of::<ToolContentPart>();
    let _ = size_of::<UserContent>();
    let _ = size_of::<AssistantContent>();
    let _ = size_of::<ToolContent>();
    let _ = size_of::<GenerateImageRequest>();
    let _ = size_of::<GenerateImageResult>();
    let _ = size_of::<GenerateVideoResult>();
    let _ = size_of::<GenerateTextContentPart>();
    let _ = size_of::<GenerateTextEndEvent>();
    let _ = size_of::<GenerateTextModelInfo>();
    let _ = size_of::<GenerateTextReasoningPart>();
    let _ = size_of::<GenerateTextResponseMetadata>();
    let _ = size_of::<GenerateTextResult<JSONValue>>();
    let _ = size_of::<GenerateTextStartEvent>();
    let _ = size_of::<GenerateTextStepEndEvent>();
    let _ = size_of::<GenerateTextStepReasoningPart>();
    let _ = size_of::<GenerateTextStepResult>();
    let _ = size_of::<GenerateTextStepStartEvent>();
    let _ = size_of::<ResponseMessage>();
    let _ = size_of::<RetryError>();
    let _ = size_of::<RetryErrorReason>();
    let _ = size_of::<TextStreamAbortPart>();
    let _ = size_of::<TextStreamCustomPart>();
    let _ = size_of::<TextStreamErrorPart>();
    let _ = size_of::<TextStreamFilePart>();
    let _ = size_of::<TextStreamFinishPart>();
    let _ = size_of::<TextStreamFinishStepPart>();
    let _ = size_of::<TextStreamPart>();
    let _ = size_of::<TextStreamRawPart>();
    let _ = size_of::<TextStreamReasoningDeltaPart>();
    let _ = size_of::<TextStreamReasoningEndPart>();
    let _ = size_of::<TextStreamReasoningFilePart>();
    let _ = size_of::<TextStreamReasoningStartPart>();
    let _ = size_of::<TextStreamStartPart>();
    let _ = size_of::<TextStreamStartStepPart>();
    let _ = size_of::<TextStreamSourcePart>();
    let _ = size_of::<TextStreamTextDeltaPart>();
    let _ = size_of::<TextStreamTextEndPart>();
    let _ = size_of::<TextStreamTextStartPart>();
    let _ = size_of::<TextStreamToolApprovalRequestPart>();
    let _ = size_of::<TextStreamToolApprovalResponsePart>();
    let _ = size_of::<TextStreamToolCallPart>();
    let _ = size_of::<TextStreamToolErrorPart>();
    let _ = size_of::<TextStreamToolInputDeltaPart>();
    let _ = size_of::<TextStreamToolInputEndPart>();
    let _ = size_of::<TextStreamToolInputStartPart>();
    let _ = size_of::<TextStreamToolOutputDeniedPart>();
    let _ = size_of::<TextStreamToolResultPart>();
    let _ = size_of::<TextOutput>();
    let _ = size_of::<CustomOutput>();
    let _ = size_of::<FileOutput>();
    let _ = size_of::<GeneratedAudioFile>();
    let _ = size_of::<GeneratedFile>();
    let _ = size_of::<GenerateImagePrompt>();
    let _ = size_of::<ImageModelProviderMetadata>();
    let _ = size_of::<ImageModelResponseMetadata>();
    let _ = size_of::<ImageModelUsage>();
    let _ = size_of::<HttpChatTransportInitOptions>();
    let _ = size_of::<InferUIDataParts>();
    let _ = size_of::<InferUIMessageChunk>();
    let _ = size_of::<InferUIMessageData>();
    let _ = size_of::<InferUIMessageMetadata>();
    let _ = size_of::<InferUIMessagePart>();
    let _ = size_of::<InferUIMessageToolCall>();
    let _ = size_of::<InferUIMessageToolOutputs>();
    let _ = size_of::<InferUIMessageTools>();
    let _ = size_of::<InvalidArgumentError>();
    let _ = size_of::<InvalidMessageRoleError>();
    let _ = size_of::<InvalidPromptError>();
    let _ = size_of::<InvalidResponseDataError>();
    let _ = size_of::<InvalidStreamPartError>();
    let _ = size_of::<InvalidToolApprovalError>();
    let _ = size_of::<JSONParseError>();
    let _ = size_of::<LanguageModelCallOptions>();
    let _ = size_of::<LanguageModelInputTokenDetails>();
    let _ = size_of::<LanguageModelOutputTokenDetails>();
    let _ = size_of::<LanguageModelReasoning>();
    let _ = size_of::<LanguageModelRequestMetadata>();
    let _ = size_of::<LanguageModelResponseMetadata>();
    let _ = size_of::<LanguageModelStreamModelCallEndPart>();
    let _ = size_of::<LanguageModelStreamModelCallResponseMetadataPart>();
    let _ = size_of::<LanguageModelStreamModelCallStartPart>();
    let _ = size_of::<LanguageModelStreamPart>();
    let _ = size_of::<ExperimentalLanguageModelStreamPart>();
    let _ = size_of::<Experimental_LanguageModelStreamPart>();
    let _ = size_of::<LanguageModelUsage>();
    let _ = size_of::<LoadAPIKeyError>();
    let _ = size_of::<LoadSettingError>();
    let _ = size_of::<ProviderMetadata>();
    let _ = size_of::<ProviderOptions>();
    let _ = size_of::<ProviderReference>();
    let _ = size_of::<PrepareReconnectToStreamRequestOptions>();
    let _ = size_of::<PrepareSendMessagesRequestOptions>();
    let _ = size_of::<PrepareStepOptions>();
    let _ = size_of::<PrepareStepResult>();
    let _ = size_of::<PreparedReconnectToStreamRequest>();
    let _ = size_of::<PreparedSendMessagesRequest>();
    let _ = size_of::<ReasoningFileOutput>();
    let _ = size_of::<ReasoningOutput>();
    let _ = size_of::<RequestCredentials>();
    let _ = size_of::<RequestOptions>();
    let _ = size_of::<RerankEndEvent>();
    let _ = size_of::<RerankRanking>();
    let _ = size_of::<RerankRankingEntry>();
    let _ = size_of::<RerankResponseMetadata>();
    let _ = size_of::<RerankResult>();
    let _ = size_of::<RerankStartEvent>();
    let _ = size_of::<RerankingModelCallEndEvent>();
    let _ = size_of::<RerankingModelCallRanking>();
    let _ = size_of::<RerankingModelCallStartEvent>();
    let _ = size_of::<Source>();
    let _ = size_of::<SourceDocumentUIPart>();
    let _ = size_of::<SourceUrlUIPart>();
    let _ = size_of::<StopCondition>();
    let _ = size_of::<StepStartUIPart>();
    let _ = size_of::<StreamRequestOptions>();
    let _ = size_of::<StreamTextChunk>();
    let _ = size_of::<StreamTextChunkEvent>();
    let _ = size_of::<StreamTextLifecycleChunk>();
    let _ = size_of::<StreamTextLifecycleChunkType>();
    let _ = size_of::<SpeechModelResponseMetadata>();
    let _ = size_of::<SpeechResult>();
    let _ = size_of::<TelemetryOptions>();
    let _ = size_of::<StaticToolCall>();
    let _ = size_of::<StaticToolError>();
    let _ = size_of::<StaticToolResult>();
    let _ = size_of::<StepResult>();
    let _ = size_of::<ModelMessageRole>();
    let _ = size_of::<SystemModelMessage>();
    let _ = size_of::<UserModelMessage>();
    let _ = size_of::<AssistantModelMessage>();
    let _ = size_of::<ToolModelMessage>();
    let _ = size_of::<ModelMessage>();
    let _ = size_of::<PromptInput>();
    let _ = size_of::<SystemPrompt>();
    let _ = size_of::<Prompt>();
    let _ = size_of::<StandardizedPrompt>();
    let _ = size_of::<PruneEmptyMessagesMode>();
    let _ = size_of::<PruneMessagesOptions>();
    let _ = size_of::<PruneReasoningMode>();
    let _ = size_of::<PruneToolCallMode>();
    let _ = size_of::<PruneToolCallRule>();
    let _ = size_of::<TimeoutConfiguration>();
    let _ = size_of::<TimeoutConfigurationSettings>();
    let _ = size_of::<TooManyEmbeddingValuesForCallError>();
    let _ = size_of::<TranscriptionModelResponseMetadata>();
    let _ = size_of::<TranscriptionResult>();
    let _ = size_of::<TranscriptionSegment>();
    let _ = size_of::<TypeValidationContext>();
    let _ = size_of::<TypeValidationError>();
    let _ = size_of::<VideoModelProviderMetadata>();
    let _ = size_of::<VideoModelResponseMetadata>();
    let _ = size_of::<ProviderOptionsMap>();
    let _ = size_of::<InvalidDataContentError>();
    let _ = size_of::<InvalidToolInputError>();
    let _ = size_of::<MissingToolResultsError>();
    let _ = size_of::<MessageConversionError>();
    let _ = size_of::<ModelMessageConversionError>();
    let _ = size_of::<NoContentGeneratedError>();
    let _ = size_of::<NoImageGeneratedError>();
    let _ = size_of::<NoObjectGeneratedError>();
    let _ = size_of::<NoOutputGeneratedError>();
    let _ = size_of::<NoSpeechGeneratedError>();
    let _ = size_of::<NoSuchModelError>();
    let _ = size_of::<NoSuchModelType>();
    let _ = size_of::<NoSuchProviderError>();
    let _ = size_of::<NoSuchProviderReferenceError>();
    let _ = size_of::<NoSuchToolError>();
    let _ = size_of::<NoTranscriptGeneratedError>();
    let _ = size_of::<NoVideoGeneratedError>();
    let _ = size_of::<PromptExecutionError>();
    let _ = size_of::<PromptValidationError>();
    let _ = size_of::<ToolCall<String, JSONValue>>();
    let _ = size_of::<ToolApprovalConfiguration>();
    let _ = size_of::<ToolApprovalDecisionContext>();
    let _ = size_of::<ToolApprovalStatus>();
    let _ = size_of::<ToolApprovalStatusDetails>();
    let _ = size_of::<ToolApprovalStatusType>();
    let _ = size_of::<ToolCallNotFoundForApprovalError>();
    let _ = size_of::<ToolCallRepairContext>();
    let _ = size_of::<ToolCallRepairError>();
    let _ = size_of::<ToolCallRepairFunctionError>();
    let _ = size_of::<ToolCallRepairResult>();
    let _ = size_of::<ToolExecutionEndEvent>();
    let _ = size_of::<ToolExecutionStartEvent>();
    let _ = size_of::<ToolResult<String, JSONValue, ToolResultOutput>>();
    let _ = size_of::<TypedToolCall>();
    let _ = size_of::<TypedToolError>();
    let _ = size_of::<TypedToolResult>();
    let _ = size_of::<UIDataPartSchemas>();
    let _ = size_of::<UIDataTypes>();
    let _ = size_of::<UIDataTypesToSchemas>();
    let _ = size_of::<DataUIMessageChunk>();
    let _ = size_of::<UIMessage>();
    let _ = size_of::<UIMessageChunk>();
    let _ = size_of::<UIMessagePart>();
    let _ = size_of::<UIMessageStreamError>();
    let _ = size_of::<UIMessageStreamOptions>();
    let _ = size_of::<UiCustomPart>();
    let _ = size_of::<UiDataPart>();
    let _ = size_of::<UiFilePart>();
    let _ = size_of::<UiMessage>();
    let _ = size_of::<UiMessageAbortChunk>();
    let _ = size_of::<UiMessageChunk<JSONValue, JSONValue>>();
    let _ = size_of::<UiMessageCustomChunk>();
    let _ = size_of::<UiMessageDataChunk<JSONValue>>();
    let _ = size_of::<UiMessageErrorChunk>();
    let _ = size_of::<UiMessageFileChunk>();
    let _ = size_of::<UiMessageFinishChunk<JSONValue>>();
    let _ = size_of::<UiMessageFinishStepChunk>();
    let _ = size_of::<UiMessageMetadataChunk<JSONValue>>();
    let _ = size_of::<UiMessagePart>();
    let _ = size_of::<UiMessageReasoningDeltaChunk>();
    let _ = size_of::<UiMessageReasoningEndChunk>();
    let _ = size_of::<UiMessageReasoningFileChunk>();
    let _ = size_of::<UiMessageReasoningStartChunk>();
    let _ = size_of::<UiMessageRole>();
    let _ = size_of::<UiMessageSourceDocumentChunk>();
    let _ = size_of::<UiMessageSourceUrlChunk>();
    let _ = size_of::<UiMessageStartChunk<JSONValue>>();
    let _ = size_of::<UiMessageStartStepChunk>();
    let _ = size_of::<UiMessageTextDeltaChunk>();
    let _ = size_of::<UiMessageTextEndChunk>();
    let _ = size_of::<UiMessageTextStartChunk>();
    let _ = size_of::<UiMessageToolApprovalRequestChunk>();
    let _ = size_of::<UiMessageToolApprovalResponseChunk>();
    let _ = size_of::<UiMessageToolInputAvailableChunk<JSONValue>>();
    let _ = size_of::<UiMessageToolInputDeltaChunk>();
    let _ = size_of::<UiMessageToolInputErrorChunk<JSONValue>>();
    let _ = size_of::<UiMessageToolInputStartChunk>();
    let _ = size_of::<UiMessageToolOutputAvailableChunk<JSONValue>>();
    let _ = size_of::<UiMessageToolOutputDeniedChunk>();
    let _ = size_of::<UiMessageToolOutputErrorChunk>();
    let _ = size_of::<UiMessageWithoutId>();
    let _ = size_of::<UiMessageStreamOptions>();
    let _ = size_of::<UiPartState>();
    let _ = size_of::<UiProviderMetadata>();
    let _ = size_of::<UiReasoningFilePart>();
    let _ = size_of::<UiReasoningPart>();
    let _ = size_of::<UiSourceDocumentPart>();
    let _ = size_of::<UiSourceUrlPart>();
    let _ = size_of::<UiTextPart>();
    let _ = size_of::<UiToolApproval>();
    let _ = size_of::<UiToolApprovalDecision>();
    let _ = size_of::<UiToolApprovalRequest>();
    let _ = size_of::<UiToolApprovedApproval>();
    let _ = size_of::<UiToolDeniedApproval>();
    let _ = size_of::<UiToolInvocation>();
    let _ = size_of::<UiToolInvocationState>();
    let _ = size_of::<UiToolKind>();
    let _ = size_of::<UiToolPart>();
    let _ = size_of::<UiToolPartState>();
    let _ = size_of::<UnsupportedFunctionalityError>();
    let _ = size_of::<UnsupportedModelVersionError>();
    let _ = size_of::<UseCompletionOptions>();
    let _ = size_of::<OnStartEvent>();
    let _ = size_of::<OnStepStartEvent>();
    let _ = size_of::<OnChunkEvent>();
    let _ = size_of::<OnStepFinishEvent>();
    let _ = size_of::<OnFinishEvent>();
    let _ = size_of::<OnToolCallStartEvent>();
    let _ = size_of::<OnToolCallFinishEvent>();
    let _ = size_of::<LlmError>();
    let _ = size_of::<*const dyn CompletionCapability>();
    let _ = size_of::<*const dyn CompletionModel>();
    let _ = size_of::<*const dyn ProviderFactory>();
    let _ = size_of::<*const dyn ImageModel>();
    let _ = size_of::<*const dyn ImageModelV4>();
    let _ = size_of::<*const dyn EmbeddingModel>();
    let _ = size_of::<*const dyn LanguageModel>();
    let _ = size_of::<*const dyn LanguageModelMiddleware>();
    let _ = size_of::<*const dyn RerankingModel>();
    let _ = size_of::<*const dyn SpeechModel>();
    let _ = size_of::<*const dyn TranscriptionModel>();
    let _ = size_of::<*const dyn VideoModel>();
    let _ = size_of::<*const dyn VideoModelV3>();
    let _ = size_of::<*const dyn VideoModelV4>();
    let _ = get_total_timeout_ms as fn(Option<&TimeoutConfiguration>) -> Option<u64>;
    let _ = get_step_timeout_ms as fn(Option<&TimeoutConfiguration>) -> Option<u64>;
    let _ = get_chunk_timeout_ms as fn(Option<&TimeoutConfiguration>) -> Option<u64>;
    let _ = get_tool_timeout_ms as fn(Option<&TimeoutConfiguration>, &str) -> Option<u64>;
    let _ = is_step_count as fn(usize) -> StopCondition;
    let _ = is_loop_finished as fn() -> StopCondition;
    let _ = is_stop_condition_met::<String, JSONValue, ToolResultOutput>
        as fn(&[StopCondition], &[GenerateTextStepResult]) -> bool;
    let _ = has_tool_call(["search"]);
    let _ = filter_active_tools::<String>
        as fn(Option<&[Tool]>, Option<&[String]>) -> Option<Vec<Tool>>;
    let _ = experimental_filter_active_tools::<String>
        as fn(Option<&[Tool]>, Option<&[String]>) -> Option<Vec<Tool>>;
    let _ = step_count_is as fn(usize) -> StopCondition;
    let _ = prune_messages as fn(Vec<ModelMessage>, PruneMessagesOptions) -> Vec<ModelMessage>;
    let _ = create_null_language_model_usage as fn() -> LanguageModelUsage;
    let _ = add_language_model_usage
        as fn(&LanguageModelUsage, &LanguageModelUsage) -> LanguageModelUsage;
    let _ = add_image_model_usage as fn(&ImageModelUsage, &ImageModelUsage) -> ImageModelUsage;
    let _ = as_language_model_usage as fn(&Usage) -> LanguageModelUsage;
    let _ = convert_data_content_to_base64_string as fn(&DataContent) -> String;
    let _ = convert_data_content_to_uint8_array
        as fn(&DataContent) -> Result<Vec<u8>, InvalidDataContentError>;
    let _ = convert_base64_to_uint8_array as fn(&str) -> Result<Vec<u8>, LlmError>;
    let _ = convert_uint8_array_to_base64 as fn(&[u8]) -> String;
    let _ = convert_to_base64 as fn(&DataContent) -> String;
    let _ = convert_uint8_array_to_text as fn(&[u8]) -> String;
    let _ = cosine_similarity::<f32, f32> as fn(&[f32], &[f32]) -> Result<f64, LlmError>;
    let _ = DEFAULT_MAX_DOWNLOAD_SIZE;
    let _ = DEFAULT_REASONING_BUDGET_PERCENTAGES;
    let _ = create_download as fn(DownloadOptions) -> Download;
    let _ = validate_download_url as fn(&str) -> Result<(), DownloadError>;
    let _ = get_text_from_data_url as fn(&str) -> Result<String, LlmError>;
    let _ = convert_image_model_file_to_data_uri
        as fn(&siumai::types::ImageEditInput) -> Result<String, LlmError>;
    let _ = is_deep_equal_data as fn(&JSONValue, &JSONValue) -> bool;
    let _ = is_text_ui_part as fn(&UiMessagePart) -> bool;
    let _ = is_custom_content_ui_part as fn(&UiMessagePart) -> bool;
    let _ = is_file_ui_part as fn(&UiMessagePart) -> bool;
    let _ = is_reasoning_file_ui_part as fn(&UiMessagePart) -> bool;
    let _ = is_reasoning_ui_part as fn(&UiMessagePart) -> bool;
    let _ = is_data_ui_part as fn(&UiMessagePart) -> bool;
    let _ = is_data_ui_message_chunk as fn(&UiMessageChunk) -> bool;
    let _ = is_static_tool_ui_part as fn(&UiMessagePart) -> bool;
    let _ = is_dynamic_tool_ui_part as fn(&UiMessagePart) -> bool;
    let _ = is_tool_ui_part as fn(&UiMessagePart) -> bool;
    let _ = get_static_tool_name as for<'a> fn(&'a UiMessagePart) -> Option<&'a str>;
    let _ = get_tool_name as for<'a> fn(&'a UiMessagePart) -> Option<&'a str>;
    let _ = get_tool_or_dynamic_tool_name as for<'a> fn(&'a UiMessagePart) -> Option<&'a str>;
    let _ = last_assistant_message_is_complete_with_tool_calls as fn(&[UiMessage]) -> bool;
    let _ = last_assistant_message_is_complete_with_approval_responses as fn(&[UiMessage]) -> bool;
    let _ = generate_text::<dyn LanguageModel>;
    let _ = experimental_generate_speech::<dyn SpeechModel>;
    let _ = experimental_generate_video::<dyn VideoModelV4>;
    let _ = experimental_transcribe::<dyn TranscriptionModel>;
    assert!(
        UI_MESSAGE_STREAM_HEADERS
            .iter()
            .any(|(name, value)| *name == "x-vercel-ai-ui-message-stream" && *value == "v1")
    );

    let schema = json_schema(serde_json::json!({ "type": "object" }));
    assert_eq!(schema.json_schema()["type"], serde_json::json!("object"));
    assert_eq!(
        serde_json::to_value(ResponseFormat::json_object().with_name("payload"))
            .expect("serialize response format"),
        serde_json::json!({ "type": "json", "name": "payload" })
    );
    assert_eq!(fix_partial_json(r#"{"value":"ok""#), r#"{"value":"ok"}"#);
    assert_eq!(
        parse_partial_json(Some(r#"{"value":"ok""#)).state,
        PartialJsonParseState::RepairedParse
    );
    let _ = partial_json_value_stream as fn(ChatStream) -> PartialJsonValueStream;
    let normalized = normalize_headers([("X-Test", "1")]);
    assert_eq!(normalized.get("x-test").map(String::as_str), Some("1"));
    let optional = normalize_optional_headers([("X-Keep", Some("yes")), ("X-Drop", None)]);
    assert_eq!(optional.get("x-keep").map(String::as_str), Some("yes"));
    assert!(!optional.contains_key("x-drop"));
    let combined = combine_headers([
        normalized.clone(),
        HeaderRecord::from([("X-Test".to_string(), "2".to_string())]),
    ]);
    assert_eq!(combined.get("x-test").map(String::as_str), Some("2"));
    let headers = with_user_agent_suffix([("User-Agent", "siumai/test")], ["ai-sdk/test"]);
    assert_eq!(
        headers.get("user-agent").map(String::as_str),
        Some("siumai/test ai-sdk/test")
    );
    let _ = normalize_header_map as fn(&reqwest::header::HeaderMap) -> HeaderRecord;
    let _ = extract_response_headers as fn(&reqwest::header::HeaderMap) -> HeaderRecord;
    assert!(is_non_nullable(&Some("value")));
    assert_eq!(
        filter_nullable([Some("a"), None, Some("b")]),
        vec!["a", "b"]
    );
    assert!(as_array::<&str>(Arrayable::none()).is_empty());
    assert_eq!(as_array(Arrayable::single("one")), vec!["one"]);
    assert_eq!(
        as_array(Arrayable::from(vec!["one", "two"])),
        vec!["one", "two"]
    );
    let filtered_entries = remove_undefined_entries([
        ("keep", Some("yes")),
        ("drop", None),
        ("also_keep", Some("ok")),
    ]);
    assert_eq!(filtered_entries.get("keep"), Some(&"yes"));
    assert!(!filtered_entries.contains_key("drop"));
    assert_eq!(
        load_api_key(
            LoadApiKeyOptions::new("SIUMAI_PUBLIC_SURFACE_KEY", "Test")
                .with_api_key("explicit-key")
        )
        .expect("explicit api key"),
        "explicit-key"
    );
    assert_eq!(
        load_setting(
            LoadSettingOptions::new("SIUMAI_PUBLIC_SURFACE_SETTING", "setting", "Test")
                .with_setting_value("explicit-setting"),
        )
        .expect("explicit setting"),
        "explicit-setting"
    );
    assert_eq!(
        load_optional_setting(
            LoadOptionalSettingOptions::new("SIUMAI_PUBLIC_SURFACE_OPTIONAL")
                .with_setting_value("explicit-optional")
        ),
        Some("explicit-optional".to_string())
    );
    assert_eq!(
        convert_image_model_file_to_data_uri(
            &siumai::types::ImageEditInput::base64_with_media_type("aGVsbG8=", "image/png")
        )
        .expect("image file data uri"),
        "data:image/png;base64,aGVsbG8="
    );
    assert_eq!(
        convert_base64_to_uint8_array("-_8").expect("base64url bytes"),
        vec![251, 255]
    );
    assert_eq!(convert_uint8_array_to_base64(&[251, 255]), "+/8=");
    assert_eq!(
        convert_to_base64(&DataContent::binary(b"hello".to_vec())),
        "aGVsbG8="
    );
    assert_eq!(media_type_to_extension("audio/mpeg"), "mp3");
    assert_eq!(strip_file_extension("archive.tar.gz"), "archive");
    assert_eq!(
        without_trailing_slash(Some("https://example.com/")).as_deref(),
        Some("https://example.com")
    );
    assert_eq!(DEFAULT_JSON_SCHEMA_PREFIX, "JSON schema:");
    assert_eq!(
        inject_json_instruction(JsonInstructionOptions::new().with_prompt("Return data")),
        "Return data\n\nYou MUST answer with JSON."
    );
    let json_messages = inject_json_instruction_into_messages(
        vec![ModelMessage::User(UserModelMessage::new(
            UserContent::text("Question"),
        ))],
        JsonInstructionMessageOptions::new(),
    );
    assert!(matches!(
        json_messages.first(),
        Some(ModelMessage::System(_))
    ));
    assert_eq!(
        parse_json(r#"{"answer":42}"#).expect("parse JSON")["answer"],
        serde_json::json!(42)
    );
    assert!(safe_parse_json("{not json}").is_failure());
    assert!(is_parsable_json(r#"{"ok":true}"#));
    let parse_schema =
        json_schema_with_validator(serde_json::json!({ "type": "object" }), |value| {
            value
                .get("answer")
                .and_then(JSONValue::as_str)
                .map(|answer| ValidationResult::success(answer.to_string()))
                .unwrap_or_else(|| {
                    ValidationResult::failure(LlmError::ParseError("missing answer".to_string()))
                })
        });
    assert_eq!(
        parse_json_with_schema(r#"{"answer":"yes"}"#, &parse_schema)
            .expect("parse and validate JSON"),
        "yes"
    );
    assert!(safe_parse_json_with_schema(r#"{"answer":42}"#, &parse_schema).is_failure());
    assert_eq!(
        validate_types(
            serde_json::json!({ "answer": "yes" }),
            &parse_schema,
            Some(TypeValidationContext {
                field: Some("answer".to_string()),
                entity_name: Some("response".to_string()),
                entity_id: Some("public-surface".to_string()),
            }),
        )
        .expect("validate types"),
        "yes"
    );
    assert!(
        safe_validate_types(serde_json::json!({ "answer": 42 }), &parse_schema, None).is_failure()
    );
    let mut provider_options = ProviderOptionsMap::new();
    provider_options.insert("openai", serde_json::json!({ "answer": "yes" }));
    assert_eq!(
        parse_provider_options("openai", Some(&provider_options), &parse_schema)
            .expect("parse provider options"),
        Some("yes".to_string())
    );
    let provider_reference = ProviderReference::from([("openai", "file-openai")]);
    assert_eq!(
        resolve_provider_reference(&provider_reference, "openai")
            .expect("resolve provider reference"),
        "file-openai"
    );
    assert!(resolve_provider_reference(&provider_reference, "anthropic").is_err());
    assert!(is_provider_reference(
        &FilePartSource::single_provider_reference("openai", "file-openai")
    ));
    let mapped_tools = [Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.web_search",
        "mySearch",
    ))];
    let mapping = create_tool_name_mapping(&mapped_tools, &[("openai.web_search", "web_search")]);
    assert_eq!(mapping.to_provider_tool_name("mySearch"), "web_search");
    assert_eq!(mapping.to_custom_tool_name("web_search"), "mySearch");
    assert!(is_custom_reasoning(Some(&LanguageModelReasoning::None)));
    assert_eq!(
        ReasoningLevel::try_from(LanguageModelReasoning::High).expect("reasoning level"),
        ReasoningLevel::High
    );
    let mut reasoning_warnings = Vec::new();
    assert_eq!(
        map_reasoning_to_provider_effort(
            ReasoningLevel::Minimal,
            &[(ReasoningLevel::Minimal, "low")],
            &mut reasoning_warnings,
        ),
        Some("low")
    );
    assert!(matches!(
        reasoning_warnings.first(),
        Some(Warning::Compatibility { .. })
    ));
    assert_eq!(
        map_reasoning_to_provider_budget(
            ReasoningBudgetOptions::new(ReasoningLevel::Medium, 10_000, 8_000),
            &mut reasoning_warnings,
        ),
        Some(3_000)
    );
    let mut streaming_tool_call_tracker = StreamingToolCallTracker::new();
    let mut streaming_tool_call_parts = Vec::new();
    streaming_tool_call_tracker
        .process_delta(
            StreamingToolCallDelta::new(Some(0))
                .with_id("call_public_surface")
                .with_type("function")
                .with_function_name("search")
                .with_arguments("{}"),
            |part| streaming_tool_call_parts.push(part),
        )
        .expect("streaming tool call delta");
    assert!(matches!(
        streaming_tool_call_parts.as_slice(),
        [
            LanguageModelV4StreamPart::ToolInputStart { id, tool_name, .. },
            LanguageModelV4StreamPart::ToolInputDelta { id: delta_id, delta, .. },
            LanguageModelV4StreamPart::ToolInputEnd { id: end_id, .. },
            LanguageModelV4StreamPart::ToolCall(call)
        ] if id == "call_public_surface"
            && tool_name == "search"
            && delta_id == "call_public_surface"
            && delta == "{}"
            && end_id == "call_public_surface"
            && call.tool_call_id == "call_public_surface"
            && call.tool_name == "search"
            && call.input == "{}"
    ));
    let object_options = GenerateObjectOptions::new()
        .with_schema_name("answer")
        .with_schema_description("Answer payload")
        .with_strict(true);
    assert_eq!(object_options.schema_name.as_deref(), Some("answer"));
    assert_eq!(
        object_options.schema_description.as_deref(),
        Some("Answer payload")
    );
    assert_eq!(object_options.strict, Some(true));
    let object_schema: GenerateObjectSchema<JSONValue> =
        serde_json::json!({ "type": "object" }).into();
    assert!(format!("{object_schema:?}").contains("Json"));
    let repaired_options = GenerateObjectOptions::new()
        .with_repair_text_fn(|context| async move { Ok(Some(context.text)) });
    assert!(format!("{repaired_options:?}").contains("has_repair_text"));

    let typed_schema = json_schema_with_validator(
        serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"]
        }),
        |value| {
            value
                .get("answer")
                .and_then(serde_json::Value::as_str)
                .map(|answer| ValidationResult::success(answer.to_string()))
                .unwrap_or_else(|| {
                    ValidationResult::failure(LlmError::ParseError(
                        "Expected answer string".to_string(),
                    ))
                })
        },
    );
    assert_eq!(
        typed_schema
            .validate(&serde_json::json!({ "answer": "42" }))
            .expect("validator present")
            .into_result()
            .expect("valid schema value"),
        "42"
    );

    let lazy = lazy_schema(|| json_schema(serde_json::json!({ "type": "string" })));
    assert_eq!(
        as_schema(lazy).json_schema()["type"],
        serde_json::json!("string")
    );
    let empty = as_schema_or_empty::<JSONValue>(None::<Schema>);
    assert_eq!(
        empty.json_schema()["additionalProperties"],
        serde_json::json!(false)
    );
    assert_eq!(
        empty_json_schema::<JSONValue>().json_schema()["properties"],
        serde_json::json!({})
    );

    let id = generate_id();
    assert_eq!(id.chars().count(), DEFAULT_ID_SIZE);
    assert!(id.chars().all(|ch| DEFAULT_ID_ALPHABET.contains(ch)));

    let prefixed_id_generator = create_id_generator(
        IdGeneratorOptions::new()
            .with_prefix("tool")
            .with_size(6)
            .with_alphabet("ab"),
    )
    .expect("valid id generator options");
    let prefixed_id = prefixed_id_generator();
    assert!(prefixed_id.starts_with("tool-"));
    assert_eq!(prefixed_id["tool-".len()..].chars().count(), 6);

    let runtime_tool = tool(Tool::function(
        "runtime_weather",
        "Runtime weather tool",
        serde_json::json!({ "type": "object" }),
    ));
    assert!(!is_executable_tool(Some(&runtime_tool)));
    assert!(
        dynamic_tool(runtime_tool.clone())
            .runtime_metadata()
            .dynamic()
    );
    let tool_options = ToolExecutionOptions::new("call_runtime")
        .try_with_chat_messages(&[ChatMessage::user("hello").build()])
        .expect("chat messages project to model messages");
    assert_eq!(tool_options.messages.len(), 1);
    let projected_messages =
        model_messages_from_chat_messages(&[ChatMessage::user("hello again").build()])
            .expect("project messages");
    assert_eq!(projected_messages.len(), 1);
    std::mem::drop(execute_tool(
        &runtime_tool,
        serde_json::json!({}),
        ToolExecutionOptions::new("call_unsupported"),
    ));

    let shared_data = DataContent::binary(vec![1, 2, 3]);
    let stt_request = SttRequest::from_data_content(shared_data.clone(), "audio/mpeg");
    assert_eq!(
        stt_request.audio_bytes().expect("stt audio bytes"),
        vec![1, 2, 3]
    );

    let translation_request =
        siumai::types::AudioTranslationRequest::from_data_content(shared_data.clone(), "audio/wav");
    assert_eq!(
        translation_request
            .audio_bytes()
            .expect("translation audio bytes"),
        vec![1, 2, 3]
    );

    let image_input =
        siumai::prelude::extensions::types::ImageEditInput::from_data_content(shared_data.clone())
            .with_media_type("image/png");
    assert_eq!(image_input.media_type(), Some("image/png"));
    assert_eq!(
        image_input
            .file_data()
            .expect("image file data")
            .as_bytes()
            .expect("image bytes"),
        vec![1, 2, 3]
    );

    let video_input = siumai::prelude::extensions::types::VideoGenerationInput::from_data_content(
        shared_data.clone(),
    )
    .with_media_type("image/png");
    assert_eq!(video_input.media_type(), Some("image/png"));
    assert_eq!(
        video_input
            .file_data()
            .expect("video file data")
            .as_bytes()
            .expect("video bytes"),
        vec![1, 2, 3]
    );

    let upload_data: DataContent = vec![1, 2, 3].into();
    assert_eq!(upload_data.as_bytes().expect("upload bytes"), vec![1, 2, 3]);
    let _ = upload_file::<dyn UploadFileApi, Vec<u8>>;
    let _ = upload_skill::<dyn UploadSkillApi>;

    let upload_data_str: DataContent = "AQID".into();
    assert_eq!(
        upload_data_str.as_bytes().expect("upload base64 bytes"),
        vec![1, 2, 3]
    );

    let invalid_data_error = DataContent::base64("***not-base64***")
        .as_bytes()
        .expect_err("invalid base64 must use shared error type");
    assert_eq!(
        invalid_data_error.to_string(),
        "Invalid data content. Content string is not a base64-encoded media."
    );
    assert_eq!(
        invalid_data_error.content(),
        &DataContent::Base64("***not-base64***".to_string())
    );

    let prompt = Prompt::prompt_text("hello").with_system_text("rules");
    let standardized = prompt.standardize().expect("prompt should standardize");
    assert_eq!(standardized.messages.len(), 1);

    let model_message =
        ModelMessage::try_from(ChatMessage::user("hi").build()).expect("user model message");
    assert!(matches!(model_message, ModelMessage::User(_)));

    let request = ChatRequest::try_from(Prompt::messages(vec![ModelMessage::System(
        SystemModelMessage::new("rules"),
    )]))
    .expect("prompt should convert into chat request");
    assert_eq!(request.messages.len(), 1);

    let missing_tool_result_prompt =
        Prompt::messages(vec![ModelMessage::Assistant(AssistantModelMessage::new(
            AssistantContent::parts(vec![AssistantContentPart::ToolCall(ToolCallPart::new(
                "call_missing_result",
                "regular_tool",
                serde_json::json!({}),
            ))]),
        ))]);
    let err = ChatRequest::try_from(missing_tool_result_prompt)
        .expect_err("prompt with unresolved tool call should fail");
    let PromptExecutionError::MissingToolResults(err) = err else {
        panic!("expected missing tool results error");
    };
    assert_eq!(err.tool_call_ids, vec!["call_missing_result".to_string()]);

    let tool_call = ToolCall::new(
        "call_1",
        "search".to_string(),
        serde_json::json!({ "q": "rust" }),
    )
    .with_provider_executed(true)
    .with_dynamic(true)
    .with_invalid(true)
    .with_error(serde_json::json!({ "message": "invalid input" }))
    .with_title("Search");
    assert_eq!(tool_call.r#type(), "tool-call");
    assert_eq!(tool_call.tool_call_id, "call_1");
    assert_eq!(tool_call.invalid, Some(true));

    let tool_result = ToolResult::new(
        "call_1",
        "search".to_string(),
        serde_json::json!({ "q": "rust" }),
        ToolResultOutput::json(serde_json::json!({ "ok": true })),
    )
    .with_provider_executed(true)
    .with_dynamic(true)
    .with_preliminary(true)
    .with_title("Search result");
    assert_eq!(tool_result.r#type(), "tool-result");
    assert_eq!(tool_result.tool_call_id, "call_1");
    assert_eq!(tool_result.preliminary, Some(true));

    let tool_error = ToolError::new(
        "call_1",
        "search".to_string(),
        serde_json::json!({ "q": "rust" }),
        serde_json::json!({ "message": "timeout" }),
    )
    .with_dynamic(true)
    .with_title("Search failed");
    assert_eq!(tool_error.r#type(), "tool-error");
    let denied = ToolOutputDenied::new("call_2", "delete".to_string())
        .with_provider_executed(false)
        .with_dynamic(false);
    assert_eq!(denied.r#type(), "tool-output-denied");

    let generated_file = GeneratedFile::from_bytes(b"hello", "text/plain");
    assert_eq!(generated_file.base64(), "aGVsbG8=");
    assert_eq!(
        generated_file.uint8_array().expect("decode generated file"),
        b"hello"
    );
    let text_output = TextOutput::new("hello");
    assert_eq!(text_output.r#type(), "text");
    let text_content_part: GenerateTextContentPart = text_output.clone().into();
    assert_eq!(text_content_part.r#type(), "text");
    let model_info = GenerateTextModelInfo::new("openai", "gpt-test");
    assert_eq!(model_info.model_id, "gpt-test");
    let step_reasoning = GenerateTextStepReasoningPart::reasoning_file("dHJhY2U=", "text/plain");
    assert_eq!(step_reasoning.r#type(), "reasoning-file");
    let text_stream_delta = TextStreamTextDeltaPart::new("text_1", "hello");
    assert_eq!(text_stream_delta.r#type(), "text-delta");
    let text_stream_part: TextStreamPart = text_stream_delta.into();
    assert_eq!(text_stream_part.r#type(), "text-delta");
    let custom_output = CustomOutput::new("openai.compaction");
    assert_eq!(custom_output.r#type(), "custom");
    let file_output = FileOutput::new(generated_file.clone());
    assert_eq!(file_output.r#type(), "file");
    let reasoning_output = ReasoningOutput::new("thinking");
    assert_eq!(reasoning_output.r#type(), "reasoning");
    let reasoning_file_output = ReasoningFileOutput::new(generated_file);
    assert_eq!(reasoning_file_output.r#type(), "reasoning-file");

    let source = Source::url_with_title("source_1", "https://example.com", "Example");
    let source_value = serde_json::to_value(&source).expect("serialize shared source");
    assert_eq!(source_value["type"], serde_json::json!("source"));
    assert_eq!(source_value["sourceType"], serde_json::json!("url"));
    let source_part: ContentPart = source.clone().into();
    assert_eq!(
        Source::try_from(source_part).expect("content source converts back"),
        source
    );

    let approval_response = ToolApprovalResponse::new("approval_1", true)
        .with_reason("approved")
        .with_provider_executed(true);
    assert_eq!(approval_response.approval_id, "approval_1");

    let approval_request_output =
        ToolApprovalRequestOutput::new("approval_1", tool_call.clone()).with_is_automatic(true);
    assert_eq!(approval_request_output.r#type(), "tool-approval-request");
    assert_eq!(
        serde_json::to_value(&approval_request_output).expect("serialize approval request output")
            ["toolCall"]["toolCallId"],
        serde_json::json!("call_1")
    );
    let approval_response_output = ToolApprovalResponseOutput::new("approval_1", tool_call, true)
        .with_reason("approved")
        .with_provider_executed(true);
    assert_eq!(approval_response_output.r#type(), "tool-approval-response");
    assert_eq!(approval_response.reason.as_deref(), Some("approved"));
    assert_eq!(approval_response.provider_executed, Some(true));

    let mut provider_options = ProviderOptionsMap::new();
    provider_options.insert(
        "anthropic",
        serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
    );

    let text_part = TextPart::new("hello").with_provider_options_map(provider_options.clone());
    assert_eq!(text_part.provider_options_map(), &provider_options);
    assert_eq!(
        text_part.provider_option("anthropic"),
        Some(&serde_json::json!({ "cacheControl": { "type": "ephemeral" } }))
    );

    let tool_call_part = ToolCallPart::new("call_1", "search", serde_json::json!({ "q": "rust" }))
        .with_provider_option("openai", serde_json::json!({ "parallelToolCalls": false }))
        .with_provider_executed(true);
    assert_eq!(
        tool_call_part.provider_option("openai"),
        Some(&serde_json::json!({ "parallelToolCalls": false }))
    );
    assert_eq!(tool_call_part.provider_executed, Some(true));

    let assistant_model_message = AssistantModelMessage::new(AssistantContent::parts(vec![
        AssistantContentPart::Text(text_part),
        AssistantContentPart::ToolCall(tool_call_part),
    ]))
    .with_provider_option(
        "anthropic",
        serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
    );
    assert_eq!(
        assistant_model_message.provider_options_map(),
        &provider_options
    );
    assert_eq!(
        assistant_model_message.provider_option("anthropic"),
        Some(&serde_json::json!({ "cacheControl": { "type": "ephemeral" } }))
    );

    let tool_result_output = ToolResultOutput::json(serde_json::json!({ "ok": true }))
        .with_provider_options_map(provider_options.clone());
    assert_eq!(tool_result_output.provider_options_map(), &provider_options);
    assert_eq!(
        tool_result_output.provider_option("anthropic"),
        Some(&serde_json::json!({ "cacheControl": { "type": "ephemeral" } }))
    );

    let file_url_part =
        siumai::types::ToolResultContentPart::file_url("https://example.com/report.pdf")
            .with_media_type("application/pdf");
    let file_url_json = serde_json::to_value(&file_url_part).expect("serialize file url part");
    assert_eq!(
        file_url_json["mediaType"],
        serde_json::json!("application/pdf")
    );

    let file_reference_part = siumai::types::ToolResultContentPart::file_reference(
        ProviderReference::from([("openai", "file_123")]),
    );
    let file_reference_json =
        serde_json::to_value(&file_reference_part).expect("serialize file reference part");
    assert_eq!(
        file_reference_json["type"],
        serde_json::json!("file-reference")
    );
    assert_eq!(
        file_reference_json["providerReference"]["openai"],
        serde_json::json!("file_123")
    );

    let legacy_file_id_part = siumai::types::ToolResultContentPart::file_id("file_legacy");
    let legacy_file_id_json =
        serde_json::to_value(&legacy_file_id_part).expect("serialize legacy file-id part");
    assert_eq!(legacy_file_id_json["type"], serde_json::json!("file-id"));
    assert_eq!(
        legacy_file_id_json["fileId"],
        serde_json::json!("file_legacy")
    );

    let image_content_part = ContentPart::image_url("https://example.com/image.webp")
        .with_image_media_type("image/webp");
    let image_content_json =
        serde_json::to_value(&image_content_part).expect("serialize image content part");
    assert_eq!(image_content_json["type"], serde_json::json!("image"));
    assert_eq!(
        image_content_json["mediaType"],
        serde_json::json!("image/webp")
    );

    let image_part = ImagePart::new(FilePartSource::url("https://example.com/image.png"))
        .with_media_type("image/png");
    assert_eq!(image_part.media_type.as_deref(), Some("image/png"));

    let file_part = FilePart::new(
        FilePartSource::url("https://example.com/report.pdf"),
        "application/pdf",
    )
    .with_filename("report.pdf");
    assert_eq!(file_part.filename.as_deref(), Some("report.pdf"));
}

#[test]
fn public_surface_ui_helpers_imports_compile() {
    use siumai::prelude::unified::{ExecutableTools, UiMessage, UiMessagePart};
    use siumai::ui::{
        SafeValidateUIMessagesResult, SafeValidateUiMessagesResult,
        ValidateUiMessagesSchemaOptions, safe_validate_ui_messages,
        safe_validate_ui_messages_with_schemas,
    };

    let _ = size_of::<SafeValidateUiMessagesResult>();
    let _ = size_of::<SafeValidateUIMessagesResult>();
    let _ = size_of::<ValidateUiMessagesSchemaOptions<'static>>();
    let _ = safe_validate_ui_messages as fn(&[UiMessage]) -> SafeValidateUiMessagesResult;
    let _ = safe_validate_ui_messages_with_schemas
        as for<'a> fn(
            &[UiMessage],
            ValidateUiMessagesSchemaOptions<'a>,
            Option<&ExecutableTools>,
            &dyn siumai::ui::UiSchemaValidator,
        ) -> SafeValidateUiMessagesResult;

    let result =
        safe_validate_ui_messages(&[UiMessage::user("user", vec![UiMessagePart::text("hello")])]);
    assert!(result.success());
}

#[tokio::test]
async fn public_surface_tooling_imports_compile() {
    use futures::StreamExt;
    use siumai::prelude::unified::{LlmError, parse_json_event_stream};
    use siumai::tooling::{
        ExecutableTool, ExecutableTools, ToolExecutionOptions, ToolExecutionResult, ToolSet,
        dynamic_tool, execute_tool, is_executable_tool, model_messages_from_chat_messages, tool,
    };
    use siumai::types::Tool;

    let tool = tool(Tool::function(
        "search",
        "Search tool",
        serde_json::json!({ "type": "object" }),
    ))
    .with_execute_stream_fn(|_args, options| {
        assert_eq!(options.tool_call_id, "call_public");
        Box::pin(futures::stream::iter(vec![Ok(
            serde_json::json!({ "ok": true }),
        )]))
    });

    assert!(is_executable_tool(Some(&tool)));

    let dynamic = dynamic_tool(Tool::function(
        "dynamic_search",
        "Dynamic search tool",
        serde_json::json!({ "type": "object" }),
    ));
    assert!(dynamic.runtime_metadata().dynamic());

    let shared_messages =
        model_messages_from_chat_messages(&[siumai::types::ChatMessage::user("hello").build()])
            .expect("project model messages");
    assert_eq!(shared_messages.len(), 1);

    let mut results = execute_tool(
        &tool,
        serde_json::json!({ "q": "rust" }),
        ToolExecutionOptions::new("call_public"),
    )
    .await
    .expect("execute tool");

    let first = results
        .next()
        .await
        .expect("preliminary result")
        .expect("preliminary ok");
    assert!(matches!(first, ToolExecutionResult::Preliminary { .. }));

    let second = results
        .next()
        .await
        .expect("final result")
        .expect("final ok");
    assert!(matches!(second, ToolExecutionResult::Final { .. }));

    let tools: ToolSet = ExecutableTools::from_tools([ExecutableTool::function(
        "echo",
        "Echo tool",
        serde_json::json!({ "type": "object" }),
        |args| async move { Ok(args) },
    )]);

    let out = tools
        .execute("echo", serde_json::json!({ "hello": "world" }))
        .await
        .expect("execute by name");
    assert_eq!(out["hello"], serde_json::json!("world"));

    let parsed = parse_json_event_stream(futures::stream::iter(vec![Ok::<_, LlmError>(
        b"data: {\"ok\":true}\n\n".to_vec(),
    )]))
    .collect::<Vec<_>>()
    .await;
    assert_eq!(
        parsed[0].as_ref().expect("json event")["ok"],
        serde_json::json!(true)
    );
}

#[test]
fn registry_handles_compile_as_family_models() {
    use siumai::completion::{CompletionModel, CompletionModelV3};
    use siumai::embedding::{EmbeddingModel, EmbeddingModelV3};
    use siumai::image::{ImageModel, ImageModelV3, ImageModelV4};
    use siumai::prelude::unified::{ModelMetadata, registry::*};
    use siumai::rerank::{RerankModelV3, RerankingModel};
    use siumai::speech::{SpeechModel, SpeechModelV3};
    use siumai::text::{LanguageModel, TextModelV3};
    use siumai::transcription::{TranscriptionModel, TranscriptionModelV3};
    use siumai::video::{VideoModel, VideoModelV3, VideoModelV4};

    fn _assert_completion_handle<T: CompletionModel + CompletionModelV3 + ModelMetadata>() {}
    fn _assert_text_handle<T: LanguageModel + TextModelV3 + ModelMetadata>() {}
    fn _assert_embedding_handle<T: EmbeddingModel + EmbeddingModelV3 + ModelMetadata>() {}
    fn _assert_image_handle<T: ImageModel + ImageModelV3 + ImageModelV4 + ModelMetadata>() {}
    fn _assert_rerank_handle<T: RerankingModel + RerankModelV3 + ModelMetadata>() {}
    fn _assert_speech_handle<T: SpeechModel + SpeechModelV3 + ModelMetadata>() {}
    fn _assert_transcription_handle<
        T: TranscriptionModel + TranscriptionModelV3 + ModelMetadata,
    >() {
    }
    fn _assert_video_handle<T: VideoModel + VideoModelV3 + VideoModelV4 + ModelMetadata>() {}

    let _ = size_of::<CompletionModelHandle>();
    let _ = size_of::<LanguageModelHandle>();
    let _ = size_of::<EmbeddingModelHandle>();
    let _ = size_of::<ImageModelHandle>();
    let _ = size_of::<RerankingModelHandle>();
    let _ = size_of::<SpeechModelHandle>();
    let _ = size_of::<TranscriptionModelHandle>();
    let _ = size_of::<VideoModelHandle>();

    _assert_completion_handle::<CompletionModelHandle>();
    _assert_text_handle::<LanguageModelHandle>();
    _assert_embedding_handle::<EmbeddingModelHandle>();
    _assert_image_handle::<ImageModelHandle>();
    _assert_rerank_handle::<RerankingModelHandle>();
    _assert_speech_handle::<SpeechModelHandle>();
    _assert_transcription_handle::<TranscriptionModelHandle>();
    _assert_video_handle::<VideoModelHandle>();
}

#[test]
fn public_family_helpers_compile_against_stable_family_models() {
    use siumai::{
        completion::{self, CompletionModel, CompletionRequest, StreamRequestOptions},
        embedding::{self, BatchEmbeddingRequest, EmbeddingModel, EmbeddingRequest},
        extensions::ImageExtras,
        image::{
            self, GenerateImagePrompt, GenerateImageRequest, ImageEditInput, ImageEditRequest,
            ImageGenerationRequest, ImageModel, ImageVariationRequest,
        },
        prelude::unified::{
            ChatMessage, JSONValue, generate_array, generate_choice, generate_enum, generate_json,
            generate_object, registry::*,
        },
        rerank::{self, RerankRequest, RerankingModel},
        speech::{self, SpeechModel, TtsRequest},
        structured_output::GenerateObjectSchema,
        text::{self, LanguageModel, TextRequest},
        transcription::{self, SttRequest, TranscriptionModel},
        video::{self, VideoGenerationRequest, VideoModel},
    };

    fn _assert_completion_surface<M: CompletionModel + ?Sized>(model: &M) {
        let request = CompletionRequest::new("hi");
        std::mem::drop(completion::complete(
            model,
            request.clone(),
            Default::default(),
        ));
        std::mem::drop(completion::stream(
            model,
            request.clone(),
            Default::default(),
        ));
        std::mem::drop(completion::stream_with_cancel(
            model,
            request,
            Default::default(),
        ));
    }

    fn _assert_text_surface<M: LanguageModel + ?Sized>(model: &M) {
        let request = TextRequest::new(vec![ChatMessage::user("hi").build()]);
        std::mem::drop(text::generate(model, request.clone(), Default::default()));
        std::mem::drop(text::stream(model, request.clone(), Default::default()));
        std::mem::drop(text::stream_with_cancel(
            model,
            request.clone(),
            Default::default(),
        ));
        let schema: GenerateObjectSchema<JSONValue> =
            serde_json::json!({ "type": "object" }).into();
        std::mem::drop(generate_object(
            model,
            request.clone(),
            schema.clone(),
            Default::default(),
        ));
        std::mem::drop(generate_array(
            model,
            request.clone(),
            schema,
            Default::default(),
        ));
        std::mem::drop(generate_enum(
            model,
            request.clone(),
            ["red", "green"],
            Default::default(),
        ));
        std::mem::drop(generate_choice(
            model,
            request.clone(),
            ["red", "green"],
            Default::default(),
        ));
        std::mem::drop(generate_json(model, request, Default::default()));
    }

    fn _assert_embedding_surface<M: EmbeddingModel + ?Sized>(model: &M) {
        let request = EmbeddingRequest::new(vec!["hello".to_string()]);
        let batch = BatchEmbeddingRequest::new(vec![request.clone()]);
        std::mem::drop(embedding::embed(model, request, Default::default()));
        std::mem::drop(embedding::embed_many(model, batch, Default::default()));
    }

    fn _assert_image_surface<M: ImageModel + ImageExtras + ?Sized>(model: &M) {
        std::mem::drop(image::generate(
            model,
            ImageGenerationRequest::default(),
            Default::default(),
        ));
        std::mem::drop(image::generate_image(
            model,
            GenerateImageRequest::from(GenerateImagePrompt::text("draw a robot")),
            Default::default(),
        ));
        std::mem::drop(image::edit(
            model,
            ImageEditRequest {
                prompt: "edit a robot".to_string(),
                images: vec![ImageEditInput::url("https://example.com/input.png")],
                ..Default::default()
            },
            Default::default(),
        ));
        std::mem::drop(image::variation(
            model,
            ImageVariationRequest::default()
                .with_image(ImageEditInput::url("https://example.com/input.png")),
            Default::default(),
        ));
    }

    fn _assert_rerank_surface<M: RerankingModel + ?Sized>(model: &M) {
        let request = RerankRequest::new(
            "rerank-model".to_string(),
            "hello".to_string(),
            vec!["a".to_string(), "b".to_string()],
        );
        std::mem::drop(rerank::rerank(model, request, Default::default()));
    }

    fn _assert_speech_surface<M: SpeechModel + ?Sized>(model: &M) {
        std::mem::drop(speech::synthesize(
            model,
            TtsRequest::new("hello".to_string()),
            Default::default(),
        ));
    }

    fn _assert_transcription_surface<M: TranscriptionModel + ?Sized>(model: &M) {
        std::mem::drop(transcription::transcribe(
            model,
            SttRequest::from_audio(Vec::new(), "audio/wav"),
            Default::default(),
        ));
    }

    fn _assert_video_surface<M: VideoModel + ?Sized>(model: &M) {
        std::mem::drop(video::create_task(
            model,
            VideoGenerationRequest::new("video-model", "animate a robot"),
            Default::default(),
        ));
        std::mem::drop(video::query_task(model, "task-123", Default::default()));
        std::mem::drop(video::wait_for_task(model, "task-123", Default::default()));
        std::mem::drop(video::generate(
            model,
            VideoGenerationRequest::new("video-model", "animate a robot"),
            Default::default(),
        ));
    }

    let _: fn(&CompletionModelHandle) = _assert_completion_surface::<CompletionModelHandle>;
    let _: fn(&LanguageModelHandle) = _assert_text_surface::<LanguageModelHandle>;
    let _: fn(&EmbeddingModelHandle) = _assert_embedding_surface::<EmbeddingModelHandle>;
    let _: fn(&ImageModelHandle) = _assert_image_surface::<ImageModelHandle>;
    let _: fn(&RerankingModelHandle) = _assert_rerank_surface::<RerankingModelHandle>;
    let _: fn(&SpeechModelHandle) = _assert_speech_surface::<SpeechModelHandle>;
    let _: fn(&TranscriptionModelHandle) =
        _assert_transcription_surface::<TranscriptionModelHandle>;
    let _: fn(&VideoModelHandle) = _assert_video_surface::<VideoModelHandle>;

    let runtime_stream_options = StreamRequestOptions::new().with_include_raw_chunks(true);
    let completion_request =
        CompletionRequest::new("hi").with_stream_options(runtime_stream_options);
    assert!(completion_request.stream_options.include_raw_chunks);

    let text_request = TextRequest::new(vec![ChatMessage::user("hi").build()]).with_stream_options(
        siumai::text::StreamRequestOptions::new().with_include_raw_chunks(true),
    );
    assert!(text_request.stream_options.include_raw_chunks);
}

#[test]
fn public_surface_video_family_imports_compile() {
    use siumai::video::{
        CreateTaskOptions, GenerateMaterializedVideoResult, GenerateOptions, GenerateVideoPrompt,
        GenerateVideoProviderMetadata, GenerateVideoResponseMetadata, GenerateVideoResult,
        GeneratedVideo, GeneratedVideoData, MaterializeVideoOptions, MaterializedVideo,
        MaterializedVideoAsset, QueryTaskOptions, VideoGenerationFileData, VideoGenerationInput,
        VideoGenerationPrompt, VideoGenerationRequest, VideoGenerationResponse, VideoModel,
        VideoModelV3, VideoModelV4, VideoTaskStatus, VideoTaskStatusResponse, WaitForTaskOptions,
    };

    let _ = size_of::<VideoGenerationFileData>();
    let _ = size_of::<VideoGenerationInput>();
    let _ = size_of::<VideoGenerationPrompt>();
    let _ = size_of::<GenerateVideoPrompt>();
    let _ = size_of::<VideoGenerationRequest>();
    let _ = size_of::<VideoGenerationResponse>();
    let _ = size_of::<VideoTaskStatus>();
    let _ = size_of::<VideoTaskStatusResponse>();
    let _ = size_of::<CreateTaskOptions>();
    let _ = size_of::<QueryTaskOptions>();
    let _ = size_of::<WaitForTaskOptions>();
    let _ = size_of::<GenerateOptions>();
    let _ = size_of::<GenerateVideoResponseMetadata>();
    let _ = size_of::<GenerateVideoResult>();
    let _ = size_of::<GenerateMaterializedVideoResult>();
    let _ = size_of::<GenerateVideoProviderMetadata>();
    let _ = size_of::<GeneratedVideo>();
    let _ = size_of::<GeneratedVideoData>();
    let _ = size_of::<MaterializedVideoAsset>();
    let _ = size_of::<MaterializeVideoOptions>();
    let _ = size_of::<MaterializedVideo>();
    let _ = size_of::<*const dyn VideoModel>();
    let _ = size_of::<*const dyn VideoModelV3>();
    let _ = size_of::<*const dyn VideoModelV4>();
    let _ = VideoGenerationRequest::new_without_prompt("video-model")
        .with_image(VideoGenerationInput::url("https://example.com/image.png"));
    let _ = VideoGenerationRequest::from_generate_prompt(
        "video-model",
        VideoGenerationPrompt::image(VideoGenerationInput::url("https://example.com/image.png")),
    );
    let status = VideoTaskStatusResponse {
        task_id: "task-1".to_string(),
        status: VideoTaskStatus::Success,
        file_id: None,
        video_url: None,
        provider_reference: Some(siumai::types::ProviderReference::single(
            "gemini",
            "files/123",
        )),
        duration: None,
        video_width: None,
        video_height: None,
        base_resp: None,
        metadata: Default::default(),
        response: None,
    };
    assert_eq!(
        status
            .effective_provider_reference("gemini")
            .and_then(|reference| reference.get("gemini").map(str::to_string)),
        Some("files/123".to_string())
    );
}

#[test]
fn public_surface_extensions_imports_compile() {
    use siumai::experimental::client::LlmClient;
    use siumai::extensions::types::*;
    use siumai::extensions::*;

    let _ = size_of::<ImageEditRequest>();
    let _ = size_of::<ModerationRequest>();
    let _ = size_of::<VideoGenerationRequest>();

    let _ = size_of::<*const dyn SpeechExtras>();
    let _ = size_of::<*const dyn TranscriptionExtras>();
    let _ = size_of::<*const dyn TimeoutCapability>();
    let _ = size_of::<*const dyn ModelListingCapability>();

    fn _assert_audio_extra_accessors<C: LlmClient + ?Sized>(client: &C) {
        let _ = client.as_speech_extras();
        let _ = client.as_transcription_extras();
    }

    let _ = _assert_audio_extra_accessors::<dyn LlmClient>;
}

#[test]
fn public_surface_compat_imports_compile() {
    use siumai::compat::{Siumai, SiumaiBuilder};

    let _ = size_of::<Siumai>();
    let _ = size_of::<SiumaiBuilder>();
}

#[test]
fn public_surface_compat_prelude_imports_compile() {
    use siumai::prelude::compat::{Provider, Siumai, SiumaiBuilder};

    let _ = size_of::<Provider>();
    let _ = size_of::<Siumai>();
    let _ = size_of::<SiumaiBuilder>();
}

#[test]
fn public_surface_streaming_aliases_compile() {
    use siumai::experimental::streaming::{
        LanguageModelV3StreamPart, LanguageModelV4File, LanguageModelV4FinishReason,
        LanguageModelV4StreamPart, LanguageModelV4ToolCall, LanguageModelV4Usage,
        SharedV4ProviderMetadata, SharedV4Warning,
    };

    let _ = size_of::<LanguageModelV3StreamPart>();
    let _ = size_of::<LanguageModelV4StreamPart>();
    let _ = size_of::<LanguageModelV4ToolCall>();
    let _ = size_of::<LanguageModelV4File>();
    let _ = size_of::<LanguageModelV4Usage>();
    let _ = size_of::<LanguageModelV4FinishReason>();
    let _ = size_of::<SharedV4ProviderMetadata>();
    let _ = size_of::<SharedV4Warning>();
}

#[cfg(feature = "openai")]
#[test]
#[allow(deprecated)]
fn public_surface_openai_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::openai::{
        OpenAIProviderSettings, OpenAiBuilder, OpenAiClient, OpenAiConfig, VERSION, create_openai,
        ext::{
            OpenAiResponsesEventConverter, moderation, responses, speech_streaming,
            transcription_streaming,
        },
        metadata::*,
        openai as openai_builder,
        options::*,
        resources::{OpenAiFiles, OpenAiModels, OpenAiModeration, OpenAiRerank},
    };

    let _ = size_of::<OpenAiBuilder>();
    let _ = size_of::<OpenAiClient>();
    let _ = size_of::<OpenAiConfig>();
    let _ = size_of::<OpenAIProviderSettings>();
    let _ = size_of::<OpenAIContextManagementConfig>();
    let _ = size_of::<OpenAIContextManagementType>();
    let _ = size_of::<OpenAILanguageModelChatOptions>();
    let _ = size_of::<OpenAIChatLanguageModelOptions>();
    let _ = size_of::<OpenAILanguageModelResponsesOptions>();
    let _ = size_of::<OpenAIResponsesProviderOptions>();
    let _ = size_of::<OpenAILanguageModelCompletionOptions>();
    let _ = size_of::<OpenAIEmbeddingModelOptions>();
    let _ = size_of::<OpenAISpeechModelOptions>();
    let _ = size_of::<OpenAITranscriptionModelOptions>();
    let _ = size_of::<OpenAIFilesOptions>();
    let _ = size_of::<OpenAiOptions>();
    let _ = size_of::<SystemMessageMode>();
    let _ = size_of::<OpenAiFiles>();
    let _ = size_of::<OpenAiModels>();
    let _ = size_of::<OpenAiModeration>();
    let _ = size_of::<OpenAiRerank>();
    let _ = size_of::<OpenAiMetadata>();
    let _ = size_of::<OpenAiSource>();
    let _ = size_of::<OpenAiResponsesEventConverter>();
    let _ = size_of::<moderation::OpenAiModerationRequest>();
    let _ = size_of::<responses::OpenAiResponsesCompactRequest>();
    let _ = size_of::<responses::OpenAiResponsesCustomEvent>();
    let _ = size_of::<responses::OpenAiSourceEvent>();
    let _ = OpenAiClient::base_url;
    let _ = OpenAiClient::set_retry_options;
    let _ = openai_builder();
    let _ = create_openai();
    let _ = VERSION;
    let _ = OpenAIProviderSettings::new();
    let _ = OpenAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/openai")
        .with_organization("org-123")
        .with_project("proj-456")
        .with_header("x-test", "1")
        .into_builder_for_model("gpt-4.1-mini");
    let _ = OpenAIProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model("gpt-4.1-mini");

    let req = ChatRequest::new(vec![user!("hi")]).with_openai_options(OpenAiOptions::new());
    let _ = req;
    let req = ChatRequest::new(vec![user!("hi")])
        .with_openai_options(OpenAILanguageModelChatOptions::new());
    let _ = req;
    let req = ChatRequest::new(vec![user!("hi")])
        .with_openai_options(OpenAILanguageModelResponsesOptions::new());
    let _ = req;
    let req = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
        .with_openai_stt_options(OpenAiSttOptions::new().with_language("en"));
    let _ = req;

    fn _assert_req_ext<T: OpenAiChatRequestExt>() {}
    fn _assert_stt_req_ext<T: OpenAiSttRequestExt>() {}
    fn _assert_resp_ext<T: OpenAiChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_stt_req_ext::<SttRequest>();
    _assert_resp_ext::<ChatResponse>();

    let _ = siumai::hosted_tools::openai::web_search().build();
    let _ = siumai::provider_ext::openai::hosted_tools::web_search().build();
    let _ = siumai::provider_ext::openai::tools::web_search();
    let _ = siumai::provider_ext::openai::provider_tools::web_search();
    let _ = siumai::Provider::openai_responses();
    let _ = siumai::Provider::openai_chat();
    let _ = speech_streaming::tts_sse_stream;
    let _ = transcription_streaming::stt_sse_stream;
}

#[cfg(feature = "openai")]
#[test]
fn public_surface_openai_compatible_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::openai_compatible::{
        ConfigurableAdapter, MetadataExtractor, OpenAICompatibleChatModelId,
        OpenAICompatibleClient, OpenAICompatibleCompletionModelId, OpenAICompatibleConfig,
        OpenAICompatibleEmbeddingModelId, OpenAICompatibleErrorData, OpenAICompatibleImageModelId,
        OpenAICompatibleProviderSettings, OpenAICompatibleRequestSettings,
        OpenAiCompatibleChatModelId, OpenAiCompatibleClient, OpenAiCompatibleCompletionModelId,
        OpenAiCompatibleConfig, OpenAiCompatibleEmbeddingModelId, OpenAiCompatibleErrorData,
        OpenAiCompatibleImageModelId, OpenAiCompatibleRequestSettings, ProviderAdapter,
        ProviderCompatibility, ProviderConfig, ProviderErrorStructure, RequestBodyTransformer,
        ResponseMetadataExtractor, VERSION, deepinfra, fireworks, generic_provider_config,
        get_provider_config, groq, list_provider_ids, moonshot, moonshotai, openrouter, options::*,
        provider_supports_capability, siliconflow, xai,
    };
    use std::sync::Arc;

    let _ = size_of::<OpenAICompatibleChatModelId>();
    let _ = size_of::<OpenAICompatibleClient>();
    let _ = size_of::<OpenAICompatibleCompletionModelId>();
    let _ = size_of::<OpenAICompatibleConfig>();
    let _ = size_of::<OpenAICompatibleEmbeddingModelId>();
    let _ = size_of::<OpenAICompatibleErrorData>();
    let _ = size_of::<OpenAICompatibleImageModelId>();
    let _ = size_of::<OpenAICompatibleProviderSettings>();
    let _ = size_of::<OpenAICompatibleRequestSettings>();
    let _ = VERSION;
    let _ = size_of::<OpenAiCompatibleChatModelId>();
    let _ = size_of::<OpenAiCompatibleClient>();
    let _ = size_of::<OpenAiCompatibleCompletionModelId>();
    let _ = size_of::<OpenAiCompatibleConfig>();
    let _ = size_of::<OpenAiCompatibleEmbeddingModelId>();
    let _ = size_of::<OpenAiCompatibleErrorData>();
    let _ = size_of::<OpenAiCompatibleImageModelId>();
    let _ = size_of::<ProviderErrorStructure<OpenAiCompatibleErrorData>>();
    let _ = size_of::<OpenAICompatibleLanguageModelChatOptions>();
    let _ = size_of::<OpenAICompatibleLanguageModelCompletionOptions>();
    let _ = size_of::<OpenAICompatibleEmbeddingModelOptions>();
    #[allow(deprecated)]
    let _ = size_of::<OpenAICompatibleProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<OpenAICompatibleCompletionProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<OpenAICompatibleEmbeddingProviderOptions>();
    let _ = size_of::<OpenAiCompatibleLanguageModelChatOptions>();
    let _ = size_of::<OpenAiCompatibleLanguageModelCompletionOptions>();
    let _ = size_of::<OpenAiCompatibleEmbeddingModelOptions>();
    #[allow(deprecated)]
    let _ = size_of::<OpenAiCompatibleProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<OpenAiCompatibleCompletionProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<OpenAiCompatibleEmbeddingProviderOptions>();
    let _ = size_of::<OpenAiCompatibleRequestSettings>();
    let _ = size_of::<ConfigurableAdapter>();
    let _ = size_of::<ProviderCompatibility>();
    let _ = size_of::<ProviderConfig>();

    struct NoopRequestBodyTransformer;
    impl RequestBodyTransformer for NoopRequestBodyTransformer {
        fn transform_request_body(
            &self,
            _body: &mut serde_json::Value,
            _model: &str,
            _request_type: siumai_provider_openai_compatible::providers::openai_compatible::RequestType,
        ) -> Result<(), siumai::prelude::LlmError> {
            Ok(())
        }
    }

    fn _assert_adapter<T: ProviderAdapter>() {}
    _assert_adapter::<ConfigurableAdapter>();
    fn _accept_request_body_transformer(_transformer: Arc<dyn RequestBodyTransformer>) {}
    fn _accept_metadata_extractor_alias(_extractor: Arc<dyn MetadataExtractor>) {}
    fn _accept_metadata_extractor(_extractor: Arc<dyn ResponseMetadataExtractor>) {}
    let request_body_transformer: Arc<dyn RequestBodyTransformer> =
        Arc::new(NoopRequestBodyTransformer);
    _accept_request_body_transformer(request_body_transformer);
    let _request_settings = OpenAICompatibleRequestSettings {
        query_params: std::collections::BTreeMap::from([(
            "api-version".to_string(),
            "2025-04-01".to_string(),
        )]),
        include_usage: Some(true),
        supports_structured_outputs: Some(false),
        request_body_transformer: None,
    };
    let _ = OpenAICompatibleProviderSettings::new("acme", "https://example.com/v1")
        .with_api_key("test-key")
        .with_header("x-test", "1")
        .with_query_param("api-version", "2025-04-01")
        .with_include_usage(true)
        .with_supports_structured_outputs(false)
        .into_config_for_model("acme-chat");
    let _ = generic_provider_config("acme", "Acme", "https://example.com/v1");
    let extractor: Arc<dyn ResponseMetadataExtractor> = Arc::new(|raw: &serde_json::Value| {
        raw.get("test").map(|value| {
            std::collections::HashMap::from([(
                "test-provider".to_string(),
                serde_json::json!({ "value": value }),
            )])
        })
    });
    _accept_metadata_extractor(extractor);
    let extractor_alias: Arc<dyn MetadataExtractor> = Arc::new(|raw: &serde_json::Value| {
        raw.get("test").map(|value| {
            std::collections::HashMap::from([(
                "test-provider".to_string(),
                serde_json::json!({ "value": value }),
            )])
        })
    });
    _accept_metadata_extractor_alias(extractor_alias);
    let error_structure = ProviderErrorStructure::<OpenAICompatibleErrorData>::serde_json(|data| {
        data.error.message.clone()
    })
    .with_is_retryable(|status, _| {
        status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
    });
    let decoded_error = error_structure
        .deserialize(&serde_json::json!({
            "error": {
                "message": "retry later"
            }
        }))
        .expect("decode compat error");
    assert_eq!(error_structure.message(&decoded_error), "retry later");
    assert_eq!(
        error_structure.is_retryable(reqwest::StatusCode::TOO_MANY_REQUESTS, Some(&decoded_error)),
        Some(true)
    );

    let _ = get_provider_config("openrouter");
    let _ = list_provider_ids();
    let _ = provider_supports_capability("openrouter", "chat");
    let _ = deepinfra::CHAT;
    let _ = deepinfra::completion::LLAMA_V3P3_70B_INSTRUCT;
    let _ = deepinfra::embedding::BGE_M3;
    let _ = deepinfra::image::FLUX_1_KONTEXT_PRO;
    let _ = fireworks::CHAT;
    let _ = fireworks::chat::KIMI_K2_THINKING;
    let _ = fireworks::completion::LLAMA_V3_8B_INSTRUCT;
    let _ = fireworks::embedding::NOMIC_EMBED_TEXT_V1_5;
    let _ = fireworks::image::FLUX_KONTEXT_MAX;
    let _ = openrouter::openai::GPT_4O;
    let _ = groq::LLAMA_3_1_8B;
    let _ = moonshotai::recommended::CHAT;
    let _ = moonshot::recommended::CHAT;
    let _ = siliconflow::DEEPSEEK_V3;
    let _ = xai::GROK_BETA;

    let _ = ChatRequest::new(vec![user!("hi")]).with_openai_compatible_options(
        OpenAICompatibleLanguageModelChatOptions::new()
            .with_user("user-123")
            .with_reasoning_effort("high")
            .with_text_verbosity("medium")
            .with_strict_json_schema(true),
    );
    let _ = CompletionRequest::new("hi").with_openai_compatible_options(
        OpenAICompatibleLanguageModelCompletionOptions::new()
            .with_echo(true)
            .with_logit_bias_token("42", 1.5)
            .with_suffix(" after")
            .with_user("user-456"),
    );
    let _ = EmbeddingRequest::single("hello").with_openai_compatible_options(
        OpenAICompatibleEmbeddingModelOptions::new()
            .with_dimensions(256)
            .with_user("user-789"),
    );
}

#[cfg(feature = "openai")]
#[test]
fn public_surface_openrouter_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::openrouter::{metadata::*, options::*};
    use std::collections::HashMap;

    let _ = size_of::<OpenRouterOptions>();
    let _ = size_of::<OpenRouterTransform>();
    let _ = size_of::<OpenRouterMetadata>();
    let _ = size_of::<OpenRouterSource>();
    let _ = size_of::<OpenRouterSourceMetadata>();
    let _ = size_of::<OpenRouterContentPartMetadata>();

    let req = ChatRequest::new(vec![user!("hi")]).with_openrouter_options(
        OpenRouterOptions::new().with_transform(OpenRouterTransform::MiddleOut),
    );
    let _ = req;

    fn _assert_req_ext<T: OpenRouterChatRequestExt>() {}
    fn _assert_resp_ext<T: OpenRouterChatResponseExt>() {}
    fn _assert_source_ext<T: OpenRouterSourceExt>() {}
    fn _assert_content_ext<T: OpenRouterContentPartExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
    _assert_source_ext::<OpenRouterSource>();
    _assert_content_ext::<ContentPart>();

    let mut resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut inner = HashMap::new();
    inner.insert(
        "logprobs".to_string(),
        serde_json::json!([{
            "token": "ok",
            "logprob": -0.1,
            "bytes": [111, 107],
            "top_logprobs": []
        }]),
    );
    inner.insert(
        "sources".to_string(),
        serde_json::json!([{
            "id": "src_1",
            "source_type": "url",
            "url": "https://openrouter.ai/docs",
            "title": "OpenRouter Docs"
        }]),
    );
    let mut outer = HashMap::new();
    outer.insert(
        "openrouter".to_string(),
        serde_json::Value::Object(inner.into_iter().collect()),
    );
    resp.provider_metadata = Some(outer);
    let _ = resp.openrouter_metadata();

    let source = OpenRouterSource {
        id: "src_1".to_string(),
        source_type: "url".to_string(),
        url: "https://openrouter.ai/docs".to_string(),
        title: Some("OpenRouter Docs".to_string()),
        tool_call_id: None,
        media_type: None,
        filename: None,
        provider_metadata: Some(serde_json::json!({
            "openrouter": {
                "fileId": "file_123",
                "containerId": "container_456",
                "index": 1
            }
        })),
        snippet: None,
    };
    let _ = source.openrouter_metadata();

    let part = ContentPart::Reasoning {
        text: "thinking".to_string(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: Some(HashMap::from([(
            "openrouter".to_string(),
            serde_json::json!({
                "itemId": "or_1",
                "reasoningEncryptedContent": "enc_456"
            }),
        )])),
    };
    let _ = part.openrouter_metadata();
}

#[cfg(feature = "openai")]
#[test]
fn public_surface_perplexity_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::perplexity::{metadata::*, options::*};

    let _ = size_of::<PerplexityOptions>();
    let _ = size_of::<PerplexityMetadata>();
    let _ = size_of::<PerplexityImage>();
    let _ = size_of::<PerplexityUsage>();

    let req = ChatRequest::new(vec![user!("hi")])
        .with_perplexity_options(PerplexityOptions::new().with_return_images(true));
    let _ = req;

    fn _assert_req_ext<T: PerplexityChatRequestExt>() {}
    fn _assert_resp_ext<T: PerplexityChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
}

#[cfg(feature = "protocol-openai")]
#[test]
fn public_surface_protocol_openai_compiles() {
    use siumai::prelude::unified::*;
    use siumai::protocol::openai::*;

    let _ = size_of::<OpenAiChatStandard>();
    let _ = size_of::<ChatRequest>();
}

#[cfg(feature = "openai")]
#[test]
fn public_surface_fireworks_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::fireworks::{models, options::*};

    let _ = size_of::<FireworksChatOptions>();
    let _ = size_of::<FireworksLanguageModelOptions>();
    let _ = size_of::<FireworksThinkingConfig>();
    let _ = size_of::<FireworksThinkingType>();
    let _ = size_of::<FireworksReasoningHistory>();
    let _ = models::fireworks::CHAT;
    let _ = models::chat::QWEN2_VL_72B_INSTRUCT;
    let _ = models::completion::LLAMA_V2_34B_CODE;
    let _ = models::embedding::NOMIC_EMBED_TEXT_V1_5;
    let _ = models::image::FLUX_1_DEV_FP8;

    let req = ChatRequest::new(vec![user!("hi")]).with_fireworks_options(
        FireworksChatOptions::new()
            .with_thinking(
                FireworksThinkingConfig::new()
                    .with_type(FireworksThinkingType::Enabled)
                    .with_budget_tokens(2048),
            )
            .with_reasoning_history(FireworksReasoningHistory::Interleaved),
    );
    let _ = req;

    fn _assert_req_ext<T: FireworksChatRequestExt>() {}
    _assert_req_ext::<ChatRequest>();
}

#[cfg(feature = "anthropic")]
#[test]
#[allow(deprecated)]
fn public_surface_anthropic_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::anthropic::{
        AnthropicBuilder, AnthropicClient, AnthropicConfig, AnthropicProviderSettings, VERSION,
        anthropic as anthropic_builder, create_anthropic,
        ext::{structured_output, thinking, tools},
        find_anthropic_container_id_from_last_step, forward_anthropic_container_id_from_last_step,
        metadata::*,
        options::*,
        resources::{AnthropicFiles, AnthropicMessageBatches, AnthropicTokens},
    };
    use std::collections::HashMap;

    let _ = size_of::<AnthropicBuilder>();
    let _ = size_of::<AnthropicClient>();
    let _ = size_of::<AnthropicConfig>();
    let _ = size_of::<AnthropicProviderSettings>();
    let _ = VERSION;
    let _ = size_of::<AnthropicOptions>();
    let _ = size_of::<AnthropicLanguageModelOptions>();
    let _ = size_of::<AnthropicProviderOptions>();
    let _ = size_of::<AnthropicFiles>();
    let _ = size_of::<AnthropicMessageBatches>();
    let _ = size_of::<AnthropicTokens>();
    let _ = size_of::<AnthropicContainerSkillType>();
    let _ = size_of::<AnthropicEffort>();
    let _ = size_of::<AnthropicInferenceGeo>();
    let _ = size_of::<AnthropicContextManagementConfig>();
    let _ = size_of::<AnthropicContextManagementEdit>();
    let _ = size_of::<AnthropicMcpServer>();
    let _ = size_of::<AnthropicToolAllowedCaller>();
    let _ = size_of::<AnthropicToolOptions>();
    let _ = size_of::<AnthropicRequestMetadata>();
    let _ = size_of::<AnthropicSpeed>();
    let _ = size_of::<AnthropicTaskBudget>();
    let _ = size_of::<AnthropicTaskBudgetType>();
    let _ = size_of::<AnthropicThinkingConfig>();
    let _ = size_of::<AnthropicThinkingDisplay>();
    let _ = anthropic_builder();
    let _ = create_anthropic();
    let _ = AnthropicProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/anthropic")
        .with_header("x-test", "1")
        .into_builder_for_model("claude-sonnet-4-5-20250929");
    let _ = AnthropicProviderSettings::new()
        .with_auth_token("test-token")
        .with_base_url("https://example.com/anthropic")
        .into_config_for_model("claude-sonnet-4-5-20250929");
    let _ = size_of::<AnthropicResponseFormat>();
    let _ = size_of::<AnthropicStructuredOutputMode>();
    let _ = size_of::<ThinkingModeConfig>();
    let _ = size_of::<AnthropicMetadata>();
    let _ = size_of::<AnthropicMessageContainerMetadata>();
    let _ = size_of::<AnthropicMessageContainerSkill>();
    let _ = size_of::<AnthropicMessageMetadata>();
    let _ = size_of::<AnthropicSource>();
    let _ = size_of::<AnthropicUsageIteration>();
    let _ = size_of::<tools::AnthropicCustomEvent>();
    let _ = size_of::<tools::AnthropicProviderToolCallEvent>();
    let _ = size_of::<tools::AnthropicProviderToolResultEvent>();
    let _ = size_of::<tools::AnthropicSourceEvent>();
    let _ = AnthropicClient::set_retry_options;

    fn _assert_req_ext<T: AnthropicChatRequestExt>() {}
    fn _assert_resp_ext<T: AnthropicChatResponseExt>() {}
    fn _assert_tool_ext<T: tools::AnthropicToolExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
    _assert_tool_ext::<Tool>();
    let empty_steps: [Option<&siumai::types::ProviderMetadataMap>; 0] = [];
    let _ = find_anthropic_container_id_from_last_step(empty_steps);
    let _ = forward_anthropic_container_id_from_last_step(empty_steps);
    let _ = thinking::assistant_message_with_thinking_metadata;
    let _ = structured_output::chat_with_json_object::<AnthropicClient>;

    let mut resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut inner = HashMap::new();
    inner.insert(
        "sources".to_string(),
        serde_json::json!([{
            "id": "src_1",
            "source_type": "document",
            "title": "Example",
            "filename": "example.pdf"
        }]),
    );
    let mut outer = HashMap::new();
    outer.insert(
        "anthropic".to_string(),
        serde_json::Value::Object(inner.into_iter().collect()),
    );
    resp.provider_metadata = Some(outer);
    let typed = resp.anthropic_metadata().expect("anthropic metadata");
    let source = typed
        .sources
        .as_ref()
        .and_then(|sources| sources.first())
        .expect("anthropic source");
    assert_eq!(source.source_type, "document");
    assert_eq!(source.filename.as_deref(), Some("example.pdf"));

    let _ = siumai::hosted_tools::anthropic::web_search_20250305().build();
    let _ = siumai::provider_ext::anthropic::hosted_tools::web_search_20250305().build();
    let _ = siumai::provider_ext::anthropic::tools::web_search_20250305();
    let _ = siumai::provider_ext::anthropic::provider_tools::web_search_20250305();
    let _ = AnthropicConfig::new("test-key")
        .with_model("claude-sonnet-4-5")
        .with_anthropic_thinking_mode(ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(1024),
        })
        .with_anthropic_structured_output_mode(AnthropicStructuredOutputMode::JsonTool)
        .with_anthropic_context_management(AnthropicContextManagementConfig::new().with_edit(
            AnthropicContextManagementEdit::ClearToolUses20250919 {
                trigger: None,
                keep: None,
                clear_at_least: Some(AnthropicContextManagementInputTokensValue::input_tokens(1)),
                clear_tool_inputs: None,
                exclude_tools: None,
            },
        ))
        .with_anthropic_tool_streaming(false)
        .with_anthropic_effort(AnthropicEffort::High)
        .with_anthropic_task_budget(AnthropicTaskBudget::tokens(400000))
        .with_anthropic_inference_geo(AnthropicInferenceGeo::Us);
    let _ = siumai::Provider::anthropic()
        .api_key("test-key")
        .model("claude-sonnet-4-5")
        .with_anthropic_thinking_mode(ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(1024),
        })
        .with_anthropic_structured_output_mode(AnthropicStructuredOutputMode::JsonTool)
        .with_anthropic_context_management(AnthropicContextManagementConfig::new().with_edit(
            AnthropicContextManagementEdit::ClearToolUses20250919 {
                trigger: None,
                keep: None,
                clear_at_least: Some(AnthropicContextManagementInputTokensValue::input_tokens(1)),
                clear_tool_inputs: None,
                exclude_tools: None,
            },
        ))
        .with_anthropic_tool_streaming(false)
        .with_anthropic_effort(AnthropicEffort::High)
        .with_anthropic_task_budget(AnthropicTaskBudget::tokens(400000))
        .with_anthropic_inference_geo(AnthropicInferenceGeo::Us);
    let _ = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .model("claude-sonnet-4-5")
        .with_anthropic_thinking_mode(ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(1024),
        })
        .with_anthropic_structured_output_mode(AnthropicStructuredOutputMode::JsonTool)
        .with_anthropic_context_management(AnthropicContextManagementConfig::new().with_edit(
            AnthropicContextManagementEdit::ClearToolUses20250919 {
                trigger: None,
                keep: None,
                clear_at_least: Some(AnthropicContextManagementInputTokensValue::input_tokens(1)),
                clear_tool_inputs: None,
                exclude_tools: None,
            },
        ))
        .with_anthropic_tool_streaming(false)
        .with_anthropic_effort(AnthropicEffort::High)
        .with_anthropic_task_budget(AnthropicTaskBudget::tokens(400000))
        .with_anthropic_inference_geo(AnthropicInferenceGeo::Us);
    let _ = siumai::Provider::anthropic();
}

#[cfg(feature = "protocol-anthropic")]
#[test]
fn public_surface_protocol_anthropic_compiles() {
    use siumai::prelude::unified::*;
    use siumai::protocol::anthropic::*;

    let _ = size_of::<AnthropicChatStandard>();
    let _ = size_of::<ChatRequest>();
}

#[cfg(feature = "google")]
#[test]
fn public_surface_gemini_provider_ext_compiles() {
    use siumai::prelude::extensions::types::VideoGenerationRequest;
    use siumai::prelude::unified::*;
    use siumai::provider_ext::gemini::{
        GeminiBuilder, GeminiClient, chat, embedding,
        ext::{code_execution, file_search_stores, tools},
        image,
        metadata::*,
        model_sets,
        options::*,
        resources::{
            GeminiCachedContents, GeminiFileSearchStores, GeminiFiles, GeminiModels, GeminiTokens,
            GeminiVideo, GoogleErrorData,
        },
        video,
    };

    let _ = size_of::<GeminiBuilder>();
    let _ = size_of::<GeminiClient>();
    let _ = size_of::<GeminiImageOptions>();
    let _ = size_of::<GoogleImageModelOptions>();
    let _ = size_of::<GoogleLanguageModelOptions>();
    let _ = size_of::<GoogleEmbeddingModelOptions>();
    let _ = size_of::<GoogleVideoModelOptions>();
    let _ = size_of::<GoogleFilesUploadOptions>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIImageProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIEmbeddingProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIVideoProviderOptions>();
    #[allow(deprecated)]
    let _: GoogleGenerativeAIVideoModelId = "veo-3.1-generate-preview".to_string();
    let _ = size_of::<GeminiOptions>();
    let _ = size_of::<GeminiThinkingConfig>();
    let _ = size_of::<GeminiMetadata>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIProviderMetadata>();
    let _ = size_of::<GoogleProviderMetadata>();
    let _ = size_of::<GeminiSource>();
    let _ = size_of::<GeminiCachedContents>();
    let _ = size_of::<GeminiFileSearchStores>();
    let _ = size_of::<GeminiFiles>();
    let _ = size_of::<GeminiModels>();
    let _ = size_of::<GeminiTokens>();
    let _ = size_of::<GeminiVideo>();
    let _ = size_of::<GoogleErrorData>();
    let _ = size_of::<code_execution::CodeExecutionConfig>();
    let _ = size_of::<code_execution::CodeExecutionResult>();
    let _ = size_of::<tools::GeminiCustomEvent>();
    let _ = size_of::<tools::GeminiSourceEvent>();
    let _ = chat::GEMINI_2_5_FLASH;
    let _ = embedding::GEMINI_EMBEDDING_001;
    let _ = image::IMAGEN_4_0_GENERATE_001;
    let _ = video::VEO_3_1_GENERATE_PREVIEW;
    let _ = model_sets::ALL_CHAT;
    let _ = model_sets::ALL_EMBEDDING;
    let _ = model_sets::ALL_IMAGE;
    let _ = model_sets::ALL_VIDEO;
    let _ = GeminiClient::base_url;
    let _ = GeminiClient::set_retry_options;

    fn _assert_req_ext<T: GeminiChatRequestExt>() {}
    fn _assert_image_req_ext<T: GeminiImageRequestExt>() {}
    fn _assert_resp_ext<T: GeminiChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_image_req_ext::<siumai::image::GenerateImageRequest>();
    _assert_resp_ext::<ChatResponse>();
    let _ = file_search_stores::stores;

    let _ = siumai::image::GenerateImageRequest::new("draw a robot").with_gemini_image_options(
        GeminiImageOptions::new()
            .with_aspect_ratio("16:9")
            .with_person_generation("allow_all"),
    );
    let _ = VideoGenerationRequest::new("veo-3.1-generate-preview", "draw a robot")
        .with_google_video_options(
            GoogleVideoModelOptions::new()
                .with_negative_prompt("no cats")
                .with_person_generation("allow_all"),
        );

    let _ = siumai::hosted_tools::google::google_search().build();
    let _ = siumai::provider_ext::gemini::hosted_tools::file_search()
        .with_file_search_store_names(vec!["fileSearchStores/store-123".to_string()])
        .build();
    let _ = siumai::provider_ext::gemini::hosted_tools::google_search().build();
    let _ = siumai::provider_ext::gemini::tools::code_execution();
    let _ = siumai::provider_ext::gemini::tools::google_maps();
    let _ = siumai::provider_ext::gemini::tools::google_search();
    let _ = siumai::provider_ext::gemini::provider_tools::google_search();
    let _ = siumai::Provider::gemini();
    let _ = siumai::Provider::gemini().language_model(chat::GEMINI_2_5_FLASH);
    let _ = siumai::Provider::gemini().chat(chat::GEMINI_2_5_PRO);
    let _ = siumai::Provider::gemini().embedding_model(embedding::GEMINI_EMBEDDING_001);
    let _ = siumai::Provider::gemini().embedding(embedding::GEMINI_EMBEDDING_2_PREVIEW);
    let _ = siumai::Provider::gemini().image_model(image::GEMINI_2_5_FLASH_IMAGE);
    let _ = siumai::Provider::gemini().image(image::IMAGEN_4_0_GENERATE_001);
    let _ = siumai::Provider::gemini().video_model(video::VEO_3_1_GENERATE_PREVIEW);
    let _ = siumai::Provider::gemini().video(video::VEO_3_1_FAST_GENERATE_PREVIEW);
}

#[cfg(feature = "google")]
#[test]
#[allow(deprecated)]
fn public_surface_google_provider_ext_compiles() {
    use siumai::compat::Siumai;
    use siumai::prelude::extensions::types::VideoGenerationRequest;
    use siumai::prelude::unified::*;
    use siumai::provider_ext::google::GoogleGenerativeAIProviderSettings;
    use siumai::provider_ext::google::{
        GeminiBuilder, GeminiClient, GeminiConfig, GoogleErrorData, GoogleProviderSettings,
        SharedIdGenerator, VERSION, chat, create_google, create_google_generative_ai, embedding,
        ext::{code_execution, file_search_stores, tools},
        google as google_builder, image,
        metadata::*,
        model_sets,
        options::*,
        resources::{
            GeminiCachedContents, GeminiFileSearchStores, GeminiFiles, GeminiModels, GeminiTokens,
            GeminiVideo,
        },
        video,
    };

    let _ = size_of::<GeminiBuilder>();
    let _ = size_of::<GeminiClient>();
    let _ = size_of::<GeminiConfig>();
    let _ = size_of::<SharedIdGenerator>();
    let _ = size_of::<GoogleProviderSettings>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIProviderSettings>();
    let _ = size_of::<GeminiImageOptions>();
    let _ = size_of::<GoogleImageModelOptions>();
    let _ = size_of::<GoogleLanguageModelOptions>();
    let _ = size_of::<GoogleEmbeddingModelOptions>();
    let _ = size_of::<GoogleVideoModelOptions>();
    let _ = size_of::<GoogleFilesUploadOptions>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIImageProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIEmbeddingProviderOptions>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIVideoProviderOptions>();
    let _ = size_of::<GoogleProviderMetadata>();
    #[allow(deprecated)]
    let _ = size_of::<GoogleGenerativeAIProviderMetadata>();
    let _: GoogleVideoModelId = "veo-3.1-generate-preview".to_string();
    #[allow(deprecated)]
    let _: GoogleGenerativeAIVideoModelId = "veo-3.1-generate-preview".to_string();
    let _ = size_of::<GeminiOptions>();
    let _ = size_of::<GeminiCachedContents>();
    let _ = size_of::<GeminiFileSearchStores>();
    let _ = size_of::<GeminiFiles>();
    let _ = size_of::<GeminiModels>();
    let _ = size_of::<GeminiTokens>();
    let _ = size_of::<GeminiVideo>();
    let _ = size_of::<GeminiThinkingConfig>();
    let _ = size_of::<GeminiMetadata>();
    let _ = size_of::<GeminiSource>();
    let _ = size_of::<GoogleErrorData>();
    let _ = size_of::<code_execution::CodeExecutionConfig>();
    let _ = size_of::<code_execution::CodeExecutionResult>();
    let _ = size_of::<tools::GeminiCustomEvent>();
    let _ = size_of::<tools::GeminiSourceEvent>();
    let _ = chat::GEMINI_2_5_FLASH;
    let _ = embedding::GEMINI_EMBEDDING_001;
    let _ = image::IMAGEN_4_0_GENERATE_001;
    let _ = video::VEO_3_1_GENERATE_PREVIEW;
    let _ = model_sets::ALL_CHAT;
    let _ = model_sets::ALL_EMBEDDING;
    let _ = model_sets::ALL_IMAGE;
    let _ = model_sets::ALL_VIDEO;
    assert!(!VERSION.is_empty());
    let _ = GeminiClient::base_url;
    let _ = GeminiClient::files;
    let _ = GeminiClient::provider_name;
    let _ = GeminiClient::set_retry_options;
    let _ = GeminiFiles::provider_name;
    let _ = GoogleProviderSettings::new();
    let _ = GoogleProviderSettings::new()
        .with_api_key("test-key")
        .with_header("x-test", "1")
        .into_builder();
    let _ = GoogleProviderSettings::new()
        .with_api_key("test-key")
        .with_name("my-gemini-proxy")
        .with_generate_id(|| "public-google-id".to_string())
        .into_builder_for_model(chat::GEMINI_2_5_FLASH);
    let _ = GoogleProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model(chat::GEMINI_2_5_FLASH);
    #[allow(deprecated)]
    let _ = GoogleGenerativeAIProviderSettings::new()
        .with_api_key("test-key")
        .into_builder_for_model(chat::GEMINI_2_5_FLASH);
    let _ = google_builder();
    let _ = create_google();
    #[allow(deprecated)]
    let _ = create_google_generative_ai();
    let _ = GoogleProviderSettings::new()
        .with_api_key("test-key")
        .into_builder()
        .files();

    fn _assert_req_ext<T: GeminiChatRequestExt>() {}
    fn _assert_image_req_ext<T: GeminiImageRequestExt>() {}
    fn _assert_google_req_ext<T: GoogleChatRequestExt>() {}
    fn _assert_google_embed_req_ext<T: GoogleEmbeddingRequestExt>() {}
    fn _assert_google_image_req_ext<T: GoogleImageRequestExt>() {}
    fn _assert_google_video_req_ext<T: GoogleVideoRequestExt>() {}
    fn _assert_resp_ext<T: GeminiChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_image_req_ext::<siumai::image::GenerateImageRequest>();
    _assert_google_req_ext::<ChatRequest>();
    _assert_google_embed_req_ext::<EmbeddingRequest>();
    _assert_google_image_req_ext::<siumai::image::GenerateImageRequest>();
    _assert_google_video_req_ext::<VideoGenerationRequest>();
    _assert_resp_ext::<ChatResponse>();
    let _ = file_search_stores::stores;

    let _ = siumai::image::GenerateImageRequest::new("draw a robot").with_gemini_image_options(
        GeminiImageOptions::new()
            .with_aspect_ratio("16:9")
            .with_person_generation("allow_all"),
    );
    let _ = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_google_options(
        GoogleLanguageModelOptions::new()
            .with_service_tier("flex")
            .with_stream_function_call_arguments(true),
    );
    let _ = EmbeddingRequest::single("hello").with_google_embedding_options(
        GoogleEmbeddingModelOptions::new()
            .with_output_dimensionality(128)
            .with_task_type(siumai::types::EmbeddingTaskType::SemanticSimilarity),
    );
    let _ = siumai::image::GenerateImageRequest::new("draw a robot").with_google_image_options(
        GoogleImageModelOptions::new()
            .with_aspect_ratio("1:1")
            .with_person_generation("allow_all"),
    );
    let _ = VideoGenerationRequest::new("veo-3.1-generate-preview", "draw a robot")
        .with_google_video_options(
            GoogleVideoModelOptions::new()
                .with_negative_prompt("no cats")
                .with_person_generation("allow_all"),
        );
    let _ = UploadFileOptions::new().with_google_upload_options(
        GoogleFilesUploadOptions::new()
            .with_display_name("spec.pdf")
            .with_poll_interval_ms(250),
    );

    let _ = siumai::provider_ext::google::hosted_tools::file_search()
        .with_file_search_store_names(vec!["fileSearchStores/store-123".to_string()])
        .build();
    let _ = siumai::provider_ext::google::hosted_tools::google_search().build();
    let _ = siumai::provider_ext::google::tools::code_execution();
    let _ = siumai::provider_ext::google::tools::enterprise_web_search();
    let _ = siumai::provider_ext::google::tools::google_maps();
    let _ = siumai::provider_ext::google::tools::url_context();
    let _ = siumai::provider_ext::google::tools::google_search();
    let _ = siumai::provider_ext::google::provider_tools::google_search();
    let _ = siumai::Provider::google();
    let _ = siumai::Provider::google().language_model(chat::GEMINI_2_5_FLASH);
    let _ = siumai::Provider::google().chat(chat::GEMINI_2_5_PRO);
    #[allow(deprecated)]
    let _ = siumai::Provider::google().generative_ai(chat::GEMINI_2_0_FLASH);
    let _ = siumai::Provider::google().embedding_model(embedding::GEMINI_EMBEDDING_001);
    let _ = siumai::Provider::google().embedding(embedding::GEMINI_EMBEDDING_2_PREVIEW);
    #[allow(deprecated)]
    let _ = siumai::Provider::google().text_embedding(embedding::GEMINI_EMBEDDING_001);
    #[allow(deprecated)]
    let _ = siumai::Provider::google().text_embedding_model(embedding::GEMINI_EMBEDDING_2_PREVIEW);
    let _ = siumai::Provider::google().image_model(image::GEMINI_2_5_FLASH_IMAGE);
    let _ = siumai::Provider::google().image(image::IMAGEN_4_0_GENERATE_001);
    let _ = siumai::Provider::google().video_model(video::VEO_3_1_GENERATE_PREVIEW);
    let _ = siumai::Provider::google().video(video::VEO_3_1_FAST_GENERATE_PREVIEW);
    let _ = siumai::Provider::google().name("my-gemini-proxy");
    let _ = siumai::Provider::google().api_key("test-key").files();
    let _ = siumai::Provider::gemini().api_key("test-key").files();
    let _ = Siumai::builder().google();
}

#[cfg(feature = "cohere")]
#[test]
#[allow(deprecated)]
fn public_surface_cohere_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::cohere::{
        CohereBuilder, CohereClient, CohereConfig, CohereProviderSettings, VERSION, chat,
        cohere as cohere_builder, create_cohere, embedding, model_sets, options::*, rerank,
    };

    let _ = size_of::<CohereBuilder>();
    let _ = size_of::<CohereClient>();
    let _ = size_of::<CohereConfig>();
    let _ = size_of::<CohereProviderSettings>();
    let _ = size_of::<CohereChatOptions>();
    let _ = size_of::<CohereLanguageModelOptions>();
    let _ = size_of::<CohereChatModelOptions>();
    let _ = size_of::<CohereThinkingConfig>();
    let _ = size_of::<CohereThinkingType>();
    let _ = size_of::<CohereEmbeddingOptions>();
    let _ = size_of::<CohereEmbeddingModelOptions>();
    let _ = size_of::<CohereEmbeddingInputType>();
    let _ = size_of::<CohereEmbeddingTruncate>();
    let _ = size_of::<CohereRerankOptions>();
    let _ = size_of::<CohereRerankingModelOptions>();
    let _ = size_of::<CohereRerankingOptions>();
    let _ = CohereClient::provider_context;
    let _ = CohereClient::base_url;
    let _ = CohereClient::http_client;
    let _ = CohereClient::retry_options;
    let _ = CohereClient::http_interceptors;
    let _ = CohereClient::http_transport;
    let _ = CohereClient::set_retry_options;
    let _ = chat::COMMAND_A_03_2025;
    let _ = embedding::EMBED_V4;
    let _ = rerank::RERANK_V3_5;
    let _ = model_sets::ALL_CHAT;
    let _ = model_sets::ALL_EMBEDDING;
    let _ = model_sets::ALL_RERANK;
    let _ = cohere_builder();
    let _ = create_cohere();
    let _ = VERSION;
    let _ = CohereProviderSettings::new();
    let _ = CohereProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/cohere")
        .with_header("x-test", "1")
        .into_builder_for_model("command-a-03-2025");
    let _ = CohereProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model("command-a-03-2025");

    fn _assert_chat_req_ext<T: CohereChatRequestExt>() {}
    fn _assert_embed_req_ext<T: CohereEmbeddingRequestExt>() {}
    fn _assert_rerank_req_ext<T: CohereRerankRequestExt>() {}
    _assert_chat_req_ext::<ChatRequest>();
    _assert_embed_req_ext::<EmbeddingRequest>();
    _assert_rerank_req_ext::<RerankRequest>();

    let chat_req = ChatRequest::new(vec![user!("hi")]).with_cohere_options(
        CohereChatOptions::new().with_thinking(
            CohereThinkingConfig::new()
                .with_type(CohereThinkingType::Enabled)
                .with_token_budget(2048),
        ),
    );
    let embed_req = EmbeddingRequest::single("hello").with_cohere_options(
        CohereEmbeddingOptions::new()
            .with_input_type(CohereEmbeddingInputType::SearchDocument)
            .with_truncate(CohereEmbeddingTruncate::End)
            .with_output_dimension(1024),
    );
    let rerank_req = RerankRequest::new(
        "rerank-v3.5".to_string(),
        "query".to_string(),
        vec!["doc-1".to_string()],
    )
    .with_cohere_options(CohereRerankOptions::new().with_priority(1));

    let _ = chat_req;
    let _ = embed_req;
    let _ = rerank_req;

    let _ = CohereConfig::new("test-key").with_model(chat::COMMAND_A_03_2025);
    let _ = siumai::Provider::cohere()
        .language_model(chat::COMMAND_A_03_2025)
        .with_http_client(reqwest::Client::new());
    let _ = siumai::Provider::cohere().embedding_model(embedding::EMBED_V4);
    let _ = siumai::Provider::cohere().reranking_model(rerank::RERANK_V3_5);
}

#[cfg(feature = "togetherai")]
#[test]
#[allow(deprecated)]
fn public_surface_togetherai_provider_ext_compiles() {
    use siumai::prelude::extensions::types::{ImageEditInput, ImageEditRequest};
    use siumai::prelude::unified::*;
    use siumai::provider_ext::togetherai::{
        TogetherAIErrorData, TogetherAIProviderSettings, TogetherAiBuilder, TogetherAiClient,
        TogetherAiConfig, VERSION, chat, completion, create_togetherai, embedding, image,
        model_sets, options::*, rerank, togetherai as togetherai_builder,
    };

    let _ = size_of::<TogetherAiBuilder>();
    let _ = size_of::<TogetherAiClient>();
    let _ = size_of::<TogetherAiConfig>();
    let _ = size_of::<TogetherAIProviderSettings>();
    let _ = size_of::<TogetherAIErrorData>();
    let _ = size_of::<TogetherAiImageOptions>();
    let _ = size_of::<TogetherAIImageModelOptions>();
    let _ = size_of::<TogetherAIImageProviderOptions>();
    let _ = size_of::<TogetherAiImageModelOptions>();
    let _ = size_of::<TogetherAiImageProviderOptions>();
    let _ = size_of::<TogetherAiRerankOptions>();
    let _ = size_of::<TogetherAIRerankingModelOptions>();
    let _ = size_of::<TogetherAIRerankingOptions>();
    let _ = size_of::<TogetherAiRerankingModelOptions>();
    let _ = size_of::<TogetherAiRerankingOptions>();
    let _ = TogetherAiClient::provider_context;
    let _ = TogetherAiClient::base_url;
    let _ = TogetherAiClient::http_client;
    let _ = TogetherAiClient::retry_options;
    let _ = TogetherAiClient::http_interceptors;
    let _ = TogetherAiClient::http_transport;
    let _ = TogetherAiClient::set_retry_options;
    let _ = chat::META_LLAMA_3_1_8B_INSTRUCT_TURBO;
    let _ = completion::QWEN_2_5_CODER_32B_INSTRUCT;
    let _ = embedding::M2_BERT_80M_8K_RETRIEVAL;
    let _ = image::FLUX_1_SCHNELL;
    let _ = rerank::LLAMA_RANK_V1;
    let _ = model_sets::ALL_CHAT;
    let _ = model_sets::ALL_COMPLETION;
    let _ = model_sets::ALL_EMBEDDING;
    let _ = model_sets::ALL_IMAGE;
    let _ = model_sets::ALL_RERANK;
    let _ = togetherai_builder();
    let _ = create_togetherai();
    let _ = VERSION;
    let _ = TogetherAIProviderSettings::new();
    let _ = TogetherAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/together")
        .with_header("x-test", "1")
        .into_builder_for_model("Salesforce/Llama-Rank-v1");
    let _ = TogetherAIProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model("Salesforce/Llama-Rank-v1");
    let decoded_error: TogetherAIErrorData = serde_json::from_value(serde_json::json!({
        "error": {
            "message": "bad request"
        }
    }))
    .expect("decode togetherai error data");
    assert_eq!(decoded_error.error.message, "bad request");

    let req = RerankRequest::new(
        rerank::LLAMA_RANK_V1.to_string(),
        "query".to_string(),
        vec!["doc-1".to_string()],
    )
    .with_togetherai_options(
        TogetherAIRerankingModelOptions::new().with_rank_fields(vec!["example".to_string()]),
    );
    let _ = req;

    let image_req = ImageGenerationRequest {
        prompt: "draw a robot".to_string(),
        ..Default::default()
    }
    .with_togetherai_image_options(
        TogetherAIImageModelOptions::new()
            .with_steps(12)
            .with_negative_prompt("blurry"),
    );
    let _ = image_req;

    let image_edit_req = ImageEditRequest {
        prompt: "edit this robot".to_string(),
        images: vec![ImageEditInput::url("https://example.com/input.png")],
        ..Default::default()
    }
    .with_togetherai_image_options(
        TogetherAIImageModelOptions::new().with_disable_safety_checker(true),
    );
    let _ = image_edit_req;

    let _ = siumai::image::GenerateImageRequest::new("draw a robot").with_togetherai_image_options(
        TogetherAIImageModelOptions::new()
            .with_steps(12)
            .with_negative_prompt("blurry"),
    );

    let _ = Provider::togetherai().model(chat::META_LLAMA_3_1_8B_INSTRUCT_TURBO);
    let _ = TogetherAiConfig::new("test-key").with_model(rerank::LLAMA_RANK_V1);
}

#[cfg(feature = "togetherai")]
#[test]
#[allow(deprecated)]
fn public_surface_togetherai_unified_builder_compiles() {
    use siumai::prelude::compat::{Provider, Siumai};

    let _ = Provider::togetherai;
    let _ = Siumai::builder().togetherai();
    let _ = Provider::togetherai().model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo");
}

#[cfg(feature = "deepinfra")]
#[test]
#[allow(deprecated)]
fn public_surface_deepinfra_unified_builder_compiles() {
    use siumai::prelude::compat::{Provider, Siumai};
    use siumai::provider_ext::deepinfra::{
        DeepInfraChatModelId, DeepInfraClient, DeepInfraCompletionModelId, DeepInfraConfig,
        DeepInfraEmbeddingModelId, DeepInfraErrorData, DeepInfraImageModelId,
        DeepInfraProviderSettings, VERSION, chat, completion, create_deepinfra, deepinfra,
        embedding, image, model_sets,
    };

    let _ = size_of::<DeepInfraChatModelId>();
    let _ = size_of::<DeepInfraClient>();
    let _ = size_of::<DeepInfraCompletionModelId>();
    let _ = size_of::<DeepInfraConfig>();
    let _ = size_of::<DeepInfraProviderSettings>();
    let _ = VERSION;
    let _ = size_of::<DeepInfraEmbeddingModelId>();
    let _ = size_of::<DeepInfraErrorData>();
    let _ = size_of::<DeepInfraImageModelId>();
    let _ = Provider::deepinfra;
    let _ = Siumai::builder().deepinfra();
    let _ = deepinfra();
    let _ = create_deepinfra();
    let _ = DeepInfraProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/deepinfra")
        .with_header("x-test", "1")
        .into_builder_for_model(chat::LLAMA_V3P3_70B_INSTRUCT);
    let _ = DeepInfraProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model(chat::LLAMA_V3P3_70B_INSTRUCT);
    let _ = chat::LLAMA_V3P3_70B_INSTRUCT;
    let _ = completion::LLAMA_V3P3_70B_INSTRUCT;
    let _ = embedding::BGE_BASE_EN_V1_5;
    let _ = image::FLUX_1_SCHNELL;
    let _ = model_sets::ALL_CHAT;
    let _ = model_sets::ALL_COMPLETION;
    let _ = model_sets::ALL_EMBEDDING;
    let _ = model_sets::ALL_IMAGE;
    let _ = Provider::deepinfra().model(chat::LLAMA_V3P3_70B_INSTRUCT);
}

#[cfg(feature = "openai")]
#[test]
#[allow(deprecated)]
fn public_surface_ai_sdk_compat_promoted_unified_builders_compile() {
    use siumai::prelude::compat::{Provider, Siumai};
    use siumai::provider_ext::fireworks::{
        FireworksProviderSettings, create_fireworks, fireworks as fireworks_builder,
    };
    use siumai::provider_ext::mistral::{
        MistralProviderSettings, create_mistral, mistral as mistral_builder,
    };
    use siumai::provider_ext::moonshotai::{
        MoonshotAIProviderSettings, create_moonshotai, moonshotai as moonshotai_builder,
    };
    use siumai::provider_ext::perplexity::{
        PerplexityProviderSettings, create_perplexity, perplexity as perplexity_builder,
    };

    let _ = Provider::mistral;
    let _ = Provider::fireworks;
    let _ = Provider::perplexity;
    let _ = Provider::moonshotai;
    let _ = Siumai::builder().mistral();
    let _ = Siumai::builder().fireworks();
    let _ = Siumai::builder().perplexity();
    let _ = Siumai::builder().moonshotai();
    let _ = mistral_builder();
    let _ = create_mistral();
    let _ = MistralProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/mistral")
        .with_header("x-test", "1")
        .into_builder_for_model("mistral-large-latest");
    let _ = MistralProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model("mistral-large-latest");
    let _ = fireworks_builder();
    let _ = create_fireworks();
    let _ = FireworksProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/fireworks")
        .with_header("x-test", "1")
        .into_config_for_model("accounts/fireworks/models/llama-v3p1-8b-instruct");
    let _ = FireworksProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/fireworks")
        .with_header("x-test", "1")
        .into_builder_for_model("accounts/fireworks/models/llama-v3p1-8b-instruct");
    let _ = perplexity_builder();
    let _ = create_perplexity();
    let _ = PerplexityProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/perplexity")
        .with_header("x-test", "1")
        .into_config_for_model("sonar");
    let _ = PerplexityProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/perplexity")
        .with_header("x-test", "1")
        .into_builder_for_model("sonar");
    let _ = moonshotai_builder();
    let _ = create_moonshotai();
    let _ = MoonshotAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/moonshot")
        .with_header("x-test", "1")
        .into_builder_for_model("kimi-k2.5");
    let _ = Provider::mistral().model("mistral-large-latest");
    let _ = Provider::fireworks().model("accounts/fireworks/models/llama-v3p1-8b-instruct");
    let _ = Provider::perplexity().model("sonar");
    let _ = Provider::moonshotai().model("kimi-k2.5");
}

#[cfg(feature = "openai")]
#[test]
#[allow(deprecated)]
fn public_surface_moonshotai_provider_ext_compile() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::moonshotai::{
        MoonshotAIChatModelId, MoonshotAIChatOptions, MoonshotAIChatRequestExt, MoonshotAIClient,
        MoonshotAIConfig, MoonshotAILanguageModelOptions, MoonshotAIProviderOptions,
        MoonshotAIProviderSettings, MoonshotAIReasoningHistory, MoonshotAIThinkingConfig,
        MoonshotAIThinkingType, VERSION, create_moonshotai, model_sets,
        moonshotai as moonshotai_builder, recommended,
    };

    let _ = size_of::<MoonshotAIClient>();
    let _ = size_of::<MoonshotAIConfig>();
    let _ = size_of::<MoonshotAIProviderSettings>();
    let _ = VERSION;
    let _ = size_of::<MoonshotAIChatModelId>();
    let _ = size_of::<MoonshotAIChatOptions>();
    let _ = size_of::<MoonshotAILanguageModelOptions>();
    let _ = size_of::<MoonshotAIProviderOptions>();

    let req = ChatRequest::new(vec![user!("hi")]).with_moonshotai_options(
        MoonshotAIChatOptions::new()
            .with_thinking(
                MoonshotAIThinkingConfig::new()
                    .with_type(MoonshotAIThinkingType::Enabled)
                    .with_budget_tokens(2048),
            )
            .with_reasoning_history(MoonshotAIReasoningHistory::Interleaved),
    );
    let _ = req;

    let _ = recommended::CHAT;
    let _ = model_sets::KIMI_K2P5;
    let _ = moonshotai_builder();
    let _ = create_moonshotai();
    let _ = MoonshotAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/moonshot")
        .with_header("x-test", "1")
        .into_config_for_model("kimi-k2.5");
}

#[cfg(feature = "bedrock")]
#[test]
#[allow(deprecated)]
fn public_surface_bedrock_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::bedrock::{
        AmazonBedrockProviderSettings, BedrockBuilder, BedrockClient, BedrockConfig,
        BedrockEmbeddingRequestExt, BedrockMessageExt, BedrockRequestContentPartExt, VERSION,
        assistant_message_with_reasoning_metadata, bedrock as bedrock_builder,
        create_amazon_bedrock, metadata::*, options::*,
    };

    let _ = size_of::<AmazonBedrockProviderSettings>();
    let _ = size_of::<BedrockBuilder>();
    let _ = size_of::<BedrockClient>();
    let _ = size_of::<BedrockConfig>();
    let _ = size_of::<BedrockChatOptions>();
    let _ = size_of::<BedrockEmbeddingOptions>();
    let _ = size_of::<AmazonBedrockEmbeddingModelOptions>();
    let _ = size_of::<BedrockEmbeddingInputType>();
    let _ = size_of::<BedrockEmbeddingPurpose>();
    let _ = size_of::<BedrockEmbeddingTruncate>();
    let _ = size_of::<AmazonBedrockLanguageModelOptions>();
    #[allow(deprecated)]
    let _ = size_of::<BedrockProviderOptions>();
    let _ = size_of::<BedrockRerankOptions>();
    let _ = size_of::<AmazonBedrockRerankingModelOptions>();
    #[allow(deprecated)]
    let _ = size_of::<BedrockRerankingOptions>();
    let _ = size_of::<BedrockMetadata>();
    let _ = size_of::<BedrockReasoningContentPartMetadata>();
    let _ = size_of::<BedrockCachePoint>();
    let _ = size_of::<BedrockCachePointConfig>();
    #[cfg(feature = "anthropic")]
    let _ = size_of::<AnthropicProviderOptions>();
    let _ = BedrockClient::runtime_base_url;
    let _ = BedrockClient::agent_runtime_base_url;
    let _ = BedrockClient::chat_provider_context;
    let _ = BedrockClient::rerank_provider_context;
    let _ = BedrockClient::http_client;
    let _ = BedrockClient::retry_options;
    let _ = BedrockClient::http_interceptors;
    let _ = BedrockClient::http_transport;
    let _ = BedrockClient::set_retry_options;
    let _ = bedrock_builder();
    let _ = create_amazon_bedrock();
    let _ = VERSION;
    let _ = AmazonBedrockProviderSettings::new();
    let _ = AmazonBedrockProviderSettings::new()
        .with_api_key("test-key")
        .with_region("us-west-2")
        .with_header("x-test", "1")
        .into_builder_for_model("amazon.nova-lite-v1:0");
    let _ = AmazonBedrockProviderSettings::new()
        .with_base_url("https://bedrock-runtime.us-east-1.amazonaws.com")
        .into_config_for_model("amazon.nova-lite-v1:0");

    let chat_req = ChatRequest::new(vec![user!("hi")]).with_bedrock_chat_options(
        BedrockChatOptions::new()
            .with_additional_model_request_fields(serde_json::json!({ "topK": 16 })),
    );
    let embed_req = EmbeddingRequest::single("bedrock embed").with_bedrock_embedding_options(
        BedrockEmbeddingOptions::new()
            .with_dimensions(512)
            .with_normalize(true)
            .with_input_type(BedrockEmbeddingInputType::SearchDocument)
            .with_truncate(BedrockEmbeddingTruncate::End),
    );
    let rerank_req = RerankRequest::new(
        "amazon.rerank-v1:0".to_string(),
        "query".to_string(),
        vec!["doc-1".to_string()],
    )
    .with_bedrock_rerank_options(
        BedrockRerankOptions::new()
            .with_region("us-east-1")
            .with_next_token("token-1"),
    );
    let mut resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut inner = std::collections::HashMap::new();
    inner.insert(
        "isJsonResponseFromTool".to_string(),
        serde_json::Value::Bool(true),
    );
    let mut outer = std::collections::HashMap::new();
    outer.insert(
        "bedrock".to_string(),
        serde_json::Value::Object(inner.into_iter().collect()),
    );
    resp.provider_metadata = Some(outer);
    let _ = resp.bedrock_metadata();
    let _ = assistant_message_with_reasoning_metadata;
    let _ = ChatMessage::user("cached")
        .with_bedrock_cache_point(BedrockCachePoint::new().with_ttl(BedrockCacheTtl::OneHour))
        .build();
    let _ = ContentPart::file_base64("AAECAw==", "application/pdf", None)
        .with_bedrock_document_citations(true);

    let _ = (chat_req, embed_req, rerank_req);
    let _ = siumai::Provider::bedrock().embedding("amazon.titan-embed-text-v2:0");
    let _ = siumai::Provider::bedrock().embedding_model("amazon.titan-embed-text-v2:0");
    let _ = siumai::Provider::bedrock().image("amazon.nova-canvas-v1:0");
    let _ = siumai::Provider::bedrock().image_model("amazon.nova-canvas-v1:0");
    let _ = siumai::Provider::bedrock().reranking("amazon.rerank-v1:0");
    let _ = siumai::Provider::bedrock().reranking_model("amazon.rerank-v1:0");
    let _ = siumai::Provider::bedrock().text_embedding("amazon.titan-embed-text-v2:0");
    let _ = siumai::Provider::bedrock().text_embedding_model("amazon.titan-embed-text-v2:0");
    #[cfg(feature = "anthropic")]
    {
        let _ = siumai::provider_ext::bedrock::tools::web_search_20260209();
        let _ = siumai::provider_ext::bedrock::provider_tools::computer_20251124();
    }
}

#[cfg(feature = "openai")]
#[test]
#[allow(deprecated)]
fn public_surface_mistral_fireworks_perplexity_provider_ext_compile() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::fireworks::{
        FireworksChatOptions, FireworksChatRequestExt, FireworksClient, FireworksConfig,
        FireworksEmbeddingModelId, FireworksEmbeddingModelOptions,
        FireworksEmbeddingProviderOptions, FireworksErrorData, FireworksImageModelId,
        FireworksProviderOptions, FireworksProviderSettings, FireworksReasoningHistory,
        FireworksThinkingConfig, FireworksThinkingType, VERSION as FIREWORKS_VERSION,
        chat as fireworks_chat, create_fireworks, fireworks as fireworks_builder,
    };
    use siumai::provider_ext::mistral::{
        MistralChatOptions, MistralChatRequestExt, MistralClient, MistralConfig,
        MistralProviderSettings, MistralReasoningEffort, VERSION as MISTRAL_VERSION,
        chat as mistral_chat, create_mistral, embedding as mistral_embedding,
        mistral as mistral_builder,
    };
    use siumai::provider_ext::perplexity::{
        PerplexityChatRequestExt, PerplexityClient, PerplexityConfig, PerplexityOptions,
        PerplexityProviderSettings, VERSION as PERPLEXITY_VERSION, chat as perplexity_chat,
        create_perplexity, perplexity as perplexity_builder,
    };

    let _ = size_of::<MistralClient>();
    let _ = size_of::<MistralConfig>();
    let _ = size_of::<MistralProviderSettings>();
    let _ = MISTRAL_VERSION;
    let _ = size_of::<MistralChatOptions>();
    let _ = size_of::<MistralReasoningEffort>();
    let _ = size_of::<FireworksClient>();
    let _ = size_of::<FireworksConfig>();
    let _ = size_of::<FireworksProviderSettings>();
    let _ = FIREWORKS_VERSION;
    let _ = size_of::<FireworksChatOptions>();
    let _ = size_of::<FireworksErrorData>();
    let _ = size_of::<FireworksEmbeddingModelId>();
    let _ = size_of::<FireworksEmbeddingModelOptions>();
    let _ = size_of::<FireworksEmbeddingProviderOptions>();
    let _ = size_of::<FireworksImageModelId>();
    let _ = size_of::<FireworksProviderOptions>();
    let _ = size_of::<PerplexityClient>();
    let _ = size_of::<PerplexityConfig>();
    let _ = size_of::<PerplexityProviderSettings>();
    let _ = PERPLEXITY_VERSION;
    let _ = size_of::<PerplexityOptions>();

    let _ = mistral_chat::MISTRAL_LARGE_LATEST;
    let _ = mistral_embedding::MISTRAL_EMBED;
    let _ = fireworks_chat::LLAMA_V3P1_8B_INSTRUCT;
    let _ = perplexity_chat::SONAR;
    let _ = mistral_builder();
    let _ = create_mistral();
    let _ = fireworks_builder();
    let _ = create_fireworks();
    let _ = perplexity_builder();
    let _ = create_perplexity();

    let _ = ChatRequest::new(vec![user!("hi")])
        .with_mistral_options(
            MistralChatOptions::new()
                .with_safe_prompt(true)
                .with_reasoning_effort(MistralReasoningEffort::High),
        )
        .with_fireworks_options(
            FireworksChatOptions::new()
                .with_thinking(
                    FireworksThinkingConfig::new()
                        .with_type(FireworksThinkingType::Enabled)
                        .with_budget_tokens(2048),
                )
                .with_reasoning_history(FireworksReasoningHistory::Interleaved),
        )
        .with_perplexity_options(PerplexityOptions::new().with_return_images(true));
}

#[cfg(feature = "google-vertex")]
#[test]
#[allow(deprecated)]
fn public_surface_google_vertex_provider_ext_compiles() {
    use siumai::compat::Siumai;
    use siumai::prelude::unified::{ChatResponse, ContentPart, MessageContent};
    use siumai::provider_ext::google_vertex::{
        GoogleVertexBuilder, GoogleVertexClient, GoogleVertexConfig, GoogleVertexProviderSettings,
        SharedIdGenerator, VERSION, chat, create_vertex, embedding, image, metadata::*, model_sets,
        options::*, vertex as vertex_builder, video,
    };

    let _ = size_of::<GoogleVertexBuilder>();
    let _ = size_of::<GoogleVertexClient>();
    let _ = size_of::<GoogleVertexConfig>();
    let _ = size_of::<GoogleVertexProviderSettings>();
    let _ = size_of::<SharedIdGenerator>();
    let _ = size_of::<GoogleVertexEmbeddingModelOptions>();
    let _ = size_of::<GoogleVertexImageModelOptions>();
    let _ = size_of::<GoogleVertexImageProviderOptions>();
    let _ = size_of::<GoogleVertexReferenceImage>();
    let _ = size_of::<GoogleVertexVideoModelId>();
    let _ = size_of::<GoogleVertexVideoModelOptions>();
    let _ = size_of::<GoogleVertexVideoProviderOptions>();
    let _ = size_of::<VertexEmbeddingOptions>();
    let _ = size_of::<VertexImagenEditMode>();
    let _ = size_of::<VertexImagenOptions>();
    let _ = size_of::<VertexImagenMaskMode>();
    let _ = size_of::<VertexMetadata>();
    let _ = size_of::<VertexSource>();
    let _ = size_of::<VertexGroundingMetadata>();
    let _ = size_of::<VertexImagenSafetySetting>();
    let _ = size_of::<VertexImagenSampleImageSize>();
    let _ = size_of::<VertexPersonGeneration>();
    let _ = size_of::<VertexUrlContextMetadata>();
    let _ = size_of::<VertexUsageMetadata>();
    let _ = size_of::<VertexSafetyRating>();
    let _ = VERSION;
    let _ = GoogleVertexClient::base_url;
    let _ = chat::GEMINI_2_5_FLASH;
    let _ = embedding::TEXT_EMBEDDING_005;
    let _ = image::IMAGEN_3_0_EDIT_001;
    let _ = image::IMAGEN_4_0_ULTRA_GENERATE_001;
    let _ = image::GEMINI_2_5_FLASH_IMAGE;
    let _ = video::VEO_3_1_GENERATE_PREVIEW;
    let _ = model_sets::ALL_CHAT;
    let _ = model_sets::ALL_EMBEDDING;
    let _ = model_sets::ALL_IMAGE;
    let _ = model_sets::ALL_VIDEO;
    let _ = vertex_builder();
    let _ = create_vertex();
    let _ = GoogleVertexProviderSettings::new();
    let _ = GoogleVertexProviderSettings::new()
        .with_api_key("test-key")
        .with_generate_id(|| "vertex-provider-settings-id".to_string())
        .with_header("x-test", "1")
        .into_builder();
    let _ = GoogleVertexProviderSettings::new()
        .with_project("demo-project")
        .with_location("global")
        .into_builder_for_model(chat::GEMINI_2_5_FLASH);
    let _ = GoogleVertexProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model(chat::GEMINI_2_5_FLASH);
    let _ = VertexEmbeddingOptions::new()
        .with_output_dimensionality(256)
        .with_task_type(siumai::types::EmbeddingTaskType::RetrievalDocument)
        .with_title("vertex-doc")
        .with_auto_truncate(true);
    let _ = vertex_builder().language_model(chat::GEMINI_2_5_FLASH);
    let _ = vertex_builder().embedding_model(embedding::TEXT_EMBEDDING_005);
    let _ = vertex_builder().with_generate_id(|| "vertex-builder-id".to_string());
    let shared_generate_id: SharedIdGenerator =
        std::sync::Arc::new(|| "vertex-shared-id".to_string());
    let _ = vertex_builder().with_shared_generate_id(shared_generate_id.clone());
    #[allow(deprecated)]
    let _ = vertex_builder().text_embedding_model(embedding::GEMINI_EMBEDDING_2_PREVIEW);
    let _ = vertex_builder().image(image::IMAGEN_4_0_ULTRA_GENERATE_001);
    let _ = vertex_builder().image_model(image::GEMINI_2_5_FLASH_IMAGE);
    let _ = vertex_builder().video(video::VEO_3_1_GENERATE_PREVIEW);
    let _ = vertex_builder().video_model(video::VEO_3_1_FAST_GENERATE_001);
    let _ = GoogleVertexConfig::new("https://example.com/custom", chat::GEMINI_2_5_FLASH)
        .with_generate_id(|| "vertex-config-id".to_string());
    let _ = GoogleVertexConfig::new("https://example.com/custom", chat::GEMINI_2_5_FLASH)
        .with_shared_generate_id(shared_generate_id);

    fn _assert_resp_ext<T: VertexChatResponseExt>() {}
    fn _assert_part_ext<T: VertexContentPartExt>() {}
    fn _assert_image_req_ext<T: VertexImagenRequestExt>() {}
    fn _assert_video_req_ext<T: VertexVideoRequestExt>() {}
    _assert_resp_ext::<ChatResponse>();
    _assert_part_ext::<ContentPart>();
    _assert_image_req_ext::<siumai::image::GenerateImageRequest>();
    _assert_video_req_ext::<siumai::extensions::types::VideoGenerationRequest>();

    let mut resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut inner = std::collections::HashMap::new();
    inner.insert(
        "usageMetadata".to_string(),
        serde_json::json!({
            "totalTokenCount": 8
        }),
    );
    let mut outer = std::collections::HashMap::new();
    outer.insert(
        "vertex".to_string(),
        serde_json::Value::Object(inner.into_iter().collect()),
    );
    resp.provider_metadata = Some(outer);
    let typed = resp.vertex_metadata().expect("vertex metadata");
    assert_eq!(
        typed
            .usage_metadata
            .as_ref()
            .and_then(|usage| usage.total_token_count),
        Some(8)
    );

    let _ = siumai::provider_ext::google_vertex::tools::code_execution();
    let _ = siumai::provider_ext::google_vertex::tools::google_search();
    let _ = siumai::provider_ext::google_vertex::tools::enterprise_web_search();
    let _ = siumai::provider_ext::google_vertex::tools::file_search(vec![
        "fileSearchStores/store-123".to_string(),
    ]);
    let _ = siumai::provider_ext::google_vertex::tools::google_maps();
    let _ = siumai::provider_ext::google_vertex::tools::url_context();
    let _ = siumai::provider_ext::google_vertex::hosted_tools::file_search()
        .with_file_search_store_names(vec!["fileSearchStores/store-123".to_string()])
        .build();
    let _ = siumai::provider_ext::google_vertex::tools::vertex_rag_store(
        "projects/demo/locations/us-central1/ragCorpora/test-corpus",
    );
    let _ = siumai::provider_ext::google_vertex::hosted_tools::google_search().build();
    let _ = siumai::provider_ext::google_vertex::hosted_tools::vertex_rag_store(
        "projects/demo/locations/us-central1/ragCorpora/test-corpus",
    )
    .build();
    let _ = siumai::image::GenerateImageRequest::new("draw a robot").with_vertex_imagen_options(
        VertexImagenOptions::new()
            .with_negative_prompt("blurry")
            .with_person_generation(VertexPersonGeneration::AllowAdult)
            .with_safety_setting(VertexImagenSafetySetting::BlockMediumAndAbove)
            .with_add_watermark(false)
            .with_sample_image_size(VertexImagenSampleImageSize::TwoK),
    );
    let _ = siumai::extensions::types::VideoGenerationRequest::new(
        "veo-3.1-generate-preview",
        "animate a tiny robot",
    )
    .with_vertex_video_options(
        GoogleVertexVideoModelOptions::new()
            .with_negative_prompt("blurry")
            .with_generate_audio(true)
            .with_reference_images(vec![
                GoogleVertexReferenceImage::new().with_gcs_uri("gs://bucket/reference.png"),
            ]),
    );
    let _ = siumai::Provider::vertex();
    let _ = siumai::Provider::vertex_maas();
    let _ = Siumai::builder().vertex();
    let _ = Siumai::builder().vertex_maas();
}

#[cfg(feature = "google-vertex")]
#[test]
#[allow(deprecated)]
fn public_surface_vertex_maas_provider_ext_compiles() {
    use siumai::prelude::compat::{Provider, Siumai};
    use siumai::provider_ext::vertex_maas::{
        GoogleVertexMaasClient, GoogleVertexMaasConfig, GoogleVertexMaasModelId,
        GoogleVertexMaasProviderSettings, VERSION, chat, completion, create_vertex_maas, embedding,
        model_sets, vertex_maas as vertex_maas_builder,
    };

    let _ = size_of::<GoogleVertexMaasClient>();
    let _ = size_of::<GoogleVertexMaasConfig>();
    let _ = size_of::<GoogleVertexMaasModelId>();
    let _ = size_of::<GoogleVertexMaasProviderSettings>();
    let _ = VERSION;
    let _ = chat::DEEPSEEK_V3_2_MAAS;
    let _ = completion::DEEPSEEK_V3_2_MAAS;
    let _ = embedding::DEEPSEEK_V3_2_MAAS;
    let _ = model_sets::ALL_CHAT;
    let _ = model_sets::ALL_COMPLETION;
    let _ = model_sets::ALL_EMBEDDING;
    let _ = vertex_maas_builder();
    let _ = create_vertex_maas();
    let _ = GoogleVertexMaasProviderSettings::new()
        .with_project("test-project")
        .with_location("us-central1")
        .with_header("Authorization", "Bearer test-token")
        .into_builder_for_model(chat::DEEPSEEK_V3_2_MAAS);
    let _ = GoogleVertexMaasProviderSettings::new()
        .with_project("test-project")
        .with_location("us-central1")
        .with_header("Authorization", "Bearer test-token")
        .into_config_for_model(chat::DEEPSEEK_V3_2_MAAS);
    let _ = Provider::vertex_maas();
    let _ = Siumai::builder().vertex_maas();
    let _ = Provider::vertex_maas().model(chat::DEEPSEEK_V3_2_MAAS);
}

#[cfg(feature = "google-vertex")]
#[test]
#[allow(deprecated)]
fn public_surface_anthropic_vertex_provider_ext_compiles() {
    use siumai::prelude::unified::{ChatRequest, ChatResponse, MessageContent, Siumai};
    use siumai::provider_ext::anthropic_vertex::{
        AnthropicChatResponseExt, AnthropicMessageContainerMetadata,
        AnthropicMessageContainerSkill, AnthropicMessageMetadata, AnthropicMetadata,
        AnthropicUsageIteration, GoogleVertexAnthropicMessagesModelId,
        GoogleVertexAnthropicProviderSettings, VertexAnthropicBuilder,
        VertexAnthropicChatRequestExt, VertexAnthropicClient, VertexAnthropicConfig,
        VertexAnthropicOptions, VertexAnthropicStructuredOutputMode, VertexAnthropicThinkingMode,
        chat, create_vertex_anthropic, model_sets, vertex_anthropic as vertex_anthropic_builder,
    };

    let _ = size_of::<VertexAnthropicBuilder>();
    let _ = size_of::<VertexAnthropicClient>();
    let _ = size_of::<VertexAnthropicConfig>();
    let _ = size_of::<GoogleVertexAnthropicMessagesModelId>();
    let _ = size_of::<GoogleVertexAnthropicProviderSettings>();
    let _ = size_of::<AnthropicMessageMetadata>();
    let _ = size_of::<AnthropicMessageContainerMetadata>();
    let _ = size_of::<AnthropicMessageContainerSkill>();
    let _ = size_of::<AnthropicMetadata>();
    let _ = size_of::<AnthropicUsageIteration>();
    let _ = size_of::<VertexAnthropicOptions>();
    let _ = size_of::<VertexAnthropicThinkingMode>();
    let _ = size_of::<VertexAnthropicStructuredOutputMode>();
    let _ = chat::CLAUDE_SONNET_4_6;
    let _ = chat::CLAUDE_OPUS_4_7;
    let _ = chat::CLAUDE_3_5_SONNET_V2_AT_20241022;
    let _ = model_sets::ALL_CHAT;
    let _ = vertex_anthropic_builder();
    let _ = create_vertex_anthropic();
    let _ = GoogleVertexAnthropicProviderSettings::new();
    let _ = GoogleVertexAnthropicProviderSettings::new()
        .with_project("demo-project")
        .with_location("global")
        .with_header("x-test", "1")
        .into_builder_for_model(chat::CLAUDE_SONNET_4_6);
    let _ = GoogleVertexAnthropicProviderSettings::new()
        .with_base_url("https://example.com/custom/")
        .into_config_for_model(chat::CLAUDE_SONNET_4_6);
    let _ = ChatRequest::new(vec![]).with_anthropic_vertex_options(
        VertexAnthropicOptions::new()
            .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048))),
    );
    let _ = ChatResponse::new(MessageContent::Text(String::new())).anthropic_metadata();
    let _ = ChatResponse::new(MessageContent::Text(String::new())).anthropic_message_metadata();
    let _ = VertexAnthropicConfig::new("https://example.com/custom", chat::CLAUDE_SONNET_4_6)
        .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_disable_parallel_tool_use(true)
        .with_send_reasoning(false);
    let _ = siumai::provider_ext::anthropic_vertex::tools::bash_20241022();
    let _ = siumai::provider_ext::anthropic_vertex::tools::bash_20250124();
    let _ = siumai::provider_ext::anthropic_vertex::tools::text_editor_20241022();
    let _ = siumai::provider_ext::anthropic_vertex::tools::text_editor_20250124();
    let _ = siumai::provider_ext::anthropic_vertex::tools::text_editor_20250429();
    let _ = siumai::provider_ext::anthropic_vertex::tools::text_editor_20250728();
    let _ = siumai::provider_ext::anthropic_vertex::tools::computer_20241022();
    let _ = siumai::provider_ext::anthropic_vertex::tools::web_search_20250305();
    let _ = siumai::provider_ext::anthropic_vertex::tools::tool_search_regex_20251119();
    let _ = siumai::provider_ext::anthropic_vertex::tools::tool_search_bm25_20251119();
    let _ = siumai::provider_ext::anthropic_vertex::hosted_tools::web_search_20250305()
        .with_user_location_typed(
            siumai::provider_ext::anthropic_vertex::hosted_tools::UserLocation::approximate()
                .with_country("US"),
        )
        .build();
    let _ = siumai::provider_ext::anthropic_vertex::provider_tools::web_search_20250305();
    let _ = siumai::provider_ext::anthropic_vertex::provider_tools::tool_search_regex_20251119();
    let _ = siumai::provider_ext::anthropic_vertex::provider_tools::tool_search_bm25_20251119();
    let _ = siumai::Provider::anthropic_vertex()
        .project("demo-project")
        .location("global")
        .base_url("https://example.com/custom")
        .model(chat::CLAUDE_SONNET_4_6)
        .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_disable_parallel_tool_use(true)
        .with_send_reasoning(false);
    let _ = siumai::Provider::vertex_anthropic();
    let _ = Siumai::builder()
        .anthropic_vertex()
        .project("demo-project")
        .location("global")
        .model(chat::CLAUDE_SONNET_4_6)
        .with_anthropic_vertex_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_anthropic_vertex_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_anthropic_vertex_disable_parallel_tool_use(true)
        .with_anthropic_vertex_send_reasoning(false);
    let _ = Siumai::builder().vertex_anthropic();
    let _ = siumai::Provider::anthropic_vertex();
}

#[cfg(feature = "protocol-gemini")]
#[test]
fn public_surface_protocol_gemini_compiles() {
    use siumai::prelude::unified::*;
    use siumai::protocol::gemini::*;

    let _ = size_of::<GeminiChatStandard>();
    let _ = size_of::<ChatRequest>();
}

#[cfg(feature = "groq")]
#[test]
fn public_surface_groq_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::groq::{
        GroqBuilder, GroqClient, GroqProviderSettings, VERSION, create_groq, ext::audio_options,
        groq as groq_builder, metadata::*, options::*,
    };

    let _ = size_of::<GroqBuilder>();
    let _ = size_of::<GroqClient>();
    let _ = size_of::<GroqProviderSettings>();
    let _ = VERSION;
    let _ = size_of::<GroqOptions>();
    let _ = size_of::<GroqLanguageModelOptions>();
    #[allow(deprecated)]
    let _ = size_of::<GroqProviderOptions>();
    let _ = size_of::<GroqTranscriptionModelOptions>();
    let _ = size_of::<GroqReasoningEffort>();
    let _ = size_of::<GroqReasoningFormat>();
    let _ = size_of::<GroqServiceTier>();
    let _ = GroqClient::provider_context;
    let _ = GroqClient::base_url;
    let _ = GroqClient::http_client;
    let _ = GroqClient::retry_options;
    let _ = GroqClient::http_interceptors;
    let _ = GroqClient::http_transport;
    let _ = GroqClient::set_retry_options;
    let _ = groq_builder();
    let _ = create_groq();
    let _ = GroqProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/groq")
        .with_header("x-test", "1")
        .into_builder_for_model("openai/gpt-oss-20b");
    let _ = GroqProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model("openai/gpt-oss-20b");
    let _ = siumai::Provider::groq().headers(Default::default());
    let _ = size_of::<audio_options::GroqTtsOptions>();
    let _ = size_of::<audio_options::GroqSttOptions>();
    let _ = size_of::<GroqMetadata>();
    let _ = size_of::<GroqSource>();
    let _ = size_of::<GroqSourceMetadata>();
    let _ = siumai::provider_ext::groq::tools::browser_search();
    let _ = siumai::provider_ext::groq::provider_tools::browser_search();
    let req = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
        .with_groq_stt_options(audio_options::GroqSttOptions::new().with_language("en"));
    let _ = req;

    fn _assert_req_ext<T: GroqChatRequestExt>() {}
    fn _assert_stt_req_ext<T: GroqSttRequestExt>() {}
    fn _assert_resp_ext<T: GroqChatResponseExt>() {}
    fn _assert_source_ext<T: GroqSourceExt>() {}
    _assert_req_ext::<siumai::prelude::unified::ChatRequest>();
    _assert_stt_req_ext::<SttRequest>();
    _assert_resp_ext::<ChatResponse>();
    _assert_source_ext::<GroqSource>();
}

#[cfg(feature = "xai")]
#[test]
fn public_surface_xai_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::xai::{
        VERSION, XaiBuilder, XaiClient, XaiConfig, XaiErrorData, XaiProviderSettings,
        XaiVideoModelId, create_xai, metadata::*, options::*, xai as xai_builder,
    };
    use std::collections::HashMap;

    let _ = size_of::<XaiBuilder>();
    let _ = size_of::<XaiClient>();
    let _ = size_of::<XaiConfig>();
    let _ = size_of::<XaiErrorData>();
    let _ = size_of::<XaiProviderSettings>();
    let _ = VERSION;
    let _ = size_of::<XaiChatOptions>();
    let _ = size_of::<XaiLanguageModelChatOptions>();
    #[allow(deprecated)]
    let _ = size_of::<XaiProviderOptions>();
    let _ = size_of::<XaiChatReasoningEffort>();
    let _ = size_of::<XaiFilesOptions>();
    let _ = size_of::<XaiImageOptions>();
    let _ = size_of::<XaiImageModelOptions>();
    #[allow(deprecated)]
    let _ = size_of::<XaiImageProviderOptions>();
    let _ = size_of::<XaiImageQuality>();
    let _ = size_of::<XaiImageResolution>();
    let _ = size_of::<XaiOptions>();
    let _ = size_of::<XaiReasoningSummary>();
    let _ = size_of::<XaiResponseInclude>();
    let _ = size_of::<XaiResponsesOptions>();
    let _ = size_of::<XaiLanguageModelResponsesOptions>();
    #[allow(deprecated)]
    let _ = size_of::<XaiResponsesProviderOptions>();
    let _ = size_of::<XaiResponsesReasoningEffort>();
    let _ = size_of::<XaiTtsOptions>();
    let _ = size_of::<XaiVideoOptions>();
    let _ = size_of::<XaiVideoModelOptions>();
    let _ = size_of::<XaiVideoModelId>();
    #[allow(deprecated)]
    let _ = size_of::<XaiVideoProviderOptions>();
    let _ = size_of::<XaiVideoMode>();
    let _ = size_of::<XaiVideoResolution>();
    let _ = XaiClient::provider_context;
    let _ = XaiClient::base_url;
    let _ = XaiClient::http_client;
    let _ = XaiClient::retry_options;
    let _ = XaiClient::http_interceptors;
    let _ = XaiClient::http_transport;
    let _ = XaiClient::set_retry_options;
    let _ = xai_builder();
    let _ = create_xai();
    let _ = XaiProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/xai")
        .with_header("x-test", "1")
        .into_builder_for_model("grok-4");
    let _ = XaiProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model("grok-4");
    let _ = size_of::<XaiMetadata>();
    let _ = size_of::<XaiSource>();
    let _ = size_of::<XaiSourceMetadata>();
    let _ = siumai::provider_ext::xai::tools::code_execution();
    let _ = siumai::provider_ext::xai::tools::file_search(vec!["store_1".to_string()]);
    let _ = siumai::provider_ext::xai::tools::mcp_server("https://example.com/mcp");
    let _ = siumai::provider_ext::xai::tools::view_image();
    let _ = siumai::provider_ext::xai::tools::view_x_video();
    let _ = siumai::provider_ext::xai::tools::web_search();
    let _ = siumai::provider_ext::xai::tools::x_search();
    let _ = siumai::provider_ext::xai::provider_tools::code_execution();
    let _ = siumai::provider_ext::xai::provider_tools::file_search(vec!["store_1".to_string()]);
    let _ = siumai::provider_ext::xai::provider_tools::mcp_server("https://example.com/mcp");
    let _ = siumai::provider_ext::xai::provider_tools::view_image();
    let _ = siumai::provider_ext::xai::provider_tools::view_x_video();
    let _ = siumai::provider_ext::xai::provider_tools::web_search();
    let _ = siumai::provider_ext::xai::provider_tools::x_search();

    fn _assert_req_ext<T: XaiChatRequestExt>() {}
    fn _assert_image_req_ext<T: XaiImageRequestExt>() {}
    fn _assert_tts_req_ext<T: XaiTtsRequestExt>() {}
    fn _assert_video_req_ext<T: XaiVideoRequestExt>() {}
    fn _assert_resp_ext<T: XaiChatResponseExt>() {}
    fn _assert_source_ext<T: XaiSourceExt>() {}
    _assert_req_ext::<siumai::prelude::unified::ChatRequest>();
    _assert_image_req_ext::<GenerateImageRequest>();
    _assert_image_req_ext::<ImageGenerationRequest>();
    _assert_image_req_ext::<siumai::extensions::types::ImageEditRequest>();
    _assert_tts_req_ext::<TtsRequest>();
    _assert_video_req_ext::<siumai::extensions::types::VideoGenerationRequest>();
    _assert_resp_ext::<ChatResponse>();
    _assert_source_ext::<XaiSource>();

    let mut resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut inner = HashMap::new();
    inner.insert(
        "sources".to_string(),
        serde_json::json!([{
            "id": "src_1",
            "source_type": "document",
            "url": "file_123",
            "title": "Doc",
            "filename": "notes.txt",
            "provider_metadata": {
                "xai": {
                    "fileId": "file_123",
                    "containerId": "container_456",
                    "index": 7
                }
            }
        }]),
    );
    let mut outer = HashMap::new();
    outer.insert(
        "xai".to_string(),
        serde_json::Value::Object(inner.into_iter().collect()),
    );
    resp.provider_metadata = Some(outer);

    let typed = resp.xai_metadata().expect("xai metadata");
    let source = typed
        .sources
        .as_ref()
        .and_then(|sources| sources.first())
        .expect("xai source");
    assert_eq!(source.source_type, "document");

    let source_meta = source.xai_metadata().expect("xai source metadata");
    assert_eq!(source_meta.file_id.as_deref(), Some("file_123"));
    assert_eq!(source_meta.container_id.as_deref(), Some("container_456"));
    assert_eq!(source_meta.index, Some(7));

    let _ = TtsRequest::new("hello".to_string())
        .with_voice("aria".to_string())
        .with_format("mp3".to_string())
        .with_xai_tts_options(
            XaiTtsOptions::new()
                .with_sample_rate(44_100)
                .with_bit_rate(192_000),
        );
    let _ = ImageGenerationRequest {
        prompt: "hi".to_string(),
        count: 1,
        ..Default::default()
    }
    .with_xai_image_options(
        XaiImageOptions::new()
            .with_aspect_ratio("16:9")
            .with_resolution("2k")
            .with_quality("high"),
    );
    let _ = siumai::extensions::types::ImageEditRequest {
        images: vec![siumai::extensions::types::ImageEditInput::file(vec![
            1, 2, 3,
        ])],
        mask: None,
        prompt: "edit".to_string(),
        model: Some("grok-imagine-image".to_string()),
        count: Some(1),
        size: None,
        aspect_ratio: None,
        seed: None,
        response_format: Some("b64_json".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
    .with_xai_image_options(XaiImageOptions::new().with_output_format("png"));
    let _ = GenerateImageRequest::new("draw a robot").with_xai_image_options(
        XaiImageOptions::new()
            .with_aspect_ratio("16:9")
            .with_output_format("png"),
    );
    let _ = siumai::extensions::types::ImageEditInput::url("https://example.com/input.png");
    let _ =
        siumai::extensions::types::VideoGenerationInput::url("https://example.com/start-frame.png");
    let _ = siumai::extensions::types::VideoGenerationRequest::new("grok-imagine-video", "hi")
        .with_count(2)
        .with_fps(24)
        .with_seed(7)
        .with_image(
            siumai::extensions::types::VideoGenerationInput::file_with_media_type(
                vec![1, 2, 3],
                "image/png",
            ),
        )
        .with_xai_video_options(
            XaiVideoOptions::new()
                .with_mode("reference-to-video")
                .with_video_url("https://example.com/video.mp4")
                .with_reference_image_urls([
                    "https://example.com/ref-1.png",
                    "https://example.com/ref-2.png",
                ])
                .with_resolution("720p"),
        );
    let _ = siumai::provider_ext::xai::tools::web_search();
    let _ = siumai::provider_ext::xai::tools::x_search();
    let _ = siumai::provider_ext::xai::tools::code_execution();
    let _ = siumai::provider_ext::xai::tools::view_image();
    let _ = siumai::provider_ext::xai::tools::view_x_video();
    let _ = siumai::provider_ext::xai::tools::file_search(vec!["collection_1".to_string()]);
    let _ = siumai::provider_ext::xai::tools::file_search_with(
        siumai::provider_ext::xai::tools::FileSearchArgs::new(["collection_1"])
            .with_max_num_results(5),
    );
    let _ = siumai::provider_ext::xai::tools::mcp("https://example.com/mcp");
    let _ = siumai::provider_ext::xai::tools::web_search_with(
        siumai::provider_ext::xai::tools::WebSearchArgs::new()
            .with_allowed_domains(["wikipedia.org"]),
    );
    let _ = siumai::provider_ext::xai::tools::x_search_with(
        siumai::provider_ext::xai::tools::XSearchArgs::new().with_allowed_x_handles(["xai"]),
    );
    let _ = siumai::provider_ext::xai::tools::mcp_server_with(
        siumai::provider_ext::xai::tools::McpArgs::new("https://example.com/mcp")
            .with_server_label("docs"),
    );
    let _ = siumai::provider_ext::xai::provider_tools::web_search();
    let _ = siumai::provider_ext::xai::provider_tools::x_search();
    let _ = siumai::provider_ext::xai::provider_tools::code_execution();
    let _ = siumai::provider_ext::xai::provider_tools::view_image();
    let _ = siumai::provider_ext::xai::provider_tools::view_x_video();
    let _ =
        siumai::provider_ext::xai::provider_tools::file_search(vec!["collection_1".to_string()]);
    let _ = siumai::provider_ext::xai::provider_tools::mcp("https://example.com/mcp");
    let _ = siumai::Provider::xai();
}

#[cfg(feature = "ollama")]
#[test]
fn public_surface_ollama_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::ollama::{
        OllamaBuilder, OllamaClient, OllamaConfig, OllamaParams, chat, embedding, metadata::*,
        model_sets, options::*,
    };

    let _ = size_of::<OllamaBuilder>();
    let _ = size_of::<OllamaClient>();
    let _ = size_of::<OllamaConfig>();
    let _ = size_of::<OllamaParams>();
    let _ = size_of::<OllamaOptions>();
    let _ = size_of::<OllamaEmbeddingOptions>();
    let _ = OllamaClient::base_url;
    let _ = OllamaClient::provider_context;
    let _ = OllamaClient::http_client;
    let _ = OllamaClient::retry_options;
    let _ = OllamaClient::http_interceptors;
    let _ = OllamaClient::http_transport;
    let _ = OllamaClient::set_retry_options;
    let _ = size_of::<OllamaMetadata>();
    let _ = chat::LLAMA_3_2_LATEST;
    let _ = embedding::NOMIC_EMBED_TEXT;
    let _ = model_sets::CHAT;

    fn _assert_req_ext<T: OllamaChatRequestExt>() {}
    fn _assert_embed_req_ext<T: OllamaEmbeddingRequestExt>() {}
    fn _assert_resp_ext<T: OllamaChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_embed_req_ext::<EmbeddingRequest>();
    _assert_resp_ext::<ChatResponse>();

    let _ = ChatRequest::new(vec![user!("hi")]).with_ollama_options(
        OllamaOptions::new()
            .with_keep_alive("1m")
            .with_param("think", serde_json::json!(true)),
    );
    let _ = EmbeddingRequest::single("hello").with_ollama_config(
        OllamaEmbeddingOptions::new()
            .with_keep_alive("5m")
            .with_truncate(false),
    );
}

#[cfg(feature = "minimaxi")]
#[test]
fn public_surface_minimaxi_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::minimaxi::{
        MinimaxiBuilder, MinimaxiClient, MinimaxiConfig, chat,
        ext::{music, structured_output, thinking, video},
        image,
        metadata::*,
        model_sets, music as music_models,
        options::*,
        resources::*,
        speech, video as video_models,
    };
    use std::collections::HashMap;

    let _ = size_of::<MinimaxiBuilder>();
    let _ = size_of::<MinimaxiClient>();
    let _ = size_of::<MinimaxiConfig>();
    let _ = size_of::<MinimaxiOptions>();
    let _ = size_of::<MinimaxiResponseFormat>();
    let _ = size_of::<MinimaxiThinkingModeConfig>();
    let _ = size_of::<MinimaxiTtsOptions>();
    let _ = size_of::<MinimaxiVideoOptions>();
    let _ = size_of::<MinimaxiTtsRequestBuilder>();
    let _ = size_of::<music::MinimaxiMusicRequestBuilder>();
    let _ = size_of::<video::MinimaxiVideoRequestBuilder>();
    let _ = size_of::<MinimaxiFiles>();
    let _ = size_of::<MinimaxiMetadata>();
    let _ = size_of::<MinimaxiSource>();
    let _ = size_of::<MinimaxiToolCallMetadata>();
    let _ = size_of::<MinimaxiToolCaller>();
    let _ = chat::MINIMAX_M2;
    let _ = speech::SPEECH_2_6_HD;
    let _ = video_models::HAILUO_2_3;
    let _ = music_models::MUSIC_2_0;
    let _ = image::IMAGE_01;
    let _ = model_sets::CHAT;

    fn _assert_chat_req_ext<T: MinimaxiChatRequestExt>() {}
    fn _assert_req_ext<T: MinimaxiTtsRequestExt>() {}
    fn _assert_video_req_ext<T: MinimaxiVideoRequestExt>() {}
    fn _assert_resp_ext<T: MinimaxiChatResponseExt>() {}
    fn _assert_part_ext<T: MinimaxiContentPartExt>() {}
    _assert_chat_req_ext::<ChatRequest>();
    _assert_req_ext::<siumai::prelude::unified::TtsRequest>();
    _assert_video_req_ext::<siumai::prelude::extensions::types::VideoGenerationRequest>();
    _assert_resp_ext::<ChatResponse>();
    _assert_part_ext::<ContentPart>();
    let _ = structured_output::chat_with_json_object::<MinimaxiClient>;
    let _ = thinking::chat_with_thinking::<MinimaxiClient>;

    let mut resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut inner = HashMap::new();
    inner.insert(
        "sources".to_string(),
        serde_json::json!([{
            "id": "src_1",
            "source_type": "document",
            "title": "Example",
            "filename": "example.pdf"
        }]),
    );
    let mut outer = HashMap::new();
    outer.insert(
        "minimaxi".to_string(),
        serde_json::Value::Object(inner.into_iter().collect()),
    );
    resp.provider_metadata = Some(outer);
    let typed = resp.minimaxi_metadata().expect("minimaxi metadata");
    let source = typed
        .sources
        .as_ref()
        .and_then(|sources| sources.first())
        .expect("minimaxi source");
    assert_eq!(source.source_type, "document");
    assert_eq!(source.filename.as_deref(), Some("example.pdf"));
}

#[cfg(feature = "azure")]
#[test]
#[allow(deprecated)]
fn public_surface_azure_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::azure::{
        AzureChatMode, AzureOpenAIProviderSettings, AzureOpenAiBuilder, AzureOpenAiClient,
        AzureOpenAiConfig, AzureOpenAiSpec, AzureUrlConfig, VERSION, azure as azure_builder,
        create_azure, metadata::*, options::*,
    };

    let _ = size_of::<AzureOpenAIProviderSettings>();
    let _ = size_of::<AzureOpenAiBuilder>();
    let _ = size_of::<AzureOpenAiClient>();
    let _ = size_of::<AzureOpenAiConfig>();
    let _ = size_of::<AzureOpenAiSpec>();
    let _ = size_of::<AzureUrlConfig>();
    let _ = size_of::<AzureChatMode>();
    let _ = size_of::<AzureOpenAiOptions>();
    let _ = size_of::<AzureResponsesApiConfig>();
    let _ = size_of::<AzureReasoningEffort>();
    let _ = size_of::<OpenAIContextManagementConfig>();
    let _ = size_of::<OpenAIContextManagementType>();
    let _ = size_of::<OpenAILanguageModelChatOptions>();
    let _ = size_of::<OpenAIChatLanguageModelOptions>();
    let _ = size_of::<OpenAILanguageModelResponsesOptions>();
    let _ = size_of::<OpenAIResponsesProviderOptions>();
    let _ = size_of::<SystemMessageMode>();
    let _ = size_of::<AzureMetadata>();
    let _ = size_of::<AzureSource>();
    let _ = size_of::<AzureSourceMetadata>();
    let _ = size_of::<AzureContentPartMetadata>();
    let _ = azure_builder();
    let _ = create_azure();
    let _ = VERSION;
    let _ = AzureOpenAIProviderSettings::new();
    let _ = AzureOpenAIProviderSettings::new()
        .with_api_key("test-key")
        .with_resource_name("demo-resource")
        .with_header("x-test", "1")
        .with_api_version("2024-10-21")
        .with_use_deployment_based_urls(true)
        .into_builder_for_model("deployment-id");
    let _ = AzureOpenAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.openai.azure.com/openai")
        .into_config_for_model("deployment-id");

    fn _assert_req_ext<T: AzureOpenAiChatRequestExt>() {}
    fn _assert_resp_ext<T: AzureChatResponseExt>() {}
    fn _assert_source_ext<T: AzureSourceExt>() {}
    fn _assert_part_ext<T: AzureContentPartExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
    _assert_source_ext::<AzureSource>();
    _assert_part_ext::<ContentPart>();

    let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_azure_options(
        AzureOpenAiOptions::new().with_reasoning_effort(AzureReasoningEffort::Medium),
    );
    let value = request
        .provider_options_map
        .get("azure")
        .expect("azure options present");
    assert_eq!(value["reasoning_effort"], serde_json::json!("medium"));

    let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
        .with_azure_options(OpenAILanguageModelChatOptions::new());
    let _ = request;
    let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
        .with_azure_options(OpenAILanguageModelResponsesOptions::new());
    let _ = request;

    let mut resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut inner = std::collections::HashMap::new();
    inner.insert("service_tier".to_string(), serde_json::json!("default"));
    let mut outer = std::collections::HashMap::new();
    outer.insert(
        "azure".to_string(),
        serde_json::Value::Object(inner.into_iter().collect()),
    );
    resp.provider_metadata = Some(outer);
    let typed = resp.azure_metadata().expect("azure metadata");
    assert_eq!(typed.service_tier.as_deref(), Some("default"));

    let _ = siumai::Provider::azure();
    let _ = siumai::Provider::azure_chat();
}

#[cfg(feature = "deepseek")]
#[test]
fn public_surface_deepseek_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::deepseek::{
        DeepSeekBuilder, DeepSeekClient, DeepSeekConfig, DeepSeekErrorData,
        DeepSeekProviderSettings, VERSION, chat, create_deepseek, deepseek as deepseek_builder,
        metadata::*, model_sets, options::*,
    };

    let _ = size_of::<DeepSeekBuilder>();
    let _ = size_of::<DeepSeekClient>();
    let _ = size_of::<DeepSeekConfig>();
    let _ = size_of::<DeepSeekErrorData>();
    let _ = size_of::<DeepSeekProviderSettings>();
    let _ = size_of::<DeepSeekOptions>();
    let _ = size_of::<DeepSeekLanguageModelOptions>();
    #[allow(deprecated)]
    let _ = size_of::<DeepSeekChatOptions>();
    let _ = DeepSeekClient::provider_context;
    let _ = DeepSeekClient::base_url;
    let _ = DeepSeekClient::http_client;
    let _ = DeepSeekClient::retry_options;
    let _ = DeepSeekClient::http_interceptors;
    let _ = DeepSeekClient::http_transport;
    let _ = DeepSeekClient::set_retry_options;
    let _ = size_of::<DeepSeekMetadata>();
    let _ = size_of::<DeepSeekSource>();
    let _ = size_of::<DeepSeekSourceMetadata>();
    let _ = chat::DEEPSEEK_CHAT;
    let _ = chat::DEEPSEEK_REASONER;
    let _ = model_sets::CHAT;
    let _ = deepseek_builder();
    let _ = create_deepseek();
    let _ = VERSION;
    let _ = DeepSeekProviderSettings::new();
    let _ = DeepSeekProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/deepseek")
        .with_header("x-test", "1")
        .into_builder_for_model("deepseek-chat");
    let _ = DeepSeekProviderSettings::new()
        .with_api_key("test-key")
        .into_config_for_model("deepseek-chat");
    let decoded_error: DeepSeekErrorData = serde_json::from_value(serde_json::json!({
        "error": {
            "message": "bad request"
        }
    }))
    .expect("decode deepseek error data");
    assert_eq!(decoded_error.error.message, "bad request");

    fn _assert_req_ext<T: DeepSeekChatRequestExt>() {}
    fn _assert_resp_ext<T: DeepSeekChatResponseExt>() {}
    fn _assert_source_ext<T: DeepSeekSourceExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
    _assert_source_ext::<DeepSeekSource>();
}
