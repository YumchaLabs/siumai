use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

#[test]
fn ai_sdk_surface_is_directory_module_with_shared_primitives_split_out() {
    let types_dir = crate_root().join("src").join("types");
    let ai_sdk_dir = types_dir.join("ai_sdk");
    let ai_sdk_file = types_dir.join("ai_sdk.rs");
    let mod_rs = ai_sdk_dir.join("mod.rs");
    let call_options_rs = ai_sdk_dir.join("call_options.rs");
    let embedding_rs = ai_sdk_dir.join("embedding.rs");
    let errors_rs = ai_sdk_dir.join("errors.rs");
    let flow_control_rs = ai_sdk_dir.join("flow_control.rs");
    let generate_object_rs = ai_sdk_dir.join("generate_object.rs");
    let generate_text_rs = ai_sdk_dir.join("generate_text.rs");
    let generated_files_rs = ai_sdk_dir.join("generated_files.rs");
    let language_model_metadata_rs = ai_sdk_dir.join("language_model_metadata.rs");
    let language_model_results_rs = ai_sdk_dir.join("language_model_results.rs");
    let language_model_v4_rs = ai_sdk_dir.join("language_model_v4.rs");
    let language_model_v4_dir = ai_sdk_dir.join("language_model_v4");
    let language_model_v4_content_rs = language_model_v4_dir.join("content.rs");
    let language_model_v4_prompt_rs = language_model_v4_dir.join("prompt.rs");
    let language_model_v4_shared_rs = language_model_v4_dir.join("shared.rs");
    let media_results_rs = ai_sdk_dir.join("media_results.rs");
    let object_stream_rs = ai_sdk_dir.join("object_stream.rs");
    let output_parts_rs = ai_sdk_dir.join("output_parts.rs");
    let rerank_rs = ai_sdk_dir.join("rerank.rs");
    let response_metadata_rs = ai_sdk_dir.join("response_metadata.rs");
    let shared_rs = ai_sdk_dir.join("shared.rs");
    let source_rs = ai_sdk_dir.join("source.rs");
    let text_stream_rs = ai_sdk_dir.join("text_stream.rs");
    let timeout_rs = ai_sdk_dir.join("timeout.rs");
    let tool_lifecycle_rs = ai_sdk_dir.join("tool_lifecycle.rs");
    let ui_message_rs = ai_sdk_dir.join("ui_message.rs");
    let ui_message_chunks_rs = ai_sdk_dir.join("ui_message_chunks.rs");
    let usage_rs = ai_sdk_dir.join("usage.rs");

    assert!(
        ai_sdk_dir.is_dir(),
        "AI SDK surface should stay as a directory module so ownership slices can be split"
    );
    assert!(
        !ai_sdk_file.exists(),
        "AI SDK surface must not collapse back into src/types/ai_sdk.rs"
    );
    assert!(mod_rs.exists(), "AI SDK directory module should own mod.rs");
    assert!(
        call_options_rs.exists(),
        "AI SDK call option carriers should stay split out as ai_sdk/call_options.rs"
    );
    assert!(
        embedding_rs.exists(),
        "AI SDK embedding result carriers should stay split out as ai_sdk/embedding.rs"
    );
    assert!(
        errors_rs.exists(),
        "AI SDK passive errors should stay split out as ai_sdk/errors.rs"
    );
    assert!(
        flow_control_rs.exists(),
        "AI SDK flow-control helpers should stay split out as ai_sdk/flow_control.rs"
    );
    assert!(
        generate_object_rs.exists(),
        "AI SDK generate-object carriers should stay split out as ai_sdk/generate_object.rs"
    );
    assert!(
        generate_text_rs.exists(),
        "AI SDK generate-text carriers should stay split out as ai_sdk/generate_text.rs"
    );
    assert!(
        generated_files_rs.exists(),
        "AI SDK generated file carriers should stay split out as ai_sdk/generated_files.rs"
    );
    assert!(
        language_model_metadata_rs.exists(),
        "AI SDK language-model metadata carriers should stay split out as ai_sdk/language_model_metadata.rs"
    );
    assert!(
        language_model_results_rs.exists(),
        "AI SDK language-model result envelopes should stay split out as ai_sdk/language_model_results.rs"
    );
    assert!(
        language_model_v4_rs.exists(),
        "AI SDK language-model V4 shell should stay split out as ai_sdk/language_model_v4.rs"
    );
    assert!(
        language_model_v4_dir.is_dir(),
        "AI SDK language-model V4 internals should stay split under ai_sdk/language_model_v4/"
    );
    assert!(
        language_model_v4_content_rs.exists(),
        "AI SDK language-model V4 generated content should stay split out as ai_sdk/language_model_v4/content.rs"
    );
    assert!(
        language_model_v4_prompt_rs.exists(),
        "AI SDK language-model V4 prompt projections should stay split out as ai_sdk/language_model_v4/prompt.rs"
    );
    assert!(
        language_model_v4_shared_rs.exists(),
        "AI SDK language-model V4 shared data/helpers should stay split out as ai_sdk/language_model_v4/shared.rs"
    );
    assert!(
        media_results_rs.exists(),
        "AI SDK media result envelopes should stay split out as ai_sdk/media_results.rs"
    );
    assert!(
        object_stream_rs.exists(),
        "AI SDK object stream parts should stay split out as ai_sdk/object_stream.rs"
    );
    assert!(
        output_parts_rs.exists(),
        "AI SDK output and tool parts should stay split out as ai_sdk/output_parts.rs"
    );
    assert!(
        rerank_rs.exists(),
        "AI SDK rerank carriers should stay split out as ai_sdk/rerank.rs"
    );
    assert!(
        response_metadata_rs.exists(),
        "AI SDK response metadata should stay split out as ai_sdk/response_metadata.rs"
    );
    assert!(
        shared_rs.exists(),
        "AI SDK shared primitives should stay split out as ai_sdk/shared.rs"
    );
    assert!(
        source_rs.exists(),
        "AI SDK source carriers should stay split out as ai_sdk/source.rs"
    );
    assert!(
        text_stream_rs.exists(),
        "AI SDK text stream parts should stay split out as ai_sdk/text_stream.rs"
    );
    assert!(
        timeout_rs.exists(),
        "AI SDK timeout carriers should stay split out as ai_sdk/timeout.rs"
    );
    assert!(
        tool_lifecycle_rs.exists(),
        "AI SDK tool lifecycle carriers should stay split out as ai_sdk/tool_lifecycle.rs"
    );
    assert!(
        ui_message_rs.exists(),
        "AI SDK UI message carriers should stay split out as ai_sdk/ui_message.rs"
    );
    assert!(
        ui_message_chunks_rs.exists(),
        "AI SDK UI message chunks should stay split out as ai_sdk/ui_message_chunks.rs"
    );
    assert!(
        usage_rs.exists(),
        "AI SDK usage carriers should stay split out as ai_sdk/usage.rs"
    );

    let mod_source = fs::read_to_string(&mod_rs).expect("read ai_sdk/mod.rs");
    let production_mod_source = mod_source
        .split("#[cfg(test)]")
        .next()
        .expect("production ai_sdk module source");
    assert!(
        mod_source.contains("mod shared;") && mod_source.contains("pub use shared::*;"),
        "ai_sdk/mod.rs should re-export shared primitives through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod call_options;") && mod_source.contains("pub use call_options::*;"),
        "ai_sdk/mod.rs should re-export call option carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod embedding;") && mod_source.contains("pub use embedding::*;"),
        "ai_sdk/mod.rs should re-export embedding result carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod errors;") && mod_source.contains("pub use errors::*;"),
        "ai_sdk/mod.rs should re-export passive error carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod flow_control;") && mod_source.contains("pub use flow_control::*;"),
        "ai_sdk/mod.rs should re-export flow-control helpers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod generate_object;")
            && mod_source.contains("pub use generate_object::*;"),
        "ai_sdk/mod.rs should re-export generate-object carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod generate_text;")
            && mod_source.contains("pub use generate_text::*;"),
        "ai_sdk/mod.rs should re-export generate-text carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod generated_files;")
            && mod_source.contains("pub use generated_files::*;"),
        "ai_sdk/mod.rs should re-export generated file carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod language_model_metadata;")
            && mod_source.contains("pub use language_model_metadata::*;"),
        "ai_sdk/mod.rs should re-export language-model metadata through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod language_model_results;")
            && mod_source.contains("pub use language_model_results::*;"),
        "ai_sdk/mod.rs should re-export language-model result envelopes through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod language_model_v4;")
            && mod_source.contains("pub use language_model_v4::*;"),
        "ai_sdk/mod.rs should re-export language-model V4 prompt/content projections through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod media_results;")
            && mod_source.contains("pub use media_results::*;"),
        "ai_sdk/mod.rs should re-export media result envelopes through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod object_stream;")
            && mod_source.contains("pub use object_stream::*;"),
        "ai_sdk/mod.rs should re-export object stream parts through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod output_parts;") && mod_source.contains("pub use output_parts::*;"),
        "ai_sdk/mod.rs should re-export output and tool parts through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod rerank;") && mod_source.contains("pub use rerank::*;"),
        "ai_sdk/mod.rs should re-export rerank carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod response_metadata;")
            && mod_source.contains("pub use response_metadata::*;"),
        "ai_sdk/mod.rs should re-export response metadata through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod source;") && mod_source.contains("pub use source::*;"),
        "ai_sdk/mod.rs should re-export source carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod text_stream;") && mod_source.contains("pub use text_stream::*;"),
        "ai_sdk/mod.rs should re-export text stream parts through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod timeout;") && mod_source.contains("pub use timeout::*;"),
        "ai_sdk/mod.rs should re-export timeout carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod tool_lifecycle;")
            && mod_source.contains("pub use tool_lifecycle::*;"),
        "ai_sdk/mod.rs should re-export tool lifecycle carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod ui_message;") && mod_source.contains("pub use ui_message::*;"),
        "ai_sdk/mod.rs should re-export UI message carriers through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod ui_message_chunks;")
            && mod_source.contains("pub use ui_message_chunks::*;"),
        "ai_sdk/mod.rs should re-export UI message chunks through the stable ai_sdk surface"
    );
    assert!(
        mod_source.contains("mod usage;") && mod_source.contains("pub use usage::*;"),
        "ai_sdk/mod.rs should re-export usage carriers through the stable ai_sdk surface"
    );
    for forbidden in ["pub struct ", "pub enum ", "pub fn ", "type ", "impl "] {
        assert!(
            !production_mod_source.contains(forbidden),
            "ai_sdk/mod.rs should stay a thin re-export shell and must not define {forbidden} items"
        );
    }
    for unexpected in [
        "pub struct LanguageModelV4CallOptions",
        "pub struct RequestOptions",
        "pub enum LanguageModelReasoning",
        "pub struct LanguageModelCallOptions",
        "pub struct CallSettings",
        "pub struct AISDKError",
        "pub struct APICallError",
        "pub struct EmptyResponseBodyError",
        "pub struct InvalidPromptError",
        "pub struct InvalidResponseDataError",
        "pub struct JSONParseError",
        "pub struct LoadAPIKeyError",
        "pub struct LoadSettingError",
        "pub struct DownloadError",
        "pub struct NoContentGeneratedError",
        "pub enum NoSuchModelType",
        "pub struct NoSuchModelError",
        "pub struct NoSuchProviderError",
        "pub struct NoSuchProviderReferenceError",
        "pub struct TooManyEmbeddingValuesForCallError",
        "pub struct TypeValidationContext",
        "pub struct TypeValidationError",
        "pub struct UnsupportedFunctionalityError",
        "pub struct InvalidArgumentError",
        "pub struct InvalidStreamPartError",
        "pub struct InvalidToolApprovalError",
        "pub struct ToolCallNotFoundForApprovalError",
        "pub struct NoImageGeneratedError",
        "pub struct NoObjectGeneratedError",
        "pub struct NoOutputGeneratedError",
        "pub struct NoSpeechGeneratedError",
        "pub struct NoTranscriptGeneratedError",
        "pub struct NoVideoGeneratedError",
        "pub struct UnsupportedModelVersionError",
        "pub struct UIMessageStreamError",
        "pub struct InvalidMessageRoleError",
        "pub struct UiMessageWithoutId",
        "pub struct MessageConversionError",
        "pub enum RetryErrorReason",
        "pub struct RetryError",
        "pub enum StopCondition",
        "pub const fn is_step_count",
        "pub const fn step_count_is",
        "pub const fn is_loop_finished",
        "pub fn has_tool_call",
        "pub fn is_stop_condition_met",
        "pub fn filter_active_tools",
        "pub fn experimental_filter_active_tools",
        "pub enum PruneReasoningMode",
        "pub enum PruneEmptyMessagesMode",
        "pub enum PruneToolCallMode",
        "pub struct PruneToolCallRule",
        "pub struct PruneMessagesOptions",
        "pub fn prune_messages",
        "pub enum GenerateObjectOutputStrategy",
        "pub struct GenerateObjectResponseMetadata",
        "pub struct GenerateObjectStartEvent",
        "pub struct GenerateObjectStepStartEvent",
        "pub struct GenerateObjectStepEndEvent",
        "pub struct GenerateObjectEndEvent",
        "pub enum GenerateTextContentPart",
        "pub enum ResponseMessage",
        "pub enum GenerateTextReasoningPart",
        "pub enum GenerateTextStepReasoningPart",
        "pub struct GenerateTextModelInfo",
        "pub struct GenerateTextResponseMetadata",
        "pub struct GenerateTextStepResult",
        "pub type StepResult",
        "pub type DefaultStepResult",
        "pub struct GenerateTextResult",
        "pub struct GenerateTextStartEvent",
        "pub struct GenerateTextStepStartEvent",
        "pub struct PrepareStepOptions",
        "pub struct PrepareStepResult",
        "pub type GenerateTextStepEndEvent",
        "pub struct GenerateTextEndEvent",
        "pub enum StreamTextLifecycleChunkType",
        "pub struct StreamTextLifecycleChunk",
        "pub enum StreamTextChunk",
        "pub struct StreamTextChunkEvent",
        "pub struct TextStreamTextStartPart",
        "pub struct TextStreamTextDeltaPart",
        "pub struct TextStreamTextEndPart",
        "pub struct TextStreamReasoningStartPart",
        "pub struct TextStreamReasoningDeltaPart",
        "pub struct TextStreamReasoningEndPart",
        "pub struct TextStreamCustomPart",
        "pub struct TextStreamToolInputStartPart",
        "pub struct TextStreamToolInputDeltaPart",
        "pub struct TextStreamToolInputEndPart",
        "pub struct TextStreamFilePart",
        "pub struct TextStreamReasoningFilePart",
        "pub type TextStreamToolCallPart",
        "pub type TextStreamSourcePart",
        "pub type TextStreamToolResultPart",
        "pub type TextStreamToolErrorPart",
        "pub type TextStreamToolOutputDeniedPart",
        "pub type TextStreamToolApprovalRequestPart",
        "pub type TextStreamToolApprovalResponsePart",
        "pub struct TextStreamStartStepPart",
        "pub struct TextStreamFinishStepPart",
        "pub struct TextStreamStartPart",
        "pub struct TextStreamFinishPart",
        "pub struct TextStreamAbortPart",
        "pub struct TextStreamErrorPart",
        "pub struct TextStreamRawPart",
        "pub enum TextStreamPart",
        "pub struct LanguageModelStreamModelCallStartPart",
        "pub struct LanguageModelStreamModelCallEndPart",
        "pub struct LanguageModelStreamModelCallResponseMetadataPart",
        "pub enum LanguageModelStreamPart",
        "pub type ExperimentalLanguageModelStreamPart",
        "pub type Experimental_LanguageModelStreamPart",
        "pub struct TimeoutConfigurationSettings",
        "pub enum TimeoutConfiguration",
        "pub const fn get_total_timeout_ms",
        "pub const fn get_step_timeout_ms",
        "pub const fn get_chunk_timeout_ms",
        "pub fn get_tool_timeout_ms",
        "pub struct CallbackModelInfo",
        "pub enum ToolApprovalStatusType",
        "pub struct ToolApprovalStatusDetails",
        "pub enum ToolApprovalStatus",
        "pub type ToolApprovalConfiguration",
        "pub struct ToolApprovalDecisionContext",
        "pub struct NoSuchToolError",
        "pub struct InvalidToolInputError",
        "pub enum ToolCallRepairFunctionError",
        "pub struct ToolCallRepairError",
        "pub struct ToolCallRepairContext",
        "pub type ToolCallRepairResult",
        "pub struct ToolExecutionStartEvent",
        "pub struct ToolExecutionEndEvent",
        "pub type OnStartEvent",
        "pub type OnStepStartEvent",
        "pub type OnChunkEvent",
        "pub type OnStepFinishEvent",
        "pub type OnFinishEvent",
        "pub type OnToolCallStartEvent",
        "pub type OnToolCallFinishEvent",
        "pub struct Source",
        "pub type ImageModelProviderMetadata",
        "pub type VideoModelProviderMetadata",
        "pub type LanguageModelV4Prompt",
        "pub enum LanguageModelV4DataContent",
        "pub enum LanguageModelV4GeneratedFileData",
        "pub enum LanguageModelV4FilePartData",
        "pub struct LanguageModelV4TextPart",
        "pub struct LanguageModelV4ReasoningPart",
        "pub struct LanguageModelV4CustomPart",
        "pub struct LanguageModelV4ToolCallPart",
        "pub enum LanguageModelV4ToolResultOutput",
        "pub enum LanguageModelV4ToolResultContentPart",
        "pub struct LanguageModelV4ToolResultPart",
        "pub struct LanguageModelV4FilePart",
        "pub struct LanguageModelV4ReasoningFilePart",
        "pub struct LanguageModelV4ToolApprovalResponsePart",
        "pub enum LanguageModelV4UserContentPart",
        "pub enum LanguageModelV4AssistantContentPart",
        "pub enum LanguageModelV4ToolContentPart",
        "pub struct LanguageModelV4SystemMessage",
        "pub struct LanguageModelV4UserMessage",
        "pub struct LanguageModelV4AssistantMessage",
        "pub struct LanguageModelV4ToolMessage",
        "pub enum LanguageModelV4Message",
        "pub fn prepare_language_model_v4_prompt",
        "pub struct LanguageModelV4Text",
        "pub struct LanguageModelV4Reasoning",
        "pub struct LanguageModelV4CustomContent",
        "pub struct LanguageModelV4Source",
        "pub struct LanguageModelV4File",
        "pub struct LanguageModelV4ReasoningFile",
        "pub struct LanguageModelV4ToolCall",
        "pub struct LanguageModelV4ToolResult",
        "pub struct LanguageModelV4ToolApprovalRequest",
        "pub enum LanguageModelV4Content",
        "pub const UI_MESSAGE_STREAM_HEADERS",
        "pub type UIDataPartSchemas",
        "pub type UIDataTypesToSchemas",
        "pub type InferUIDataParts",
        "pub type UIDataTypes",
        "pub type InferUIMessageMetadata",
        "pub type InferUIMessageData",
        "pub type InferUIMessageTools",
        "pub type InferUIMessageToolOutputs",
        "pub type InferUIMessageToolCall",
        "pub type InferUIMessagePart",
        "pub struct UITool",
        "pub type InferUITool",
        "pub type UITools",
        "pub type InferUITools",
        "pub type UIMessage",
        "pub type UIMessagePart",
        "pub type TextUIPart",
        "pub type CustomContentUIPart",
        "pub type ReasoningUIPart",
        "pub type FileUIPart",
        "pub type ReasoningFileUIPart",
        "pub type SourceUrlUIPart",
        "pub type SourceDocumentUIPart",
        "pub type DataUIPart",
        "pub type ToolUIPart",
        "pub type DynamicToolUIPart",
        "pub type UIToolInvocation",
        "pub type StepStartUIPart",
        "pub fn is_text_ui_part",
        "pub fn is_custom_content_ui_part",
        "pub fn is_file_ui_part",
        "pub fn is_reasoning_file_ui_part",
        "pub fn is_reasoning_ui_part",
        "pub fn is_data_ui_part",
        "pub fn is_static_tool_ui_part",
        "pub fn is_dynamic_tool_ui_part",
        "pub fn is_tool_ui_part",
        "pub fn get_static_tool_name",
        "pub fn get_tool_name",
        "pub fn get_tool_or_dynamic_tool_name",
        "pub fn last_assistant_message_is_complete_with_tool_calls",
        "pub fn last_assistant_message_is_complete_with_approval_responses",
        "pub struct CreateUIMessage",
        "pub struct ChatRequestOptions",
        "pub enum ChatStatus",
        "pub struct ChatState",
        "pub struct ChatInit",
        "pub enum ChatTransportTrigger",
        "pub struct ChatTransportSendMessagesOptions",
        "pub struct ChatTransportReconnectToStreamOptions",
        "pub struct HttpChatTransportInitOptions",
        "pub struct PrepareSendMessagesRequestOptions",
        "pub struct PreparedSendMessagesRequest",
        "pub struct PrepareReconnectToStreamRequestOptions",
        "pub struct PreparedReconnectToStreamRequest",
        "pub struct CompletionRequestOptions",
        "pub enum RequestCredentials",
        "pub enum CompletionStreamProtocol",
        "pub struct UseCompletionOptions",
        "pub struct UiMessageStreamOptions",
        "pub type UIMessageStreamOptions",
        "fn deserialize_ui_message_data_chunk_type",
        "pub struct UiMessageTextStartChunk",
        "pub struct UiMessageTextDeltaChunk",
        "pub struct UiMessageTextEndChunk",
        "pub struct UiMessageReasoningStartChunk",
        "pub struct UiMessageReasoningDeltaChunk",
        "pub struct UiMessageReasoningEndChunk",
        "pub struct UiMessageCustomChunk",
        "pub struct UiMessageErrorChunk",
        "pub struct UiMessageToolInputStartChunk",
        "pub struct UiMessageToolInputDeltaChunk",
        "pub struct UiMessageToolInputAvailableChunk",
        "pub struct UiMessageToolInputErrorChunk",
        "pub struct UiMessageToolApprovalRequestChunk",
        "pub struct UiMessageToolApprovalResponseChunk",
        "pub struct UiMessageToolOutputAvailableChunk",
        "pub struct UiMessageToolOutputErrorChunk",
        "pub struct UiMessageToolOutputDeniedChunk",
        "pub struct UiMessageSourceUrlChunk",
        "pub struct UiMessageSourceDocumentChunk",
        "pub struct UiMessageFileChunk",
        "pub struct UiMessageReasoningFileChunk",
        "pub struct UiMessageDataChunk",
        "pub struct UiMessageStartStepChunk",
        "pub struct UiMessageFinishStepChunk",
        "pub struct UiMessageStartChunk",
        "pub struct UiMessageFinishChunk",
        "pub struct UiMessageAbortChunk",
        "pub struct UiMessageMetadataChunk",
        "pub enum UiMessageChunk",
        "pub type UIMessageChunk",
        "pub type DataUIMessageChunk",
        "pub type InferUIMessageChunk",
        "pub fn is_data_ui_message_chunk",
        "macro_rules! ui_message_id_provider_metadata_chunk",
        "pub struct GeneratedFile",
        "pub type DefaultGeneratedFile",
        "pub struct DefaultGeneratedFileWithType",
        "pub struct GeneratedAudioFile",
        "pub type DefaultGeneratedAudioFile",
        "pub struct DefaultGeneratedAudioFileWithType",
        "pub type LanguageModelV4RequestMetadata",
        "pub struct LanguageModelV4ResponseMetadata",
        "pub struct LanguageModelV4GenerateResponseMetadata",
        "pub struct LanguageModelV4StreamResponseMetadata",
        "pub struct LanguageModelRequestMetadata",
        "pub struct LanguageModelResponseMetadata",
        "fn string_body_to_json_value",
        "pub struct LanguageModelV4FinishReason",
        "pub struct LanguageModelV4GenerateResult",
        "pub struct LanguageModelV4StreamResult",
        "pub struct GenerateImageResult",
        "pub type Experimental_GenerateImageResult",
        "pub struct GenerateVideoResult",
        "pub struct SpeechResult",
        "pub type Experimental_SpeechResult",
        "pub struct TranscriptionSegment",
        "pub struct TranscriptionResult",
        "pub type Experimental_TranscriptionResult",
        "pub struct ObjectStreamObjectPart",
        "pub struct ObjectStreamTextDeltaPart",
        "pub struct ObjectStreamErrorPart",
        "pub struct ObjectStreamFinishPart",
        "pub enum ObjectStreamPart",
        "pub struct TextOutput",
        "pub struct CustomOutput",
        "pub struct FileOutput",
        "pub struct ReasoningOutput",
        "pub struct ReasoningFileOutput",
        "pub struct ToolCall<",
        "pub type StaticToolCall",
        "pub type DynamicToolCall",
        "pub type TypedToolCall",
        "pub struct ToolResult<",
        "pub type StaticToolResult",
        "pub type DynamicToolResult",
        "pub type TypedToolResult",
        "pub struct ToolError<",
        "pub type StaticToolError",
        "pub type DynamicToolError",
        "pub type TypedToolError",
        "pub enum ToolOutput<",
        "pub struct ToolOutputDenied<",
        "pub type StaticToolOutputDenied",
        "pub type TypedToolOutputDenied",
        "pub struct ToolApprovalRequestOutput<",
        "pub struct ToolApprovalResponseOutput<",
        "pub struct ImageModelResponseMetadata",
        "pub struct VideoModelResponseMetadata",
        "pub struct SpeechModelResponseMetadata",
        "pub struct TranscriptionModelResponseMetadata",
        "pub struct LanguageModelV4InputTokens",
        "pub struct LanguageModelV4OutputTokens",
        "pub struct LanguageModelV4Usage",
        "pub struct LanguageModelInputTokenDetails",
        "pub struct LanguageModelOutputTokenDetails",
        "pub struct LanguageModelUsage",
        "pub struct EmbeddingModelUsage",
        "pub struct ImageModelUsage",
        "pub fn add_language_model_usage",
        "pub fn add_image_model_usage",
        "pub struct ModelCallResponseData",
        "pub enum EmbedValue",
        "pub enum EmbedOutput",
        "pub enum EmbedResponseData",
        "pub struct EmbedResult",
        "pub struct EmbedManyResult",
        "pub struct EmbedStartEvent",
        "pub struct EmbedEndEvent",
        "pub struct EmbeddingModelCallStartEvent",
        "pub struct EmbeddingModelCallEndEvent",
        "pub struct RerankResponseMetadata",
        "pub struct RerankRanking",
        "pub struct RerankResult",
        "pub struct RerankStartEvent",
        "pub struct RerankEndEvent",
        "pub struct RerankingModelCallStartEvent",
        "pub struct RerankingModelCallRanking",
        "pub struct RerankingModelCallEndEvent",
    ] {
        assert!(
            !mod_source.contains(unexpected),
            "ai_sdk/mod.rs should not directly own `{unexpected}`"
        );
    }

    let call_options_source =
        fs::read_to_string(&call_options_rs).expect("read ai_sdk/call_options.rs");
    for expected in [
        "pub struct LanguageModelV4CallOptions",
        "pub struct RequestOptions",
        "pub enum LanguageModelReasoning",
        "pub struct LanguageModelCallOptions",
        "pub struct CallSettings",
    ] {
        assert!(
            call_options_source.contains(expected),
            "ai_sdk/call_options.rs should own `{expected}`"
        );
    }

    let embedding_source = fs::read_to_string(&embedding_rs).expect("read ai_sdk/embedding.rs");
    for expected in [
        "pub struct ModelCallResponseData",
        "pub enum EmbedValue",
        "pub enum EmbedOutput",
        "pub enum EmbedResponseData",
        "pub struct EmbedResult",
        "pub struct EmbedManyResult",
        "pub struct EmbedStartEvent",
        "pub struct EmbedEndEvent",
        "pub struct EmbeddingModelCallStartEvent",
        "pub struct EmbeddingModelCallEndEvent",
    ] {
        assert!(
            embedding_source.contains(expected),
            "ai_sdk/embedding.rs should own `{expected}`"
        );
    }

    let errors_source = fs::read_to_string(&errors_rs).expect("read ai_sdk/errors.rs");
    for expected in [
        "pub struct AISDKError",
        "pub struct APICallError",
        "pub struct EmptyResponseBodyError",
        "pub struct InvalidPromptError",
        "pub struct InvalidResponseDataError",
        "pub struct JSONParseError",
        "pub struct LoadAPIKeyError",
        "pub struct LoadSettingError",
        "pub struct DownloadError",
        "pub struct NoContentGeneratedError",
        "pub enum NoSuchModelType",
        "pub struct NoSuchModelError",
        "pub struct NoSuchProviderError",
        "pub struct NoSuchProviderReferenceError",
        "pub struct TooManyEmbeddingValuesForCallError",
        "pub struct TypeValidationContext",
        "pub struct TypeValidationError",
        "pub struct UnsupportedFunctionalityError",
        "pub struct InvalidArgumentError",
        "pub struct InvalidStreamPartError",
        "pub struct InvalidToolApprovalError",
        "pub struct ToolCallNotFoundForApprovalError",
        "pub struct NoImageGeneratedError",
        "pub struct NoObjectGeneratedError",
        "pub struct NoOutputGeneratedError",
        "pub struct NoSpeechGeneratedError",
        "pub struct NoTranscriptGeneratedError",
        "pub struct NoVideoGeneratedError",
        "pub struct UnsupportedModelVersionError",
        "pub struct UIMessageStreamError",
        "pub struct InvalidMessageRoleError",
        "pub struct UiMessageWithoutId",
        "pub struct MessageConversionError",
        "pub enum RetryErrorReason",
        "pub struct RetryError",
    ] {
        assert!(
            errors_source.contains(expected),
            "ai_sdk/errors.rs should own `{expected}`"
        );
    }

    let flow_control_source =
        fs::read_to_string(&flow_control_rs).expect("read ai_sdk/flow_control.rs");
    for expected in [
        "pub enum StopCondition",
        "pub const fn is_step_count",
        "pub const fn step_count_is",
        "pub const fn is_loop_finished",
        "pub fn has_tool_call",
        "pub fn is_stop_condition_met",
        "pub fn filter_active_tools",
        "pub fn experimental_filter_active_tools",
        "pub enum PruneReasoningMode",
        "pub enum PruneEmptyMessagesMode",
        "pub enum PruneToolCallMode",
        "pub struct PruneToolCallRule",
        "pub struct PruneMessagesOptions",
        "pub fn prune_messages",
    ] {
        assert!(
            flow_control_source.contains(expected),
            "ai_sdk/flow_control.rs should own `{expected}`"
        );
    }

    let generate_object_source =
        fs::read_to_string(&generate_object_rs).expect("read ai_sdk/generate_object.rs");
    for expected in [
        "pub enum GenerateObjectOutputStrategy",
        "pub struct GenerateObjectResponseMetadata",
        "pub struct GenerateObjectStartEvent",
        "pub struct GenerateObjectStepStartEvent",
        "pub struct GenerateObjectStepEndEvent",
        "pub struct GenerateObjectEndEvent",
    ] {
        assert!(
            generate_object_source.contains(expected),
            "ai_sdk/generate_object.rs should own `{expected}`"
        );
    }

    let generate_text_source =
        fs::read_to_string(&generate_text_rs).expect("read ai_sdk/generate_text.rs");
    for expected in [
        "pub enum GenerateTextContentPart",
        "pub enum ResponseMessage",
        "pub enum GenerateTextReasoningPart",
        "pub enum GenerateTextStepReasoningPart",
        "pub struct GenerateTextModelInfo",
        "pub struct GenerateTextResponseMetadata",
        "pub struct GenerateTextStepResult",
        "pub type StepResult",
        "pub type DefaultStepResult",
        "pub struct GenerateTextResult",
        "pub struct GenerateTextStartEvent",
        "pub struct GenerateTextStepStartEvent",
        "pub struct PrepareStepOptions",
        "pub struct PrepareStepResult",
        "pub type GenerateTextStepEndEvent",
        "pub struct GenerateTextEndEvent",
        "pub enum StreamTextLifecycleChunkType",
        "pub struct StreamTextLifecycleChunk",
        "pub enum StreamTextChunk",
        "pub struct StreamTextChunkEvent",
    ] {
        assert!(
            generate_text_source.contains(expected),
            "ai_sdk/generate_text.rs should own `{expected}`"
        );
    }

    let generated_files_source =
        fs::read_to_string(&generated_files_rs).expect("read ai_sdk/generated_files.rs");
    for expected in [
        "pub struct GeneratedFile",
        "pub type DefaultGeneratedFile",
        "pub struct DefaultGeneratedFileWithType",
        "pub type Experimental_GeneratedImage",
        "pub struct GeneratedAudioFile",
        "pub type DefaultGeneratedAudioFile",
        "pub struct DefaultGeneratedAudioFileWithType",
    ] {
        assert!(
            generated_files_source.contains(expected),
            "ai_sdk/generated_files.rs should own `{expected}`"
        );
    }

    let language_model_metadata_source = fs::read_to_string(&language_model_metadata_rs)
        .expect("read ai_sdk/language_model_metadata.rs");
    for expected in [
        "pub type LanguageModelV4RequestMetadata",
        "pub struct LanguageModelV4ResponseMetadata",
        "pub struct LanguageModelV4GenerateResponseMetadata",
        "pub struct LanguageModelV4StreamResponseMetadata",
        "pub struct LanguageModelRequestMetadata",
        "pub struct LanguageModelResponseMetadata",
        "fn string_body_to_json_value",
    ] {
        assert!(
            language_model_metadata_source.contains(expected),
            "ai_sdk/language_model_metadata.rs should own `{expected}`"
        );
    }

    let language_model_results_source = fs::read_to_string(&language_model_results_rs)
        .expect("read ai_sdk/language_model_results.rs");
    for expected in [
        "pub struct LanguageModelV4FinishReason",
        "pub struct LanguageModelV4GenerateResult",
        "pub struct LanguageModelV4StreamResult",
    ] {
        assert!(
            language_model_results_source.contains(expected),
            "ai_sdk/language_model_results.rs should own `{expected}`"
        );
    }

    let language_model_v4_source =
        fs::read_to_string(&language_model_v4_rs).expect("read ai_sdk/language_model_v4.rs");
    let production_language_model_v4_source = language_model_v4_source
        .split("#[cfg(test)]")
        .next()
        .expect("production language_model_v4 shell source");
    assert!(
        language_model_v4_source.contains("mod content;")
            && language_model_v4_source.contains("pub use content::*;"),
        "ai_sdk/language_model_v4.rs should re-export generated content through the stable V4 surface"
    );
    assert!(
        language_model_v4_source.contains("mod prompt;")
            && language_model_v4_source.contains("pub use prompt::*;"),
        "ai_sdk/language_model_v4.rs should re-export prompt projections through the stable V4 surface"
    );
    assert!(
        language_model_v4_source.contains("mod shared;")
            && language_model_v4_source.contains("pub use shared::*;"),
        "ai_sdk/language_model_v4.rs should re-export shared V4 data through the stable V4 surface"
    );
    for expected in [
        "pub type LanguageModelV4Prompt",
        "pub enum LanguageModelV4DataContent",
        "pub struct LanguageModelV4Text",
        "pub enum LanguageModelV4Content",
    ] {
        assert!(
            !language_model_v4_source.contains(expected),
            "ai_sdk/language_model_v4.rs should stay a thin shell and must not own `{expected}`"
        );
    }
    for forbidden in [
        "pub struct ",
        "pub enum ",
        "pub fn ",
        "pub type ",
        "impl ",
        "macro_rules!",
    ] {
        assert!(
            !production_language_model_v4_source.contains(forbidden),
            "ai_sdk/language_model_v4.rs should stay a thin re-export shell and must not define {forbidden} items"
        );
    }

    let language_model_v4_shared_source = fs::read_to_string(&language_model_v4_shared_rs)
        .expect("read ai_sdk/language_model_v4/shared.rs");
    for expected in [
        "macro_rules! fixed_language_model_v4_type_marker",
        "pub enum LanguageModelV4DataContent",
        "pub enum LanguageModelV4GeneratedFileData",
        "pub enum LanguageModelV4FilePartData",
        "pub(super) fn language_model_v4_provider_options_from_stable",
        "pub(crate) fn serialize_optional_language_model_v4_provider_metadata",
        "pub(crate) fn deserialize_optional_language_model_v4_provider_metadata",
        "pub(super) fn is_language_model_v4_custom_kind",
    ] {
        assert!(
            language_model_v4_shared_source.contains(expected),
            "ai_sdk/language_model_v4/shared.rs should own `{expected}`"
        );
    }

    let language_model_v4_prompt_source = fs::read_to_string(&language_model_v4_prompt_rs)
        .expect("read ai_sdk/language_model_v4/prompt.rs");
    for expected in [
        "pub type LanguageModelV4Prompt",
        "pub struct LanguageModelV4TextPart",
        "pub struct LanguageModelV4ReasoningPart",
        "pub struct LanguageModelV4CustomPart",
        "pub struct LanguageModelV4ToolCallPart",
        "pub enum LanguageModelV4ToolResultOutput",
        "pub enum LanguageModelV4ToolResultContentPart",
        "pub struct LanguageModelV4ToolResultPart",
        "pub struct LanguageModelV4FilePart",
        "pub struct LanguageModelV4ReasoningFilePart",
        "pub struct LanguageModelV4ToolApprovalResponsePart",
        "pub enum LanguageModelV4UserContentPart",
        "pub enum LanguageModelV4AssistantContentPart",
        "pub enum LanguageModelV4ToolContentPart",
        "pub struct LanguageModelV4SystemMessage",
        "pub struct LanguageModelV4UserMessage",
        "pub struct LanguageModelV4AssistantMessage",
        "pub struct LanguageModelV4ToolMessage",
        "pub enum LanguageModelV4Message",
        "pub fn prepare_language_model_v4_prompt",
    ] {
        assert!(
            language_model_v4_prompt_source.contains(expected),
            "ai_sdk/language_model_v4/prompt.rs should own `{expected}`"
        );
    }
    for forbidden in ["providerMetadata", "provider_metadata"] {
        assert!(
            !language_model_v4_prompt_source.contains(forbidden),
            "ai_sdk/language_model_v4/prompt.rs is request-side input data and must not carry response-side {forbidden}"
        );
    }

    let language_model_v4_content_source = fs::read_to_string(&language_model_v4_content_rs)
        .expect("read ai_sdk/language_model_v4/content.rs");
    for expected in [
        "pub struct LanguageModelV4Text",
        "pub struct LanguageModelV4Reasoning",
        "pub struct LanguageModelV4CustomContent",
        "pub struct LanguageModelV4Source",
        "pub struct LanguageModelV4File",
        "pub struct LanguageModelV4ReasoningFile",
        "pub struct LanguageModelV4ToolCall",
        "pub struct LanguageModelV4ToolResult",
        "pub struct LanguageModelV4ToolApprovalRequest",
        "pub enum LanguageModelV4Content",
    ] {
        assert!(
            language_model_v4_content_source.contains(expected),
            "ai_sdk/language_model_v4/content.rs should own `{expected}`"
        );
    }
    for forbidden in ["providerOptions", "provider_options"] {
        assert!(
            !language_model_v4_content_source.contains(forbidden),
            "ai_sdk/language_model_v4/content.rs is response-side generated content and must not carry request-side {forbidden}"
        );
    }

    let media_results_source =
        fs::read_to_string(&media_results_rs).expect("read ai_sdk/media_results.rs");
    for expected in [
        "pub struct GenerateImageResult",
        "pub type Experimental_GenerateImageResult",
        "pub struct GenerateVideoResult",
        "pub struct SpeechResult",
        "pub type Experimental_SpeechResult",
        "pub struct TranscriptionSegment",
        "pub struct TranscriptionResult",
        "pub type Experimental_TranscriptionResult",
    ] {
        assert!(
            media_results_source.contains(expected),
            "ai_sdk/media_results.rs should own `{expected}`"
        );
    }

    let object_stream_source =
        fs::read_to_string(&object_stream_rs).expect("read ai_sdk/object_stream.rs");
    for expected in [
        "pub struct ObjectStreamObjectPart",
        "pub struct ObjectStreamTextDeltaPart",
        "pub struct ObjectStreamErrorPart",
        "pub struct ObjectStreamFinishPart",
        "pub enum ObjectStreamPart",
    ] {
        assert!(
            object_stream_source.contains(expected),
            "ai_sdk/object_stream.rs should own `{expected}`"
        );
    }

    let output_parts_source =
        fs::read_to_string(&output_parts_rs).expect("read ai_sdk/output_parts.rs");
    for expected in [
        "pub struct TextOutput",
        "pub struct CustomOutput",
        "pub struct FileOutput",
        "pub struct ReasoningOutput",
        "pub struct ReasoningFileOutput",
        "pub struct ToolCall<",
        "pub type StaticToolCall",
        "pub type DynamicToolCall",
        "pub type TypedToolCall",
        "pub struct ToolResult<",
        "pub type StaticToolResult",
        "pub type DynamicToolResult",
        "pub type TypedToolResult",
        "pub struct ToolError<",
        "pub type StaticToolError",
        "pub type DynamicToolError",
        "pub type TypedToolError",
        "pub enum ToolOutput<",
        "pub struct ToolOutputDenied<",
        "pub type StaticToolOutputDenied",
        "pub type TypedToolOutputDenied",
        "pub struct ToolApprovalRequestOutput<",
        "pub struct ToolApprovalResponseOutput<",
    ] {
        assert!(
            output_parts_source.contains(expected),
            "ai_sdk/output_parts.rs should own `{expected}`"
        );
    }

    let rerank_source = fs::read_to_string(&rerank_rs).expect("read ai_sdk/rerank.rs");
    for expected in [
        "pub struct RerankResponseMetadata",
        "pub struct RerankRanking",
        "pub struct RerankResult",
        "pub struct RerankStartEvent",
        "pub struct RerankEndEvent",
        "pub struct RerankingModelCallStartEvent",
        "pub struct RerankingModelCallRanking",
        "pub struct RerankingModelCallEndEvent",
    ] {
        assert!(
            rerank_source.contains(expected),
            "ai_sdk/rerank.rs should own `{expected}`"
        );
    }

    let response_metadata_source =
        fs::read_to_string(&response_metadata_rs).expect("read ai_sdk/response_metadata.rs");
    for expected in [
        "pub struct ImageModelResponseMetadata",
        "pub struct VideoModelResponseMetadata",
        "pub struct SpeechModelResponseMetadata",
        "pub struct TranscriptionModelResponseMetadata",
    ] {
        assert!(
            response_metadata_source.contains(expected),
            "ai_sdk/response_metadata.rs should own `{expected}`"
        );
    }

    let shared_source = fs::read_to_string(&shared_rs).expect("read ai_sdk/shared.rs");
    for expected in [
        "pub type JSONValue",
        "pub enum CallWarning",
        "pub type ProviderMetadata",
        "pub type ProviderOptions",
        "pub struct TelemetryOptions",
    ] {
        assert!(
            shared_source.contains(expected),
            "ai_sdk/shared.rs should own `{expected}`"
        );
    }

    let source_source = fs::read_to_string(&source_rs).expect("read ai_sdk/source.rs");
    for expected in [
        "pub(super) enum SourceMarker",
        "pub struct Source",
        "pub type ImageModelProviderMetadata",
        "pub type VideoModelProviderMetadata",
    ] {
        assert!(
            source_source.contains(expected),
            "ai_sdk/source.rs should own `{expected}`"
        );
    }

    let text_stream_source =
        fs::read_to_string(&text_stream_rs).expect("read ai_sdk/text_stream.rs");
    for expected in [
        "pub struct TextStreamTextStartPart",
        "pub struct TextStreamTextDeltaPart",
        "pub struct TextStreamTextEndPart",
        "pub struct TextStreamReasoningStartPart",
        "pub struct TextStreamReasoningDeltaPart",
        "pub struct TextStreamReasoningEndPart",
        "pub struct TextStreamCustomPart",
        "pub struct TextStreamToolInputStartPart",
        "pub struct TextStreamToolInputDeltaPart",
        "pub struct TextStreamToolInputEndPart",
        "pub struct TextStreamFilePart",
        "pub struct TextStreamReasoningFilePart",
        "pub type TextStreamToolCallPart",
        "pub type TextStreamSourcePart",
        "pub type TextStreamToolResultPart",
        "pub type TextStreamToolErrorPart",
        "pub type TextStreamToolOutputDeniedPart",
        "pub type TextStreamToolApprovalRequestPart",
        "pub type TextStreamToolApprovalResponsePart",
        "pub struct TextStreamStartStepPart",
        "pub struct TextStreamFinishStepPart",
        "pub struct TextStreamStartPart",
        "pub struct TextStreamFinishPart",
        "pub struct TextStreamAbortPart",
        "pub struct TextStreamErrorPart",
        "pub struct TextStreamRawPart",
        "pub enum TextStreamPart",
        "pub struct LanguageModelStreamModelCallStartPart",
        "pub struct LanguageModelStreamModelCallEndPart",
        "pub struct LanguageModelStreamModelCallResponseMetadataPart",
        "pub enum LanguageModelStreamPart",
        "pub type ExperimentalLanguageModelStreamPart",
        "pub type Experimental_LanguageModelStreamPart",
    ] {
        assert!(
            text_stream_source.contains(expected),
            "ai_sdk/text_stream.rs should own `{expected}`"
        );
    }

    let timeout_source = fs::read_to_string(&timeout_rs).expect("read ai_sdk/timeout.rs");
    for expected in [
        "pub struct TimeoutConfigurationSettings",
        "pub enum TimeoutConfiguration",
        "pub const fn get_total_timeout_ms",
        "pub const fn get_step_timeout_ms",
        "pub const fn get_chunk_timeout_ms",
        "pub fn get_tool_timeout_ms",
    ] {
        assert!(
            timeout_source.contains(expected),
            "ai_sdk/timeout.rs should own `{expected}`"
        );
    }

    let tool_lifecycle_source =
        fs::read_to_string(&tool_lifecycle_rs).expect("read ai_sdk/tool_lifecycle.rs");
    for expected in [
        "pub struct CallbackModelInfo",
        "pub enum ToolApprovalStatusType",
        "pub struct ToolApprovalStatusDetails",
        "pub enum ToolApprovalStatus",
        "pub type ToolApprovalConfiguration",
        "pub struct ToolApprovalDecisionContext",
        "pub struct NoSuchToolError",
        "pub struct InvalidToolInputError",
        "pub enum ToolCallRepairFunctionError",
        "pub struct ToolCallRepairError",
        "pub struct ToolCallRepairContext",
        "pub type ToolCallRepairResult",
        "pub struct ToolExecutionStartEvent",
        "pub struct ToolExecutionEndEvent",
        "pub type OnStartEvent",
        "pub type OnStepStartEvent",
        "pub type OnChunkEvent",
        "pub type OnStepFinishEvent",
        "pub type OnFinishEvent",
        "pub type OnToolCallStartEvent",
        "pub type OnToolCallFinishEvent",
    ] {
        assert!(
            tool_lifecycle_source.contains(expected),
            "ai_sdk/tool_lifecycle.rs should own `{expected}`"
        );
    }

    let ui_message_source = fs::read_to_string(&ui_message_rs).expect("read ai_sdk/ui_message.rs");
    for expected in [
        "pub const UI_MESSAGE_STREAM_HEADERS",
        "pub type UIDataPartSchemas",
        "pub type UIDataTypesToSchemas",
        "pub type InferUIDataParts",
        "pub type UIDataTypes",
        "pub type InferUIMessageMetadata",
        "pub type InferUIMessageData",
        "pub type InferUIMessageTools",
        "pub type InferUIMessageToolOutputs",
        "pub type InferUIMessageToolCall",
        "pub type InferUIMessagePart",
        "pub struct UITool",
        "pub type InferUITool",
        "pub type UITools",
        "pub type InferUITools",
        "pub type UIMessage",
        "pub type UIMessagePart",
        "pub type TextUIPart",
        "pub type CustomContentUIPart",
        "pub type ReasoningUIPart",
        "pub type FileUIPart",
        "pub type ReasoningFileUIPart",
        "pub type SourceUrlUIPart",
        "pub type SourceDocumentUIPart",
        "pub type DataUIPart",
        "pub type ToolUIPart",
        "pub type DynamicToolUIPart",
        "pub type UIToolInvocation",
        "pub type StepStartUIPart",
        "pub fn is_text_ui_part",
        "pub fn is_custom_content_ui_part",
        "pub fn is_file_ui_part",
        "pub fn is_reasoning_file_ui_part",
        "pub fn is_reasoning_ui_part",
        "pub fn is_data_ui_part",
        "pub fn is_static_tool_ui_part",
        "pub fn is_dynamic_tool_ui_part",
        "pub fn is_tool_ui_part",
        "pub fn get_static_tool_name",
        "pub fn get_tool_name",
        "pub fn get_tool_or_dynamic_tool_name",
        "pub fn last_assistant_message_is_complete_with_tool_calls",
        "pub fn last_assistant_message_is_complete_with_approval_responses",
        "pub struct CreateUIMessage",
        "pub struct ChatRequestOptions",
        "pub enum ChatStatus",
        "pub struct ChatState",
        "pub struct ChatInit",
        "pub enum ChatTransportTrigger",
        "pub struct ChatTransportSendMessagesOptions",
        "pub struct ChatTransportReconnectToStreamOptions",
        "pub struct HttpChatTransportInitOptions",
        "pub struct PrepareSendMessagesRequestOptions",
        "pub struct PreparedSendMessagesRequest",
        "pub struct PrepareReconnectToStreamRequestOptions",
        "pub struct PreparedReconnectToStreamRequest",
        "pub struct CompletionRequestOptions",
        "pub enum RequestCredentials",
        "pub enum CompletionStreamProtocol",
        "pub struct UseCompletionOptions",
        "pub struct UiMessageStreamOptions",
        "pub type UIMessageStreamOptions",
    ] {
        assert!(
            ui_message_source.contains(expected),
            "ai_sdk/ui_message.rs should own `{expected}`"
        );
    }

    let ui_message_chunks_source =
        fs::read_to_string(&ui_message_chunks_rs).expect("read ai_sdk/ui_message_chunks.rs");
    for expected in [
        "fn deserialize_ui_message_data_chunk_type",
        "macro_rules! ui_message_id_provider_metadata_chunk",
        "UiMessageTextStartChunk",
        "pub struct UiMessageTextDeltaChunk",
        "UiMessageTextEndChunk",
        "UiMessageReasoningStartChunk",
        "pub struct UiMessageReasoningDeltaChunk",
        "UiMessageReasoningEndChunk",
        "pub struct UiMessageCustomChunk",
        "pub struct UiMessageErrorChunk",
        "pub struct UiMessageToolInputStartChunk",
        "pub struct UiMessageToolInputDeltaChunk",
        "pub struct UiMessageToolInputAvailableChunk",
        "pub struct UiMessageToolInputErrorChunk",
        "pub struct UiMessageToolApprovalRequestChunk",
        "pub struct UiMessageToolApprovalResponseChunk",
        "pub struct UiMessageToolOutputAvailableChunk",
        "pub struct UiMessageToolOutputErrorChunk",
        "pub struct UiMessageToolOutputDeniedChunk",
        "pub struct UiMessageSourceUrlChunk",
        "pub struct UiMessageSourceDocumentChunk",
        "pub struct UiMessageFileChunk",
        "pub struct UiMessageReasoningFileChunk",
        "pub struct UiMessageDataChunk",
        "pub struct UiMessageStartStepChunk",
        "pub struct UiMessageFinishStepChunk",
        "pub struct UiMessageStartChunk",
        "pub struct UiMessageFinishChunk",
        "pub struct UiMessageAbortChunk",
        "pub struct UiMessageMetadataChunk",
        "pub enum UiMessageChunk",
        "pub type UIMessageChunk",
        "pub type DataUIMessageChunk",
        "pub type InferUIMessageChunk",
        "pub fn is_data_ui_message_chunk",
    ] {
        assert!(
            ui_message_chunks_source.contains(expected),
            "ai_sdk/ui_message_chunks.rs should own `{expected}`"
        );
    }

    let usage_source = fs::read_to_string(&usage_rs).expect("read ai_sdk/usage.rs");
    for expected in [
        "pub struct LanguageModelV4InputTokens",
        "pub struct LanguageModelV4OutputTokens",
        "pub struct LanguageModelV4Usage",
        "pub struct LanguageModelInputTokenDetails",
        "pub struct LanguageModelOutputTokenDetails",
        "pub struct LanguageModelUsage",
        "pub struct EmbeddingModelUsage",
        "pub struct ImageModelUsage",
        "pub fn add_language_model_usage",
        "pub fn add_image_model_usage",
    ] {
        assert!(
            usage_source.contains(expected),
            "ai_sdk/usage.rs should own `{expected}`"
        );
    }
}
