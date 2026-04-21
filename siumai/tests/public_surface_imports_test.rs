use std::mem::size_of;

#[test]
#[allow(deprecated)]
fn public_surface_unified_imports_compile() {
    use siumai::prelude::unified::*;

    let _ = size_of::<ChatRequest>();
    let _ = size_of::<ChatResponse>();
    let _ = size_of::<CallSettings>();
    let _ = size_of::<JSONValue>();
    let _ = size_of::<CallWarning>();
    let _ = size_of::<CancelHandle>();
    let _ = size_of::<Context>();
    let _ = size_of::<DataContent>();
    let _ = size_of::<CompletionRequest>();
    let _ = size_of::<CompletionResponse>();
    let _ = size_of::<EmbeddingModelUsage>();
    let _ = size_of::<TextPart>();
    let _ = size_of::<ImagePart>();
    let _ = size_of::<FilePart>();
    let _ = size_of::<ReasoningPart>();
    let _ = size_of::<CustomPart>();
    let _ = size_of::<ReasoningFilePart>();
    let _ = size_of::<ToolCallPart>();
    let _ = size_of::<ToolResultPart>();
    let _ = size_of::<ToolApprovalRequest>();
    let _ = size_of::<ToolApprovalResponse>();
    let _ = size_of::<UserContentPart>();
    let _ = size_of::<AssistantContentPart>();
    let _ = size_of::<ToolContentPart>();
    let _ = size_of::<UserContent>();
    let _ = size_of::<AssistantContent>();
    let _ = size_of::<ToolContent>();
    let _ = size_of::<GenerateImageRequest>();
    let _ = size_of::<ImageModelProviderMetadata>();
    let _ = size_of::<ImageModelResponseMetadata>();
    let _ = size_of::<ImageModelUsage>();
    let _ = size_of::<LanguageModelCallOptions>();
    let _ = size_of::<LanguageModelInputTokenDetails>();
    let _ = size_of::<LanguageModelOutputTokenDetails>();
    let _ = size_of::<LanguageModelReasoning>();
    let _ = size_of::<LanguageModelRequestMetadata>();
    let _ = size_of::<LanguageModelResponseMetadata>();
    let _ = size_of::<LanguageModelUsage>();
    let _ = size_of::<ProviderMetadata>();
    let _ = size_of::<ProviderOptions>();
    let _ = size_of::<RequestOptions>();
    let _ = size_of::<StreamRequestOptions>();
    let _ = size_of::<SpeechModelResponseMetadata>();
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
    let _ = size_of::<TimeoutConfiguration>();
    let _ = size_of::<TimeoutConfigurationSettings>();
    let _ = size_of::<TranscriptionModelResponseMetadata>();
    let _ = size_of::<VideoModelProviderMetadata>();
    let _ = size_of::<VideoModelResponseMetadata>();
    let _ = size_of::<ProviderOptionsMap>();
    let _ = size_of::<ModelMessageConversionError>();
    let _ = size_of::<PromptValidationError>();
    let _ = size_of::<ToolCall<String, JSONValue>>();
    let _ = size_of::<ToolResult<String, JSONValue, ToolResultOutput>>();
    let _ = size_of::<LlmError>();
    let _ = size_of::<*const dyn CompletionCapability>();
    let _ = size_of::<*const dyn CompletionModel>();
    let _ = size_of::<*const dyn ImageModel>();
    let _ = size_of::<*const dyn ImageModelV4>();
    let _ = size_of::<*const dyn LanguageModel>();
    let _ = size_of::<*const dyn RerankingModel>();
    let _ = size_of::<*const dyn SpeechModel>();
    let _ = size_of::<*const dyn TranscriptionModel>();
    let _ = get_total_timeout_ms as fn(Option<&TimeoutConfiguration>) -> Option<u64>;
    let _ = get_step_timeout_ms as fn(Option<&TimeoutConfiguration>) -> Option<u64>;
    let _ = get_chunk_timeout_ms as fn(Option<&TimeoutConfiguration>) -> Option<u64>;
    let _ = get_tool_timeout_ms as fn(Option<&TimeoutConfiguration>, &str) -> Option<u64>;
    let _ = convert_data_content_to_base64_string as fn(&DataContent) -> String;

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

    let tool_call = ToolCall::new(
        "call_1",
        "search".to_string(),
        serde_json::json!({ "q": "rust" }),
    )
    .with_provider_executed(true)
    .with_dynamic(true);
    assert_eq!(tool_call.tool_call_id, "call_1");

    let tool_result = ToolResult::new(
        "call_1",
        "search".to_string(),
        serde_json::json!({ "q": "rust" }),
        ToolResultOutput::json(serde_json::json!({ "ok": true })),
    )
    .with_provider_executed(true)
    .with_dynamic(true);
    assert_eq!(tool_result.tool_call_id, "call_1");

    let approval_response = ToolApprovalResponse::new("approval_1", true)
        .with_reason("approved")
        .with_provider_executed(true);
    assert_eq!(approval_response.approval_id, "approval_1");
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
}

#[tokio::test]
async fn public_surface_tooling_imports_compile() {
    use futures::StreamExt;
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
            self, GenerateImageRequest, ImageEditInput, ImageEditRequest, ImageGenerationRequest,
            ImageModel, ImageVariationRequest,
        },
        prelude::unified::{ChatMessage, registry::*},
        rerank::{self, RerankRequest, RerankingModel},
        speech::{self, SpeechModel, TtsRequest},
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
        std::mem::drop(text::stream_with_cancel(model, request, Default::default()));
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
            GenerateImageRequest::new("draw a robot"),
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
        QueryTaskOptions, VideoGenerationFileData, VideoGenerationInput, VideoGenerationPrompt,
        VideoGenerationRequest, VideoGenerationResponse, VideoModel, VideoModelV3, VideoModelV4,
        VideoTaskStatus, VideoTaskStatusResponse, WaitForTaskOptions,
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
        OpenAiBuilder, OpenAiClient, OpenAiConfig,
        ext::{
            OpenAiResponsesEventConverter, moderation, responses, speech_streaming,
            transcription_streaming,
        },
        metadata::*,
        options::*,
        resources::{OpenAiFiles, OpenAiModels, OpenAiModeration, OpenAiRerank},
    };

    let _ = size_of::<OpenAiBuilder>();
    let _ = size_of::<OpenAiClient>();
    let _ = size_of::<OpenAiConfig>();
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
        OpenAICompatibleRequestSettings, OpenAiCompatibleChatModelId, OpenAiCompatibleClient,
        OpenAiCompatibleCompletionModelId, OpenAiCompatibleConfig,
        OpenAiCompatibleEmbeddingModelId, OpenAiCompatibleErrorData, OpenAiCompatibleImageModelId,
        OpenAiCompatibleRequestSettings, ProviderAdapter, ProviderCompatibility, ProviderConfig,
        ProviderErrorStructure, RequestBodyTransformer, ResponseMetadataExtractor, deepinfra,
        fireworks, get_provider_config, groq, list_provider_ids, moonshot, moonshotai, openrouter,
        options::*, provider_supports_capability, siliconflow, xai,
    };
    use std::sync::Arc;

    let _ = size_of::<OpenAICompatibleChatModelId>();
    let _ = size_of::<OpenAICompatibleClient>();
    let _ = size_of::<OpenAICompatibleCompletionModelId>();
    let _ = size_of::<OpenAICompatibleConfig>();
    let _ = size_of::<OpenAICompatibleEmbeddingModelId>();
    let _ = size_of::<OpenAICompatibleErrorData>();
    let _ = size_of::<OpenAICompatibleImageModelId>();
    let _ = size_of::<OpenAICompatibleRequestSettings>();
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
        AnthropicBuilder, AnthropicClient, AnthropicConfig,
        ext::{structured_output, thinking, tools},
        metadata::*,
        options::*,
        resources::{AnthropicFiles, AnthropicMessageBatches, AnthropicTokens},
    };
    use std::collections::HashMap;

    let _ = size_of::<AnthropicBuilder>();
    let _ = size_of::<AnthropicClient>();
    let _ = size_of::<AnthropicConfig>();
    let _ = size_of::<AnthropicOptions>();
    let _ = size_of::<AnthropicLanguageModelOptions>();
    let _ = size_of::<AnthropicProviderOptions>();
    let _ = size_of::<AnthropicFiles>();
    let _ = size_of::<AnthropicMessageBatches>();
    let _ = size_of::<AnthropicTokens>();
    let _ = size_of::<AnthropicContainerSkillType>();
    let _ = size_of::<AnthropicEffort>();
    let _ = size_of::<AnthropicContextManagementConfig>();
    let _ = size_of::<AnthropicContextManagementEdit>();
    let _ = size_of::<AnthropicMcpServer>();
    let _ = size_of::<AnthropicToolAllowedCaller>();
    let _ = size_of::<AnthropicToolOptions>();
    let _ = size_of::<AnthropicRequestMetadata>();
    let _ = size_of::<AnthropicSpeed>();
    let _ = size_of::<AnthropicThinkingConfig>();
    let _ = size_of::<AnthropicResponseFormat>();
    let _ = size_of::<AnthropicStructuredOutputMode>();
    let _ = size_of::<ThinkingModeConfig>();
    let _ = size_of::<AnthropicMetadata>();
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
        .with_anthropic_effort(AnthropicEffort::High);
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
        .with_anthropic_effort(AnthropicEffort::High);
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
        .with_anthropic_effort(AnthropicEffort::High);
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
        GeminiBuilder, GeminiClient,
        ext::{code_execution, file_search_stores, tools},
        metadata::*,
        options::*,
        resources::{
            GeminiCachedContents, GeminiFileSearchStores, GeminiFiles, GeminiModels, GeminiTokens,
            GeminiVideo, GoogleErrorData,
        },
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
}

#[cfg(feature = "google")]
#[test]
fn public_surface_google_provider_ext_compiles() {
    use siumai::prelude::extensions::types::VideoGenerationRequest;
    use siumai::prelude::unified::*;
    use siumai::provider_ext::google::{
        GeminiBuilder, GeminiClient, GeminiConfig, GoogleErrorData,
        ext::{code_execution, file_search_stores, tools},
        metadata::*,
        options::*,
        resources::{
            GeminiCachedContents, GeminiFileSearchStores, GeminiFiles, GeminiModels, GeminiTokens,
            GeminiVideo,
        },
    };

    let _ = size_of::<GeminiBuilder>();
    let _ = size_of::<GeminiClient>();
    let _ = size_of::<GeminiConfig>();
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
    let _ = GeminiClient::base_url;
    let _ = GeminiClient::set_retry_options;

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
}

#[cfg(feature = "cohere")]
#[test]
#[allow(deprecated)]
fn public_surface_cohere_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::cohere::{
        CohereBuilder, CohereClient, CohereConfig, chat, embedding, model_sets, options::*, rerank,
    };

    let _ = size_of::<CohereBuilder>();
    let _ = size_of::<CohereClient>();
    let _ = size_of::<CohereConfig>();
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
        TogetherAIErrorData, TogetherAiBuilder, TogetherAiClient, TogetherAiConfig, chat,
        completion, embedding, image, model_sets, options::*, rerank,
    };

    let _ = size_of::<TogetherAiBuilder>();
    let _ = size_of::<TogetherAiClient>();
    let _ = size_of::<TogetherAiConfig>();
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
        DeepInfraClient, DeepInfraConfig, DeepInfraErrorData, chat, completion, embedding, image,
        model_sets,
    };

    let _ = size_of::<DeepInfraClient>();
    let _ = size_of::<DeepInfraConfig>();
    let _ = size_of::<DeepInfraErrorData>();
    let _ = Provider::deepinfra;
    let _ = Siumai::builder().deepinfra();
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

    let _ = Provider::mistral;
    let _ = Provider::fireworks;
    let _ = Provider::perplexity;
    let _ = Provider::moonshotai;
    let _ = Siumai::builder().mistral();
    let _ = Siumai::builder().fireworks();
    let _ = Siumai::builder().perplexity();
    let _ = Siumai::builder().moonshotai();
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
        MoonshotAIReasoningHistory, MoonshotAIThinkingConfig, MoonshotAIThinkingType, model_sets,
        recommended,
    };

    let _ = size_of::<MoonshotAIClient>();
    let _ = size_of::<MoonshotAIConfig>();
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
}

#[cfg(feature = "bedrock")]
#[test]
#[allow(deprecated)]
fn public_surface_bedrock_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::bedrock::{
        BedrockBuilder, BedrockClient, BedrockConfig, BedrockEmbeddingRequestExt,
        BedrockMessageExt, BedrockRequestContentPartExt, assistant_message_with_reasoning_metadata,
        metadata::*, options::*,
    };

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
        FireworksProviderOptions, FireworksReasoningHistory, FireworksThinkingConfig,
        FireworksThinkingType, chat as fireworks_chat,
    };
    use siumai::provider_ext::mistral::{
        MistralChatOptions, MistralChatRequestExt, MistralClient, MistralConfig,
        MistralReasoningEffort, chat as mistral_chat, embedding as mistral_embedding,
    };
    use siumai::provider_ext::perplexity::{
        PerplexityChatRequestExt, PerplexityClient, PerplexityConfig, PerplexityOptions,
        chat as perplexity_chat,
    };

    let _ = size_of::<MistralClient>();
    let _ = size_of::<MistralConfig>();
    let _ = size_of::<MistralChatOptions>();
    let _ = size_of::<MistralReasoningEffort>();
    let _ = size_of::<FireworksClient>();
    let _ = size_of::<FireworksConfig>();
    let _ = size_of::<FireworksChatOptions>();
    let _ = size_of::<FireworksErrorData>();
    let _ = size_of::<FireworksEmbeddingModelId>();
    let _ = size_of::<FireworksEmbeddingModelOptions>();
    let _ = size_of::<FireworksEmbeddingProviderOptions>();
    let _ = size_of::<FireworksImageModelId>();
    let _ = size_of::<FireworksProviderOptions>();
    let _ = size_of::<PerplexityClient>();
    let _ = size_of::<PerplexityConfig>();
    let _ = size_of::<PerplexityOptions>();

    let _ = mistral_chat::MISTRAL_LARGE_LATEST;
    let _ = mistral_embedding::MISTRAL_EMBED;
    let _ = fireworks_chat::LLAMA_V3P1_8B_INSTRUCT;
    let _ = perplexity_chat::SONAR;

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
        GoogleVertexBuilder, GoogleVertexClient, GoogleVertexConfig, chat, embedding, image,
        metadata::*, model_sets, options::*, video,
    };

    let _ = size_of::<GoogleVertexBuilder>();
    let _ = size_of::<GoogleVertexClient>();
    let _ = size_of::<GoogleVertexConfig>();
    let _ = size_of::<GoogleVertexEmbeddingModelOptions>();
    let _ = size_of::<GoogleVertexImageModelOptions>();
    let _ = size_of::<GoogleVertexImageProviderOptions>();
    let _ = size_of::<GoogleVertexReferenceImage>();
    let _ = size_of::<GoogleVertexVideoModelId>();
    let _ = size_of::<GoogleVertexVideoModelOptions>();
    let _ = size_of::<GoogleVertexVideoProviderOptions>();
    let _ = size_of::<VertexEmbeddingOptions>();
    let _ = size_of::<VertexImagenOptions>();
    let _ = size_of::<VertexMetadata>();
    let _ = size_of::<VertexSource>();
    let _ = size_of::<VertexGroundingMetadata>();
    let _ = size_of::<VertexUrlContextMetadata>();
    let _ = size_of::<VertexUsageMetadata>();
    let _ = size_of::<VertexSafetyRating>();
    let _ = GoogleVertexClient::base_url;
    let _ = chat::GEMINI_2_5_FLASH;
    let _ = embedding::TEXT_EMBEDDING_004;
    let _ = image::IMAGEN_3_0_EDIT_001;
    let _ = video::VEO_3_1_GENERATE_PREVIEW;
    let _ = model_sets::ALL_CHAT;
    let _ = model_sets::ALL_EMBEDDING;
    let _ = model_sets::ALL_IMAGE;
    let _ = model_sets::ALL_VIDEO;

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
            .with_person_generation("allow_adult")
            .with_safety_setting("block_medium_and_above")
            .with_add_watermark(false)
            .with_sample_image_size("2K"),
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
    use siumai::provider_ext::vertex_maas::{chat, completion, embedding, model_sets};

    let _ = chat::DEEPSEEK_V3_2_MAAS;
    let _ = completion::DEEPSEEK_V3_2_MAAS;
    let _ = embedding::DEEPSEEK_V3_2_MAAS;
    let _ = model_sets::ALL_CHAT;
    let _ = model_sets::ALL_COMPLETION;
    let _ = model_sets::ALL_EMBEDDING;
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
        AnthropicChatResponseExt, AnthropicMetadata, VertexAnthropicBuilder,
        VertexAnthropicChatRequestExt, VertexAnthropicClient, VertexAnthropicConfig,
        VertexAnthropicOptions, VertexAnthropicStructuredOutputMode, VertexAnthropicThinkingMode,
        chat, model_sets,
    };

    let _ = size_of::<VertexAnthropicBuilder>();
    let _ = size_of::<VertexAnthropicClient>();
    let _ = size_of::<VertexAnthropicConfig>();
    let _ = size_of::<AnthropicMetadata>();
    let _ = size_of::<VertexAnthropicOptions>();
    let _ = size_of::<VertexAnthropicThinkingMode>();
    let _ = size_of::<VertexAnthropicStructuredOutputMode>();
    let _ = chat::CLAUDE_SONNET_4_5_LATEST;
    let _ = model_sets::ALL_CHAT;
    let _ = ChatRequest::new(vec![]).with_anthropic_vertex_options(
        VertexAnthropicOptions::new()
            .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048))),
    );
    let _ = ChatResponse::new(MessageContent::Text(String::new())).anthropic_metadata();
    let _ = VertexAnthropicConfig::new("https://example.com/custom", "claude-sonnet-4-5-latest")
        .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_disable_parallel_tool_use(true)
        .with_send_reasoning(false);
    let _ = siumai::Provider::anthropic_vertex()
        .base_url("https://example.com/custom")
        .model("claude-sonnet-4-5-latest")
        .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_disable_parallel_tool_use(true)
        .with_send_reasoning(false);
    let _ = Siumai::builder()
        .anthropic_vertex()
        .base_url("https://example.com/custom")
        .model("claude-sonnet-4-5-latest")
        .with_anthropic_vertex_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_anthropic_vertex_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_anthropic_vertex_disable_parallel_tool_use(true)
        .with_anthropic_vertex_send_reasoning(false);
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
        GroqBuilder, GroqClient, ext::audio_options, metadata::*, options::*,
    };

    let _ = size_of::<GroqBuilder>();
    let _ = size_of::<GroqClient>();
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
        XaiBuilder, XaiClient, XaiConfig, XaiErrorData, XaiVideoModelId, metadata::*, options::*,
    };
    use std::collections::HashMap;

    let _ = size_of::<XaiBuilder>();
    let _ = size_of::<XaiClient>();
    let _ = size_of::<XaiConfig>();
    let _ = size_of::<XaiErrorData>();
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
        AzureChatMode, AzureOpenAiBuilder, AzureOpenAiClient, AzureOpenAiConfig, AzureOpenAiSpec,
        AzureUrlConfig, metadata::*, options::*,
    };

    let _ = size_of::<AzureOpenAiBuilder>();
    let _ = size_of::<AzureOpenAiClient>();
    let _ = size_of::<AzureOpenAiConfig>();
    let _ = size_of::<AzureOpenAiSpec>();
    let _ = size_of::<AzureUrlConfig>();
    let _ = size_of::<AzureChatMode>();
    let _ = size_of::<AzureOpenAiOptions>();
    let _ = size_of::<AzureResponsesApiConfig>();
    let _ = size_of::<AzureReasoningEffort>();
    let _ = size_of::<OpenAILanguageModelChatOptions>();
    let _ = size_of::<OpenAIChatLanguageModelOptions>();
    let _ = size_of::<OpenAILanguageModelResponsesOptions>();
    let _ = size_of::<OpenAIResponsesProviderOptions>();
    let _ = size_of::<AzureMetadata>();
    let _ = size_of::<AzureSource>();
    let _ = size_of::<AzureSourceMetadata>();
    let _ = size_of::<AzureContentPartMetadata>();

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
        DeepSeekBuilder, DeepSeekClient, DeepSeekConfig, DeepSeekErrorData, chat, metadata::*,
        model_sets, options::*,
    };

    let _ = size_of::<DeepSeekBuilder>();
    let _ = size_of::<DeepSeekClient>();
    let _ = size_of::<DeepSeekConfig>();
    let _ = size_of::<DeepSeekErrorData>();
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
