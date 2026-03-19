use std::mem::size_of;

#[test]
fn public_surface_unified_imports_compile() {
    use siumai::prelude::unified::*;

    let _ = size_of::<ChatRequest>();
    let _ = size_of::<ChatResponse>();
    let _ = size_of::<ProviderOptionsMap>();
    let _ = size_of::<LlmError>();
    let _ = size_of::<*const dyn LanguageModel>();
    let _ = size_of::<*const dyn RerankingModel>();
    let _ = size_of::<*const dyn SpeechModel>();
    let _ = size_of::<*const dyn TranscriptionModel>();
}

#[test]
fn registry_handles_compile_as_family_models() {
    use siumai::embedding::{EmbeddingModel, EmbeddingModelV3};
    use siumai::image::{ImageModel, ImageModelV3};
    use siumai::prelude::unified::{ModelMetadata, registry::*};
    use siumai::rerank::{RerankModelV3, RerankingModel};
    use siumai::speech::{SpeechModel, SpeechModelV3};
    use siumai::text::{LanguageModel, TextModelV3};
    use siumai::transcription::{TranscriptionModel, TranscriptionModelV3};

    fn _assert_text_handle<T: LanguageModel + TextModelV3 + ModelMetadata>() {}
    fn _assert_embedding_handle<T: EmbeddingModel + EmbeddingModelV3 + ModelMetadata>() {}
    fn _assert_image_handle<T: ImageModel + ImageModelV3 + ModelMetadata>() {}
    fn _assert_rerank_handle<T: RerankingModel + RerankModelV3 + ModelMetadata>() {}
    fn _assert_speech_handle<T: SpeechModel + SpeechModelV3 + ModelMetadata>() {}
    fn _assert_transcription_handle<
        T: TranscriptionModel + TranscriptionModelV3 + ModelMetadata,
    >() {
    }

    let _ = size_of::<LanguageModelHandle>();
    let _ = size_of::<EmbeddingModelHandle>();
    let _ = size_of::<ImageModelHandle>();
    let _ = size_of::<RerankingModelHandle>();
    let _ = size_of::<SpeechModelHandle>();
    let _ = size_of::<TranscriptionModelHandle>();

    _assert_text_handle::<LanguageModelHandle>();
    _assert_embedding_handle::<EmbeddingModelHandle>();
    _assert_image_handle::<ImageModelHandle>();
    _assert_rerank_handle::<RerankingModelHandle>();
    _assert_speech_handle::<SpeechModelHandle>();
    _assert_transcription_handle::<TranscriptionModelHandle>();
}

#[test]
fn public_family_helpers_compile_against_stable_family_models() {
    use siumai::{
        embedding::{self, BatchEmbeddingRequest, EmbeddingModel, EmbeddingRequest},
        image::{self, ImageGenerationRequest, ImageModel},
        prelude::unified::{ChatMessage, registry::*},
        rerank::{self, RerankRequest, RerankingModel},
        speech::{self, SpeechModel, TtsRequest},
        text::{self, LanguageModel, TextRequest},
        transcription::{self, SttRequest, TranscriptionModel},
    };

    fn _assert_text_surface<M: LanguageModel + ?Sized>(model: &M) {
        let request = TextRequest::new(vec![ChatMessage::user("hi").build()]);
        let _ = text::generate(model, request.clone(), Default::default());
        let _ = text::stream(model, request.clone(), Default::default());
        let _ = text::stream_with_cancel(model, request, Default::default());
    }

    fn _assert_embedding_surface<M: EmbeddingModel + ?Sized>(model: &M) {
        let request = EmbeddingRequest::new(vec!["hello".to_string()]);
        let batch = BatchEmbeddingRequest::new(vec![request.clone()]);
        let _ = embedding::embed(model, request, Default::default());
        let _ = embedding::embed_many(model, batch, Default::default());
    }

    fn _assert_image_surface<M: ImageModel + ?Sized>(model: &M) {
        let _ = image::generate(model, ImageGenerationRequest::default(), Default::default());
    }

    fn _assert_rerank_surface<M: RerankingModel + ?Sized>(model: &M) {
        let request = RerankRequest::new(
            "rerank-model".to_string(),
            "hello".to_string(),
            vec!["a".to_string(), "b".to_string()],
        );
        let _ = rerank::rerank(model, request, Default::default());
    }

    fn _assert_speech_surface<M: SpeechModel + ?Sized>(model: &M) {
        let _ = speech::synthesize(
            model,
            TtsRequest::new("hello".to_string()),
            Default::default(),
        );
    }

    fn _assert_transcription_surface<M: TranscriptionModel + ?Sized>(model: &M) {
        let _ = transcription::transcribe(
            model,
            SttRequest::from_audio(Vec::new()),
            Default::default(),
        );
    }

    let _: fn(&LanguageModelHandle) = _assert_text_surface::<LanguageModelHandle>;
    let _: fn(&EmbeddingModelHandle) = _assert_embedding_surface::<EmbeddingModelHandle>;
    let _: fn(&ImageModelHandle) = _assert_image_surface::<ImageModelHandle>;
    let _: fn(&RerankingModelHandle) = _assert_rerank_surface::<RerankingModelHandle>;
    let _: fn(&SpeechModelHandle) = _assert_speech_surface::<SpeechModelHandle>;
    let _: fn(&TranscriptionModelHandle) =
        _assert_transcription_surface::<TranscriptionModelHandle>;
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

#[cfg(feature = "openai")]
#[test]
fn public_surface_openai_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::openai::{
        OpenAiClient, OpenAiConfig,
        ext::{
            OpenAiResponsesEventConverter, moderation, responses, speech_streaming,
            transcription_streaming,
        },
        metadata::*,
        options::*,
        resources::{OpenAiFiles, OpenAiModels, OpenAiModeration, OpenAiRerank},
    };

    let _ = size_of::<OpenAiClient>();
    let _ = size_of::<OpenAiConfig>();
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

    fn _assert_req_ext<T: OpenAiChatRequestExt>() {}
    fn _assert_resp_ext<T: OpenAiChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
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
    use siumai::provider_ext::openai_compatible::{
        ConfigurableAdapter, OpenAiCompatibleClient, OpenAiCompatibleConfig, ProviderAdapter,
        ProviderCompatibility, ProviderConfig, get_provider_config, list_provider_ids,
        provider_supports_capability,
    };

    let _ = size_of::<OpenAiCompatibleClient>();
    let _ = size_of::<OpenAiCompatibleConfig>();
    let _ = size_of::<ConfigurableAdapter>();
    let _ = size_of::<ProviderCompatibility>();
    let _ = size_of::<ProviderConfig>();

    fn _assert_adapter<T: ProviderAdapter>() {}
    _assert_adapter::<ConfigurableAdapter>();

    let _ = get_provider_config("openrouter");
    let _ = list_provider_ids();
    let _ = provider_supports_capability("openrouter", "chat");
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
    outer.insert("openrouter".to_string(), inner);
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

#[cfg(feature = "anthropic")]
#[test]
#[allow(deprecated)]
fn public_surface_anthropic_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::anthropic::{
        AnthropicClient, AnthropicConfig,
        ext::{structured_output, thinking, tools},
        metadata::*,
        options::*,
        resources::{AnthropicFiles, AnthropicMessageBatches, AnthropicTokens},
    };
    use std::collections::HashMap;

    let _ = size_of::<AnthropicClient>();
    let _ = size_of::<AnthropicConfig>();
    let _ = size_of::<AnthropicOptions>();
    let _ = size_of::<AnthropicFiles>();
    let _ = size_of::<AnthropicMessageBatches>();
    let _ = size_of::<AnthropicTokens>();
    let _ = size_of::<AnthropicEffort>();
    let _ = size_of::<AnthropicResponseFormat>();
    let _ = size_of::<AnthropicStructuredOutputMode>();
    let _ = size_of::<ThinkingModeConfig>();
    let _ = size_of::<AnthropicMetadata>();
    let _ = size_of::<AnthropicSource>();
    let _ = size_of::<tools::AnthropicCustomEvent>();
    let _ = size_of::<tools::AnthropicProviderToolCallEvent>();
    let _ = size_of::<tools::AnthropicProviderToolResultEvent>();
    let _ = size_of::<tools::AnthropicSourceEvent>();
    let _ = AnthropicClient::set_retry_options;

    fn _assert_req_ext<T: AnthropicChatRequestExt>() {}
    fn _assert_resp_ext<T: AnthropicChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
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
    outer.insert("anthropic".to_string(), inner);
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
        .with_anthropic_context_management(serde_json::json!({
            "clear_at_least": 1
        }))
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
        .with_anthropic_context_management(serde_json::json!({
            "clear_at_least": 1
        }))
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
        .with_anthropic_context_management(serde_json::json!({
            "clear_at_least": 1
        }))
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
    use siumai::prelude::unified::*;
    use siumai::provider_ext::gemini::{
        GeminiClient,
        ext::{code_execution, file_search_stores, tools},
        metadata::*,
        options::*,
        resources::{
            GeminiCachedContents, GeminiFileSearchStores, GeminiFiles, GeminiModels, GeminiTokens,
        },
    };

    let _ = size_of::<GeminiClient>();
    let _ = size_of::<GeminiOptions>();
    let _ = size_of::<GeminiThinkingConfig>();
    let _ = size_of::<GeminiMetadata>();
    let _ = size_of::<GeminiSource>();
    let _ = size_of::<GeminiCachedContents>();
    let _ = size_of::<GeminiFileSearchStores>();
    let _ = size_of::<GeminiFiles>();
    let _ = size_of::<GeminiModels>();
    let _ = size_of::<GeminiTokens>();
    let _ = size_of::<code_execution::CodeExecutionConfig>();
    let _ = size_of::<code_execution::CodeExecutionResult>();
    let _ = size_of::<tools::GeminiCustomEvent>();
    let _ = size_of::<tools::GeminiSourceEvent>();
    let _ = GeminiClient::base_url;
    let _ = GeminiClient::set_retry_options;

    fn _assert_req_ext<T: GeminiChatRequestExt>() {}
    fn _assert_resp_ext<T: GeminiChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
    let _ = file_search_stores::stores;

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
    use siumai::prelude::unified::*;
    use siumai::provider_ext::google::{
        GeminiClient, GeminiConfig,
        ext::{code_execution, file_search_stores, tools},
        metadata::*,
        options::*,
        resources::{
            GeminiCachedContents, GeminiFileSearchStores, GeminiFiles, GeminiModels, GeminiTokens,
        },
    };

    let _ = size_of::<GeminiClient>();
    let _ = size_of::<GeminiConfig>();
    let _ = size_of::<GeminiOptions>();
    let _ = size_of::<GeminiCachedContents>();
    let _ = size_of::<GeminiFileSearchStores>();
    let _ = size_of::<GeminiFiles>();
    let _ = size_of::<GeminiModels>();
    let _ = size_of::<GeminiTokens>();
    let _ = size_of::<GeminiThinkingConfig>();
    let _ = size_of::<GeminiMetadata>();
    let _ = size_of::<GeminiSource>();
    let _ = size_of::<code_execution::CodeExecutionConfig>();
    let _ = size_of::<code_execution::CodeExecutionResult>();
    let _ = size_of::<tools::GeminiCustomEvent>();
    let _ = size_of::<tools::GeminiSourceEvent>();
    let _ = GeminiClient::base_url;
    let _ = GeminiClient::set_retry_options;

    fn _assert_req_ext<T: GeminiChatRequestExt>() {}
    fn _assert_resp_ext<T: GeminiChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
    let _ = file_search_stores::stores;

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
fn public_surface_cohere_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::cohere::{CohereClient, CohereConfig, options::*};

    let _ = size_of::<CohereClient>();
    let _ = size_of::<CohereConfig>();
    let _ = size_of::<CohereRerankOptions>();
    let _ = CohereClient::provider_context;
    let _ = CohereClient::base_url;
    let _ = CohereClient::http_client;
    let _ = CohereClient::retry_options;
    let _ = CohereClient::http_interceptors;
    let _ = CohereClient::http_transport;
    let _ = CohereClient::set_retry_options;

    let req = RerankRequest::new(
        "rerank-english-v3.0".to_string(),
        "query".to_string(),
        vec!["doc-1".to_string()],
    )
    .with_cohere_options(CohereRerankOptions::new().with_priority(1));
    let _ = req;
}

#[cfg(feature = "togetherai")]
#[test]
fn public_surface_togetherai_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::togetherai::{TogetherAiClient, TogetherAiConfig, options::*};

    let _ = size_of::<TogetherAiClient>();
    let _ = size_of::<TogetherAiConfig>();
    let _ = size_of::<TogetherAiRerankOptions>();
    let _ = TogetherAiClient::provider_context;
    let _ = TogetherAiClient::base_url;
    let _ = TogetherAiClient::http_client;
    let _ = TogetherAiClient::retry_options;
    let _ = TogetherAiClient::http_interceptors;
    let _ = TogetherAiClient::http_transport;
    let _ = TogetherAiClient::set_retry_options;

    let req = RerankRequest::new(
        "Salesforce/Llama-Rank-v1".to_string(),
        "query".to_string(),
        vec!["doc-1".to_string()],
    )
    .with_togetherai_options(
        TogetherAiRerankOptions::new().with_rank_fields(vec!["example".to_string()]),
    );
    let _ = req;
}

#[cfg(feature = "bedrock")]
#[test]
fn public_surface_bedrock_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::bedrock::{BedrockClient, BedrockConfig, metadata::*, options::*};

    let _ = size_of::<BedrockClient>();
    let _ = size_of::<BedrockConfig>();
    let _ = size_of::<BedrockChatOptions>();
    let _ = size_of::<BedrockRerankOptions>();
    let _ = size_of::<BedrockMetadata>();
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
    outer.insert("bedrock".to_string(), inner);
    resp.provider_metadata = Some(outer);
    let _ = resp.bedrock_metadata();

    let _ = (chat_req, rerank_req);
}

#[cfg(feature = "google-vertex")]
#[test]
fn public_surface_google_vertex_provider_ext_compiles() {
    use siumai::prelude::unified::{ChatResponse, ContentPart, MessageContent};
    use siumai::provider_ext::google_vertex::{
        GoogleVertexClient, GoogleVertexConfig, metadata::*, options::*,
    };

    let _ = size_of::<GoogleVertexClient>();
    let _ = size_of::<GoogleVertexConfig>();
    let _ = size_of::<VertexEmbeddingOptions>();
    let _ = size_of::<VertexImagenOptions>();
    let _ = size_of::<VertexMetadata>();
    let _ = size_of::<VertexSource>();
    let _ = size_of::<VertexGroundingMetadata>();
    let _ = size_of::<VertexUrlContextMetadata>();
    let _ = size_of::<VertexUsageMetadata>();
    let _ = size_of::<VertexSafetyRating>();
    let _ = GoogleVertexClient::base_url;

    fn _assert_resp_ext<T: VertexChatResponseExt>() {}
    fn _assert_part_ext<T: VertexContentPartExt>() {}
    _assert_resp_ext::<ChatResponse>();
    _assert_part_ext::<ContentPart>();

    let mut resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut inner = std::collections::HashMap::new();
    inner.insert(
        "usageMetadata".to_string(),
        serde_json::json!({
            "totalTokenCount": 8
        }),
    );
    let mut outer = std::collections::HashMap::new();
    outer.insert("vertex".to_string(), inner);
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
    let _ = siumai::Provider::vertex();
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
    };

    let _ = size_of::<VertexAnthropicBuilder>();
    let _ = size_of::<VertexAnthropicClient>();
    let _ = size_of::<VertexAnthropicConfig>();
    let _ = size_of::<AnthropicMetadata>();
    let _ = size_of::<VertexAnthropicOptions>();
    let _ = size_of::<VertexAnthropicThinkingMode>();
    let _ = size_of::<VertexAnthropicStructuredOutputMode>();
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
    use siumai::provider_ext::groq::{GroqClient, ext::audio_options, metadata::*, options::*};

    let _ = size_of::<GroqClient>();
    let _ = size_of::<GroqOptions>();
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
    let _ = size_of::<audio_options::GroqTtsOptions>();
    let _ = size_of::<audio_options::GroqSttOptions>();
    let _ = size_of::<GroqMetadata>();
    let _ = size_of::<GroqSource>();
    let _ = size_of::<GroqSourceMetadata>();

    fn _assert_req_ext<T: GroqChatRequestExt>() {}
    fn _assert_resp_ext<T: GroqChatResponseExt>() {}
    fn _assert_source_ext<T: GroqSourceExt>() {}
    _assert_req_ext::<siumai::prelude::unified::ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
    _assert_source_ext::<GroqSource>();
}

#[cfg(feature = "xai")]
#[test]
fn public_surface_xai_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::xai::{XaiClient, XaiConfig, metadata::*, options::*};
    use std::collections::HashMap;

    let _ = size_of::<XaiClient>();
    let _ = size_of::<XaiConfig>();
    let _ = size_of::<XaiOptions>();
    let _ = size_of::<XaiTtsOptions>();
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

    fn _assert_req_ext<T: XaiChatRequestExt>() {}
    fn _assert_tts_req_ext<T: XaiTtsRequestExt>() {}
    fn _assert_resp_ext<T: XaiChatResponseExt>() {}
    fn _assert_source_ext<T: XaiSourceExt>() {}
    _assert_req_ext::<siumai::prelude::unified::ChatRequest>();
    _assert_tts_req_ext::<TtsRequest>();
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
    outer.insert("xai".to_string(), inner);
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
    let _ = siumai::provider_ext::xai::provider_tools::web_search();
    let _ = siumai::provider_ext::xai::provider_tools::x_search();
    let _ = siumai::provider_ext::xai::provider_tools::code_execution();
    let _ = siumai::Provider::xai();
}

#[cfg(feature = "ollama")]
#[test]
fn public_surface_ollama_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::ollama::{
        OllamaClient, OllamaConfig, OllamaParams, metadata::*, options::*,
    };

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
        MinimaxiClient, MinimaxiConfig,
        ext::{music, structured_output, thinking, video},
        metadata::*,
        options::*,
        resources::*,
    };
    use std::collections::HashMap;

    let _ = size_of::<MinimaxiClient>();
    let _ = size_of::<MinimaxiConfig>();
    let _ = size_of::<MinimaxiOptions>();
    let _ = size_of::<MinimaxiResponseFormat>();
    let _ = size_of::<MinimaxiThinkingModeConfig>();
    let _ = size_of::<MinimaxiTtsOptions>();
    let _ = size_of::<MinimaxiTtsRequestBuilder>();
    let _ = size_of::<music::MinimaxiMusicRequestBuilder>();
    let _ = size_of::<video::MinimaxiVideoRequestBuilder>();
    let _ = size_of::<MinimaxiFiles>();
    let _ = size_of::<MinimaxiMetadata>();
    let _ = size_of::<MinimaxiSource>();
    let _ = size_of::<MinimaxiToolCallMetadata>();
    let _ = size_of::<MinimaxiToolCaller>();

    fn _assert_chat_req_ext<T: MinimaxiChatRequestExt>() {}
    fn _assert_req_ext<T: MinimaxiTtsRequestExt>() {}
    fn _assert_resp_ext<T: MinimaxiChatResponseExt>() {}
    fn _assert_part_ext<T: MinimaxiContentPartExt>() {}
    _assert_chat_req_ext::<ChatRequest>();
    _assert_req_ext::<siumai::prelude::unified::TtsRequest>();
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
    outer.insert("minimaxi".to_string(), inner);
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

    let mut resp = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut inner = std::collections::HashMap::new();
    inner.insert("service_tier".to_string(), serde_json::json!("default"));
    let mut outer = std::collections::HashMap::new();
    outer.insert("azure".to_string(), inner);
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
    use siumai::provider_ext::deepseek::{DeepSeekClient, DeepSeekConfig, metadata::*, options::*};

    let _ = size_of::<DeepSeekClient>();
    let _ = size_of::<DeepSeekConfig>();
    let _ = size_of::<DeepSeekOptions>();
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

    fn _assert_req_ext<T: DeepSeekChatRequestExt>() {}
    fn _assert_resp_ext<T: DeepSeekChatResponseExt>() {}
    fn _assert_source_ext<T: DeepSeekSourceExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();
    _assert_source_ext::<DeepSeekSource>();
}
