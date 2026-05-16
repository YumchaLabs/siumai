//! AI SDK-aligned shared surface aliases and metadata helpers.
//!
//! These names intentionally mirror the shared `packages/ai/src/types/*` contract where
//! Siumai already has a stable equivalent or can expose a passive data structure honestly
//! without pretending the runtime wiring is more complete than it is today.

mod call_options;
mod embedding;
mod errors;
mod flow_control;
mod generate_object;
mod generate_text;
mod generated_files;
mod language_model_metadata;
mod language_model_results;
mod language_model_v4;
mod media_results;
mod object_stream;
mod output_parts;
mod rerank;
mod response_metadata;
mod shared;
mod source;
mod text_stream;
mod timeout;
mod tool_lifecycle;
mod ui_message;
mod ui_message_chunks;
mod usage;

pub use call_options::*;
pub use embedding::*;
pub use errors::*;
pub use flow_control::*;
pub use generate_object::*;
pub use generate_text::*;
pub use generated_files::*;
pub use language_model_metadata::*;
pub use language_model_results::*;
pub use language_model_v4::*;
pub use media_results::*;
pub use object_stream::*;
pub use output_parts::*;
pub use rerank::*;
pub use response_metadata::*;
pub use shared::*;
pub use source::*;
pub use text_stream::*;
pub use timeout::*;
pub use tool_lifecycle::*;
pub use ui_message::*;
pub use ui_message_chunks::*;
pub use usage::*;

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use std::collections::HashMap;
    use std::time::Duration;

    use super::super::{
        AssistantContent, AssistantContentPart, AssistantModelMessage, ContentPart, CustomPart,
        EmbeddingUsage, FilePart, FilePartSource, FinishReason, HttpRequestInfo, HttpResponseInfo,
        ImagePart, LanguageModelV4Tool, LanguageModelV4ToolChoice, MediaSource, ModelMessage,
        PromptInput, ProviderOptionsMap, ProviderReference, ReasoningFilePart, ReasoningPart,
        ResponseMetadata, StandardizedPrompt, SystemModelMessage, SystemPrompt, TextPart, Tool,
        ToolApprovalRequest, ToolApprovalResponse, ToolCallPart, ToolChoice, ToolContentPart,
        ToolModelMessage, ToolResultContentPart, ToolResultOutput, ToolResultPart, UiMessage,
        UiMessagePart, UiMessageRole, Usage, UserContent, UserContentPart, UserModelMessage,
        Warning,
    };
    use super::*;

    #[test]
    fn source_shape_matches_language_model_source_contract() {
        let mut provider_metadata = ProviderMetadata::new();
        provider_metadata.insert("anthropic".to_string(), serde_json::json!({ "foo": "bar" }));

        let source = Source::url_with_title("source-0", "https://example.com", "Example")
            .with_provider_metadata(provider_metadata);

        let value = serde_json::to_value(&source).expect("serialize source");
        assert_eq!(value["type"], serde_json::json!("source"));
        assert_eq!(value["sourceType"], serde_json::json!("url"));
        assert_eq!(value["id"], serde_json::json!("source-0"));
        assert_eq!(value["url"], serde_json::json!("https://example.com"));
        assert_eq!(value["title"], serde_json::json!("Example"));
        assert_eq!(
            value["providerMetadata"]["anthropic"],
            serde_json::json!({ "foo": "bar" })
        );

        let roundtrip: Source = serde_json::from_value(value).expect("deserialize source");
        assert_eq!(roundtrip.r#type(), "source");
        assert_eq!(roundtrip.source_type(), "url");
        assert_eq!(roundtrip, source);

        let content_part: ContentPart = source.clone().into();
        let converted = Source::try_from(&content_part).expect("convert source content part");
        assert_eq!(converted, source);
    }

    #[test]
    fn source_rejects_non_source_type_marker() {
        let error = serde_json::from_value::<Source>(serde_json::json!({
            "type": "text",
            "sourceType": "url",
            "id": "source-0",
            "url": "https://example.com"
        }))
        .expect_err("non-source marker should be rejected");

        assert!(error.to_string().contains("expected source type marker"));
    }

    #[test]
    fn language_model_request_metadata_parses_json_body_and_falls_back_to_string() {
        let json = LanguageModelRequestMetadata::from(HttpRequestInfo {
            body: Some("{\"ok\":true}".to_string()),
        });
        let text = LanguageModelRequestMetadata::from(HttpRequestInfo {
            body: Some("plain-text".to_string()),
        });

        assert_eq!(json.body, Some(serde_json::json!({ "ok": true })));
        assert_eq!(text.body, Some(JSONValue::String("plain-text".to_string())));
    }

    #[test]
    fn call_warning_uses_shared_v4_warning_shape() {
        let warning = CallWarning::from(Warning::UnsupportedSetting {
            setting: "topK".to_string(),
            details: Some("not supported".to_string()),
        });

        let value = serde_json::to_value(&warning).expect("serialize call warning");
        assert_eq!(value["type"], serde_json::json!("unsupported"));
        assert_eq!(value["feature"], serde_json::json!("topK"));
        assert_eq!(value["details"], serde_json::json!("not supported"));

        let legacy = serde_json::json!({
            "type": "unsupported-setting",
            "setting": "topK"
        });
        assert!(serde_json::from_value::<CallWarning>(legacy).is_err());
    }

    #[test]
    fn telemetry_options_match_ai_sdk_shape() {
        let options = TelemetryOptions::new()
            .with_is_enabled(true)
            .with_record_inputs(false)
            .with_record_outputs(true)
            .with_function_id("ai.generateText");

        let json = serde_json::to_value(&options).expect("serialize telemetry options");
        assert_eq!(json["isEnabled"], serde_json::json!(true));
        assert_eq!(json["recordInputs"], serde_json::json!(false));
        assert_eq!(json["recordOutputs"], serde_json::json!(true));
        assert_eq!(json["functionId"], serde_json::json!("ai.generateText"));
        assert!(json.get("integrations").is_none());

        let roundtrip: TelemetryOptions = serde_json::from_value(serde_json::json!({
            "is_enabled": false,
            "record_inputs": true,
            "record_outputs": false,
            "function_id": "ai.streamText",
            "integrations": [{ "ignored": true }]
        }))
        .expect("deserialize telemetry options");
        assert_eq!(roundtrip.is_enabled, Some(false));
        assert_eq!(roundtrip.record_inputs, Some(true));
        assert_eq!(roundtrip.record_outputs, Some(false));
        assert_eq!(roundtrip.function_id.as_deref(), Some("ai.streamText"));
    }

    #[test]
    fn language_model_response_metadata_requires_main_fields() {
        let metadata = ResponseMetadata {
            id: Some("resp_123".to_string()),
            model: Some("gpt-4o".to_string()),
            created: Some(
                DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                    .expect("valid timestamp")
                    .with_timezone(&Utc),
            ),
            provider: "openai".to_string(),
            request_id: None,
            headers: Some(HashMap::from([(
                "x-request-id".to_string(),
                "req_123".to_string(),
            )])),
            body: None,
        };

        let converted =
            LanguageModelResponseMetadata::try_from(&metadata).expect("convert response metadata");

        assert_eq!(converted.id, "resp_123");
        assert_eq!(converted.model_id, "gpt-4o");
        assert_eq!(
            converted
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-request-id"))
                .map(String::as_str),
            Some("req_123")
        );
    }

    #[test]
    fn language_model_v4_generate_response_metadata_preserves_body() {
        let metadata = ResponseMetadata {
            id: Some("resp_123".to_string()),
            model: Some("gpt-4o".to_string()),
            created: Some(
                DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                    .expect("valid timestamp")
                    .with_timezone(&Utc),
            ),
            provider: "openai".to_string(),
            request_id: None,
            headers: None,
            body: Some(serde_json::json!({ "choices": [{ "text": "hello" }] })),
        };

        let converted = LanguageModelV4GenerateResponseMetadata::from(&metadata);

        assert_eq!(converted.id.as_deref(), Some("resp_123"));
        assert_eq!(
            converted
                .body
                .as_ref()
                .and_then(|body| body["choices"][0]["text"].as_str()),
            Some("hello")
        );
    }

    #[test]
    fn image_and_transcription_response_metadata_require_model_id() {
        let response = HttpResponseInfo {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: Some("model-1".to_string()),
            headers: HashMap::from([("x-test".to_string(), "1".to_string())]),
            body: None,
        };

        let image =
            ImageModelResponseMetadata::try_from(&response).expect("convert image metadata");
        let transcription = TranscriptionModelResponseMetadata::try_from(&response)
            .expect("convert transcription metadata");

        assert_eq!(image.model_id, "model-1");
        assert_eq!(transcription.model_id, "model-1");
    }

    #[test]
    fn response_metadata_body_matrix_matches_ai_sdk_surfaces() {
        let response = HttpResponseInfo {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: Some("model-1".to_string()),
            headers: HashMap::from([("x-test".to_string(), "1".to_string())]),
            body: Some(serde_json::json!({ "raw": true })),
        };

        let rerank = RerankResponseMetadata::from(&response);
        let speech =
            SpeechModelResponseMetadata::try_from(&response).expect("convert speech metadata");
        let transcription = TranscriptionModelResponseMetadata::try_from(&response)
            .expect("convert transcription metadata");

        assert_eq!(rerank.body.as_ref(), response.body.as_ref());
        assert_eq!(speech.body.as_ref(), response.body.as_ref());
        assert_eq!(transcription.body.as_ref(), response.body.as_ref());

        let image =
            ImageModelResponseMetadata::try_from(&response).expect("convert image metadata");
        let video =
            VideoModelResponseMetadata::try_from(&response).expect("convert video metadata");

        let image_json = serde_json::to_value(&image).expect("serialize image metadata");
        let video_json = serde_json::to_value(&video).expect("serialize video metadata");

        assert!(image_json.get("body").is_none());
        assert!(video_json.get("body").is_none());
    }

    #[test]
    fn generate_object_events_and_stream_parts_match_ai_sdk_shape() {
        let response_metadata = LanguageModelResponseMetadata {
            id: "resp_1".to_string(),
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "gpt-4o-mini".to_string(),
            headers: None,
        };
        let response = GenerateObjectResponseMetadata::new(response_metadata.clone())
            .with_body(serde_json::json!({ "raw": true }));
        let request = LanguageModelRequestMetadata {
            body: Some(serde_json::json!({ "messages": [] })),
        };

        let mut provider_options = ProviderOptionsMap::new();
        provider_options.insert("openai", serde_json::json!({ "strictJsonSchema": true }));

        let start = GenerateObjectStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.generateObject".to_string(),
            provider: "openai".to_string(),
            model_id: "gpt-4o-mini".to_string(),
            system: Some(SystemPrompt::Text("return JSON".to_string())),
            prompt: Some(PromptInput::Text("extract".to_string())),
            messages: None,
            max_output_tokens: Some(128),
            temperature: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: Some(7),
            max_retries: 2,
            headers: Some(HashMap::from([(
                "x-test".to_string(),
                Some("1".to_string()),
            )])),
            provider_options: Some(provider_options),
            output: GenerateObjectOutputStrategy::NoSchema,
            schema: None,
            schema_name: None,
            schema_description: None,
        };
        let start_json = serde_json::to_value(&start).expect("serialize start event");
        assert_eq!(start_json["callId"], serde_json::json!("call_1"));
        assert_eq!(start_json["output"], serde_json::json!("no-schema"));
        assert_eq!(
            start_json["providerOptions"]["openai"]["strictJsonSchema"],
            serde_json::json!(true)
        );

        let step_end = GenerateObjectStepEndEvent {
            call_id: "call_1".to_string(),
            step_number: 0,
            provider: "openai".to_string(),
            model_id: "gpt-4o-mini".to_string(),
            finish_reason: FinishReason::Stop,
            usage: LanguageModelUsage::new(5, 7),
            object_text: "{\"answer\":42}".to_string(),
            reasoning: Some("parsed as JSON".to_string()),
            warnings: None,
            request: request.clone(),
            response: response.clone(),
            provider_metadata: None,
            ms_to_first_chunk: Some(12),
        };
        let step_end_json = serde_json::to_value(&step_end).expect("serialize step end event");
        assert_eq!(
            step_end_json["objectText"],
            serde_json::json!("{\"answer\":42}")
        );
        assert_eq!(
            step_end_json["response"]["body"]["raw"],
            serde_json::json!(true)
        );
        assert_eq!(step_end_json["msToFirstChunk"], serde_json::json!(12));

        let end: GenerateObjectEndEvent = GenerateObjectEndEvent {
            call_id: "call_1".to_string(),
            object: Some(serde_json::json!({ "answer": 42 })),
            error: None,
            reasoning: Some("parsed as JSON".to_string()),
            finish_reason: FinishReason::Stop,
            usage: LanguageModelUsage::new(5, 7),
            warnings: None,
            request,
            response,
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "responseId": "resp_1" }),
            )])),
        };
        let end_json = serde_json::to_value(&end).expect("serialize end event");
        assert_eq!(end_json["object"]["answer"], serde_json::json!(42));
        assert_eq!(
            end_json["providerMetadata"]["openai"]["responseId"],
            serde_json::json!("resp_1")
        );

        let stream_parts: Vec<ObjectStreamPart> = vec![
            ObjectStreamObjectPart::new(serde_json::json!({ "answer": 4 })).into(),
            ObjectStreamTextDeltaPart::new("{\"answer\"").into(),
            ObjectStreamErrorPart::new(serde_json::json!({ "message": "transient" })).into(),
            ObjectStreamFinishPart::new(
                FinishReason::Stop,
                LanguageModelUsage::new(5, 7),
                response_metadata,
            )
            .into(),
        ];
        let stream_json = serde_json::to_value(&stream_parts).expect("serialize object parts");
        assert_eq!(stream_json[0]["type"], serde_json::json!("object"));
        assert_eq!(stream_json[0]["object"]["answer"], serde_json::json!(4));
        assert_eq!(
            stream_json[1]["textDelta"],
            serde_json::json!("{\"answer\"")
        );
        assert_eq!(
            stream_json[2]["error"]["message"],
            serde_json::json!("transient")
        );
        assert_eq!(stream_json[3]["finishReason"], serde_json::json!("stop"));

        let roundtrip: Vec<ObjectStreamPart> =
            serde_json::from_value(stream_json).expect("deserialize object parts");
        assert_eq!(roundtrip[0].r#type(), "object");
        assert_eq!(roundtrip[1].r#type(), "text-delta");
        assert_eq!(roundtrip[2].r#type(), "error");
        assert_eq!(roundtrip[3].r#type(), "finish");
    }

    #[test]
    fn ui_message_chunks_match_ai_sdk_stream_shape() {
        assert_eq!(
            UI_MESSAGE_STREAM_HEADERS
                .iter()
                .find(|(key, _)| *key == "x-vercel-ai-ui-message-stream")
                .map(|(_, value)| *value),
            Some("v1")
        );

        let create_message = CreateUIMessage::new()
            .with_id("msg_new")
            .with_role(UiMessageRole::User)
            .with_metadata(serde_json::json!({ "draft": true }))
            .with_parts(vec![UiMessagePart::text("hello")]);
        let create_message_json =
            serde_json::to_value(&create_message).expect("serialize create UI message");
        assert_eq!(create_message_json["id"], serde_json::json!("msg_new"));
        assert_eq!(create_message_json["role"], serde_json::json!("user"));
        assert_eq!(
            create_message_json["metadata"]["draft"],
            serde_json::json!(true)
        );
        assert_eq!(
            create_message_json["parts"][0]["type"],
            serde_json::json!("text")
        );

        let _: UIMessage = UiMessage::assistant("msg_alias", vec![UiMessagePart::text("hello")]);
        let _: UIMessagePart = UiMessagePart::StepStart;
        let _: InferUIMessageMetadata = serde_json::json!({ "thread": "t1" });
        let _: InferUIMessageData =
            HashMap::from([("status".to_string(), serde_json::json!({ "ok": true }))]);
        let _: InferUIMessageTools = HashMap::new();
        let _: InferUIMessageToolOutputs = serde_json::json!({ "results": [] });
        let _: InferUIMessageToolCall =
            ToolCall::new("call_inferred", "search".to_string(), serde_json::json!({}));
        let _: InferUIMessagePart = UiMessagePart::text("inferred");
        let text_part: TextUIPart = TextUIPart::new("hello");
        let data_part: DataUIPart = DataUIPart::new("status", serde_json::json!({ "ok": true }));
        let tool_part: ToolUIPart = ToolUIPart::named(
            "search",
            "call_1",
            crate::types::chat::UiToolPartState::InputStreaming,
        );
        let _: DynamicToolUIPart = DynamicToolUIPart::dynamic(
            "dynamic-search",
            "call_2",
            crate::types::chat::UiToolPartState::InputStreaming,
        );
        let _: StepStartUIPart = UiMessagePart::StepStart;
        assert_eq!(
            serde_json::to_value(UiMessagePart::Text(text_part)).expect("serialize TextUIPart")["type"],
            serde_json::json!("text")
        );
        assert_eq!(
            serde_json::to_value(UiMessagePart::Data(data_part)).expect("serialize DataUIPart")["type"],
            serde_json::json!("data-status")
        );
        assert_eq!(
            serde_json::to_value(UiMessagePart::Tool(tool_part)).expect("serialize ToolUIPart")["type"],
            serde_json::json!("tool-search")
        );

        let ui_tool = UITool {
            input: serde_json::json!({ "query": "rust" }),
            output: Some(serde_json::json!({ "results": [] })),
        };
        let mut ui_tools = UITools::new();
        ui_tools.insert("search".to_string(), ui_tool.clone());
        let _: InferUITool = ui_tool;
        let _: InferUITools = ui_tools.clone();
        let ui_tools_json = serde_json::to_value(&ui_tools).expect("serialize UITools");
        assert_eq!(
            ui_tools_json["search"]["input"]["query"],
            serde_json::json!("rust")
        );

        let chat_request_options = ChatRequestOptions::new()
            .with_headers(HashMap::from([(
                "x-trace-id".to_string(),
                "trace_1".to_string(),
            )]))
            .with_body(serde_json::json!({ "sessionId": "sess_1" }))
            .with_metadata(serde_json::json!({ "source": "ui" }));
        let chat_request_json =
            serde_json::to_value(&chat_request_options).expect("serialize chat request options");
        assert_eq!(
            chat_request_json["headers"]["x-trace-id"],
            serde_json::json!("trace_1")
        );
        assert_eq!(
            chat_request_json["body"]["sessionId"],
            serde_json::json!("sess_1")
        );
        assert_eq!(
            chat_request_json["metadata"]["source"],
            serde_json::json!("ui")
        );
        assert_eq!(
            serde_json::to_value(ChatStatus::Streaming).expect("serialize chat status"),
            serde_json::json!("streaming")
        );

        let chat_state = ChatState::ready(vec![UiMessage::user(
            "msg_user",
            vec![UiMessagePart::text("hello")],
        )])
        .with_error(serde_json::json!({ "message": "network" }));
        let chat_state_json = serde_json::to_value(&chat_state).expect("serialize chat state");
        assert_eq!(chat_state_json["status"], serde_json::json!("error"));
        assert_eq!(
            chat_state_json["error"]["message"],
            serde_json::json!("network")
        );
        assert!(chat_state_json.get("pushMessage").is_none());

        let chat_init =
            ChatInit::new()
                .with_id("chat_1")
                .with_messages(vec![UiMessage::assistant(
                    "msg_assistant",
                    vec![UiMessagePart::text("ready")],
                )]);
        let chat_init_json = serde_json::to_value(&chat_init).expect("serialize chat init");
        assert_eq!(chat_init_json["id"], serde_json::json!("chat_1"));
        assert_eq!(
            chat_init_json["messages"][0]["role"],
            serde_json::json!("assistant")
        );
        assert!(chat_init_json.get("transport").is_none());
        assert!(chat_init_json.get("onFinish").is_none());

        let send_options = ChatTransportSendMessagesOptions {
            trigger: ChatTransportTrigger::SubmitMessage,
            chat_id: "chat_1".to_string(),
            message_id: None,
            messages: vec![UiMessage::user(
                "msg_user",
                vec![UiMessagePart::text("hello")],
            )],
            request_options: chat_request_options.clone(),
        };
        let send_options_json =
            serde_json::to_value(&send_options).expect("serialize send messages options");
        assert_eq!(
            send_options_json["trigger"],
            serde_json::json!("submit-message")
        );
        assert_eq!(send_options_json["chatId"], serde_json::json!("chat_1"));
        assert_eq!(
            send_options_json["headers"]["x-trace-id"],
            serde_json::json!("trace_1")
        );

        let http_transport_options = HttpChatTransportInitOptions::new()
            .with_api("/api/chat")
            .with_credentials(RequestCredentials::Include);
        let http_transport_json = serde_json::to_value(&http_transport_options)
            .expect("serialize HTTP chat transport options");
        assert_eq!(http_transport_json["api"], serde_json::json!("/api/chat"));
        assert_eq!(
            http_transport_json["credentials"],
            serde_json::json!("include")
        );
        assert!(http_transport_json.get("fetch").is_none());
        assert!(
            http_transport_json
                .get("prepareSendMessagesRequest")
                .is_none()
        );

        let prepare_send = PrepareSendMessagesRequestOptions {
            id: "chat_1".to_string(),
            messages: vec![UiMessage::user(
                "msg_user",
                vec![UiMessagePart::text("hello")],
            )],
            request_metadata: Some(serde_json::json!({ "source": "ui" })),
            body: Some(serde_json::json!({ "sessionId": "sess_1" })),
            credentials: Some(RequestCredentials::SameOrigin),
            headers: Some(HashMap::from([(
                "x-trace-id".to_string(),
                "trace_1".to_string(),
            )])),
            api: "/api/chat".to_string(),
            trigger: ChatTransportTrigger::RegenerateMessage,
            message_id: Some("msg_assistant".to_string()),
        };
        let prepare_send_json =
            serde_json::to_value(&prepare_send).expect("serialize prepare send options");
        assert_eq!(
            prepare_send_json["requestMetadata"]["source"],
            serde_json::json!("ui")
        );
        assert_eq!(
            prepare_send_json["trigger"],
            serde_json::json!("regenerate-message")
        );

        let prepared_send = PreparedSendMessagesRequest {
            body: serde_json::json!({ "id": "chat_1" }),
            headers: Some(HashMap::from([("x-prepared".to_string(), "1".to_string())])),
            credentials: Some(RequestCredentials::Omit),
            api: Some("/api/custom-chat".to_string()),
        };
        let prepared_send_json =
            serde_json::to_value(&prepared_send).expect("serialize prepared send request");
        assert_eq!(prepared_send_json["credentials"], serde_json::json!("omit"));
        assert_eq!(
            prepared_send_json["api"],
            serde_json::json!("/api/custom-chat")
        );

        let reconnect_options = ChatTransportReconnectToStreamOptions {
            chat_id: "chat_1".to_string(),
            request_options: ChatRequestOptions::new().with_metadata(serde_json::json!({
                "resume": true
            })),
        };
        let reconnect_json =
            serde_json::to_value(&reconnect_options).expect("serialize reconnect options");
        assert_eq!(
            reconnect_json["metadata"]["resume"],
            serde_json::json!(true)
        );

        let prepare_reconnect = PrepareReconnectToStreamRequestOptions {
            id: "chat_1".to_string(),
            request_metadata: Some(serde_json::json!({ "resume": true })),
            body: None,
            credentials: Some(RequestCredentials::SameOrigin),
            headers: None,
            api: "/api/chat".to_string(),
        };
        let prepare_reconnect_json =
            serde_json::to_value(&prepare_reconnect).expect("serialize prepare reconnect options");
        assert_eq!(
            prepare_reconnect_json["credentials"],
            serde_json::json!("same-origin")
        );
        assert_eq!(
            serde_json::to_value(PreparedReconnectToStreamRequest {
                api: Some("/api/chat/chat_1/stream".to_string()),
                ..PreparedReconnectToStreamRequest::default()
            })
            .expect("serialize prepared reconnect request")["api"],
            serde_json::json!("/api/chat/chat_1/stream")
        );

        let completion_request_options = CompletionRequestOptions::new()
            .with_headers(HashMap::from([("x-mode".to_string(), "test".to_string())]))
            .with_body(serde_json::json!({ "tenant": "acme" }));
        let completion_request_json = serde_json::to_value(&completion_request_options)
            .expect("serialize completion request options");
        assert_eq!(
            completion_request_json["headers"]["x-mode"],
            serde_json::json!("test")
        );
        assert_eq!(
            completion_request_json["body"]["tenant"],
            serde_json::json!("acme")
        );

        let use_completion_options = UseCompletionOptions::new()
            .with_api("/api/completion")
            .with_id("completion_1")
            .with_initial_input("question")
            .with_initial_completion("answer")
            .with_credentials(RequestCredentials::SameOrigin)
            .with_headers(HashMap::from([("x-ui".to_string(), "1".to_string())]))
            .with_body(serde_json::json!({ "sessionId": "sess_1" }))
            .with_stream_protocol(CompletionStreamProtocol::Text);
        let use_completion_json = serde_json::to_value(&use_completion_options)
            .expect("serialize use completion options");
        assert_eq!(
            use_completion_json["initialInput"],
            serde_json::json!("question")
        );
        assert_eq!(
            use_completion_json["initialCompletion"],
            serde_json::json!("answer")
        );
        assert_eq!(
            use_completion_json["credentials"],
            serde_json::json!("same-origin")
        );
        assert_eq!(
            use_completion_json["streamProtocol"],
            serde_json::json!("text")
        );
        assert!(use_completion_json.get("onFinish").is_none());
        assert!(use_completion_json.get("onError").is_none());
        assert!(use_completion_json.get("fetch").is_none());

        let options = UiMessageStreamOptions::new()
            .with_original_messages(vec![UiMessage::user(
                "msg_user",
                vec![UiMessagePart::text("hello")],
            )])
            .with_send_reasoning(false)
            .with_send_sources(true)
            .with_send_finish(false)
            .with_send_start(true);
        let options_json =
            serde_json::to_value(&options).expect("serialize UI message stream options");
        assert_eq!(
            options_json["originalMessages"][0]["id"],
            serde_json::json!("msg_user")
        );
        assert_eq!(options_json["sendReasoning"], serde_json::json!(false));
        assert_eq!(options_json["sendSources"], serde_json::json!(true));
        assert_eq!(options_json["sendFinish"], serde_json::json!(false));
        assert_eq!(options_json["sendStart"], serde_json::json!(true));
        assert!(options_json.get("generateMessageId").is_none());
        assert!(options_json.get("onFinish").is_none());
        assert!(options_json.get("messageMetadata").is_none());
        assert!(options_json.get("onError").is_none());

        let options_roundtrip: UIMessageStreamOptions =
            serde_json::from_value(options_json).expect("deserialize UI message stream options");
        assert_eq!(
            options_roundtrip
                .original_messages
                .as_ref()
                .expect("original messages")[0]
                .id,
            "msg_user"
        );
        let empty_options_json = serde_json::to_value(UiMessageStreamOptions::<UiMessage>::new())
            .expect("serialize empty UI message stream options");
        assert_eq!(empty_options_json, serde_json::json!({}));

        let mut start = UiMessageStartChunk::new();
        start.message_id = Some("msg_1".to_string());
        start.message_metadata = Some(serde_json::json!({ "turn": 1 }));

        let mut finish = UiMessageFinishChunk::new();
        finish.finish_reason = Some(FinishReason::Stop);
        finish.message_metadata = Some(serde_json::json!({ "done": true }));

        let mut data = UiMessageDataChunk::new("weather", serde_json::json!({ "city": "Paris" }));
        data.id = Some("data_1".to_string());
        data.transient = Some(true);

        let chunks: Vec<UiMessageChunk> = vec![
            UiMessageChunk::Start(start),
            UiMessageTextStartChunk::new("text_1").into(),
            UiMessageTextDeltaChunk::new("text_1", "hello").into(),
            UiMessageChunk::ToolInputAvailable(UiMessageToolInputAvailableChunk::new(
                "call_1",
                "weather",
                serde_json::json!({ "city": "Paris" }),
            )),
            UiMessageChunk::Data(data),
            UiMessageChunk::SourceDocument(UiMessageSourceDocumentChunk::new(
                "src_1",
                "text/plain",
                "Notes",
            )),
            UiMessageChunk::Finish(finish),
        ];

        let json = serde_json::to_value(&chunks).expect("serialize UI message chunks");
        assert_eq!(json[0]["type"], serde_json::json!("start"));
        assert_eq!(json[0]["messageId"], serde_json::json!("msg_1"));
        assert_eq!(json[2]["type"], serde_json::json!("text-delta"));
        assert_eq!(json[2]["delta"], serde_json::json!("hello"));
        assert_eq!(json[3]["toolCallId"], serde_json::json!("call_1"));
        assert_eq!(json[4]["type"], serde_json::json!("data-weather"));
        assert_eq!(json[4]["transient"], serde_json::json!(true));
        assert_eq!(json[5]["sourceId"], serde_json::json!("src_1"));
        assert_eq!(json[6]["finishReason"], serde_json::json!("stop"));

        let roundtrip: Vec<UiMessageChunk> =
            serde_json::from_value(json).expect("deserialize UI message chunks");
        assert_eq!(roundtrip[0].r#type(), "start");
        assert_eq!(roundtrip[2].r#type(), "text-delta");
        assert_eq!(roundtrip[4].r#type(), "data-weather");
        assert_eq!(roundtrip[6].r#type(), "finish");

        let data_chunk: DataUIMessageChunk =
            UiMessageDataChunk::new("status", serde_json::json!({ "ok": true }));
        let data_chunk_union: UIMessageChunk = data_chunk.into();
        let _: InferUIMessageChunk = data_chunk_union.clone();
        assert!(is_data_ui_message_chunk(&data_chunk_union));
        let text_delta_chunk: UIMessageChunk =
            UiMessageTextDeltaChunk::new("text_1", "hello").into();
        assert!(!is_data_ui_message_chunk(&text_delta_chunk));

        let invalid_data_chunk = serde_json::json!({
            "type": "invalid-data",
            "data": { "city": "Paris" }
        });
        assert!(serde_json::from_value::<UiMessageDataChunk>(invalid_data_chunk.clone()).is_err());
        assert!(serde_json::from_value::<UiMessageChunk>(invalid_data_chunk).is_err());

        let missing_type_chunk = serde_json::json!({ "messageId": "msg_1" });
        assert!(serde_json::from_value::<UiMessageStartChunk>(missing_type_chunk.clone()).is_err());
        assert!(serde_json::from_value::<UiMessageChunk>(missing_type_chunk).is_err());
        assert!(serde_json::from_value::<UiMessageChunk>(serde_json::json!({})).is_err());
    }

    #[test]
    fn ui_message_helper_functions_match_ai_sdk_semantics() {
        use crate::types::chat::{UiToolPart, UiToolPartState};

        let text = UiMessagePart::Text(TextUIPart::new("hello"));
        let custom = UiMessagePart::Custom(CustomContentUIPart::new("openai.redacted"));
        let file = UiMessagePart::File(FileUIPart::new(
            "https://example.com/file.txt",
            "text/plain",
        ));
        let reasoning_file = UiMessagePart::ReasoningFile(ReasoningFileUIPart::new(
            "https://example.com/reasoning.txt",
            "text/plain",
        ));
        let reasoning = UiMessagePart::Reasoning(ReasoningUIPart::new("thinking"));
        let data =
            UiMessagePart::Data(DataUIPart::new("status", serde_json::json!({ "ok": true })));
        let static_tool = UiMessagePart::Tool(UiToolPart::named(
            "search",
            "call_static",
            UiToolPartState::OutputAvailable,
        ));
        let dynamic_tool = UiMessagePart::Tool(UiToolPart::dynamic(
            "code-runner",
            "call_dynamic",
            UiToolPartState::OutputError,
        ));

        assert!(is_text_ui_part(&text));
        assert!(is_custom_content_ui_part(&custom));
        assert!(is_file_ui_part(&file));
        assert!(is_reasoning_file_ui_part(&reasoning_file));
        assert!(is_reasoning_ui_part(&reasoning));
        assert!(is_data_ui_part(&data));
        assert!(is_static_tool_ui_part(&static_tool));
        assert!(!is_static_tool_ui_part(&dynamic_tool));
        assert!(is_dynamic_tool_ui_part(&dynamic_tool));
        assert!(!is_dynamic_tool_ui_part(&static_tool));
        assert!(is_tool_ui_part(&static_tool));
        assert!(is_tool_ui_part(&dynamic_tool));
        assert!(!is_tool_ui_part(&text));
        assert_eq!(get_static_tool_name(&static_tool), Some("search"));
        assert_eq!(get_static_tool_name(&dynamic_tool), None);
        assert_eq!(get_tool_name(&static_tool), Some("search"));
        assert_eq!(get_tool_name(&dynamic_tool), Some("code-runner"));
        assert_eq!(
            get_tool_or_dynamic_tool_name(&dynamic_tool),
            Some("code-runner")
        );
        assert_eq!(get_tool_name(&text), None);

        let complete_tool_call = UiMessage::assistant(
            "msg_complete_tools",
            vec![
                UiMessagePart::Tool(UiToolPart::named(
                    "previous",
                    "call_previous",
                    UiToolPartState::InputAvailable,
                )),
                UiMessagePart::StepStart,
                UiMessagePart::Tool(UiToolPart::named(
                    "search",
                    "call_1",
                    UiToolPartState::OutputAvailable,
                )),
                UiMessagePart::Tool(UiToolPart::dynamic(
                    "code-runner",
                    "call_2",
                    UiToolPartState::OutputError,
                )),
            ],
        );
        assert!(last_assistant_message_is_complete_with_tool_calls(&[
            complete_tool_call
        ]));

        let mut provider_executed = UiToolPart::named(
            "provider-search",
            "call_provider",
            UiToolPartState::OutputAvailable,
        );
        provider_executed.provider_executed = Some(true);
        let only_provider_executed = UiMessage::assistant(
            "msg_provider_tools",
            vec![UiMessagePart::Tool(provider_executed)],
        );
        assert!(!last_assistant_message_is_complete_with_tool_calls(&[
            only_provider_executed,
        ]));

        let incomplete_tool_call = UiMessage::assistant(
            "msg_incomplete_tools",
            vec![UiMessagePart::Tool(UiToolPart::named(
                "search",
                "call_pending",
                UiToolPartState::InputAvailable,
            ))],
        );
        assert!(!last_assistant_message_is_complete_with_tool_calls(&[
            incomplete_tool_call
        ]));

        let approval_responded = UiMessage::assistant(
            "msg_approvals",
            vec![
                UiMessagePart::Tool(UiToolPart::named(
                    "approval",
                    "call_approval",
                    UiToolPartState::ApprovalResponded,
                )),
                UiMessagePart::Tool(UiToolPart::named(
                    "search",
                    "call_result",
                    UiToolPartState::OutputAvailable,
                )),
            ],
        );
        assert!(last_assistant_message_is_complete_with_approval_responses(
            &[approval_responded,]
        ));

        let pending_approval = UiMessage::assistant(
            "msg_pending_approval",
            vec![
                UiMessagePart::Tool(UiToolPart::named(
                    "approval",
                    "call_approval",
                    UiToolPartState::ApprovalResponded,
                )),
                UiMessagePart::Tool(UiToolPart::named(
                    "search",
                    "call_pending",
                    UiToolPartState::InputAvailable,
                )),
            ],
        );
        assert!(!last_assistant_message_is_complete_with_approval_responses(
            &[pending_approval,]
        ));

        assert!(!last_assistant_message_is_complete_with_tool_calls(&[
            UiMessage::user("msg_user", vec![UiMessagePart::text("hello")])
        ]));
        assert!(!last_assistant_message_is_complete_with_approval_responses(
            &[]
        ));
    }

    #[test]
    fn image_speech_and_transcription_results_match_ai_sdk_shape() {
        let image_file = GeneratedFile::from_bytes(b"image", "image/png");
        let image_response = ImageModelResponseMetadata {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "image-model".to_string(),
            headers: None,
        };
        let image_result = GenerateImageResult::new(image_file.clone(), vec![image_file])
            .with_responses(vec![image_response])
            .with_provider_metadata(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "images": [{ "id": "img_1" }] }),
            )]))
            .with_usage(ImageModelUsage::new(Some(1), Some(2), Some(3)));
        let image_json = serde_json::to_value(&image_result).expect("serialize image result");

        assert_eq!(
            image_json["image"]["mediaType"],
            serde_json::json!("image/png")
        );
        assert_eq!(
            image_json["images"][0]["base64"],
            serde_json::json!("aW1hZ2U=")
        );
        assert_eq!(image_json["usage"]["totalTokens"], serde_json::json!(3));
        assert_eq!(
            image_json["providerMetadata"]["openai"]["images"][0]["id"],
            serde_json::json!("img_1")
        );
        let _: Experimental_GenerateImageResult =
            serde_json::from_value(image_json).expect("deserialize image result");

        let video_file = GeneratedFile::from_bytes(b"video", "video/mp4");
        let video_response = VideoModelResponseMetadata {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:30Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "video-model".to_string(),
            headers: None,
            provider_metadata: Some(HashMap::from([(
                "xai".to_string(),
                serde_json::json!({ "requestId": "vid_1" }),
            )])),
        };
        let video_result = GenerateVideoResult::new(video_file.clone(), vec![video_file])
            .with_responses(vec![video_response])
            .with_provider_metadata(HashMap::from([(
                "xai".to_string(),
                serde_json::json!({ "videos": [{ "id": "vid_1" }] }),
            )]));
        let video_json = serde_json::to_value(&video_result).expect("serialize video result");

        assert_eq!(
            video_json["video"]["mediaType"],
            serde_json::json!("video/mp4")
        );
        assert_eq!(
            video_json["videos"][0]["base64"],
            serde_json::json!("dmlkZW8=")
        );
        assert_eq!(
            video_json["responses"][0]["providerMetadata"]["xai"]["requestId"],
            serde_json::json!("vid_1")
        );
        assert_eq!(
            video_json["providerMetadata"]["xai"]["videos"][0]["id"],
            serde_json::json!("vid_1")
        );
        let _: GenerateVideoResult =
            serde_json::from_value(video_json).expect("deserialize video result");

        let typed_file = DefaultGeneratedFileWithType::from_bytes(b"file", "text/plain");
        let typed_file_json =
            serde_json::to_value(&typed_file).expect("serialize generated file with type");
        assert_eq!(typed_file_json["type"], serde_json::json!("file"));
        assert_eq!(typed_file_json["base64"], serde_json::json!("ZmlsZQ=="));
        assert_eq!(
            typed_file_json["mediaType"],
            serde_json::json!("text/plain")
        );
        let _: DefaultGeneratedFileWithType =
            serde_json::from_value(typed_file_json).expect("deserialize generated file with type");

        let audio = GeneratedAudioFile::from_bytes(b"audio", "audio/mpeg");
        assert_eq!(audio.format, "mp3");
        assert_eq!(audio.base64(), "YXVkaW8=");
        let default_audio: DefaultGeneratedAudioFile = audio.clone();
        assert_eq!(default_audio.media_type(), "audio/mpeg");
        let typed_audio = DefaultGeneratedAudioFileWithType::new(default_audio);
        let typed_audio_json =
            serde_json::to_value(&typed_audio).expect("serialize generated audio with type");
        assert_eq!(typed_audio_json["type"], serde_json::json!("audio"));
        assert_eq!(typed_audio_json["format"], serde_json::json!("mp3"));
        assert_eq!(
            typed_audio_json["mediaType"],
            serde_json::json!("audio/mpeg")
        );
        let _: DefaultGeneratedAudioFileWithType = serde_json::from_value(typed_audio_json)
            .expect("deserialize generated audio with type");

        let speech_response = SpeechModelResponseMetadata {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:01:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "speech-model".to_string(),
            headers: None,
            body: Some(serde_json::json!({ "ok": true })),
        };
        let speech_result = SpeechResult::new(audio)
            .with_responses(vec![speech_response])
            .with_provider_metadata(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "voice": "alloy" }),
            )]));
        let speech_json = serde_json::to_value(&speech_result).expect("serialize speech result");

        assert_eq!(speech_json["audio"]["format"], serde_json::json!("mp3"));
        assert_eq!(
            speech_json["audio"]["mediaType"],
            serde_json::json!("audio/mpeg")
        );
        assert_eq!(
            speech_json["providerMetadata"]["openai"]["voice"],
            serde_json::json!("alloy")
        );
        let _: Experimental_SpeechResult =
            serde_json::from_value(speech_json).expect("deserialize speech result");

        let transcription_response = TranscriptionModelResponseMetadata {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:02:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "stt-model".to_string(),
            headers: None,
            body: Some(serde_json::json!({ "text": "hello" })),
        };
        let transcription_result = TranscriptionResult::new("hello")
            .with_segments(vec![TranscriptionSegment::new("hello", 0.0, 0.5)])
            .with_language("en")
            .with_duration_in_seconds(0.5)
            .with_responses(vec![transcription_response])
            .with_provider_metadata(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "transcriptId": "tr_1" }),
            )]));
        let transcription_json =
            serde_json::to_value(&transcription_result).expect("serialize transcription result");

        assert_eq!(transcription_json["text"], serde_json::json!("hello"));
        assert_eq!(
            transcription_json["segments"][0]["startSecond"],
            serde_json::json!(0.0)
        );
        assert_eq!(
            transcription_json["durationInSeconds"],
            serde_json::json!(0.5)
        );
        assert_eq!(
            transcription_json["providerMetadata"]["openai"]["transcriptId"],
            serde_json::json!("tr_1")
        );
        let _: Experimental_TranscriptionResult =
            serde_json::from_value(transcription_json).expect("deserialize transcription result");
    }

    #[test]
    fn request_timeout_helpers_work_without_runtime_abort_handle() {
        let timeout = TimeoutConfiguration::settings(
            TimeoutConfigurationSettings::new()
                .with_total_ms(3_000)
                .with_step_ms(1_000)
                .with_chunk_ms(250)
                .with_tool_ms(900)
                .with_tool_timeout_ms("weather", 1200),
        );

        let request = RequestOptions::new()
            .with_max_retries(2)
            .with_abort_signal(())
            .with_header("x-test", "1")
            .without_header("x-drop")
            .with_timeout(timeout.clone());

        assert_eq!(timeout.total_timeout_ms(), Some(3_000));
        assert_eq!(timeout.step_timeout_ms(), Some(1_000));
        assert_eq!(timeout.chunk_timeout_ms(), Some(250));
        assert_eq!(timeout.tool_timeout_ms("weather"), Some(1_200));
        assert_eq!(timeout.tool_timeout_ms("other"), Some(900));
        assert_eq!(request.max_attempts(), Some(3));
        assert_eq!(
            request.effective_headers(),
            HashMap::from([("x-test".to_string(), "1".to_string())])
        );
        assert_eq!(request.total_timeout(), Some(Duration::from_millis(3_000)));
        assert_eq!(request.step_timeout(), Some(Duration::from_millis(1_000)));
        assert_eq!(request.chunk_timeout(), Some(Duration::from_millis(250)));
        assert_eq!(request.tool_timeout_ms("weather"), Some(1_200));
        assert!(request.abort_signal.is_some());
    }

    #[test]
    fn embedding_result_and_event_payloads_match_ai_sdk_shape() {
        let response_data = ModelCallResponseData::new()
            .with_headers(HashMap::from([(
                "x-request-id".to_string(),
                "req_1".to_string(),
            )]))
            .with_body(serde_json::json!({ "ok": true }));
        let provider_metadata = HashMap::from([(
            "openai".to_string(),
            serde_json::json!({ "embeddingId": "emb_1" }),
        )]);

        let result = EmbedResult::new("hello", vec![1.0, 2.0], EmbeddingModelUsage::new(2))
            .with_provider_metadata(provider_metadata.clone())
            .with_response(response_data.clone());
        let result_json = serde_json::to_value(&result).expect("serialize embed result");

        assert_eq!(result_json["value"], serde_json::json!("hello"));
        assert_eq!(result_json["embedding"][0], serde_json::json!(1.0));
        assert_eq!(result_json["usage"]["tokens"], serde_json::json!(2));
        assert_eq!(
            result_json["providerMetadata"]["openai"]["embeddingId"],
            serde_json::json!("emb_1")
        );
        assert_eq!(
            result_json["response"]["headers"]["x-request-id"],
            serde_json::json!("req_1")
        );
        let _: EmbedResult = serde_json::from_value(result_json).expect("deserialize embed result");

        let many = EmbedManyResult::new(
            vec!["a".to_string(), "b".to_string()],
            vec![vec![1.0], vec![2.0]],
            EmbeddingModelUsage::new(4),
        )
        .with_provider_metadata(provider_metadata.clone())
        .with_responses(vec![Some(response_data.clone()), None]);
        let many_json = serde_json::to_value(&many).expect("serialize embedMany result");

        assert_eq!(many_json["values"][1], serde_json::json!("b"));
        assert_eq!(many_json["embeddings"][1][0], serde_json::json!(2.0));
        assert!(many_json["responses"][1].is_null());
        let _: EmbedManyResult =
            serde_json::from_value(many_json).expect("deserialize embedMany result");

        let start = EmbedStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.embedMany".to_string(),
            provider: "openai".to_string(),
            model_id: "text-embedding-3-small".to_string(),
            value: EmbedValue::from(vec!["a".to_string(), "b".to_string()]),
            max_retries: 2,
            headers: Some(HashMap::from([(
                "x-test".to_string(),
                Some("1".to_string()),
            )])),
            provider_options: None,
        };
        let start_json = serde_json::to_value(&start).expect("serialize embed start event");
        assert_eq!(start_json["callId"], serde_json::json!("call_1"));
        assert_eq!(start_json["value"][0], serde_json::json!("a"));

        let end = EmbedEndEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.embedMany".to_string(),
            provider: "openai".to_string(),
            model_id: "text-embedding-3-small".to_string(),
            value: EmbedValue::from(vec!["a".to_string(), "b".to_string()]),
            embedding: EmbedOutput::from(vec![vec![1.0], vec![2.0]]),
            usage: EmbeddingModelUsage::new(4),
            warnings: Vec::new(),
            provider_metadata: Some(provider_metadata),
            response: Some(EmbedResponseData::from(vec![Some(response_data), None])),
        };
        let end_json = serde_json::to_value(&end).expect("serialize embed end event");
        assert_eq!(end_json["embedding"][1][0], serde_json::json!(2.0));
        assert!(end_json["response"][1].is_null());

        let model_call_start = EmbeddingModelCallStartEvent {
            call_id: "call_1".to_string(),
            embed_call_id: "embed_1".to_string(),
            operation_id: "ai.embedMany.doEmbed".to_string(),
            provider: "openai".to_string(),
            model_id: "text-embedding-3-small".to_string(),
            values: vec!["a".to_string()],
        };
        let model_call_end = EmbeddingModelCallEndEvent {
            call_id: "call_1".to_string(),
            embed_call_id: "embed_1".to_string(),
            operation_id: "ai.embedMany.doEmbed".to_string(),
            provider: "openai".to_string(),
            model_id: "text-embedding-3-small".to_string(),
            values: vec!["a".to_string()],
            embeddings: vec![vec![1.0]],
            usage: EmbeddingModelUsage::new(1),
        };
        let model_start_json =
            serde_json::to_value(&model_call_start).expect("serialize embedding model start");
        let model_end_json =
            serde_json::to_value(&model_call_end).expect("serialize embedding model end");
        assert_eq!(
            model_start_json["embedCallId"],
            serde_json::json!("embed_1")
        );
        assert_eq!(model_end_json["embeddings"][0][0], serde_json::json!(1.0));
    }

    #[test]
    fn rerank_result_and_event_payloads_match_ai_sdk_shape() {
        let timestamp = DateTime::parse_from_rfc3339("2026-04-21T09:03:00Z")
            .expect("valid timestamp")
            .with_timezone(&Utc);
        let response = RerankResponseMetadata::new(timestamp, "rerank-model")
            .with_id("rr_1")
            .with_headers(HashMap::from([(
                "x-request-id".to_string(),
                "req_1".to_string(),
            )]))
            .with_body(serde_json::json!({ "ok": true }));
        let original_documents = vec!["apple".to_string(), "banana".to_string()];
        let ranking = vec![
            RerankRanking::new(1, 0.9, "banana".to_string()),
            RerankRanking::new(0, 0.7, "apple".to_string()),
        ];
        let result = RerankResult::new(
            original_documents.clone(),
            ranking.clone(),
            response.clone(),
        )
        .with_provider_metadata(HashMap::from([(
            "cohere".to_string(),
            serde_json::json!({ "searchUnits": 1 }),
        )]));
        let result_json = serde_json::to_value(&result).expect("serialize rerank result");

        assert_eq!(
            result_json["originalDocuments"][0],
            serde_json::json!("apple")
        );
        assert_eq!(
            result_json["rerankedDocuments"][0],
            serde_json::json!("banana")
        );
        assert_eq!(
            result_json["ranking"][0]["originalIndex"],
            serde_json::json!(1)
        );
        assert_eq!(result_json["ranking"][0]["score"], serde_json::json!(0.9));
        assert_eq!(result_json["response"]["id"], serde_json::json!("rr_1"));
        assert_eq!(
            result_json["providerMetadata"]["cohere"]["searchUnits"],
            serde_json::json!(1)
        );
        let _: RerankResult<String> =
            serde_json::from_value(result_json).expect("deserialize rerank result");

        let documents = vec![serde_json::json!("apple"), serde_json::json!("banana")];
        let start = RerankStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.rerank".to_string(),
            provider: "cohere".to_string(),
            model_id: "rerank-model".to_string(),
            documents: documents.clone(),
            query: "fruit".to_string(),
            top_n: Some(2),
            max_retries: 2,
            headers: None,
            provider_options: None,
        };
        let end = RerankEndEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.rerank".to_string(),
            provider: "cohere".to_string(),
            model_id: "rerank-model".to_string(),
            documents: documents.clone(),
            query: "fruit".to_string(),
            ranking: vec![
                RerankRanking::new(1, 0.9, serde_json::json!("banana")),
                RerankRanking::new(0, 0.7, serde_json::json!("apple")),
            ],
            warnings: Vec::new(),
            provider_metadata: None,
            response: response.clone(),
        };
        let start_json = serde_json::to_value(&start).expect("serialize rerank start event");
        let end_json = serde_json::to_value(&end).expect("serialize rerank end event");
        assert_eq!(start_json["documents"][0], serde_json::json!("apple"));
        assert_eq!(start_json["topN"], serde_json::json!(2));
        assert_eq!(
            end_json["ranking"][0]["document"],
            serde_json::json!("banana")
        );

        let model_call_start = RerankingModelCallStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.rerank.doRerank".to_string(),
            provider: "cohere".to_string(),
            model_id: "rerank-model".to_string(),
            documents,
            documents_type: "text".to_string(),
            query: "fruit".to_string(),
            top_n: Some(2),
        };
        let model_call_end = RerankingModelCallEndEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.rerank.doRerank".to_string(),
            provider: "cohere".to_string(),
            model_id: "rerank-model".to_string(),
            documents_type: "text".to_string(),
            ranking: vec![RerankingModelCallRanking::new(1, 0.9)],
        };
        let model_start_json =
            serde_json::to_value(&model_call_start).expect("serialize rerank model start");
        let model_end_json =
            serde_json::to_value(&model_call_end).expect("serialize rerank model end");
        assert_eq!(model_start_json["documentsType"], serde_json::json!("text"));
        assert_eq!(
            model_end_json["ranking"][0]["relevanceScore"],
            serde_json::json!(0.9)
        );
    }

    #[test]
    fn embedding_and_image_usage_shared_shapes_are_available() {
        let embedding = EmbeddingModelUsage::from(EmbeddingUsage::new(12, 12));
        let mut image = ImageModelUsage::new(Some(3), Some(5), Some(8));
        image.merge(&ImageModelUsage::new(Some(2), None, Some(2)));
        let added_image =
            add_image_model_usage(&image, &ImageModelUsage::new(None, Some(1), Some(1)));
        let call_options = LanguageModelCallOptions::from(&super::super::CommonParams {
            model: "gpt-5".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(128),
            max_completion_tokens: Some(256),
            top_p: Some(0.9),
            top_k: Some(40.0),
            stop_sequences: Some(vec!["END".to_string()]),
            seed: Some(7),
            frequency_penalty: Some(0.2),
            presence_penalty: Some(0.1),
        });
        let mut language = LanguageModelUsage::from(
            Usage::builder()
                .with_input_tokens(super::super::UsageInputTokens {
                    total: Some(10),
                    no_cache: Some(7),
                    cache_read: Some(2),
                    cache_write: Some(1),
                })
                .with_output_tokens(super::super::UsageOutputTokens {
                    total: Some(6),
                    text: Some(4),
                    reasoning: Some(2),
                })
                .with_raw_usage_value(serde_json::json!({
                    "provider_total_tokens": 16
                }))
                .build(),
        );
        language.merge(&LanguageModelUsage {
            input_tokens: Some(2),
            input_token_details: LanguageModelInputTokenDetails {
                no_cache_tokens: Some(1),
                cache_read_tokens: Some(1),
                cache_write_tokens: None,
            },
            output_tokens: Some(1),
            output_token_details: LanguageModelOutputTokenDetails {
                text_tokens: Some(1),
                reasoning_tokens: None,
            },
            total_tokens: Some(3),
            reasoning_tokens: None,
            cached_input_tokens: Some(1),
            raw: Some(serde_json::Map::new()),
        });
        let added_language = add_language_model_usage(
            &create_null_language_model_usage(),
            &LanguageModelUsage {
                input_tokens: Some(4),
                input_token_details: LanguageModelInputTokenDetails::default(),
                output_tokens: Some(3),
                output_token_details: LanguageModelOutputTokenDetails::default(),
                total_tokens: Some(7),
                reasoning_tokens: None,
                cached_input_tokens: None,
                raw: Some(serde_json::Map::new()),
            },
        );
        let projected_language = as_language_model_usage(
            &Usage::builder()
                .with_input_tokens(super::super::UsageInputTokens {
                    total: Some(3),
                    no_cache: Some(2),
                    cache_read: Some(1),
                    cache_write: None,
                })
                .with_output_tokens(super::super::UsageOutputTokens {
                    total: Some(5),
                    text: Some(4),
                    reasoning: Some(1),
                })
                .build(),
        );

        assert_eq!(embedding.tokens, 12);
        assert_eq!(image.input_tokens, Some(5));
        assert_eq!(image.output_tokens, Some(5));
        assert_eq!(image.total_tokens, Some(10));
        assert_eq!(added_image.input_tokens, Some(5));
        assert_eq!(added_image.output_tokens, Some(6));
        assert_eq!(added_image.total_tokens, Some(11));
        assert_eq!(call_options.max_output_tokens, Some(256));
        assert_eq!(call_options.temperature, Some(0.7));
        assert_eq!(call_options.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(call_options.reasoning, None);
        assert_eq!(language.input_tokens, Some(12));
        assert_eq!(language.output_tokens, Some(7));
        assert_eq!(language.total_tokens, Some(19));
        assert_eq!(language.input_token_details.no_cache_tokens, Some(8));
        assert_eq!(language.cached_input_tokens, Some(3));
        assert_eq!(language.output_token_details.reasoning_tokens, Some(2));
        assert_eq!(language.raw, None);
        assert_eq!(added_language.input_tokens, Some(4));
        assert_eq!(added_language.output_tokens, Some(3));
        assert_eq!(added_language.total_tokens, Some(7));
        assert_eq!(added_language.raw, None);
        assert_eq!(projected_language.input_tokens, Some(3));
        assert_eq!(projected_language.output_tokens, Some(5));
        assert_eq!(projected_language.total_tokens, Some(8));
        assert_eq!(projected_language.cached_input_tokens, Some(1));
        assert_eq!(
            projected_language.output_token_details.reasoning_tokens,
            Some(1)
        );
    }

    #[test]
    fn video_model_response_metadata_attaches_non_empty_provider_metadata() {
        let metadata = VideoModelResponseMetadata::try_from(&HttpResponseInfo {
            timestamp: Utc::now(),
            model_id: Some("video-model".to_string()),
            headers: HashMap::from([("x-video".to_string(), "1".to_string())]),
            body: None,
        })
        .expect("valid video response metadata")
        .with_provider_metadata(HashMap::from([(
            "fake-video".to_string(),
            serde_json::json!({ "taskId": "task-1" }),
        )]));

        assert_eq!(metadata.model_id, "video-model");
        assert_eq!(
            metadata
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-video")),
            Some(&"1".to_string())
        );
        assert_eq!(
            metadata
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("fake-video"))
                .and_then(|value| value.get("taskId"))
                .and_then(serde_json::Value::as_str),
            Some("task-1")
        );
    }

    #[test]
    fn timeout_helper_functions_follow_ai_sdk_semantics() {
        let timeout = TimeoutConfiguration::settings(
            TimeoutConfigurationSettings::new()
                .with_total_ms(1_000)
                .with_step_ms(200)
                .with_chunk_ms(50)
                .with_tool_ms(300)
                .with_tool_timeout_ms("search", 450),
        );

        assert_eq!(get_total_timeout_ms(Some(&timeout)), Some(1_000));
        assert_eq!(get_step_timeout_ms(Some(&timeout)), Some(200));
        assert_eq!(get_chunk_timeout_ms(Some(&timeout)), Some(50));
        assert_eq!(get_tool_timeout_ms(Some(&timeout), "search"), Some(450));
        assert_eq!(get_tool_timeout_ms(Some(&timeout), "other"), Some(300));
        assert_eq!(get_total_timeout_ms(None), None);
    }

    #[test]
    fn provider_utils_style_tool_and_context_types_are_available() {
        let mut context = Context::new();
        context.insert("tenant".to_string(), serde_json::json!("acme"));

        let mut provider_options = ProviderOptions::default();
        provider_options.insert("openai", serde_json::json!({ "serviceTier": "flex" }));

        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "itemId": "item_1" }),
        )]);

        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
        )
        .with_provider_executed(true)
        .with_dynamic(true)
        .with_invalid(true)
        .with_error(serde_json::json!({ "message": "unknown tool" }))
        .with_title("Search")
        .with_provider_metadata(provider_metadata.clone());

        let tool_result = ToolResult::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
            ToolResultOutput::json(serde_json::json!({ "ok": true })),
        )
        .with_provider_executed(true)
        .with_dynamic(true)
        .with_preliminary(true)
        .with_title("Search result")
        .with_provider_metadata(provider_metadata);

        assert_eq!(context.get("tenant"), Some(&serde_json::json!("acme")));
        assert_eq!(
            provider_options
                .get("openai")
                .and_then(|value| value.get("serviceTier"))
                .and_then(serde_json::Value::as_str),
            Some("flex")
        );
        assert_eq!(tool_call.tool_call_id, "call_1");
        assert_eq!(tool_call.provider_executed, Some(true));
        assert_eq!(tool_call.invalid, Some(true));
        assert_eq!(tool_result.tool_call_id, "call_1");
        assert_eq!(tool_result.dynamic, Some(true));

        let tool_call_json = serde_json::to_value(&tool_call).expect("serialize tool call");
        assert_eq!(tool_call.r#type(), "tool-call");
        assert_eq!(tool_call_json["type"], serde_json::json!("tool-call"));
        assert_eq!(
            tool_call_json["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("item_1")
        );
        assert_eq!(tool_call_json["title"], serde_json::json!("Search"));
        assert_eq!(tool_call_json["invalid"], serde_json::json!(true));
        assert_eq!(
            tool_call_json["error"],
            serde_json::json!({ "message": "unknown tool" })
        );

        let tool_result_json = serde_json::to_value(&tool_result).expect("serialize tool result");
        assert_eq!(tool_result.r#type(), "tool-result");
        assert_eq!(tool_result_json["type"], serde_json::json!("tool-result"));
        assert_eq!(
            tool_result_json["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("item_1")
        );
        assert_eq!(
            tool_result_json["title"],
            serde_json::json!("Search result")
        );
        assert_eq!(tool_result_json["preliminary"], serde_json::json!(true));
    }

    #[test]
    fn ai_sdk_error_index_passive_shapes_match_exported_errors() {
        let base_error = AISDKError::new("AI_TestError", "boom", Some(serde_json::json!("cause")));
        assert_eq!(
            serde_json::to_value(&base_error).expect("serialize base error"),
            serde_json::json!({
                "name": "AI_TestError",
                "message": "boom",
                "cause": "cause"
            })
        );

        let api_error = APICallError::new(
            "rate limited",
            "https://api.example.test",
            serde_json::json!({ "model": "test" }),
            Some(429),
        );
        let api_error_json = serde_json::to_value(&api_error).expect("serialize api error");
        assert_eq!(api_error_json["statusCode"], serde_json::json!(429));
        assert_eq!(api_error_json["isRetryable"], serde_json::json!(true));

        assert_eq!(
            serde_json::to_value(EmptyResponseBodyError::new())
                .expect("serialize empty body error"),
            serde_json::json!({ "message": "Empty response body" })
        );

        assert_eq!(
            serde_json::to_value(InvalidPromptError::new(
                serde_json::json!({ "role": "bad" }),
                "unsupported role",
                None,
            ))
            .expect("serialize invalid prompt")["message"],
            serde_json::json!("Invalid prompt: unsupported role")
        );

        assert_eq!(
            serde_json::to_value(InvalidResponseDataError::new(
                serde_json::json!({ "ok": false }),
            ))
            .expect("serialize invalid response data")["message"],
            serde_json::json!(r#"Invalid response data: {"ok":false}."#)
        );

        assert_eq!(
            serde_json::to_value(JSONParseError::new(
                "not json",
                Some(serde_json::json!("SyntaxError")),
            ))
            .expect("serialize json parse error")["message"],
            serde_json::json!("JSON parsing failed: Text: not json.\nError message: SyntaxError")
        );

        assert_eq!(
            serde_json::to_value(LoadAPIKeyError::new("missing key"))
                .expect("serialize load api key error")["message"],
            serde_json::json!("missing key")
        );
        assert_eq!(
            serde_json::to_value(LoadSettingError::new("missing setting"))
                .expect("serialize load setting error")["message"],
            serde_json::json!("missing setting")
        );

        let download_status =
            DownloadError::from_status("https://example.com/video.mp4", 404, "Not Found");
        let download_status_json =
            serde_json::to_value(&download_status).expect("serialize download status error");
        assert_eq!(
            download_status_json,
            serde_json::json!({
                "message": "Failed to download https://example.com/video.mp4: 404 Not Found",
                "url": "https://example.com/video.mp4",
                "statusCode": 404,
                "statusText": "Not Found"
            })
        );

        let download_cause =
            DownloadError::from_cause("https://example.com/video.mp4", serde_json::json!("boom"));
        let download_cause_json =
            serde_json::to_value(&download_cause).expect("serialize download cause error");
        assert_eq!(
            download_cause_json["message"],
            serde_json::json!("Failed to download https://example.com/video.mp4: boom")
        );
        assert_eq!(download_cause_json["cause"], serde_json::json!("boom"));

        assert_eq!(
            serde_json::to_value(NoContentGeneratedError::new())
                .expect("serialize no content error")["message"],
            serde_json::json!("No content generated.")
        );

        let no_such_model = NoSuchModelError::new("gpt-test", NoSuchModelType::LanguageModel);
        let no_such_model_json =
            serde_json::to_value(&no_such_model).expect("serialize no such model error");
        assert_eq!(
            no_such_model_json["modelType"],
            serde_json::json!("languageModel")
        );
        assert_eq!(
            no_such_model_json["message"],
            serde_json::json!("No such languageModel: gpt-test")
        );

        let no_such_provider = NoSuchProviderError::new(
            "missing:gpt-test",
            NoSuchModelType::LanguageModel,
            "missing",
            vec!["openai".to_string(), "anthropic".to_string()],
        );
        let no_such_provider_json =
            serde_json::to_value(&no_such_provider).expect("serialize no such provider error");
        assert_eq!(
            no_such_provider_json["providerId"],
            serde_json::json!("missing")
        );
        assert_eq!(
            no_such_provider_json["availableProviders"],
            serde_json::json!(["openai", "anthropic"])
        );
        assert_eq!(
            no_such_provider_json["message"],
            serde_json::json!("No such provider: missing (available providers: openai,anthropic)")
        );

        let no_provider_reference = NoSuchProviderReferenceError::new(
            "anthropic",
            HashMap::from([("openai".to_string(), "file_1".to_string())]),
        );
        assert_eq!(
            serde_json::to_value(&no_provider_reference)
                .expect("serialize no provider reference error")["message"],
            serde_json::json!(
                "No provider reference found for provider 'anthropic'. Available providers: openai"
            )
        );

        let too_many_embeddings = TooManyEmbeddingValuesForCallError::new(
            "openai",
            "text-embedding-test",
            1,
            vec![serde_json::json!("a"), serde_json::json!("b")],
        );
        assert_eq!(
            serde_json::to_value(&too_many_embeddings)
                .expect("serialize too many embeddings error")["message"],
            serde_json::json!(
                "Too many values for a single embedding call. The openai model \"text-embedding-test\" can only embed up to 1 values per call, but 2 values were provided."
            )
        );

        let type_validation = TypeValidationError::new(
            serde_json::json!({ "x": 1 }),
            Some(serde_json::json!("bad type")),
            Some(TypeValidationContext {
                field: Some("message.parts[0]".to_string()),
                entity_name: Some("tool".to_string()),
                entity_id: Some("call_1".to_string()),
            }),
        );
        assert_eq!(
            serde_json::to_value(&type_validation).expect("serialize type validation error")["message"],
            serde_json::json!(
                "Type validation failed for message.parts[0] (tool, id: \"call_1\"): Value: {\"x\":1}.\nError message: bad type"
            )
        );

        assert_eq!(
            serde_json::to_value(UnsupportedFunctionalityError::new("vision"))
                .expect("serialize unsupported functionality")["message"],
            serde_json::json!("'vision' functionality not supported.")
        );

        let invalid_argument =
            InvalidArgumentError::new("temperature", serde_json::json!(2), "must be <= 1");
        assert_eq!(
            serde_json::to_value(&invalid_argument).expect("serialize invalid argument"),
            serde_json::json!({
                "message": "Invalid argument for parameter temperature: must be <= 1",
                "parameter": "temperature",
                "value": 2
            })
        );

        let invalid_stream_part = InvalidStreamPartError::new(
            LanguageModelStreamPart::<String, JSONValue, ToolResultOutput>::TextDelta(
                TextStreamTextDeltaPart::new("txt_1", "hello"),
            ),
            "Unexpected chunk",
        );
        let invalid_stream_part_json =
            serde_json::to_value(&invalid_stream_part).expect("serialize invalid stream part");
        assert_eq!(
            invalid_stream_part_json["chunk"]["type"],
            serde_json::json!("text-delta")
        );
        assert_eq!(
            invalid_stream_part_json["message"],
            serde_json::json!("Unexpected chunk")
        );

        assert_eq!(
            serde_json::to_value(InvalidToolApprovalError::new("approval_1"))
                .expect("serialize invalid tool approval")["message"],
            serde_json::json!(
                "Tool approval response references unknown approvalId: \"approval_1\". \
                 No matching tool-approval-request found in message history."
            )
        );

        assert_eq!(
            serde_json::to_value(ToolCallNotFoundForApprovalError::new(
                "call_1",
                "approval_1"
            ))
            .expect("serialize missing approval tool call")["message"],
            serde_json::json!(
                "Tool call \"call_1\" not found for approval request \"approval_1\"."
            )
        );

        let unsupported = UnsupportedModelVersionError::new("v1", "test", "model");
        assert_eq!(
            serde_json::to_value(&unsupported).expect("serialize unsupported model version"),
            serde_json::json!({
                "message": "Unsupported model version v1 for provider \"test\" and model \"model\". AI SDK 5 only supports models that implement specification version \"v2\".",
                "version": "v1",
                "provider": "test",
                "modelId": "model"
            })
        );

        let ui_stream_error = UIMessageStreamError::new("text-delta", "txt_1", "Missing start");
        assert_eq!(
            serde_json::to_value(&ui_stream_error).expect("serialize UI stream error"),
            serde_json::json!({
                "message": "Missing start",
                "chunkType": "text-delta",
                "chunkId": "txt_1"
            })
        );

        let invalid_role = InvalidMessageRoleError::new("developer");
        assert_eq!(
            serde_json::to_value(&invalid_role).expect("serialize invalid role")["message"],
            serde_json::json!(
                "Invalid message role: 'developer'. Must be one of: \"system\", \"user\", \"assistant\", \"tool\"."
            )
        );

        let message_conversion = MessageConversionError::new(
            UiMessageWithoutId::new(UiMessageRole::Assistant, vec![UiMessagePart::text("hello")]),
            "Cannot convert assistant message",
        );
        let message_conversion_json =
            serde_json::to_value(&message_conversion).expect("serialize message conversion error");
        assert_eq!(
            message_conversion_json["originalMessage"]["role"],
            serde_json::json!("assistant")
        );
        assert_eq!(
            message_conversion_json["originalMessage"]["parts"][0]["type"],
            serde_json::json!("text")
        );

        let retry = RetryError::new(
            "Retries exhausted",
            RetryErrorReason::MaxRetriesExceeded,
            vec![
                serde_json::json!("first"),
                serde_json::json!({ "message": "last" }),
            ],
        );
        let retry_json = serde_json::to_value(&retry).expect("serialize retry error");
        assert_eq!(
            retry_json["reason"],
            serde_json::json!("maxRetriesExceeded")
        );
        assert_eq!(
            retry_json["lastError"]["message"],
            serde_json::json!("last")
        );

        assert_eq!(
            serde_json::to_value(NoImageGeneratedError::new(None, None))
                .expect("serialize no image error"),
            serde_json::json!({ "message": "No image generated." })
        );
        assert_eq!(
            serde_json::to_value(NoObjectGeneratedError::new(
                Some("{}".to_string()),
                None,
                Some(LanguageModelUsage::new(1, 2)),
                Some(FinishReason::Stop),
                None,
            ))
            .expect("serialize no object error")["finishReason"],
            serde_json::json!("stop")
        );
        assert_eq!(
            serde_json::to_value(NoOutputGeneratedError::new(Some(serde_json::json!(
                "empty response"
            ),)))
            .expect("serialize no output error")["cause"],
            serde_json::json!("empty response")
        );
        assert_eq!(
            serde_json::to_value(NoSpeechGeneratedError::new(Vec::new()))
                .expect("serialize no speech error")["message"],
            serde_json::json!("No speech audio generated.")
        );
        assert_eq!(
            serde_json::to_value(NoTranscriptGeneratedError::new(Vec::new()))
                .expect("serialize no transcript error")["message"],
            serde_json::json!("No transcript generated.")
        );
        assert_eq!(
            serde_json::to_value(NoVideoGeneratedError::new(Vec::new(), None))
                .expect("serialize no video error")["message"],
            serde_json::json!("No video generated.")
        );
    }

    #[test]
    fn generate_text_basic_content_outputs_match_ai_sdk_shape() {
        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "itemId": "msg_1" }),
        )]);

        let text = TextOutput::new("hello").with_provider_metadata(provider_metadata.clone());
        assert_eq!(text.r#type(), "text");
        assert_eq!(
            serde_json::to_value(&text).expect("serialize text output"),
            serde_json::json!({
                "type": "text",
                "text": "hello",
                "providerMetadata": {
                    "openai": { "itemId": "msg_1" }
                }
            })
        );

        let custom = CustomOutput::new("openai.compaction")
            .with_provider_metadata(provider_metadata.clone());
        assert_eq!(custom.r#type(), "custom");
        assert_eq!(
            serde_json::to_value(&custom).expect("serialize custom output"),
            serde_json::json!({
                "type": "custom",
                "kind": "openai.compaction",
                "providerMetadata": {
                    "openai": { "itemId": "msg_1" }
                }
            })
        );

        let file = FileOutput::new(GeneratedFile::from_bytes(b"hello", "text/plain"))
            .with_provider_metadata(provider_metadata);
        assert_eq!(file.r#type(), "file");
        assert_eq!(
            serde_json::to_value(&file).expect("serialize file output"),
            serde_json::json!({
                "type": "file",
                "file": {
                    "base64": "aGVsbG8=",
                    "mediaType": "text/plain"
                },
                "providerMetadata": {
                    "openai": { "itemId": "msg_1" }
                }
            })
        );

        let wrong_type = serde_json::json!({
            "type": "image",
            "file": {
                "base64": "aGVsbG8=",
                "mediaType": "text/plain"
            }
        });
        assert!(serde_json::from_value::<FileOutput>(wrong_type).is_err());
    }

    #[test]
    fn generate_text_content_part_union_roundtrips_ai_sdk_shape() {
        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
        )
        .with_dynamic(true);
        let parts: Vec<GenerateTextContentPart> = vec![
            TextOutput::new("hello").into(),
            CustomOutput::new("openai.compaction").into(),
            ReasoningOutput::new("thinking").into(),
            ReasoningFileOutput::new(GeneratedFile::from_bytes(b"trace", "text/plain")).into(),
            Source::url("source_1", "https://example.com").into(),
            FileOutput::new(GeneratedFile::from_bytes(b"hello", "text/plain")).into(),
            tool_call.clone().into(),
            ToolResult::new(
                "call_1",
                "search".to_string(),
                serde_json::json!({ "q": "rust" }),
                ToolResultOutput::json(serde_json::json!({ "ok": true })),
            )
            .into(),
            ToolError::new(
                "call_2",
                "fetch".to_string(),
                serde_json::json!({ "url": "https://example.com" }),
                serde_json::json!({ "message": "timeout" }),
            )
            .into(),
            ToolApprovalRequestOutput::new("approval_1", tool_call.clone()).into(),
            ToolApprovalResponseOutput::new("approval_1", tool_call, true).into(),
        ];

        let part_types: Vec<&'static str> =
            parts.iter().map(GenerateTextContentPart::r#type).collect();
        assert_eq!(
            part_types,
            vec![
                "text",
                "custom",
                "reasoning",
                "reasoning-file",
                "source",
                "file",
                "tool-call",
                "tool-result",
                "tool-error",
                "tool-approval-request",
                "tool-approval-response"
            ]
        );

        let json = serde_json::to_value(&parts).expect("serialize generate text content parts");
        assert_eq!(json[0]["type"], serde_json::json!("text"));
        assert_eq!(
            json[5]["file"]["mediaType"],
            serde_json::json!("text/plain")
        );
        assert_eq!(json[6]["type"], serde_json::json!("tool-call"));
        assert_eq!(json[7]["type"], serde_json::json!("tool-result"));
        assert_eq!(json[9]["toolCall"]["type"], serde_json::json!("tool-call"));

        let roundtrip: Vec<GenerateTextContentPart> =
            serde_json::from_value(json).expect("deserialize generate text content parts");
        let roundtrip_types: Vec<&'static str> = roundtrip
            .iter()
            .map(GenerateTextContentPart::r#type)
            .collect();
        assert_eq!(roundtrip_types, part_types);
    }

    #[test]
    fn generate_text_result_envelope_matches_ai_sdk_shape() {
        let timestamp = DateTime::parse_from_rfc3339("2026-04-24T00:00:00Z")
            .expect("timestamp")
            .with_timezone(&Utc);
        let response_metadata = LanguageModelResponseMetadata {
            id: "resp_1".to_string(),
            timestamp,
            model_id: "gpt-test".to_string(),
            headers: None,
        };
        let response = GenerateTextResponseMetadata::new(response_metadata)
            .with_messages(vec![
                AssistantModelMessage::new(AssistantContent::text("hello")).into(),
            ])
            .with_body(serde_json::json!({ "id": "resp_1" }));
        let request = LanguageModelRequestMetadata {
            body: Some(serde_json::json!({ "messages": [] })),
        };
        let usage = LanguageModelUsage {
            input_tokens: Some(3),
            output_tokens: Some(2),
            total_tokens: Some(5),
            ..LanguageModelUsage::default()
        };
        let source = Source::url("source_1", "https://example.com");
        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
        );
        let tool_result = ToolResult::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
            ToolResultOutput::json(serde_json::json!({ "ok": true })),
        );
        let reasoning_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "reasoningId": "rs_1" }),
        )]);
        let content: Vec<GenerateTextContentPart> = vec![
            TextOutput::new("hello").into(),
            source.clone().into(),
            tool_call.clone().into(),
            tool_result.clone().into(),
        ];
        let reasoning_output =
            ReasoningOutput::new("because").with_provider_metadata(reasoning_metadata.clone());
        let reasoning = vec![reasoning_output.clone().into()];
        let step_reasoning = vec![
            GenerateTextStepReasoningPart::from(reasoning_output),
            GenerateTextStepReasoningPart::from(
                ReasoningFileOutput::new(GeneratedFile::from_base64("dHJhY2U=", "text/plain"))
                    .with_provider_metadata(reasoning_metadata),
            ),
        ];
        let files = vec![GeneratedFile::from_bytes(b"hello", "text/plain")];
        let model = GenerateTextModelInfo::new("openai", "gpt-test");
        let step = GenerateTextStepResult {
            call_id: "call_root".to_string(),
            step_number: 0,
            model,
            tools_context: Context::new(),
            runtime_context: Context::new(),
            content: content.clone(),
            text: "hello".to_string(),
            reasoning: step_reasoning,
            reasoning_text: Some("because".to_string()),
            files: files.clone(),
            sources: vec![source.clone()],
            tool_calls: vec![tool_call.clone()],
            static_tool_calls: vec![tool_call.clone()],
            dynamic_tool_calls: Vec::new(),
            tool_results: vec![tool_result.clone()],
            static_tool_results: vec![tool_result.clone()],
            dynamic_tool_results: Vec::new(),
            finish_reason: FinishReason::Stop,
            raw_finish_reason: Some("stop".to_string()),
            usage: usage.clone(),
            warnings: None,
            request: request.clone(),
            response: response.clone(),
            provider_metadata: None,
        };
        let result = GenerateTextResult {
            content,
            text: "hello".to_string(),
            reasoning,
            reasoning_text: Some("because".to_string()),
            files,
            sources: vec![source],
            tool_calls: vec![tool_call.clone()],
            static_tool_calls: vec![tool_call],
            dynamic_tool_calls: Vec::new(),
            tool_results: vec![tool_result.clone()],
            static_tool_results: vec![tool_result],
            dynamic_tool_results: Vec::new(),
            finish_reason: FinishReason::Stop,
            raw_finish_reason: Some("stop".to_string()),
            usage: usage.clone(),
            total_usage: usage,
            warnings: None,
            request,
            response,
            provider_metadata: None,
            steps: vec![step],
            output: serde_json::json!("hello"),
        };

        let json = serde_json::to_value(&result).expect("serialize generate text result");
        assert_eq!(json["content"][0]["type"], serde_json::json!("text"));
        assert_eq!(json["reasoning"][0]["type"], serde_json::json!("reasoning"));
        assert_eq!(
            json["reasoning"][0]["providerMetadata"]["openai"]["reasoningId"],
            serde_json::json!("rs_1")
        );
        assert_eq!(
            json["files"][0]["mediaType"],
            serde_json::json!("text/plain")
        );
        assert_eq!(json["finishReason"], serde_json::json!("stop"));
        assert_eq!(json["rawFinishReason"], serde_json::json!("stop"));
        assert_eq!(json["totalUsage"]["totalTokens"], serde_json::json!(5));
        assert_eq!(
            json["response"]["messages"][0]["role"],
            serde_json::json!("assistant")
        );
        assert_eq!(json["steps"][0]["callId"], serde_json::json!("call_root"));
        assert_eq!(
            json["steps"][0]["model"]["modelId"],
            serde_json::json!("gpt-test")
        );
        assert_eq!(
            json["steps"][0]["reasoning"][0]["providerOptions"]["openai"]["reasoningId"],
            serde_json::json!("rs_1")
        );
        assert_eq!(
            json["steps"][0]["reasoning"][1],
            serde_json::json!({
                "type": "reasoning-file",
                "data": "dHJhY2U=",
                "mediaType": "text/plain",
                "providerOptions": {
                    "openai": { "reasoningId": "rs_1" }
                }
            })
        );

        let roundtrip: GenerateTextResult =
            serde_json::from_value(json).expect("deserialize generate text result");
        assert_eq!(roundtrip.text, "hello");
        assert_eq!(roundtrip.steps.len(), 1);
        assert_eq!(roundtrip.content[2].r#type(), "tool-call");
        assert_eq!(roundtrip.steps[0].reasoning[1].r#type(), "reasoning-file");
    }

    #[test]
    fn text_stream_parts_match_ai_sdk_stream_text_result_shape() {
        let timestamp = DateTime::parse_from_rfc3339("2026-04-24T00:00:00Z")
            .expect("timestamp")
            .with_timezone(&Utc);
        let response = LanguageModelResponseMetadata {
            id: "resp_1".to_string(),
            timestamp,
            model_id: "gpt-test".to_string(),
            headers: None,
        };
        let request = LanguageModelRequestMetadata {
            body: Some(serde_json::json!({ "messages": [] })),
        };
        let usage = LanguageModelUsage {
            input_tokens: Some(3),
            output_tokens: Some(2),
            total_tokens: Some(5),
            ..LanguageModelUsage::default()
        };
        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "responseId": "resp_1" }),
        )]);
        let mut finish_step =
            TextStreamFinishStepPart::new(response, usage.clone(), FinishReason::Stop);
        finish_step.raw_finish_reason = Some("stop".to_string());
        finish_step.provider_metadata = Some(provider_metadata);
        let mut finish = TextStreamFinishPart::new(FinishReason::Stop, usage);
        finish.raw_finish_reason = Some("stop".to_string());

        let parts: Vec<TextStreamPart> = vec![
            TextStreamStartPart::new().into(),
            TextStreamTextStartPart::new("text_1").into(),
            TextStreamTextDeltaPart::new("text_1", "hello").into(),
            TextStreamReasoningDeltaPart::new("reasoning_1", "because").into(),
            TextStreamToolInputStartPart::new("call_1", "search").into(),
            ToolCall::new(
                "call_1",
                "search".to_string(),
                serde_json::json!({ "q": "rust" }),
            )
            .into(),
            TextStreamFilePart::new(GeneratedFile::from_bytes(b"hello", "text/plain")).into(),
            TextStreamStartStepPart::new(request, Vec::new()).into(),
            finish_step.into(),
            finish.into(),
            TextStreamAbortPart::new().with_reason("user").into(),
            TextStreamRawPart::new(serde_json::json!({ "chunk": 1 })).into(),
        ];

        let json = serde_json::to_value(&parts).expect("serialize text stream parts");
        assert_eq!(json[0]["type"], serde_json::json!("start"));
        assert_eq!(json[2]["type"], serde_json::json!("text-delta"));
        assert_eq!(json[2]["text"], serde_json::json!("hello"));
        assert!(json[2]["delta"].is_null());
        assert_eq!(json[3]["type"], serde_json::json!("reasoning-delta"));
        assert_eq!(json[3]["text"], serde_json::json!("because"));
        assert_eq!(json[5]["type"], serde_json::json!("tool-call"));
        assert_eq!(
            json[6]["file"]["mediaType"],
            serde_json::json!("text/plain")
        );
        assert_eq!(json[8]["type"], serde_json::json!("finish-step"));
        assert_eq!(json[8]["finishReason"], serde_json::json!("stop"));
        assert_eq!(json[8]["rawFinishReason"], serde_json::json!("stop"));
        assert_eq!(
            json[8]["providerMetadata"]["openai"]["responseId"],
            serde_json::json!("resp_1")
        );
        assert_eq!(json[9]["type"], serde_json::json!("finish"));
        assert_eq!(json[9]["totalUsage"]["totalTokens"], serde_json::json!(5));
        assert!(json[9]["usage"].is_null());
        assert_eq!(json[10]["reason"], serde_json::json!("user"));
        assert_eq!(json[11]["rawValue"]["chunk"], serde_json::json!(1));

        let roundtrip: Vec<TextStreamPart> =
            serde_json::from_value(json).expect("deserialize text stream parts");
        assert_eq!(roundtrip[2].r#type(), "text-delta");
        assert_eq!(roundtrip[8].r#type(), "finish-step");
        assert_eq!(roundtrip[11].r#type(), "raw");
    }

    #[test]
    fn language_model_stream_parts_match_ai_sdk_model_call_shape() {
        let timestamp = DateTime::parse_from_rfc3339("2026-04-24T00:00:00Z")
            .expect("timestamp")
            .with_timezone(&Utc);
        let usage = LanguageModelUsage {
            input_tokens: Some(7),
            output_tokens: Some(5),
            total_tokens: Some(12),
            ..LanguageModelUsage::default()
        };
        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "responseId": "resp_1" }),
        )]);
        let mut end = LanguageModelStreamModelCallEndPart::new(FinishReason::Stop, usage);
        end.raw_finish_reason = Some("stop".to_string());
        end.provider_metadata = Some(provider_metadata);

        let parts: Vec<LanguageModelStreamPart> = vec![
            LanguageModelStreamModelCallStartPart::new(Vec::new()).into(),
            LanguageModelStreamModelCallResponseMetadataPart::new()
                .with_id("resp_1")
                .with_timestamp(timestamp)
                .with_model_id("gpt-test")
                .into(),
            TextStreamTextStartPart::new("text_1").into(),
            TextStreamTextDeltaPart::new("text_1", "hello").into(),
            TextStreamReasoningDeltaPart::new("reasoning_1", "because").into(),
            TextStreamToolInputStartPart::new("call_1", "search").into(),
            ToolCall::new(
                "call_1",
                "search".to_string(),
                serde_json::json!({ "query": "rust" }),
            )
            .into(),
            ToolResult::new(
                "call_1",
                "search".to_string(),
                serde_json::json!({ "query": "rust" }),
                ToolResultOutput::json(serde_json::json!({ "answer": "ok" })),
            )
            .into(),
            TextStreamErrorPart::new(serde_json::json!({ "message": "recoverable" })).into(),
            TextStreamRawPart::new(serde_json::json!({ "chunk": 1 })).into(),
            end.into(),
        ];

        let json = serde_json::to_value(&parts).expect("serialize language model stream parts");
        assert_eq!(json[0]["type"], serde_json::json!("model-call-start"));
        assert_eq!(
            json[1]["type"],
            serde_json::json!("model-call-response-metadata")
        );
        assert_eq!(json[1]["id"], serde_json::json!("resp_1"));
        assert_eq!(json[1]["modelId"], serde_json::json!("gpt-test"));
        assert_eq!(json[3]["type"], serde_json::json!("text-delta"));
        assert_eq!(json[6]["type"], serde_json::json!("tool-call"));
        assert_eq!(json[7]["type"], serde_json::json!("tool-result"));
        assert_eq!(json[10]["type"], serde_json::json!("model-call-end"));
        assert_eq!(json[10]["finishReason"], serde_json::json!("stop"));
        assert_eq!(json[10]["rawFinishReason"], serde_json::json!("stop"));
        assert_eq!(json[10]["usage"]["totalTokens"], serde_json::json!(12));
        assert_eq!(
            json[10]["providerMetadata"]["openai"]["responseId"],
            serde_json::json!("resp_1")
        );

        let roundtrip: Vec<LanguageModelStreamPart> =
            serde_json::from_value(json).expect("deserialize language model stream parts");
        assert_eq!(roundtrip[0].r#type(), "model-call-start");
        assert_eq!(roundtrip[1].r#type(), "model-call-response-metadata");
        assert_eq!(roundtrip[10].r#type(), "model-call-end");
    }

    #[test]
    fn generate_text_callback_start_events_match_ai_sdk_shape() {
        let mut provider_options = ProviderOptionsMap::new();
        provider_options.insert("openai", serde_json::json!({ "reasoningEffort": "low" }));
        let mut tools_context = Context::new();
        tools_context.insert("tenant".to_string(), serde_json::json!("docs"));
        let mut runtime_context = Context::new();
        runtime_context.insert("traceId".to_string(), serde_json::json!("trace_1"));
        let prompt = StandardizedPrompt {
            system: None,
            messages: vec![ModelMessage::Assistant(AssistantModelMessage::new(
                AssistantContent::text("ready"),
            ))],
        };

        let start_event = GenerateTextStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.generateText".to_string(),
            model: CallbackModelInfo::new("openai", "gpt-test"),
            tools: Some(vec![Tool::function(
                "search",
                "Search docs",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    }
                }),
            )]),
            tool_choice: Some(ToolChoice::tool("search")),
            active_tools: Some(vec!["search".to_string()]),
            max_retries: 2,
            timeout: Some(TimeoutConfiguration::settings(
                TimeoutConfigurationSettings::new().with_total_ms(30_000),
            )),
            headers: Some(HashMap::from([
                ("x-test".to_string(), Some("1".to_string())),
                ("x-drop".to_string(), None),
            ])),
            provider_options: Some(provider_options.clone()),
            stop_when: vec![StopCondition::is_step_count(2)],
            output: Some(serde_json::json!({ "type": "object" })),
            tools_context: tools_context.clone(),
            runtime_context: runtime_context.clone(),
            call_options: LanguageModelCallOptions {
                temperature: Some(0.2),
                ..LanguageModelCallOptions::default()
            },
            prompt: prompt.clone(),
        };

        let json = serde_json::to_value(&start_event).expect("serialize start event");

        assert_eq!(json["callId"], serde_json::json!("call_1"));
        assert_eq!(json["operationId"], serde_json::json!("ai.generateText"));
        assert_eq!(json["provider"], serde_json::json!("openai"));
        assert_eq!(json["modelId"], serde_json::json!("gpt-test"));
        assert_eq!(json["toolChoice"]["toolName"], serde_json::json!("search"));
        assert_eq!(json["activeTools"][0], serde_json::json!("search"));
        assert_eq!(
            json["providerOptions"]["openai"]["reasoningEffort"],
            serde_json::json!("low")
        );
        assert_eq!(json["toolsContext"]["tenant"], serde_json::json!("docs"));
        assert_eq!(
            json["runtimeContext"]["traceId"],
            serde_json::json!("trace_1")
        );
        assert_eq!(json["maxRetries"], serde_json::json!(2));
        assert_eq!(json["timeout"]["totalMs"], serde_json::json!(30_000));
        assert_eq!(
            json["stopWhen"][0],
            serde_json::json!({ "type": "step-count", "stepCount": 2 })
        );
        assert_eq!(json["headers"]["x-test"], serde_json::json!("1"));
        assert!(json["headers"]["x-drop"].is_null());
        assert_eq!(json["messages"][0]["role"], serde_json::json!("assistant"));
        assert_eq!(json["temperature"], serde_json::json!(0.2));

        let step_event = GenerateTextStepStartEvent {
            call_id: "call_1".to_string(),
            model: CallbackModelInfo::new("openai", "gpt-test"),
            step_number: 1,
            tools: None,
            tool_choice: Some(ToolChoice::Auto),
            active_tools: Some(vec!["search".to_string()]),
            steps: Vec::<GenerateTextStepResult>::new(),
            provider_options: Some(provider_options),
            output: Some(serde_json::json!({ "type": "object" })),
            runtime_context,
            tools_context,
            prompt,
        };
        let step_json = serde_json::to_value(&step_event).expect("serialize step start event");

        assert_eq!(step_json["callId"], serde_json::json!("call_1"));
        assert_eq!(step_json["stepNumber"], serde_json::json!(1));
        assert_eq!(step_json["toolChoice"], serde_json::json!("auto"));
        assert_eq!(step_json["activeTools"][0], serde_json::json!("search"));
        assert!(step_json["steps"].as_array().is_some_and(Vec::is_empty));
    }

    #[test]
    #[allow(deprecated)]
    fn stop_condition_helpers_match_ai_sdk_builtin_semantics() {
        fn step_with_tool_calls(
            step_number: u32,
            tool_calls: Vec<ToolCall<String, JSONValue>>,
        ) -> GenerateTextStepResult {
            let timestamp = DateTime::parse_from_rfc3339("2026-04-24T00:00:00Z")
                .expect("timestamp")
                .with_timezone(&Utc);
            let response = GenerateTextResponseMetadata::new(LanguageModelResponseMetadata {
                id: format!("resp_{step_number}"),
                timestamp,
                model_id: "gpt-test".to_string(),
                headers: None,
            });

            GenerateTextStepResult {
                call_id: "call_1".to_string(),
                step_number,
                model: GenerateTextModelInfo::new("openai", "gpt-test"),
                tools_context: Context::new(),
                runtime_context: Context::new(),
                content: Vec::new(),
                text: String::new(),
                reasoning: Vec::new(),
                reasoning_text: None,
                files: Vec::new(),
                sources: Vec::new(),
                tool_calls: tool_calls.clone(),
                static_tool_calls: tool_calls,
                dynamic_tool_calls: Vec::new(),
                tool_results: Vec::new(),
                static_tool_results: Vec::new(),
                dynamic_tool_results: Vec::new(),
                finish_reason: FinishReason::Stop,
                raw_finish_reason: None,
                usage: LanguageModelUsage::default(),
                warnings: None,
                request: LanguageModelRequestMetadata { body: None },
                response,
                provider_metadata: None,
            }
        }

        let steps = vec![
            step_with_tool_calls(0, Vec::new()),
            step_with_tool_calls(
                1,
                vec![ToolCall::new(
                    "call_final",
                    "finalAnswer".to_string(),
                    serde_json::json!({}),
                )],
            ),
        ];

        assert!(StopCondition::is_step_count(2).is_met(&steps));
        assert!(step_count_is(2).is_met(&steps));
        assert!(!StopCondition::is_step_count(1).is_met(&steps));
        assert!(!StopCondition::is_loop_finished().is_met(&steps));
        assert!(StopCondition::has_tool_call(["search", "finalAnswer"]).is_met(&steps));
        assert!(!StopCondition::has_tool_call(["search"]).is_met(&steps));
        assert!(!StopCondition::custom(serde_json::json!({ "name": "custom" })).is_met(&steps));
        assert!(is_stop_condition_met(
            &[is_loop_finished(), has_tool_call(["finalAnswer"])],
            &steps
        ));
    }

    #[test]
    fn filter_active_tools_matches_ai_sdk_helper_semantics() {
        let tools = vec![
            Tool::function(
                "search",
                "Search docs",
                serde_json::json!({ "type": "object" }),
            ),
            Tool::function(
                "weather",
                "Get weather",
                serde_json::json!({ "type": "object" }),
            ),
            Tool::provider_defined("openai.web_search", "webSearch"),
        ];

        let all_tools = filter_active_tools::<String>(Some(&tools), None)
            .expect("tools should be returned when active tools are absent");
        assert_eq!(all_tools.len(), 3);

        let filtered = filter_active_tools(Some(&tools), Some(&["weather", "webSearch"]))
            .expect("filtered tools should be returned");
        let filtered_names: Vec<&str> = filtered
            .iter()
            .map(|tool| match tool {
                Tool::Function { function } => function.name.as_str(),
                Tool::ProviderDefined(tool) => tool.name.as_str(),
            })
            .collect();
        assert_eq!(filtered_names, vec!["weather", "webSearch"]);

        let experimental_filtered =
            experimental_filter_active_tools(Some(&tools), Some(&["search"]))
                .expect("experimental alias should return filtered tools");
        let experimental_names: Vec<&str> = experimental_filtered
            .iter()
            .map(|tool| match tool {
                Tool::Function { function } => function.name.as_str(),
                Tool::ProviderDefined(tool) => tool.name.as_str(),
            })
            .collect();
        assert_eq!(experimental_names, vec!["search"]);

        assert!(filter_active_tools::<String>(None, Some(&[])).is_none());
    }

    #[test]
    fn prune_messages_matches_ai_sdk_reasoning_and_tool_pruning() {
        let messages = vec![
            ModelMessage::Assistant(AssistantModelMessage::new(AssistantContent::parts(vec![
                AssistantContentPart::Text(TextPart::new("hello")),
                AssistantContentPart::Reasoning(ReasoningPart::new("remove this")),
                AssistantContentPart::ToolCall(ToolCallPart::new(
                    "call_1",
                    "search",
                    serde_json::json!({ "query": "rust" }),
                )),
                AssistantContentPart::ToolApprovalRequest(ToolApprovalRequest::new(
                    "approval_1",
                    "call_1",
                )),
            ]))),
            ModelMessage::Tool(ToolModelMessage::new(vec![
                ToolContentPart::ToolResult(ToolResultPart::new(
                    "call_1",
                    "search",
                    ToolResultOutput::json(serde_json::json!({ "answer": "ok" })),
                )),
                ToolContentPart::ToolApprovalResponse(ToolApprovalResponse::new(
                    "approval_1",
                    true,
                )),
            ])),
            ModelMessage::Assistant(AssistantModelMessage::new(AssistantContent::parts(vec![
                AssistantContentPart::Reasoning(ReasoningPart::new("keep this")),
                AssistantContentPart::ToolCall(ToolCallPart::new(
                    "call_2",
                    "weather",
                    serde_json::json!({ "city": "HK" }),
                )),
            ]))),
        ];

        let pruned = prune_messages(
            messages,
            PruneMessagesOptions::new()
                .with_reasoning(PruneReasoningMode::BeforeLastMessage)
                .with_tool_calls(vec![PruneToolCallRule::before_last_message()]),
        );

        assert_eq!(pruned.len(), 2);

        let ModelMessage::Assistant(first) = &pruned[0] else {
            panic!("first pruned message should be assistant");
        };
        let AssistantContent::Parts(first_parts) = &first.content else {
            panic!("first assistant message should keep parts");
        };
        assert_eq!(first_parts.len(), 1);
        assert!(matches!(first_parts[0], AssistantContentPart::Text(_)));

        let ModelMessage::Assistant(last) = &pruned[1] else {
            panic!("last pruned message should be assistant");
        };
        let AssistantContent::Parts(last_parts) = &last.content else {
            panic!("last assistant message should keep parts");
        };
        assert!(
            last_parts
                .iter()
                .any(|part| matches!(part, AssistantContentPart::Reasoning(_)))
        );
        assert!(
            last_parts
                .iter()
                .any(|part| matches!(part, AssistantContentPart::ToolCall(_)))
        );
    }

    #[test]
    fn stream_text_chunk_event_accepts_parts_and_lifecycle_markers() {
        let part_event: StreamTextChunkEvent = StreamTextChunkEvent::new(TextStreamPart::from(
            TextStreamTextDeltaPart::new("text_1", "hello"),
        ));
        let part_json = serde_json::to_value(&part_event).expect("serialize part chunk event");

        assert_eq!(part_json["chunk"]["type"], serde_json::json!("text-delta"));
        assert_eq!(part_json["chunk"]["text"], serde_json::json!("hello"));

        let lifecycle_event: StreamTextChunkEvent = StreamTextChunkEvent::new(
            StreamTextLifecycleChunk::first_chunk("call_1", 0)
                .with_attribute("phase", serde_json::json!("first")),
        );
        let lifecycle_json =
            serde_json::to_value(&lifecycle_event).expect("serialize lifecycle chunk event");

        assert_eq!(
            lifecycle_json["chunk"]["type"],
            serde_json::json!("ai.stream.firstChunk")
        );
        assert_eq!(
            lifecycle_json["chunk"]["callId"],
            serde_json::json!("call_1")
        );
        assert_eq!(lifecycle_json["chunk"]["stepNumber"], serde_json::json!(0));
        assert_eq!(
            lifecycle_json["chunk"]["attributes"]["phase"],
            serde_json::json!("first")
        );

        let roundtrip: StreamTextChunkEvent =
            serde_json::from_value(lifecycle_json).expect("deserialize lifecycle chunk event");
        assert_eq!(roundtrip.chunk.r#type(), "ai.stream.firstChunk");
    }

    #[test]
    fn generated_file_and_reasoning_outputs_match_ai_sdk_shape() {
        let file = GeneratedFile::from_bytes(b"hello", "text/plain");
        assert_eq!(file.base64(), "aGVsbG8=");
        assert_eq!(file.uint8_array().expect("decode generated file"), b"hello");
        assert_eq!(
            serde_json::to_value(&file).expect("serialize generated file"),
            serde_json::json!({
                "base64": "aGVsbG8=",
                "mediaType": "text/plain"
            })
        );

        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "reasoningId": "rs_1" }),
        )]);
        let reasoning = ReasoningOutput::new("internal reasoning")
            .with_provider_metadata(provider_metadata.clone());
        let reasoning_json = serde_json::to_value(&reasoning).expect("serialize reasoning output");
        assert_eq!(reasoning.r#type(), "reasoning");
        assert_eq!(
            reasoning_json,
            serde_json::json!({
                "type": "reasoning",
                "text": "internal reasoning",
                "providerMetadata": {
                    "openai": { "reasoningId": "rs_1" }
                }
            })
        );
        let reasoning_roundtrip: ReasoningOutput =
            serde_json::from_value(reasoning_json).expect("deserialize reasoning output");
        assert_eq!(reasoning_roundtrip.text, "internal reasoning");

        let reasoning_file =
            ReasoningFileOutput::new(file).with_provider_metadata(provider_metadata);
        let reasoning_file_json =
            serde_json::to_value(&reasoning_file).expect("serialize reasoning-file output");
        assert_eq!(reasoning_file.r#type(), "reasoning-file");
        assert_eq!(
            reasoning_file_json,
            serde_json::json!({
                "type": "reasoning-file",
                "file": {
                    "base64": "aGVsbG8=",
                    "mediaType": "text/plain"
                },
                "providerMetadata": {
                    "openai": { "reasoningId": "rs_1" }
                }
            })
        );
        let wrong_type = serde_json::json!({
            "type": "reasoning-text",
            "text": "nope"
        });
        assert!(serde_json::from_value::<ReasoningOutput>(wrong_type).is_err());
    }

    #[test]
    fn tool_error_and_output_denied_match_ai_sdk_shape() {
        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "itemId": "item_error" }),
        )]);
        let tool_error = ToolError::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
            serde_json::json!({ "message": "timeout" }),
        )
        .with_provider_executed(true)
        .with_provider_metadata(provider_metadata)
        .with_dynamic(true)
        .with_title("Search failed");
        let tool_error_json = serde_json::to_value(&tool_error).expect("serialize tool error");

        assert_eq!(tool_error.r#type(), "tool-error");
        assert_eq!(
            tool_error_json,
            serde_json::json!({
                "type": "tool-error",
                "toolCallId": "call_1",
                "toolName": "search",
                "input": { "q": "rust" },
                "error": { "message": "timeout" },
                "providerExecuted": true,
                "providerMetadata": {
                    "openai": { "itemId": "item_error" }
                },
                "dynamic": true,
                "title": "Search failed"
            })
        );
        let roundtrip: ToolError =
            serde_json::from_value(tool_error_json).expect("deserialize tool error");
        assert_eq!(roundtrip.tool_call_id, "call_1");

        let denied = ToolOutputDenied::new("call_2", "delete".to_string())
            .with_provider_executed(false)
            .with_dynamic(false);
        let denied_json = serde_json::to_value(&denied).expect("serialize output denied");

        assert_eq!(denied.r#type(), "tool-output-denied");
        assert_eq!(
            denied_json,
            serde_json::json!({
                "type": "tool-output-denied",
                "toolCallId": "call_2",
                "toolName": "delete",
                "providerExecuted": false,
                "dynamic": false
            })
        );
        let wrong_type = serde_json::json!({
            "type": "tool-denied",
            "toolCallId": "call_2",
            "toolName": "delete"
        });
        assert!(serde_json::from_value::<ToolOutputDenied>(wrong_type).is_err());
    }

    #[test]
    fn prepare_step_approval_and_repair_shapes_match_ai_sdk_options() {
        let mut provider_options = ProviderOptionsMap::new();
        provider_options.insert("anthropic", serde_json::json!({ "container": "ctn_1" }));
        let mut tools_context = Context::new();
        tools_context.insert("tenant".to_string(), serde_json::json!("docs"));
        let mut runtime_context = Context::new();
        runtime_context.insert("traceId".to_string(), serde_json::json!("trace_1"));
        let messages = vec![ModelMessage::Assistant(AssistantModelMessage::new(
            AssistantContent::text("calling search"),
        ))];

        let prepare_result = PrepareStepResult {
            model: Some(CallbackModelInfo::new("anthropic", "claude-test")),
            tool_choice: Some(ToolChoice::tool("search")),
            active_tools: Some(vec!["search".to_string()]),
            system: Some(SystemPrompt::Text("Use the docs.".to_string())),
            messages: Some(messages.clone()),
            tools_context: Some(tools_context.clone()),
            runtime_context: Some(runtime_context.clone()),
            provider_options: Some(provider_options),
        };
        let prepare_json =
            serde_json::to_value(&prepare_result).expect("serialize prepare-step result");

        assert_eq!(
            prepare_json["model"]["provider"],
            serde_json::json!("anthropic")
        );
        assert_eq!(
            prepare_json["model"]["modelId"],
            serde_json::json!("claude-test")
        );
        assert_eq!(
            prepare_json["toolChoice"]["toolName"],
            serde_json::json!("search")
        );
        assert_eq!(prepare_json["activeTools"][0], serde_json::json!("search"));
        assert_eq!(prepare_json["system"], serde_json::json!("Use the docs."));
        assert_eq!(
            prepare_json["toolsContext"]["tenant"],
            serde_json::json!("docs")
        );
        assert_eq!(
            prepare_json["runtimeContext"]["traceId"],
            serde_json::json!("trace_1")
        );
        assert_eq!(
            prepare_json["providerOptions"]["anthropic"]["container"],
            serde_json::json!("ctn_1")
        );

        let approval_status = ToolApprovalStatus::detailed(
            ToolApprovalStatusType::Denied,
            Some("policy denied".to_string()),
        );
        let approval_status_json =
            serde_json::to_value(&approval_status).expect("serialize approval status");
        assert_eq!(
            approval_status_json,
            serde_json::json!({ "type": "denied", "reason": "policy denied" })
        );
        assert_eq!(
            serde_json::to_value(ToolApprovalStatus::user_approval())
                .expect("serialize simple approval status"),
            serde_json::json!("user-approval")
        );

        let approval_config = ToolApprovalConfiguration::from([(
            "search".to_string(),
            ToolApprovalStatus::approved(),
        )]);
        let approval_config_json =
            serde_json::to_value(&approval_config).expect("serialize approval config");
        assert_eq!(
            approval_config_json["search"],
            serde_json::json!("approved")
        );

        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "query": "rust" }),
        );
        let approval_context = ToolApprovalDecisionContext {
            tool_call: tool_call.clone(),
            tools: Some(vec![Tool::function(
                "search",
                "Search docs",
                serde_json::json!({ "type": "object" }),
            )]),
            tools_context: tools_context.clone(),
            runtime_context: runtime_context.clone(),
            messages: messages.clone(),
        };
        let approval_context_json =
            serde_json::to_value(&approval_context).expect("serialize approval context");
        assert_eq!(
            approval_context_json["toolCall"]["toolName"],
            serde_json::json!("search")
        );
        assert_eq!(
            approval_context_json["toolsContext"]["tenant"],
            serde_json::json!("docs")
        );

        let repair_context = ToolCallRepairContext {
            system: Some(SystemPrompt::Text("Use tools carefully.".to_string())),
            messages,
            tool_call: tool_call.clone(),
            tools: vec![Tool::function(
                "lookup",
                "Lookup docs",
                serde_json::json!({ "type": "object" }),
            )],
            input_schemas: HashMap::from([(
                "lookup".to_string(),
                serde_json::json!({ "type": "object" }),
            )]),
            error: ToolCallRepairFunctionError::NoSuchTool(NoSuchToolError::new(
                "search",
                Some(vec!["lookup".to_string()]),
            )),
        };
        let repair_json = serde_json::to_value(&repair_context).expect("serialize repair context");
        assert_eq!(
            repair_json["toolCall"]["toolName"],
            serde_json::json!("search")
        );
        assert_eq!(
            repair_json["inputSchemas"]["lookup"]["type"],
            serde_json::json!("object")
        );
        assert_eq!(
            repair_json["error"],
            serde_json::json!({
                "type": "no-such-tool",
                "toolName": "search",
                "availableTools": ["lookup"],
                "message": "Model tried to call unavailable tool 'search'. Available tools: lookup."
            })
        );

        let invalid_tool_input_error =
            ToolCallRepairFunctionError::from(InvalidToolInputError::new(
                "search",
                "{",
                Some(serde_json::json!({ "message": "bad input" })),
            ));
        let invalid_tool_input_json =
            serde_json::to_value(&invalid_tool_input_error).expect("serialize invalid input error");
        assert_eq!(
            invalid_tool_input_json,
            serde_json::json!({
                "type": "invalid-tool-input",
                "toolName": "search",
                "toolInput": "{",
                "message": r#"Invalid input for tool search: {"message":"bad input"}"#,
                "cause": { "message": "bad input" }
            })
        );

        let repair_error = ToolCallRepairError::new(
            serde_json::from_value(repair_json["error"].clone()).expect("deserialize repair error"),
            Some(serde_json::json!({ "message": "repair failed" })),
        );
        let repair_error_json =
            serde_json::to_value(&repair_error).expect("serialize repair error wrapper");
        assert_eq!(
            repair_error_json["message"],
            serde_json::json!(r#"Error repairing tool call: {"message":"repair failed"}"#)
        );
        assert_eq!(
            repair_error_json["originalError"]["type"],
            serde_json::json!("no-such-tool")
        );
        assert_eq!(
            repair_error_json["cause"]["message"],
            serde_json::json!("repair failed")
        );

        let repair_result: ToolCallRepairResult = Some(ToolCall::new(
            "call_1",
            "lookup".to_string(),
            serde_json::json!({}),
        ));
        let repair_result_json =
            serde_json::to_value(&repair_result).expect("serialize repair result");
        assert_eq!(repair_result_json["type"], serde_json::json!("tool-call"));
        assert_eq!(repair_result_json["toolName"], serde_json::json!("lookup"));
    }

    #[test]
    fn tool_execution_events_and_tool_output_match_ai_sdk_shape() {
        let messages = vec![ModelMessage::Assistant(AssistantModelMessage::new(
            AssistantContent::text("calling search"),
        ))];
        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "query": "rust" }),
        );
        let start_event = ToolExecutionStartEvent {
            call_id: "call_1".to_string(),
            messages: messages.clone(),
            tool_call: tool_call.clone(),
            tool_context: Some(serde_json::json!({ "tenant": "docs" })),
        };
        let start_json =
            serde_json::to_value(&start_event).expect("serialize tool execution start event");

        assert_eq!(start_json["callId"], serde_json::json!("call_1"));
        assert_eq!(
            start_json["toolCall"]["toolCallId"],
            serde_json::json!("call_1")
        );
        assert_eq!(
            start_json["toolContext"]["tenant"],
            serde_json::json!("docs")
        );
        assert_eq!(
            start_json["messages"][0]["role"],
            serde_json::json!("assistant")
        );

        let tool_output = ToolOutput::from(ToolResult::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "query": "rust" }),
            ToolResultOutput::json(serde_json::json!({ "answer": "ok" })),
        ));
        let end_event = ToolExecutionEndEvent {
            call_id: "call_1".to_string(),
            duration_ms: 42,
            messages,
            tool_call,
            tool_context: Some(serde_json::json!({ "tenant": "docs" })),
            tool_output,
        };
        let end_json =
            serde_json::to_value(&end_event).expect("serialize tool execution end event");

        assert_eq!(end_json["durationMs"], serde_json::json!(42));
        assert_eq!(
            end_json["toolCall"]["toolName"],
            serde_json::json!("search")
        );
        assert_eq!(end_json["toolContext"]["tenant"], serde_json::json!("docs"));
        assert_eq!(
            end_json["toolOutput"]["type"],
            serde_json::json!("tool-result")
        );
        assert_eq!(
            end_json["toolOutput"]["output"]["value"]["answer"],
            serde_json::json!("ok")
        );

        let roundtrip: ToolExecutionEndEvent =
            serde_json::from_value(end_json).expect("deserialize tool execution end event");
        assert_eq!(roundtrip.tool_output.r#type(), "tool-result");

        let error_output: ToolOutput = ToolError::new(
            "call_2",
            "search".to_string(),
            serde_json::json!({ "query": "rust" }),
            serde_json::json!({ "message": "timeout" }),
        )
        .into();
        assert_eq!(error_output.r#type(), "tool-error");
    }

    #[test]
    fn generate_text_tool_approval_outputs_match_ai_sdk_shape() {
        let tool_call = ToolCall::new(
            "call_approval",
            "dangerous_tool".to_string(),
            serde_json::json!({ "path": "/tmp/file" }),
        )
        .with_provider_executed(true)
        .with_title("Dangerous tool");

        let request =
            ToolApprovalRequestOutput::new("approval_1", tool_call.clone()).with_is_automatic(true);
        let request_json = serde_json::to_value(&request).expect("serialize request output");

        assert_eq!(request.r#type(), "tool-approval-request");
        assert_eq!(
            request_json,
            serde_json::json!({
                "type": "tool-approval-request",
                "approvalId": "approval_1",
                "toolCall": {
                    "type": "tool-call",
                    "toolCallId": "call_approval",
                    "toolName": "dangerous_tool",
                    "input": { "path": "/tmp/file" },
                    "providerExecuted": true,
                    "title": "Dangerous tool"
                },
                "isAutomatic": true
            })
        );
        let roundtrip: ToolApprovalRequestOutput =
            serde_json::from_value(request_json).expect("deserialize request output");
        assert_eq!(roundtrip.approval_id, "approval_1");
        assert_eq!(roundtrip.tool_call.tool_call_id, "call_approval");

        let response = ToolApprovalResponseOutput::new("approval_1", tool_call, false)
            .with_reason("denied by policy")
            .with_provider_executed(true);
        let response_json = serde_json::to_value(&response).expect("serialize response output");

        assert_eq!(response.r#type(), "tool-approval-response");
        assert_eq!(
            response_json,
            serde_json::json!({
                "type": "tool-approval-response",
                "approvalId": "approval_1",
                "toolCall": {
                    "type": "tool-call",
                    "toolCallId": "call_approval",
                    "toolName": "dangerous_tool",
                    "input": { "path": "/tmp/file" },
                    "providerExecuted": true,
                    "title": "Dangerous tool"
                },
                "approved": false,
                "reason": "denied by policy",
                "providerExecuted": true
            })
        );
        let wrong_type = serde_json::json!({
            "type": "tool-approval",
            "approvalId": "approval_1",
            "toolCall": {
                "toolCallId": "call_approval",
                "toolName": "dangerous_tool",
                "input": {}
            }
        });
        assert!(serde_json::from_value::<ToolApprovalRequestOutput>(wrong_type).is_err());
    }

    #[test]
    fn language_model_v4_call_options_overlay_keeps_model_facing_fields_together() {
        let stable_prompt = vec![ModelMessage::System(SystemModelMessage::new("Be concise"))];
        let prompt = prepare_language_model_v4_prompt(stable_prompt.clone());
        let tool = Tool::function(
            "weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                }
            }),
        );

        let options: LanguageModelV4CallOptions =
            LanguageModelV4CallOptions::from_model_messages(stable_prompt)
                .with_max_output_tokens(u64::from(u32::MAX) + 1)
                .with_stable_tools(vec![tool])
                .with_tool_choice(ToolChoice::Required)
                .with_header("x-test", "1")
                .without_header("x-drop");

        assert_eq!(options.prompt, prompt);
        assert_eq!(options.max_output_tokens, Some(u64::from(u32::MAX) + 1));
        assert_eq!(
            options.tool_choice,
            Some(LanguageModelV4ToolChoice::Required)
        );
        assert_eq!(
            options.effective_headers().get("x-test"),
            Some(&"1".to_string())
        );
        assert!(!options.effective_headers().contains_key("x-drop"));

        let tools = options.tools.expect("tools are projected");
        assert!(matches!(tools[0], LanguageModelV4Tool::Function(_)));
        assert_eq!(
            serde_json::to_value(&tools[0]).expect("serialize model-facing tool")["type"],
            serde_json::json!("function")
        );
    }

    #[test]
    fn language_model_v4_prompt_projection_matches_provider_prompt_shape() {
        let user = UserModelMessage::new(UserContent::parts(vec![
            UserContentPart::Text(TextPart::new("hello")),
            UserContentPart::Image(ImagePart::new(FilePartSource::url(
                "https://example.com/image.png",
            ))),
            UserContentPart::File(
                FilePart::new(
                    FilePartSource::provider_reference(ProviderReference::single(
                        "openai", "file_1",
                    )),
                    "application/pdf",
                )
                .with_filename("paper.pdf"),
            ),
        ]));
        let assistant = AssistantModelMessage::new(AssistantContent::parts(vec![
            AssistantContentPart::Text(TextPart::new("")),
            AssistantContentPart::ReasoningFile(ReasoningFilePart::new(
                MediaSource::base64("cmVhc29u"),
                "text/plain",
            )),
            AssistantContentPart::ToolApprovalRequest(ToolApprovalRequest::new(
                "approval_1",
                "call_1",
            )),
        ]));
        let first_tool = ToolModelMessage::new(vec![ToolContentPart::ToolResult(
            ToolResultPart::new("call_1", "weather", ToolResultOutput::text("sunny")),
        )]);
        let mut approval_provider_options = ProviderOptionsMap::default();
        approval_provider_options
            .insert("anthropic", serde_json::json!({ "approvalMode": "manual" }));
        let second_tool = ToolModelMessage::new(vec![ToolContentPart::ToolApprovalResponse(
            ToolApprovalResponse::new("approval_2", true)
                .with_provider_executed(true)
                .with_provider_options_map(approval_provider_options),
        )]);

        let prompt = prepare_language_model_v4_prompt(vec![
            ModelMessage::User(user),
            ModelMessage::Assistant(assistant),
            ModelMessage::Tool(first_tool),
            ModelMessage::Tool(second_tool),
        ]);

        let json = serde_json::to_value(&prompt).expect("serialize projected V4 prompt");
        assert_eq!(json[0]["role"], serde_json::json!("user"));
        assert_eq!(json[0]["content"][1]["type"], serde_json::json!("file"));
        assert_eq!(
            json[0]["content"][1]["mediaType"],
            serde_json::json!("image/*")
        );
        assert_eq!(
            json[0]["content"][1]["data"],
            serde_json::json!("https://example.com/image.png")
        );
        assert_eq!(
            json[0]["content"][2]["data"],
            serde_json::json!({ "openai": "file_1" })
        );
        assert_eq!(
            json[1]["content"],
            serde_json::json!([
                {
                    "type": "reasoning-file",
                    "data": "cmVhc29u",
                    "mediaType": "text/plain"
                }
            ])
        );
        assert_eq!(json[2]["role"], serde_json::json!("tool"));
        assert_eq!(
            json[2]["content"].as_array().expect("tool content").len(),
            2
        );
        assert_eq!(
            json[2]["content"][1],
            serde_json::json!({
                "type": "tool-approval-response",
                "approvalId": "approval_2",
                "approved": true,
                "providerOptions": {
                    "anthropic": {
                        "approvalMode": "manual"
                    }
                }
            })
        );
    }

    #[test]
    fn language_model_v4_provider_options_require_object_values() {
        assert!(
            serde_json::from_value::<LanguageModelV4TextPart>(serde_json::json!({
                "type": "text",
                "text": "hello",
                "providerOptions": {
                    "openai": "not-an-object"
                }
            }))
            .is_err()
        );

        let mut invalid_provider_options = ProviderOptionsMap::default();
        invalid_provider_options.insert("openai", serde_json::json!("not-an-object"));
        let invalid_part = LanguageModelV4TextPart::new("hello")
            .with_provider_options_map(invalid_provider_options);
        assert!(serde_json::to_value(&invalid_part).is_err());

        let valid_json = serde_json::json!({
            "type": "text",
            "text": "hello",
            "providerOptions": {
                "openai": {
                    "cacheControl": {
                        "type": "ephemeral"
                    }
                }
            }
        });
        let valid_part: LanguageModelV4TextPart = serde_json::from_value(valid_json.clone())
            .expect("deserialize V4 text provider options");
        assert_eq!(
            serde_json::to_value(&valid_part).expect("serialize V4 text provider options"),
            valid_json
        );
    }

    #[test]
    fn language_model_v4_prompt_projection_filters_non_object_provider_options() {
        let mut message_provider_options = ProviderOptionsMap::default();
        message_provider_options.insert("openai", serde_json::json!({ "store": false }));
        message_provider_options.insert("legacy", serde_json::json!("drop"));

        let mut text_provider_options = ProviderOptionsMap::default();
        text_provider_options.insert(
            "anthropic",
            serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
        );
        text_provider_options.insert("legacy", serde_json::json!(true));

        let user = UserModelMessage::new(UserContent::parts(vec![UserContentPart::Text(
            TextPart::new("hello").with_provider_options_map(text_provider_options),
        )]))
        .with_provider_options_map(message_provider_options);

        let prompt = prepare_language_model_v4_prompt(vec![ModelMessage::User(user)]);
        let json = serde_json::to_value(&prompt).expect("serialize projected V4 prompt");

        assert_eq!(
            json[0]["providerOptions"],
            serde_json::json!({
                "openai": {
                    "store": false
                }
            })
        );
        assert!(json[0]["providerOptions"].get("legacy").is_none());
        assert_eq!(
            json[0]["content"][0]["providerOptions"],
            serde_json::json!({
                "anthropic": {
                    "cacheControl": {
                        "type": "ephemeral"
                    }
                }
            })
        );
        assert!(
            json[0]["content"][0]["providerOptions"]
                .get("legacy")
                .is_none()
        );
    }

    #[test]
    fn language_model_v4_tool_result_projection_filters_non_object_provider_options() {
        let output = ToolResultOutput::text("ok")
            .with_provider_option("openai", serde_json::json!({ "store": false }))
            .with_provider_option("legacy", serde_json::json!(false));
        let projected = LanguageModelV4ToolResultOutput::from_tool_result_output(&output);
        let json = serde_json::to_value(&projected).expect("serialize V4 tool output");

        assert_eq!(
            json["providerOptions"],
            serde_json::json!({
                "openai": {
                    "store": false
                }
            })
        );
        assert!(json["providerOptions"].get("legacy").is_none());

        let content_output = ToolResultOutput::content(vec![
            ToolResultContentPart::text("ok")
                .with_provider_option("anthropic", serde_json::json!({ "type": "tool-reference" }))
                .with_provider_option("legacy", serde_json::json!(null)),
        ]);
        let projected = LanguageModelV4ToolResultOutput::from_tool_result_output(&content_output);
        let json = serde_json::to_value(&projected).expect("serialize V4 content tool output");

        assert_eq!(json["type"], serde_json::json!("content"));
        assert_eq!(
            json["value"][0]["providerOptions"],
            serde_json::json!({
                "anthropic": {
                    "type": "tool-reference"
                }
            })
        );
        assert!(json["value"][0]["providerOptions"].get("legacy").is_none());
    }

    #[test]
    fn language_model_v4_provider_metadata_requires_object_values() {
        assert!(
            serde_json::from_value::<LanguageModelV4Text>(serde_json::json!({
                "type": "text",
                "text": "hello",
                "providerMetadata": {
                    "openai": "not-an-object"
                }
            }))
            .is_err()
        );

        let invalid_provider_metadata =
            ProviderMetadata::from([("openai".to_string(), serde_json::json!("not-an-object"))]);
        let invalid_content =
            LanguageModelV4Text::new("hello").with_provider_metadata(invalid_provider_metadata);
        assert!(serde_json::to_value(&invalid_content).is_err());

        let invalid_result = serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": "hello"
                }
            ],
            "finishReason": {
                "unified": "stop"
            },
            "usage": {
                "inputTokens": {},
                "outputTokens": {}
            },
            "providerMetadata": {
                "openai": 42
            },
            "warnings": []
        });
        assert!(serde_json::from_value::<LanguageModelV4GenerateResult>(invalid_result).is_err());

        let valid_json = serde_json::json!({
            "type": "text",
            "text": "hello",
            "providerMetadata": {
                "openai": {
                    "itemId": "msg_1"
                }
            }
        });
        let valid_content: LanguageModelV4Text =
            serde_json::from_value(valid_json.clone()).expect("deserialize V4 text metadata");
        assert_eq!(
            serde_json::to_value(&valid_content).expect("serialize V4 text metadata"),
            valid_json
        );
    }

    #[test]
    fn language_model_v4_content_projection_filters_non_object_provider_metadata() {
        let provider_metadata = ProviderMetadata::from([
            (
                "openai".to_string(),
                serde_json::json!({ "itemId": "msg_1" }),
            ),
            ("legacy".to_string(), serde_json::json!("drop")),
        ]);

        let text = TextOutput::new("hello").with_provider_metadata(provider_metadata.clone());
        let projected_text = LanguageModelV4Text::from_text_output(&text);
        let text_json = serde_json::to_value(&projected_text).expect("serialize projected text");
        assert_eq!(
            text_json["providerMetadata"],
            serde_json::json!({
                "openai": {
                    "itemId": "msg_1"
                }
            })
        );
        assert!(text_json["providerMetadata"].get("legacy").is_none());

        let source = Source::url("src_1", "https://example.com")
            .with_provider_metadata(provider_metadata.clone());
        let projected_source = LanguageModelV4Source::from_source(&source);
        let source_json =
            serde_json::to_value(&projected_source).expect("serialize projected source");
        assert_eq!(
            source_json["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("msg_1")
        );
        assert!(source_json["providerMetadata"].get("legacy").is_none());

        let custom = CustomOutput::new("openai.compaction")
            .with_provider_metadata(provider_metadata.clone());
        let projected_custom = LanguageModelV4CustomContent::try_from_custom_output(&custom)
            .expect("valid custom output");
        let custom_json =
            serde_json::to_value(&projected_custom).expect("serialize projected custom output");
        assert_eq!(
            custom_json["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("msg_1")
        );
        assert!(custom_json["providerMetadata"].get("legacy").is_none());

        let content: LanguageModelV4Content = text.into();
        let content_json = serde_json::to_value(&content).expect("serialize projected content");
        assert!(content_json["providerMetadata"].get("legacy").is_none());
    }

    #[test]
    fn language_model_v4_custom_kind_enforces_provider_prefix_shape() {
        let mut provider_options = ProviderOptionsMap::default();
        provider_options.insert("openai", serde_json::json!({ "mode": "compact" }));

        let prompt_part = LanguageModelV4CustomPart::try_new("openai.compaction")
            .expect("valid custom prompt kind")
            .with_provider_options_map(provider_options);
        let prompt_json =
            serde_json::to_value(&prompt_part).expect("serialize V4 custom prompt part");
        assert_eq!(prompt_json["type"], serde_json::json!("custom"));
        assert_eq!(prompt_json["kind"], serde_json::json!("openai.compaction"));
        assert_eq!(
            prompt_json["providerOptions"]["openai"]["mode"],
            serde_json::json!("compact")
        );

        assert!(
            serde_json::from_value::<LanguageModelV4CustomPart>(serde_json::json!({
                "type": "custom",
                "kind": "compaction"
            }))
            .is_err()
        );
        assert!(serde_json::to_value(LanguageModelV4CustomPart::new("compaction")).is_err());

        let content = LanguageModelV4CustomContent::try_new("openai.compaction")
            .expect("valid custom output kind")
            .with_provider_metadata(ProviderMetadata::from([(
                "openai".to_string(),
                serde_json::json!({ "itemId": "cmp_1" }),
            )]));
        let content_json =
            serde_json::to_value(&content).expect("serialize V4 custom content part");
        assert_eq!(content_json["type"], serde_json::json!("custom"));
        assert_eq!(
            content_json["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("cmp_1")
        );
        assert!(
            serde_json::from_value::<LanguageModelV4CustomContent>(serde_json::json!({
                "type": "custom",
                "kind": "compaction"
            }))
            .is_err()
        );
    }

    #[test]
    fn language_model_v4_prompt_projection_drops_invalid_custom_kinds() {
        let assistant = AssistantModelMessage::new(AssistantContent::parts(vec![
            AssistantContentPart::Custom(CustomPart::new("compaction")),
            AssistantContentPart::Custom(CustomPart::new("openai.compaction")),
        ]));

        let prompt = prepare_language_model_v4_prompt(vec![ModelMessage::Assistant(assistant)]);
        let json = serde_json::to_value(&prompt).expect("serialize projected V4 prompt");

        assert_eq!(
            json[0]["content"],
            serde_json::json!([
                {
                    "type": "custom",
                    "kind": "openai.compaction"
                }
            ])
        );
    }

    #[test]
    fn language_model_v4_custom_content_projects_valid_stable_outputs_only() {
        assert!(
            LanguageModelV4CustomContent::try_from_custom_output(&CustomOutput::new("compaction"))
                .is_none()
        );

        let projected = LanguageModelV4CustomContent::try_from_custom_output(&CustomOutput::new(
            "openai.compaction",
        ))
        .expect("valid stable custom output");
        let json = serde_json::to_value(&projected).expect("serialize projected custom output");

        assert_eq!(json["type"], serde_json::json!("custom"));
        assert_eq!(json["kind"], serde_json::json!("openai.compaction"));
    }

    #[test]
    fn language_model_v4_tool_result_output_canonicalizes_content_parts() {
        let output = ToolResultOutput::content(vec![
            ToolResultContentPart::text("ok"),
            ToolResultContentPart::image_data("aGVsbG8=", "image/png"),
            ToolResultContentPart::image_url("https://example.com/image.png"),
            ToolResultContentPart::file_reference(ProviderReference::single("openai", "file_1")),
            ToolResultContentPart::file_id(HashMap::from([(
                "anthropic".to_string(),
                "file_ant".to_string(),
            )])),
            ToolResultContentPart::custom()
                .with_provider_option("anthropic", serde_json::json!({ "type": "tool-reference" })),
        ]);

        let projected = LanguageModelV4ToolResultOutput::from_tool_result_output(&output);
        let json = serde_json::to_value(&projected).expect("serialize V4 tool output");

        assert_eq!(json["type"], serde_json::json!("content"));
        assert_eq!(json["value"][0]["type"], serde_json::json!("text"));
        assert_eq!(json["value"][1]["type"], serde_json::json!("file-data"));
        assert_eq!(
            json["value"][1]["mediaType"],
            serde_json::json!("image/png")
        );
        assert_eq!(json["value"][2]["type"], serde_json::json!("file-url"));
        assert_eq!(json["value"][2]["mediaType"], serde_json::json!("image/*"));
        assert_eq!(
            json["value"][3]["type"],
            serde_json::json!("file-reference")
        );
        assert_eq!(
            json["value"][3]["providerReference"]["openai"],
            serde_json::json!("file_1")
        );
        assert_eq!(
            json["value"][4]["type"],
            serde_json::json!("file-reference")
        );
        assert_eq!(
            json["value"][4]["providerReference"]["anthropic"],
            serde_json::json!("file_ant")
        );
        assert_eq!(json["value"][5]["type"], serde_json::json!("custom"));
        assert_eq!(
            json["value"][5]["providerOptions"]["anthropic"]["type"],
            serde_json::json!("tool-reference")
        );
    }

    #[test]
    fn language_model_v4_tool_result_output_falls_back_for_unrepresentable_legacy_content() {
        let output = ToolResultOutput::content(vec![
            ToolResultContentPart::file_id("file_without_provider"),
            ToolResultContentPart::file_url("https://example.com/report.pdf"),
        ]);

        let projected = LanguageModelV4ToolResultOutput::from_tool_result_output(&output);
        let json = serde_json::to_value(&projected).expect("serialize V4 tool output");

        assert_eq!(json["type"], serde_json::json!("json"));
        assert_eq!(json["value"][0]["type"], serde_json::json!("file-id"));
        assert_eq!(
            json["value"][0]["fileId"],
            serde_json::json!("file_without_provider")
        );
        assert_eq!(json["value"][1]["type"], serde_json::json!("file-url"));
        assert!(json["providerOptions"].is_null());
    }

    #[test]
    fn language_model_v4_prompt_projection_uses_canonical_tool_result_output() {
        let tool = ToolModelMessage::new(vec![ToolContentPart::ToolResult(ToolResultPart::new(
            "call_1",
            "vision",
            ToolResultOutput::content(vec![ToolResultContentPart::image_url(
                "https://example.com/image.png",
            )]),
        ))]);

        let prompt = prepare_language_model_v4_prompt(vec![ModelMessage::Tool(tool)]);
        let json = serde_json::to_value(&prompt).expect("serialize projected V4 prompt");

        assert_eq!(
            json[0]["content"][0]["output"]["type"],
            serde_json::json!("content")
        );
        assert_eq!(
            json[0]["content"][0]["output"]["value"][0]["type"],
            serde_json::json!("file-url")
        );
        assert_eq!(
            json[0]["content"][0]["output"]["value"][0]["mediaType"],
            serde_json::json!("image/*")
        );
    }

    #[test]
    fn language_model_v4_generate_result_matches_provider_content_shape() {
        let tool_call = LanguageModelV4ToolCall::from_json_input(
            "call_1",
            "weather",
            serde_json::json!({ "city": "Paris" }),
        )
        .expect("stringify tool input")
        .with_provider_executed(true);
        let usage = LanguageModelV4Usage::new(
            LanguageModelV4InputTokens {
                total: Some(10),
                no_cache: Some(7),
                cache_read: Some(3),
                cache_write: None,
            },
            LanguageModelV4OutputTokens {
                total: Some(4),
                text: Some(2),
                reasoning: Some(2),
            },
        );
        let response = LanguageModelV4GenerateResponseMetadata {
            id: Some("resp_1".to_string()),
            timestamp: None,
            model_id: Some("model-1".to_string()),
            headers: Some(HashMap::from([(
                "x-request-id".to_string(),
                "req_1".to_string(),
            )])),
            body: Some(serde_json::json!({ "raw": true })),
        };

        let result = LanguageModelV4GenerateResult::new(
            vec![
                LanguageModelV4Text::new("hello").into(),
                LanguageModelV4ReasoningFile::new("cmVhc29u", "text/plain").into(),
                LanguageModelV4File::new(vec![1_u8, 2, 3], "application/octet-stream").into(),
                tool_call.into(),
                LanguageModelV4ToolResult::new(
                    "call_1",
                    "weather",
                    serde_json::json!({ "temperature": 21 }),
                )
                .with_preliminary(true)
                .into(),
                LanguageModelV4ToolApprovalRequest::new("approval_1", "call_2").into(),
            ],
            LanguageModelV4FinishReason::new(FinishReason::ToolCalls, Some("tool_calls".into())),
            usage,
        )
        .with_response(response);

        let json = serde_json::to_value(&result).expect("serialize V4 generate result");
        assert_eq!(
            json["finishReason"]["unified"],
            serde_json::json!("tool-calls")
        );
        assert_eq!(json["finishReason"]["raw"], serde_json::json!("tool_calls"));
        assert_eq!(
            json["usage"]["inputTokens"]["cacheRead"],
            serde_json::json!(3)
        );
        assert_eq!(json["response"]["modelId"], serde_json::json!("model-1"));
        assert_eq!(json["content"][0]["type"], serde_json::json!("text"));
        assert_eq!(
            json["content"][1]["type"],
            serde_json::json!("reasoning-file")
        );
        assert_eq!(json["content"][1]["data"], serde_json::json!("cmVhc29u"));
        assert_eq!(json["content"][2]["type"], serde_json::json!("file"));
        assert_eq!(
            json["content"][2]["data"],
            serde_json::json!([1_u8, 2_u8, 3_u8])
        );
        assert_eq!(json["content"][3]["type"], serde_json::json!("tool-call"));
        assert_eq!(
            json["content"][3]["input"],
            serde_json::json!(r#"{"city":"Paris"}"#)
        );
        assert_eq!(json["content"][4]["type"], serde_json::json!("tool-result"));
        assert_eq!(
            json["content"][4]["result"],
            serde_json::json!({ "temperature": 21 })
        );
        assert_eq!(
            json["content"][5],
            serde_json::json!({
                "type": "tool-approval-request",
                "approvalId": "approval_1",
                "toolCallId": "call_2"
            })
        );

        let roundtrip: LanguageModelV4GenerateResult =
            serde_json::from_value(json).expect("deserialize V4 generate result");
        assert_eq!(roundtrip.content[3].r#type(), "tool-call");
    }

    #[test]
    fn language_model_v4_generated_files_use_generated_file_data_shape() {
        let file = LanguageModelV4File::new(
            LanguageModelV4GeneratedFileData::string("ZmFrZQ=="),
            "image/png",
        );
        let reasoning_file = LanguageModelV4ReasoningFile::new(
            LanguageModelV4GeneratedFileData::bytes([1_u8, 2, 3]),
            "application/octet-stream",
        );

        let file_json = serde_json::to_value(&file).expect("serialize generated file");
        let reasoning_file_json =
            serde_json::to_value(&reasoning_file).expect("serialize reasoning file");

        assert_eq!(file_json["type"], serde_json::json!("file"));
        assert_eq!(file_json["data"], serde_json::json!("ZmFrZQ=="));
        assert_eq!(
            reasoning_file_json["type"],
            serde_json::json!("reasoning-file")
        );
        assert_eq!(reasoning_file_json["data"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn language_model_v4_tool_result_rejects_null_result_payload() {
        let invalid = serde_json::json!({
            "type": "tool-result",
            "toolCallId": "call_1",
            "toolName": "weather",
            "result": null
        });

        assert!(serde_json::from_value::<LanguageModelV4ToolResult>(invalid).is_err());

        let result = LanguageModelV4ToolResult::new("call_1", "weather", serde_json::Value::Null);
        assert!(serde_json::to_value(&result).is_err());
    }

    #[test]
    fn language_model_v4_usage_keeps_provider_sized_token_counts() {
        let above_u32 = u64::from(u32::MAX) + 1;
        let usage = LanguageModelV4Usage::new(
            LanguageModelV4InputTokens {
                total: Some(above_u32),
                no_cache: Some(above_u32 - 20),
                cache_read: Some(10),
                cache_write: Some(10),
            },
            LanguageModelV4OutputTokens {
                total: Some(above_u32 + 100),
                text: Some(above_u32 + 80),
                reasoning: Some(20),
            },
        );

        let json = serde_json::to_value(&usage).expect("serialize V4 usage");
        assert_eq!(json["inputTokens"]["total"], serde_json::json!(above_u32));
        assert_eq!(
            json["outputTokens"]["text"],
            serde_json::json!(above_u32 + 80)
        );

        let roundtrip: LanguageModelV4Usage =
            serde_json::from_value(json).expect("deserialize V4 usage");
        assert_eq!(roundtrip.input_tokens.total, Some(above_u32));
        assert_eq!(roundtrip.output_tokens.text, Some(above_u32 + 80));

        let stable_usage = LanguageModelV4Usage::new(
            super::super::UsageInputTokens {
                total: Some(10),
                no_cache: Some(7),
                cache_read: Some(2),
                cache_write: Some(1),
            },
            super::super::UsageOutputTokens {
                total: Some(5),
                text: Some(4),
                reasoning: Some(1),
            },
        );
        assert_eq!(stable_usage.input_tokens.total, Some(10_u64));
        assert_eq!(stable_usage.output_tokens.reasoning, Some(1_u64));
    }

    #[test]
    #[allow(deprecated)]
    fn call_settings_projects_onto_call_and_request_options() {
        let settings = CallSettings::new()
            .with_max_output_tokens(256)
            .with_temperature(0.4)
            .with_top_p(0.8)
            .with_stop_sequences(vec!["END".to_string()])
            .with_seed(7)
            .with_reasoning(LanguageModelReasoning::Medium)
            .with_max_retries(2)
            .with_abort_signal(())
            .with_header("x-test", "1")
            .without_header("x-drop");

        let call_options = settings.language_model_call_options();
        let request_options = settings.request_options();

        assert_eq!(call_options.max_output_tokens, Some(256));
        assert_eq!(call_options.temperature, Some(0.4));
        assert_eq!(call_options.top_p, Some(0.8));
        assert_eq!(call_options.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(call_options.seed, Some(7));
        assert_eq!(call_options.reasoning, Some(LanguageModelReasoning::Medium));
        assert_eq!(request_options.max_retries, Some(2));
        assert_eq!(request_options.max_attempts(), Some(3));
        assert!(request_options.abort_signal.is_some());
        assert_eq!(
            request_options.effective_headers().get("x-test"),
            Some(&"1".to_string())
        );
        assert!(!request_options.effective_headers().contains_key("x-drop"));
        assert!(request_options.timeout.is_none());
    }
}
