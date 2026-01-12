use super::*;
use async_trait::async_trait;

#[derive(serde::Deserialize, Debug, PartialEq)]
struct User {
    name: String,
    age: u32,
}

struct StreamOnlyModel {
    deltas: Vec<&'static str>,
}

#[async_trait]
impl ChatCapability for StreamOnlyModel {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation("non-stream".into()))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let chunks = self.deltas.clone();
        let s = async_stream::try_stream! {
            for d in chunks {
                yield ChatStreamEvent::ContentDelta { delta: d.to_string(), index: None };
            }
            yield ChatStreamEvent::StreamEnd {
                response: ChatResponse::new(
                    MessageContent::Text(String::new())
                ),
            };
        };
        Ok(Box::pin(s))
    }
}

#[tokio::test]
async fn stream_object_emits_partial_on_balanced_block() {
    // JSON appears across multiple deltas; partial should emit once balanced
    let model = StreamOnlyModel {
        deltas: vec!["prefix ", "{", "\"id\":1", "}", " suffix"],
    };
    let mut s = stream_object::<serde_json::Value>(&model, vec![], None, Default::default())
        .await
        .expect("stream");
    use futures::StreamExt;
    let mut saw_partial = false;
    let mut saw_final = false;
    while let Some(ev) = s.next().await {
        match ev.expect("ok") {
            StreamObjectEvent::PartialObject { partial } => {
                saw_partial = true;
                assert_eq!(partial.get("id").and_then(|v| v.as_u64()), Some(1));
            }
            StreamObjectEvent::Final { object, .. } => {
                saw_final = true;
                assert_eq!(object.get("id").and_then(|v| v.as_u64()), Some(1));
            }
            _ => {}
        }
    }
    assert!(saw_partial && saw_final);
}

#[tokio::test]
async fn stream_object_repairs_trailing_comma() {
    // Trailing comma appears before closing brace; repair should handle
    let model = StreamOnlyModel {
        deltas: vec!["{\"a\":1,", "}\n"],
    };
    let mut s = stream_object::<serde_json::Value>(&model, vec![], None, Default::default())
        .await
        .expect("stream");
    use futures::StreamExt;
    let mut final_obj: Option<serde_json::Value> = None;
    while let Some(ev) = s.next().await {
        if let StreamObjectEvent::Final { object, .. } = ev.expect("ok") {
            final_obj = Some(object);
        }
    }
    let obj = final_obj.expect("final");
    assert_eq!(obj.get("a").and_then(|v| v.as_u64()), Some(1));
}

struct MockModel;

#[async_trait]
impl ChatCapability for MockModel {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse::new(MessageContent::Text(
            "{\"name\":\"Ada\",\"age\":36}".to_string(),
        )))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation("no stream".into()))
    }
}

#[tokio::test]
async fn generate_object_happy_path() {
    let model = MockModel;
    let schema = serde_json::json!({
        "type":"object",
        "properties":{
            "name":{"type":"string"},
            "age":{"type":"integer","minimum":0}
        },
        "required":["name","age"]
    });
    let (user, _resp): (User, _) = generate_object(
        &model,
        vec![ChatMessage::user("give me user json").build()],
        None,
        GenerateObjectOptions {
            schema: Some(schema),
            output: OutputKind::Object,
            ..Default::default()
        },
    )
    .await
    .expect("object");
    assert_eq!(
        user,
        User {
            name: "Ada".into(),
            age: 36
        }
    );
}

#[tokio::test]
async fn stream_object_emits_partial_updates() {
    use futures::StreamExt;

    struct MockStreamModel;

    #[async_trait]
    impl ChatCapability for MockStreamModel {
        async fn chat_with_tools(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatResponse, LlmError> {
            Err(LlmError::UnsupportedOperation("no sync".into()))
        }

        async fn chat_stream(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatStream, LlmError> {
            let s = async_stream::try_stream! {
                yield ChatStreamEvent::ContentDelta { delta: "{".into(), index: None };
                yield ChatStreamEvent::ContentDelta { delta: "\"name\"".into(), index: None };
                yield ChatStreamEvent::ContentDelta { delta: ":\"Ada\",".into(), index: None };
                yield ChatStreamEvent::ContentDelta { delta: "\"age\":36}".into(), index: None };
                yield ChatStreamEvent::StreamEnd {
                    response: ChatResponse::new(
                        MessageContent::Text("".into())
                    ),
                };
            };
            Ok(Box::pin(s))
        }
    }

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct U {
        name: String,
        age: u32,
    }

    let model = MockStreamModel;
    let mut s = stream_object::<U>(
        &model,
        vec![ChatMessage::user("user").build()],
        None,
        StreamObjectOptions {
            emit_partial_object: true,
            ..Default::default()
        },
    )
    .await
    .expect("stream");
    let mut saw_partial = false;
    let mut saw_final = false;
    while let Some(ev) = s.next().await {
        match ev.expect("ok") {
            StreamObjectEvent::PartialObject { partial } => {
                // eventually becomes object with both fields
                if partial.get("name").is_some() {
                    saw_partial = true;
                }
            }
            StreamObjectEvent::Final { object, .. } => {
                assert_eq!(
                    object,
                    U {
                        name: "Ada".into(),
                        age: 36
                    }
                );
                saw_final = true;
            }
            _ => {}
        }
    }
    assert!(saw_partial, "should emit at least one partial object");
    assert!(saw_final, "should emit final object");
}
