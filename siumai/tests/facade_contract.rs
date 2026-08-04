use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use siumai::prelude::*;
use siumai::{EmbeddingLimits, ModelId, ResponseMetadata};

#[cfg(feature = "registry")]
use siumai::ProviderRegistration;

#[derive(Debug)]
struct FakeEmbedding {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
}

impl Model for FakeEmbedding {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl EmbeddingModel for FakeEmbedding {
    fn limits(&self) -> EmbeddingLimits {
        EmbeddingLimits {
            max_inputs: Some(4),
            max_input_tokens: None,
        }
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        _options: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        self.limits().validate(&request)?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        let response = EmbeddingResponse {
            embeddings: request.inputs().iter().map(|_| vec![1.0, 2.0]).collect(),
            metadata: ResponseMetadata::default(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };
        response.validate(&request)?;
        Ok(response)
    }
}

fn fake(model: ModelId, calls: Arc<AtomicUsize>) -> FakeEmbedding {
    FakeEmbedding {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fake").unwrap(),
            model,
            ModelFamily::Embedding,
        ),
        calls,
    }
}

#[tokio::test]
async fn direct_family_helper_preserves_one_call_per_batch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fake(ModelId::new("embed-v1").unwrap(), calls.clone());
    let request = EmbeddingRequest::new(["one", "two"]).unwrap();

    let response = embedding::embed(&model, request).await.unwrap();

    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "registry")]
#[tokio::test]
async fn direct_and_registry_paths_use_the_same_family_contract() {
    use siumai::core::{ModelPolicy, ModelPolicyContext, ModelPolicyDecision};
    use siumai::registry::{Registry, RouteId};

    #[derive(Debug)]
    struct UnknownPolicy;

    impl ModelPolicy for UnknownPolicy {
        fn evaluate(&self, _context: &ModelPolicyContext) -> ModelPolicyDecision {
            ModelPolicyDecision::unknown_model()
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let direct = fake(ModelId::new("embed-v1").unwrap(), calls.clone());
    let factory_calls = calls.clone();
    let registration =
        ProviderRegistration::new(ProviderId::new("fake").unwrap(), Arc::new(UnknownPolicy))
            .with_embedding(Arc::new(move |model| {
                Ok(Arc::new(fake(model, factory_calls.clone())) as Arc<dyn EmbeddingModel>)
            }));
    let mut builder = Registry::builder();
    builder
        .register(RouteId::new("primary").unwrap(), registration)
        .unwrap();
    let registry = builder.build().unwrap();
    let erased = registry.embedding_model("primary:embed-v1").unwrap();

    let direct_response = embedding::embed(&direct, EmbeddingRequest::single("one").unwrap())
        .await
        .unwrap();
    let erased_response = embedding::embed(&erased, EmbeddingRequest::single("one").unwrap())
        .await
        .unwrap();

    assert_eq!(direct_response, erased_response);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[cfg(all(feature = "registry", feature = "openai"))]
#[tokio::test]
async fn openai_direct_registry_and_helper_paths_share_one_wire_pipeline() {
    use serde_json::{Value, json};
    use siumai::providers::openai::configured::{OpenAiCredential, OpenAiProvider};
    use siumai::registry::{Registry, RouteId};
    use siumai_transport::{EndpointConfig, TransportEvent, TransportObserver};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[derive(Default)]
    struct CountingObserver {
        attempts: AtomicUsize,
        completed: AtomicUsize,
    }

    impl TransportObserver for CountingObserver {
        fn observe(&self, event: &TransportEvent) {
            match event {
                TransportEvent::AttemptStarted { .. } => {
                    self.attempts.fetch_add(1, Ordering::SeqCst);
                }
                TransportEvent::Completed { .. } => {
                    self.completed.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
        }
    }

    fn request() -> LanguageRequest {
        LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
    }

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_123",
            "created_at": 1,
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [{
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "hello back",
                    "annotations": []
                }]
            }]
        })))
        .expect(3)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chat_123",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-5.6-sol",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello back"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let observer = Arc::new(CountingObserver::default());
    let provider = OpenAiProvider::builder(OpenAiCredential::unauthenticated())
        .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap())
        .with_transport_observer(observer.clone())
        .build()
        .unwrap();
    let direct = provider.responses("gpt-5.6-sol").unwrap();
    let mut builder = Registry::builder();
    builder
        .register(
            RouteId::new("openai-responses").unwrap(),
            provider.responses_registration(),
        )
        .unwrap()
        .register(
            RouteId::new("openai-chat").unwrap(),
            provider.chat_completions_registration(),
        )
        .unwrap();
    let registry = builder.build().unwrap();
    let erased = registry
        .language_model("openai-responses:gpt-5.6-sol")
        .unwrap();
    let chat = registry.language_model("openai-chat:gpt-5.6-sol").unwrap();
    assert_eq!(erased.provider_id(), chat.provider_id());
    assert_eq!(erased.descriptor().api_mode(), Some("responses"));
    assert_eq!(chat.descriptor().api_mode(), Some("chat-completions"));

    let direct_response = direct
        .generate(request(), CallOptions::default())
        .await
        .unwrap();
    let erased_response = erased
        .generate(request(), CallOptions::default())
        .await
        .unwrap();
    let helper_response = language::generate(&direct, request()).await.unwrap();
    let chat_response = chat
        .generate(request(), CallOptions::default())
        .await
        .unwrap();

    assert_eq!(direct_response, erased_response);
    assert_eq!(erased_response, helper_response);
    assert!(matches!(
        &chat_response.content()[0],
        ContentPart::Text { text } if text == "hello back"
    ));
    assert_eq!(observer.attempts.load(Ordering::SeqCst), 4);
    assert_eq!(observer.completed.load(Ordering::SeqCst), 4);

    let requests = server.received_requests().await.unwrap();
    let bodies = requests
        .iter()
        .filter(|request| request.url.path() == "/v1/responses")
        .map(|request| serde_json::from_slice::<Value>(&request.body).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(bodies[0], bodies[1]);
    assert_eq!(bodies[1], bodies[2]);
}

#[cfg(feature = "openai-realtime")]
#[test]
fn facade_exposes_realtime_as_typed_provider_sessions() {
    use siumai::providers::openai::configured::experimental::realtime::{
        OPENAI_REALTIME_MODEL, OPENAI_REALTIME_TRANSLATION_MODEL, OpenAiRealtimeClientEvent,
        OpenAiTranslationClientEvent,
    };
    use siumai::providers::openai::configured::{OpenAiCredential, OpenAiProvider};

    let provider = OpenAiProvider::builder(OpenAiCredential::api_key("test-key"))
        .build()
        .unwrap();
    let conversation = provider.realtime(OPENAI_REALTIME_MODEL).unwrap();
    let translation = provider
        .translation(OPENAI_REALTIME_TRANSLATION_MODEL)
        .unwrap();

    assert_eq!(conversation.model(), OPENAI_REALTIME_MODEL);
    assert_eq!(translation.model(), OPENAI_REALTIME_TRANSLATION_MODEL);
    assert!(conversation.endpoint().is_official());
    assert!(translation.endpoint().is_official());
    assert_eq!(
        OpenAiRealtimeClientEvent::ResponseCreate {
            event_id: None,
            response: None,
        }
        .event_type(),
        "response.create"
    );
    assert_eq!(
        OpenAiTranslationClientEvent::SessionClose { event_id: None }.event_type(),
        "session.close"
    );
}
