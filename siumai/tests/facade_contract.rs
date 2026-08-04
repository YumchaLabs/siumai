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
