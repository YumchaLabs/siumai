use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use siumai_core::{
    EmbeddingModel, ImageModel, LanguageModel, ModelId, ModelOperation, ModelPolicyDecision,
    ProviderRegistration, RerankModel, RouteId, SpeechModel, TranscriptionModel,
};

use crate::middleware::{
    MiddlewareStack, RegistryMiddleware, RegistryModelContext, contextualize_lookup,
};
use crate::{ModelReference, RegistryBuildError, RegistryResolveError};

/// Immutable route table shared by cheap [`Registry`] handles.
#[derive(Debug, Clone, Default)]
pub struct RegistrySnapshot {
    routes: BTreeMap<RouteId, ProviderRegistration>,
    aliases: BTreeMap<RouteId, RouteId>,
    middleware: MiddlewareStack,
}

impl RegistrySnapshot {
    pub fn routes(&self) -> impl ExactSizeIterator<Item = (&RouteId, &ProviderRegistration)> {
        self.routes.iter()
    }

    /// Return aliases flattened to their final provider route.
    pub fn aliases(&self) -> impl ExactSizeIterator<Item = (&RouteId, &RouteId)> {
        self.aliases.iter()
    }

    pub fn middlewares(&self) -> impl ExactSizeIterator<Item = &Arc<dyn RegistryMiddleware>> {
        self.middleware.iter()
    }

    pub fn registration(&self, route: &RouteId) -> Option<&ProviderRegistration> {
        self.canonical_route(route)
            .and_then(|route| self.routes.get(route))
    }

    fn canonical_route(&self, route: &RouteId) -> Option<&RouteId> {
        if let Some((route, _)) = self.routes.get_key_value(route) {
            Some(route)
        } else {
            self.aliases.get(route)
        }
    }
}

/// A cheap immutable router over configured provider registrations.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    snapshot: Arc<RegistrySnapshot>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn builder() -> RegistryBuilder {
        RegistryBuilder::new()
    }

    pub fn from_snapshot(snapshot: Arc<RegistrySnapshot>) -> Self {
        Self { snapshot }
    }

    pub fn snapshot(&self) -> Arc<RegistrySnapshot> {
        self.snapshot.clone()
    }

    /// Start a new snapshot from the current immutable route table.
    pub fn edit(&self) -> RegistryBuilder {
        RegistryBuilder {
            routes: self.snapshot.routes.clone(),
            aliases: self.snapshot.aliases.clone(),
            middleware: self.snapshot.middleware.clone(),
        }
    }

    pub fn language_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn LanguageModel>, RegistryResolveError> {
        let (registration, model, context) = self.resolve(reference)?;
        let model =
            registration
                .language_model(model)
                .map_err(|error| RegistryResolveError::Model {
                    context: context.clone(),
                    source: contextualize_lookup(error, context.route()),
                })?;
        self.snapshot.middleware.language(&context, model)
    }

    pub fn embedding_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn EmbeddingModel>, RegistryResolveError> {
        let (registration, model, context) = self.resolve(reference)?;
        let model =
            registration
                .embedding_model(model)
                .map_err(|error| RegistryResolveError::Model {
                    context: context.clone(),
                    source: contextualize_lookup(error, context.route()),
                })?;
        self.snapshot.middleware.embedding(&context, model)
    }

    pub fn rerank_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn RerankModel>, RegistryResolveError> {
        let (registration, model, context) = self.resolve(reference)?;
        let model =
            registration
                .rerank_model(model)
                .map_err(|error| RegistryResolveError::Model {
                    context: context.clone(),
                    source: contextualize_lookup(error, context.route()),
                })?;
        self.snapshot.middleware.rerank(&context, model)
    }

    pub fn image_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn ImageModel>, RegistryResolveError> {
        let (registration, model, context) = self.resolve(reference)?;
        let model =
            registration
                .image_model(model)
                .map_err(|error| RegistryResolveError::Model {
                    context: context.clone(),
                    source: contextualize_lookup(error, context.route()),
                })?;
        self.snapshot.middleware.image(&context, model)
    }

    pub fn speech_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn SpeechModel>, RegistryResolveError> {
        let (registration, model, context) = self.resolve(reference)?;
        let model =
            registration
                .speech_model(model)
                .map_err(|error| RegistryResolveError::Model {
                    context: context.clone(),
                    source: contextualize_lookup(error, context.route()),
                })?;
        self.snapshot.middleware.speech(&context, model)
    }

    pub fn transcription_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn TranscriptionModel>, RegistryResolveError> {
        let (registration, model, context) = self.resolve(reference)?;
        let model = registration.transcription_model(model).map_err(|error| {
            RegistryResolveError::Model {
                context: context.clone(),
                source: contextualize_lookup(error, context.route()),
            }
        })?;
        self.snapshot.middleware.transcription(&context, model)
    }

    pub fn evaluate(
        &self,
        reference: impl AsRef<str>,
        family: siumai_core::ModelFamily,
        operation: ModelOperation,
    ) -> Result<ModelPolicyDecision, RegistryResolveError> {
        let (registration, model, _) = self.resolve(reference)?;
        Ok(registration.evaluate(model, family, operation))
    }

    fn resolve(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<(&ProviderRegistration, ModelId, RegistryModelContext), RegistryResolveError> {
        let reference = ModelReference::parse(reference)?;
        let (route, model) = reference.into_parts();
        let canonical_route = self
            .snapshot
            .canonical_route(&route)
            .cloned()
            .ok_or_else(|| RegistryResolveError::UnknownRoute {
                route: route.clone(),
            })?;
        let registration = self
            .snapshot
            .routes
            .get(&canonical_route)
            .expect("canonical route must have a registration");
        Ok((
            registration,
            model,
            RegistryModelContext::new(route, canonical_route),
        ))
    }
}

/// Mutable assembly for one immutable registry snapshot.
#[derive(Debug, Clone, Default)]
pub struct RegistryBuilder {
    routes: BTreeMap<RouteId, ProviderRegistration>,
    aliases: BTreeMap<RouteId, RouteId>,
    middleware: MiddlewareStack,
}

impl RegistryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one identity-preserving model wrapper layer.
    pub fn middleware(&mut self, middleware: Arc<dyn RegistryMiddleware>) -> &mut Self {
        self.middleware.push(middleware);
        self
    }

    pub fn register(
        &mut self,
        route: RouteId,
        registration: ProviderRegistration,
    ) -> Result<&mut Self, RegistryBuildError> {
        if self.aliases.contains_key(&route) {
            return Err(RegistryBuildError::RouteAliasConflict { route });
        }
        if self.routes.contains_key(&route) {
            return Err(RegistryBuildError::DuplicateRoute { route });
        }
        self.routes.insert(route, registration);
        Ok(self)
    }

    /// Parse and register a route from its textual form.
    pub fn register_named(
        &mut self,
        route: impl AsRef<str>,
        registration: ProviderRegistration,
    ) -> Result<&mut Self, RegistryBuildError> {
        self.register(RouteId::new(route)?, registration)
    }

    /// Replace an existing route without mutating any previously built snapshot.
    pub fn replace(
        &mut self,
        route: RouteId,
        registration: ProviderRegistration,
    ) -> Result<&mut Self, RegistryBuildError> {
        let entry =
            self.routes
                .get_mut(&route)
                .ok_or_else(|| RegistryBuildError::MissingRoute {
                    route: route.clone(),
                })?;
        *entry = registration;
        Ok(self)
    }

    /// Parse and replace a route from its textual form.
    pub fn replace_named(
        &mut self,
        route: impl AsRef<str>,
        registration: ProviderRegistration,
    ) -> Result<&mut Self, RegistryBuildError> {
        self.replace(RouteId::new(route)?, registration)
    }

    pub fn alias(
        &mut self,
        alias: RouteId,
        target: RouteId,
    ) -> Result<&mut Self, RegistryBuildError> {
        if self.routes.contains_key(&alias) {
            return Err(RegistryBuildError::RouteAliasConflict { route: alias });
        }
        if self.aliases.contains_key(&alias) {
            return Err(RegistryBuildError::DuplicateAlias { alias });
        }
        self.aliases.insert(alias, target);
        Ok(self)
    }

    /// Parse and register an alias from textual route IDs.
    pub fn alias_named(
        &mut self,
        alias: impl AsRef<str>,
        target: impl AsRef<str>,
    ) -> Result<&mut Self, RegistryBuildError> {
        self.alias(RouteId::new(alias)?, RouteId::new(target)?)
    }

    pub fn build(self) -> Result<Registry, RegistryBuildError> {
        let aliases = flatten_aliases(&self.routes, &self.aliases)?;
        Ok(Registry::from_snapshot(Arc::new(RegistrySnapshot {
            routes: self.routes,
            aliases,
            middleware: self.middleware,
        })))
    }
}

fn flatten_aliases(
    routes: &BTreeMap<RouteId, ProviderRegistration>,
    aliases: &BTreeMap<RouteId, RouteId>,
) -> Result<BTreeMap<RouteId, RouteId>, RegistryBuildError> {
    aliases
        .keys()
        .map(|alias| {
            let mut current = alias;
            let mut visited = BTreeSet::new();
            loop {
                if !visited.insert(current.clone()) {
                    return Err(RegistryBuildError::AliasCycle {
                        route: current.clone(),
                    });
                }
                let Some(target) = aliases.get(current) else {
                    if routes.contains_key(current) {
                        return Ok((alias.clone(), current.clone()));
                    }
                    return Err(RegistryBuildError::UnknownAliasTarget {
                        alias: alias.clone(),
                        target: current.clone(),
                    });
                };
                current = target;
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use siumai_core::{
        ApiModeId, CallOptions, EmbeddingLimits, EmbeddingRequest, EmbeddingResponse, Error,
        ErrorContext, ErrorKind, FinishReason, ImageRequest, ImageResponse, LanguageRequest,
        LanguageResponse, LanguageStream, Model, ModelDescriptor, ModelFamily, ModelLookupError,
        ModelPolicy, ModelPolicyContext, ModelPolicyDecision, ProtocolId, ProviderId,
        ProviderScope, RerankRequest, RerankResponse, SpeechRequest, SpeechResponse,
        TranscriptionRequest, TranscriptionResponse, Usage,
    };

    use super::*;

    #[derive(Debug)]
    struct AdvisoryPolicy;

    impl ModelPolicy for AdvisoryPolicy {
        fn evaluate(&self, _context: &ModelPolicyContext) -> ModelPolicyDecision {
            ModelPolicyDecision::unknown_model()
        }
    }

    #[derive(Debug)]
    struct FakeLanguageModel {
        descriptor: ModelDescriptor,
        runtime: Arc<usize>,
    }

    impl Model for FakeLanguageModel {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }
    }

    #[async_trait]
    impl LanguageModel for FakeLanguageModel {
        async fn generate(
            &self,
            _request: LanguageRequest,
            _options: CallOptions,
        ) -> Result<LanguageResponse, Error> {
            Ok(
                LanguageResponse::completed(Vec::new(), FinishReason::Stop, Usage::default())
                    .unwrap()
                    .with_id(self.runtime.to_string())
                    .with_model(self.descriptor.model().clone()),
            )
        }

        async fn stream(
            &self,
            _request: LanguageRequest,
            _options: CallOptions,
        ) -> Result<LanguageStream, Error> {
            unreachable!("registry tests only exercise construction")
        }
    }

    macro_rules! fake_family_model {
        ($name:ident, $trait:ident, $method:ident, $request:ty, $response:ty) => {
            #[derive(Debug)]
            struct $name {
                descriptor: ModelDescriptor,
            }

            impl Model for $name {
                fn descriptor(&self) -> &ModelDescriptor {
                    &self.descriptor
                }
            }

            #[async_trait]
            impl $trait for $name {
                async fn $method(
                    &self,
                    _request: $request,
                    _options: CallOptions,
                ) -> Result<$response, Error> {
                    unreachable!("registry tests only exercise construction")
                }
            }
        };
    }

    fake_family_model!(
        FakeEmbeddingModel,
        EmbeddingModel,
        embed,
        EmbeddingRequest,
        EmbeddingResponse
    );
    fake_family_model!(
        FakeRerankModel,
        RerankModel,
        rerank,
        RerankRequest,
        RerankResponse
    );
    fake_family_model!(
        FakeImageModel,
        ImageModel,
        generate_image,
        ImageRequest,
        ImageResponse
    );
    fake_family_model!(
        FakeSpeechModel,
        SpeechModel,
        synthesize,
        SpeechRequest,
        SpeechResponse
    );
    fake_family_model!(
        FakeTranscriptionModel,
        TranscriptionModel,
        transcribe,
        TranscriptionRequest,
        TranscriptionResponse
    );

    #[derive(Debug)]
    struct FailingEmbeddingModel {
        descriptor: ModelDescriptor,
        calls: Arc<AtomicUsize>,
    }

    #[derive(Debug)]
    struct CountingMiddleware {
        wraps: Arc<AtomicUsize>,
    }

    impl RegistryMiddleware for CountingMiddleware {
        fn wrap_embedding(
            &self,
            context: &RegistryModelContext,
            model: Arc<dyn EmbeddingModel>,
        ) -> Arc<dyn EmbeddingModel> {
            assert_eq!(context.requested_route().as_str(), "recommended");
            assert_eq!(context.route().as_str(), "production");
            self.wraps.fetch_add(1, Ordering::SeqCst);
            model
        }
    }

    #[derive(Debug)]
    struct IdentityDriftMiddleware;

    impl RegistryMiddleware for IdentityDriftMiddleware {
        fn wrap_embedding(
            &self,
            _context: &RegistryModelContext,
            model: Arc<dyn EmbeddingModel>,
        ) -> Arc<dyn EmbeddingModel> {
            Arc::new(FakeEmbeddingModel {
                descriptor: ModelDescriptor::new(
                    ProviderId::new("wrong-provider").unwrap(),
                    model.model_id().clone(),
                    ModelFamily::Embedding,
                ),
            })
        }
    }

    impl Model for FailingEmbeddingModel {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }
    }

    #[async_trait]
    impl EmbeddingModel for FailingEmbeddingModel {
        fn limits(&self) -> EmbeddingLimits {
            EmbeddingLimits {
                max_inputs: Some(7),
                max_input_tokens: None,
            }
        }

        async fn embed(
            &self,
            _request: EmbeddingRequest,
            _options: CallOptions,
        ) -> Result<EmbeddingResponse, Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(
                Error::new(ErrorKind::Provider, "scripted provider failure").with_context(
                    ErrorContext {
                        operation: Some(ModelOperation::Embed),
                        provider: Some(self.provider_id().clone()),
                        route: None,
                        model: Some(self.model_id().clone()),
                    },
                ),
            )
        }
    }

    fn registration(
        runtime: Arc<usize>,
        constructions: Arc<AtomicUsize>,
        mode: &str,
    ) -> ProviderRegistration {
        let scope = Arc::new(
            ProviderScope::new(ProviderId::new("fake").unwrap())
                .with_protocol(ProtocolId::new("native").unwrap())
                .with_api_mode(ApiModeId::new(mode).unwrap()),
        );
        let factory_scope = scope.clone();
        ProviderRegistration::from_scope(scope, Arc::new(AdvisoryPolicy)).with_language(Arc::new(
            move |model| {
                constructions.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(FakeLanguageModel {
                    descriptor: ModelDescriptor::from_scope(
                        factory_scope.clone(),
                        model,
                        ModelFamily::Language,
                    ),
                    runtime: runtime.clone(),
                }) as Arc<dyn LanguageModel>)
            },
        ))
    }

    fn all_family_registration() -> ProviderRegistration {
        let scope = Arc::new(ProviderScope::new(ProviderId::new("all-families").unwrap()));

        macro_rules! factory {
            ($model:ident, $trait:ident, $family:ident) => {{
                let scope = scope.clone();
                Arc::new(move |model| {
                    Ok(Arc::new($model {
                        descriptor: ModelDescriptor::from_scope(
                            scope.clone(),
                            model,
                            ModelFamily::$family,
                        ),
                    }) as Arc<dyn $trait>)
                })
            }};
        }

        ProviderRegistration::from_scope(scope.clone(), Arc::new(AdvisoryPolicy))
            .with_language(Arc::new(move |model| {
                Ok(Arc::new(FakeLanguageModel {
                    descriptor: ModelDescriptor::new(
                        ProviderId::new("all-families").unwrap(),
                        model,
                        ModelFamily::Language,
                    ),
                    runtime: Arc::new(1),
                }) as Arc<dyn LanguageModel>)
            }))
            .with_embedding(factory!(FakeEmbeddingModel, EmbeddingModel, Embedding))
            .with_rerank(factory!(FakeRerankModel, RerankModel, Rerank))
            .with_image(factory!(FakeImageModel, ImageModel, Image))
            .with_speech(factory!(FakeSpeechModel, SpeechModel, Speech))
            .with_transcription(factory!(
                FakeTranscriptionModel,
                TranscriptionModel,
                Transcription
            ))
    }

    fn route(value: &str) -> RouteId {
        RouteId::new(value).unwrap()
    }

    #[test]
    fn resolves_alias_chains_and_preserves_model_colons() {
        let constructions = Arc::new(AtomicUsize::new(0));
        let mut builder = Registry::builder();
        builder
            .register(
                route("account-a"),
                registration(Arc::new(1), constructions.clone(), "responses"),
            )
            .unwrap()
            .alias(route("recommended"), route("primary"))
            .unwrap()
            .alias(route("primary"), route("account-a"))
            .unwrap();
        let registry = builder.build().unwrap();

        let model = registry
            .language_model("recommended:publisher:model:v2")
            .unwrap();
        assert_eq!(model.model_id().as_str(), "publisher:model:v2");
        assert_eq!(model.descriptor().api_mode(), Some("responses"));
        assert_eq!(constructions.load(Ordering::SeqCst), 1);
        assert_eq!(
            registry
                .snapshot()
                .aliases()
                .find(|(alias, _)| alias.as_str() == "recommended")
                .map(|(_, target)| target.as_str()),
            Some("account-a")
        );
    }

    #[test]
    fn resolves_all_six_stable_model_families() {
        let mut builder = Registry::builder();
        builder
            .register_named("all", all_family_registration())
            .unwrap();
        let registry = builder.build().unwrap();

        assert_eq!(
            registry
                .language_model("all:model")
                .unwrap()
                .descriptor()
                .family(),
            ModelFamily::Language
        );
        assert_eq!(
            registry
                .embedding_model("all:model")
                .unwrap()
                .descriptor()
                .family(),
            ModelFamily::Embedding
        );
        assert_eq!(
            registry
                .rerank_model("all:model")
                .unwrap()
                .descriptor()
                .family(),
            ModelFamily::Rerank
        );
        assert_eq!(
            registry
                .image_model("all:model")
                .unwrap()
                .descriptor()
                .family(),
            ModelFamily::Image
        );
        assert_eq!(
            registry
                .speech_model("all:model")
                .unwrap()
                .descriptor()
                .family(),
            ModelFamily::Speech
        );
        assert_eq!(
            registry
                .transcription_model("all:model")
                .unwrap()
                .descriptor()
                .family(),
            ModelFamily::Transcription
        );
    }

    #[test]
    fn reports_unknown_route_and_unsupported_family() {
        let mut builder = Registry::builder();
        builder
            .register(
                route("known"),
                registration(Arc::new(1), Arc::new(AtomicUsize::new(0)), "responses"),
            )
            .unwrap()
            .alias(route("recommended"), route("known"))
            .unwrap();
        let registry = builder.build().unwrap();

        assert!(matches!(
            registry.language_model("missing:model"),
            Err(RegistryResolveError::UnknownRoute { .. })
        ));
        let error = match registry.image_model("recommended:model") {
            Ok(_) => panic!("unsupported image family unexpectedly resolved"),
            Err(error) => error,
        };
        let RegistryResolveError::Model { context, source } = error else {
            panic!("expected a route-aware model lookup error");
        };
        assert_eq!(context.requested_route().as_str(), "recommended");
        assert_eq!(context.route().as_str(), "known");
        assert!(matches!(
            source,
            ModelLookupError::UnsupportedFamily {
                family: ModelFamily::Image,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn resolved_models_preserve_identity_limits_and_canonical_route_context() {
        let calls = Arc::new(AtomicUsize::new(0));
        let wraps = Arc::new(AtomicUsize::new(0));
        let scope = Arc::new(ProviderScope::new(ProviderId::new("fake").unwrap()));
        let factory_scope = scope.clone();
        let factory_calls = calls.clone();
        let registration = ProviderRegistration::from_scope(scope, Arc::new(AdvisoryPolicy))
            .with_embedding(Arc::new(move |model| {
                Ok(Arc::new(FailingEmbeddingModel {
                    descriptor: ModelDescriptor::from_scope(
                        factory_scope.clone(),
                        model,
                        ModelFamily::Embedding,
                    ),
                    calls: factory_calls.clone(),
                }) as Arc<dyn EmbeddingModel>)
            }));
        let mut builder = Registry::builder();
        builder
            .middleware(Arc::new(CountingMiddleware {
                wraps: wraps.clone(),
            }))
            .register(route("production"), registration)
            .unwrap()
            .alias(route("recommended"), route("production"))
            .unwrap();
        let registry = builder.build().unwrap();
        let model = registry.embedding_model("recommended:embed-v1").unwrap();

        assert_eq!(model.descriptor().provider().as_str(), "fake");
        assert_eq!(model.descriptor().model().as_str(), "embed-v1");
        assert_eq!(model.descriptor().family(), ModelFamily::Embedding);
        assert_eq!(model.limits().max_inputs, Some(7));

        let error = model
            .embed(
                EmbeddingRequest::single("hello").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(wraps.load(Ordering::SeqCst), 1);
        assert_eq!(
            error.context().route.as_ref().map(RouteId::as_str),
            Some("production")
        );
        assert_eq!(
            error.context().provider.as_ref().map(ProviderId::as_str),
            Some("fake")
        );
        assert_eq!(
            error.context().model.as_ref().map(ModelId::as_str),
            Some("embed-v1")
        );
    }

    #[test]
    fn middleware_cannot_change_model_identity() {
        let mut builder = Registry::builder();
        builder
            .middleware(Arc::new(IdentityDriftMiddleware))
            .register(route("production"), all_family_registration())
            .unwrap();
        let registry = builder.build().unwrap();

        let error = match registry.embedding_model("production:embed-v1") {
            Ok(_) => panic!("identity-changing middleware unexpectedly resolved"),
            Err(error) => error,
        };
        let RegistryResolveError::Model { context, source } = error else {
            panic!("expected a route-aware identity error");
        };
        assert_eq!(context.requested_route().as_str(), "production");
        assert_eq!(context.route().as_str(), "production");
        assert!(matches!(source, ModelLookupError::IdentityMismatch { .. }));
    }

    #[test]
    fn rejects_duplicates_conflicts_unknown_targets_and_cycles() {
        let registration = || registration(Arc::new(1), Arc::new(AtomicUsize::new(0)), "responses");
        let mut duplicate = Registry::builder();
        duplicate.register(route("one"), registration()).unwrap();
        assert!(matches!(
            duplicate.register(route("one"), registration()),
            Err(RegistryBuildError::DuplicateRoute { .. })
        ));

        let mut unknown = Registry::builder();
        unknown.alias(route("alias"), route("missing")).unwrap();
        assert!(matches!(
            unknown.build(),
            Err(RegistryBuildError::UnknownAliasTarget { .. })
        ));

        let mut cycle = Registry::builder();
        cycle.alias(route("a"), route("b")).unwrap();
        cycle.alias(route("b"), route("a")).unwrap();
        assert!(matches!(
            cycle.build(),
            Err(RegistryBuildError::AliasCycle { .. })
        ));

        let mut invalid = Registry::builder();
        assert!(matches!(
            invalid.register_named("route:model", registration()),
            Err(RegistryBuildError::InvalidRoute(_))
        ));
    }

    #[tokio::test]
    async fn replacement_creates_a_new_snapshot_without_rebinding_old_models() {
        let old_runtime = Arc::new(1usize);
        let new_runtime = Arc::new(2usize);
        let constructions = Arc::new(AtomicUsize::new(0));
        let mut builder = Registry::builder();
        builder
            .register(
                route("openai"),
                registration(old_runtime.clone(), constructions.clone(), "responses"),
            )
            .unwrap();
        let old_registry = builder.build().unwrap();
        let old_model = old_registry.language_model("openai:model").unwrap();

        let mut replacement = old_registry.edit();
        replacement
            .replace(
                route("openai"),
                registration(new_runtime, constructions, "chat-completions"),
            )
            .unwrap();
        let new_registry = replacement.build().unwrap();
        let new_model = new_registry.language_model("openai:model").unwrap();

        let old_response = old_model
            .generate(LanguageRequest::new(Vec::new()), CallOptions::default())
            .await
            .unwrap();
        let new_response = new_model
            .generate(LanguageRequest::new(Vec::new()), CallOptions::default())
            .await
            .unwrap();

        assert_eq!(old_model.descriptor().api_mode(), Some("responses"));
        assert_eq!(new_model.descriptor().api_mode(), Some("chat-completions"));
        assert_eq!(old_response.id(), Some("1"));
        assert_eq!(new_response.id(), Some("2"));
        assert_eq!(old_registry.snapshot().routes().len(), 1);
        assert_eq!(new_registry.snapshot().routes().len(), 1);
    }

    #[tokio::test]
    async fn distinct_routes_for_one_provider_keep_their_captured_runtimes() {
        let constructions = Arc::new(AtomicUsize::new(0));
        let mut builder = Registry::builder();
        builder
            .register(
                route("account-a"),
                registration(Arc::new(11), constructions.clone(), "chat-completions"),
            )
            .unwrap()
            .register(
                route("account-b"),
                registration(Arc::new(22), constructions, "responses"),
            )
            .unwrap();
        let registry = builder.build().unwrap();

        let account_a = registry.language_model("account-a:model").unwrap();
        let account_b = registry.language_model("account-b:model").unwrap();
        let response_a = account_a
            .generate(LanguageRequest::new(Vec::new()), CallOptions::default())
            .await
            .unwrap();
        let response_b = account_b
            .generate(LanguageRequest::new(Vec::new()), CallOptions::default())
            .await
            .unwrap();

        assert_eq!(account_a.provider_id(), account_b.provider_id());
        assert_eq!(account_a.descriptor().api_mode(), Some("chat-completions"));
        assert_eq!(account_b.descriptor().api_mode(), Some("responses"));
        assert_eq!(response_a.id(), Some("11"));
        assert_eq!(response_b.id(), Some("22"));
    }

    #[test]
    fn resolution_constructs_fresh_handles_without_a_registry_cache() {
        let constructions = Arc::new(AtomicUsize::new(0));
        let runtime = Arc::new(7usize);
        let mut builder = Registry::builder();
        builder
            .register(
                route("route"),
                registration(runtime, constructions.clone(), "responses"),
            )
            .unwrap();
        let registry = builder.build().unwrap();

        let handles = (0..64)
            .map(|_| {
                let registry = registry.clone();
                std::thread::spawn(move || registry.language_model("route:model").unwrap())
            })
            .collect::<Vec<_>>();
        for handle in handles {
            let _ = handle.join().unwrap();
        }
        assert_eq!(constructions.load(Ordering::SeqCst), 64);
    }
}
