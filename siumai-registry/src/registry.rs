use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use siumai_core::{
    EmbeddingModel, ImageModel, LanguageModel, ModelId, ProviderRegistration, RerankModel, RouteId,
    SpeechModel, TranscriptionModel,
};

use crate::route::{
    RegistryModelContext, contextualize_lookup, with_embedding_route, with_image_route,
    with_language_route, with_rerank_route, with_speech_route, with_transcription_route,
};
use crate::{ModelReference, RegistryBuildError, RegistryResolveError};

/// Immutable route table shared by cheap [`Registry`] handles.
#[derive(Debug, Clone, Default)]
pub struct RegistrySnapshot {
    routes: BTreeMap<RouteId, ProviderRegistration>,
    aliases: BTreeMap<RouteId, RouteId>,
}

impl RegistrySnapshot {
    pub fn routes(&self) -> impl ExactSizeIterator<Item = (&RouteId, &ProviderRegistration)> {
        self.routes.iter()
    }

    /// Return aliases flattened to their final provider route.
    pub fn aliases(&self) -> impl ExactSizeIterator<Item = (&RouteId, &RouteId)> {
        self.aliases.iter()
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
                    source: Box::new(contextualize_lookup(error, context.route())),
                })?;
        Ok(with_language_route(&context, model))
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
                    source: Box::new(contextualize_lookup(error, context.route())),
                })?;
        Ok(with_embedding_route(&context, model))
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
                    source: Box::new(contextualize_lookup(error, context.route())),
                })?;
        Ok(with_rerank_route(&context, model))
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
                    source: Box::new(contextualize_lookup(error, context.route())),
                })?;
        Ok(with_image_route(&context, model))
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
                    source: Box::new(contextualize_lookup(error, context.route())),
                })?;
        Ok(with_speech_route(&context, model))
    }

    pub fn transcription_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn TranscriptionModel>, RegistryResolveError> {
        let (registration, model, context) = self.resolve(reference)?;
        let model = registration.transcription_model(model).map_err(|error| {
            RegistryResolveError::Model {
                context: context.clone(),
                source: Box::new(contextualize_lookup(error, context.route())),
            }
        })?;
        Ok(with_transcription_route(&context, model))
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
}

impl RegistryBuilder {
    pub fn new() -> Self {
        Self::default()
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
        ErrorContext, ErrorKind, ImageRequest, ImageResponse, LanguageCallError,
        LanguageCompletionReason, LanguageRequest, LanguageResponse, LanguageStream, Model,
        ModelDescriptor, ModelFamily, ModelLookupError, ModelOperation, ProtocolId, ProviderId,
        ProviderInstanceId, ProviderScope, RerankRequest, RerankResponse, SpeechRequest,
        SpeechResponse, TranscriptionRequest, TranscriptionResponse, Usage,
    };

    use super::*;

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
        ) -> Result<LanguageResponse, LanguageCallError> {
            Ok(LanguageResponse::completed(
                Vec::new(),
                LanguageCompletionReason::Stop,
                Usage::default(),
            )
            .unwrap()
            .with_id(self.runtime.to_string())
            .with_model(self.descriptor.model().clone()))
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
    struct FailingLanguageModel {
        descriptor: ModelDescriptor,
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

    impl Model for FailingLanguageModel {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }
    }

    #[async_trait]
    impl LanguageModel for FailingLanguageModel {
        async fn generate(
            &self,
            _request: LanguageRequest,
            _options: CallOptions,
        ) -> Result<LanguageResponse, LanguageCallError> {
            Err(Error::new(ErrorKind::Provider, "scripted language failure")
                .with_context(ErrorContext {
                    operation: Some(ModelOperation::Generate),
                    provider: Some(self.provider_id().clone()),
                    route: None,
                    model: Some(self.model_id().clone()),
                })
                .into())
        }

        async fn stream(
            &self,
            _request: LanguageRequest,
            _options: CallOptions,
        ) -> Result<LanguageStream, Error> {
            Err(Error::new(
                ErrorKind::Transport,
                "scripted stream establishment failure",
            )
            .with_context(ErrorContext {
                operation: Some(ModelOperation::Stream),
                provider: Some(self.provider_id().clone()),
                route: None,
                model: Some(self.model_id().clone()),
            }))
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
        let instance_id = ProviderInstanceId::new();
        ProviderRegistration::from_language(
            scope,
            Arc::new(move |model| {
                constructions.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(FakeLanguageModel {
                    descriptor: ModelDescriptor::from_scope(
                        factory_scope.clone(),
                        model,
                        ModelFamily::Language,
                        instance_id.clone(),
                    ),
                    runtime: runtime.clone(),
                }) as Arc<dyn LanguageModel>)
            }),
        )
    }

    fn all_family_registration() -> ProviderRegistration {
        let scope = Arc::new(ProviderScope::new(ProviderId::new("all-families").unwrap()));
        let instance_id = ProviderInstanceId::new();

        macro_rules! factory {
            ($model:ident, $trait:ident, $family:ident) => {{
                let scope = scope.clone();
                let instance_id = instance_id.clone();
                Arc::new(move |model| {
                    Ok(Arc::new($model {
                        descriptor: ModelDescriptor::from_scope(
                            scope.clone(),
                            model,
                            ModelFamily::$family,
                            instance_id.clone(),
                        ),
                    }) as Arc<dyn $trait>)
                })
            }};
        }

        ProviderRegistration::from_language(
            scope.clone(),
            Arc::new({
                let scope = scope.clone();
                let instance_id = instance_id.clone();
                move |model| {
                    Ok(Arc::new(FakeLanguageModel {
                        descriptor: ModelDescriptor::from_scope(
                            scope.clone(),
                            model,
                            ModelFamily::Language,
                            instance_id.clone(),
                        ),
                        runtime: Arc::new(1),
                    }) as Arc<dyn LanguageModel>)
                }
            }),
        )
        .bind_embedding(
            scope.clone(),
            factory!(FakeEmbeddingModel, EmbeddingModel, Embedding),
        )
        .unwrap()
        .bind_rerank(
            scope.clone(),
            factory!(FakeRerankModel, RerankModel, Rerank),
        )
        .unwrap()
        .bind_image(scope.clone(), factory!(FakeImageModel, ImageModel, Image))
        .unwrap()
        .bind_speech(
            scope.clone(),
            factory!(FakeSpeechModel, SpeechModel, Speech),
        )
        .unwrap()
        .bind_transcription(
            scope.clone(),
            factory!(FakeTranscriptionModel, TranscriptionModel, Transcription),
        )
        .unwrap()
    }

    fn failing_language_registration() -> ProviderRegistration {
        let scope = Arc::new(ProviderScope::new(ProviderId::new("failing").unwrap()));
        let factory_scope = scope.clone();
        let instance_id = ProviderInstanceId::new();
        ProviderRegistration::from_language(
            scope,
            Arc::new(move |model| {
                Ok(Arc::new(FailingLanguageModel {
                    descriptor: ModelDescriptor::from_scope(
                        factory_scope.clone(),
                        model,
                        ModelFamily::Language,
                        instance_id.clone(),
                    ),
                }) as Arc<dyn LanguageModel>)
            }),
        )
    }

    fn route(value: &str) -> RouteId {
        RouteId::new(value).unwrap()
    }

    #[test]
    fn resolves_unknown_models_through_aliases_and_preserves_model_colons() {
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
    fn resolves_all_six_stable_model_families_with_canonical_route_context() {
        let mut builder = Registry::builder();
        builder
            .register_named("all", all_family_registration())
            .unwrap()
            .alias_named("recommended", "all")
            .unwrap();
        let registry = builder.build().unwrap();

        macro_rules! assert_family_route {
            ($method:ident, $family:ident) => {
                for reference in ["all:model", "recommended:model"] {
                    let model = registry.$method(reference).unwrap();
                    assert_eq!(model.descriptor().family(), ModelFamily::$family);
                    assert_eq!(model.route_id().map(RouteId::as_str), Some("all"));
                }
            };
        }

        assert_family_route!(language_model, Language);
        assert_family_route!(embedding_model, Embedding);
        assert_family_route!(rerank_model, Rerank);
        assert_family_route!(image_model, Image);
        assert_family_route!(speech_model, Speech);
        assert_family_route!(transcription_model, Transcription);
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
            *source,
            ModelLookupError::UnsupportedFamily {
                family: ModelFamily::Image,
                ..
            }
        ));
    }

    #[test]
    fn construction_errors_preserve_requested_and_canonical_routes() {
        let scope = Arc::new(ProviderScope::new(ProviderId::new("failing").unwrap()));
        let registration = ProviderRegistration::from_language(
            scope,
            Arc::new(|_| {
                Err(ModelLookupError::Construction {
                    source: Error::new(ErrorKind::Configuration, "scripted construction failure"),
                })
            }),
        );
        let mut builder = Registry::builder();
        builder
            .register_named("production", registration)
            .unwrap()
            .alias_named("recommended", "production")
            .unwrap();
        let registry = builder.build().unwrap();

        let error = match registry.language_model("recommended:model") {
            Ok(_) => panic!("failing model unexpectedly resolved"),
            Err(error) => error,
        };
        let RegistryResolveError::Model { context, source } = error else {
            panic!("expected a route-aware construction error");
        };
        assert_eq!(context.requested_route().as_str(), "recommended");
        assert_eq!(context.route().as_str(), "production");
        let ModelLookupError::Construction { source } = *source else {
            panic!("expected a construction error source");
        };
        assert_eq!(
            source.context().route.as_ref().map(RouteId::as_str),
            Some("production")
        );
    }

    #[tokio::test]
    async fn language_call_errors_preserve_canonical_route_context() {
        let mut builder = Registry::builder();
        builder
            .register_named("production", failing_language_registration())
            .unwrap()
            .alias_named("recommended", "production")
            .unwrap();
        let registry = builder.build().unwrap();
        let model = registry.language_model("recommended:language-v1").unwrap();

        let direct = model
            .generate(LanguageRequest::new(Vec::new()), CallOptions::default())
            .await
            .unwrap_err();
        assert_eq!(
            direct.context().route.as_ref().map(RouteId::as_str),
            Some("production")
        );

        let stream = model
            .stream(LanguageRequest::new(Vec::new()), CallOptions::default())
            .await
            .unwrap_err();
        assert_eq!(
            stream.context().route.as_ref().map(RouteId::as_str),
            Some("production")
        );
    }

    #[tokio::test]
    async fn resolved_models_preserve_identity_limits_and_canonical_route_context() {
        let calls = Arc::new(AtomicUsize::new(0));
        let scope = Arc::new(ProviderScope::new(ProviderId::new("fake").unwrap()));
        let factory_scope = scope.clone();
        let instance_id = ProviderInstanceId::new();
        let factory_calls = calls.clone();
        let registration = ProviderRegistration::from_embedding(
            scope,
            Arc::new(move |model| {
                Ok(Arc::new(FailingEmbeddingModel {
                    descriptor: ModelDescriptor::from_scope(
                        factory_scope.clone(),
                        model,
                        ModelFamily::Embedding,
                        instance_id.clone(),
                    ),
                    calls: factory_calls.clone(),
                }) as Arc<dyn EmbeddingModel>)
            }),
        );
        let mut builder = Registry::builder();
        builder
            .register(route("production"), registration)
            .unwrap()
            .alias(route("recommended"), route("production"))
            .unwrap();
        let registry = builder.build().unwrap();
        let model = registry.embedding_model("recommended:embed-v1").unwrap();

        assert_eq!(model.descriptor().provider().as_str(), "fake");
        assert_eq!(model.descriptor().model().as_str(), "embed-v1");
        assert_eq!(model.descriptor().family(), ModelFamily::Embedding);
        assert_eq!(model.route_id().map(RouteId::as_str), Some("production"));
        assert_eq!(model.limits().max_inputs, Some(7));

        let error = model
            .embed(
                EmbeddingRequest::single("hello").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
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
