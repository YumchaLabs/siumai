use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use siumai_core::{
    EmbeddingModel, ImageModel, LanguageModel, ModelId, ModelOperation, ModelPolicyDecision,
    ProviderRegistration, RerankModel, RouteId, SpeechModel, TranscriptionModel,
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
        let route = self.aliases.get(route).unwrap_or(route);
        self.routes.get(route)
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
        let (registration, model) = self.resolve(reference)?;
        Ok(registration.language_model(model)?)
    }

    pub fn embedding_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn EmbeddingModel>, RegistryResolveError> {
        let (registration, model) = self.resolve(reference)?;
        Ok(registration.embedding_model(model)?)
    }

    pub fn rerank_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn RerankModel>, RegistryResolveError> {
        let (registration, model) = self.resolve(reference)?;
        Ok(registration.rerank_model(model)?)
    }

    pub fn image_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn ImageModel>, RegistryResolveError> {
        let (registration, model) = self.resolve(reference)?;
        Ok(registration.image_model(model)?)
    }

    pub fn speech_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn SpeechModel>, RegistryResolveError> {
        let (registration, model) = self.resolve(reference)?;
        Ok(registration.speech_model(model)?)
    }

    pub fn transcription_model(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<Arc<dyn TranscriptionModel>, RegistryResolveError> {
        let (registration, model) = self.resolve(reference)?;
        Ok(registration.transcription_model(model)?)
    }

    pub fn evaluate(
        &self,
        reference: impl AsRef<str>,
        family: siumai_core::ModelFamily,
        operation: ModelOperation,
    ) -> Result<ModelPolicyDecision, RegistryResolveError> {
        let (registration, model) = self.resolve(reference)?;
        Ok(registration.evaluate(model, family, operation))
    }

    fn resolve(
        &self,
        reference: impl AsRef<str>,
    ) -> Result<(&ProviderRegistration, ModelId), RegistryResolveError> {
        let reference = ModelReference::parse(reference)?;
        let (route, model) = reference.into_parts();
        let registration = self
            .snapshot
            .registration(&route)
            .ok_or(RegistryResolveError::UnknownRoute { route })?;
        Ok((registration, model))
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
        ApiModeId, CallOptions, Error, FinishReason, LanguageRequest, LanguageResponse,
        LanguageStream, Model, ModelDescriptor, ModelFamily, ModelLookupError, ModelPolicy,
        ModelPolicyContext, ModelPolicyDecision, ProtocolId, ProviderId, ProviderScope, Usage,
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
    fn reports_unknown_route_and_unsupported_family() {
        let mut builder = Registry::builder();
        builder
            .register(
                route("known"),
                registration(Arc::new(1), Arc::new(AtomicUsize::new(0)), "responses"),
            )
            .unwrap();
        let registry = builder.build().unwrap();

        assert!(matches!(
            registry.language_model("missing:model"),
            Err(RegistryResolveError::UnknownRoute { .. })
        ));
        assert!(matches!(
            registry.image_model("known:model"),
            Err(RegistryResolveError::Model(
                ModelLookupError::UnsupportedFamily {
                    family: ModelFamily::Image,
                    ..
                }
            ))
        ));
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
