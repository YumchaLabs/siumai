use siumai_core::{InvalidId, ModelLookupError, RouteId};
use thiserror::Error;

use crate::ModelReferenceError;
use crate::middleware::RegistryModelContext;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistryBuildError {
    #[error(transparent)]
    InvalidRoute(#[from] InvalidId),
    #[error("route `{route}` is already registered")]
    DuplicateRoute { route: RouteId },
    #[error("route `{route}` is not registered and cannot be replaced")]
    MissingRoute { route: RouteId },
    #[error("alias `{alias}` is already registered")]
    DuplicateAlias { alias: RouteId },
    #[error("route ID `{route}` cannot be both a provider route and an alias")]
    RouteAliasConflict { route: RouteId },
    #[error("alias `{alias}` targets unknown route `{target}`")]
    UnknownAliasTarget { alias: RouteId, target: RouteId },
    #[error("alias cycle contains route `{route}`")]
    AliasCycle { route: RouteId },
}

#[derive(Debug, Error)]
pub enum RegistryResolveError {
    #[error(transparent)]
    InvalidReference(#[from] ModelReferenceError),
    #[error("unknown provider route `{route}`")]
    UnknownRoute { route: RouteId },
    #[error("failed to resolve model for {context}: {source}")]
    Model {
        context: RegistryModelContext,
        #[source]
        source: Box<ModelLookupError>,
    },
}
