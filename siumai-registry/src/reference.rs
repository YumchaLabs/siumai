use std::fmt;
use std::str::FromStr;

use siumai_core::{ModelId, RouteId};
use thiserror::Error;

/// A parsed `route:model` reference.
///
/// Only the first colon is a routing separator. Remaining colons are retained
/// as part of the provider-owned model ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelReference {
    route: RouteId,
    model: ModelId,
}

impl ModelReference {
    pub fn new(route: RouteId, model: ModelId) -> Self {
        Self { route, model }
    }

    pub fn parse(value: impl AsRef<str>) -> Result<Self, ModelReferenceError> {
        value.as_ref().parse()
    }

    pub fn route(&self) -> &RouteId {
        &self.route
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn into_parts(self) -> (RouteId, ModelId) {
        (self.route, self.model)
    }
}

impl fmt::Display for ModelReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.route, self.model)
    }
}

impl FromStr for ModelReference {
    type Err = ModelReferenceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (route, model) = value
            .split_once(':')
            .ok_or(ModelReferenceError::MissingSeparator)?;
        let route = RouteId::new(route)
            .map_err(|error| ModelReferenceError::InvalidRoute(error.to_string()))?;
        let model = ModelId::new(model)
            .map_err(|error| ModelReferenceError::InvalidModel(error.to_string()))?;
        Ok(Self::new(route, model))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ModelReferenceError {
    #[error("model reference must use the `route:model` form")]
    MissingSeparator,
    #[error("invalid route in model reference: {0}")]
    InvalidRoute(String),
    #[error("invalid model ID in model reference: {0}")]
    InvalidModel(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_only_the_first_colon() {
        let reference = ModelReference::parse("Primary:publisher:model/v2").unwrap();
        assert_eq!(reference.route().as_str(), "primary");
        assert_eq!(reference.model().as_str(), "publisher:model/v2");
        assert_eq!(reference.to_string(), "primary:publisher:model/v2");
    }

    #[test]
    fn rejects_malformed_references() {
        assert_eq!(
            ModelReference::parse("model-only").unwrap_err(),
            ModelReferenceError::MissingSeparator
        );
        assert!(matches!(
            ModelReference::parse(":model"),
            Err(ModelReferenceError::InvalidRoute(_))
        ));
        assert!(matches!(
            ModelReference::parse("route:"),
            Err(ModelReferenceError::InvalidModel(_))
        ));
    }
}
