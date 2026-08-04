//! Immutable, provider-agnostic model routing for Siumai.
//!
//! Providers are configured in their own crates and contribute a
//! [`ProviderRegistration`]. This crate only maps route IDs to those captured
//! registrations and resolves one of the six stable model families.

#![deny(unsafe_code)]

mod error;
mod reference;
mod registry;

pub use error::{RegistryBuildError, RegistryResolveError};
pub use reference::{ModelReference, ModelReferenceError};
pub use registry::{Registry, RegistryBuilder, RegistrySnapshot};
pub use siumai_core::{ModelFamily, ProviderRegistration, RouteId};
