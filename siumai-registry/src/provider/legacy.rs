//! Compatibility helpers for legacy `ProviderType` classification.
//!
//! Provider ids are the primary registry identity. This module is the narrow boundary that still
//! maps open provider ids to the historical closed `ProviderType` enum for public compatibility.

use crate::types::ProviderType;

/// Derive the legacy `ProviderType` classification from a provider id.
///
/// Keep this helper out of core/provider-id lookup paths. It exists only for compatibility fields
/// such as `ProviderInfo::provider_type` and `Siumai::metadata().provider_type`.
pub(crate) fn provider_type_for_id(provider_id: &str) -> ProviderType {
    ProviderType::from_name(provider_id)
}
