//! Capability gate choreography for executor-level hard guards.
//!
//! `ProviderCapabilities` answers whether a provider advertises a capability. This module owns the
//! runtime decision that follows when a requested capability is absent: reject, warn, or delegate to
//! provider fallback. Executors should depend on named requirements here instead of hand-assembling
//! policies and resolving them locally.

use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::{UnsupportedCapabilityBehavior, UnsupportedCapabilityPolicy, Warning};

/// Named hard capability requirement used by executors before provider execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityRequirement {
    /// Provider capability flag name.
    pub feature: &'static str,
    /// Public failure detail for this concrete operation.
    pub details: &'static str,
}

impl CapabilityRequirement {
    pub const fn new(feature: &'static str, details: &'static str) -> Self {
        Self { feature, details }
    }

    pub fn reject_policy(self) -> UnsupportedCapabilityPolicy {
        UnsupportedCapabilityPolicy::reject(self.feature, Some(self.details))
    }

    pub fn warn_policy(self) -> UnsupportedCapabilityPolicy {
        UnsupportedCapabilityPolicy::warn(self.feature, Some(self.details))
    }

    pub fn provider_fallback_policy(self) -> UnsupportedCapabilityPolicy {
        UnsupportedCapabilityPolicy::provider_fallback(self.feature, Some(self.details))
    }

    pub fn ensure_supported(self, capabilities: &ProviderCapabilities) -> Result<(), LlmError> {
        UnsupportedCapabilityGate::new(capabilities).reject_if_unsupported(self)
    }
}

/// Resolves capability policy decisions against a concrete provider capability snapshot.
pub struct UnsupportedCapabilityGate<'a> {
    capabilities: &'a ProviderCapabilities,
}

impl<'a> UnsupportedCapabilityGate<'a> {
    pub const fn new(capabilities: &'a ProviderCapabilities) -> Self {
        Self { capabilities }
    }

    /// Reject when a required capability is absent.
    pub fn reject_if_unsupported(
        &self,
        requirement: CapabilityRequirement,
    ) -> Result<(), LlmError> {
        if self.capabilities.supports(requirement.feature) {
            Ok(())
        } else {
            resolve_unsupported_capability_policy(requirement.reject_policy()).map(|_| ())
        }
    }

    /// Warn when a capability is absent but continuing preserves caller intent.
    pub fn warn_if_unsupported(
        &self,
        requirement: CapabilityRequirement,
    ) -> Result<Option<Warning>, LlmError> {
        if self.capabilities.supports(requirement.feature) {
            Ok(None)
        } else {
            resolve_unsupported_capability_policy(requirement.warn_policy())
        }
    }

    /// Surface a compatibility warning when support is deliberately delegated to the provider.
    pub fn provider_fallback_if_unsupported(
        &self,
        requirement: CapabilityRequirement,
    ) -> Result<Option<Warning>, LlmError> {
        if self.capabilities.supports(requirement.feature) {
            Ok(None)
        } else {
            resolve_unsupported_capability_policy(requirement.provider_fallback_policy())
        }
    }
}

/// Resolve an unsupported-capability policy into the shared runtime outcome.
///
/// `Reject` becomes `LlmError::UnsupportedOperation`. Non-reject policies return a warning that
/// callers can merge into response or stream-start warnings.
pub fn resolve_unsupported_capability_policy(
    policy: UnsupportedCapabilityPolicy,
) -> Result<Option<Warning>, LlmError> {
    match policy.behavior {
        UnsupportedCapabilityBehavior::Reject => Err(LlmError::UnsupportedOperation(
            unsupported_capability_message(&policy),
        )),
        UnsupportedCapabilityBehavior::Warn | UnsupportedCapabilityBehavior::ProviderFallback => {
            Ok(policy.warning())
        }
    }
}

fn unsupported_capability_message(policy: &UnsupportedCapabilityPolicy) -> String {
    match policy.details.as_deref() {
        Some(details) if !details.is_empty() => {
            format!("unsupported capability `{}`: {}", policy.feature, details)
        }
        _ => format!("unsupported capability `{}`", policy.feature),
    }
}

/// Standard executor requirements. Keep operation-specific details here so executors do not repeat
/// failure copy or policy construction.
pub mod requirements {
    use super::CapabilityRequirement;

    pub const EMBEDDING: CapabilityRequirement =
        CapabilityRequirement::new("embedding", "Embedding is not supported by this provider");
    pub const RERANK: CapabilityRequirement =
        CapabilityRequirement::new("rerank", "Rerank is not supported by this provider");
    pub const SPEECH: CapabilityRequirement =
        CapabilityRequirement::new("speech", "Text-to-speech is not supported by this provider");
    pub const TRANSCRIPTION: CapabilityRequirement = CapabilityRequirement::new(
        "transcription",
        "Speech-to-text is not supported by this provider",
    );
    pub const IMAGE_GENERATION: CapabilityRequirement = CapabilityRequirement::new(
        "image_generation",
        "Image generation is not supported by this provider",
    );
    pub const IMAGE_EDIT: CapabilityRequirement = CapabilityRequirement::new(
        "image_generation",
        "Image editing is not supported by this provider",
    );
    pub const IMAGE_VARIATION: CapabilityRequirement = CapabilityRequirement::new(
        "image_generation",
        "Image variation is not supported by this provider",
    );
    pub const FILE_UPLOAD: CapabilityRequirement = CapabilityRequirement::new(
        "file_management",
        "File management is not supported by this provider",
    );
    pub const FILE_LIST: CapabilityRequirement = CapabilityRequirement::new(
        "file_management",
        "File listing is not supported by this provider",
    );
    pub const FILE_RETRIEVE: CapabilityRequirement = CapabilityRequirement::new(
        "file_management",
        "File retrieve is not supported by this provider",
    );
    pub const FILE_DELETE: CapabilityRequirement = CapabilityRequirement::new(
        "file_management",
        "File delete is not supported by this provider",
    );
    pub const FILE_CONTENT: CapabilityRequirement = CapabilityRequirement::new(
        "file_management",
        "File content download is not supported by this provider",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_if_unsupported_returns_error() {
        let capabilities = ProviderCapabilities::new();
        let error = requirements::EMBEDDING
            .ensure_supported(&capabilities)
            .expect_err("missing embedding capability should reject");

        assert!(matches!(error, LlmError::UnsupportedOperation(_)));
        assert!(
            error
                .to_string()
                .contains("unsupported capability `embedding`: Embedding is not supported")
        );
    }

    #[test]
    fn reject_if_unsupported_passes_supported_capability() {
        let capabilities = ProviderCapabilities::new().with_embedding();

        requirements::EMBEDDING
            .ensure_supported(&capabilities)
            .expect("supported embedding capability should pass");
    }

    #[test]
    fn warning_and_provider_fallback_share_gate_resolution() {
        let capabilities = ProviderCapabilities::new();
        let gate = UnsupportedCapabilityGate::new(&capabilities);

        let warning = gate
            .warn_if_unsupported(requirements::IMAGE_EDIT)
            .expect("warning policy")
            .expect("warning value");
        assert!(matches!(warning, Warning::Unsupported { .. }));

        let warning = gate
            .provider_fallback_if_unsupported(requirements::IMAGE_EDIT)
            .expect("provider fallback policy")
            .expect("warning value");
        assert!(matches!(warning, Warning::Compatibility { .. }));
    }
}
