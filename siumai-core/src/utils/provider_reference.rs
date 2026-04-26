//! AI SDK-style provider reference utility helpers.

use std::collections::HashMap;

use crate::types::{FilePartSource, NoSuchProviderReferenceError, ProviderReference};

/// Check whether a file source is backed by a provider reference.
pub fn is_provider_reference(source: &FilePartSource) -> bool {
    source.is_provider_reference()
}

/// Resolve a provider-specific reference from a provider reference map.
///
/// This mirrors AI SDK `resolveProviderReference`: missing providers return a
/// `NoSuchProviderReferenceError` carrier with the available reference map.
pub fn resolve_provider_reference<'a>(
    reference: &'a ProviderReference,
    provider: &str,
) -> Result<&'a str, NoSuchProviderReferenceError> {
    reference.get(provider).ok_or_else(|| {
        NoSuchProviderReferenceError::new(
            provider,
            reference
                .0
                .iter()
                .map(|(provider, reference)| (provider.clone(), reference.clone()))
                .collect::<HashMap<_, _>>(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_provider_reference_or_returns_error_carrier() {
        let reference = ProviderReference::from([("openai", "file-openai")]);

        assert_eq!(
            resolve_provider_reference(&reference, "openai").expect("provider reference"),
            "file-openai"
        );

        let error = resolve_provider_reference(&reference, "anthropic")
            .expect_err("missing provider should return error carrier");
        assert_eq!(error.provider, "anthropic");
        assert_eq!(
            error.reference.get("openai").map(String::as_str),
            Some("file-openai")
        );
    }

    #[test]
    fn checks_file_part_source_provider_reference() {
        assert!(is_provider_reference(
            &FilePartSource::single_provider_reference("openai", "file-openai")
        ));
        assert!(!is_provider_reference(&FilePartSource::url(
            "https://example.com/image.png"
        )));
    }
}
