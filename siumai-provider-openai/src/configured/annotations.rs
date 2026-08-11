use serde::{Deserialize, Serialize};
use siumai_core::{
    ContentAnnotationTarget, ContentAnnotations, Error, ErrorKind, TypedProviderAnnotation,
};
use siumai_protocol_openai::{PromptCacheAnnotationResolver, PromptCacheNodeOptions};

/// OpenAI controls attached to one canonical message-content node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct OpenAiContentOptions {}

impl OpenAiContentOptions {
    /// Mark this content node as an explicit OpenAI prompt-cache breakpoint.
    ///
    /// Read and write outcomes are selected by the provider. The request does
    /// not attempt to classify a breakpoint as historical or newly written.
    pub const fn prompt_cache_breakpoint() -> Self {
        Self {}
    }
}

impl TypedProviderAnnotation for OpenAiContentOptions {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "openai";
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OpenAiAnnotationResolver;

impl PromptCacheAnnotationResolver for OpenAiAnnotationResolver {
    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<PromptCacheNodeOptions, Error> {
        annotations
            .decode::<OpenAiContentOptions>()
            .map(|value| {
                value.map_or_else(PromptCacheNodeOptions::default, |_| {
                    PromptCacheNodeOptions::default().with_explicit_breakpoint(true)
                })
            })
            .map_err(|source| {
                Error::new(ErrorKind::InvalidInput, "invalid OpenAI content annotation")
                    .with_source(source)
            })
    }
}

#[cfg(test)]
mod tests {
    use siumai_core::{
        ContentPart, LanguageRequest, Message, MessagePart, MessageRole, ProviderAnnotationError,
    };

    use super::*;

    fn annotated_part(text: impl Into<String>, options: OpenAiContentOptions) -> MessagePart {
        MessagePart::text(text)
            .with_provider_annotation(&options)
            .expect("annotation")
    }

    #[test]
    fn every_annotated_node_is_the_same_explicit_breakpoint() {
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            (0..12).map(|index| {
                annotated_part(
                    format!("prefix-{index}"),
                    OpenAiContentOptions::prompt_cache_breakpoint(),
                )
            }),
        )]);
        for part in request.messages[0].content() {
            let resolved = OpenAiAnnotationResolver
                .resolve_content(part.annotations())
                .expect("breakpoint annotation");
            assert!(resolved.explicit_breakpoint());
        }
    }

    #[test]
    fn one_content_node_cannot_carry_duplicate_openai_annotations() {
        let part = MessagePart::new(ContentPart::Text {
            text: "prefix".to_string(),
        })
        .with_provider_annotation(&OpenAiContentOptions::prompt_cache_breakpoint())
        .expect("first annotation");
        let error = part
            .with_provider_annotation(&OpenAiContentOptions::prompt_cache_breakpoint())
            .expect_err("duplicate namespace");
        assert!(matches!(
            error,
            ProviderAnnotationError::DuplicateNamespace { .. }
        ));
    }
}
