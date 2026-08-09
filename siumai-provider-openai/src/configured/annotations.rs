use serde::{Deserialize, Serialize};
use siumai_core::{
    ContentAnnotationTarget, ContentAnnotations, Error, ErrorKind, LanguageRequest,
    ProviderAnnotationError, TypedProviderAnnotation,
};
use siumai_protocol_openai::{PromptCacheAnnotationResolver, PromptCacheNodeOptions};
use thiserror::Error;

use super::options::OpenAiPromptCacheMode;

/// The durable role of one OpenAI prompt-cache marker.
///
/// Historical markers describe prefixes that may already exist in the provider
/// cache. Write candidates are the current request's explicit write budget. The
/// OpenAI wire uses the same block marker for both roles, so Siumai preserves the
/// distinction in the typed request node and validates it before encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiPromptCacheMarker {
    Historical,
    WriteCandidate,
}

/// OpenAI controls attached to one canonical message-content node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiContentOptions {
    prompt_cache: OpenAiPromptCacheMarker,
}

impl OpenAiContentOptions {
    /// Mark a prefix that may already be present in OpenAI's cache history.
    pub const fn historical_cache_marker() -> Self {
        Self {
            prompt_cache: OpenAiPromptCacheMarker::Historical,
        }
    }

    /// Spend one explicit write-candidate slot on this content block.
    pub const fn cache_write_candidate() -> Self {
        Self {
            prompt_cache: OpenAiPromptCacheMarker::WriteCandidate,
        }
    }

    pub const fn prompt_cache_marker(self) -> OpenAiPromptCacheMarker {
        self.prompt_cache
    }
}

impl TypedProviderAnnotation for OpenAiContentOptions {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "openai";
}

/// Invalid OpenAI node-scoped request controls.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OpenAiAnnotationError {
    #[error("invalid OpenAI content annotation: {0}")]
    Annotation(#[from] ProviderAnnotationError),
    #[error(
        "OpenAI historical prompt-cache markers must precede current write candidates; marker at message {message_index}, content {content_index} is out of order"
    )]
    HistoricalMarkerAfterWrite {
        message_index: usize,
        content_index: usize,
    },
    #[error(
        "OpenAI prompt-cache mode allows at most {maximum} explicit write candidates, but the request contains {actual}"
    )]
    TooManyWriteCandidates { actual: usize, maximum: usize },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct OpenAiPromptCacheSummary {
    marker_count: usize,
    write_candidate_count: usize,
}

impl OpenAiPromptCacheSummary {
    pub(crate) const fn has_markers(self) -> bool {
        self.marker_count != 0
    }

    #[cfg(test)]
    pub(crate) const fn marker_count(self) -> usize {
        self.marker_count
    }

    #[cfg(test)]
    pub(crate) const fn write_candidate_count(self) -> usize {
        self.write_candidate_count
    }
}

pub(crate) fn validate_prompt_cache_annotations(
    request: &LanguageRequest,
    mode: OpenAiPromptCacheMode,
) -> Result<OpenAiPromptCacheSummary, OpenAiAnnotationError> {
    let maximum_writes = match mode {
        OpenAiPromptCacheMode::Implicit => 3,
        OpenAiPromptCacheMode::Explicit => 4,
    };
    let mut summary = OpenAiPromptCacheSummary::default();
    let mut saw_write_candidate = false;

    for (message_index, message) in request.messages.iter().enumerate() {
        for (content_index, part) in message.content().iter().enumerate() {
            let Some(options) = part.annotations().decode::<OpenAiContentOptions>()? else {
                continue;
            };
            summary.marker_count = summary.marker_count.saturating_add(1);
            match options.prompt_cache_marker() {
                OpenAiPromptCacheMarker::Historical if saw_write_candidate => {
                    return Err(OpenAiAnnotationError::HistoricalMarkerAfterWrite {
                        message_index,
                        content_index,
                    });
                }
                OpenAiPromptCacheMarker::Historical => {}
                OpenAiPromptCacheMarker::WriteCandidate => {
                    saw_write_candidate = true;
                    summary.write_candidate_count = summary.write_candidate_count.saturating_add(1);
                    if summary.write_candidate_count > maximum_writes {
                        return Err(OpenAiAnnotationError::TooManyWriteCandidates {
                            actual: summary.write_candidate_count,
                            maximum: maximum_writes,
                        });
                    }
                }
            }
        }
    }

    Ok(summary)
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
    use siumai_core::{ContentPart, Message, MessagePart, MessageRole};

    use super::*;

    fn annotated_part(text: impl Into<String>, options: OpenAiContentOptions) -> MessagePart {
        MessagePart::text(text)
            .with_provider_annotation(&options)
            .expect("annotation")
    }

    #[test]
    fn implicit_and_explicit_modes_enforce_only_current_write_budget() {
        let historical = (0..90).map(|index| {
            annotated_part(
                format!("history-{index}"),
                OpenAiContentOptions::historical_cache_marker(),
            )
        });
        let writes = (0..3).map(|index| {
            annotated_part(
                format!("write-{index}"),
                OpenAiContentOptions::cache_write_candidate(),
            )
        });
        let implicit = LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            historical.chain(writes),
        )]);
        let summary = validate_prompt_cache_annotations(&implicit, OpenAiPromptCacheMode::Implicit)
            .expect("implicit markers");
        assert_eq!(summary.marker_count(), 93);
        assert_eq!(summary.write_candidate_count(), 3);

        let four_writes = LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            (0..4).map(|index| {
                annotated_part(
                    format!("write-{index}"),
                    OpenAiContentOptions::cache_write_candidate(),
                )
            }),
        )]);
        assert!(matches!(
            validate_prompt_cache_annotations(&four_writes, OpenAiPromptCacheMode::Implicit),
            Err(OpenAiAnnotationError::TooManyWriteCandidates {
                actual: 4,
                maximum: 3
            })
        ));
        validate_prompt_cache_annotations(&four_writes, OpenAiPromptCacheMode::Explicit)
            .expect("four explicit writes");
    }

    #[test]
    fn fifth_explicit_write_and_interleaved_history_fail_closed() {
        let five_writes = LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            (0..5).map(|index| {
                annotated_part(
                    format!("write-{index}"),
                    OpenAiContentOptions::cache_write_candidate(),
                )
            }),
        )]);
        assert!(matches!(
            validate_prompt_cache_annotations(&five_writes, OpenAiPromptCacheMode::Explicit),
            Err(OpenAiAnnotationError::TooManyWriteCandidates {
                actual: 5,
                maximum: 4
            })
        ));

        let interleaved = LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            [
                annotated_part("write", OpenAiContentOptions::cache_write_candidate()),
                annotated_part("history", OpenAiContentOptions::historical_cache_marker()),
            ],
        )]);
        assert!(matches!(
            validate_prompt_cache_annotations(&interleaved, OpenAiPromptCacheMode::Explicit),
            Err(OpenAiAnnotationError::HistoricalMarkerAfterWrite {
                message_index: 0,
                content_index: 1
            })
        ));
    }

    #[test]
    fn one_content_node_cannot_carry_conflicting_openai_roles() {
        let part = MessagePart::new(ContentPart::Text {
            text: "prefix".to_string(),
        })
        .with_provider_annotation(&OpenAiContentOptions::historical_cache_marker())
        .expect("first annotation");
        let error = part
            .with_provider_annotation(&OpenAiContentOptions::cache_write_candidate())
            .expect_err("duplicate namespace");
        assert!(matches!(
            error,
            ProviderAnnotationError::DuplicateNamespace { .. }
        ));
    }
}
