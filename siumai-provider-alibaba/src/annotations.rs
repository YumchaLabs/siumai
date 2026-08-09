use serde::{Deserialize, Serialize};
use siumai_core::{
    ContentAnnotationTarget, ContentAnnotations, Error, ErrorKind, MessageAnnotationTarget,
    MessageAnnotations, ProviderAnnotationError, ToolAnnotationTarget, ToolAnnotations,
    TypedProviderAnnotation,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, CacheControl, CacheTtl, ContentNodeOptions, MessageNodeOptions,
    MessagesAnnotationResolver, MessagesCodecError, ToolNodeOptions,
};
use siumai_protocol_openai::{PromptCacheAnnotationResolver, PromptCacheNodeOptions};

macro_rules! cache_marker {
    ($name:ident, $target:ty, $api_mode:expr) => {
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name {}

        impl $name {
            pub const fn new() -> Self {
                Self {}
            }
        }

        impl TypedProviderAnnotation for $name {
            type Target = $target;

            const NAMESPACE: &'static str = "alibaba";
            const API_MODE: Option<&'static str> = $api_mode;
        }
    };
}

cache_marker!(
    AlibabaMessageCache,
    MessageAnnotationTarget,
    Some(API_MODE_ID)
);
cache_marker!(AlibabaContentCache, ContentAnnotationTarget, None);
cache_marker!(AlibabaToolCache, ToolAnnotationTarget, Some(API_MODE_ID));

/// Project Alibaba's fixed ephemeral cache markers into Messages wire controls.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AlibabaAnnotationResolver;

impl MessagesAnnotationResolver for AlibabaAnnotationResolver {
    fn resolve_message(
        &self,
        annotations: &MessageAnnotations,
    ) -> Result<MessageNodeOptions, MessagesCodecError> {
        resolve::<AlibabaMessageCache, _>(annotations, "message").map(|marker| {
            marker.map_or_else(MessageNodeOptions::default, |_| {
                MessageNodeOptions::default().with_cache_control(fixed_cache())
            })
        })
    }

    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<ContentNodeOptions, MessagesCodecError> {
        resolve::<AlibabaContentCache, _>(annotations, "content").map(|marker| {
            marker.map_or_else(ContentNodeOptions::default, |_| {
                ContentNodeOptions::default().with_cache_control(fixed_cache())
            })
        })
    }

    fn resolve_tool(
        &self,
        annotations: &ToolAnnotations,
    ) -> Result<ToolNodeOptions, MessagesCodecError> {
        resolve::<AlibabaToolCache, _>(annotations, "tool").map(|marker| {
            marker.map_or_else(ToolNodeOptions::default, |_| {
                ToolNodeOptions::default().with_cache_control(fixed_cache())
            })
        })
    }
}

/// Project Alibaba content cache annotations into its OpenAI-shaped Chat wire.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AlibabaChatAnnotationResolver;

impl PromptCacheAnnotationResolver for AlibabaChatAnnotationResolver {
    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<PromptCacheNodeOptions, Error> {
        annotations
            .decode::<AlibabaContentCache>()
            .map(|marker| PromptCacheNodeOptions::new().with_explicit_breakpoint(marker.is_some()))
            .map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "invalid Alibaba content annotation",
                )
                .with_source(source)
            })
    }
}

fn resolve<T, Target>(
    annotations: &siumai_core::ProviderAnnotations<Target>,
    node: &'static str,
) -> Result<Option<T>, MessagesCodecError>
where
    T: serde::de::DeserializeOwned + TypedProviderAnnotation<Target = Target>,
    Target: siumai_core::ProviderAnnotationTarget,
{
    annotations
        .decode::<T>()
        .map_err(|source| invalid_annotation(node, source))
}

fn invalid_annotation(node: &'static str, source: ProviderAnnotationError) -> MessagesCodecError {
    MessagesCodecError::InvalidAnnotation { node, source }
}

const fn fixed_cache() -> CacheControl {
    CacheControl::new(CacheTtl::FiveMinutes)
}
