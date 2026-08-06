use serde::{Deserialize, Serialize};
use siumai_core::{
    ContentAnnotationTarget, ContentAnnotations, MessageAnnotationTarget, MessageAnnotations,
    ProviderAnnotationError, ToolAnnotationTarget, ToolAnnotations, TypedProviderAnnotation,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, CacheControl, CacheTtl, ContentNodeOptions, MessageNodeOptions,
    MessagesAnnotationResolver, MessagesCodecError, ToolNodeOptions,
};

macro_rules! cache_marker {
    ($name:ident, $target:ty) => {
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

            const NAMESPACE: &'static str = "minimax";
            const API_MODE: Option<&'static str> = Some(API_MODE_ID);
        }
    };
}

cache_marker!(MinimaxMessageCache, MessageAnnotationTarget);
cache_marker!(MinimaxContentCache, ContentAnnotationTarget);
cache_marker!(MinimaxToolCache, ToolAnnotationTarget);

/// Strict projection from typed MiniMax cache markers to Messages wire controls.
///
/// MiniMax currently exposes only the fixed ephemeral cache marker. The
/// compatible encoder's `FiveMinutesImplicit` dialect rule therefore emits
/// `{ "type": "ephemeral" }` and never invents an Anthropic TTL extension.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinimaxAnnotationResolver;

impl MessagesAnnotationResolver for MinimaxAnnotationResolver {
    fn resolve_message(
        &self,
        annotations: &MessageAnnotations,
    ) -> Result<MessageNodeOptions, MessagesCodecError> {
        resolve::<MinimaxMessageCache, _>(annotations, "message").map(|marker| {
            marker.map_or_else(MessageNodeOptions::default, |_| {
                MessageNodeOptions::default().with_cache_control(fixed_cache())
            })
        })
    }

    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<ContentNodeOptions, MessagesCodecError> {
        resolve::<MinimaxContentCache, _>(annotations, "content").map(|marker| {
            marker.map_or_else(ContentNodeOptions::default, |_| {
                ContentNodeOptions::default().with_cache_control(fixed_cache())
            })
        })
    }

    fn resolve_tool(
        &self,
        annotations: &ToolAnnotations,
    ) -> Result<ToolNodeOptions, MessagesCodecError> {
        resolve::<MinimaxToolCache, _>(annotations, "tool").map(|marker| {
            marker.map_or_else(ToolNodeOptions::default, |_| {
                ToolNodeOptions::default().with_cache_control(fixed_cache())
            })
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
