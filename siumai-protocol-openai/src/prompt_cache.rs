use siumai_core::{ContentAnnotations, Error};

/// Provider-owned prompt-cache intent projected onto one semantic content node.
///
/// The protocol codec deliberately does not interpret annotation payloads. A
/// configured provider validates its own annotation namespace and supplies this
/// small, wire-focused projection instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptCacheNodeOptions {
    explicit_breakpoint: bool,
}

impl PromptCacheNodeOptions {
    pub const fn new() -> Self {
        Self {
            explicit_breakpoint: false,
        }
    }

    pub const fn with_explicit_breakpoint(mut self, explicit_breakpoint: bool) -> Self {
        self.explicit_breakpoint = explicit_breakpoint;
        self
    }

    pub const fn explicit_breakpoint(self) -> bool {
        self.explicit_breakpoint
    }
}

/// Resolves provider-owned content annotations into protocol cache intent.
///
/// Implementations must validate only the provider namespace and API mode they
/// own. The protocol crate invokes the resolver for every
/// [`siumai_core::MessagePart`]
/// before deciding whether that node can be represented as a wire content
/// block.
pub trait PromptCacheAnnotationResolver: Send + Sync {
    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<PromptCacheNodeOptions, Error>;
}

/// Resolver used by the provider-neutral request-encoding entry points.
///
/// It intentionally ignores annotations so existing protocol-only callers keep
/// the historical behavior. Provider-facing entry points should pass their
/// validated resolver explicitly.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPromptCacheAnnotations;

impl PromptCacheAnnotationResolver for NoPromptCacheAnnotations {
    fn resolve_content(
        &self,
        _annotations: &ContentAnnotations,
    ) -> Result<PromptCacheNodeOptions, Error> {
        Ok(PromptCacheNodeOptions::default())
    }
}
