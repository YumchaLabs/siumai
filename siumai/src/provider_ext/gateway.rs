pub use siumai_provider_gateway::providers::gateway::{
    GatewayBuilder, GatewayClient, GatewayConfig,
};

/// Create the Vercel AI Gateway provider builder.
pub fn gateway() -> GatewayBuilder {
    crate::compat::Provider::gateway()
}

/// Create the Vercel AI Gateway provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createGateway()`.
pub fn create_gateway() -> GatewayBuilder {
    gateway()
}

/// Typed provider options (`provider_options_map["gateway"]`).
pub mod options {
    pub use siumai_provider_gateway::provider_options::{
        GatewayEmbeddingModelOptions, GatewayLanguageModelOptions, GatewayOptions,
        GatewayProviderTimeouts, GatewayServiceTier, GatewaySort,
    };
    pub use siumai_provider_gateway::providers::gateway::{
        GatewayChatRequestExt, GatewayEmbeddingRequestExt,
    };
}

pub use options::{
    GatewayChatRequestExt, GatewayEmbeddingModelOptions, GatewayEmbeddingRequestExt,
    GatewayLanguageModelOptions, GatewayOptions, GatewayProviderTimeouts, GatewayServiceTier,
    GatewaySort,
};
