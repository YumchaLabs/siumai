//! siumai-core
//!
//! Provider-agnostic runtime, types, and shared execution primitives.
#![deny(unsafe_code)]

pub mod auth;
pub mod builder;
pub mod client;
pub mod core;
pub mod custom_provider;
pub mod defaults;
pub mod error;
pub mod execution;
pub mod hosted_tools;
pub mod observability;
pub mod params;
pub mod retry;
pub mod retry_api;
pub mod standards;
pub mod streaming;
pub mod tools;
pub mod traits;
pub mod types;
pub mod utils;

pub use error::LlmError;

/// Enumerates all supported OpenAI-compatible provider ids.
///
/// This list is intentionally hosted in `siumai-core` so that both the facade
/// (`siumai`) and the registry (`siumai-registry`) can generate vendor preset
/// helpers without depending on an umbrella crate.
#[macro_export]
macro_rules! siumai_for_each_openai_compatible_provider {
    ($mac:ident) => {
        $mac!(deepseek, "deepseek");
        $mac!(openrouter, "openrouter");
        $mac!(siliconflow, "siliconflow");
        $mac!(together, "together");
        $mac!(fireworks, "fireworks");
        $mac!(github_copilot, "github_copilot");
        $mac!(perplexity, "perplexity");
        $mac!(mistral, "mistral");
        $mac!(cohere, "cohere");
        $mac!(zhipu, "zhipu");
        $mac!(moonshot, "moonshot");
        $mac!(yi, "yi");
        $mac!(doubao, "doubao");
        $mac!(baichuan, "baichuan");
        $mac!(qwen, "qwen");
        // OpenAI-compatible variants of native providers
        $mac!(groq_openai_compatible, "groq");
        $mac!(xai_openai_compatible, "xai");
        // International providers
        $mac!(nvidia, "nvidia");
        $mac!(hyperbolic, "hyperbolic");
        $mac!(jina, "jina");
        $mac!(github, "github");
        $mac!(voyageai, "voyageai");
        $mac!(poe, "poe");
        // Chinese providers
        $mac!(stepfun, "stepfun");
        $mac!(minimax, "minimax");
        $mac!(infini, "infini");
        $mac!(modelscope, "modelscope");
        // Extended providers seen in vendor presets
        $mac!(hunyuan, "hunyuan");
        $mac!(baidu_cloud, "baidu_cloud");
        $mac!(tencent_cloud_ti, "tencent_cloud_ti");
        $mac!(xirang, "xirang");
        $mac!(ai302, "302ai");
        $mac!(aihubmix, "aihubmix");
        $mac!(ppio, "ppio");
        $mac!(ocoolai, "ocoolai");
    };
}
