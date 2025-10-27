//! Centralized list of OpenAI-compatible providers for generating builder methods.
//!
//! This macro enumerates all supported OpenAI-compatible providers. It accepts
//! a callback macro `$mac` and invokes it as `$mac!(method_name, provider_id)`
//! for each provider entry. Use it to generate methods on different builders
//! without duplicating the list of providers.
//!
//! Example:
//!
//! ```ignore
//! macro_rules! gen_llmbuilder_method {
//!     ($name:ident, $id:expr) => {
//!         #[cfg(feature = "openai")]
//!         pub fn $name(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
//!             crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, $id)
//!         }
//!     };
//! }
//! siumai_for_each_openai_compatible_provider!(gen_llmbuilder_method);
//! ```
#![allow(unused_macros)]

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
        $mac!(groq_openai_compatible, "groq_openai_compatible");
        $mac!(xai_openai_compatible, "xai_openai_compatible");
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
        // Extended providers seen in SiumaiBuilder convenience
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
