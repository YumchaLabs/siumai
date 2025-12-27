//! Provider-side macros.

/// Enumerates all supported OpenAI-compatible providers.
///
/// This macro is defined at the crate root so it is always available
/// regardless of feature flags. It accepts a callback macro `$mac` and
/// invokes it as `$mac!(method_name, provider_id)` for each provider.
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
