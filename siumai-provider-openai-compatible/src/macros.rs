/// Enumerates all supported OpenAI-compatible provider ids.
///
/// This macro is used to generate convenience builder methods (e.g. `.openrouter()`)
/// without duplicating the provider list across crates.
///
/// Note: these are vendor presets for the OpenAI-compatible adapter surface
/// (`providers::openai_compatible`), not native providers.
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
