use std::mem::size_of;

#[test]
fn public_surface_unified_imports_compile() {
    use siumai::prelude::unified::*;

    let _ = size_of::<ChatRequest>();
    let _ = size_of::<ChatResponse>();
    let _ = size_of::<ProviderOptionsMap>();
    let _ = size_of::<LlmError>();
}

#[test]
fn public_surface_extensions_imports_compile() {
    use siumai::extensions::types::*;
    use siumai::extensions::*;

    let _ = size_of::<ImageEditRequest>();
    let _ = size_of::<ModerationRequest>();
    let _ = size_of::<VideoGenerationRequest>();

    let _ = size_of::<*const dyn TimeoutCapability>();
    let _ = size_of::<*const dyn ModelListingCapability>();
}

#[cfg(feature = "openai")]
#[test]
fn public_surface_openai_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::openai::{OpenAiClient, OpenAiConfig, metadata::*, options::*};

    let _ = size_of::<OpenAiClient>();
    let _ = size_of::<OpenAiConfig>();
    let _ = size_of::<OpenAiOptions>();
    let _ = size_of::<OpenAiMetadata>();
    let _ = size_of::<OpenAiSource>();

    let req = ChatRequest::new(vec![user!("hi")]).with_openai_options(OpenAiOptions::new());
    let _ = req;

    fn _assert_req_ext<T: OpenAiChatRequestExt>() {}
    fn _assert_resp_ext<T: OpenAiChatResponseExt>() {}
    _assert_req_ext::<ChatRequest>();
    _assert_resp_ext::<ChatResponse>();

    let _ = siumai::hosted_tools::openai::web_search().build();
}

#[cfg(feature = "anthropic")]
#[test]
fn public_surface_anthropic_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::anthropic::{AnthropicClient, metadata::*, options::*};

    let _ = size_of::<AnthropicClient>();
    let _ = size_of::<AnthropicOptions>();
    let _ = size_of::<AnthropicMetadata>();
    let _ = size_of::<AnthropicSource>();

    fn _assert_resp_ext<T: AnthropicChatResponseExt>() {}
    _assert_resp_ext::<ChatResponse>();

    let _ = siumai::hosted_tools::anthropic::web_search_20250305().build();
}

#[cfg(feature = "google")]
#[test]
fn public_surface_gemini_provider_ext_compiles() {
    use siumai::prelude::unified::*;
    use siumai::provider_ext::gemini::{GeminiClient, metadata::*, options::*};

    let _ = size_of::<GeminiClient>();
    let _ = size_of::<GeminiOptions>();
    let _ = size_of::<GeminiMetadata>();
    let _ = size_of::<GeminiSource>();

    fn _assert_resp_ext<T: GeminiChatResponseExt>() {}
    _assert_resp_ext::<ChatResponse>();

    let _ = siumai::hosted_tools::google::google_search().build();
}

#[cfg(feature = "groq")]
#[test]
fn public_surface_groq_provider_ext_compiles() {
    use siumai::provider_ext::groq::{GroqClient, options::*};

    let _ = size_of::<GroqClient>();
    let _ = size_of::<GroqOptions>();

    fn _assert_req_ext<T: GroqChatRequestExt>() {}
    _assert_req_ext::<siumai::prelude::unified::ChatRequest>();
}

#[cfg(feature = "xai")]
#[test]
fn public_surface_xai_provider_ext_compiles() {
    use siumai::provider_ext::xai::{XaiClient, options::*};

    let _ = size_of::<XaiClient>();
    let _ = size_of::<XaiOptions>();

    fn _assert_req_ext<T: XaiChatRequestExt>() {}
    _assert_req_ext::<siumai::prelude::unified::ChatRequest>();
}

#[cfg(feature = "ollama")]
#[test]
fn public_surface_ollama_provider_ext_compiles() {
    use siumai::provider_ext::ollama::{OllamaClient, OllamaConfig, options::*};

    let _ = size_of::<OllamaClient>();
    let _ = size_of::<OllamaConfig>();
    let _ = size_of::<OllamaOptions>();

    fn _assert_req_ext<T: OllamaChatRequestExt>() {}
    _assert_req_ext::<siumai::prelude::unified::ChatRequest>();
}

#[cfg(feature = "minimaxi")]
#[test]
fn public_surface_minimaxi_provider_ext_compiles() {
    use siumai::provider_ext::minimaxi::{
        MinimaxiClient, MinimaxiConfig, options::*, resources::*,
    };

    let _ = size_of::<MinimaxiClient>();
    let _ = size_of::<MinimaxiConfig>();
    let _ = size_of::<MinimaxiTtsOptions>();
    let _ = size_of::<MinimaxiFiles>();
}
