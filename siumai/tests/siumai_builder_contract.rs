use siumai::Siumai;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "openai-compatible"
))]
fn accepts_language_model<M: siumai::LanguageModel + ?Sized>(model: &M) {
    let _ = model;
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "openai-compatible",
    feature = "alibaba",
    feature = "moonshotai",
    feature = "volcengine",
    feature = "groq",
    feature = "xai",
    feature = "minimax",
    feature = "deepseek",
    feature = "cohere",
    feature = "deepgram",
    feature = "google-vertex-anthropic",
    feature = "elevenlabs"
))]
fn error_diagnostics(error: &(dyn std::error::Error + 'static)) -> String {
    let mut output = String::new();
    let mut current = Some(error);
    while let Some(error) = current {
        output.push_str(&format!("{error:?}\n{error}\n"));
        current = error.source();
    }
    output
}

#[test]
fn zero_state_builder_is_available_without_default_features() {
    let _builder = Siumai::builder();
}

#[path = "siumai_builder_contract/generic_hub.rs"]
mod generic_hub;

#[cfg(feature = "openai")]
#[path = "siumai_builder_contract/openai.rs"]
mod openai;

#[cfg(feature = "anthropic")]
#[path = "siumai_builder_contract/anthropic.rs"]
mod anthropic;

#[cfg(feature = "google")]
#[path = "siumai_builder_contract/gemini.rs"]
mod gemini;

#[cfg(feature = "openai-compatible")]
#[path = "siumai_builder_contract/openai_compatible.rs"]
mod openai_compatible;

#[cfg(feature = "alibaba")]
#[path = "siumai_builder_contract/alibaba.rs"]
mod alibaba;

#[cfg(feature = "moonshotai")]
#[path = "siumai_builder_contract/moonshot.rs"]
mod moonshot;

#[cfg(feature = "volcengine")]
#[path = "siumai_builder_contract/volcengine.rs"]
mod volcengine;

#[cfg(feature = "groq")]
#[path = "siumai_builder_contract/groq.rs"]
mod groq;

#[cfg(feature = "xai")]
#[path = "siumai_builder_contract/xai.rs"]
mod xai;

#[cfg(feature = "minimax")]
#[path = "siumai_builder_contract/minimax.rs"]
mod minimax;

#[cfg(feature = "deepseek")]
#[path = "siumai_builder_contract/deepseek.rs"]
mod deepseek;

#[cfg(feature = "cohere")]
#[path = "siumai_builder_contract/cohere.rs"]
mod cohere;

#[cfg(feature = "deepgram")]
#[path = "siumai_builder_contract/deepgram.rs"]
mod deepgram;

#[cfg(feature = "google-vertex-anthropic")]
#[path = "siumai_builder_contract/vertex_anthropic.rs"]
mod vertex_anthropic;

#[cfg(feature = "elevenlabs")]
#[path = "siumai_builder_contract/elevenlabs.rs"]
mod elevenlabs;

#[cfg(all(feature = "groq", feature = "openai-compatible"))]
#[path = "siumai_builder_contract/branded_compatible_options.rs"]
mod branded_compatible_options;
