use siumai::language;
use siumai::providers::openai::models::GPT_5_6;
use siumai::registry::{Registry, RegistryBuilderExt};
use siumai::{LanguageCallError, LanguageModel, LanguageResponse, Model, Siumai};

async fn application_call<M>(model: &M) -> Result<LanguageResponse, LanguageCallError>
where
    M: LanguageModel + ?Sized,
{
    language::generate(model, "Hello").await
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = Siumai::builder()
        .openai()
        .api_key("example-openai-key")
        .build()?;
    let direct = ai.language(GPT_5_6)?;

    let mut registry = Registry::builder();
    registry.register_provider("production", ai.provider())?;
    let registry = registry.build()?;
    let routed = registry.language_model(format!("production:{GPT_5_6}"))?;

    assert_eq!(direct.descriptor().scope(), routed.descriptor().scope());
    assert_eq!(
        direct.descriptor().instance_id(),
        routed.descriptor().instance_id()
    );
    assert_eq!(
        routed.route_id().map(siumai::core::RouteId::as_str),
        Some("production")
    );

    // The same generic function accepts a typed facade client and an explicitly
    // erased Registry model. Dropping both futures keeps the example offline.
    drop(application_call(&direct));
    drop(application_call(routed.as_ref()));

    Ok(())
}
