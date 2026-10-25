use siumai::providers::openai_compatible::{
    OpenAiCompatibleApiMode, OpenAiCompatibleProfile, OpenAiCompatibleProvider,
};
use siumai::{ReplayDomainId, Siumai};

fn local_provider() -> Result<Siumai<OpenAiCompatibleProvider>, Box<dyn std::error::Error>> {
    // Unreleased: trust this developer-controlled origin, not unchecked user URLs.
    // Prefer HTTPS when available; HTTP sends all request and response data in cleartext.
    let profile = OpenAiCompatibleProfile::custom(
        siumai::ProviderId::new("self-hosted")?,
        "http://127.0.0.1:11434/v1",
        ReplayDomainId::new("self-hosted")?,
        OpenAiCompatibleApiMode::ChatCompletions,
    )?;
    Ok(Siumai::builder()
        .openai_compatible()
        .profile(profile)
        .unauthenticated()
        .build()?)
}

async fn generate() -> Result<(), Box<dyn std::error::Error>> {
    let ai = local_provider()?;
    // Replace this ID with a model served by your local endpoint.
    let client = ai.language("your-served-model")?;
    let response = client.generate("Hello from Siumai.").await?;
    if let Some(text) = response.output_text() {
        println!("{text}");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = local_provider()?;
    let _client = ai.language("your-served-model")?;
    // Compile the request path without requiring a running local server.
    drop(generate());
    Ok(())
}
