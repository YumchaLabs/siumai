use siumai::providers::openai::{OpenAiConfigError, OpenAiCredential, OpenAiProvider};
use siumai::transport::{HttpTransportRoute, ProviderHttpTransportSettings, ProxyEndpoint};

fn build_openai_provider(
    credential: OpenAiCredential,
    settings: ProviderHttpTransportSettings,
) -> Result<OpenAiProvider, OpenAiConfigError> {
    OpenAiProvider::builder(credential)
        .with_http_transport_settings(settings)
        .build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let route =
        HttpTransportRoute::trusted_connect(ProxyEndpoint::https("https://proxy.example.com")?);
    let settings = ProviderHttpTransportSettings::default().with_route(route)?;

    assert!(settings.route().proxy().is_some());

    let _provider_factory: fn(
        OpenAiCredential,
        ProviderHttpTransportSettings,
    ) -> Result<OpenAiProvider, OpenAiConfigError> = build_openai_provider;

    Ok(())
}
