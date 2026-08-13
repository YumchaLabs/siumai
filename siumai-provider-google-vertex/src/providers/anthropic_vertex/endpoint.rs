use siumai_transport::{EndpointConfig, EndpointError, OfficialOrigin};
use thiserror::Error;

const MAX_PROJECT_BYTES: usize = 256;
const MAX_LOCATION_BYTES: usize = 63;

pub(crate) fn official_endpoint(
    project: &str,
    location: &str,
) -> Result<EndpointConfig, GoogleVertexAnthropicEndpointError> {
    validate_project(project)?;
    validate_location(location)?;
    let host = endpoint_host(location);
    let origin = format!("https://{host}");
    let base_url =
        format!("{origin}/v1/projects/{project}/locations/{location}/publishers/anthropic/");
    Ok(EndpointConfig::official(
        base_url,
        OfficialOrigin::new(origin)?,
    )?)
}

pub(crate) fn endpoint_host(location: &str) -> String {
    match location {
        "global" => "aiplatform.googleapis.com".to_string(),
        "eu" => "aiplatform.eu.rep.googleapis.com".to_string(),
        "us" => "aiplatform.us.rep.googleapis.com".to_string(),
        _ => format!("{location}-aiplatform.googleapis.com"),
    }
}

fn validate_project(project: &str) -> Result<(), GoogleVertexAnthropicEndpointError> {
    if project.is_empty()
        || project.len() > MAX_PROJECT_BYTES
        || !project
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~'))
    {
        return Err(GoogleVertexAnthropicEndpointError::InvalidProject);
    }
    Ok(())
}

fn validate_location(location: &str) -> Result<(), GoogleVertexAnthropicEndpointError> {
    if location.is_empty()
        || location.len() > MAX_LOCATION_BYTES
        || !location
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || !location
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        || !location
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
    {
        return Err(GoogleVertexAnthropicEndpointError::InvalidLocation);
    }
    Ok(())
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GoogleVertexAnthropicEndpointError {
    #[error("Google Cloud project must be one safe, bounded path segment")]
    InvalidProject,
    #[error("Vertex location must be one bounded lowercase DNS label")]
    InvalidLocation,
    #[error("invalid Anthropic-on-Vertex endpoint: {0}")]
    Endpoint(#[from] EndpointError),
}
