use serde_json::Value;
use siumai_anthropic_compatible::{
    MessagesRequestProjection, MessagesRequestProjectionContext, ProjectedMessagesRequest,
};
use siumai_core::{Error, ErrorKind};
use siumai_transport::RequestHeaders;

/// Vertex AI envelope projection for one canonical Anthropic Messages body.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GoogleVertexAnthropicProjection;

impl MessagesRequestProjection for GoogleVertexAnthropicProjection {
    fn project(
        &self,
        context: &MessagesRequestProjectionContext<'_>,
        mut body: Value,
    ) -> Result<ProjectedMessagesRequest, Error> {
        if context.beta_header().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Anthropic beta headers are not supported by Claude on Vertex AI",
            ));
        }
        let model = context.model().as_str();
        validate_model_segment(model)?;
        let object = body.as_object_mut().ok_or_else(|| {
            Error::new(
                ErrorKind::Internal,
                "canonical Anthropic Messages request body was not an object",
            )
        })?;
        if object.remove("model").is_none() {
            return Err(Error::new(
                ErrorKind::Internal,
                "canonical Anthropic Messages request omitted its model",
            ));
        }
        object.insert(
            "anthropic_version".to_string(),
            Value::String(context.api_version().to_string()),
        );
        let operation = if context.is_streaming() {
            "streamRawPredict"
        } else {
            "rawPredict"
        };
        ProjectedMessagesRequest::try_new(
            format!("models/{model}:{operation}"),
            body,
            RequestHeaders::new(),
        )
    }
}

pub(crate) fn validate_model_segment(model: &str) -> Result<(), Error> {
    if model.is_empty()
        || model.len() > 256
        || !model
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'@'))
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Vertex model ID is not a safe request-path segment",
        ));
    }
    Ok(())
}
