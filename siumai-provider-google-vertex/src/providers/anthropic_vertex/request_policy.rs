use siumai_anthropic_compatible::{
    MessagesCallOptions, MessagesRequestPolicy, MessagesRequestRequirements,
};
use siumai_core::{
    ContentPart, Error, ErrorKind, LanguageRequest, MediaData, MessageRole, ModelId,
};
use siumai_protocol_anthropic::messages::{OutputEffort, ThinkingConfig};

use super::models::{
    CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5_20251001, CLAUDE_OPUS_4_1_20250805, CLAUDE_OPUS_4_5_20251101,
    CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7, CLAUDE_OPUS_4_8, CLAUDE_OPUS_4_20250514, CLAUDE_OPUS_5,
    CLAUDE_SONNET_4_5_20250929, CLAUDE_SONNET_4_6, CLAUDE_SONNET_4_20250514, CLAUDE_SONNET_5,
    uses_strict_sampling,
};

const MAX_OUTPUT_TOKENS_128K: u64 = 128_000;
const MAX_OUTPUT_TOKENS_64K: u64 = 64_000;

/// Structural and exact-model request restrictions for Claude models served through Vertex AI.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GoogleVertexAnthropicRequestPolicy;

impl MessagesRequestPolicy for GoogleVertexAnthropicRequestPolicy {
    fn prepare(
        &self,
        model: &ModelId,
        request: &LanguageRequest,
        options: &mut MessagesCallOptions,
    ) -> Result<MessagesRequestRequirements, Error> {
        validate_model_request(model, request, options)?;
        reject_mid_conversation_system_messages(request)?;
        reject_url_media(request)?;
        reject_unverified_options(options)?;

        if request
            .structured_output
            .as_ref()
            .is_some_and(|output| !output.strict)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Claude on Vertex AI requires strict structured output",
            ));
        }
        Ok(MessagesRequestRequirements::new())
    }
}

fn validate_model_request(
    model: &ModelId,
    request: &LanguageRequest,
    options: &MessagesCallOptions,
) -> Result<(), Error> {
    if uses_strict_sampling(model.as_str()) {
        if options.top_k().is_some() {
            return Err(invalid("Claude 4.7 and later do not accept top_k sampling"));
        }
        if request
            .generation
            .temperature
            .is_some_and(|temperature| temperature != 1.0)
        {
            return Err(invalid(
                "Claude 4.7 and later accept only the default temperature value of 1",
            ));
        }
        if request.generation.top_p.is_some_and(|top_p| top_p < 0.99) {
            return Err(invalid(
                "Claude 4.7 and later accept only the default top_p range of 0.99 to 1",
            ));
        }
    }

    validate_model_options(model.as_str(), options.thinking(), options.output_effort())?;
    validate_output_limit(model.as_str(), request.generation.max_output_tokens)
}

fn validate_model_options(
    model: &str,
    thinking: Option<ThinkingConfig>,
    effort: Option<OutputEffort>,
) -> Result<(), Error> {
    match model {
        CLAUDE_OPUS_5 => {
            reject_manual_thinking(thinking)?;
            if thinking == Some(ThinkingConfig::Disabled)
                && matches!(effort, Some(OutputEffort::XHigh | OutputEffort::Max))
            {
                return Err(invalid(
                    "Claude Opus 5 cannot disable thinking at xhigh or max effort",
                ));
            }
        }
        CLAUDE_SONNET_5 | CLAUDE_OPUS_4_8 | CLAUDE_OPUS_4_7 => {
            reject_manual_thinking(thinking)?;
        }
        CLAUDE_FABLE_5 => {
            if matches!(
                thinking,
                Some(ThinkingConfig::Disabled | ThinkingConfig::Enabled { .. })
            ) {
                return Err(invalid(
                    "Claude Fable 5 requires adaptive thinking and cannot use disabled or manual thinking",
                ));
            }
        }
        CLAUDE_HAIKU_4_5_20251001
        | CLAUDE_SONNET_4_5_20250929
        | CLAUDE_OPUS_4_1_20250805
        | CLAUDE_OPUS_4_20250514
        | CLAUDE_SONNET_4_20250514 => {
            if matches!(thinking, Some(ThinkingConfig::Adaptive { .. })) {
                return Err(invalid(
                    "adaptive thinking is not verified for this Vertex Claude model",
                ));
            }
            if effort.is_some() {
                return Err(invalid(
                    "output effort is not supported by this Vertex Claude model",
                ));
            }
        }
        CLAUDE_OPUS_4_5_20251101 => {
            if matches!(thinking, Some(ThinkingConfig::Adaptive { .. })) {
                return Err(invalid(
                    "adaptive thinking is not verified for Claude Opus 4.5 on Vertex AI",
                ));
            }
            if matches!(effort, Some(OutputEffort::XHigh | OutputEffort::Max)) {
                return Err(invalid(
                    "Claude Opus 4.5 supports low, medium, or high effort",
                ));
            }
        }
        CLAUDE_OPUS_4_6 | CLAUDE_SONNET_4_6 => {
            reject_xhigh_effort(effort)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_output_limit(model: &str, requested: Option<u64>) -> Result<(), Error> {
    let maximum = match model {
        CLAUDE_OPUS_5 | CLAUDE_SONNET_5 | CLAUDE_FABLE_5 | CLAUDE_OPUS_4_8 | CLAUDE_OPUS_4_7
        | CLAUDE_OPUS_4_6 | CLAUDE_SONNET_4_6 => Some(MAX_OUTPUT_TOKENS_128K),
        CLAUDE_HAIKU_4_5_20251001 => Some(MAX_OUTPUT_TOKENS_64K),
        _ => None,
    };
    if requested
        .zip(maximum)
        .is_some_and(|(value, max)| value > max)
    {
        return Err(invalid(
            "max_output_tokens exceeds the verified Vertex Claude model limit",
        ));
    }
    Ok(())
}

fn reject_mid_conversation_system_messages(request: &LanguageRequest) -> Result<(), Error> {
    let mut conversation_started = false;
    for message in &request.messages {
        match message.role() {
            MessageRole::System if conversation_started => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "mid-conversation system messages are not verified for Claude on Vertex AI",
                ));
            }
            MessageRole::User | MessageRole::Assistant | MessageRole::Tool => {
                conversation_started = true;
            }
            _ => {}
        }
    }
    Ok(())
}

fn reject_url_media(request: &LanguageRequest) -> Result<(), Error> {
    if request.messages.iter().any(|message| {
        message.content().iter().any(|part| {
            matches!(
                part.content(),
                ContentPart::Media(media) if matches!(&media.data, MediaData::Url(_))
            )
        })
    }) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "Claude on Vertex AI does not accept URL media sources",
        ));
    }
    Ok(())
}

fn reject_unverified_options(options: &MessagesCallOptions) -> Result<(), Error> {
    if options.fallbacks().is_some() {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "Anthropic server-side fallbacks are not supported by Vertex AI",
        ));
    }
    Ok(())
}

fn reject_manual_thinking(thinking: Option<ThinkingConfig>) -> Result<(), Error> {
    if matches!(thinking, Some(ThinkingConfig::Enabled { .. })) {
        return Err(invalid(
            "this model accepts adaptive or disabled thinking, not legacy manual thinking",
        ));
    }
    Ok(())
}

fn reject_xhigh_effort(effort: Option<OutputEffort>) -> Result<(), Error> {
    if effort == Some(OutputEffort::XHigh) {
        return Err(invalid(
            "Claude 4.6 supports low, medium, high, or max effort, not xhigh",
        ));
    }
    Ok(())
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
