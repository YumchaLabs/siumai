//! Small advisory catalog for the current GPT-5.5 and GPT-5.6 families.
//!
//! Model IDs remain open. These constants drive policy advice only and never
//! form an execution allowlist.

/// Rolling GPT-5.5 model ID.
pub const GPT_5_5: &str = "gpt-5.5";
/// GPT-5.5 Pro model ID.
pub const GPT_5_5_PRO: &str = "gpt-5.5-pro";
/// Rolling GPT-5.6 alias. OpenAI currently routes it to GPT-5.6 Sol.
pub const GPT_5_6: &str = "gpt-5.6";
/// Frontier-capability GPT-5.6 tier.
pub const GPT_5_6_SOL: &str = "gpt-5.6-sol";
/// Balanced GPT-5.6 tier.
pub const GPT_5_6_TERRA: &str = "gpt-5.6-terra";
/// Efficient high-volume GPT-5.6 tier.
pub const GPT_5_6_LUNA: &str = "gpt-5.6-luna";

/// Provider-owned advisory classification for current OpenAI model policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpenAiModelClass {
    Gpt55,
    Gpt55Pro,
    Gpt56Alias,
    Gpt56Sol,
    Gpt56Terra,
    Gpt56Luna,
    Unknown,
}

impl OpenAiModelClass {
    pub const fn is_gpt_5_5(self) -> bool {
        matches!(self, Self::Gpt55 | Self::Gpt55Pro)
    }

    pub const fn is_gpt_5_6(self) -> bool {
        matches!(
            self,
            Self::Gpt56Alias | Self::Gpt56Sol | Self::Gpt56Terra | Self::Gpt56Luna
        )
    }
}

/// Classify exact current model IDs without blocking future model IDs.
pub fn classify_model(model: &str) -> OpenAiModelClass {
    match model {
        GPT_5_5 => OpenAiModelClass::Gpt55,
        GPT_5_5_PRO => OpenAiModelClass::Gpt55Pro,
        GPT_5_6 => OpenAiModelClass::Gpt56Alias,
        GPT_5_6_SOL => OpenAiModelClass::Gpt56Sol,
        GPT_5_6_TERRA => OpenAiModelClass::Gpt56Terra,
        GPT_5_6_LUNA => OpenAiModelClass::Gpt56Luna,
        _ => OpenAiModelClass::Unknown,
    }
}
