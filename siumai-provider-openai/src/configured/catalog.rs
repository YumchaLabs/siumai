//! Small dated catalog of OpenAI language-model hints.
//!
//! Model IDs remain open. These constants support explicit introspection only
//! and never form an execution allowlist.

/// GPT-6 Astra model ID, verified against the official catalog on 2026-10-25.
pub const GPT_6_ASTRA: &str = "gpt-6-astra";
/// GPT-6.1 Sol model ID, verified against the official catalog on 2026-10-25.
pub const GPT_6_1_SOL: &str = "gpt-6.1-sol";
/// GPT-6 Luna model ID, verified against the official catalog on 2026-10-25.
pub const GPT_6_LUNA: &str = "gpt-6-luna";

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

/// Provider-owned exact-ID classification for catalog introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpenAiModelClass {
    Gpt6Astra,
    Gpt61Sol,
    Gpt6Luna,
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
        GPT_6_ASTRA => OpenAiModelClass::Gpt6Astra,
        GPT_6_1_SOL => OpenAiModelClass::Gpt61Sol,
        GPT_6_LUNA => OpenAiModelClass::Gpt6Luna,
        GPT_5_5 => OpenAiModelClass::Gpt55,
        GPT_5_5_PRO => OpenAiModelClass::Gpt55Pro,
        GPT_5_6 => OpenAiModelClass::Gpt56Alias,
        GPT_5_6_SOL => OpenAiModelClass::Gpt56Sol,
        GPT_5_6_TERRA => OpenAiModelClass::Gpt56Terra,
        GPT_5_6_LUNA => OpenAiModelClass::Gpt56Luna,
        _ => OpenAiModelClass::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_6_classification_uses_exact_ids_without_guessing_future_variants() {
        for (id, class) in [
            (GPT_6_ASTRA, OpenAiModelClass::Gpt6Astra),
            (GPT_6_1_SOL, OpenAiModelClass::Gpt61Sol),
            (GPT_6_LUNA, OpenAiModelClass::Gpt6Luna),
        ] {
            assert_eq!(classify_model(id), class);
            assert_eq!(
                classify_model(&format!("{id}-future")),
                OpenAiModelClass::Unknown
            );
        }
    }
}
