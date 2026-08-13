use thiserror::Error;

/// Validated upper-bound rule for the Messages `temperature` field.
///
/// Canonical request validation already guarantees a finite, non-negative
/// temperature. This rule captures the remaining dialect-specific upper bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureEncodingRule {
    maximum: f64,
}

impl TemperatureEncodingRule {
    /// Construct a finite, non-negative temperature upper bound.
    pub fn new(maximum: f64) -> Result<Self, MessagesEncodingRuleError> {
        if !maximum.is_finite() || maximum < 0.0 {
            return Err(MessagesEncodingRuleError::InvalidTemperatureMaximum);
        }
        Ok(Self { maximum })
    }

    /// Native Anthropic Messages upper bound.
    pub const fn native() -> Self {
        Self { maximum: 1.0 }
    }

    pub const fn maximum(self) -> f64 {
        self.maximum
    }
}

impl Default for TemperatureEncodingRule {
    fn default() -> Self {
        Self::native()
    }
}

/// How a compatible Messages dialect represents prompt-cache TTLs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheControlWireStyle {
    /// Emit both `type: "ephemeral"` and an explicit `ttl` field.
    #[default]
    ExplicitTtl,
    /// Emit only `type: "ephemeral"` for five-minute cache entries.
    ///
    /// One-hour cache entries are rejected because omitting their TTL would
    /// silently weaken their requested semantics.
    FiveMinutesImplicit,
}

/// How a Messages dialect represents system instructions after the conversation starts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum MidConversationSystemEncoding {
    /// Reject the request because the compatible dialect has not proved an encoding.
    #[default]
    Unsupported,
    /// Emit a normal Messages entry with `role: "system"`.
    InlineSystemRole,
}

/// Bounded request-encoding rules for one Anthropic Messages-compatible dialect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MessagesEncodingRules {
    temperature: TemperatureEncodingRule,
    cache_control: CacheControlWireStyle,
    video_input: bool,
    mid_conversation_system: MidConversationSystemEncoding,
}

impl MessagesEncodingRules {
    /// Conservative baseline for an Anthropic-compatible dialect.
    pub const fn compatible_baseline() -> Self {
        Self {
            temperature: TemperatureEncodingRule::native(),
            cache_control: CacheControlWireStyle::ExplicitTtl,
            video_input: false,
            mid_conversation_system: MidConversationSystemEncoding::Unsupported,
        }
    }

    /// Native Anthropic Messages encoding rules.
    pub const fn native() -> Self {
        Self {
            temperature: TemperatureEncodingRule::native(),
            cache_control: CacheControlWireStyle::ExplicitTtl,
            video_input: false,
            mid_conversation_system: MidConversationSystemEncoding::InlineSystemRole,
        }
    }

    pub const fn temperature(self) -> TemperatureEncodingRule {
        self.temperature
    }

    pub const fn cache_control(self) -> CacheControlWireStyle {
        self.cache_control
    }

    /// Whether user content may contain provider-compatible video blocks.
    /// Native Anthropic Messages keeps this disabled; compatible providers can
    /// opt in without changing the canonical core request model.
    pub const fn video_input(self) -> bool {
        self.video_input
    }

    pub const fn mid_conversation_system(self) -> MidConversationSystemEncoding {
        self.mid_conversation_system
    }

    pub const fn with_temperature(mut self, rule: TemperatureEncodingRule) -> Self {
        self.temperature = rule;
        self
    }

    pub const fn with_cache_control(mut self, style: CacheControlWireStyle) -> Self {
        self.cache_control = style;
        self
    }

    pub const fn with_video_input(mut self, enabled: bool) -> Self {
        self.video_input = enabled;
        self
    }

    pub const fn with_mid_conversation_system(
        mut self,
        encoding: MidConversationSystemEncoding,
    ) -> Self {
        self.mid_conversation_system = encoding;
        self
    }
}

impl Default for MessagesEncodingRules {
    fn default() -> Self {
        Self::compatible_baseline()
    }
}

/// Invalid compatible-dialect encoding configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum MessagesEncodingRuleError {
    #[error("temperature maximum must be finite and non-negative")]
    InvalidTemperatureMaximum,
}
