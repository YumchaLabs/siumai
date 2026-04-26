//! AI SDK-style JSON instruction prompt helpers.

use crate::types::{JSONSchema7, ModelMessage, SystemModelMessage};

/// Default schema prefix used by AI SDK `injectJsonInstruction`.
pub const DEFAULT_JSON_SCHEMA_PREFIX: &str = "JSON schema:";

/// Default schema suffix used by AI SDK `injectJsonInstruction`.
pub const DEFAULT_JSON_SCHEMA_SUFFIX: &str =
    "You MUST answer with a JSON object that matches the JSON schema above.";

/// Default generic JSON suffix used when no schema is provided.
pub const DEFAULT_JSON_GENERIC_SUFFIX: &str = "You MUST answer with JSON.";

/// Options for [`inject_json_instruction`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JsonInstructionOptions {
    /// Existing prompt text to prepend before the injected instruction.
    pub prompt: Option<String>,
    /// Optional JSON Schema to include in the instruction.
    pub schema: Option<JSONSchema7>,
    /// Optional schema prefix. When omitted, schema prompts use [`DEFAULT_JSON_SCHEMA_PREFIX`].
    pub schema_prefix: Option<String>,
    /// Optional suffix. When omitted, the default depends on whether `schema` is present.
    pub schema_suffix: Option<String>,
}

impl JsonInstructionOptions {
    /// Create empty JSON instruction options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set existing prompt text.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set the JSON Schema included in the instruction.
    pub fn with_schema(mut self, schema: JSONSchema7) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Override the schema prefix.
    pub fn with_schema_prefix(mut self, schema_prefix: impl Into<String>) -> Self {
        self.schema_prefix = Some(schema_prefix.into());
        self
    }

    /// Override the final instruction suffix.
    pub fn with_schema_suffix(mut self, schema_suffix: impl Into<String>) -> Self {
        self.schema_suffix = Some(schema_suffix.into());
        self
    }
}

/// Options for [`inject_json_instruction_into_messages`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JsonInstructionMessageOptions {
    /// Optional JSON Schema to include in the instruction.
    pub schema: Option<JSONSchema7>,
    /// Optional schema prefix. When omitted, schema prompts use [`DEFAULT_JSON_SCHEMA_PREFIX`].
    pub schema_prefix: Option<String>,
    /// Optional suffix. When omitted, the default depends on whether `schema` is present.
    pub schema_suffix: Option<String>,
}

impl JsonInstructionMessageOptions {
    /// Create empty message-injection options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the JSON Schema included in the instruction.
    pub fn with_schema(mut self, schema: JSONSchema7) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Override the schema prefix.
    pub fn with_schema_prefix(mut self, schema_prefix: impl Into<String>) -> Self {
        self.schema_prefix = Some(schema_prefix.into());
        self
    }

    /// Override the final instruction suffix.
    pub fn with_schema_suffix(mut self, schema_suffix: impl Into<String>) -> Self {
        self.schema_suffix = Some(schema_suffix.into());
        self
    }
}

/// Inject AI SDK-style JSON instructions into prompt text.
pub fn inject_json_instruction(options: JsonInstructionOptions) -> String {
    let has_schema = options.schema.is_some();
    let schema_prefix = options
        .schema_prefix
        .or_else(|| has_schema.then(|| DEFAULT_JSON_SCHEMA_PREFIX.to_string()));
    let schema_suffix = options.schema_suffix.or_else(|| {
        Some(if has_schema {
            DEFAULT_JSON_SCHEMA_SUFFIX.to_string()
        } else {
            DEFAULT_JSON_GENERIC_SUFFIX.to_string()
        })
    });

    let mut lines = Vec::new();

    if let Some(prompt) = options.prompt
        && !prompt.is_empty()
    {
        lines.push(prompt);
        lines.push(String::new());
    }

    if let Some(schema_prefix) = schema_prefix {
        lines.push(schema_prefix);
    }

    if let Some(schema) = options.schema {
        lines.push(serde_json::to_string(&schema).expect("JSON Schema value should serialize"));
    }

    if let Some(schema_suffix) = schema_suffix {
        lines.push(schema_suffix);
    }

    lines.join("\n")
}

/// Inject AI SDK-style JSON instructions into the first system model message.
///
/// If the prompt does not start with a system message, a new system message is prepended.
pub fn inject_json_instruction_into_messages(
    messages: Vec<ModelMessage>,
    options: JsonInstructionMessageOptions,
) -> Vec<ModelMessage> {
    let mut messages = messages.into_iter();
    let instruction_options = |prompt: String| JsonInstructionOptions {
        prompt: Some(prompt),
        schema: options.schema.clone(),
        schema_prefix: options.schema_prefix.clone(),
        schema_suffix: options.schema_suffix.clone(),
    };

    match messages.next() {
        Some(ModelMessage::System(mut system)) => {
            system.content = inject_json_instruction(instruction_options(system.content));
            let mut output = vec![ModelMessage::System(system)];
            output.extend(messages);
            output
        }
        Some(first) => {
            let system = SystemModelMessage::new(inject_json_instruction(instruction_options(
                String::new(),
            )));
            let mut output = vec![ModelMessage::System(system), first];
            output.extend(messages);
            output
        }
        None => vec![ModelMessage::System(SystemModelMessage::new(
            inject_json_instruction(instruction_options(String::new())),
        ))],
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{ModelMessage, SystemModelMessage, UserContent, UserModelMessage};

    use super::*;

    #[test]
    fn inject_json_instruction_matches_generic_prompt_shape() {
        assert_eq!(
            inject_json_instruction(JsonInstructionOptions::new().with_prompt("Return data")),
            "Return data\n\nYou MUST answer with JSON."
        );

        assert_eq!(
            inject_json_instruction(JsonInstructionOptions::new()),
            "You MUST answer with JSON."
        );
    }

    #[test]
    fn inject_json_instruction_includes_schema_defaults() {
        let instruction = inject_json_instruction(
            JsonInstructionOptions::new()
                .with_prompt("Return data")
                .with_schema(serde_json::json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } }
                })),
        );

        assert_eq!(
            instruction,
            concat!(
                "Return data\n\n",
                "JSON schema:\n",
                r#"{"type":"object","properties":{"answer":{"type":"string"}}}"#,
                "\nYou MUST answer with a JSON object that matches the JSON schema above."
            )
        );
    }

    #[test]
    fn inject_json_instruction_supports_custom_prefix_and_suffix() {
        assert_eq!(
            inject_json_instruction(
                JsonInstructionOptions::new()
                    .with_schema(serde_json::json!({ "type": "array" }))
                    .with_schema_prefix("Schema:")
                    .with_schema_suffix("Only JSON.")
            ),
            "Schema:\n{\"type\":\"array\"}\nOnly JSON."
        );
    }

    #[test]
    fn inject_json_instruction_into_messages_updates_existing_system() {
        let messages = vec![
            ModelMessage::System(SystemModelMessage::new("Rules")),
            ModelMessage::User(UserModelMessage::new(UserContent::text("Question"))),
        ];

        let output = inject_json_instruction_into_messages(
            messages,
            JsonInstructionMessageOptions::new()
                .with_schema(serde_json::json!({ "type": "object" })),
        );

        assert_eq!(output.len(), 2);
        let ModelMessage::System(system) = &output[0] else {
            panic!("first message should stay system");
        };
        assert!(system.content.starts_with("Rules\n\nJSON schema:"));
    }

    #[test]
    fn inject_json_instruction_into_messages_prepends_system_when_missing() {
        let messages = vec![ModelMessage::User(UserModelMessage::new(
            UserContent::text("Question"),
        ))];

        let output =
            inject_json_instruction_into_messages(messages, JsonInstructionMessageOptions::new());

        assert_eq!(output.len(), 2);
        let ModelMessage::System(system) = &output[0] else {
            panic!("first message should be injected system");
        };
        assert_eq!(system.content, "You MUST answer with JSON.");
    }
}
