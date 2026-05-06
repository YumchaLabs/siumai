//! AI SDK-style ID generation helpers.

use std::sync::Arc;

use crate::error::LlmError;

/// Default alphabet used by AI SDK `generateId`.
pub const DEFAULT_ID_ALPHABET: &str =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Default random suffix size used by AI SDK `generateId`.
pub const DEFAULT_ID_SIZE: usize = 16;

/// Cloneable ID generator function.
///
/// This mirrors AI SDK's `IdGenerator = () => string` contract.
pub type IdGenerator = Arc<dyn Fn() -> String + Send + Sync + 'static>;

/// Options for creating an ID generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdGeneratorOptions {
    /// Optional prefix added before the random part.
    pub prefix: Option<String>,
    /// Separator between prefix and random part.
    pub separator: String,
    /// Number of random characters.
    pub size: usize,
    /// Alphabet used for random characters.
    pub alphabet: String,
}

impl Default for IdGeneratorOptions {
    fn default() -> Self {
        Self {
            prefix: None,
            separator: "-".to_string(),
            size: DEFAULT_ID_SIZE,
            alphabet: DEFAULT_ID_ALPHABET.to_string(),
        }
    }
}

impl IdGeneratorOptions {
    /// Create default ID generator options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ID prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Set the prefix separator.
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    /// Set the random suffix size.
    pub const fn with_size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    /// Set the random alphabet.
    pub fn with_alphabet(mut self, alphabet: impl Into<String>) -> Self {
        self.alphabet = alphabet.into();
        self
    }
}

/// Create an ID generator.
///
/// The generator is intentionally not cryptographically secure, matching AI SDK's helper.
pub fn create_id_generator(options: IdGeneratorOptions) -> Result<IdGenerator, LlmError> {
    let alphabet: Vec<char> = options.alphabet.chars().collect();
    if alphabet.is_empty() {
        return Err(LlmError::InvalidParameter(
            "ID generator alphabet must not be empty".to_string(),
        ));
    }

    if options
        .prefix
        .as_ref()
        .is_some_and(|_| options.alphabet.contains(&options.separator))
    {
        return Err(LlmError::InvalidParameter(format!(
            "ID generator separator {:?} must not be part of the alphabet {:?}",
            options.separator, options.alphabet
        )));
    }

    let alphabet = Arc::new(alphabet);
    let size = options.size;

    let random_part = move || {
        (0..size)
            .map(|_| {
                let index = rand::random_range(0..alphabet.len());
                alphabet[index]
            })
            .collect::<String>()
    };

    match options.prefix {
        Some(prefix) => {
            let separator = options.separator;
            Ok(Arc::new(move || {
                format!("{prefix}{separator}{}", random_part())
            }))
        }
        None => Ok(Arc::new(random_part)),
    }
}

/// Generate a default 16-character random ID.
///
/// The value is intentionally not cryptographically secure.
pub fn generate_id() -> String {
    let generator = create_id_generator(IdGeneratorOptions::default())
        .expect("default ID generator options are valid");
    generator()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_id_uses_default_size() {
        let id = generate_id();
        assert_eq!(id.chars().count(), DEFAULT_ID_SIZE);
        assert!(id.chars().all(|ch| DEFAULT_ID_ALPHABET.contains(ch)));
    }

    #[test]
    fn create_id_generator_supports_prefix_and_custom_alphabet() {
        let generator = create_id_generator(
            IdGeneratorOptions::new()
                .with_prefix("tool")
                .with_size(8)
                .with_alphabet("ab"),
        )
        .expect("valid options");

        let id = generator();
        assert_eq!(id.chars().count(), "tool-".chars().count() + 8);
        assert!(id.starts_with("tool-"));
        assert!(id["tool-".len()..].chars().all(|ch| "ab".contains(ch)));
    }

    #[test]
    fn create_id_generator_rejects_empty_alphabet() {
        let Err(error) = create_id_generator(IdGeneratorOptions::new().with_alphabet("")) else {
            panic!("empty alphabet should fail");
        };
        assert!(error.to_string().contains("must not be empty"));
    }

    #[test]
    fn create_id_generator_rejects_ambiguous_prefixed_separator() {
        let Err(error) = create_id_generator(
            IdGeneratorOptions::new()
                .with_prefix("tool")
                .with_separator("-")
                .with_alphabet("a-b"),
        ) else {
            panic!("separator in alphabet should fail with prefix");
        };
        assert!(error.to_string().contains("must not be part"));
    }
}
