//! Generic tag extraction for streaming text.
//!
//! This module provides a state machine-based tag extractor that can handle
//! streaming text input and extract content between opening and closing tags.
//!
//! The implementation is based on Cherry Studio's TagExtractor and Vercel AI SDK's
//! tag extraction logic, which correctly handles tags that are split across chunks.

/// Tag configuration for extraction.
///
/// Defines the opening and closing tags to extract, and an optional separator
/// to insert when switching between tag content and regular text.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::middleware::TagConfig;
///
/// let config = TagConfig::new("<think>", "</think>")
///     .with_separator("\n");
/// ```
#[derive(Debug, Clone)]
pub struct TagConfig {
    /// Opening tag (e.g., "<think>")
    pub opening_tag: String,
    /// Closing tag (e.g., "</think>")
    pub closing_tag: String,
    /// Optional separator to insert when switching contexts
    pub separator: Option<String>,
}

impl TagConfig {
    /// Create a new tag configuration.
    ///
    /// # Arguments
    ///
    /// * `opening_tag` - The opening tag to detect
    /// * `closing_tag` - The closing tag to detect
    pub fn new(opening_tag: impl Into<String>, closing_tag: impl Into<String>) -> Self {
        Self {
            opening_tag: opening_tag.into(),
            closing_tag: closing_tag.into(),
            separator: None,
        }
    }

    /// Set the separator to insert when switching contexts.
    ///
    /// # Arguments
    ///
    /// * `separator` - The separator string (e.g., "\n")
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = Some(separator.into());
        self
    }
}

/// Internal state for tag extraction.
#[derive(Debug, Clone)]
struct TagExtractionState {
    /// Buffer for accumulating text
    text_buffer: String,
    /// Whether we're currently inside a tag
    is_inside_tag: bool,
    /// Whether this is the first tag encountered
    is_first_tag: bool,
    /// Whether this is the first text encountered
    is_first_text: bool,
    /// Whether we just switched contexts
    after_switch: bool,
    /// Accumulated content from within tags
    accumulated_tag_content: String,
    /// Whether we have any tag content
    has_tag_content: bool,
}

impl Default for TagExtractionState {
    fn default() -> Self {
        Self {
            text_buffer: String::new(),
            is_inside_tag: false,
            is_first_tag: true,
            is_first_text: true,
            after_switch: false,
            accumulated_tag_content: String::new(),
            has_tag_content: false,
        }
    }
}

/// Result of tag extraction.
///
/// Each call to `process_text` may return multiple results as it processes
/// the buffered text.
#[derive(Debug, Clone)]
pub struct TagExtractionResult {
    /// The extracted content
    pub content: String,
    /// Whether this content is from inside a tag
    pub is_tag_content: bool,
    /// Whether tag extraction is complete
    pub complete: bool,
    /// The complete extracted tag content (only present when complete=true)
    pub tag_content_extracted: Option<String>,
}

/// Generic tag extractor for streaming text.
///
/// This extractor uses a state machine to handle tags that may be split across
/// multiple chunks of streaming text. It correctly handles partial tags at chunk
/// boundaries.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::middleware::{TagConfig, TagExtractor};
///
/// let config = TagConfig::new("<think>", "</think>");
/// let mut extractor = TagExtractor::new(config);
///
/// // Process chunks
/// let results1 = extractor.process_text("Hello <thin");
/// let results2 = extractor.process_text("k>thinking</think> world");
///
/// // Finalize
/// if let Some(final_result) = extractor.finalize() {
///     println!("Final thinking: {:?}", final_result.tag_content_extracted);
/// }
/// ```
pub struct TagExtractor {
    config: TagConfig,
    state: TagExtractionState,
}

impl TagExtractor {
    /// Create a new tag extractor with the given configuration.
    pub fn new(config: TagConfig) -> Self {
        Self {
            config,
            state: TagExtractionState::default(),
        }
    }

    /// Process a chunk of text and return extraction results.
    ///
    /// This method may return multiple results as it processes the buffered text.
    /// Each result indicates whether the content is from inside or outside a tag.
    ///
    /// # Arguments
    ///
    /// * `new_text` - The new text chunk to process
    ///
    /// # Returns
    ///
    /// A vector of extraction results. When a tag is complete, one of the results
    /// will have `complete=true` and `tag_content_extracted` will contain the
    /// complete tag content.
    pub fn process_text(&mut self, new_text: &str) -> Vec<TagExtractionResult> {
        self.state.text_buffer.push_str(new_text);
        let mut results = Vec::new();

        loop {
            let next_tag = if self.state.is_inside_tag {
                self.config.closing_tag.clone()
            } else {
                self.config.opening_tag.clone()
            };

            let start_index = get_potential_start_index(&self.state.text_buffer, &next_tag);

            if start_index.is_none() {
                // No tag found, output all buffered content
                let content = self.state.text_buffer.clone();
                if !content.is_empty() {
                    let prefixed_content = self.add_prefix(&content);
                    results.push(TagExtractionResult {
                        content: prefixed_content.clone(),
                        is_tag_content: self.state.is_inside_tag,
                        complete: false,
                        tag_content_extracted: None,
                    });

                    if self.state.is_inside_tag {
                        self.state
                            .accumulated_tag_content
                            .push_str(&prefixed_content);
                        self.state.has_tag_content = true;
                    }
                }
                self.state.text_buffer.clear();
                break;
            }

            let start_index = start_index.unwrap();

            // Process content before the tag
            if start_index > 0 {
                let content_before_tag = self.state.text_buffer[..start_index].to_string();
                let prefixed_content = self.add_prefix(&content_before_tag);
                results.push(TagExtractionResult {
                    content: prefixed_content.clone(),
                    is_tag_content: self.state.is_inside_tag,
                    complete: false,
                    tag_content_extracted: None,
                });

                if self.state.is_inside_tag {
                    self.state
                        .accumulated_tag_content
                        .push_str(&prefixed_content);
                    self.state.has_tag_content = true;
                }
            }

            let found_full_match = start_index + next_tag.len() <= self.state.text_buffer.len();

            if found_full_match {
                // Found complete tag
                self.state.text_buffer =
                    self.state.text_buffer[start_index + next_tag.len()..].to_string();

                // If just finished tag content, generate complete result
                if self.state.is_inside_tag && self.state.has_tag_content {
                    results.push(TagExtractionResult {
                        content: String::new(),
                        is_tag_content: false,
                        complete: true,
                        tag_content_extracted: Some(self.state.accumulated_tag_content.clone()),
                    });
                    self.state.accumulated_tag_content.clear();
                    self.state.has_tag_content = false;
                }

                self.state.is_inside_tag = !self.state.is_inside_tag;
                self.state.after_switch = true;

                if self.state.is_inside_tag {
                    self.state.is_first_tag = false;
                } else {
                    self.state.is_first_text = false;
                }
            } else {
                // Partial tag match, keep in buffer
                self.state.text_buffer = self.state.text_buffer[start_index..].to_string();
                break;
            }
        }

        results
    }

    /// Finalize processing and return any remaining tag content.
    ///
    /// This should be called when the stream is complete to extract any
    /// unclosed tag content.
    ///
    /// # Returns
    ///
    /// An optional extraction result containing the remaining tag content.
    pub fn finalize(&mut self) -> Option<TagExtractionResult> {
        if self.state.has_tag_content && !self.state.accumulated_tag_content.is_empty() {
            let result = TagExtractionResult {
                content: String::new(),
                is_tag_content: false,
                complete: true,
                tag_content_extracted: Some(self.state.accumulated_tag_content.clone()),
            };
            self.state.accumulated_tag_content.clear();
            self.state.has_tag_content = false;
            Some(result)
        } else {
            None
        }
    }

    /// Reset the extractor state.
    ///
    /// This clears all buffers and resets the state machine to its initial state.
    pub fn reset(&mut self) {
        self.state = TagExtractionState::default();
    }

    /// Add prefix to content based on context switching.
    fn add_prefix(&mut self, text: &str) -> String {
        let needs_prefix = self.state.after_switch
            && if self.state.is_inside_tag {
                !self.state.is_first_tag
            } else {
                !self.state.is_first_text
            };

        let prefix = if needs_prefix {
            self.config.separator.as_deref().unwrap_or("")
        } else {
            ""
        };

        self.state.after_switch = false;
        format!("{}{}", prefix, text)
    }
}

/// Get the potential start index of searched text in text.
///
/// This function handles both complete matches and partial matches, which is
/// crucial for streaming text where tags may be split across chunks.
///
/// # Algorithm
///
/// 1. First, check if the searched text exists as a complete substring
/// 2. If not, check if the end of the text matches the beginning of the searched text
///    (this handles the case where a tag is split across chunks)
///
/// # Example
///
/// ```rust,ignore
/// // Complete match
/// assert_eq!(get_potential_start_index("Hello <thinking>", "<thinking>"), Some(6));
///
/// // Partial match (tag split across chunks)
/// assert_eq!(get_potential_start_index("Hello <thin", "<thinking>"), Some(6));
/// assert_eq!(get_potential_start_index("Hello <", "<thinking>"), Some(6));
/// ```
///
/// # Arguments
///
/// * `text` - The text to search in
/// * `searched_text` - The text to search for
///
/// # Returns
///
/// The index where the searched text starts (or might start), or None if not found.
fn get_potential_start_index(text: &str, searched_text: &str) -> Option<usize> {
    if searched_text.is_empty() {
        return None;
    }

    // Check for complete match
    if let Some(index) = text.find(searched_text) {
        return Some(index);
    }

    // Check for suffix-prefix match (streaming case)
    // Look for the largest suffix of text that matches a prefix of searched_text
    for i in (0..text.len()).rev() {
        let suffix = &text[i..];
        if searched_text.starts_with(suffix) {
            return Some(i);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_potential_start_index_complete_match() {
        assert_eq!(
            get_potential_start_index("Hello <thinking>", "<thinking>"),
            Some(6)
        );
        assert_eq!(get_potential_start_index("Hello world", "world"), Some(6));
    }

    #[test]
    fn test_get_potential_start_index_partial_match() {
        assert_eq!(
            get_potential_start_index("Hello <thin", "<thinking>"),
            Some(6)
        );
        assert_eq!(get_potential_start_index("Hello <", "<thinking>"), Some(6));
        assert_eq!(get_potential_start_index("Hello w", "world"), Some(6));
    }

    #[test]
    fn test_get_potential_start_index_no_match() {
        assert_eq!(get_potential_start_index("Hello world", "xyz"), None);
        assert_eq!(get_potential_start_index("", "test"), None);
    }

    #[test]
    fn test_get_potential_start_index_empty_search() {
        assert_eq!(get_potential_start_index("Hello", ""), None);
    }
}
