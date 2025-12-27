//! UTF-8 Stream Decoder
//!
//! This module provides a UTF-8 stream decoder that handles incomplete byte sequences
//! gracefully. It buffers incomplete UTF-8 byte sequences and only emits complete,
//! valid UTF-8 strings, preventing corruption when multi-byte characters are split
//! across network chunks.

use std::str;

/// A UTF-8 stream decoder that handles incomplete byte sequences gracefully.
///
/// This decoder buffers incomplete UTF-8 byte sequences and only emits
/// complete, valid UTF-8 strings. This prevents corruption when
/// multi-byte characters are split across network chunks.
///
/// # Example
///
/// ```rust
/// use siumai::utils::Utf8StreamDecoder;
///
/// let mut decoder = Utf8StreamDecoder::new();
///
/// // Process chunks that might split UTF-8 characters
/// let chunk1 = vec![0xE4, 0xB8]; // First 2 bytes of CJK char U+4E2D (incomplete)
/// let chunk2 = vec![0xAD]; // Last byte of that CJK char
///
/// let result1 = decoder.decode(&chunk1); // Returns empty string (incomplete)
/// let result2 = decoder.decode(&chunk2); // Returns the full CJK char
///
/// // Don't forget to flush at the end
/// let remaining = decoder.flush();
/// ```
#[derive(Debug, Default)]
pub struct Utf8StreamDecoder {
    buffer: Vec<u8>,
}

impl Utf8StreamDecoder {
    /// Create a new UTF-8 stream decoder.
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Decode a chunk of bytes, returning only complete UTF-8 strings.
    ///
    /// Incomplete UTF-8 sequences are buffered until the next chunk.
    /// Returns an empty string if no complete sequences are available.
    ///
    /// # Arguments
    ///
    /// * `chunk` - The byte chunk to decode
    ///
    /// # Returns
    ///
    /// A string containing all complete UTF-8 characters from the current
    /// and buffered bytes. Returns empty string if no complete characters
    /// are available.
    pub fn decode(&mut self, chunk: &[u8]) -> String {
        if chunk.is_empty() {
            return String::new();
        }

        // Add new bytes to buffer
        self.buffer.extend_from_slice(chunk);

        // Find the last complete UTF-8 sequence
        let last_complete_index = self.find_last_complete_utf8_index();

        if last_complete_index.is_none() {
            // No complete sequences, keep buffering
            return String::new();
        }

        let last_complete_index = last_complete_index.unwrap();

        // Extract complete bytes for decoding
        let complete_bytes = &self.buffer[..=last_complete_index];

        // Convert to string - this should always succeed due to our validation
        let result = match str::from_utf8(complete_bytes) {
            Ok(s) => s.to_string(),
            Err(_) => {
                // This shouldn't happen with our logic, but handle gracefully
                self.buffer.clear();
                return String::new();
            }
        };

        // Keep incomplete bytes for next chunk
        let remaining_bytes = self.buffer[last_complete_index + 1..].to_vec();
        self.buffer = remaining_bytes;

        result
    }

    /// Flush any remaining buffered bytes.
    ///
    /// Call this when the stream ends to get any remaining partial data.
    /// This may return invalid UTF-8 if the buffer contains incomplete sequences.
    ///
    /// # Returns
    ///
    /// A string containing any remaining buffered data, or empty string if
    /// the buffer is empty or contains invalid UTF-8.
    pub fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let result = match str::from_utf8(&self.buffer) {
            Ok(s) => s.to_string(),
            Err(_) => {
                // Invalid UTF-8 sequence, return empty string
                String::new()
            }
        };

        self.buffer.clear();
        result
    }

    /// Clear the internal buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Check if there are buffered bytes waiting for completion.
    pub fn has_buffered_bytes(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of buffered bytes.
    pub fn buffered_byte_count(&self) -> usize {
        self.buffer.len()
    }

    /// Find the index of the last complete UTF-8 character in the buffer.
    ///
    /// Returns None if no complete characters are found.
    fn find_last_complete_utf8_index(&self) -> Option<usize> {
        if self.buffer.is_empty() {
            return None;
        }

        // Start from the end and work backwards
        for i in (0..self.buffer.len()).rev() {
            let byte = self.buffer[i];

            // ASCII character (0xxxxxxx) - always complete
            if byte <= 0x7F {
                return Some(i);
            }

            // Start of multi-byte sequence (11xxxxxx)
            if (byte & 0xC0) == 0xC0 {
                // Determine expected sequence length
                let expected_length = if (byte & 0xE0) == 0xC0 {
                    2 // 110xxxxx
                } else if (byte & 0xF0) == 0xE0 {
                    3 // 1110xxxx
                } else if (byte & 0xF8) == 0xF0 {
                    4 // 11110xxx
                } else {
                    // Invalid start byte, skip
                    continue;
                };

                // Check if we have enough bytes for complete sequence
                let available_length = self.buffer.len() - i;
                if available_length >= expected_length {
                    // Verify all continuation bytes are valid
                    let mut is_valid = true;
                    for j in 1..expected_length {
                        if i + j >= self.buffer.len() || (self.buffer[i + j] & 0xC0) != 0x80 {
                            is_valid = false;
                            break;
                        }
                    }

                    if is_valid {
                        return Some(i + expected_length - 1);
                    }
                }

                // Incomplete sequence, check previous character
                if i > 0 {
                    // Create a temporary buffer with bytes up to current position
                    let temp_buffer = &self.buffer[..i];
                    let temp_decoder = Utf8StreamDecoder {
                        buffer: temp_buffer.to_vec(),
                    };
                    return temp_decoder.find_last_complete_utf8_index();
                } else {
                    return None;
                }
            }

            // Continuation byte (10xxxxxx) - keep looking backwards
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_characters() {
        let mut decoder = Utf8StreamDecoder::new();

        let result = decoder.decode(b"Hello");
        assert_eq!(result, "Hello");

        let result = decoder.decode(b" World");
        assert_eq!(result, " World");

        assert!(!decoder.has_buffered_bytes());
    }

    #[test]
    fn test_complete_utf8_characters() {
        let mut decoder = Utf8StreamDecoder::new();

        // Complete Chinese character
        let result = decoder.decode("你好".as_bytes());
        assert_eq!(result, "你好");

        assert!(!decoder.has_buffered_bytes());
    }

    #[test]
    fn test_incomplete_utf8_sequences() {
        let mut decoder = Utf8StreamDecoder::new();

        // Split CJK character U+4E2D (0xE4 0xB8 0xAD)
        let chunk1 = vec![0xE4, 0xB8]; // First 2 bytes (incomplete)
        let chunk2 = vec![0xAD]; // Last byte

        let result1 = decoder.decode(&chunk1);
        assert_eq!(result1, ""); // Should be empty (incomplete)
        assert!(decoder.has_buffered_bytes());
        assert_eq!(decoder.buffered_byte_count(), 2);

        let result2 = decoder.decode(&chunk2);
        assert_eq!(result2, "中"); // Should return the full CJK char
        assert!(!decoder.has_buffered_bytes());
    }

    #[test]
    fn test_mixed_content() {
        let mut decoder = Utf8StreamDecoder::new();

        // Mix of ASCII and multi-byte characters
        let text = "Hello 你好 World 🌍";
        let bytes = text.as_bytes();

        // Split into small chunks
        let mut result = String::new();
        for chunk in bytes.chunks(3) {
            result.push_str(&decoder.decode(chunk));
        }
        result.push_str(&decoder.flush());

        assert_eq!(result, text);
    }

    #[test]
    fn test_emoji_sequences() {
        let mut decoder = Utf8StreamDecoder::new();

        let emoji = "🌍🚀✨";
        let bytes = emoji.as_bytes();

        // Split emoji bytes awkwardly
        let mut result = String::new();
        for chunk in bytes.chunks(2) {
            result.push_str(&decoder.decode(chunk));
        }
        result.push_str(&decoder.flush());

        assert_eq!(result, emoji);
    }

    #[test]
    fn test_flush() {
        let mut decoder = Utf8StreamDecoder::new();

        // Add incomplete sequence
        let incomplete = vec![0xE4, 0xB8]; // First 2 bytes of CJK char U+4E2D
        let result = decoder.decode(&incomplete);
        assert_eq!(result, "");
        assert!(decoder.has_buffered_bytes());

        // Flush should return empty for invalid UTF-8
        let flushed = decoder.flush();
        assert_eq!(flushed, "");
        assert!(!decoder.has_buffered_bytes());
    }

    #[test]
    fn test_reset() {
        let mut decoder = Utf8StreamDecoder::new();

        // Add some data
        decoder.decode(b"Hello");
        decoder.decode(&[0xE4, 0xB8]); // Incomplete sequence

        assert!(decoder.has_buffered_bytes());

        decoder.reset();
        assert!(!decoder.has_buffered_bytes());
        assert_eq!(decoder.buffered_byte_count(), 0);
    }

    #[test]
    fn test_empty_input() {
        let mut decoder = Utf8StreamDecoder::new();

        let result = decoder.decode(&[]);
        assert_eq!(result, "");

        let flushed = decoder.flush();
        assert_eq!(flushed, "");
    }
}
